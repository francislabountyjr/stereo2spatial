"""Batch-window flow-matching loss computation."""

from __future__ import annotations

from dataclasses import replace

import torch
from accelerate import Accelerator

from .loss_terms import (
    _binaural_cue_loss,
    _channel_correlation_l1_loss,
    _channel_routing_kl_loss,
    _downmix_consistency_loss,
    _multi_resolution_stft_loss,
    _stereo_log_mel_perceptual_loss,
)
from .losses_windowed import (
    WindowMetadata,
    apply_mix_style_dropout,
    compute_flow_matching_window_loss,
    forward_window,
    init_memory_if_enabled,
    prepare_flow_matching_inputs,
    resolve_window_plan,
    resolve_window_weight,
    slice_and_pad_tensor4d,
    slice_and_pad_window,
)
from .scheduled_sampling import (
    ReflexFlowOptions,
    apply_flow_matching_scheduled_sampling,
)


def _resolve_effective_sequence_frames(
    *,
    t_max: int,
    force_full_sequence: bool,
    randomize_per_batch: bool,
    seq_choices_frames: list[int],
    max_choice_frames: int,
    global_step: int,
    seed: int,
) -> int:
    """Resolve per-step effective sequence length for strided training batches."""
    if not force_full_sequence and t_max < max_choice_frames:
        raise ValueError(
            f"Dataset emitted Tmax={t_max} frames but training.sequence_seconds_choices "
            f"requires up to {max_choice_frames}. Increase data.sequence_seconds."
        )
    if force_full_sequence:
        return int(t_max)
    if randomize_per_batch and len(seq_choices_frames) > 1:
        step_rng = torch.Generator(device="cpu")
        step_rng.manual_seed((int(seed) << 32) ^ int(global_step))
        choice_idx = int(
            torch.randint(len(seq_choices_frames), size=(1,), generator=step_rng).item()
        )
        return int(seq_choices_frames[choice_idx])
    return int(t_max)


def _apply_aux_losses_and_collect(
    *,
    window_loss: torch.Tensor,
    pred: torch.Tensor,
    z1_w: torch.Tensor,
    zc_w: torch.Tensor,
    target_downmix_w: torch.Tensor | None,
    vm_w: torch.Tensor,
    weight: torch.Tensor,
    routing_kl_weight: float,
    routing_kl_temperature: float,
    routing_kl_eps: float,
    corr_weight: float,
    corr_eps: float,
    corr_offdiag_only: bool,
    corr_use_correlation: bool,
    downmix_consistency_weight: float,
    downmix_consistency_loss: str,
    downmix_channel_order: list[str] | None,
    mrstft_loss_weight: float,
    mrstft_fft_sizes: list[int],
    mrstft_hop_lengths: list[int],
    mrstft_win_lengths: list[int],
    mrstft_sc_weight: float,
    mrstft_log_mag_weight: float,
    mrstft_eps: float,
    perceptual_loss_weight: float,
    perceptual_sample_rate: int,
    perceptual_n_fft: int,
    perceptual_hop_length: int,
    perceptual_win_length: int,
    perceptual_n_mels: int,
    perceptual_f_min: float,
    perceptual_f_max: float | None,
    perceptual_band_weight: float,
    perceptual_band_low_hz: float,
    perceptual_band_high_hz: float,
    perceptual_eps: float,
    binaural_ild_loss_weight: float,
    binaural_ipd_loss_weight: float,
    binaural_ccf_loss_weight: float,
    binaural_loss_warmup_steps: int,
    binaural_sample_rate: int,
    binaural_loss_eps: float,
    global_step: int,
    collect_gan_aux: bool,
    gan_cond_chunks: list[torch.Tensor] | None,
    gan_real_chunks: list[torch.Tensor] | None,
    gan_fake_chunks: list[torch.Tensor] | None,
    gan_mask_chunks: list[torch.Tensor] | None,
) -> torch.Tensor:
    """Apply optional aux losses and collect GAN window tensors when requested."""
    vm4 = vm_w[:, None, None, :].to(dtype=pred.dtype, device=pred.device)
    clean_pred_w = pred * vm4
    clean_tgt_w = z1_w.to(dtype=pred.dtype) * vm4

    w_t = weight[None, :].to(dtype=pred.dtype, device=pred.device)
    wm_t = vm_w.to(dtype=pred.dtype, device=pred.device) * w_t
    mask_dt = wm_t[:, None, :].expand(-1, pred.shape[2], -1)

    if float(routing_kl_weight) > 0.0:
        l_route = _channel_routing_kl_loss(
            prediction_x1=clean_pred_w,
            target_x1=clean_tgt_w,
            mask_dt=mask_dt,
            temperature=float(routing_kl_temperature),
            eps=float(routing_kl_eps),
        )
        window_loss = window_loss + float(routing_kl_weight) * l_route

    if float(corr_weight) > 0.0:
        l_corr = _channel_correlation_l1_loss(
            prediction_x1=clean_pred_w,
            target_x1=clean_tgt_w,
            mask_dt=mask_dt,
            eps=float(corr_eps),
            offdiag_only=bool(corr_offdiag_only),
            use_correlation=bool(corr_use_correlation),
        )
        window_loss = window_loss + float(corr_weight) * l_corr

    if float(downmix_consistency_weight) > 0.0:
        if target_downmix_w is None:
            raise KeyError(
                "batch must include target_downmix_signal when "
                "downmix_consistency_weight > 0"
            )
        l_downmix = _downmix_consistency_loss(
            prediction_x1=clean_pred_w,
            target_downmix_signal=target_downmix_w.to(dtype=pred.dtype) * vm4,
            mask_dt=mask_dt,
            channel_order=downmix_channel_order,
            loss_type=downmix_consistency_loss,
        )
        window_loss = window_loss + float(downmix_consistency_weight) * l_downmix

    if float(mrstft_loss_weight) > 0.0:
        l_mrstft = _multi_resolution_stft_loss(
            prediction_x1=clean_pred_w,
            target_x1=clean_tgt_w,
            mask_dt=mask_dt,
            fft_sizes=mrstft_fft_sizes,
            hop_lengths=mrstft_hop_lengths,
            win_lengths=mrstft_win_lengths,
            spectral_convergence_weight=mrstft_sc_weight,
            log_magnitude_weight=mrstft_log_mag_weight,
            eps=mrstft_eps,
        )
        window_loss = window_loss + float(mrstft_loss_weight) * l_mrstft

    if float(perceptual_loss_weight) > 0.0:
        l_perceptual = _stereo_log_mel_perceptual_loss(
            prediction_x1=clean_pred_w,
            target_x1=clean_tgt_w,
            target_downmix_signal=(
                target_downmix_w.to(dtype=pred.dtype) * vm4
                if target_downmix_w is not None
                else None
            ),
            mask_dt=mask_dt,
            sample_rate=int(perceptual_sample_rate),
            channel_order=downmix_channel_order,
            n_fft=int(perceptual_n_fft),
            hop_length=int(perceptual_hop_length),
            win_length=int(perceptual_win_length),
            n_mels=int(perceptual_n_mels),
            f_min=float(perceptual_f_min),
            f_max=perceptual_f_max,
            band_weight=float(perceptual_band_weight),
            band_low_hz=float(perceptual_band_low_hz),
            band_high_hz=float(perceptual_band_high_hz),
            eps=float(perceptual_eps),
        )
        window_loss = window_loss + float(perceptual_loss_weight) * l_perceptual

    if (
        float(binaural_ild_loss_weight) > 0.0
        or float(binaural_ipd_loss_weight) > 0.0
        or float(binaural_ccf_loss_weight) > 0.0
    ):
        warmup_steps = max(0, int(binaural_loss_warmup_steps))
        warmup = (
            1.0
            if warmup_steps <= 0
            else min(1.0, float(max(0, int(global_step))) / float(warmup_steps))
        )
        if warmup > 0.0:
            l_binaural = _binaural_cue_loss(
                prediction_x1=clean_pred_w,
                target_x1=clean_tgt_w,
                mask_dt=mask_dt,
                sample_rate=int(binaural_sample_rate),
                fft_sizes=mrstft_fft_sizes,
                hop_lengths=mrstft_hop_lengths,
                win_lengths=mrstft_win_lengths,
                ild_weight=float(binaural_ild_loss_weight) * warmup,
                ipd_weight=float(binaural_ipd_loss_weight) * warmup,
                ccf_weight=float(binaural_ccf_loss_weight) * warmup,
                eps=float(binaural_loss_eps),
            )
            window_loss = window_loss + l_binaural

    if collect_gan_aux:
        mask_disc = vm4.expand(-1, 1, pred.shape[2], -1)
        if (
            gan_fake_chunks is None
            or gan_real_chunks is None
            or gan_cond_chunks is None
            or gan_mask_chunks is None
        ):
            raise RuntimeError("GAN aux buffers were not initialized.")
        gan_fake_chunks.append(clean_pred_w)
        gan_real_chunks.append(clean_tgt_w)
        gan_cond_chunks.append(zc_w.to(dtype=pred.dtype) * vm4)
        gan_mask_chunks.append(mask_disc)

    return window_loss


def _finalize_gan_aux(
    *,
    collect_gan_aux: bool,
    gan_cond_chunks: list[torch.Tensor] | None,
    gan_real_chunks: list[torch.Tensor] | None,
    gan_fake_chunks: list[torch.Tensor] | None,
    gan_mask_chunks: list[torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    """Materialize concatenated GAN aux tensors from collected per-window buffers."""
    if not collect_gan_aux:
        return None
    if (
        gan_fake_chunks is None
        or gan_real_chunks is None
        or gan_cond_chunks is None
        or gan_mask_chunks is None
        or not gan_fake_chunks
    ):
        raise RuntimeError("GAN aux collection requested but no windows were produced.")
    return {
        "cond": torch.cat(gan_cond_chunks, dim=0),
        "real": torch.cat(gan_real_chunks, dim=0),
        "fake": torch.cat(gan_fake_chunks, dim=0),
        "mask": torch.cat(gan_mask_chunks, dim=0),
    }


def _compute_batch_flow_matching_loss(
    accelerator: Accelerator,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    seq_choices_frames: list[int],
    max_choice_frames: int,
    window_frames: int,
    overlap_frames: int,
    randomize_per_batch: bool,
    detach_memory: bool,
    global_step: int,
    seed: int,
    window_metadata: WindowMetadata | None = None,
    force_full_sequence: bool = False,
    collect_gan_aux: bool = False,
    scheduled_sampling_config: object | None = None,
    enable_scheduled_sampling: bool = True,
    # Optional routing/crosstalk regularizers in signal space.
    routing_kl_weight: float = 0.0,
    routing_kl_temperature: float = 1.0,
    routing_kl_eps: float = 1e-6,
    corr_weight: float = 0.0,
    corr_eps: float = 1e-6,
    corr_offdiag_only: bool = True,
    corr_use_correlation: bool = True,
    downmix_consistency_weight: float = 0.0,
    downmix_consistency_loss: str = "mse",
    downmix_channel_order: list[str] | None = None,
    mrstft_loss_weight: float = 0.0,
    mrstft_fft_sizes: list[int] | None = None,
    mrstft_hop_lengths: list[int] | None = None,
    mrstft_win_lengths: list[int] | None = None,
    mrstft_sc_weight: float = 1.0,
    mrstft_log_mag_weight: float = 1.0,
    mrstft_eps: float = 1e-7,
    perceptual_loss_weight: float = 0.0,
    perceptual_sample_rate: int = 48000,
    perceptual_n_fft: int = 1024,
    perceptual_hop_length: int = 256,
    perceptual_win_length: int = 1024,
    perceptual_n_mels: int = 80,
    perceptual_f_min: float = 40.0,
    perceptual_f_max: float | None = None,
    perceptual_band_weight: float = 1.0,
    perceptual_band_low_hz: float = 150.0,
    perceptual_band_high_hz: float = 8000.0,
    perceptual_eps: float = 1e-5,
    binaural_ild_loss_weight: float = 0.0,
    binaural_ipd_loss_weight: float = 0.0,
    binaural_ccf_loss_weight: float = 0.0,
    binaural_loss_warmup_steps: int = 0,
    binaural_sample_rate: int = 48000,
    binaural_loss_eps: float = 1e-7,
) -> tuple[torch.Tensor, int, int, dict[str, torch.Tensor] | None]:
    """Compute flow-matching loss over fixed-size waveform windows."""
    z1 = batch["target_signal"]  # [B,C,P,Tmax]
    z_cond = batch["cond_signal"]  # [B,Cc,P,Tmax]
    target_downmix = batch.get("target_downmix_signal")  # [B,2,P,Tmax]
    valid_mask = batch["valid_mask"]  # [B,Tmax]
    mix_style, mix_style_mask = apply_mix_style_dropout(
        mix_style=batch.get("mix_style"),
        training_config=scheduled_sampling_config,
        model=model,
    )

    t_max = z1.shape[-1]
    t_eff = min(
        _resolve_effective_sequence_frames(
            t_max=t_max,
            force_full_sequence=force_full_sequence,
            randomize_per_batch=randomize_per_batch,
            seq_choices_frames=seq_choices_frames,
            max_choice_frames=max_choice_frames,
            global_step=global_step,
            seed=seed,
        ),
        int(t_max),
    )
    inputs = prepare_flow_matching_inputs(
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t_eff=t_eff,
        training_config=scheduled_sampling_config,
    )
    reflexflow = ReflexFlowOptions(enabled=False, alpha=1.0, beta1=0.0, beta2=1.0)
    reflex_clean_pred: torch.Tensor | None = None
    reflex_biased_pred: torch.Tensor | None = None
    if enable_scheduled_sampling and scheduled_sampling_config is not None:
        rollout = apply_flow_matching_scheduled_sampling(
            model=model,
            z1=inputs.z1,
            z_cond=inputs.z_cond,
            valid_mask=inputs.valid_mask,
            t=inputs.t,
            zt=inputs.zt,
            z0=inputs.z0,
            mix_style=mix_style,
            mix_style_mask=mix_style_mask,
            training_config=scheduled_sampling_config,
            global_step=global_step,
            window_frames=window_frames,
            overlap_frames=overlap_frames,
        )
        inputs = replace(inputs, t=rollout.t, zt=rollout.zt)
        reflexflow = rollout.reflexflow
        reflex_clean_pred = rollout.reflex_clean_pred
        reflex_biased_pred = rollout.reflex_biased_pred
    mem = init_memory_if_enabled(
        accelerator=accelerator,
        model=model,
        batch_size=inputs.batch_size,
        device=inputs.z1.device,
        dtype=inputs.zt.dtype,
    )
    starts, cached_weights = resolve_window_plan(
        t_eff=t_eff,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        window_metadata=window_metadata,
    )
    num_windows = len(starts)

    need_clean_pred = (
        collect_gan_aux
        or (float(routing_kl_weight) > 0.0)
        or (float(corr_weight) > 0.0)
        or (float(downmix_consistency_weight) > 0.0)
        or (float(mrstft_loss_weight) > 0.0)
        or (float(perceptual_loss_weight) > 0.0)
        or (float(binaural_ild_loss_weight) > 0.0)
        or (float(binaural_ipd_loss_weight) > 0.0)
        or (float(binaural_ccf_loss_weight) > 0.0)
    )
    resolved_mrstft_fft_sizes = (
        [512, 1024, 2048] if mrstft_fft_sizes is None else mrstft_fft_sizes
    )
    resolved_mrstft_hop_lengths = (
        [128, 256, 512] if mrstft_hop_lengths is None else mrstft_hop_lengths
    )
    resolved_mrstft_win_lengths = (
        [512, 1024, 2048] if mrstft_win_lengths is None else mrstft_win_lengths
    )

    total_loss: torch.Tensor = torch.zeros((), device=z1.device, dtype=torch.float32)

    gan_cond_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None
    gan_real_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None
    gan_fake_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None
    gan_mask_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None

    for idx, start in enumerate(starts):
        end = min(start + window_frames, t_eff)
        zt_w, zc_w, vm_w, z1_w = slice_and_pad_window(
            zt=inputs.zt,
            z_cond=inputs.z_cond,
            valid_mask=inputs.valid_mask,
            start=start,
            end=end,
            window_frames=window_frames,
            batch_size=inputs.batch_size,
            z1=inputs.z1,
        )
        target_downmix_w = (
            slice_and_pad_tensor4d(
                tensor=target_downmix[..., :t_eff],
                start=start,
                end=end,
                window_frames=window_frames,
            )
            if target_downmix is not None
            else None
        )
        if z1_w is None:
            raise RuntimeError("z1 window is required for batch-window loss.")

        weight = resolve_window_weight(
            idx=idx,
            num_windows=num_windows,
            window_frames=window_frames,
            overlap_frames=overlap_frames,
            cached_weights=cached_weights,
            device=inputs.z1.device,
            dtype=inputs.zt.dtype,
        )
        pred, mem = forward_window(
            model=model,
            zt_w=zt_w,
            zc_w=zc_w,
            vm_w=vm_w,
            t=inputs.t,
            mem=mem,
            detach_memory=detach_memory,
            mix_style=mix_style,
            mix_style_mask=mix_style_mask,
        )

        # ---- main clean-prediction loss ----
        clean_pred_w: torch.Tensor | None = None
        biased_pred_w: torch.Tensor | None = None
        adr_target_w: torch.Tensor | None = None
        if (
            reflexflow.enabled
            and reflex_clean_pred is not None
            and reflex_biased_pred is not None
        ):
            clean_pred_w = slice_and_pad_tensor4d(
                tensor=reflex_clean_pred,
                start=start,
                end=end,
                window_frames=window_frames,
            ).to(device=pred.device, dtype=pred.dtype)
            biased_pred_w = slice_and_pad_tensor4d(
                tensor=reflex_biased_pred,
                start=start,
                end=end,
                window_frames=window_frames,
            ).to(device=pred.device, dtype=pred.dtype)
        if reflexflow.enabled:
            adr_target_w = (z1_w.to(dtype=pred.dtype) - zt_w.to(dtype=pred.dtype)).to(
                device=pred.device
            )
        adr_prediction_w = (
            (pred.to(dtype=zt_w.dtype) - zt_w).to(device=pred.device)
            if reflexflow.enabled
            else None
        )

        loss_fm_w = compute_flow_matching_window_loss(
            prediction=pred,
            target_clean=z1_w.to(dtype=pred.dtype, device=pred.device),
            valid_mask=vm_w,
            frame_weight=weight,
            sample_loss_weight=inputs.loss_weight,
            reflex_enabled=reflexflow.enabled,
            reflex_clean_pred=clean_pred_w,
            reflex_biased_pred=biased_pred_w,
            reflex_prediction_vector=adr_prediction_w,
            reflex_target_vector=adr_target_w,
            reflex_alpha=reflexflow.alpha,
            reflex_beta1=reflexflow.beta1,
            reflex_beta2=reflexflow.beta2,
        )
        window_loss = loss_fm_w.float()

        # ---- optional: routing + correlation losses on clean prediction ----
        if need_clean_pred:
            window_loss = _apply_aux_losses_and_collect(
                window_loss=window_loss,
                pred=pred,
                z1_w=z1_w,
                zc_w=zc_w,
                target_downmix_w=target_downmix_w,
                vm_w=vm_w,
                weight=weight,
                routing_kl_weight=routing_kl_weight,
                routing_kl_temperature=routing_kl_temperature,
                routing_kl_eps=routing_kl_eps,
                corr_weight=corr_weight,
                corr_eps=corr_eps,
                corr_offdiag_only=corr_offdiag_only,
                corr_use_correlation=corr_use_correlation,
                downmix_consistency_weight=downmix_consistency_weight,
                downmix_consistency_loss=downmix_consistency_loss,
                downmix_channel_order=downmix_channel_order,
                mrstft_loss_weight=mrstft_loss_weight,
                mrstft_fft_sizes=resolved_mrstft_fft_sizes,
                mrstft_hop_lengths=resolved_mrstft_hop_lengths,
                mrstft_win_lengths=resolved_mrstft_win_lengths,
                mrstft_sc_weight=mrstft_sc_weight,
                mrstft_log_mag_weight=mrstft_log_mag_weight,
                mrstft_eps=mrstft_eps,
                perceptual_loss_weight=perceptual_loss_weight,
                perceptual_sample_rate=perceptual_sample_rate,
                perceptual_n_fft=perceptual_n_fft,
                perceptual_hop_length=perceptual_hop_length,
                perceptual_win_length=perceptual_win_length,
                perceptual_n_mels=perceptual_n_mels,
                perceptual_f_min=perceptual_f_min,
                perceptual_f_max=perceptual_f_max,
                perceptual_band_weight=perceptual_band_weight,
                perceptual_band_low_hz=perceptual_band_low_hz,
                perceptual_band_high_hz=perceptual_band_high_hz,
                perceptual_eps=perceptual_eps,
                binaural_ild_loss_weight=binaural_ild_loss_weight,
                binaural_ipd_loss_weight=binaural_ipd_loss_weight,
                binaural_ccf_loss_weight=binaural_ccf_loss_weight,
                binaural_loss_warmup_steps=binaural_loss_warmup_steps,
                binaural_sample_rate=binaural_sample_rate,
                binaural_loss_eps=binaural_loss_eps,
                global_step=global_step,
                collect_gan_aux=collect_gan_aux,
                gan_cond_chunks=gan_cond_chunks,
                gan_real_chunks=gan_real_chunks,
                gan_fake_chunks=gan_fake_chunks,
                gan_mask_chunks=gan_mask_chunks,
            )

        total_loss = total_loss + window_loss

    gan_aux = _finalize_gan_aux(
        collect_gan_aux=collect_gan_aux,
        gan_cond_chunks=gan_cond_chunks,
        gan_real_chunks=gan_real_chunks,
        gan_fake_chunks=gan_fake_chunks,
        gan_mask_chunks=gan_mask_chunks,
    )
    return total_loss / max(num_windows, 1), t_eff, num_windows, gan_aux
