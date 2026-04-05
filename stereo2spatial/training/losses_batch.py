"""Batch-window flow-matching loss computation."""

from __future__ import annotations

from dataclasses import replace

import torch
from accelerate import Accelerator

from .loss_terms import (
    _channel_correlation_l1_loss,
    _channel_routing_kl_loss,
)
from .losses_windowed import (
    WindowMetadata,
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
    zt_w: torch.Tensor,
    t: torch.Tensor,
    z1_w: torch.Tensor,
    zc_w: torch.Tensor,
    vm_w: torch.Tensor,
    weight: torch.Tensor,
    routing_kl_weight: float,
    routing_kl_temperature: float,
    routing_kl_eps: float,
    corr_weight: float,
    corr_eps: float,
    corr_offdiag_only: bool,
    corr_use_correlation: bool,
    collect_gan_aux: bool,
    gan_cond_chunks: list[torch.Tensor] | None,
    gan_real_chunks: list[torch.Tensor] | None,
    gan_fake_chunks: list[torch.Tensor] | None,
    gan_mask_chunks: list[torch.Tensor] | None,
) -> torch.Tensor:
    """Apply optional aux losses and collect GAN window tensors when requested."""
    vm4 = vm_w[:, None, None, :].to(dtype=pred.dtype, device=pred.device)
    x1_hat_w = zt_w + (1.0 - t[:, None, None, None]) * pred
    x1_hat_w = x1_hat_w * vm4
    x1_tgt_w = z1_w.to(dtype=pred.dtype) * vm4

    w_t = weight[None, :].to(dtype=pred.dtype, device=pred.device)
    wm_t = vm_w.to(dtype=pred.dtype, device=pred.device) * w_t
    mask_dt = wm_t[:, None, :].expand(-1, pred.shape[2], -1)

    if float(routing_kl_weight) > 0.0:
        l_route = _channel_routing_kl_loss(
            prediction_x1=x1_hat_w,
            target_x1=x1_tgt_w,
            mask_dt=mask_dt,
            temperature=float(routing_kl_temperature),
            eps=float(routing_kl_eps),
        )
        window_loss = window_loss + float(routing_kl_weight) * l_route

    if float(corr_weight) > 0.0:
        l_corr = _channel_correlation_l1_loss(
            prediction_x1=x1_hat_w,
            target_x1=x1_tgt_w,
            mask_dt=mask_dt,
            eps=float(corr_eps),
            offdiag_only=bool(corr_offdiag_only),
            use_correlation=bool(corr_use_correlation),
        )
        window_loss = window_loss + float(corr_weight) * l_corr

    if collect_gan_aux:
        mask_disc = vm4.expand(-1, 1, pred.shape[2], -1)
        if (
            gan_fake_chunks is None
            or gan_real_chunks is None
            or gan_cond_chunks is None
            or gan_mask_chunks is None
        ):
            raise RuntimeError("GAN aux buffers were not initialized.")
        gan_fake_chunks.append(x1_hat_w)
        gan_real_chunks.append(x1_tgt_w)
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
    # Optional routing/crosstalk regularizers in latent space.
    routing_kl_weight: float = 0.0,
    routing_kl_temperature: float = 1.0,
    routing_kl_eps: float = 1e-6,
    corr_weight: float = 0.0,
    corr_eps: float = 1e-6,
    corr_offdiag_only: bool = True,
    corr_use_correlation: bool = True,
) -> tuple[torch.Tensor, int, int, dict[str, torch.Tensor] | None]:
    """Compute flow-matching loss over fixed-size training windows."""
    z1 = batch["target_latent"]  # [B,C,D,Tmax]
    z_cond = batch["cond_latent"]  # [B,Cc,D,Tmax]
    valid_mask = batch["valid_mask"]  # [B,Tmax]

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
        z0 = inputs.z1 - inputs.target_velocity
        rollout = apply_flow_matching_scheduled_sampling(
            model=model,
            z1=inputs.z1,
            z_cond=inputs.z_cond,
            valid_mask=inputs.valid_mask,
            t=inputs.t,
            zt=inputs.zt,
            z0=z0,
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

    need_x1_hat = (
        collect_gan_aux
        or (float(routing_kl_weight) > 0.0)
        or (float(corr_weight) > 0.0)
    )

    total_loss: torch.Tensor = torch.zeros((), device=z1.device, dtype=torch.float32)

    gan_cond_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None
    gan_real_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None
    gan_fake_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None
    gan_mask_chunks: list[torch.Tensor] | None = [] if collect_gan_aux else None

    for idx, start in enumerate(starts):
        end = min(start + window_frames, t_eff)
        zt_w, zc_w, tv_w, vm_w, z1_w = slice_and_pad_window(
            zt=inputs.zt,
            z_cond=inputs.z_cond,
            target_velocity=inputs.target_velocity,
            valid_mask=inputs.valid_mask,
            start=start,
            end=end,
            window_frames=window_frames,
            batch_size=inputs.batch_size,
            z1=inputs.z1,
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
        )

        # ---- main FM velocity loss ----
        clean_pred_w: torch.Tensor | None = None
        biased_pred_w: torch.Tensor | None = None
        adr_target_w: torch.Tensor | None = None
        if reflexflow.enabled and reflex_clean_pred is not None and reflex_biased_pred is not None:
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

        loss_fm_w = compute_flow_matching_window_loss(
            prediction=pred,
            target_velocity=tv_w,
            valid_mask=vm_w,
            frame_weight=weight,
            sample_loss_weight=inputs.loss_weight,
            reflex_enabled=reflexflow.enabled,
            reflex_clean_pred=clean_pred_w,
            reflex_biased_pred=biased_pred_w,
            reflex_target_vector=adr_target_w,
            reflex_alpha=reflexflow.alpha,
            reflex_beta1=reflexflow.beta1,
            reflex_beta2=reflexflow.beta2,
        )
        window_loss = loss_fm_w.float()

        # ---- optional: routing + correlation losses on x1_hat ----
        if need_x1_hat:
            window_loss = _apply_aux_losses_and_collect(
                window_loss=window_loss,
                pred=pred,
                zt_w=zt_w,
                t=inputs.t,
                z1_w=z1_w,
                zc_w=zc_w,
                vm_w=vm_w,
                weight=weight,
                routing_kl_weight=routing_kl_weight,
                routing_kl_temperature=routing_kl_temperature,
                routing_kl_eps=routing_kl_eps,
                corr_weight=corr_weight,
                corr_eps=corr_eps,
                corr_offdiag_only=corr_offdiag_only,
                corr_use_correlation=corr_use_correlation,
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
