"""Shared loss-term primitives for flow-matching training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from stereo2spatial.common.channel_layouts import (
    AC3_DOWNMIX_COEFFICIENTS,
    CHANNEL_COUNT_FALLBACKS,
    CHANNEL_ORDER_7_1_4,
)

DEFAULT_DOWNMIX_CHANNEL_ORDER_7_1_4 = CHANNEL_ORDER_7_1_4


def _compute_loss_weighted(
    prediction: torch.Tensor,  # [B,C,D,T]
    target_clean: torch.Tensor,  # [B,C,D,T]
    valid_mask: torch.Tensor,  # [B,T]
    frame_weight: torch.Tensor,  # [T]
) -> torch.Tensor:
    """Compute frame-weighted masked MSE for clean endpoint targets."""
    mse = (prediction - target_clean).pow(2)  # [B,C,D,T]
    return _reduce_masked_reconstruction_loss(
        loss=mse,
        prediction=prediction,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
    )


def _reduce_masked_reconstruction_loss(
    *,
    loss: torch.Tensor,
    prediction: torch.Tensor,
    valid_mask: torch.Tensor,
    frame_weight: torch.Tensor,
) -> torch.Tensor:
    """Reduce a `[B,C,D,T]` reconstruction loss with frame and validity weights."""
    w = frame_weight[None, :].to(
        dtype=prediction.dtype, device=prediction.device
    )  # [1,T]
    m = valid_mask.to(dtype=prediction.dtype, device=prediction.device)  # [B,T]
    wm = (w * m)[:, None, None, :]  # [B,1,1,T]

    weighted = loss * wm
    denom = wm.sum() * prediction.shape[1] * prediction.shape[2]
    denom = torch.clamp(denom, min=1.0)
    return weighted.sum() / denom


def _charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-3,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute smooth L1-style Charbonnier reconstruction loss."""
    eps_f = float(eps)
    if eps_f <= 0:
        raise ValueError("charbonnier eps must be > 0")
    loss = torch.sqrt((prediction - target).float().pow(2) + eps_f * eps_f) - eps_f
    reduction_name = str(reduction).strip().lower()
    if reduction_name == "mean":
        return loss.mean()
    if reduction_name == "sum":
        return loss.sum()
    if reduction_name == "none":
        return loss
    raise ValueError("charbonnier reduction must be one of: mean, sum, none")


def _masked_waveform_reconstruction_loss(
    *,
    prediction: torch.Tensor,
    target_clean: torch.Tensor,
    valid_mask: torch.Tensor,
    frame_weight: torch.Tensor,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
    charbonnier_weight: float = 0.0,
    charbonnier_eps: float = 1e-3,
    sample_loss_weight: torch.Tensor | None = None,
    element_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute a weighted sum of masked waveform reconstruction losses."""
    pred_f = prediction.float()
    target_f = target_clean.float()
    total = pred_f.new_zeros(())
    used = False

    def apply_extra_weights(loss: torch.Tensor) -> torch.Tensor:
        if element_weight is not None:
            loss = loss * torch.as_tensor(
                element_weight,
                dtype=loss.dtype,
                device=loss.device,
            )
        if sample_loss_weight is not None:
            if (
                sample_loss_weight.dim() != 1
                or sample_loss_weight.shape[0] != prediction.shape[0]
            ):
                raise ValueError(
                    "sample_loss_weight must be shape [B] matching prediction batch size."
                )
            loss = loss * sample_loss_weight[:, None, None, None].to(
                dtype=loss.dtype,
                device=loss.device,
            )
        return loss

    if float(mse_weight) > 0.0:
        total = total + float(mse_weight) * _reduce_masked_reconstruction_loss(
            loss=apply_extra_weights((pred_f - target_f).pow(2)),
            prediction=prediction,
            valid_mask=valid_mask,
            frame_weight=frame_weight,
        )
        used = True
    if float(l1_weight) > 0.0:
        total = total + float(l1_weight) * _reduce_masked_reconstruction_loss(
            loss=apply_extra_weights((pred_f - target_f).abs()),
            prediction=prediction,
            valid_mask=valid_mask,
            frame_weight=frame_weight,
        )
        used = True
    if float(charbonnier_weight) > 0.0:
        total = total + float(charbonnier_weight) * _reduce_masked_reconstruction_loss(
            loss=apply_extra_weights(
                _charbonnier_loss(
                    pred_f,
                    target_f,
                    eps=float(charbonnier_eps),
                    reduction="none",
                )
            ),
            prediction=prediction,
            valid_mask=valid_mask,
            frame_weight=frame_weight,
        )
        used = True

    if not used:
        raise ValueError(
            "At least one waveform reconstruction loss weight must be > 0 "
            "(waveform_mse_loss_weight, waveform_l1_loss_weight, or "
            "waveform_charbonnier_loss_weight)."
        )
    return total


def _downmix_to_stereo(
    signal: torch.Tensor,
    channel_order: list[str] | None = None,
) -> torch.Tensor:
    """Apply the configured spatial-to-stereo training downmix over waveform patches."""
    if signal.dim() != 4:
        raise ValueError(f"signal must be [B,C,P,T], got {tuple(signal.shape)}")
    order = (
        CHANNEL_COUNT_FALLBACKS.get(
            signal.shape[1], DEFAULT_DOWNMIX_CHANNEL_ORDER_7_1_4
        )
        if channel_order is None
        else channel_order
    )
    if len(order) != signal.shape[1]:
        raise ValueError(
            "downmix_channel_order length must match signal channels "
            f"({len(order)} != {signal.shape[1]})"
        )

    stereo = torch.zeros(
        (signal.shape[0], 2, signal.shape[2], signal.shape[3]),
        dtype=signal.dtype,
        device=signal.device,
    )
    for index, label in enumerate(order):
        left, right = AC3_DOWNMIX_COEFFICIENTS.get(label.upper(), (0.5, 0.5))
        if left:
            stereo[:, 0] = stereo[:, 0] + signal[:, index] * float(left)
        if right:
            stereo[:, 1] = stereo[:, 1] + signal[:, index] * float(right)
    return stereo


def _align_downmix_to_conditioning(
    downmix: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Match predicted stereo downmix channels to mono or stereo reference."""
    if reference.shape[1] == 2:
        return downmix
    if reference.shape[1] == 1:
        return downmix.mean(dim=1, keepdim=True)
    raise ValueError(
        "target_downmix_signal must have one or two waveform channels for downmix "
        f"consistency, got {reference.shape[1]}"
    )


def _downmix_consistency_loss(
    *,
    prediction_x1: torch.Tensor,
    target_downmix_signal: torch.Tensor,
    mask_dt: torch.Tensor,
    channel_order: list[str] | None = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Penalize predicted spatial waveforms whose downmix diverges from target downmix."""
    pred_downmix = _downmix_to_stereo(
        prediction_x1.float(), channel_order=channel_order
    )
    target = target_downmix_signal.float()
    if (
        pred_downmix.shape[0] != target.shape[0]
        or pred_downmix.shape[2:] != target.shape[2:]
    ):
        raise ValueError(
            "Prediction downmix and target downmix shapes do not align: "
            f"downmix={tuple(pred_downmix.shape)} target={tuple(target.shape)}"
        )
    pred_cmp = _align_downmix_to_conditioning(pred_downmix, target)

    loss_name = str(loss_type).strip().lower()
    if loss_name in {"l1", "mae"}:
        err = (pred_cmp - target).abs()
    elif loss_name in {"mse", "l2"}:
        err = (pred_cmp - target).pow(2)
    else:
        raise ValueError("downmix_consistency_loss must be one of: mse, l1")

    mask = mask_dt[:, None, :, :].to(dtype=err.dtype, device=err.device)
    denom = torch.clamp(mask.sum() * err.shape[1], min=1.0)
    return (err * mask).sum() / denom


def _unpatch_signal_to_audio(
    signal: torch.Tensor,
    mask_dt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Flatten waveform patches `[B,C,P,T]` into audio `[B*C,S]`."""
    if signal.dim() != 4:
        raise ValueError(f"signal must be [B,C,P,T], got {tuple(signal.shape)}")
    audio = (
        signal.float()
        .permute(0, 1, 3, 2)
        .reshape(signal.shape[0] * signal.shape[1], -1)
    )
    if mask_dt is None:
        return audio, None
    if mask_dt.shape != (signal.shape[0], signal.shape[2], signal.shape[3]):
        raise ValueError(
            "mask_dt must have shape [B,P,T] matching signal patches, "
            f"got {tuple(mask_dt.shape)} for signal {tuple(signal.shape)}"
        )
    mask = mask_dt.float().permute(0, 2, 1).reshape(signal.shape[0], -1)
    mask = mask.repeat_interleave(signal.shape[1], dim=0).to(device=signal.device)
    return audio * mask, mask


def _unpatch_binaural_signal_to_audio(
    signal: torch.Tensor,
    mask_dt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flatten binaural waveform patches `[B,2,P,T]` into `[B,2,S]` audio."""
    if signal.dim() != 4:
        raise ValueError(f"signal must be [B,C,P,T], got {tuple(signal.shape)}")
    if signal.shape[1] != 2:
        raise ValueError(
            "binaural cue losses require exactly two target channels, "
            f"got {signal.shape[1]}"
        )
    audio = signal.float().permute(0, 1, 3, 2).reshape(signal.shape[0], 2, -1)
    if mask_dt is None:
        return audio
    if mask_dt.shape != (signal.shape[0], signal.shape[2], signal.shape[3]):
        raise ValueError(
            "mask_dt must have shape [B,P,T] matching signal patches, "
            f"got {tuple(mask_dt.shape)} for signal {tuple(signal.shape)}"
        )
    mask = mask_dt.float().permute(0, 2, 1).reshape(signal.shape[0], -1)
    return audio * mask[:, None, :].to(device=signal.device)


def _multi_resolution_stft_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    fft_sizes: list[int],
    hop_lengths: list[int],
    win_lengths: list[int],
    spectral_convergence_weight: float = 1.0,
    log_magnitude_weight: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution STFT loss over unpatched waveform predictions."""
    if prediction_x1.shape != target_x1.shape:
        raise ValueError(
            "prediction_x1 and target_x1 must have matching shape, "
            f"got {tuple(prediction_x1.shape)} and {tuple(target_x1.shape)}"
        )
    if not (len(fft_sizes) == len(hop_lengths) == len(win_lengths)):
        raise ValueError("STFT fft_sizes, hop_lengths, and win_lengths must align")
    if not fft_sizes:
        raise ValueError("At least one STFT resolution is required")

    pred_audio, _ = _unpatch_signal_to_audio(prediction_x1, mask_dt=mask_dt)
    target_audio, _ = _unpatch_signal_to_audio(target_x1, mask_dt=mask_dt)
    if pred_audio.shape[-1] <= 0:
        raise ValueError("Cannot compute STFT loss on empty audio")

    total = pred_audio.new_zeros(())
    used = 0
    for n_fft, hop_length, win_length in zip(fft_sizes, hop_lengths, win_lengths):
        n_fft_i = int(n_fft)
        hop_i = int(hop_length)
        win_i = int(win_length)
        if n_fft_i <= 0 or hop_i <= 0 or win_i <= 0:
            raise ValueError("STFT fft_sizes, hop_lengths, and win_lengths must be > 0")
        if win_i > n_fft_i:
            raise ValueError("STFT win_length cannot exceed n_fft")

        pad_amount = max(0, n_fft_i - pred_audio.shape[-1])
        pred_in = F.pad(pred_audio, (0, pad_amount)) if pad_amount else pred_audio
        target_in = F.pad(target_audio, (0, pad_amount)) if pad_amount else target_audio
        window = torch.hann_window(
            win_i,
            periodic=True,
            dtype=pred_in.dtype,
            device=pred_in.device,
        )
        pred_stft = torch.stft(
            pred_in,
            n_fft=n_fft_i,
            hop_length=hop_i,
            win_length=win_i,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        target_stft = torch.stft(
            target_in,
            n_fft=n_fft_i,
            hop_length=hop_i,
            win_length=win_i,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        resolution_loss = pred_audio.new_zeros(())
        if float(spectral_convergence_weight) > 0.0:
            numerator = torch.linalg.vector_norm(target_mag - pred_mag, ord=2)
            denominator = torch.linalg.vector_norm(target_mag, ord=2).clamp_min(
                float(eps)
            )
            resolution_loss = resolution_loss + float(spectral_convergence_weight) * (
                numerator / denominator
            )
        if float(log_magnitude_weight) > 0.0:
            log_diff = (
                torch.log(pred_mag + float(eps)) - torch.log(target_mag + float(eps))
            ).abs()
            resolution_loss = (
                resolution_loss + float(log_magnitude_weight) * log_diff.mean()
            )
        total = total + resolution_loss
        used += 1

    return total / float(max(used, 1))


def _binaural_stft_cue_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    fft_sizes: list[int],
    hop_lengths: list[int],
    win_lengths: list[int],
    ild_weight: float = 0.0,
    ipd_weight: float = 0.0,
    eps: float = 1e-7,
    energy_weight_power: float = 0.3,
) -> torch.Tensor:
    """Compare binaural ILD/IPD cues for direct two-channel headphone targets."""
    if float(ild_weight) <= 0.0 and float(ipd_weight) <= 0.0:
        return prediction_x1.new_zeros(())
    if prediction_x1.shape != target_x1.shape:
        raise ValueError(
            "prediction_x1 and target_x1 must have matching shape, "
            f"got {tuple(prediction_x1.shape)} and {tuple(target_x1.shape)}"
        )
    if prediction_x1.shape[1] != 2:
        return prediction_x1.new_zeros(())
    if not (len(fft_sizes) == len(hop_lengths) == len(win_lengths)):
        raise ValueError(
            "binaural STFT fft_sizes, hop_lengths, and win_lengths must align"
        )
    if not fft_sizes:
        raise ValueError("At least one binaural STFT resolution is required")

    pred_audio = _unpatch_binaural_signal_to_audio(prediction_x1, mask_dt=mask_dt)
    target_audio = _unpatch_binaural_signal_to_audio(target_x1, mask_dt=mask_dt)
    total = pred_audio.new_zeros(())
    used = 0
    eps_f = float(max(eps, 1e-12))

    for n_fft, hop_length, win_length in zip(fft_sizes, hop_lengths, win_lengths):
        n_fft_i = int(n_fft)
        hop_i = int(hop_length)
        win_i = int(win_length)
        if n_fft_i <= 0 or hop_i <= 0 or win_i <= 0:
            raise ValueError(
                "binaural STFT fft_sizes, hop_lengths, and win_lengths must be > 0"
            )
        if win_i > n_fft_i:
            raise ValueError("binaural STFT win_length cannot exceed n_fft")

        pad_amount = max(0, n_fft_i - int(pred_audio.shape[-1]))
        pred_in = F.pad(pred_audio, (0, pad_amount)) if pad_amount else pred_audio
        target_in = F.pad(target_audio, (0, pad_amount)) if pad_amount else target_audio
        batch_size, channels, num_samples = pred_in.shape
        window = torch.hann_window(
            win_i,
            periodic=True,
            dtype=pred_in.dtype,
            device=pred_in.device,
        )
        pred_stft = torch.stft(
            pred_in.reshape(batch_size * channels, num_samples),
            n_fft=n_fft_i,
            hop_length=hop_i,
            win_length=win_i,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).reshape(batch_size, channels, n_fft_i // 2 + 1, -1)
        target_stft = torch.stft(
            target_in.reshape(batch_size * channels, num_samples),
            n_fft=n_fft_i,
            hop_length=hop_i,
            win_length=win_i,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).reshape(batch_size, channels, n_fft_i // 2 + 1, -1)

        target_energy = target_stft.abs().sum(dim=1).clamp_min(eps_f)
        weight = target_energy.pow(float(energy_weight_power))
        weight = weight / weight.mean(dim=(-2, -1), keepdim=True).clamp_min(eps_f)
        weight = weight.detach()

        resolution_loss = pred_audio.new_zeros(())
        if float(ild_weight) > 0.0:
            pred_ild = torch.log(pred_stft[:, 0].abs().clamp_min(eps_f)) - torch.log(
                pred_stft[:, 1].abs().clamp_min(eps_f)
            )
            target_ild = torch.log(
                target_stft[:, 0].abs().clamp_min(eps_f)
            ) - torch.log(target_stft[:, 1].abs().clamp_min(eps_f))
            resolution_loss = (
                resolution_loss
                + float(ild_weight) * ((pred_ild - target_ild).abs() * weight).mean()
            )

        if float(ipd_weight) > 0.0:
            pred_cross = pred_stft[:, 0] * pred_stft[:, 1].conj()
            target_cross = target_stft[:, 0] * target_stft[:, 1].conj()
            pred_unit = pred_cross / pred_cross.abs().clamp_min(eps_f)
            target_unit = target_cross / target_cross.abs().clamp_min(eps_f)
            ipd = (1.0 - (pred_unit * target_unit.conj()).real) * weight
            resolution_loss = resolution_loss + float(ipd_weight) * ipd.mean()

        total = total + resolution_loss
        used += 1

    return total / float(max(used, 1))


def _normalized_binaural_frames(
    audio: torch.Tensor,
    *,
    frame_length: int,
    hop_length: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized left/right frames and detached frame weights."""
    if audio.shape[-1] < frame_length:
        audio = F.pad(audio, (0, frame_length - audio.shape[-1]))
    frames = audio.unfold(dimension=-1, size=frame_length, step=hop_length)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    left = frames[:, 0]
    right = frames[:, 1]
    frame_energy = (left.pow(2).mean(dim=-1) + right.pow(2).mean(dim=-1)).detach()
    frame_weight = frame_energy / frame_energy.mean(dim=-1, keepdim=True).clamp_min(eps)
    left = left / left.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    right = right / right.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return left, right, frame_weight


def _binaural_ccf_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    sample_rate: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    max_delay_ms: float = 0.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compare short-window interaural cross-correlation profiles."""
    if prediction_x1.shape != target_x1.shape:
        raise ValueError(
            "prediction_x1 and target_x1 must have matching shape, "
            f"got {tuple(prediction_x1.shape)} and {tuple(target_x1.shape)}"
        )
    if prediction_x1.shape[1] != 2:
        return prediction_x1.new_zeros(())
    sample_rate_i = int(sample_rate)
    if sample_rate_i <= 0:
        raise ValueError("sample_rate must be > 0 for binaural CCF loss")

    frame_length = max(8, int(round(sample_rate_i * float(frame_ms) / 1000.0)))
    hop_length = max(1, int(round(sample_rate_i * float(hop_ms) / 1000.0)))
    max_lag = max(1, int(round(sample_rate_i * float(max_delay_ms) / 1000.0)))
    eps_f = float(max(eps, 1e-12))

    pred_audio = _unpatch_binaural_signal_to_audio(prediction_x1, mask_dt=mask_dt)
    target_audio = _unpatch_binaural_signal_to_audio(target_x1, mask_dt=mask_dt)
    pred_left, pred_right, _ = _normalized_binaural_frames(
        pred_audio,
        frame_length=frame_length,
        hop_length=hop_length,
        eps=eps_f,
    )
    target_left, target_right, frame_weight = _normalized_binaural_frames(
        target_audio,
        frame_length=frame_length,
        hop_length=hop_length,
        eps=eps_f,
    )
    pred_right_pad = F.pad(pred_right, (max_lag, max_lag))
    target_right_pad = F.pad(target_right, (max_lag, max_lag))

    total = pred_audio.new_zeros(())
    for offset in range(2 * max_lag + 1):
        pred_shift = pred_right_pad[..., offset : offset + frame_length]
        target_shift = target_right_pad[..., offset : offset + frame_length]
        pred_ccf = (pred_left * pred_shift).mean(dim=-1)
        target_ccf = (target_left * target_shift).mean(dim=-1)
        total = total + ((pred_ccf - target_ccf).abs() * frame_weight).mean()

    return total / float(2 * max_lag + 1)


def _frame_rms_ild_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    frame_size: int = 2048,
    hop_size: int = 1024,
    eps: float = 1e-6,
    silence_threshold: float = 1e-4,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Compare frame-level left/right log-RMS ratios for headphone targets."""
    if prediction_x1.shape != target_x1.shape:
        raise ValueError(
            "prediction_x1 and target_x1 must have matching shape, "
            f"got {tuple(prediction_x1.shape)} and {tuple(target_x1.shape)}"
        )
    if prediction_x1.shape[1] != 2:
        return prediction_x1.new_zeros(())
    frame_size_i = int(frame_size)
    hop_size_i = int(hop_size)
    if frame_size_i <= 0 or hop_size_i <= 0:
        raise ValueError("frame RMS ILD frame_size and hop_size must be > 0")
    eps_f = float(max(eps, 1e-12))

    def frame_rms(channel_audio: torch.Tensor) -> torch.Tensor:
        if channel_audio.shape[-1] < frame_size_i:
            channel_audio = F.pad(
                channel_audio,
                (0, frame_size_i - channel_audio.shape[-1]),
            )
        frames = channel_audio.unfold(
            dimension=-1,
            size=frame_size_i,
            step=hop_size_i,
        )
        return frames.pow(2).mean(dim=-1).add(eps_f).sqrt()

    pred_audio = _unpatch_binaural_signal_to_audio(prediction_x1, mask_dt=mask_dt)
    target_audio = _unpatch_binaural_signal_to_audio(target_x1, mask_dt=mask_dt)

    pred_l_rms = frame_rms(pred_audio[:, 0])
    pred_r_rms = frame_rms(pred_audio[:, 1])
    targ_l_rms = frame_rms(target_audio[:, 0])
    targ_r_rms = frame_rms(target_audio[:, 1])

    pred_ild = torch.log(pred_l_rms.clamp_min(eps_f)) - torch.log(
        pred_r_rms.clamp_min(eps_f)
    )
    targ_ild = torch.log(targ_l_rms.clamp_min(eps_f)) - torch.log(
        targ_r_rms.clamp_min(eps_f)
    )

    target_energy = 0.5 * (targ_l_rms + targ_r_rms)
    active = target_energy > float(silence_threshold)
    if not bool(active.any().item()):
        return prediction_x1.new_zeros(())

    weight = target_energy / target_energy[active].mean().clamp_min(eps_f)
    weight = weight.clamp(max=float(max_weight)).detach()
    return ((pred_ild - targ_ild).abs() * weight)[active].mean()


def _loss_by_name(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    charbonnier_eps: float,
) -> torch.Tensor:
    loss_name = str(loss_type).strip().lower()
    if loss_name in {"mse", "l2"}:
        return (prediction.float() - target.float()).pow(2).mean()
    if loss_name in {"l1", "mae"}:
        return (prediction.float() - target.float()).abs().mean()
    if loss_name in {"charbonnier", "charb"}:
        return _charbonnier_loss(
            prediction,
            target,
            eps=float(charbonnier_eps),
            reduction="mean",
        )
    raise ValueError("loss_type must be one of: mse, l2, l1, mae, charbonnier")


def _mid_side_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    loss_type: str = "charbonnier",
    charbonnier_eps: float = 1e-3,
    mid_weight: float = 0.0,
    side_weight: float = 1.0,
) -> torch.Tensor:
    """Compare headphone mid and side waveforms with a selectable loss."""
    if prediction_x1.shape != target_x1.shape:
        raise ValueError(
            "prediction_x1 and target_x1 must have matching shape, "
            f"got {tuple(prediction_x1.shape)} and {tuple(target_x1.shape)}"
        )
    if prediction_x1.shape[1] != 2:
        return prediction_x1.new_zeros(())

    pred_audio = _unpatch_binaural_signal_to_audio(prediction_x1, mask_dt=mask_dt)
    target_audio = _unpatch_binaural_signal_to_audio(target_x1, mask_dt=mask_dt)

    pred_mid = 0.5 * (pred_audio[:, 0] + pred_audio[:, 1])
    pred_side = 0.5 * (pred_audio[:, 0] - pred_audio[:, 1])
    targ_mid = 0.5 * (target_audio[:, 0] + target_audio[:, 1])
    targ_side = 0.5 * (target_audio[:, 0] - target_audio[:, 1])

    total = pred_audio.new_zeros(())
    if float(mid_weight) > 0.0:
        total = total + float(mid_weight) * _loss_by_name(
            pred_mid,
            targ_mid,
            loss_type=loss_type,
            charbonnier_eps=charbonnier_eps,
        )
    if float(side_weight) > 0.0:
        total = total + float(side_weight) * _loss_by_name(
            pred_side,
            targ_side,
            loss_type=loss_type,
            charbonnier_eps=charbonnier_eps,
        )
    return total


def _binaural_cue_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    sample_rate: int,
    fft_sizes: list[int],
    hop_lengths: list[int],
    win_lengths: list[int],
    ild_weight: float = 0.0,
    ipd_weight: float = 0.0,
    ccf_weight: float = 0.0,
    frame_ild_weight: float = 0.0,
    frame_ild_frame_size: int = 2048,
    frame_ild_hop_size: int = 1024,
    frame_ild_silence_threshold: float = 1e-4,
    frame_ild_max_weight: float = 4.0,
    mid_side_weight: float = 0.0,
    mid_side_loss_type: str = "charbonnier",
    mid_side_mid_weight: float = 0.0,
    mid_side_side_weight: float = 1.0,
    mid_side_charbonnier_eps: float = 1e-3,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Aggregate optional binaural cue losses for direct headphone targets."""
    if prediction_x1.shape[1] != 2:
        return prediction_x1.new_zeros(())
    total = prediction_x1.new_zeros(())
    if float(ild_weight) > 0.0 or float(ipd_weight) > 0.0:
        total = total + _binaural_stft_cue_loss(
            prediction_x1=prediction_x1,
            target_x1=target_x1,
            mask_dt=mask_dt,
            fft_sizes=fft_sizes,
            hop_lengths=hop_lengths,
            win_lengths=win_lengths,
            ild_weight=float(ild_weight),
            ipd_weight=float(ipd_weight),
            eps=float(eps),
        )
    if float(ccf_weight) > 0.0:
        total = total + float(ccf_weight) * _binaural_ccf_loss(
            prediction_x1=prediction_x1,
            target_x1=target_x1,
            mask_dt=mask_dt,
            sample_rate=int(sample_rate),
            eps=float(eps),
        )
    if float(frame_ild_weight) > 0.0:
        total = total + float(frame_ild_weight) * _frame_rms_ild_loss(
            prediction_x1=prediction_x1,
            target_x1=target_x1,
            mask_dt=mask_dt,
            frame_size=int(frame_ild_frame_size),
            hop_size=int(frame_ild_hop_size),
            eps=float(eps),
            silence_threshold=float(frame_ild_silence_threshold),
            max_weight=float(frame_ild_max_weight),
        )
    if float(mid_side_weight) > 0.0:
        total = total + float(mid_side_weight) * _mid_side_loss(
            prediction_x1=prediction_x1,
            target_x1=target_x1,
            mask_dt=mask_dt,
            loss_type=mid_side_loss_type,
            charbonnier_eps=float(mid_side_charbonnier_eps),
            mid_weight=float(mid_side_mid_weight),
            side_weight=float(mid_side_side_weight),
        )
    return total


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a triangular mel filterbank and its center frequencies."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if n_fft <= 0 or n_mels <= 0:
        raise ValueError("n_fft and n_mels must be > 0")
    nyquist = float(sample_rate) / 2.0
    f_max_f = nyquist if f_max is None else min(float(f_max), nyquist)
    f_min_f = max(0.0, float(f_min))
    if f_max_f <= f_min_f:
        raise ValueError("perceptual mel f_max must be greater than f_min")

    mel_min = _hz_to_mel(torch.tensor(f_min_f, device=device, dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(f_max_f, device=device, dtype=dtype))
    mel_points = torch.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
        device=device,
        dtype=dtype,
    )
    hz_points = _mel_to_hz(mel_points)
    fft_freqs = torch.linspace(
        0.0,
        nyquist,
        n_fft // 2 + 1,
        device=device,
        dtype=dtype,
    )

    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    freqs = fft_freqs[None, :]

    lower_slope = (freqs - lower) / (center - lower).clamp_min(1e-12)
    upper_slope = (upper - freqs) / (upper - center).clamp_min(1e-12)
    filters = torch.minimum(lower_slope, upper_slope).clamp_min(0.0)
    filters = filters / filters.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return filters, hz_points[1:-1]


def _resolve_perceptual_stereo_signals(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    target_downmix_signal: torch.Tensor | None,
    channel_order: list[str] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve stereo comparison signals for broad perceptual matching."""
    if prediction_x1.shape[1] == 2 and target_x1.shape[1] == 2:
        return prediction_x1, target_x1

    pred_stereo = _downmix_to_stereo(prediction_x1.float(), channel_order=channel_order)
    if target_downmix_signal is not None:
        target_stereo = target_downmix_signal.float()
        if target_stereo.shape[1] == 1:
            target_stereo = target_stereo.expand(-1, 2, -1, -1).contiguous()
        if target_stereo.shape[1] != 2:
            raise ValueError(
                "target_downmix_signal must have one or two channels for perceptual loss"
            )
        return pred_stereo, target_stereo

    target_stereo = _downmix_to_stereo(target_x1.float(), channel_order=channel_order)
    return pred_stereo, target_stereo


def _stereo_log_mel_perceptual_loss(
    *,
    prediction_x1: torch.Tensor,
    target_x1: torch.Tensor,
    mask_dt: torch.Tensor,
    sample_rate: int,
    target_downmix_signal: torch.Tensor | None = None,
    channel_order: list[str] | None = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 40.0,
    f_max: float | None = None,
    band_weight: float = 1.0,
    band_low_hz: float = 150.0,
    band_high_hz: float = 8000.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Log-mel perceptual loss on stereo targets or spatial-to-stereo downmixes."""
    if prediction_x1.shape != target_x1.shape:
        raise ValueError(
            "prediction_x1 and target_x1 must have matching shape, "
            f"got {tuple(prediction_x1.shape)} and {tuple(target_x1.shape)}"
        )
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0 or n_mels <= 0:
        raise ValueError("perceptual FFT/hop/window/mel sizes must be > 0")
    if win_length > n_fft:
        raise ValueError("perceptual win_length cannot exceed n_fft")
    if eps <= 0:
        raise ValueError("perceptual eps must be > 0")

    pred_stereo, target_stereo = _resolve_perceptual_stereo_signals(
        prediction_x1=prediction_x1,
        target_x1=target_x1,
        target_downmix_signal=target_downmix_signal,
        channel_order=channel_order,
    )
    if pred_stereo.shape != target_stereo.shape:
        raise ValueError(
            "perceptual prediction and target stereo shapes do not align: "
            f"{tuple(pred_stereo.shape)} vs {tuple(target_stereo.shape)}"
        )

    pred_audio, _ = _unpatch_signal_to_audio(pred_stereo, mask_dt=mask_dt)
    target_audio, _ = _unpatch_signal_to_audio(target_stereo, mask_dt=mask_dt)
    pad_amount = max(0, int(n_fft) - int(pred_audio.shape[-1]))
    if pad_amount:
        pred_audio = F.pad(pred_audio, (0, pad_amount))
        target_audio = F.pad(target_audio, (0, pad_amount))

    window = torch.hann_window(
        int(win_length),
        periodic=True,
        dtype=pred_audio.dtype,
        device=pred_audio.device,
    )
    pred_mag = (
        torch.stft(
            pred_audio,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .pow(2.0)
    )
    target_mag = (
        torch.stft(
            target_audio,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .pow(2.0)
    )

    filters, mel_centers = _mel_filterbank(
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        f_min=float(f_min),
        f_max=f_max,
        device=pred_audio.device,
        dtype=pred_audio.dtype,
    )
    pred_mel = torch.einsum("mf,nft->nmt", filters, pred_mag)
    target_mel = torch.einsum("mf,nft->nmt", filters, target_mag)
    log_diff = (
        torch.log(pred_mel + float(eps)) - torch.log(target_mel + float(eps))
    ).abs()

    mel_weights = torch.ones_like(mel_centers)
    if float(band_weight) != 1.0:
        band_mask = (mel_centers >= float(band_low_hz)) & (
            mel_centers <= float(band_high_hz)
        )
        mel_weights = torch.where(
            band_mask,
            torch.full_like(mel_weights, float(band_weight)),
            mel_weights,
        )
    weighted = log_diff * mel_weights[None, :, None]
    return weighted.mean()


def _channel_routing_kl_loss(
    prediction_x1: torch.Tensor,  # [B,C,D,T]
    target_x1: torch.Tensor,  # [B,C,D,T]
    mask_dt: torch.Tensor,  # [B,D,T]  (valid_mask * overlap_weight)
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Encourage per-(D,T) energy distribution across channels to match target.

    Uses:
    - p(c|d,t) from target energy
    - q(c|d,t) from prediction energy
    - KL(p || q), averaged with mask_dt.
    """
    pred_sq = prediction_x1.float().pow(2)
    tgt_sq = target_x1.float().pow(2)

    tau = max(float(temperature), 1e-6)
    eps_f = max(float(eps), 1e-12)

    tgt_logits = torch.log(tgt_sq + eps_f) / tau  # [B,C,D,T]
    pred_logits = torch.log(pred_sq + eps_f) / tau

    p = torch.softmax(tgt_logits, dim=1)
    log_p = torch.log_softmax(tgt_logits, dim=1)
    log_q = torch.log_softmax(pred_logits, dim=1)

    kl_dt = torch.sum(p * (log_p - log_q), dim=1)  # [B,D,T]

    w = mask_dt.to(dtype=kl_dt.dtype, device=kl_dt.device)
    denom = torch.clamp(w.sum(), min=1.0)
    return (kl_dt * w).sum() / denom


def _channel_correlation_l1_loss(
    prediction_x1: torch.Tensor,  # [B,C,D,T]
    target_x1: torch.Tensor,  # [B,C,D,T]
    mask_dt: torch.Tensor,  # [B,D,T]  (valid_mask * overlap_weight)
    eps: float = 1e-6,
    offdiag_only: bool = True,
    use_correlation: bool = True,
) -> torch.Tensor:
    """
    Match cross-channel structure between prediction and target.

    Default behavior compares off-diagonal entries of correlation matrices.
    """
    pred = prediction_x1.float()
    tgt = target_x1.float()

    mask = mask_dt[:, None, :, :].to(dtype=pred.dtype, device=pred.device)
    n = torch.clamp(
        mask.sum(dim=(2, 3), keepdim=True), min=max(float(eps), 1e-6)
    )  # [B,1,1,1]

    pred_mean = (pred * mask).sum(dim=(2, 3), keepdim=True) / n
    tgt_mean = (tgt * mask).sum(dim=(2, 3), keepdim=True) / n

    # Use sqrt(weights) so covariance is weighted linearly by mask_dt.
    sqrt_mask = torch.sqrt(mask.clamp_min(0.0))
    pred_centered = (pred - pred_mean) * sqrt_mask
    tgt_centered = (tgt - tgt_mean) * sqrt_mask

    pred_flat = pred_centered.reshape(pred.shape[0], pred.shape[1], -1)  # [B,C,N]
    tgt_flat = tgt_centered.reshape(tgt.shape[0], tgt.shape[1], -1)  # [B,C,N]
    n_scalar = n.reshape(n.shape[0], 1, 1)  # [B,1,1]

    cov_pred = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / n_scalar  # [B,C,C]
    cov_tgt = torch.bmm(tgt_flat, tgt_flat.transpose(1, 2)) / n_scalar  # [B,C,C]

    if use_correlation:
        diag_pred = torch.diagonal(cov_pred, dim1=1, dim2=2).clamp_min(eps)  # [B,C]
        diag_tgt = torch.diagonal(cov_tgt, dim1=1, dim2=2).clamp_min(eps)  # [B,C]

        std_pred = torch.sqrt(diag_pred)  # [B,C]
        std_tgt = torch.sqrt(diag_tgt)  # [B,C]

        denom_pred = std_pred[:, :, None] * std_pred[:, None, :]
        denom_tgt = std_tgt[:, :, None] * std_tgt[:, None, :]

        corr_pred = cov_pred / denom_pred.clamp_min(eps)
        corr_tgt = cov_tgt / denom_tgt.clamp_min(eps)
        mat_pred, mat_tgt = corr_pred, corr_tgt
    else:
        mat_pred, mat_tgt = cov_pred, cov_tgt

    diff = (mat_pred - mat_tgt).abs()  # [B,C,C]
    if offdiag_only:
        c = diff.shape[1]
        eye = torch.eye(c, device=diff.device, dtype=diff.dtype)[None, :, :]
        diff = diff * (1.0 - eye)
        denom = float(max(c * (c - 1), 1))
        return diff.sum() / denom / diff.shape[0]

    return diff.mean()
