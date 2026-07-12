"""Shared training-space amplitude lifting for raw waveform modeling."""

from __future__ import annotations

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional runtime fallback
    torchaudio = None


def compute_shared_rms_gain(
    reference: torch.Tensor,
    *,
    target_rms: float,
    eps: float,
) -> torch.Tensor:
    """Return one unclamped RMS-normalization gain per item, preserving channel balances."""
    if reference.dim() == 2:
        rms = reference.float().pow(2).mean(dim=(0, 1), keepdim=True).add(eps).sqrt()
    elif reference.dim() == 3:
        rms = reference.float().pow(2).mean(dim=(1, 2), keepdim=True).add(eps).sqrt()
    else:
        raise ValueError(
            f"reference must have shape [C,S] or [B,C,S], got {tuple(reference.shape)}"
        )
    return torch.as_tensor(
        float(target_rms), dtype=rms.dtype, device=rms.device
    ) / rms


def resolve_amplitude_lift_gain(
    reference: torch.Tensor,
    *,
    mode: str,
    target_rms: float,
    eps: float,
) -> torch.Tensor:
    """Return the raw/unclamped model-space lift gain for RMS-normalized or pure-scale mode."""
    mode_name = str(mode).strip().lower()

    if mode_name == "scale":
        return torch.ones(
            _gain_shape_for_reference(reference),
            dtype=torch.float32,
            device=reference.device,
        )

    if mode_name in {"rms", "wavflow"}:
        return compute_shared_rms_gain(
            reference,
            target_rms=target_rms,
            eps=eps,
        )

    raise ValueError("amplitude lift mode must be one of: rms, scale, wavflow")


def _gain_shape_for_reference(reference: torch.Tensor) -> tuple[int, ...]:
    """Return the broadcastable gain shape for a reference audio tensor."""
    if reference.dim() == 2:
        return (1, 1)
    if reference.dim() == 3:
        return (reference.shape[0], 1, 1)
    raise ValueError(
        f"reference must have shape [C,S] or [B,C,S], got {tuple(reference.shape)}"
    )


def resolve_effective_amplitude_gain(
    gain: torch.Tensor,
    *,
    scale: float,
    gain_power: float = 1.0,
    gain_clip_value: float | None = None,
    gain_min_value: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Resolve the actual gain used in model space.

    This is the single function apply/undo/conditioning should all call.

    gain_power:
      1.0 = full RMS normalization; 0.5 = soft/partial; 0.0 = ignore RMS gain.

    gain_clip_value:
      max gain clamp (replaces old waveform clipping).

    gain_min_value:
      optional min gain clamp, avoids crushing loud tracks too aggressively.

    scale:
      global multiplier applied after all gain transforms.
    """
    effective = gain.float().clamp_min(float(eps))

    if float(gain_power) != 1.0:
        effective = effective.pow(float(gain_power))

    if gain_min_value is not None and float(gain_min_value) > 0.0:
        effective = effective.clamp_min(float(gain_min_value))

    if gain_clip_value is not None and float(gain_clip_value) > 0.0:
        effective = effective.clamp_max(float(gain_clip_value))

    return effective * float(scale)


def amplitude_lift_log_gain(
    gain: torch.Tensor,
    *,
    scale: float,
    gain_power: float = 1.0,
    gain_clip_value: float | None = None,
    gain_min_value: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return log of the effective gain for model conditioning.

    Always condition on this — never on the raw gain — so the model
    sees the actual transform that was applied.
    """
    effective_gain = resolve_effective_amplitude_gain(
        gain,
        scale=scale,
        gain_power=gain_power,
        gain_clip_value=gain_clip_value,
        gain_min_value=gain_min_value,
        eps=eps,
    )
    return torch.log(effective_gain.clamp_min(float(eps)))


def apply_amplitude_lift(
    audio: torch.Tensor,
    *,
    gain: torch.Tensor,
    scale: float,
    clip_value: float | None = None,
    gain_power: float = 1.0,
    gain_min_value: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply shared-gain amplitude normalization and global scale.

    clip_value is now a max *gain* clamp, not a waveform sample clamp.
    This function does NOT clamp waveform values.
    """
    effective_gain = resolve_effective_amplitude_gain(
        gain,
        scale=scale,
        gain_power=gain_power,
        gain_clip_value=clip_value,
        gain_min_value=gain_min_value,
        eps=eps,
    )
    return audio.float() * effective_gain


def undo_amplitude_lift(
    audio: torch.Tensor,
    *,
    gain: torch.Tensor,
    scale: float,
    eps: float,
    clip_value: float | None = None,
    gain_power: float = 1.0,
    gain_min_value: float | None = None,
) -> torch.Tensor:
    """Map model-space lifted audio back toward raw waveform scale."""
    effective_gain = resolve_effective_amplitude_gain(
        gain,
        scale=scale,
        gain_power=gain_power,
        gain_clip_value=clip_value,
        gain_min_value=gain_min_value,
        eps=eps,
    )
    return audio.float() / effective_gain.clamp_min(float(eps))


def _shared_rms(audio: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Return one RMS value per item over channels and samples."""
    if audio.dim() == 2:
        return audio.float().pow(2).mean(dim=(0, 1), keepdim=True).add(eps).sqrt()
    if audio.dim() == 3:
        return audio.float().pow(2).mean(dim=(1, 2), keepdim=True).add(eps).sqrt()
    raise ValueError(f"audio must have shape [C,S] or [B,C,S], got {tuple(audio.shape)}")


def _shared_peak(audio: torch.Tensor) -> torch.Tensor:
    """Return one absolute peak value per item over channels and samples."""
    if audio.dim() == 2:
        return audio.float().abs().amax(dim=(0, 1), keepdim=True)
    if audio.dim() == 3:
        return audio.float().abs().amax(dim=(1, 2), keepdim=True)
    raise ValueError(f"audio must have shape [C,S] or [B,C,S], got {tuple(audio.shape)}")


def wavflow_target_transform(
    audio: torch.Tensor,
    *,
    target_rms: float = 0.33,
    scale: float = 3.0,
    peak_limit: float = 1.0,
    peak_rescale_min_rms: float = 0.3,
    waveform_clamp: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply WavFlow-style target lifting and return ``(audio, effective_gain)``."""
    gain = compute_shared_rms_gain(audio, target_rms=target_rms, eps=eps)
    normalized = audio.float() * gain
    peak = _shared_peak(normalized)
    rms_after_gain = _shared_rms(normalized, eps=eps)
    peak_scale = torch.as_tensor(float(peak_limit), device=peak.device) / peak.clamp_min(
        float(eps)
    )
    new_rms = rms_after_gain * peak_scale
    should_peak_rescale = (peak > float(peak_limit)) & (
        new_rms > float(peak_rescale_min_rms)
    )
    peak_scale = torch.where(should_peak_rescale, peak_scale, torch.ones_like(peak))
    normalized = normalized * peak_scale
    effective_gain = gain * peak_scale * float(scale)
    if waveform_clamp:
        normalized = normalized.clamp(-float(peak_limit), float(peak_limit))
    return normalized * float(scale), effective_gain


def wavflow_source_transform(
    audio: torch.Tensor,
    *,
    target_rms: float = 0.33,
    scale: float = 3.0,
    peak_limit: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-normalize source audio, then peak-rescale instead of clipping."""
    gain = compute_shared_rms_gain(audio, target_rms=target_rms, eps=eps)
    normalized = audio.float() * gain
    peak = _shared_peak(normalized)
    peak_scale = torch.as_tensor(float(peak_limit), device=peak.device) / peak.clamp_min(
        float(eps)
    )
    peak_scale = torch.where(peak > float(peak_limit), peak_scale, torch.ones_like(peak))
    effective_gain = gain * peak_scale * float(scale)
    return normalized * peak_scale * float(scale), effective_gain


def wavflow_target_clip_stats(
    audio: torch.Tensor,
    *,
    target_rms: float = 0.33,
    peak_limit: float = 1.0,
    peak_rescale_min_rms: float = 0.3,
    eps: float = 1e-8,
) -> dict[str, float | int | bool]:
    """Return target-clamp stats using the same pre-clamp path as WavFlow mode."""
    gain = compute_shared_rms_gain(audio, target_rms=target_rms, eps=eps)
    normalized = audio.float() * gain
    peak_after_rms = float(_shared_peak(normalized).flatten()[0].item())
    rms_after_gain = float(_shared_rms(normalized, eps=eps).flatten()[0].item())
    peak_rescale_applied = False
    peak_scale = 1.0
    if peak_after_rms > float(peak_limit):
        candidate_scale = float(peak_limit) / max(peak_after_rms, float(eps))
        if rms_after_gain * candidate_scale > float(peak_rescale_min_rms):
            peak_rescale_applied = True
            peak_scale = candidate_scale
            normalized = normalized * peak_scale
    pre_clamp_peak = float(_shared_peak(normalized).flatten()[0].item())
    would_clip_mask = normalized.abs() > float(peak_limit)
    return {
        "rms_gain": float(gain.flatten()[0].item()),
        "peak_after_rms": peak_after_rms,
        "rms_after_gain": rms_after_gain,
        "peak_rescale_applied": peak_rescale_applied,
        "peak_rescale_scale": peak_scale,
        "pre_clamp_peak": pre_clamp_peak,
        "would_clip": bool(would_clip_mask.any().item()),
        "clipped_samples": int(would_clip_mask.sum().item()),
    }


def wavflow_source_peak_stats(
    audio: torch.Tensor,
    *,
    target_rms: float = 0.33,
    peak_limit: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, float | bool]:
    """Return source peak-rescale stats using the same path as WavFlow mode."""
    gain = compute_shared_rms_gain(audio, target_rms=target_rms, eps=eps)
    normalized = audio.float() * gain
    peak_after_rms = float(_shared_peak(normalized).flatten()[0].item())
    scale = 1.0
    if peak_after_rms > float(peak_limit):
        scale = float(peak_limit) / max(peak_after_rms, float(eps))
    return {
        "rms_gain": float(gain.flatten()[0].item()),
        "peak_after_rms": peak_after_rms,
        "peak_rescale_required": peak_after_rms > float(peak_limit),
        "peak_rescale_scale": scale,
    }


def normalize_lufs(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    target_lufs: float = -23.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize integrated loudness to ``target_lufs`` when measurable."""
    audio_f = audio.float()
    loudness: torch.Tensor | None = None
    if torchaudio is not None and hasattr(torchaudio.functional, "loudness"):
        try:
            loudness = torchaudio.functional.loudness(audio_f.cpu(), int(sample_rate))
        except Exception:
            loudness = None
    if loudness is None or not bool(torch.isfinite(loudness).all().item()):
        rms = audio_f.pow(2).mean().sqrt()
        if not bool(torch.isfinite(rms).item()) or float(rms.item()) <= float(eps):
            return audio_f
        loudness = 20.0 * torch.log10(rms.clamp_min(float(eps)))

    gain_db = float(target_lufs) - float(loudness.item())
    gain = 10.0 ** (gain_db / 20.0)
    return audio_f * float(gain)


def undo_wavflow_output_lift(
    audio: torch.Tensor,
    *,
    scale: float,
    sample_rate: int,
    target_lufs: float = -23.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply WavFlow inference inverse: divide global scale, then LUFS-normalize."""
    decoded = audio.float() / max(float(scale), float(eps))
    return normalize_lufs(
        decoded,
        sample_rate=int(sample_rate),
        target_lufs=float(target_lufs),
        eps=float(eps),
    )


__all__ = [
    "amplitude_lift_log_gain",
    "apply_amplitude_lift",
    "compute_shared_rms_gain",
    "resolve_amplitude_lift_gain",
    "resolve_effective_amplitude_gain",
    "normalize_lufs",
    "undo_amplitude_lift",
    "undo_wavflow_output_lift",
    "wavflow_source_peak_stats",
    "wavflow_source_transform",
    "wavflow_target_clip_stats",
    "wavflow_target_transform",
]
