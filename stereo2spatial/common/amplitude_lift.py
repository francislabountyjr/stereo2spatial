"""Shared training-space amplitude lifting for raw waveform modeling."""

from __future__ import annotations

import torch


def compute_shared_rms_gain(
    reference: torch.Tensor,
    *,
    target_rms: float,
    eps: float,
) -> torch.Tensor:
    """Return one RMS-normalization gain per item, preserving channel balances."""
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


def apply_amplitude_lift(
    audio: torch.Tensor,
    *,
    gain: torch.Tensor,
    scale: float,
    clip_value: float | None = 1.0,
) -> torch.Tensor:
    """Apply shared-gain RMS normalization, optional clipping, and global scale."""
    lifted = audio.float().mul(gain)
    if clip_value is not None and float(clip_value) > 0.0:
        clip = float(clip_value)
        lifted = lifted.clamp(-clip, clip)
    return lifted.mul(float(scale))


def undo_amplitude_lift(
    audio: torch.Tensor,
    *,
    gain: torch.Tensor,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """Map model-space lifted audio back toward raw waveform scale."""
    denom = gain.float().mul(float(scale)).clamp_min(float(eps))
    return audio.float() / denom


__all__ = [
    "apply_amplitude_lift",
    "compute_shared_rms_gain",
    "undo_amplitude_lift",
]
