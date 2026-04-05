"""Shared temporal window/chunk utilities."""

from __future__ import annotations

import torch


def segment_starts(
    total_frames: int,
    window_frames: int,
    stride_frames: int,
) -> list[int]:
    """Return left-aligned window starts with guaranteed tail coverage."""
    if total_frames <= window_frames:
        return [0]

    starts = list(range(0, total_frames - window_frames + 1, stride_frames))
    last_start = total_frames - window_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def chunk_weight(
    chunk_length: int,
    overlap_frames: int,
    is_first: bool,
    is_last: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build triangular overlap-add weights for a chunk."""
    weight = torch.ones(chunk_length, dtype=dtype, device=device)
    if overlap_frames <= 0:
        return weight

    fade_len = min(overlap_frames, chunk_length)
    if fade_len <= 1:
        return weight

    if not is_first:
        weight[:fade_len] *= torch.linspace(0.0, 1.0, fade_len, dtype=dtype, device=device)
    if not is_last:
        weight[-fade_len:] *= torch.linspace(1.0, 0.0, fade_len, dtype=dtype, device=device)
    return weight
