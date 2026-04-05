"""Window/chunk utilities for overlapping temporal processing."""

from __future__ import annotations

import torch

from stereo2spatial.common.windowing import (
    chunk_weight as _common_chunk_weight,
)
from stereo2spatial.common.windowing import (
    segment_starts as _common_segment_starts,
)


def _segment_starts(
    total_frames: int,
    window_frames: int,
    stride_frames: int,
) -> list[int]:
    """Backward-compatible wrapper over shared segment start computation."""
    return _common_segment_starts(
        total_frames=total_frames,
        window_frames=window_frames,
        stride_frames=stride_frames,
    )


def _chunk_weight(
    chunk_length: int,
    overlap_frames: int,
    is_first: bool,
    is_last: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Backward-compatible wrapper over shared chunk overlap weights."""
    return _common_chunk_weight(
        chunk_length=chunk_length,
        overlap_frames=overlap_frames,
        is_first=is_first,
        is_last=is_last,
        device=device,
        dtype=dtype,
    )


def _build_window_metadata(
    frame_lengths: list[int],
    window_frames: int,
    overlap_frames: int,
) -> dict[int, tuple[list[int], list[torch.Tensor]]]:
    """Precompute segment starts and overlap weights for each frame length."""
    unique_lengths = sorted(
        {int(length) for length in frame_lengths if int(length) > 0}
    )
    stride_frames = window_frames - overlap_frames
    cpu = torch.device("cpu")

    metadata: dict[int, tuple[list[int], list[torch.Tensor]]] = {}
    for total_frames in unique_lengths:
        starts = _segment_starts(
            total_frames=total_frames,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )
        weights = [
            _chunk_weight(
                chunk_length=window_frames,
                overlap_frames=overlap_frames,
                is_first=(idx == 0),
                is_last=(idx == len(starts) - 1),
                device=cpu,
                dtype=torch.float32,
            )
            for idx in range(len(starts))
        ]
        metadata[total_frames] = (starts, weights)
    return metadata
