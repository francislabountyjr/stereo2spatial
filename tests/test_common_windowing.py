from __future__ import annotations

import torch

from stereo2spatial.common.windowing import chunk_weight, segment_starts


def test_segment_starts_covers_tail() -> None:
    starts = segment_starts(total_frames=10, window_frames=4, stride_frames=3)
    assert starts == [0, 3, 6]


def test_segment_starts_single_window() -> None:
    starts = segment_starts(total_frames=3, window_frames=4, stride_frames=1)
    assert starts == [0]


def test_chunk_weight_handles_overlap() -> None:
    weights = chunk_weight(
        chunk_length=6,
        overlap_frames=3,
        is_first=False,
        is_last=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.5, 0.0], dtype=torch.float32)
    assert torch.allclose(weights, expected)


def test_chunk_weight_no_overlap_is_all_ones() -> None:
    weights = chunk_weight(
        chunk_length=5,
        overlap_frames=0,
        is_first=True,
        is_last=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(weights, torch.ones(5, dtype=torch.float32))
