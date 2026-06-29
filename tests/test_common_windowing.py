from __future__ import annotations

import torch

from stereo2spatial.common.windowing import chunk_weight, segment_starts


def test_segment_starts_covers_tail() -> None:
    starts = segment_starts(total_frames=10, window_frames=4, stride_frames=3)
    assert starts == [0, 3, 6]


def test_segment_starts_keeps_regular_stride_for_near_tail() -> None:
    starts = segment_starts(total_frames=1688, window_frames=562, stride_frames=374)
    assert starts == [0, 374, 748, 1122, 1496]


def test_segment_starts_appends_regular_padded_tail_window() -> None:
    starts = segment_starts(total_frames=15, window_frames=4, stride_frames=3)
    assert starts == [0, 3, 6, 9, 12]


def test_segment_starts_does_not_shift_final_window_backward() -> None:
    starts = segment_starts(total_frames=240, window_frames=10, stride_frames=8)
    assert starts == list(range(0, 233, 8))


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
