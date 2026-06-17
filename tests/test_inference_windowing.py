from __future__ import annotations

import pytest
import torch

from stereo2spatial.inference.windowing import (
    extract_fixed_window,
    fixed_window_specs,
    resolve_fixed_window_frames,
    stitch_fixed_windows,
)


def test_resolve_fixed_window_frames_uses_patch_rate() -> None:
    window_frames, overlap_frames = resolve_fixed_window_frames(
        sample_rate=48_000,
        patch_size=200,
        window_seconds=10.0,
        overlap_seconds=2.0,
    )

    assert window_frames == 2400
    assert overlap_frames == 480


def test_fixed_window_specs_short_sequence_pads_one_window() -> None:
    specs = fixed_window_specs(total_frames=5, window_frames=10, overlap_frames=2)

    assert [(s.start_frame, s.valid_frames, s.padded_frames) for s in specs] == [
        (0, 5, 5)
    ]
    assert specs[0].is_padded is True


def test_fixed_window_specs_exact_window_has_no_padding() -> None:
    specs = fixed_window_specs(total_frames=10, window_frames=10, overlap_frames=2)

    assert [(s.start_frame, s.valid_frames, s.padded_frames) for s in specs] == [
        (0, 10, 0)
    ]
    assert specs[0].is_padded is False


def test_fixed_window_specs_preserves_stride_when_tail_needs_padding() -> None:
    specs = fixed_window_specs(total_frames=19, window_frames=10, overlap_frames=2)

    assert [(s.start_frame, s.valid_frames, s.padded_frames) for s in specs] == [
        (0, 10, 0),
        (8, 10, 0),
        (16, 3, 7),
    ]


def test_fixed_window_specs_does_not_add_extra_window_when_tail_is_covered() -> None:
    specs = fixed_window_specs(total_frames=18, window_frames=10, overlap_frames=2)

    assert [(s.start_frame, s.valid_frames, s.padded_frames) for s in specs] == [
        (0, 10, 0),
        (8, 10, 0),
    ]


def test_extract_fixed_window_pads_final_window() -> None:
    sequence = torch.arange(19, dtype=torch.float32).view(1, 19)
    spec = fixed_window_specs(total_frames=19, window_frames=10, overlap_frames=2)[-1]

    window = extract_fixed_window(sequence, spec, pad_value=-1.0)

    assert window.shape == (1, 10)
    assert torch.equal(window[0, :3], torch.tensor([16.0, 17.0, 18.0]))
    assert torch.equal(window[0, 3:], torch.full((7,), -1.0))


def test_stitch_fixed_windows_reconstructs_sequence_from_extracted_windows() -> None:
    sequence = torch.randn(2, 3, 19)
    specs = fixed_window_specs(total_frames=19, window_frames=10, overlap_frames=2)
    windows = [extract_fixed_window(sequence, spec) for spec in specs]

    stitched = stitch_fixed_windows(windows, specs, overlap_frames=2)

    assert stitched.shape == sequence.shape
    assert torch.allclose(stitched, sequence, atol=1.0e-6)


def test_stitch_fixed_windows_crops_padding_from_short_sequence() -> None:
    sequence = torch.randn(2, 5)
    specs = fixed_window_specs(total_frames=5, window_frames=10, overlap_frames=2)
    windows = [extract_fixed_window(sequence, spec) for spec in specs]

    stitched = stitch_fixed_windows(windows, specs, overlap_frames=2)

    assert stitched.shape == sequence.shape
    assert torch.allclose(stitched, sequence, atol=1.0e-6)


def test_fixed_window_specs_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_frames"):
        fixed_window_specs(total_frames=10, window_frames=10, overlap_frames=10)
