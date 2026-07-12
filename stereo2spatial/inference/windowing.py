"""Fixed-window utilities for inference scheduling and stitching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stereo2spatial.common.windowing import chunk_weight


@dataclass(frozen=True)
class FixedWindowSpec:
    """One fixed-size inference window over a longer sequence."""

    index: int
    start_frame: int
    valid_frames: int
    window_frames: int
    total_frames: int

    @property
    def end_frame(self) -> int:
        """Exclusive end frame in the unpadded original sequence."""
        return self.start_frame + self.valid_frames

    @property
    def padded_frames(self) -> int:
        """Number of padded frames required to reach ``window_frames``."""
        return self.window_frames - self.valid_frames

    @property
    def is_padded(self) -> bool:
        """Whether this window contains right-side padding."""
        return self.padded_frames > 0


def resolve_fixed_window_frames(
    *,
    sample_rate: int,
    patch_size: int,
    window_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
) -> tuple[int, int]:
    """Return fixed window and overlap lengths in patch frames."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")

    patch_fps = float(sample_rate) / float(patch_size)
    window_frames = max(1, int(round(float(window_seconds) * patch_fps)))
    overlap_frames = int(round(float(overlap_seconds) * patch_fps))
    overlap_frames = min(overlap_frames, max(0, window_frames - 1))
    return window_frames, overlap_frames


def fixed_window_specs(
    *,
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
) -> list[FixedWindowSpec]:
    """Build fixed-stride windows, padding only the final uncovered tail."""
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")
    if overlap_frames >= window_frames:
        raise ValueError("overlap_frames must be smaller than window_frames")

    stride_frames = window_frames - overlap_frames
    starts = [0]
    while starts[-1] + window_frames < total_frames:
        starts.append(starts[-1] + stride_frames)

    specs: list[FixedWindowSpec] = []
    for index, start_frame in enumerate(starts):
        valid_frames = min(window_frames, total_frames - start_frame)
        specs.append(
            FixedWindowSpec(
                index=index,
                start_frame=start_frame,
                valid_frames=valid_frames,
                window_frames=window_frames,
                total_frames=total_frames,
            )
        )
    return specs


def extract_fixed_window(
    sequence: torch.Tensor,
    spec: FixedWindowSpec,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Extract one fixed-size window from a tensor with frames on the last axis."""
    if sequence.dim() == 0:
        raise ValueError("sequence must have at least one dimension")
    if sequence.shape[-1] != spec.total_frames:
        raise ValueError(
            "sequence last dimension must match spec.total_frames "
            f"({sequence.shape[-1]} != {spec.total_frames})"
        )
    window = sequence[..., spec.start_frame : spec.end_frame]
    if spec.padded_frames <= 0:
        return window.contiguous()
    pad = torch.full(
        (*sequence.shape[:-1], spec.padded_frames),
        float(pad_value),
        dtype=sequence.dtype,
        device=sequence.device,
    )
    return torch.cat([window, pad], dim=-1).contiguous()


def stitch_fixed_windows(
    windows: list[torch.Tensor],
    specs: list[FixedWindowSpec],
    *,
    overlap_frames: int,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Overlap-add fixed-size windows into a sequence with frames on the last axis."""
    if not windows:
        raise ValueError("windows cannot be empty")
    if len(windows) != len(specs):
        raise ValueError("windows and specs must have the same length")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")

    total_frames = specs[0].total_frames
    prefix_shape = windows[0].shape[:-1]
    dtype = windows[0].dtype
    device = windows[0].device
    assembled = torch.zeros((*prefix_shape, total_frames), dtype=dtype, device=device)
    weight_sum = torch.zeros((total_frames,), dtype=dtype, device=device)

    for idx, (window, spec) in enumerate(zip(windows, specs)):
        if spec.total_frames != total_frames:
            raise ValueError("all specs must share total_frames")
        if window.shape[:-1] != prefix_shape:
            raise ValueError("all windows must share non-frame dimensions")
        if window.shape[-1] != spec.window_frames:
            raise ValueError(
                "window last dimension must match spec.window_frames "
                f"({window.shape[-1]} != {spec.window_frames})"
            )
        if window.dtype != dtype or window.device != device:
            raise ValueError("all windows must share dtype and device")

        valid = window[..., : spec.valid_frames]
        weight = chunk_weight(
            chunk_length=spec.valid_frames,
            overlap_frames=overlap_frames,
            is_first=(idx == 0),
            is_last=(idx == len(specs) - 1),
            device=device,
            dtype=dtype,
        )
        assembled[..., spec.start_frame : spec.end_frame] += valid * weight
        weight_sum[spec.start_frame : spec.end_frame] += weight

    return assembled / weight_sum.clamp_min(float(eps))
