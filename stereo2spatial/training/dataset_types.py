"""Typed records used by the signal-song dataset runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class ConditioningSource(IntEnum):
    """Enumerated conditioning signal source used for one training sample."""

    STEREO = 0
    MONO = 1
    DOWNMIX = 2


@dataclass(frozen=True)
class SongRecord:
    """Manifest-level metadata for one signal-song sample directory."""

    stream_hash: str
    sample_dir: Path
    target_frames: int
    target_channels: int
    sample_rate: int | None
    input_samples: int | None
    mix_style: tuple[float, ...] | None = None
    signal_rms: dict[str, float] | None = None


@dataclass(frozen=True)
class EpochSegment:
    """One scheduled segment slice selected for the current training epoch."""

    song_index: int
    start_frame: int
    num_valid_frames: int
    conditioning_source: ConditioningSource
