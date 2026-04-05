"""Epoch planning and latent-fps resolution helpers for dataset scheduling."""

from __future__ import annotations

import random
from statistics import median

from .dataset_types import ConditioningSource, EpochSegment, SongRecord


def _resolve_latent_fps(latent_fps: float | str, songs: list[SongRecord]) -> float:
    """Resolve configured latent FPS, supporting numeric values and ``auto``."""
    if isinstance(latent_fps, (int, float)):
        value = float(latent_fps)
        if value <= 0:
            raise ValueError("latent_fps must be > 0")
        return value

    if not isinstance(latent_fps, str) or latent_fps.lower() != "auto":
        raise ValueError("latent_fps must be a number or 'auto'")

    fps_values: list[float] = []
    for song in songs:
        if song.sample_rate is None or song.input_samples is None:
            continue
        if song.input_samples <= 0 or song.target_frames <= 0:
            continue
        fps_values.append(song.sample_rate * song.target_frames / song.input_samples)

    if fps_values:
        return float(median(fps_values))

    return 50.0


def _sample_condition_source(
    *,
    rng: random.Random,
    mono_probability: float,
    downmix_probability: float,
) -> ConditioningSource:
    """Sample stereo/mono/downmix conditioning source for one segment."""
    draw = rng.random()
    if draw < mono_probability:
        return ConditioningSource.MONO
    if draw < mono_probability + downmix_probability:
        return ConditioningSource.DOWNMIX
    return ConditioningSource.STEREO


def _segments_for_song(
    *,
    total_frames: int,
    window_frames: int,
    stride_frames: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Build deterministic strided segment ranges for one song."""
    if total_frames <= window_frames:
        return [(0, total_frames)]

    last_start = max(0, total_frames - window_frames)

    offset = 0
    if stride_frames > 1:
        offset = rng.randrange(stride_frames)

    starts = {0, last_start}
    cursor = offset
    while cursor <= last_start:
        starts.add(cursor)
        cursor += stride_frames

    out: list[tuple[int, int]] = []
    for start in sorted(starts):
        seg_len = min(window_frames, total_frames - start)
        if seg_len > 0:
            out.append((start, seg_len))
    return out


def _build_epoch_segments(
    *,
    epoch: int,
    songs: list[SongRecord],
    seed: int,
    shuffle_segments_within_epoch: bool,
    sequence_mode: str,
    sequence_frames: int,
    stride_frames: int,
    mono_probability: float,
    downmix_probability: float,
) -> list[EpochSegment]:
    """Build the full epoch segment schedule across all songs."""
    rng = random.Random(seed + int(epoch))
    song_indices = list(range(len(songs)))
    if shuffle_segments_within_epoch:
        rng.shuffle(song_indices)

    segments: list[EpochSegment] = []
    if sequence_mode == "full_song":
        for song_index in song_indices:
            song = songs[song_index]
            segments.append(
                EpochSegment(
                    song_index=song_index,
                    start_frame=0,
                    num_valid_frames=song.target_frames,
                    conditioning_source=_sample_condition_source(
                        rng=rng,
                        mono_probability=mono_probability,
                        downmix_probability=downmix_probability,
                    ),
                )
            )
        return segments

    for song_index in song_indices:
        song = songs[song_index]
        segment_ranges = _segments_for_song(
            total_frames=song.target_frames,
            window_frames=sequence_frames,
            stride_frames=stride_frames,
            rng=rng,
        )
        if shuffle_segments_within_epoch and len(segment_ranges) > 1:
            rng.shuffle(segment_ranges)
        for start, seg_len in segment_ranges:
            segments.append(
                EpochSegment(
                    song_index=song_index,
                    start_frame=start,
                    num_valid_frames=seg_len,
                    conditioning_source=_sample_condition_source(
                        rng=rng,
                        mono_probability=mono_probability,
                        downmix_probability=downmix_probability,
                    ),
                )
            )
    return segments
