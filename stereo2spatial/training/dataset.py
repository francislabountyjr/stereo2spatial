"""Latent-song dataset with per-epoch segment scheduling."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .dataset_epoch import (
    _build_epoch_segments as _epoch_build_segments,
)
from .dataset_epoch import (
    _resolve_latent_fps as _epoch_resolve_latent_fps,
)
from .dataset_epoch import (
    _sample_condition_source as _epoch_sample_condition_source,
)
from .dataset_epoch import (
    _segments_for_song as _epoch_segments_for_song,
)
from .dataset_io import (
    METADATA_FILENAME,
    SAMPLE_BUNDLE_FILENAME,
    SOURCE_DOWNMIX_LATENT_FILENAME,
    SOURCE_MONO_LATENT_FILENAME,
    SOURCE_STEREO_LATENT_FILENAME,
    TARGET_LATENT_FILENAME,
    _slice_with_right_pad,
)
from .dataset_io import (
    _load_latents_from_sample as _io_load_latents_from_sample,
)
from .dataset_io import (
    _load_manifest_records as _io_load_manifest_records,
)
from .dataset_types import ConditioningSource, EpochSegment, SongRecord

__all__ = [
    "ConditioningSource",
    "LatentSongDataset",
    "EpochSegment",
    "SongRecord",
    "TARGET_LATENT_FILENAME",
    "SOURCE_STEREO_LATENT_FILENAME",
    "SOURCE_MONO_LATENT_FILENAME",
    "SOURCE_DOWNMIX_LATENT_FILENAME",
    "SAMPLE_BUNDLE_FILENAME",
    "METADATA_FILENAME",
]


class LatentSongDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Map-style dataset with per-epoch segment schedule.

    sequence_mode='strided_crops':
      - segment_seconds still exists (nominal window size), but returned samples use
        sequence_seconds.
      - sequence_seconds controls the fixed crop length emitted by __getitem__.
      - stride_seconds controls spacing of start_frame positions along the song.

    sequence_mode='full_song':
      - each epoch segment is one song (start_frame=0).
      - __getitem__ returns an unpadded variable-length sequence.
      - optional full_song_max_seconds can cap very long songs by taking a leading
        contiguous span from start_frame=0.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        sample_artifact_mode: str,
        segment_seconds: float,
        latent_fps: float | str,
        mono_probability: float,
        downmix_probability: float,
        cache_size: int,
        shuffle_segments_within_epoch: bool,
        seed: int,
        sequence_seconds: float | None = None,
        stride_seconds: float | None = None,
        sequence_mode: str = "strided_crops",
        full_song_max_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.sample_artifact_mode = sample_artifact_mode.strip().lower()
        self.segment_seconds = float(segment_seconds)
        self.latent_fps = latent_fps
        self.mono_probability = float(mono_probability)
        self.downmix_probability = float(downmix_probability)
        self.cache_size = int(cache_size)
        self.shuffle_segments_within_epoch = bool(shuffle_segments_within_epoch)
        self.seed = int(seed)
        self.sequence_mode = str(sequence_mode).strip().lower()
        self.full_song_max_seconds = (
            float(full_song_max_seconds)
            if full_song_max_seconds is not None
            else None
        )

        self.sequence_seconds = (
            float(sequence_seconds)
            if sequence_seconds is not None
            else self.segment_seconds
        )
        self.stride_seconds = (
            float(stride_seconds)
            if stride_seconds is not None
            else self.sequence_seconds
        )

        if self.sample_artifact_mode not in {"bundle", "split"}:
            raise ValueError(
                "sample_artifact_mode must be one of: bundle, split "
                f"(got {self.sample_artifact_mode!r})"
            )
        if self.sequence_mode not in {"strided_crops", "full_song"}:
            raise ValueError(
                "sequence_mode must be one of: strided_crops, full_song "
                f"(got {self.sequence_mode!r})"
            )
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be > 0")
        if self.sequence_seconds <= 0:
            raise ValueError("sequence_seconds must be > 0")
        if self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be > 0")
        if self.full_song_max_seconds is not None and self.full_song_max_seconds <= 0:
            raise ValueError("full_song_max_seconds must be > 0")
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        if self.mono_probability < 0 or self.mono_probability > 1:
            raise ValueError("mono_probability must be in [0, 1]")
        if self.downmix_probability < 0 or self.downmix_probability > 1:
            raise ValueError("downmix_probability must be in [0, 1]")
        if self.mono_probability + self.downmix_probability > 1:
            raise ValueError("mono_probability + downmix_probability must be <= 1")

        self._songs = self._load_manifest_records()
        if not self._songs:
            raise RuntimeError(f"No samples found from manifest: {self.manifest_path}")

        self.resolved_latent_fps = self._resolve_latent_fps()
        self.segment_frames = max(
            1, int(round(self.segment_seconds * self.resolved_latent_fps))
        )
        self.sequence_frames = max(
            1, int(round(self.sequence_seconds * self.resolved_latent_fps))
        )
        self.stride_frames = max(
            1, int(round(self.stride_seconds * self.resolved_latent_fps))
        )
        self.full_song_max_frames = (
            max(1, int(round(self.full_song_max_seconds * self.resolved_latent_fps)))
            if self.full_song_max_seconds is not None
            else None
        )

        self._segments: list[EpochSegment] = []
        self._bundle_cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()
        self._epoch = 0
        self.set_epoch(0)

    def _load_manifest_records(self) -> list[SongRecord]:
        """Load and validate manifest entries under ``dataset_root``."""
        return _io_load_manifest_records(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
        )

    def _resolve_latent_fps(self) -> float:
        """Resolve latent FPS from config value (numeric or ``auto``)."""
        return _epoch_resolve_latent_fps(latent_fps=self.latent_fps, songs=self._songs)

    def _sample_condition_source(self, rng: random.Random) -> ConditioningSource:
        """Sample conditioning source according to configured probabilities."""
        return _epoch_sample_condition_source(
            rng=rng,
            mono_probability=self.mono_probability,
            downmix_probability=self.downmix_probability,
        )

    def _segments_for_song(
        self,
        total_frames: int,
        window_frames: int,
        stride_frames: int,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        """Generate per-song segment ranges for one epoch schedule."""
        return _epoch_segments_for_song(
            total_frames=total_frames,
            window_frames=window_frames,
            stride_frames=stride_frames,
            rng=rng,
        )

    def _build_epoch_segments(self, epoch: int) -> list[EpochSegment]:
        """Build deterministic epoch segment metadata for all songs."""
        return _epoch_build_segments(
            epoch=epoch,
            songs=self._songs,
            seed=self.seed,
            shuffle_segments_within_epoch=self.shuffle_segments_within_epoch,
            sequence_mode=self.sequence_mode,
            sequence_frames=self.sequence_frames,
            stride_frames=self.stride_frames,
            mono_probability=self.mono_probability,
            downmix_probability=self.downmix_probability,
        )

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the epoch segment schedule using deterministic epoch seeding."""
        self._epoch = int(epoch)
        self._segments = self._build_epoch_segments(self._epoch)

    def _load_latents_from_sample(self, sample_dir: Path) -> dict[str, torch.Tensor]:
        """Load one sample's latent tensors from bundle or split artifacts."""
        return _io_load_latents_from_sample(sample_dir)

    def _get_song_latents(self, song: SongRecord) -> dict[str, torch.Tensor]:
        """Return song latent tensors, using an LRU cache when enabled."""
        if song.sample_dir in self._bundle_cache:
            value = self._bundle_cache.pop(song.sample_dir)
            self._bundle_cache[song.sample_dir] = value
            return value

        latents = self._load_latents_from_sample(song.sample_dir)
        if self.cache_size > 0:
            self._bundle_cache[song.sample_dir] = latents
            while len(self._bundle_cache) > self.cache_size:
                self._bundle_cache.popitem(last=False)
        return latents

    def _conditioning_key(self, source: ConditioningSource) -> str:
        """Map conditioning source enum to the matching latent dictionary key."""
        if source == ConditioningSource.MONO:
            return "source_mono_latent"
        if source == ConditioningSource.DOWNMIX:
            return "source_downmix_latent"
        return "source_stereo_latent"

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        segment = self._segments[index]
        song = self._songs[segment.song_index]
        latents = self._get_song_latents(song)

        target_latent = latents["target_latent"]
        cond_key = self._conditioning_key(segment.conditioning_source)
        cond_latent = latents[cond_key]

        if self.sequence_mode == "full_song":
            max_total_frames = min(target_latent.shape[-1], cond_latent.shape[-1])
            total_frames = min(int(segment.num_valid_frames), int(max_total_frames))
            if total_frames <= 0:
                raise RuntimeError(
                    f"Invalid full-song slice length: total_frames={total_frames}"
                )

            start_frame = 0
            num_valid_frames = total_frames
            if (
                self.full_song_max_frames is not None
                and num_valid_frames > self.full_song_max_frames
            ):
                num_valid_frames = self.full_song_max_frames

            end_frame = start_frame + num_valid_frames
            target_chunk = target_latent[..., start_frame:end_frame].contiguous()
            cond_chunk = cond_latent[..., start_frame:end_frame].contiguous()
            valid_frames = min(target_chunk.shape[-1], cond_chunk.shape[-1])
            if valid_frames <= 0:
                raise RuntimeError(
                    "Invalid full-song chunk: "
                    f"target={tuple(target_chunk.shape)} cond={tuple(cond_chunk.shape)}"
                )
            target_chunk = target_chunk[..., :valid_frames]
            cond_chunk = cond_chunk[..., :valid_frames]
            valid_mask = torch.ones(
                valid_frames, dtype=torch.bool, device=target_chunk.device
            )

            return {
                "target_latent": target_chunk,
                "cond_latent": cond_chunk,
                "valid_mask": valid_mask,
                "song_index": torch.tensor(segment.song_index, dtype=torch.long),
                "start_frame": torch.tensor(start_frame, dtype=torch.long),
                "conditioning_source": torch.tensor(
                    int(segment.conditioning_source), dtype=torch.long
                ),
            }

        target_chunk, target_mask = _slice_with_right_pad(
            target_latent,
            segment.start_frame,
            segment.num_valid_frames,
            self.sequence_frames,
        )
        cond_chunk, cond_mask = _slice_with_right_pad(
            cond_latent,
            segment.start_frame,
            segment.num_valid_frames,
            self.sequence_frames,
        )
        valid_mask = target_mask & cond_mask

        return {
            "target_latent": target_chunk,
            "cond_latent": cond_chunk,
            "valid_mask": valid_mask,
            "song_index": torch.tensor(segment.song_index, dtype=torch.long),
            "start_frame": torch.tensor(segment.start_frame, dtype=torch.long),
            "conditioning_source": torch.tensor(
                int(segment.conditioning_source), dtype=torch.long
            ),
        }

    def describe(self) -> dict[str, Any]:
        """Return a compact runtime summary of resolved dataset settings."""
        return {
            "num_songs": len(self._songs),
            "epoch": self._epoch,
            "epoch_num_segments": len(self._segments),
            "sequence_mode": self.sequence_mode,
            "segment_seconds": self.segment_seconds,
            "sequence_seconds": self.sequence_seconds,
            "stride_seconds": self.stride_seconds,
            "full_song_max_seconds": self.full_song_max_seconds,
            "resolved_latent_fps": self.resolved_latent_fps,
            "segment_frames": self.segment_frames,
            "sequence_frames": self.sequence_frames,
            "stride_frames": self.stride_frames,
            "full_song_max_frames": self.full_song_max_frames,
            "mono_probability": self.mono_probability,
            "downmix_probability": self.downmix_probability,
        }
