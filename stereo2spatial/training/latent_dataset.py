"""Legacy precomputed-latent dataset with current trainer-facing batch keys."""

from __future__ import annotations

from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .dataset_epoch import (
    _build_epoch_segments as _epoch_build_segments,
)
from .dataset_epoch import _resolve_patch_fps
from .dataset_types import ConditioningSource, EpochSegment, SongRecord
from .latent_dataset_io import (
    METADATA_FILENAME,
    SAMPLE_BUNDLE_FILENAME,
    SOURCE_DOWNMIX_LATENT_FILENAME,
    SOURCE_MONO_LATENT_FILENAME,
    SOURCE_STEREO_LATENT_FILENAME,
    TARGET_LATENT_FILENAME,
    _filter_songs_by_sample_exclusion,
    _load_latents_from_sample,
    _load_manifest_records,
    _slice_with_right_pad,
)


class LatentSongDataset(Dataset[dict[str, torch.Tensor]]):
    """Read old EAR-VAE latent artifacts while exposing the current batch API."""

    def __init__(
        self,
        dataset_root: str | PathLike[str] | list[str | PathLike[str]],
        manifest_path: str | PathLike[str] | list[str | PathLike[str]],
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
        shuffle_segments_within_song: bool = True,
        sample_exclusion_path: str | Path | list[str | Path] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_roots, self.manifest_paths = self._coerce_dataset_paths(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
        )
        # Singular aliases preserve the shape expected by older callers.
        self.dataset_root = self.dataset_roots[0]
        self.manifest_path = self.manifest_paths[0]
        self.sample_artifact_mode = str(sample_artifact_mode).strip().lower()
        self.segment_seconds = float(segment_seconds)
        self.latent_fps = latent_fps
        self.mono_probability = float(mono_probability)
        self.downmix_probability = float(downmix_probability)
        self.cache_size = int(cache_size)
        self.shuffle_segments_within_epoch = bool(shuffle_segments_within_epoch)
        self.shuffle_segments_within_song = bool(shuffle_segments_within_song)
        self.sample_exclusion_path = sample_exclusion_path
        self.seed = int(seed)
        self.sequence_mode = str(sequence_mode).strip().lower()
        self.full_song_max_seconds = (
            None if full_song_max_seconds is None else float(full_song_max_seconds)
        )
        self.sequence_seconds = (
            self.segment_seconds
            if sequence_seconds is None
            else float(sequence_seconds)
        )
        self.stride_seconds = (
            self.sequence_seconds if stride_seconds is None else float(stride_seconds)
        )

        self._validate_settings()
        songs: list[SongRecord] = []
        for root, manifest in zip(self.dataset_roots, self.manifest_paths):
            songs.extend(
                _load_manifest_records(dataset_root=root, manifest_path=manifest)
            )
        self._songs, self._excluded_song_count = _filter_songs_by_sample_exclusion(
            songs,
            sample_exclusion_path=self.sample_exclusion_path,
        )
        if not self._songs:
            manifests = ", ".join(str(path) for path in self.manifest_paths)
            raise RuntimeError(f"No samples found from latent manifest(s): {manifests}")

        self.resolved_latent_fps = _resolve_patch_fps(
            patch_fps=self.latent_fps,
            songs=self._songs,
        )
        # Current sequence planning consumes this architecture-neutral alias.
        self.resolved_patch_fps = self.resolved_latent_fps
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

    @staticmethod
    def _path_list(
        value: str | PathLike[str] | list[str | PathLike[str]],
        name: str,
    ) -> list[Path]:
        if isinstance(value, (str, PathLike)):
            return [Path(value)]
        if not isinstance(value, list) or not value:
            raise TypeError(f"{name} must be a path or non-empty path list.")
        return [Path(item) for item in value]

    @classmethod
    def _coerce_dataset_paths(
        cls,
        *,
        dataset_root: str | PathLike[str] | list[str | PathLike[str]],
        manifest_path: str | PathLike[str] | list[str | PathLike[str]],
    ) -> tuple[list[Path], list[Path]]:
        roots = cls._path_list(dataset_root, "dataset_root")
        manifests = cls._path_list(manifest_path, "manifest_path")
        if len(roots) != len(manifests):
            raise ValueError(
                "dataset_root and manifest_path must contain the same number of paths."
            )
        resolved_manifests = [
            manifest if manifest.is_absolute() or manifest.exists() else root / manifest
            for root, manifest in zip(roots, manifests)
        ]
        return roots, resolved_manifests

    def _validate_settings(self) -> None:
        if self.sample_artifact_mode not in {"bundle", "split"}:
            raise ValueError("sample_artifact_mode must be one of: bundle, split")
        if self.sequence_mode not in {"strided_crops", "full_song"}:
            raise ValueError("sequence_mode must be one of: strided_crops, full_song")
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
        if not 0.0 <= self.mono_probability <= 1.0:
            raise ValueError("mono_probability must be in [0, 1]")
        if not 0.0 <= self.downmix_probability <= 1.0:
            raise ValueError("downmix_probability must be in [0, 1]")
        if self.mono_probability + self.downmix_probability > 1.0:
            raise ValueError("mono_probability + downmix_probability must be <= 1")

    def _build_epoch_segments(self, epoch: int) -> list[EpochSegment]:
        segments: list[EpochSegment] = _epoch_build_segments(
            epoch=int(epoch),
            songs=self._songs,
            seed=self.seed,
            shuffle_segments_within_epoch=self.shuffle_segments_within_epoch,
            shuffle_segments_within_song=self.shuffle_segments_within_song,
            sequence_mode=self.sequence_mode,
            sequence_frames=self.sequence_frames,
            stride_frames=self.stride_frames,
            mono_probability=self.mono_probability,
            downmix_probability=self.downmix_probability,
        )
        return segments

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the deterministic segment schedule for one epoch."""
        self._epoch = int(epoch)
        self._segments = self._build_epoch_segments(self._epoch)

    def set_global_step(self, step: int) -> None:
        """Accept the current trainer hook; latent loading has no step ramp."""
        del step

    def _get_song_latents(self, song: SongRecord) -> dict[str, torch.Tensor]:
        if song.sample_dir in self._bundle_cache:
            value = self._bundle_cache.pop(song.sample_dir)
            self._bundle_cache[song.sample_dir] = value
            return value

        latents = _load_latents_from_sample(
            song.sample_dir,
            sample_artifact_mode=self.sample_artifact_mode,
        )
        if self.cache_size > 0:
            self._bundle_cache[song.sample_dir] = latents
            while len(self._bundle_cache) > self.cache_size:
                self._bundle_cache.popitem(last=False)
        return latents

    @staticmethod
    def _conditioning_key(source: ConditioningSource) -> str:
        if source == ConditioningSource.MONO:
            return "source_mono_latent"
        if source == ConditioningSource.DOWNMIX:
            return "source_downmix_latent"
        return "source_stereo_latent"

    @staticmethod
    def _metadata(
        segment: EpochSegment,
        *,
        start_frame: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "song_index": torch.tensor(segment.song_index, dtype=torch.long),
            "start_frame": torch.tensor(start_frame, dtype=torch.long),
            "conditioning_source": torch.tensor(
                int(segment.conditioning_source), dtype=torch.long
            ),
        }

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        segment = self._segments[index]
        song = self._songs[segment.song_index]
        latents = self._get_song_latents(song)
        target = latents["target_latent"]
        conditioning = latents[self._conditioning_key(segment.conditioning_source)]

        if target.shape[1] != conditioning.shape[1]:
            raise ValueError(
                "Target and conditioning latent feature dimensions differ: "
                f"target={tuple(target.shape)} cond={tuple(conditioning.shape)}"
            )

        if self.sequence_mode == "full_song":
            valid_frames = min(
                int(segment.num_valid_frames),
                int(target.shape[-1]),
                int(conditioning.shape[-1]),
            )
            if self.full_song_max_frames is not None:
                valid_frames = min(valid_frames, self.full_song_max_frames)
            if valid_frames <= 0:
                raise RuntimeError(
                    f"Invalid full-song latent slice for {song.sample_dir}"
                )
            target_chunk = target[..., :valid_frames].contiguous()
            cond_chunk = conditioning[..., :valid_frames].contiguous()
            sample = {
                "target_signal": target_chunk,
                "cond_signal": cond_chunk,
                "valid_mask": torch.ones(valid_frames, dtype=torch.bool),
            }
            sample.update(self._metadata(segment, start_frame=0))
            return sample

        target_chunk, target_mask = _slice_with_right_pad(
            target,
            segment.start_frame,
            segment.num_valid_frames,
            self.sequence_frames,
        )
        cond_chunk, cond_mask = _slice_with_right_pad(
            conditioning,
            segment.start_frame,
            segment.num_valid_frames,
            self.sequence_frames,
        )
        sample = {
            "target_signal": target_chunk,
            "cond_signal": cond_chunk,
            "valid_mask": target_mask & cond_mask,
        }
        sample.update(self._metadata(segment, start_frame=segment.start_frame))
        return sample

    def describe(self) -> dict[str, Any]:
        """Return architecture-neutral and legacy-specific runtime metadata."""
        return {
            "sample_domain": "vae_latent",
            "num_songs": len(self._songs),
            "num_datasets": len(self.dataset_roots),
            "dataset_roots": [str(path) for path in self.dataset_roots],
            "manifest_paths": [str(path) for path in self.manifest_paths],
            "sample_artifact_mode": self.sample_artifact_mode,
            "sample_exclusion_path": self.sample_exclusion_path,
            "excluded_song_count": self._excluded_song_count,
            "epoch": self._epoch,
            "epoch_num_segments": len(self._segments),
            "sequence_mode": self.sequence_mode,
            "segment_seconds": self.segment_seconds,
            "sequence_seconds": self.sequence_seconds,
            "stride_seconds": self.stride_seconds,
            "full_song_max_seconds": self.full_song_max_seconds,
            "shuffle_segments_within_epoch": self.shuffle_segments_within_epoch,
            "shuffle_segments_within_song": self.shuffle_segments_within_song,
            "resolved_latent_fps": self.resolved_latent_fps,
            "resolved_patch_fps": self.resolved_patch_fps,
            "segment_frames": self.segment_frames,
            "sequence_frames": self.sequence_frames,
            "stride_frames": self.stride_frames,
            "full_song_max_frames": self.full_song_max_frames,
            "mono_probability": self.mono_probability,
            "downmix_probability": self.downmix_probability,
        }


__all__ = [
    "ConditioningSource",
    "EpochSegment",
    "LatentSongDataset",
    "METADATA_FILENAME",
    "SAMPLE_BUNDLE_FILENAME",
    "SOURCE_DOWNMIX_LATENT_FILENAME",
    "SOURCE_MONO_LATENT_FILENAME",
    "SOURCE_STEREO_LATENT_FILENAME",
    "SongRecord",
    "TARGET_LATENT_FILENAME",
]
