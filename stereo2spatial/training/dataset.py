"""Signal-song dataset with per-epoch segment scheduling."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import replace
from multiprocessing import Value
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset

from stereo2spatial.common.amplitude_lift import (
    amplitude_lift_log_gain,
    apply_amplitude_lift,
    resolve_amplitude_lift_gain,
    wavflow_source_transform,
    wavflow_target_transform,
)
from stereo2spatial.common.codec_augmentation import codec_roundtrip
from stereo2spatial.common.mix_style import DEFAULT_MIX_STYLE_VECTOR

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional dependency path
    torchaudio = None

from .dataset_epoch import (
    _build_epoch_segments as _epoch_build_segments,
)
from .dataset_epoch import (
    _resolve_patch_fps as _epoch_resolve_patch_fps,
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
    SOURCE_DOWNMIX_SIGNAL_FILENAME,
    SOURCE_MONO_SIGNAL_FILENAME,
    SOURCE_STEREO_SIGNAL_FILENAME,
    TARGET_SIGNAL_FILENAME,
    _filter_songs_by_min_source_rms,
    _filter_songs_by_sample_exclusion,
    _patch_audio,
    _slice_with_right_pad,
)
from .dataset_io import (
    _load_manifest_records as _io_load_manifest_records,
)
from .dataset_io import (
    _load_raw_signals_from_sample as _io_load_raw_signals_from_sample,
)
from .dataset_io import (
    _load_signals_from_sample as _io_load_signals_from_sample,
)
from .dataset_types import ConditioningSource, EpochSegment, SongRecord

__all__ = [
    "ConditioningSource",
    "WaveformSongDataset",
    "EpochSegment",
    "SongRecord",
    "TARGET_SIGNAL_FILENAME",
    "SOURCE_STEREO_SIGNAL_FILENAME",
    "SOURCE_MONO_SIGNAL_FILENAME",
    "SOURCE_DOWNMIX_SIGNAL_FILENAME",
    "SAMPLE_BUNDLE_FILENAME",
    "METADATA_FILENAME",
]


class WaveformSongDataset(Dataset[dict[str, torch.Tensor]]):
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
        dataset_root: str | PathLike[str] | Sequence[str | PathLike[str]],
        manifest_path: str | PathLike[str] | Sequence[str | PathLike[str]],
        sample_artifact_mode: str,
        segment_seconds: float,
        patch_fps: float | str,
        patch_size: int,
        mono_probability: float,
        downmix_probability: float,
        cache_size: int,
        shuffle_segments_within_epoch: bool,
        seed: int,
        sample_exclusion_path: str
        | PathLike[str]
        | Sequence[str | PathLike[str]]
        | None = None,
        shuffle_segments_within_song: bool = True,
        materialize_cached_signals: bool = False,
        sequence_seconds: float | None = None,
        stride_seconds: float | None = None,
        sample_rate: int = 48_000,
        training_sample_rate: int | None = None,
        sequence_mode: str = "strided_crops",
        full_song_max_seconds: float | None = None,
        amplitude_lift_enabled: bool = False,
        amplitude_lift_mode: str = "rms",
        amplitude_lift_reference: str = "source",
        amplitude_lift_target_rms: float = 0.33,
        amplitude_lift_scale: float = 3.0,
        amplitude_lift_clip_value: float | None = 4.0,
        amplitude_lift_gain_power: float = 1.0,
        amplitude_lift_gain_min_value: float | None = None,
        amplitude_lift_waveform_clamp: bool = True,
        amplitude_lift_peak_limit: float = 1.0,
        amplitude_lift_peak_rescale_min_rms: float = 0.3,
        amplitude_lift_eps: float = 1.0e-8,
        min_source_rms: float | None = None,
        source_resample_aug_enabled: bool = False,
        source_resample_aug_probability: float = 0.0,
        source_resample_aug_rates: list[int] | None = None,
        source_resample_aug_weights: list[float] | None = None,
        source_resample_aug_sample_rate: int = 48_000,
        source_codec_aug_enabled: bool = False,
        source_codec_aug_probability: float = 0.0,
        source_codec_aug_start_step: int = 0,
        source_codec_aug_full_strength_step: int = 0,
        source_codec_aug_backend: str = "auto",
        source_codec_aug_ffmpeg_path: str = "ffmpeg",
        source_codec_aug_codecs: list[str] | None = None,
        source_codec_aug_codec_weights: list[float] | None = None,
        source_codec_aug_bitrates: dict[str, list[int]] | None = None,
        source_codec_aug_max_chunk_seconds: float | None = 12.0,
        source_codec_aug_align_max_lag: int = 8192,
        source_codec_aug_timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__()
        self.dataset_roots, self.manifest_paths = self._coerce_dataset_paths(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
        )
        self.dataset_root = self.dataset_roots[0]
        self.manifest_path = self.manifest_paths[0]
        self.sample_artifact_mode = sample_artifact_mode.strip().lower()
        self.segment_seconds = float(segment_seconds)
        self.patch_fps = patch_fps
        self.patch_size = int(patch_size)
        self.mono_probability = float(mono_probability)
        self.downmix_probability = float(downmix_probability)
        self.cache_size = int(cache_size)
        self.shuffle_segments_within_epoch = bool(shuffle_segments_within_epoch)
        self.shuffle_segments_within_song = bool(shuffle_segments_within_song)
        self.materialize_cached_signals = bool(materialize_cached_signals)
        self.sample_exclusion_path = sample_exclusion_path
        self.seed = int(seed)
        self.sequence_mode = str(sequence_mode).strip().lower()
        self.sample_rate = int(sample_rate)
        self.training_sample_rate = (
            self.sample_rate
            if training_sample_rate is None
            else int(training_sample_rate)
        )
        self.full_song_max_seconds = (
            float(full_song_max_seconds) if full_song_max_seconds is not None else None
        )
        self.amplitude_lift_enabled = bool(amplitude_lift_enabled)
        self.amplitude_lift_mode = str(amplitude_lift_mode).strip().lower()
        self.amplitude_lift_reference = str(amplitude_lift_reference).strip().lower()
        self.amplitude_lift_target_rms = float(amplitude_lift_target_rms)
        self.amplitude_lift_scale = float(amplitude_lift_scale)
        self.amplitude_lift_clip_value = (
            None
            if amplitude_lift_clip_value is None
            else float(amplitude_lift_clip_value)
        )
        self.amplitude_lift_gain_power = float(amplitude_lift_gain_power)
        self.amplitude_lift_gain_min_value = (
            None
            if amplitude_lift_gain_min_value is None
            else float(amplitude_lift_gain_min_value)
        )
        self.amplitude_lift_waveform_clamp = bool(amplitude_lift_waveform_clamp)
        self.amplitude_lift_peak_limit = float(amplitude_lift_peak_limit)
        self.amplitude_lift_peak_rescale_min_rms = float(
            amplitude_lift_peak_rescale_min_rms
        )
        self.amplitude_lift_eps = float(amplitude_lift_eps)
        self.min_source_rms = None if min_source_rms is None else float(min_source_rms)
        self.source_resample_aug_enabled = bool(source_resample_aug_enabled)
        self.source_resample_aug_probability = float(source_resample_aug_probability)
        self.source_resample_aug_rates = (
            [int(rate) for rate in source_resample_aug_rates]
            if source_resample_aug_rates is not None
            else [44_100]
        )
        self.source_resample_aug_weights = (
            [float(weight) for weight in source_resample_aug_weights]
            if source_resample_aug_weights is not None
            else None
        )
        self.source_resample_aug_sample_rate = int(source_resample_aug_sample_rate)
        self.source_codec_aug_enabled = bool(source_codec_aug_enabled)
        self.source_codec_aug_probability = float(source_codec_aug_probability)
        self.source_codec_aug_start_step = int(source_codec_aug_start_step)
        self.source_codec_aug_full_strength_step = int(
            source_codec_aug_full_strength_step
        )
        self.source_codec_aug_backend = str(source_codec_aug_backend).strip().lower()
        self.source_codec_aug_ffmpeg_path = str(source_codec_aug_ffmpeg_path)
        self.source_codec_aug_codecs = (
            [str(codec).strip().lower() for codec in source_codec_aug_codecs]
            if source_codec_aug_codecs is not None
            else ["mp3", "aac", "opus"]
        )
        self.source_codec_aug_codec_weights = (
            [float(weight) for weight in source_codec_aug_codec_weights]
            if source_codec_aug_codec_weights is not None
            else [0.4, 0.35, 0.25]
        )
        default_bitrates = {
            "mp3": [128, 160, 192, 256, 320],
            "aac": [128, 160, 192, 256],
            "opus": [96, 128, 160, 192],
        }
        self.source_codec_aug_bitrates = (
            {
                str(codec).strip().lower(): [int(rate) for rate in rates]
                for codec, rates in source_codec_aug_bitrates.items()
            }
            if source_codec_aug_bitrates is not None
            else default_bitrates
        )
        self.source_codec_aug_max_chunk_seconds = (
            float(source_codec_aug_max_chunk_seconds)
            if source_codec_aug_max_chunk_seconds is not None
            else None
        )
        self.source_codec_aug_align_max_lag = int(source_codec_aug_align_max_lag)
        self.source_codec_aug_timeout_seconds = float(source_codec_aug_timeout_seconds)
        self._global_step_value = Value("q", 0, lock=False)

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

        if self.sample_artifact_mode not in {"bundle", "split", "flac"}:
            raise ValueError(
                "sample_artifact_mode must be one of: bundle, split, flac "
                f"(got {self.sample_artifact_mode!r})"
            )
        if self.sequence_mode not in {"strided_crops", "full_song"}:
            raise ValueError(
                "sequence_mode must be one of: strided_crops, full_song "
                f"(got {self.sequence_mode!r})"
            )
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be > 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.sequence_seconds <= 0:
            raise ValueError("sequence_seconds must be > 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.training_sample_rate <= 0:
            raise ValueError("training_sample_rate must be > 0")
        if self.training_sample_rate > self.sample_rate:
            raise ValueError("training_sample_rate must be <= sample_rate")
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
        if self.amplitude_lift_reference not in {"source", "target"}:
            raise ValueError("amplitude_lift_reference must be one of: source, target")
        if self.amplitude_lift_mode not in {"rms", "scale", "wavflow"}:
            raise ValueError("amplitude_lift_mode must be one of: rms, scale, wavflow")
        if self.amplitude_lift_target_rms <= 0:
            raise ValueError("amplitude_lift_target_rms must be > 0")
        if self.amplitude_lift_scale <= 0:
            raise ValueError("amplitude_lift_scale must be > 0")
        if (
            self.amplitude_lift_clip_value is not None
            and self.amplitude_lift_clip_value <= 0
        ):
            raise ValueError("amplitude_lift_clip_value must be > 0 when set")
        if self.amplitude_lift_eps <= 0:
            raise ValueError("amplitude_lift_eps must be > 0")
        if self.amplitude_lift_peak_limit <= 0:
            raise ValueError("amplitude_lift_peak_limit must be > 0")
        if self.amplitude_lift_peak_rescale_min_rms <= 0:
            raise ValueError("amplitude_lift_peak_rescale_min_rms must be > 0")
        if self.min_source_rms is not None and not 0 < self.min_source_rms < 1:
            raise ValueError("min_source_rms must be in (0, 1) when set")
        if not 0 <= self.source_resample_aug_probability <= 1:
            raise ValueError("source_resample_aug_probability must be in [0, 1]")
        if self.source_resample_aug_sample_rate <= 0:
            raise ValueError("source_resample_aug_sample_rate must be > 0")
        if self.source_resample_aug_enabled:
            if not self.source_resample_aug_rates:
                raise ValueError(
                    "source_resample_aug_rates must be non-empty when enabled"
                )
            for rate in self.source_resample_aug_rates:
                if rate <= 0:
                    raise ValueError("source_resample_aug_rates must all be > 0")
            if self.source_resample_aug_weights is not None:
                if len(self.source_resample_aug_weights) != len(
                    self.source_resample_aug_rates
                ):
                    raise ValueError(
                        "source_resample_aug_weights length must match rates"
                    )
                if any(weight < 0 for weight in self.source_resample_aug_weights):
                    raise ValueError("source_resample_aug_weights must be >= 0")
                if sum(self.source_resample_aug_weights) <= 0:
                    raise ValueError("source_resample_aug_weights must sum to > 0")
        if not 0 <= self.source_codec_aug_probability <= 1:
            raise ValueError("source_codec_aug_probability must be in [0, 1]")
        if self.source_codec_aug_start_step < 0:
            raise ValueError("source_codec_aug_start_step must be >= 0")
        if self.source_codec_aug_full_strength_step < 0:
            raise ValueError("source_codec_aug_full_strength_step must be >= 0")
        if self.source_codec_aug_backend not in {"auto", "torchaudio", "ffmpeg"}:
            raise ValueError(
                "source_codec_aug_backend must be one of: auto, torchaudio, ffmpeg"
            )
        if self.source_codec_aug_max_chunk_seconds is not None:
            if self.source_codec_aug_max_chunk_seconds <= 0:
                raise ValueError("source_codec_aug_max_chunk_seconds must be > 0")
        if self.source_codec_aug_align_max_lag <= 0:
            raise ValueError("source_codec_aug_align_max_lag must be > 0")
        if self.source_codec_aug_timeout_seconds <= 0:
            raise ValueError("source_codec_aug_timeout_seconds must be > 0")
        if self.source_codec_aug_enabled:
            if not self.source_codec_aug_codecs:
                raise ValueError("source_codec_aug_codecs must be non-empty")
            if len(self.source_codec_aug_codec_weights) != len(
                self.source_codec_aug_codecs
            ):
                raise ValueError(
                    "source_codec_aug_codec_weights length must match codecs"
                )
            if sum(self.source_codec_aug_codec_weights) <= 0:
                raise ValueError("source_codec_aug_codec_weights must sum to > 0")
            for codec in self.source_codec_aug_codecs:
                if codec not in {"mp3", "aac", "opus"}:
                    raise ValueError("source_codec_aug_codecs must be mp3/aac/opus")
                bitrates = self.source_codec_aug_bitrates.get(codec)
                if not bitrates:
                    raise ValueError("source_codec_aug_bitrates must be non-empty")
                if any(int(rate) <= 0 for rate in bitrates):
                    raise ValueError("source_codec_aug_bitrates must all be > 0")

        self._songs = self._load_manifest_records()
        if not self._songs:
            manifests = ", ".join(str(path) for path in self.manifest_paths)
            detail = (
                f" (after min_source_rms={self.min_source_rms} filtering)"
                if self.min_source_rms is not None
                else ""
            )
            raise RuntimeError(
                f"No samples found from manifest(s){detail}: {manifests}"
            )

        self.resolved_patch_fps = self._resolve_patch_fps()
        self.segment_frames = max(
            1, int(round(self.segment_seconds * self.resolved_patch_fps))
        )
        self.sequence_frames = max(
            1, int(round(self.sequence_seconds * self.resolved_patch_fps))
        )
        self.stride_frames = max(
            1, int(round(self.stride_seconds * self.resolved_patch_fps))
        )
        self.full_song_max_frames = (
            max(1, int(round(self.full_song_max_seconds * self.resolved_patch_fps)))
            if self.full_song_max_seconds is not None
            else None
        )

        self._segments: list[EpochSegment] = []
        self._bundle_cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()
        self._epoch = 0
        self.set_epoch(0)

    @staticmethod
    def _path_list(
        value: str | PathLike[str] | Sequence[str | PathLike[str]],
        name: str,
    ) -> list[Path]:
        """Normalize a path or non-empty path list."""
        if isinstance(value, (str, PathLike)):
            return [Path(value)]
        if not isinstance(value, Sequence) or not value:
            raise TypeError(f"{name} must be a path or non-empty path list.")
        return [Path(item) for item in value]

    @classmethod
    def _coerce_dataset_paths(
        cls,
        *,
        dataset_root: str | PathLike[str] | Sequence[str | PathLike[str]],
        manifest_path: str | PathLike[str] | Sequence[str | PathLike[str]],
    ) -> tuple[list[Path], list[Path]]:
        """Normalize dataset root/manifest inputs and keep them paired."""
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

    def _mix_style_tensor(self, song: SongRecord) -> torch.Tensor:
        """Return this song's normalized mix-style vector."""
        values = (
            song.mix_style if song.mix_style is not None else DEFAULT_MIX_STYLE_VECTOR
        )
        return torch.tensor(values, dtype=torch.float32)

    def _load_manifest_records(self) -> list[SongRecord]:
        """Load and validate manifest entries under all configured dataset roots."""
        songs: list[SongRecord] = []
        for dataset_root, manifest_path in zip(self.dataset_roots, self.manifest_paths):
            songs.extend(
                _io_load_manifest_records(
                    dataset_root=dataset_root,
                    manifest_path=manifest_path,
                    patch_size=self.patch_size,
                )
            )
        songs = [self._song_for_training_sample_rate(song) for song in songs]
        songs, excluded = _filter_songs_by_sample_exclusion(
            songs,
            sample_exclusion_path=self.sample_exclusion_path,
        )
        if self.sample_exclusion_path is not None:
            print(
                "[sample_exclusion] "
                f"files={self.sample_exclusion_path} kept={len(songs)} dropped={excluded}"
            )
        songs, dropped, missing = _filter_songs_by_min_source_rms(
            songs,
            min_source_rms=self.min_source_rms,
        )
        if self.min_source_rms is not None:
            message = (
                f"[min_source_rms] threshold={self.min_source_rms} "
                f"kept={len(songs)} dropped={dropped}"
            )
            if missing > 0:
                message += f" kept_without_rms_metadata={missing}"
            print(message)
        return songs

    def _song_for_training_sample_rate(self, song: SongRecord) -> SongRecord:
        """Return song metadata with frame counts scaled to training sample rate."""
        source_rate = int(song.sample_rate or self.sample_rate)
        if self.training_sample_rate == source_rate:
            return song
        ratio = float(self.training_sample_rate) / float(source_rate)
        if song.input_samples is not None and song.input_samples > 0:
            resampled_samples = max(1, int(round(int(song.input_samples) * ratio)))
            target_frames = max(1, math.ceil(resampled_samples / self.patch_size))
            return replace(
                song,
                target_frames=target_frames,
                input_samples=resampled_samples,
            )
        target_frames = max(1, int(math.floor(float(song.target_frames) * ratio)))
        return replace(song, target_frames=target_frames)

    def _resolve_patch_fps(self) -> float:
        """Resolve signal FPS from config value (numeric or ``auto``)."""
        return _epoch_resolve_patch_fps(patch_fps=self.patch_fps, songs=self._songs)

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
            shuffle_segments_within_song=self.shuffle_segments_within_song,
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

    def set_global_step(self, step: int) -> None:
        """Update the shared global step used by worker-side augmentation ramps."""
        self._global_step_value.value = int(max(0, step))

    def _global_step(self) -> int:
        """Return the most recent trainer global step visible to this worker."""
        return int(self._global_step_value.value)

    def _load_signals_from_sample(self, sample_dir: Path) -> dict[str, torch.Tensor]:
        """Load one sample's signal tensors from bundle or split artifacts."""
        return _io_load_signals_from_sample(
            sample_dir,
            patch_size=self.patch_size,
            amplitude_lift_enabled=self.amplitude_lift_enabled,
            amplitude_lift_mode=self.amplitude_lift_mode,
            amplitude_lift_reference=self.amplitude_lift_reference,
            amplitude_lift_target_rms=self.amplitude_lift_target_rms,
            amplitude_lift_scale=self.amplitude_lift_scale,
            amplitude_lift_clip_value=self.amplitude_lift_clip_value,
            amplitude_lift_eps=self.amplitude_lift_eps,
            amplitude_lift_gain_power=self.amplitude_lift_gain_power,
            amplitude_lift_gain_min_value=self.amplitude_lift_gain_min_value,
            amplitude_lift_waveform_clamp=self.amplitude_lift_waveform_clamp,
            amplitude_lift_peak_limit=self.amplitude_lift_peak_limit,
            amplitude_lift_peak_rescale_min_rms=(
                self.amplitude_lift_peak_rescale_min_rms
            ),
        )

    def _load_raw_signals_from_sample(
        self, sample_dir: Path
    ) -> dict[str, torch.Tensor]:
        """Load one sample's raw signal tensors without patching full songs."""
        return _io_load_raw_signals_from_sample(sample_dir)

    def _apply_wavflow_lift_to_signals(
        self,
        signals: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply WavFlow-style target lifting to full-song target-side signals."""
        if "target_signal" not in signals:
            raise KeyError("WavFlow amplitude lifting requires target_signal")
        lifted = dict(signals)
        lifted["target_signal"], _target_gain = wavflow_target_transform(
            signals["target_signal"],
            target_rms=self.amplitude_lift_target_rms,
            scale=self.amplitude_lift_scale,
            peak_limit=self.amplitude_lift_peak_limit,
            peak_rescale_min_rms=self.amplitude_lift_peak_rescale_min_rms,
            waveform_clamp=self.amplitude_lift_waveform_clamp,
            eps=self.amplitude_lift_eps,
        )
        if "source_downmix_signal" in signals:
            lifted["source_downmix_signal"], _downmix_gain = wavflow_target_transform(
                signals["source_downmix_signal"],
                target_rms=self.amplitude_lift_target_rms,
                scale=self.amplitude_lift_scale,
                peak_limit=self.amplitude_lift_peak_limit,
                peak_rescale_min_rms=self.amplitude_lift_peak_rescale_min_rms,
                waveform_clamp=self.amplitude_lift_waveform_clamp,
                eps=self.amplitude_lift_eps,
            )

        for key in ("source_stereo_signal", "source_mono_signal"):
            if key not in signals:
                continue
            _lifted_source, source_gain = wavflow_source_transform(
                signals[key],
                target_rms=self.amplitude_lift_target_rms,
                scale=self.amplitude_lift_scale,
                peak_limit=self.amplitude_lift_peak_limit,
                eps=self.amplitude_lift_eps,
            )
            lifted[f"_wavflow_gain_{key}"] = source_gain.detach().cpu()

        return lifted

    def _maybe_resample_training_signals(
        self,
        signals: dict[str, torch.Tensor],
        *,
        source_rate: int,
    ) -> dict[str, torch.Tensor]:
        """Optionally resample full-song waveform signals to training sample rate."""
        if self.training_sample_rate == int(source_rate):
            return signals
        if torchaudio is None:
            raise RuntimeError(
                "data.training_sample_rate requires torchaudio for resampling."
            )
        out: dict[str, torch.Tensor] = {}
        for key, value in signals.items():
            if key.startswith("_"):
                out[key] = value
                continue
            out[key] = torchaudio.functional.resample(
                value.float(),
                orig_freq=int(source_rate),
                new_freq=int(self.training_sample_rate),
            ).contiguous()
        return out

    def _get_song_signals(self, song: SongRecord) -> dict[str, torch.Tensor]:
        """Return raw song signal tensors, using an LRU cache when enabled."""
        if song.sample_dir in self._bundle_cache:
            value = self._bundle_cache.pop(song.sample_dir)
            self._bundle_cache[song.sample_dir] = value
            return value

        signals = self._load_raw_signals_from_sample(song.sample_dir)
        if self.amplitude_lift_enabled and self.amplitude_lift_mode == "wavflow":
            signals = self._apply_wavflow_lift_to_signals(signals)
        signals = self._maybe_resample_training_signals(
            signals,
            source_rate=int(song.sample_rate or self.sample_rate),
        )
        if self.cache_size > 0:
            if self.materialize_cached_signals:
                signals = self._materialize_signals(signals)
            self._bundle_cache[song.sample_dir] = signals
            while len(self._bundle_cache) > self.cache_size:
                self._bundle_cache.popitem(last=False)
        return signals

    @staticmethod
    def _materialize_signals(
        signals: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Force mmap-backed tensors into normal CPU RAM before cache reuse."""
        return {
            key: value.detach().cpu().contiguous().clone()
            for key, value in signals.items()
        }

    def _conditioning_key(self, source: ConditioningSource) -> str:
        """Map conditioning source enum to the matching signal dictionary key."""
        if source == ConditioningSource.MONO:
            return "source_mono_signal"
        if source == ConditioningSource.DOWNMIX:
            return "source_downmix_signal"
        return "source_stereo_signal"

    def _amplitude_lift_gain(
        self,
        *,
        song: SongRecord,
        signals: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Compute the shared full-song gain used for chunk-local amplitude lift."""
        if not self.amplitude_lift_enabled:
            return None
        if self.amplitude_lift_mode == "wavflow":
            return None
        reference_key = (
            "target_signal"
            if self.amplitude_lift_reference == "target"
            else "source_stereo_signal"
        )
        if self.amplitude_lift_mode == "scale":
            return torch.ones((1, 1), dtype=torch.float32)
        if song.signal_rms is not None and reference_key in song.signal_rms:
            rms = torch.tensor(
                float(song.signal_rms[reference_key]), dtype=torch.float32
            )
            return torch.as_tensor(
                self.amplitude_lift_target_rms,
                dtype=torch.float32,
            ) / rms.clamp_min(float(self.amplitude_lift_eps))
        if reference_key not in signals:
            raise KeyError(
                f"Amplitude lifting reference signal missing: {reference_key}"
            )
        reference = signals[reference_key]
        if reference.dim() != 2:
            return None
        return resolve_amplitude_lift_gain(
            reference.float(),
            mode=self.amplitude_lift_mode,
            target_rms=self.amplitude_lift_target_rms,
            eps=self.amplitude_lift_eps,
        )

    def _maybe_lift_chunk(
        self,
        chunk: torch.Tensor,
        *,
        gain: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply model-space amplitude lift to one sliced waveform chunk."""
        if gain is None:
            return chunk.float()
        if self.amplitude_lift_mode == "wavflow":
            return chunk.float()
        return apply_amplitude_lift(
            chunk.float(),
            gain=gain,
            scale=self.amplitude_lift_scale,
            clip_value=(
                None
                if self.amplitude_lift_mode == "scale"
                else self.amplitude_lift_clip_value
            ),
            gain_power=self.amplitude_lift_gain_power,
            gain_min_value=self.amplitude_lift_gain_min_value,
        )

    def _maybe_lift_conditioning_chunk(
        self,
        chunk: torch.Tensor,
        *,
        gain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply conditioning-side WavFlow lift after source augmentations."""
        if not self.amplitude_lift_enabled or self.amplitude_lift_mode != "wavflow":
            return chunk.float(), None
        if gain is not None:
            return chunk.float() * gain.to(dtype=torch.float32), gain
        lifted, gain = wavflow_source_transform(
            chunk.float(),
            target_rms=self.amplitude_lift_target_rms,
            scale=self.amplitude_lift_scale,
            peak_limit=self.amplitude_lift_peak_limit,
            eps=self.amplitude_lift_eps,
        )
        return lifted, gain

    def _amplitude_lift_log_gain_tensor(
        self, gain: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return a scalar [1] log-gain tensor for model conditioning, or None."""
        if gain is None:
            return None
        if self.amplitude_lift_mode == "wavflow":
            return (
                torch.log(gain.float().clamp_min(self.amplitude_lift_eps))
                .flatten()[:1]
                .contiguous()
            )
        log_g = amplitude_lift_log_gain(
            gain,
            scale=self.amplitude_lift_scale,
            gain_power=self.amplitude_lift_gain_power,
            gain_clip_value=(
                None
                if self.amplitude_lift_mode == "scale"
                else self.amplitude_lift_clip_value
            ),
            gain_min_value=self.amplitude_lift_gain_min_value,
            eps=self.amplitude_lift_eps,
        )
        return log_g.flatten()[:1].contiguous()

    def _slice_waveform_with_right_pad(
        self,
        signal: torch.Tensor,
        *,
        start_frame: int,
        num_valid_frames: int,
        window_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice raw ``[C,S]`` waveform by patch frames and pad to window length."""
        if signal.dim() != 2:
            raise ValueError(f"Expected raw [C,S] signal, got {tuple(signal.shape)}")
        if num_valid_frames <= 0:
            raise ValueError(f"num_valid_frames must be > 0, got {num_valid_frames}")
        if num_valid_frames > window_frames:
            raise ValueError(
                f"num_valid_frames={num_valid_frames} cannot exceed window_frames={window_frames}"
            )

        start_sample = int(start_frame) * self.patch_size
        requested_samples = int(num_valid_frames) * self.patch_size
        window_samples = int(window_frames) * self.patch_size
        end_sample = min(signal.shape[-1], start_sample + requested_samples)
        clipped = signal[..., start_sample:end_sample]
        valid_samples = int(clipped.shape[-1])
        if valid_samples <= 0:
            raise RuntimeError(
                "Invalid waveform slice window: "
                f"start_frame={start_frame} total_samples={signal.shape[-1]}"
            )

        padded = torch.zeros(
            (signal.shape[0], window_samples),
            dtype=signal.dtype,
            device=signal.device,
        )
        padded[..., :valid_samples] = clipped
        valid_frames = min(
            int(window_frames),
            max(1, (valid_samples + self.patch_size - 1) // self.patch_size),
        )
        mask = torch.zeros(window_frames, dtype=torch.bool, device=signal.device)
        mask[:valid_frames] = True
        return padded, mask

    def _patch_waveform_chunk(self, signal: torch.Tensor, name: str) -> torch.Tensor:
        """Patch a pre-windowed raw waveform chunk."""
        return _patch_audio(
            signal.float().contiguous(), patch_size=self.patch_size, name=name
        )

    def _augmentation_rng(self, index: int, *, stream: int = 0) -> random.Random:
        """Return deterministic item-local RNG for stochastic audio augmentations."""
        seed = (
            (self.seed + 1) * 1_000_003
            + (self._epoch + 1) * 97_409
            + (int(index) + 1) * 9176
            + int(stream) * 6_364_136_223_846_793_005
        )
        return random.Random(seed)

    @staticmethod
    def _cpt_to_waveform(signal: torch.Tensor) -> torch.Tensor:
        """Convert ``[C, P, T]`` patches to continuous ``[C, S]`` waveform."""
        if signal.dim() != 3:
            raise ValueError(f"Expected [C,P,T] signal, got {tuple(signal.shape)}")
        return signal.permute(0, 2, 1).reshape(signal.shape[0], -1).contiguous()

    @staticmethod
    def _waveform_to_cpt(
        waveform: torch.Tensor,
        *,
        patch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Convert continuous ``[C, S]`` waveform back to fixed ``[C, P, T]``."""
        target_samples = int(patch_size) * int(num_frames)
        if waveform.shape[-1] > target_samples:
            waveform = waveform[..., :target_samples]
        elif waveform.shape[-1] < target_samples:
            pad = torch.zeros(
                (*waveform.shape[:-1], target_samples - waveform.shape[-1]),
                dtype=waveform.dtype,
                device=waveform.device,
            )
            waveform = torch.cat([waveform, pad], dim=-1)
        return (
            waveform.reshape(waveform.shape[0], int(num_frames), int(patch_size))
            .permute(0, 2, 1)
            .contiguous()
        )

    def _apply_source_resample_augmentation(
        self,
        cond_chunk: torch.Tensor,
        *,
        index: int,
    ) -> torch.Tensor:
        """Optionally degrade conditioning audio by sample-rate roundtrip only."""
        if (
            not self.source_resample_aug_enabled
            or self.source_resample_aug_probability <= 0
        ):
            return cond_chunk

        rng = self._augmentation_rng(index, stream=1)
        if rng.random() >= self.source_resample_aug_probability:
            return cond_chunk

        if torchaudio is None:
            raise RuntimeError(
                "source_resample_augmentation requires torchaudio to be installed"
            )

        rate = rng.choices(
            self.source_resample_aug_rates,
            weights=self.source_resample_aug_weights,
            k=1,
        )[0]
        source_rate = int(self.source_resample_aug_sample_rate)
        if int(rate) == source_rate:
            return cond_chunk

        original_dtype = cond_chunk.dtype
        input_was_patched = cond_chunk.dim() == 3
        if input_was_patched:
            waveform = self._cpt_to_waveform(cond_chunk.float())
            num_frames = int(cond_chunk.shape[-1])
        elif cond_chunk.dim() == 2:
            waveform = cond_chunk.float()
            num_frames = 0
        else:
            raise ValueError(
                "source resample augmentation expects [C,S] waveform or [C,P,T] "
                f"patches, got {tuple(cond_chunk.shape)}"
            )
        degraded = cast(
            torch.Tensor,
            torchaudio.functional.resample(
                waveform=waveform,
                orig_freq=source_rate,
                new_freq=int(rate),
            ),
        )
        restored = cast(
            torch.Tensor,
            torchaudio.functional.resample(
                waveform=degraded,
                orig_freq=int(rate),
                new_freq=source_rate,
            ),
        )
        restored = restored.to(dtype=original_dtype)
        if not input_was_patched:
            target_samples = int(cond_chunk.shape[-1])
            if restored.shape[-1] > target_samples:
                return restored[..., :target_samples].contiguous()
            if restored.shape[-1] < target_samples:
                pad = torch.zeros(
                    (*restored.shape[:-1], target_samples - restored.shape[-1]),
                    dtype=restored.dtype,
                    device=restored.device,
                )
                restored = torch.cat([restored, pad], dim=-1)
            return restored.contiguous()
        return self._waveform_to_cpt(
            restored,
            patch_size=self.patch_size,
            num_frames=num_frames,
        )

    def _source_codec_aug_strength(self) -> float:
        """Return current source codec augmentation strength in [0, 1]."""
        if not self.source_codec_aug_enabled:
            return 0.0
        step = self._global_step()
        start = int(self.source_codec_aug_start_step)
        full = int(self.source_codec_aug_full_strength_step)
        if step < start:
            return 0.0
        if full <= start:
            return 1.0
        return min(1.0, max(0.0, float(step - start) / float(full - start)))

    def _apply_source_codec_augmentation(
        self,
        cond_waveform: torch.Tensor,
        *,
        index: int,
    ) -> torch.Tensor:
        """Optionally apply source-only lossy codec encode/decode augmentation."""
        strength = self._source_codec_aug_strength()
        if strength <= 0.0 or self.source_codec_aug_probability <= 0.0:
            return cond_waveform
        if self.source_codec_aug_max_chunk_seconds is not None:
            chunk_seconds = float(cond_waveform.shape[-1]) / float(
                self.source_resample_aug_sample_rate
            )
            if chunk_seconds > self.source_codec_aug_max_chunk_seconds:
                return cond_waveform

        rng = self._augmentation_rng(index, stream=2)
        if rng.random() >= self.source_codec_aug_probability * strength:
            return cond_waveform

        codec = rng.choices(
            self.source_codec_aug_codecs,
            weights=self.source_codec_aug_codec_weights,
            k=1,
        )[0]
        bitrate = rng.choice(self.source_codec_aug_bitrates[codec])
        try:
            return codec_roundtrip(
                cond_waveform,
                sample_rate=self.source_resample_aug_sample_rate,
                codec=codec,
                bitrate_kbps=int(bitrate),
                backend=self.source_codec_aug_backend,
                ffmpeg_path=self.source_codec_aug_ffmpeg_path,
                align=True,
                align_max_lag=self.source_codec_aug_align_max_lag,
                timeout_seconds=self.source_codec_aug_timeout_seconds,
            )
        except Exception:
            # Codec augmentation must never take down a long training job because
            # an optional codec backend is missing or one worker hits a bad encode.
            return cond_waveform

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        segment = self._segments[index]
        song = self._songs[segment.song_index]
        signals = self._get_song_signals(song)
        mix_style = self._mix_style_tensor(song)

        target_signal = signals["target_signal"]
        target_downmix_signal = signals.get("source_downmix_signal")
        cond_key = self._conditioning_key(segment.conditioning_source)
        if cond_key not in signals:
            raise RuntimeError(
                f"Conditioning source {segment.conditioning_source.name.lower()} "
                f"requires {cond_key}, but this sample does not include it. "
                "Set downmix_probability=0.0 for direct stereo/headphone datasets."
            )
        cond_signal = signals[cond_key]
        source_lift_gain: torch.Tensor | None = None
        lift_gain: torch.Tensor | None = None

        if self.sequence_mode == "full_song":
            if target_signal.dim() != 2 or cond_signal.dim() != 2:
                full_signals = self._load_signals_from_sample(song.sample_dir)
                target_signal = full_signals["target_signal"]
                target_downmix_signal = full_signals.get("source_downmix_signal")
                cond_signal = full_signals[cond_key]
                frame_lengths = [target_signal.shape[-1], cond_signal.shape[-1]]
                if target_downmix_signal is not None:
                    frame_lengths.append(target_downmix_signal.shape[-1])
                max_total_frames = min(frame_lengths)
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
                target_chunk = target_signal[..., start_frame:end_frame].contiguous()
                cond_chunk = cond_signal[..., start_frame:end_frame].contiguous()
                target_downmix_chunk = (
                    target_downmix_signal[..., start_frame:end_frame].contiguous()
                    if target_downmix_signal is not None
                    else None
                )
                valid_frames = min(target_chunk.shape[-1], cond_chunk.shape[-1])
                if target_downmix_chunk is not None:
                    valid_frames = min(valid_frames, target_downmix_chunk.shape[-1])
                if valid_frames <= 0:
                    raise RuntimeError(
                        "Invalid full-song chunk: "
                        f"target={tuple(target_chunk.shape)} cond={tuple(cond_chunk.shape)}"
                    )
                target_chunk = target_chunk[..., :valid_frames]
                cond_chunk = cond_chunk[..., :valid_frames]
                if target_downmix_chunk is not None:
                    target_downmix_chunk = target_downmix_chunk[..., :valid_frames]
                cond_chunk = self._apply_source_resample_augmentation(
                    cond_chunk,
                    index=index,
                )
                valid_mask = torch.ones(
                    valid_frames, dtype=torch.bool, device=target_chunk.device
                )
                sample = {
                    "target_signal": target_chunk,
                    "cond_signal": cond_chunk,
                    "valid_mask": valid_mask,
                    "mix_style": mix_style,
                    "song_index": torch.tensor(segment.song_index, dtype=torch.long),
                    "start_frame": torch.tensor(start_frame, dtype=torch.long),
                    "conditioning_source": torch.tensor(
                        int(segment.conditioning_source), dtype=torch.long
                    ),
                }
                if target_downmix_chunk is not None:
                    sample["target_downmix_signal"] = target_downmix_chunk
                if self.amplitude_lift_enabled:
                    sample["amplitude_lift_log_gain"] = torch.zeros(
                        1, dtype=torch.float32
                    )
                return sample

            lift_gain = self._amplitude_lift_gain(song=song, signals=signals)
            frame_lengths = [target_signal.shape[-1], cond_signal.shape[-1]]
            if target_downmix_signal is not None:
                frame_lengths.append(target_downmix_signal.shape[-1])
            max_total_samples = min(frame_lengths)
            max_total_frames = max(
                1, (max_total_samples + self.patch_size - 1) // self.patch_size
            )
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
            del end_frame
            target_waveform, target_mask = self._slice_waveform_with_right_pad(
                target_signal,
                start_frame=start_frame,
                num_valid_frames=num_valid_frames,
                window_frames=num_valid_frames,
            )
            cond_waveform, cond_mask = self._slice_waveform_with_right_pad(
                cond_signal,
                start_frame=start_frame,
                num_valid_frames=num_valid_frames,
                window_frames=num_valid_frames,
            )
            cond_waveform = self._apply_source_codec_augmentation(
                cond_waveform,
                index=index,
            )
            cond_waveform = self._apply_source_resample_augmentation(
                cond_waveform,
                index=index,
            )
            target_chunk = self._patch_waveform_chunk(
                self._maybe_lift_chunk(target_waveform, gain=lift_gain),
                "target_signal",
            )
            cond_waveform = self._maybe_lift_chunk(cond_waveform, gain=lift_gain)
            if cond_key in {"source_stereo_signal", "source_mono_signal"}:
                cond_waveform, source_lift_gain = self._maybe_lift_conditioning_chunk(
                    cond_waveform,
                    gain=signals.get(f"_wavflow_gain_{cond_key}"),
                )
            cond_chunk = self._patch_waveform_chunk(
                cond_waveform,
                "cond_signal",
            )
            target_downmix_chunk = (
                self._patch_waveform_chunk(
                    self._maybe_lift_chunk(
                        self._slice_waveform_with_right_pad(
                            target_downmix_signal,
                            start_frame=start_frame,
                            num_valid_frames=num_valid_frames,
                            window_frames=num_valid_frames,
                        )[0],
                        gain=lift_gain,
                    ),
                    "target_downmix_signal",
                )
                if target_downmix_signal is not None
                else None
            )
            valid_mask = target_mask & cond_mask
            valid_frame_lengths = [target_chunk.shape[-1], cond_chunk.shape[-1]]
            if target_downmix_chunk is not None:
                valid_frame_lengths.append(target_downmix_chunk.shape[-1])
            valid_frames = min(valid_frame_lengths)
            if valid_frames <= 0:
                raise RuntimeError(
                    "Invalid full-song chunk: "
                    f"target={tuple(target_chunk.shape)} cond={tuple(cond_chunk.shape)}"
                )
            target_chunk = target_chunk[..., :valid_frames]
            cond_chunk = cond_chunk[..., :valid_frames]
            if target_downmix_chunk is not None:
                target_downmix_chunk = target_downmix_chunk[..., :valid_frames]
            valid_mask = valid_mask[:valid_frames]
            sample = {
                "target_signal": target_chunk,
                "cond_signal": cond_chunk,
                "valid_mask": valid_mask,
                "mix_style": mix_style,
                "song_index": torch.tensor(segment.song_index, dtype=torch.long),
                "start_frame": torch.tensor(start_frame, dtype=torch.long),
                "conditioning_source": torch.tensor(
                    int(segment.conditioning_source), dtype=torch.long
                ),
            }
            if target_downmix_chunk is not None:
                sample["target_downmix_signal"] = target_downmix_chunk
            log_gain_t = self._amplitude_lift_log_gain_tensor(
                lift_gain if lift_gain is not None else source_lift_gain
            )
            if log_gain_t is not None:
                sample["amplitude_lift_log_gain"] = log_gain_t
            return sample

        if target_signal.dim() != 2 or cond_signal.dim() != 2:
            full_signals = self._load_signals_from_sample(song.sample_dir)
            target_signal = full_signals["target_signal"]
            target_downmix_signal = full_signals.get("source_downmix_signal")
            cond_signal = full_signals[cond_key]
            target_chunk, target_mask = _slice_with_right_pad(
                target_signal,
                segment.start_frame,
                segment.num_valid_frames,
                self.sequence_frames,
            )
            cond_chunk, cond_mask = _slice_with_right_pad(
                cond_signal,
                segment.start_frame,
                segment.num_valid_frames,
                self.sequence_frames,
            )
            target_downmix_chunk = None
            valid_mask = target_mask & cond_mask
            if target_downmix_signal is not None:
                target_downmix_chunk, target_downmix_mask = _slice_with_right_pad(
                    target_downmix_signal,
                    segment.start_frame,
                    segment.num_valid_frames,
                    self.sequence_frames,
                )
                valid_mask = valid_mask & target_downmix_mask
        else:
            lift_gain = self._amplitude_lift_gain(song=song, signals=signals)
            target_waveform, target_mask = self._slice_waveform_with_right_pad(
                target_signal,
                start_frame=segment.start_frame,
                num_valid_frames=segment.num_valid_frames,
                window_frames=self.sequence_frames,
            )
            cond_waveform, cond_mask = self._slice_waveform_with_right_pad(
                cond_signal,
                start_frame=segment.start_frame,
                num_valid_frames=segment.num_valid_frames,
                window_frames=self.sequence_frames,
            )
            cond_waveform = self._apply_source_codec_augmentation(
                cond_waveform,
                index=index,
            )
            cond_waveform = self._apply_source_resample_augmentation(
                cond_waveform,
                index=index,
            )
            target_chunk = self._patch_waveform_chunk(
                self._maybe_lift_chunk(target_waveform, gain=lift_gain),
                "target_signal",
            )
            cond_waveform = self._maybe_lift_chunk(cond_waveform, gain=lift_gain)
            if cond_key in {"source_stereo_signal", "source_mono_signal"}:
                cond_waveform, source_lift_gain = self._maybe_lift_conditioning_chunk(
                    cond_waveform,
                    gain=signals.get(f"_wavflow_gain_{cond_key}"),
                )
            cond_chunk = self._patch_waveform_chunk(
                cond_waveform,
                "cond_signal",
            )
            target_downmix_chunk = None
            valid_mask = target_mask & cond_mask
            if target_downmix_signal is not None:
                (
                    target_downmix_waveform,
                    target_downmix_mask,
                ) = self._slice_waveform_with_right_pad(
                    target_downmix_signal,
                    start_frame=segment.start_frame,
                    num_valid_frames=segment.num_valid_frames,
                    window_frames=self.sequence_frames,
                )
                target_downmix_chunk = self._patch_waveform_chunk(
                    self._maybe_lift_chunk(target_downmix_waveform, gain=lift_gain),
                    "target_downmix_signal",
                )
                valid_mask = valid_mask & target_downmix_mask

        if target_signal.dim() != 2 or cond_signal.dim() != 2:
            cond_chunk = self._apply_source_resample_augmentation(
                cond_chunk,
                index=index,
            )

        sample = {
            "target_signal": target_chunk,
            "cond_signal": cond_chunk,
            "valid_mask": valid_mask,
            "mix_style": mix_style,
            "song_index": torch.tensor(segment.song_index, dtype=torch.long),
            "start_frame": torch.tensor(segment.start_frame, dtype=torch.long),
            "conditioning_source": torch.tensor(
                int(segment.conditioning_source), dtype=torch.long
            ),
        }
        if target_downmix_chunk is not None:
            sample["target_downmix_signal"] = target_downmix_chunk
        log_gain_t = self._amplitude_lift_log_gain_tensor(
            lift_gain if lift_gain is not None else source_lift_gain
        )
        if log_gain_t is not None:
            sample["amplitude_lift_log_gain"] = log_gain_t
        return sample

    def describe(self) -> dict[str, Any]:
        """Return a compact runtime summary of resolved dataset settings."""
        return {
            "num_songs": len(self._songs),
            "num_datasets": len(self.dataset_roots),
            "dataset_roots": [str(path) for path in self.dataset_roots],
            "sample_exclusion_path": self.sample_exclusion_path,
            "sample_rate": self.sample_rate,
            "training_sample_rate": self.training_sample_rate,
            "epoch": self._epoch,
            "epoch_num_segments": len(self._segments),
            "sequence_mode": self.sequence_mode,
            "segment_seconds": self.segment_seconds,
            "sequence_seconds": self.sequence_seconds,
            "stride_seconds": self.stride_seconds,
            "full_song_max_seconds": self.full_song_max_seconds,
            "shuffle_segments_within_epoch": self.shuffle_segments_within_epoch,
            "shuffle_segments_within_song": self.shuffle_segments_within_song,
            "materialize_cached_signals": self.materialize_cached_signals,
            "resolved_patch_fps": self.resolved_patch_fps,
            "segment_frames": self.segment_frames,
            "sequence_frames": self.sequence_frames,
            "stride_frames": self.stride_frames,
            "full_song_max_frames": self.full_song_max_frames,
            "mono_probability": self.mono_probability,
            "downmix_probability": self.downmix_probability,
            "amplitude_lift_enabled": self.amplitude_lift_enabled,
            "amplitude_lift_mode": self.amplitude_lift_mode,
            "amplitude_lift_reference": self.amplitude_lift_reference,
            "amplitude_lift_target_rms": self.amplitude_lift_target_rms,
            "amplitude_lift_scale": self.amplitude_lift_scale,
            "amplitude_lift_clip_value": self.amplitude_lift_clip_value,
            "amplitude_lift_waveform_clamp": self.amplitude_lift_waveform_clamp,
            "amplitude_lift_peak_limit": self.amplitude_lift_peak_limit,
            "amplitude_lift_peak_rescale_min_rms": (
                self.amplitude_lift_peak_rescale_min_rms
            ),
            "source_resample_aug_enabled": self.source_resample_aug_enabled,
            "source_resample_aug_probability": self.source_resample_aug_probability,
            "source_resample_aug_rates": list(self.source_resample_aug_rates),
            "source_codec_aug_enabled": self.source_codec_aug_enabled,
            "source_codec_aug_probability": self.source_codec_aug_probability,
            "source_codec_aug_start_step": self.source_codec_aug_start_step,
            "source_codec_aug_full_strength_step": (
                self.source_codec_aug_full_strength_step
            ),
            "source_codec_aug_backend": self.source_codec_aug_backend,
            "source_codec_aug_codecs": list(self.source_codec_aug_codecs),
            "source_codec_aug_max_chunk_seconds": (
                self.source_codec_aug_max_chunk_seconds
            ),
        }
