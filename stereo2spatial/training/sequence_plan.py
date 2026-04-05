"""Sequence/window planning for trainer runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TrainConfig
from .dataset import LatentSongDataset
from .windowing import _build_window_metadata


@dataclass(frozen=True)
class SequenceTrainingPlan:
    """Resolved frame-level sequencing/window choices consumed by trainer steps."""

    sequence_mode: str
    tbptt_windows: int
    seq_choices_frames: list[int]
    max_choice_frames: int
    window_frames: int
    overlap_frames: int
    window_metadata: dict[int, tuple[list[int], list[torch.Tensor]]] | None
    detach_memory: bool
    randomize_per_batch: bool


def build_sequence_training_plan(
    *,
    config: TrainConfig,
    dataset: LatentSongDataset,
) -> SequenceTrainingPlan:
    """Resolve per-batch temporal slicing and overlap parameters."""
    fps = float(dataset.resolved_latent_fps)
    sequence_mode = (
        str(getattr(config.training, "sequence_mode", "strided_crops")).strip().lower()
    )
    tbptt_windows = int(getattr(config.training, "tbptt_windows", 0))
    seq_choices_sec = getattr(config.training, "sequence_seconds_choices", None)
    if not seq_choices_sec:
        seq_choices_sec = [float(dataset.sequence_seconds)]
    seq_choices_frames = [max(1, int(round(float(value) * fps))) for value in seq_choices_sec]
    max_choice_frames = max(seq_choices_frames)

    window_frames = max(1, int(round(float(config.training.window_seconds) * fps)))
    overlap_frames = int(round(float(config.training.overlap_seconds) * fps))
    overlap_frames = min(overlap_frames, max(0, window_frames - 1))
    window_metadata: dict[int, tuple[list[int], list[torch.Tensor]]] | None = None
    if sequence_mode != "full_song":
        window_metadata = _build_window_metadata(
            frame_lengths=seq_choices_frames + [max_choice_frames, int(dataset.sequence_frames)],
            window_frames=window_frames,
            overlap_frames=overlap_frames,
        )

    detach_memory = bool(getattr(config.training, "detach_memory", True))
    randomize_per_batch = bool(
        getattr(config.training, "randomize_sequence_per_batch", True)
    )
    if sequence_mode == "full_song":
        randomize_per_batch = False

    return SequenceTrainingPlan(
        sequence_mode=sequence_mode,
        tbptt_windows=tbptt_windows,
        seq_choices_frames=seq_choices_frames,
        max_choice_frames=max_choice_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        window_metadata=window_metadata,
        detach_memory=detach_memory,
        randomize_per_batch=randomize_per_batch,
    )
