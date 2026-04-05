from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from stereo2spatial.training.trainer_loop import (
    _compute_conditioning_counts,
    _resolve_epoch_iterator,
    _should_run_validation,
)
from stereo2spatial.training.trainer_settings import TrainerRuntimeSettings


class _FakeAccelerator:
    def __init__(self) -> None:
        self.skip_calls: list[int] = []

    def skip_first_batches(self, dataloader: list[int], count: int) -> list[int]:
        self.skip_calls.append(count)
        return dataloader[count:]

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        return tensor


def test_resolve_epoch_iterator_returns_full_loader_without_resume() -> None:
    accelerator = _FakeAccelerator()
    loader = [1, 2, 3]
    iterator, seen, resume_left, skip_epoch, message = _resolve_epoch_iterator(
        accelerator=accelerator,
        dataloader=cast(DataLoader[Any], loader),
        resume_batches_seen=0,
        total_batches_this_epoch=3,
    )
    assert iterator == loader
    assert seen == 0
    assert resume_left == 0
    assert skip_epoch is False
    assert message is None
    assert accelerator.skip_calls == []


def test_resolve_epoch_iterator_skips_whole_epoch_when_resume_exhausted() -> None:
    accelerator = _FakeAccelerator()
    iterator, seen, resume_left, skip_epoch, message = _resolve_epoch_iterator(
        accelerator=accelerator,
        dataloader=cast(DataLoader[Any], [1, 2, 3]),
        resume_batches_seen=3,
        total_batches_this_epoch=3,
    )
    assert iterator == [1, 2, 3]
    assert seen == 0
    assert resume_left == 0
    assert skip_epoch is True
    assert message is None


def test_resolve_epoch_iterator_skips_prefix_and_emits_message() -> None:
    accelerator = _FakeAccelerator()
    iterator, seen, resume_left, skip_epoch, message = _resolve_epoch_iterator(
        accelerator=accelerator,
        dataloader=cast(DataLoader[Any], [10, 20, 30, 40]),
        resume_batches_seen=2,
        total_batches_this_epoch=4,
    )
    assert iterator == [30, 40]
    assert seen == 2
    assert resume_left == 0
    assert skip_epoch is False
    assert message is not None and "2/4" in message
    assert accelerator.skip_calls == [2]


def test_compute_conditioning_counts_reduces_batch_histogram() -> None:
    accelerator = _FakeAccelerator()
    batch = {
        "conditioning_source": torch.tensor([0, 1, 2, 2, 0], dtype=torch.long),
        "target_latent": torch.zeros(5, 1, 1, 1),
    }
    stereo, mono, downmix = _compute_conditioning_counts(
        batch=batch,
        accelerator=accelerator,
    )
    assert (stereo, mono, downmix) == (2, 1, 2)


def test_should_run_validation_uses_step_cadence() -> None:
    settings = cast(
        TrainerRuntimeSettings,
        SimpleNamespace(validation_steps=4),
    )
    assert _should_run_validation(settings=settings, global_step=4) is True
    assert _should_run_validation(settings=settings, global_step=8) is True
    assert _should_run_validation(settings=settings, global_step=5) is False
    assert _should_run_validation(
        settings=cast(
            TrainerRuntimeSettings,
            SimpleNamespace(validation_steps=0),
        ),
        global_step=100,
    ) is False
