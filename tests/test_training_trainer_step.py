from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import stereo2spatial.training.trainer_step as trainer_step_module
from stereo2spatial.training.sequence_plan import SequenceTrainingPlan
from stereo2spatial.training.trainer_settings import TrainerRuntimeSettings


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise RuntimeError("forward is not expected in this unit test")


class _FakeAccelerator:
    def __init__(self) -> None:
        self.sync_gradients = True
        self.backward_calls = 0

    @contextmanager
    def accumulate(self, _model: torch.nn.Module) -> Iterator[None]:
        yield

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def clip_grad_norm_(self, _parameters: Any, _max_norm: float) -> None:
        return


def _build_sequence_plan(*, tbptt_windows: int) -> SequenceTrainingPlan:
    return SequenceTrainingPlan(
        sequence_mode="full_song",
        tbptt_windows=tbptt_windows,
        seq_choices_frames=[8],
        max_choice_frames=8,
        window_frames=8,
        overlap_frames=0,
        window_metadata=None,
        detach_memory=True,
        randomize_per_batch=False,
    )


def _build_settings() -> TrainerRuntimeSettings:
    return cast(
        TrainerRuntimeSettings,
        SimpleNamespace(
            use_gan=False,
            use_channel_aux_losses=False,
            routing_kl_weight=0.0,
            routing_kl_temperature=1.0,
            routing_kl_eps=1e-6,
            corr_weight=0.0,
            corr_eps=1e-6,
            corr_offdiag_only=True,
            corr_use_correlation=True,
        ),
    )


def _build_config() -> Any:
    return SimpleNamespace(
        training=SimpleNamespace(max_steps=100),
        optimizer=SimpleNamespace(lr=0.1),
        scheduler=SimpleNamespace(min_lr=0.1, warmup_steps=0, type="constant"),
    )


def test_run_training_step_skips_second_backward_for_full_song_tbptt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    def _tbptt_loss(**kwargs: Any) -> tuple[torch.Tensor, int, int, None]:
        loss = kwargs["model"].weight.square().sum()
        kwargs["accelerator"].backward(loss)
        return loss.detach(), 8, 2, None

    monkeypatch.setattr(
        trainer_step_module,
        "_compute_full_song_flow_matching_loss",
        _tbptt_loss,
    )

    result = trainer_step_module._run_training_step(
        accelerator=cast(Any, accelerator),
        model=model,
        optimizer=optimizer,
        discriminator=None,
        discriminator_optimizer=None,
        batch={},
        sequence_plan=_build_sequence_plan(tbptt_windows=2),
        global_step=0,
        seed=0,
        settings=_build_settings(),
        grad_clip_norm=0.0,
        config=cast(Any, _build_config()),
    )

    assert accelerator.backward_calls == 1
    assert result.loss.requires_grad is False
    assert model.weight.item() == pytest.approx(0.8, abs=1e-6)


def test_run_training_step_backprops_loss_when_tbptt_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    def _full_song_loss(**kwargs: Any) -> tuple[torch.Tensor, int, int, None]:
        loss = kwargs["model"].weight.square().sum()
        return loss, 8, 2, None

    monkeypatch.setattr(
        trainer_step_module,
        "_compute_full_song_flow_matching_loss",
        _full_song_loss,
    )

    result = trainer_step_module._run_training_step(
        accelerator=cast(Any, accelerator),
        model=model,
        optimizer=optimizer,
        discriminator=None,
        discriminator_optimizer=None,
        batch={},
        sequence_plan=_build_sequence_plan(tbptt_windows=0),
        global_step=0,
        seed=0,
        settings=_build_settings(),
        grad_clip_norm=0.0,
        config=cast(Any, _build_config()),
    )

    assert accelerator.backward_calls == 1
    assert result.loss.requires_grad is True
    assert model.weight.item() == pytest.approx(0.8, abs=1e-6)
