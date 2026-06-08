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
            downmix_consistency_weight=0.0,
            downmix_consistency_loss="mse",
            downmix_channel_order=None,
            mrstft_loss_weight=0.0,
            mrstft_fft_sizes=[512, 1024, 2048],
            mrstft_hop_lengths=[128, 256, 512],
            mrstft_win_lengths=[512, 1024, 2048],
            mrstft_sc_weight=1.0,
            mrstft_log_mag_weight=1.0,
            mrstft_eps=1e-7,
            waveform_mse_loss_weight=1.0,
            waveform_l1_loss_weight=0.0,
            waveform_charbonnier_loss_weight=0.0,
            waveform_charbonnier_eps=1e-3,
            perceptual_loss_weight=0.0,
            perceptual_sample_rate=48000,
            perceptual_n_fft=1024,
            perceptual_hop_length=256,
            perceptual_win_length=1024,
            perceptual_n_mels=80,
            perceptual_f_min=40.0,
            perceptual_f_max=None,
            perceptual_band_weight=1.0,
            perceptual_band_low_hz=150.0,
            perceptual_band_high_hz=8000.0,
            perceptual_eps=1e-5,
            binaural_ild_loss_weight=0.0,
            binaural_ipd_loss_weight=0.0,
            binaural_ccf_loss_weight=0.0,
            binaural_frame_ild_loss_weight=0.0,
            binaural_frame_ild_frame_size=2048,
            binaural_frame_ild_hop_size=1024,
            binaural_frame_ild_silence_threshold=1e-4,
            binaural_frame_ild_max_weight=4.0,
            binaural_mid_side_loss_weight=0.0,
            binaural_mid_side_loss_type="charbonnier",
            binaural_mid_side_mid_weight=0.0,
            binaural_mid_side_side_weight=1.0,
            binaural_mid_side_charbonnier_eps=1e-3,
            binaural_loss_warmup_steps=0,
            binaural_sample_rate=48000,
            binaural_loss_eps=1e-7,
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


def test_run_training_step_skips_nonfinite_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    def _batch_loss(**_kwargs: Any) -> tuple[torch.Tensor, int, int, None]:
        return model.weight * torch.tensor(float("nan")), 8, 1, None

    monkeypatch.setattr(
        trainer_step_module,
        "_compute_batch_flow_matching_loss",
        _batch_loss,
    )

    result = trainer_step_module._run_training_step(
        accelerator=cast(Any, accelerator),
        model=model,
        optimizer=optimizer,
        discriminator=None,
        discriminator_optimizer=None,
        batch={},
        sequence_plan=SequenceTrainingPlan(
            sequence_mode="strided_crops",
            tbptt_windows=0,
            seq_choices_frames=[8],
            max_choice_frames=8,
            window_frames=8,
            overlap_frames=0,
            window_metadata=None,
            detach_memory=True,
            randomize_per_batch=False,
        ),
        global_step=0,
        seed=0,
        settings=_build_settings(),
        grad_clip_norm=1.0,
        config=cast(Any, _build_config()),
    )

    assert result.skipped_step is True
    assert result.skip_reason == "nonfinite_loss"
    assert accelerator.backward_calls == 0
    assert model.weight.item() == pytest.approx(1.0)
