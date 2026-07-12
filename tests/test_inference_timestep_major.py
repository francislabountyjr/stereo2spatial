from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from stereo2spatial.inference.sampling import (
    _resolve_time_grid as resolve_window_major_time_grid,
)
from stereo2spatial.inference.sampling import generate_spatial_signal
from stereo2spatial.inference.timestep_sampling import (
    _resolve_time_grid as resolve_timestep_major_time_grid,
)
from stereo2spatial.inference.timestep_sampling import (
    generate_spatial_signal_timestep_major,
)


class _DeterministicSamplingModel(torch.nn.Module):
    target_channels = 2
    cond_channels = 2
    patch_size = 2

    def init_memory(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        del batch_size, device, dtype
        return None

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> torch.Tensor:
        del valid_mask, mem
        if return_mem:
            raise AssertionError("memoryless model must not request returned memory")
        t4 = t[:, None, None, None].to(device=zt.device, dtype=zt.dtype)
        return 0.35 * zt + 0.2 * z_cond + (1.0 - t4) * 0.05


class _PaddingRecordingModel(_DeterministicSamplingModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> torch.Tensor:
        self.calls.append(
            (
                zt.detach().cpu().clone(),
                z_cond.detach().cpu().clone(),
                None if valid_mask is None else valid_mask.detach().cpu().clone(),
            )
        )
        return super().forward(
            zt=zt,
            t=t,
            z_cond=z_cond,
            valid_mask=valid_mask,
            mem=mem,
            return_mem=return_mem,
        )


class _MemorySweepRecordingModel(torch.nn.Module):
    target_channels = 2
    cond_channels = 2
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, float | None, float | None, float | None]] = []

    def init_memory(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        self.events.append(("init", None, None, None))
        return torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del valid_mask
        if mem is None or not return_mem:
            raise AssertionError(
                "each timestep-major window must carry and return memory"
            )
        self.events.append(
            (
                "call",
                float(mem.mean().item()),
                float(z_cond[0, 0, 0, 0].item()),
                float(t[0].item()),
            )
        )
        t4 = t[:, None, None, None].to(device=zt.device, dtype=zt.dtype)
        clean = zt + (1.0 - t4) * 0.05
        return clean, mem + 1.0


def _sampling_kwargs(
    model: torch.nn.Module,
    cond_signal: torch.Tensor,
    *,
    solver: str = "euler",
    solver_steps: int = 2,
    seed: int = 123,
) -> dict[str, Any]:
    return {
        "model": cast(Any, model),
        "cond_signal": cond_signal,
        "chunk_frames": cond_signal.shape[-1],
        "overlap_frames": 0,
        "solver": solver,
        "solver_steps": solver_steps,
        "solver_rtol": 1e-5,
        "solver_atol": 1e-5,
        "seed": seed,
    }


@pytest.mark.parametrize(
    "resolver",
    [resolve_window_major_time_grid, resolve_timestep_major_time_grid],
)
def test_adaptive_solver_time_grids_remain_float32_in_low_precision(
    resolver: Any,
) -> None:
    grid = resolver(method="dopri5", num_steps=4, device=torch.device("cpu"))

    assert grid.dtype == torch.float32
    assert float(grid[-1].item()) < 1.0


def test_timestep_major_is_deterministic_for_same_seed() -> None:
    model = _DeterministicSamplingModel()
    cond_signal = torch.randn(2, 2, 11)
    kwargs = _sampling_kwargs(model, cond_signal, solver="heun", solver_steps=3)
    kwargs.update(chunk_frames=5, overlap_frames=2)

    pred_a = generate_spatial_signal_timestep_major(**kwargs)
    pred_b = generate_spatial_signal_timestep_major(**kwargs)
    pred_c = generate_spatial_signal_timestep_major(**{**kwargs, "seed": 999})

    assert pred_a.shape == (2, 2, 11)
    assert torch.allclose(pred_a, pred_b)
    assert not torch.allclose(pred_a, pred_c)


def test_timestep_major_right_pads_every_tail_evaluation_and_sets_valid_mask() -> None:
    model = _PaddingRecordingModel()
    frame_values = torch.arange(1, 9, dtype=torch.float32)
    cond_signal = frame_values.view(1, 1, -1).expand(2, 2, -1).clone()

    pred = generate_spatial_signal_timestep_major(
        model=cast(Any, model),
        cond_signal=cond_signal,
        chunk_frames=4,
        overlap_frames=1,
        solver="euler",
        solver_steps=1,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
    )

    assert pred.shape == (2, 2, 8)
    assert model.calls
    assert len(model.calls) % 3 == 0

    expected_masks = (
        torch.tensor([[True, True, True, True]]),
        torch.tensor([[True, True, True, True]]),
        torch.tensor([[True, True, False, False]]),
    )
    for sweep_start in range(0, len(model.calls), 3):
        sweep = model.calls[sweep_start : sweep_start + 3]
        assert [float(call[1][0, 0, 0, 0].item()) for call in sweep] == [
            1.0,
            4.0,
            7.0,
        ]
        for (zt, z_cond, valid_mask), expected_mask in zip(
            sweep, expected_masks, strict=True
        ):
            assert zt.shape == (1, 2, 2, 4)
            assert z_cond.shape == (1, 2, 2, 4)
            assert valid_mask is not None
            assert torch.equal(valid_mask, expected_mask)

        tail_zt, tail_cond, _ = sweep[-1]
        assert torch.equal(tail_cond[0, 0, 0], torch.tensor([7.0, 8.0, 0.0, 0.0]))
        assert torch.count_nonzero(tail_zt[..., 2:]).item() == 0


def test_timestep_major_resets_memory_for_each_left_to_right_global_sweep() -> None:
    model = _MemorySweepRecordingModel()
    frame_values = torch.arange(1, 9, dtype=torch.float32)
    cond_signal = frame_values.view(1, 1, -1).expand(2, 2, -1).clone()

    pred = generate_spatial_signal_timestep_major(
        model=cast(Any, model),
        cond_signal=cond_signal,
        chunk_frames=4,
        overlap_frames=1,
        solver="euler",
        solver_steps=2,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
    )

    assert pred.shape == (2, 2, 8)
    # Two Euler field evaluations plus the final accepted-state clean prediction.
    assert len(model.events) == 3 * 4
    for sweep_start in range(0, len(model.events), 4):
        init_event, *call_events = model.events[sweep_start : sweep_start + 4]
        assert init_event[0] == "init"
        assert [event[0] for event in call_events] == ["call", "call", "call"]
        assert [event[1] for event in call_events] == pytest.approx([0.0, 1.0, 2.0])
        assert [event[2] for event in call_events] == pytest.approx([1.0, 4.0, 7.0])
        assert len({event[3] for event in call_events}) == 1


@pytest.mark.parametrize("solver", ["euler", "heun", "midpoint_rk2", "res6s"])
def test_timestep_major_matches_window_major_for_one_window(solver: str) -> None:
    model = _DeterministicSamplingModel()
    cond_signal = torch.randn(2, 2, 6)
    kwargs = _sampling_kwargs(model, cond_signal, solver=solver, solver_steps=3)

    window_major = generate_spatial_signal(**kwargs)
    timestep_major = generate_spatial_signal_timestep_major(**kwargs)

    assert torch.allclose(timestep_major, window_major, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("one_step_input", ["zeros", "cond"])
def test_timestep_major_one_step_matches_window_major_for_one_window(
    one_step_input: str,
) -> None:
    model = _DeterministicSamplingModel()
    cond_signal = torch.randn(2, 2, 6)
    kwargs = _sampling_kwargs(model, cond_signal)
    kwargs.update(one_step=True, one_step_input=one_step_input)

    window_major = generate_spatial_signal(**kwargs)
    timestep_major = generate_spatial_signal_timestep_major(**kwargs)

    assert torch.allclose(timestep_major, window_major, rtol=1e-6, atol=1e-6)
