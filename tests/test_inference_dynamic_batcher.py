from __future__ import annotations

import torch

from stereo2spatial.inference.dynamic_batcher import (
    group_compatible_queries,
    run_dynamic_controller_scheduler,
)
from stereo2spatial.inference.solvers import FixedStepSolverController, ModelQuery


class _RecordingCleanPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.t_batches: list[torch.Tensor] = []
        self.return_mem_batches = 0

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
        mix_style: torch.Tensor | None = None,
        amplitude_gain: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del valid_mask
        self.batch_sizes.append(int(zt.shape[0]))
        self.t_batches.append(t.detach().cpu())
        clean = 0.2 * zt + 0.4 * z_cond + t.view(-1, 1, 1, 1).to(zt) * 0.1
        if mix_style is not None:
            clean = clean + mix_style.mean(dim=-1).view(-1, 1, 1, 1) * 0.01
        if amplitude_gain is not None:
            clean = clean + amplitude_gain.view(-1, 1, 1, 1).to(clean) * 0.001
        if return_mem:
            self.return_mem_batches += 1
            if mem is None:
                raise AssertionError("return_mem requires mem")
            return clean, mem + 1.0
        return clean


def _controller(
    *,
    request_id: str,
    solver: str = "euler",
    solver_steps: int = 2,
    mem: torch.Tensor | None = None,
) -> FixedStepSolverController:
    generator = torch.Generator().manual_seed(123 + len(request_id))
    z0 = torch.randn(1, 2, 3, 5, generator=generator)
    cond = torch.randn(1, 2, 3, 5, generator=generator)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)
    return FixedStepSolverController(
        request_id=request_id,
        window_index=0,
        solver=solver,
        solver_steps=solver_steps,
        z0_chunk=z0,
        z_cond=cond,
        valid_mask=valid_mask,
        mem=mem,
        mix_style=torch.zeros(1, 2),
        amplitude_gain=torch.zeros(1, 1),
    )


def test_group_compatible_queries_splits_by_shape_and_batch_size() -> None:
    zt = torch.zeros(1, 2, 3, 5)
    z_cond = torch.zeros(1, 2, 3, 5)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)
    queries = [
        ModelQuery("a", 0, 0, zt, z_cond, valid_mask, 0.0),
        ModelQuery("b", 0, 0, zt, z_cond, valid_mask, 0.1),
        ModelQuery("c", 0, 0, zt[..., :4], z_cond[..., :4], valid_mask[:, :4], 0.2),
    ]

    batches = group_compatible_queries(queries, max_batch_size=1)
    batch_lengths = [len(batch) for batch in batches]

    assert batch_lengths == [1, 1, 1]
    assert batches[0][0].request_id == "a"
    assert batches[1][0].request_id == "b"
    assert batches[2][0].request_id == "c"


def test_dynamic_scheduler_batches_mixed_timestep_queries() -> None:
    model = _RecordingCleanPredictor()
    controllers = [
        _controller(request_id="short", solver="euler", solver_steps=1),
        _controller(request_id="long", solver="euler", solver_steps=3),
    ]

    stats = run_dynamic_controller_scheduler(
        controllers=controllers,
        model=model,
        max_batch_size=2,
    )

    assert stats.completed_controllers == 2
    assert stats.max_observed_batch_size == 2
    assert any(batch.numel() == 2 and batch.unique().numel() == 2 for batch in model.t_batches)
    for controller in controllers:
        assert controller.is_done
        assert controller.result().shape == (1, 2, 3, 5)


def test_dynamic_scheduler_batches_memory_update_queries() -> None:
    model = _RecordingCleanPredictor()
    mem = torch.zeros(1, 4, 8)
    controllers = [
        _controller(request_id="a", solver="heun", solver_steps=1, mem=mem),
        _controller(request_id="b", solver="heun", solver_steps=1, mem=mem),
    ]

    stats = run_dynamic_controller_scheduler(
        controllers=controllers,
        model=model,
        max_batch_size=2,
    )

    assert stats.completed_controllers == 2
    assert model.return_mem_batches == 1
    for controller in controllers:
        assert controller.mem_out is not None
        assert torch.allclose(controller.mem_out, mem + 1.0)


def test_dynamic_scheduler_respects_max_active_controllers() -> None:
    model = _RecordingCleanPredictor()
    controllers = [
        _controller(request_id="a", solver="euler", solver_steps=1),
        _controller(request_id="b", solver="euler", solver_steps=1),
        _controller(request_id="c", solver="euler", solver_steps=1),
    ]

    stats = run_dynamic_controller_scheduler(
        controllers=controllers,
        model=model,
        max_batch_size=8,
        max_active_controllers=2,
    )

    assert stats.completed_controllers == 3
    assert stats.max_observed_active_controllers == 2
    assert stats.max_observed_batch_size == 2
