from __future__ import annotations

import pytest
import torch

from stereo2spatial.inference.sampling import _sample_chunk_signal
from stereo2spatial.inference.solvers import (
    FixedStepSolverController,
    ModelQuery,
    stack_model_queries,
)


class _CleanPredictor(torch.nn.Module):
    target_channels = 2
    cond_channels = 2
    patch_size = 3

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
        clean = 0.25 * zt + 0.5 * z_cond
        clean = clean + t.view(-1, 1, 1, 1).to(clean) * 0.1
        if mix_style is not None:
            clean = clean + mix_style.mean(dim=-1).view(-1, 1, 1, 1) * 0.01
        if amplitude_gain is not None:
            clean = clean + amplitude_gain.view(-1, 1, 1, 1).to(clean) * 0.001
        if return_mem:
            if mem is None:
                raise AssertionError("return_mem requires mem")
            return clean, mem + 1.0
        return clean


def _run_controller(
    controller: FixedStepSolverController,
    model: _CleanPredictor,
) -> torch.Tensor:
    while not controller.is_done:
        query = controller.next_query()
        assert query is not None
        output = model(
            zt=query.zt,
            t=torch.tensor([query.t_value], dtype=query.zt.dtype, device=query.zt.device),
            z_cond=query.z_cond,
            valid_mask=query.valid_mask,
            mem=query.mem,
            return_mem=query.return_mem,
            mix_style=query.mix_style,
            amplitude_gain=query.amplitude_gain,
        )
        if query.return_mem:
            clean, mem_out = output
            controller.accept_output(clean, mem_out=mem_out)
        else:
            assert isinstance(output, torch.Tensor)
            controller.accept_output(output)
    return controller.result()


@pytest.mark.parametrize("solver", ["euler", "heun", "midpoint_rk2", "res6s"])
def test_fixed_step_controller_matches_existing_sampler(solver: str) -> None:
    torch.manual_seed(123)
    model = _CleanPredictor()
    cond_chunk = torch.randn(1, 2, 3, 5)
    z0_chunk = torch.randn(1, 2, 3, 5)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)
    mix_style = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    amplitude_gain = torch.tensor([[0.5]], dtype=torch.float32)

    expected, _ = _sample_chunk_signal(
        model=model,  # type: ignore[arg-type]
        cond_chunk=cond_chunk,
        valid_mask=valid_mask,
        z0_chunk=z0_chunk,
        solver=solver,
        solver_steps=3,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        mem=None,
        mix_style=mix_style,
        amplitude_gain=amplitude_gain,
    )
    controller = FixedStepSolverController(
        request_id="song-a",
        window_index=0,
        solver=solver,
        solver_steps=3,
        z0_chunk=z0_chunk,
        z_cond=cond_chunk,
        valid_mask=valid_mask,
        mem=None,
        mix_style=mix_style,
        amplitude_gain=amplitude_gain,
    )

    actual = _run_controller(controller, model)

    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_stack_model_queries_allows_mixed_timestep_values() -> None:
    zt = torch.zeros(1, 2, 3, 5)
    z_cond = torch.ones(1, 2, 3, 5)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)
    mem = torch.zeros(1, 4, 8)
    mix_style = torch.zeros(1, 2)
    amplitude_gain = torch.zeros(1, 1)
    queries = [
        ModelQuery(
            request_id="a",
            window_index=0,
            query_index=0,
            zt=zt,
            z_cond=z_cond,
            valid_mask=valid_mask,
            t_value=0.0,
            mem=mem,
            mix_style=mix_style,
            amplitude_gain=amplitude_gain,
        ),
        ModelQuery(
            request_id="b",
            window_index=2,
            query_index=4,
            zt=zt + 1.0,
            z_cond=z_cond + 2.0,
            valid_mask=valid_mask,
            t_value=0.375,
            mem=mem + 3.0,
            mix_style=mix_style + 4.0,
            amplitude_gain=amplitude_gain + 5.0,
        ),
    ]

    batch = stack_model_queries(queries)

    assert batch["zt"].shape == (2, 2, 3, 5)
    assert batch["z_cond"].shape == (2, 2, 3, 5)
    assert batch["valid_mask"].shape == (2, 5)
    assert torch.equal(batch["t"], torch.tensor([0.0, 0.375]))
    assert batch["mem"].shape == (2, 4, 8)
    assert batch["mix_style"].shape == (2, 2)
    assert batch["amplitude_gain"].shape == (2, 1)


def test_stack_model_queries_rejects_mixed_return_memory_queries() -> None:
    zt = torch.zeros(1, 2, 3, 5)
    z_cond = torch.ones(1, 2, 3, 5)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="return_mem"):
        stack_model_queries(
            [
                ModelQuery("a", 0, 0, zt, z_cond, valid_mask, 0.0),
                ModelQuery("b", 0, 0, zt, z_cond, valid_mask, 0.0, return_mem=True),
            ]
        )


def test_controller_emits_memory_update_query_after_final_prediction() -> None:
    model = _CleanPredictor()
    cond_chunk = torch.zeros(1, 2, 3, 5)
    z0_chunk = torch.randn(1, 2, 3, 5)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)
    mem = torch.zeros(1, 4, 8)
    controller = FixedStepSolverController(
        request_id="song-a",
        window_index=1,
        solver="euler",
        solver_steps=1,
        z0_chunk=z0_chunk,
        z_cond=cond_chunk,
        valid_mask=valid_mask,
        mem=mem,
    )

    result = _run_controller(controller, model)

    assert controller.is_done is True
    assert controller.mem_out is not None
    assert torch.allclose(controller.mem_out, mem + 1.0)
    assert result.shape == z0_chunk.shape
