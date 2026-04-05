from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stereo2spatial.training.scheduled_sampling import (
    ScheduledSamplingPlan,
    apply_flow_matching_scheduled_sampling,
    build_rollout_plan,
    resolve_scheduled_sampling_probability,
)


class _TimeVelocityModel(torch.nn.Module):
    timestep_scale = 11.0

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del z_cond, valid_mask
        return t[:, None, None, None].to(dtype=zt.dtype, device=zt.device).expand_as(zt)


class _StateAwareVelocityModel(torch.nn.Module):
    timestep_scale = 11.0

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del z_cond, valid_mask
        return zt + t[:, None, None, None].to(dtype=zt.dtype, device=zt.device)


class _MemoryCarryVelocityModel(torch.nn.Module):
    timestep_scale = 11.0
    num_memory_tokens = 1

    def __init__(self) -> None:
        super().__init__()
        self.return_mem_calls = 0

    def init_memory(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del t, z_cond, valid_mask
        if mem is None:
            raise RuntimeError("mem is required for memory-carry rollout test model")
        velocity = mem[:, :, :, None].to(dtype=zt.dtype, device=zt.device).expand_as(zt)
        if return_mem:
            self.return_mem_calls += 1
            return velocity, mem + 1.0
        return velocity


def test_resolve_scheduled_sampling_probability_ramp() -> None:
    start = resolve_scheduled_sampling_probability(
        max_step_offset=4,
        probability=0.9,
        prob_start=0.1,
        prob_end=0.5,
        ramp_steps=100,
        ramp_start_step=10,
        ramp_shape="cosine",
        global_step=10,
    )
    mid = resolve_scheduled_sampling_probability(
        max_step_offset=4,
        probability=0.9,
        prob_start=0.1,
        prob_end=0.5,
        ramp_steps=100,
        ramp_start_step=10,
        ramp_shape="cosine",
        global_step=60,
    )
    end = resolve_scheduled_sampling_probability(
        max_step_offset=4,
        probability=0.9,
        prob_start=0.1,
        prob_end=0.5,
        ramp_steps=100,
        ramp_start_step=10,
        ramp_shape="cosine",
        global_step=200,
    )
    disabled = resolve_scheduled_sampling_probability(
        max_step_offset=0,
        probability=0.9,
        prob_start=0.1,
        prob_end=0.5,
        ramp_steps=100,
        ramp_start_step=10,
        ramp_shape="cosine",
        global_step=200,
    )

    assert start == pytest.approx(0.1)
    assert 0.1 < mid < 0.5
    assert end == pytest.approx(0.5)
    assert disabled == pytest.approx(0.0)


def test_build_rollout_plan_respects_probability_mask() -> None:
    base_steps = torch.tensor([2, 5, 9], dtype=torch.long)
    plan = build_rollout_plan(
        base_steps=base_steps,
        max_step_offset=4,
        apply_probability=0.0,
        strategy="uniform",
        num_time_steps=1000,
    )

    assert torch.equal(plan.target_steps, base_steps)
    assert torch.equal(plan.source_steps, base_steps)
    assert torch.equal(plan.rollout_steps, torch.zeros_like(base_steps))


def test_apply_flow_matching_scheduled_sampling_noop_when_disabled() -> None:
    model = _TimeVelocityModel()
    z1 = torch.ones(1, 1, 1, 2, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.7], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=0,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="heun",
        scheduled_sampling_reflexflow=None,
    )
    result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
    )

    assert torch.allclose(result.t, t)
    assert torch.allclose(result.zt, zt)
    assert result.reflexflow.enabled is False
    assert result.reflex_clean_pred is None
    assert result.reflex_biased_pred is None


def test_apply_flow_matching_scheduled_sampling_supports_heun_solver() -> None:
    model = _TimeVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([4], dtype=torch.long),
        rollout_steps=torch.tensor([4], dtype=torch.long),
    )

    euler_cfg = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="euler",
        scheduled_sampling_reflexflow=False,
    )
    heun_cfg = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="heun",
        scheduled_sampling_reflexflow=False,
    )

    euler_result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=euler_cfg,
        global_step=0,
        plan=plan,
    )
    heun_result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=heun_cfg,
        global_step=0,
        plan=plan,
    )
    euler_zt = euler_result.zt
    heun_zt = heun_result.zt

    assert euler_zt.item() == pytest.approx(0.62, abs=1e-5)
    assert heun_zt.item() == pytest.approx(0.64, abs=1e-5)
    assert not torch.allclose(euler_zt, heun_zt)


def test_apply_flow_matching_scheduled_sampling_supports_unipc_solver() -> None:
    model = _StateAwareVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([4], dtype=torch.long),
        rollout_steps=torch.tensor([4], dtype=torch.long),
    )

    euler_cfg = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="euler",
        scheduled_sampling_reflexflow=False,
    )
    heun_cfg = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="heun",
        scheduled_sampling_reflexflow=False,
    )
    unipc_cfg = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="unipc",
        scheduled_sampling_reflexflow=False,
    )

    euler_result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=euler_cfg,
        global_step=0,
        plan=plan,
    )
    heun_result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=heun_cfg,
        global_step=0,
        plan=plan,
    )
    unipc_result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=unipc_cfg,
        global_step=0,
        plan=plan,
    )

    assert euler_result.zt.item() == pytest.approx(0.83538, abs=1e-5)
    assert heun_result.zt.item() == pytest.approx(0.8836237, abs=1e-5)
    assert unipc_result.zt.item() == pytest.approx(0.8847706, abs=1e-5)
    assert not torch.allclose(unipc_result.zt, euler_result.zt)
    assert not torch.allclose(unipc_result.zt, heun_result.zt)


def test_apply_flow_matching_scheduled_sampling_rollout_uses_plan_target_steps() -> None:
    model = _TimeVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.83], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([4], dtype=torch.long),
        rollout_steps=torch.tensor([4], dtype=torch.long),
    )
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="euler",
        scheduled_sampling_reflexflow=False,
    )

    result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
        plan=plan,
    )

    assert result.t.item() == pytest.approx(0.8, abs=1e-6)
    assert result.zt.item() == pytest.approx(0.62, abs=1e-5)


def test_apply_flow_matching_scheduled_sampling_rollout_threads_memory_state() -> None:
    model = _MemoryCarryVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([4], dtype=torch.long),
        rollout_steps=torch.tensor([2], dtype=torch.long),
    )
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="euler",
        scheduled_sampling_reflexflow=False,
    )

    result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
        plan=plan,
    )

    assert model.return_mem_calls == 2
    assert result.zt.item() == pytest.approx(0.6, abs=1e-5)


def test_heun_rollout_memory_should_commit_only_accepted_steps() -> None:
    model = _MemoryCarryVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([4], dtype=torch.long),
        rollout_steps=torch.tensor([2], dtype=torch.long),
    )
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="heun",
        scheduled_sampling_reflexflow=False,
    )

    apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
        plan=plan,
    )

    # Compile-stable probe calls use return_mem=True but do not commit state.
    # With rollout_steps=2 (Heun), we expect 2 accepted-step calls + 2 probes.
    assert model.return_mem_calls == 4


def test_unipc_rollout_memory_should_commit_only_accepted_steps() -> None:
    model = _MemoryCarryVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([4], dtype=torch.long),
        rollout_steps=torch.tensor([2], dtype=torch.long),
    )
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="unipc",
        scheduled_sampling_reflexflow=False,
    )

    apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
        plan=plan,
    )

    # Compile-stable probe calls use return_mem=True but do not commit state.
    # With rollout_steps=2 (UniPC), we expect 2 accepted-step calls + 2 probes.
    assert model.return_mem_calls == 4


def test_windowed_rollout_commits_memory_once_per_window_endpoint() -> None:
    model = _MemoryCarryVelocityModel()
    z1 = torch.ones(1, 1, 1, 2, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8], dtype=torch.long),
        source_steps=torch.tensor([5], dtype=torch.long),
        rollout_steps=torch.tensor([3], dtype=torch.long),
    )
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="euler",
        scheduled_sampling_reflexflow=False,
    )

    result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
        plan=plan,
        window_frames=1,
        overlap_frames=0,
    )

    # With 2 windows and 3 rollout steps per window (Euler), probe-style calls
    # contribute 3 calls/window and endpoint commit contributes 1/window.
    assert model.return_mem_calls == 8
    assert result.t.item() == pytest.approx(0.8, abs=1e-6)
    assert result.zt[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert result.zt[0, 0, 0, 1].item() == pytest.approx(0.8, abs=1e-5)


def test_reflexflow_auto_enabled_and_populates_prediction_caches() -> None:
    model = _StateAwareVelocityModel()
    z1 = torch.ones(2, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.8, 0.8], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(2, 1, dtype=torch.bool)
    plan = ScheduledSamplingPlan(
        target_steps=torch.tensor([8, 8], dtype=torch.long),
        source_steps=torch.tensor([4, 8], dtype=torch.long),
        rollout_steps=torch.tensor([4, 0], dtype=torch.long),
    )
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=4,
        scheduled_sampling_probability=1.0,
        scheduled_sampling_sampler="heun",
        scheduled_sampling_reflexflow=None,
        scheduled_sampling_reflexflow_alpha=1.0,
        scheduled_sampling_reflexflow_beta1=10.0,
        scheduled_sampling_reflexflow_beta2=1.0,
    )

    result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
        plan=plan,
    )

    assert result.reflexflow.enabled is True
    assert result.reflex_clean_pred is not None
    assert result.reflex_biased_pred is not None
    # rollout sample differs, non-rollout sample stays equal
    assert not torch.allclose(result.reflex_clean_pred[0], result.reflex_biased_pred[0])
    assert torch.allclose(result.reflex_clean_pred[1], result.reflex_biased_pred[1])


def test_reflexflow_can_be_enabled_without_scheduled_sampling_rollout() -> None:
    model = _TimeVelocityModel()
    z1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    z0 = torch.zeros_like(z1)
    t = torch.tensor([0.7], dtype=torch.float32)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    z_cond = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    config = SimpleNamespace(
        scheduled_sampling_max_step_offset=0,
        scheduled_sampling_probability=0.0,
        scheduled_sampling_sampler="heun",
        scheduled_sampling_reflexflow=True,
        scheduled_sampling_reflexflow_alpha=1.0,
        scheduled_sampling_reflexflow_beta1=10.0,
        scheduled_sampling_reflexflow_beta2=1.0,
    )

    result = apply_flow_matching_scheduled_sampling(
        model=model,
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        zt=zt,
        z0=z0,
        training_config=config,
        global_step=0,
    )

    assert result.reflexflow.enabled is True
    assert result.reflex_clean_pred is None
    assert result.reflex_biased_pred is None
    assert torch.allclose(result.zt, zt)
