from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stereo2spatial.training.loss_terms import _compute_loss_weighted
from stereo2spatial.training.losses_batch import _resolve_effective_sequence_frames
from stereo2spatial.training.losses_windowed import (
    compute_flow_matching_window_loss,
    forward_window,
    prepare_flow_matching_inputs,
    resolve_window_plan,
    resolve_window_weight,
    slice_and_pad_window,
)


def test_resolve_effective_sequence_frames_errors_when_choices_exceed_tmax() -> None:
    with pytest.raises(ValueError, match="sequence_seconds_choices"):
        _resolve_effective_sequence_frames(
            t_max=8,
            force_full_sequence=False,
            randomize_per_batch=False,
            seq_choices_frames=[8, 16],
            max_choice_frames=16,
            global_step=0,
            seed=0,
        )


def test_prepare_flow_matching_inputs_masks_invalid_frames() -> None:
    torch.manual_seed(0)
    z1 = torch.ones(1, 2, 3, 4)
    z_cond = torch.full((1, 1, 3, 4), 2.0)
    valid_mask = torch.tensor([[True, True, False, False]])

    prepared = prepare_flow_matching_inputs(
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t_eff=4,
    )

    assert prepared.batch_size == 1
    assert prepared.t_eff == 4
    assert prepared.z1[..., 2:].abs().sum().item() == 0.0
    assert prepared.z_cond[..., 2:].abs().sum().item() == 0.0
    assert prepared.target_velocity.shape == z1.shape
    assert prepared.t.shape == (1,)


def test_prepare_flow_matching_inputs_custom_timesteps_map_to_flipped_t() -> None:
    torch.manual_seed(0)
    z1 = torch.zeros(1, 1, 1, 2)
    z_cond = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    config = SimpleNamespace(
        flow_timestep_sampling="custom",
        flow_custom_timesteps=[1000.0],
        flow_fast_schedule=False,
        flow_loss_weighting="none",
        flow_schedule_shift=None,
        flow_schedule_auto_shift=False,
    )

    prepared = prepare_flow_matching_inputs(
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t_eff=2,
        training_config=config,
    )

    assert prepared.sigma.item() == pytest.approx(1.0, abs=1e-7)
    assert prepared.t.item() == pytest.approx(0.0, abs=1e-7)


def test_prepare_flow_matching_inputs_applies_sd3_sigma_weighting() -> None:
    torch.manual_seed(0)
    z1 = torch.zeros(1, 1, 1, 2)
    z_cond = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    config = SimpleNamespace(
        flow_timestep_sampling="custom",
        flow_custom_timesteps=[500.0],
        flow_fast_schedule=False,
        flow_loss_weighting="sigma_sqrt",
        flow_schedule_shift=None,
        flow_schedule_auto_shift=False,
    )

    prepared = prepare_flow_matching_inputs(
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t_eff=2,
        training_config=config,
    )

    assert prepared.sigma.item() == pytest.approx(0.5, abs=1e-7)
    assert prepared.loss_weight.item() == pytest.approx(4.0, abs=1e-6)


def test_prepare_flow_matching_inputs_applies_flow_schedule_shift_on_sigma() -> None:
    torch.manual_seed(0)
    z1 = torch.zeros(1, 1, 1, 2)
    z_cond = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    config = SimpleNamespace(
        flow_timestep_sampling="custom",
        flow_custom_timesteps=[500.0],
        flow_fast_schedule=False,
        flow_loss_weighting="none",
        flow_schedule_shift=2.0,
        flow_schedule_auto_shift=False,
    )

    prepared = prepare_flow_matching_inputs(
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t_eff=2,
        training_config=config,
    )

    assert prepared.sigma.item() == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert prepared.t.item() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_resolve_window_plan_uses_cached_metadata() -> None:
    cached_weight = torch.tensor([1.0, 0.5, 0.25])
    starts, weights = resolve_window_plan(
        t_eff=6,
        window_frames=3,
        overlap_frames=1,
        window_metadata={6: ([0, 2, 4], [cached_weight, cached_weight, cached_weight])},
    )
    assert starts == [0, 2, 4]
    assert weights is not None
    assert len(weights) == 3


def test_slice_and_pad_window_right_pads_partial_segment() -> None:
    zt = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 3)
    zc = torch.ones_like(zt)
    tv = torch.full_like(zt, 4.0)
    z1 = torch.full_like(zt, 7.0)
    vm = torch.tensor([[True, True, True]])

    zt_w, zc_w, tv_w, vm_w, z1_w = slice_and_pad_window(
        zt=zt,
        z_cond=zc,
        target_velocity=tv,
        valid_mask=vm,
        start=1,
        end=3,
        window_frames=4,
        batch_size=1,
        z1=z1,
    )

    assert zt_w.shape == (1, 1, 1, 4)
    assert zc_w.shape == (1, 1, 1, 4)
    assert tv_w.shape == (1, 1, 1, 4)
    assert vm_w.shape == (1, 4)
    assert z1_w is not None and z1_w.shape == (1, 1, 1, 4)
    assert torch.allclose(zt_w[..., 0], torch.tensor([[[1.0]]]))
    assert torch.allclose(zt_w[..., 1], torch.tensor([[[2.0]]]))
    assert zt_w[..., 2:].abs().sum().item() == 0.0
    assert vm_w.tolist() == [[True, True, False, False]]


def test_resolve_window_weight_uses_cache_with_cast() -> None:
    cached = [torch.tensor([1.0, 0.5], dtype=torch.float64)]
    weight = resolve_window_weight(
        idx=0,
        num_windows=1,
        window_frames=2,
        overlap_frames=1,
        cached_weights=cached,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert weight.dtype == torch.float32
    assert torch.allclose(weight, torch.tensor([1.0, 0.5], dtype=torch.float32))


def test_compute_flow_matching_window_loss_matches_baseline_when_reflex_disabled() -> None:
    prediction = torch.tensor([[[[0.1, 0.2, 0.3]]]], dtype=torch.float32)
    target_velocity = torch.tensor([[[[0.0, 0.1, 0.2]]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True, False]])
    frame_weight = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)

    baseline = _compute_loss_weighted(
        prediction=prediction,
        target_velocity=target_velocity,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
    )
    reflex_disabled = compute_flow_matching_window_loss(
        prediction=prediction,
        target_velocity=target_velocity,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
        reflex_enabled=False,
    )

    assert torch.allclose(reflex_disabled, baseline, atol=1e-7, rtol=1e-7)


def test_compute_flow_matching_window_loss_applies_sample_weights() -> None:
    prediction = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    target_velocity = torch.ones(2, 1, 1, 1, dtype=torch.float32)
    valid_mask = torch.ones(2, 1, dtype=torch.bool)
    frame_weight = torch.ones(1, dtype=torch.float32)

    unweighted = compute_flow_matching_window_loss(
        prediction=prediction,
        target_velocity=target_velocity,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
        reflex_enabled=False,
    )
    weighted = compute_flow_matching_window_loss(
        prediction=prediction,
        target_velocity=target_velocity,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
        sample_loss_weight=torch.tensor([1.0, 2.0], dtype=torch.float32),
        reflex_enabled=False,
    )

    assert unweighted.item() == pytest.approx(1.0, abs=1e-7)
    assert weighted.item() == pytest.approx(1.5, abs=1e-7)


def test_compute_flow_matching_window_loss_reflex_adr_uses_biased_direction() -> None:
    prediction = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    target_velocity = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True]])
    frame_weight = torch.tensor([1.0], dtype=torch.float32)

    loss_without_biased_vector = compute_flow_matching_window_loss(
        prediction=prediction,
        target_velocity=target_velocity,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
        reflex_enabled=True,
        reflex_alpha=0.0,
        reflex_beta1=1.0,
        reflex_beta2=1.0,
    )
    loss_with_biased_vector = compute_flow_matching_window_loss(
        prediction=prediction,
        target_velocity=target_velocity,
        valid_mask=valid_mask,
        frame_weight=frame_weight,
        reflex_enabled=True,
        reflex_target_vector=torch.tensor([[[[-1.0]]]], dtype=torch.float32),
        reflex_alpha=0.0,
        reflex_beta1=1.0,
        reflex_beta2=1.0,
    )

    assert loss_without_biased_vector.item() == pytest.approx(0.0, abs=1e-7)
    assert loss_with_biased_vector.item() == pytest.approx(4.0, abs=1e-7)


class _DummyMemModel(torch.nn.Module):
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
        pred = zt + z_cond + t[:, None, None, None]
        if return_mem:
            if mem is None:
                raise RuntimeError("mem required when return_mem=True")
            return pred, mem + 1.0
        return pred


def test_forward_window_returns_detached_memory_when_requested() -> None:
    model = _DummyMemModel()
    zt = torch.zeros(2, 1, 1, 3)
    zc = torch.ones(2, 1, 1, 3)
    vm = torch.ones(2, 3, dtype=torch.bool)
    t = torch.zeros(2)
    mem = torch.randn(2, 1, 4, requires_grad=True)

    pred, mem_out = forward_window(
        model=model,
        zt_w=zt,
        zc_w=zc,
        vm_w=vm,
        t=t,
        mem=mem,
        detach_memory=True,
    )
    assert pred.shape == zt.shape
    assert mem_out is not None
    assert mem_out.requires_grad is False


class _RecurrentBridgeModel(torch.nn.Module):
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
            raise RuntimeError("mem required for recurrent bridge test model")
        pred = zt + mem[:, :, :, None].to(dtype=zt.dtype, device=zt.device).expand_as(zt)
        next_mem = mem + zt.mean(dim=(1, 2, 3), keepdim=True).squeeze(-1)
        if return_mem:
            return pred, next_mem
        return pred


def test_forward_window_detach_memory_controls_cross_window_gradients() -> None:
    model = _RecurrentBridgeModel()
    zc = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    vm = torch.ones(1, 2, dtype=torch.bool)
    t = torch.tensor([0.5], dtype=torch.float32)

    # detach_memory=False should allow later-window loss to backprop through memory
    # into earlier-window states.
    zt_w1 = torch.full((1, 1, 1, 2), 2.0, dtype=torch.float32, requires_grad=True)
    zt_w2 = torch.zeros((1, 1, 1, 2), dtype=torch.float32, requires_grad=True)
    mem0 = torch.zeros((1, 1, 1), dtype=torch.float32, requires_grad=True)
    _, mem1 = forward_window(
        model=model,
        zt_w=zt_w1,
        zc_w=zc,
        vm_w=vm,
        t=t,
        mem=mem0,
        detach_memory=False,
    )
    pred2, _ = forward_window(
        model=model,
        zt_w=zt_w2,
        zc_w=zc,
        vm_w=vm,
        t=t,
        mem=mem1,
        detach_memory=False,
    )
    loss = pred2.sum()
    loss.backward()
    assert zt_w2.grad is not None
    assert float(zt_w2.grad.abs().sum().item()) > 0.0
    assert zt_w1.grad is not None
    assert float(zt_w1.grad.abs().sum().item()) > 0.0

    # detach_memory=True should cut gradient from later windows into earlier windows.
    zt_w1_detached = torch.full(
        (1, 1, 1, 2), 2.0, dtype=torch.float32, requires_grad=True
    )
    zt_w2_detached = torch.zeros((1, 1, 1, 2), dtype=torch.float32, requires_grad=True)
    mem0_detached = torch.zeros((1, 1, 1), dtype=torch.float32, requires_grad=True)
    _, mem1_detached = forward_window(
        model=model,
        zt_w=zt_w1_detached,
        zc_w=zc,
        vm_w=vm,
        t=t,
        mem=mem0_detached,
        detach_memory=True,
    )
    pred2_detached, _ = forward_window(
        model=model,
        zt_w=zt_w2_detached,
        zc_w=zc,
        vm_w=vm,
        t=t,
        mem=mem1_detached,
        detach_memory=True,
    )
    loss_detached = pred2_detached.sum()
    loss_detached.backward()
    assert zt_w2_detached.grad is not None
    assert float(zt_w2_detached.grad.abs().sum().item()) > 0.0
    if zt_w1_detached.grad is None:
        pass
    else:
        assert float(zt_w1_detached.grad.abs().sum().item()) == pytest.approx(0.0)
