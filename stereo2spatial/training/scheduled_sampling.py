"""Flow-matching scheduled-sampling rollout helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch

from stereo2spatial.common.windowing import (
    chunk_weight as _common_chunk_weight,
)
from stereo2spatial.common.windowing import (
    segment_starts as _common_segment_starts,
)

_ALLOWED_ROLLOUT_STRATEGIES = {"uniform", "biased_early", "biased_late"}
_ALLOWED_RAMP_SHAPES = {"linear", "cosine"}
_ALLOWED_SAMPLERS = {"euler", "heun", "unipc"}


def _iter_wrapped_modules(model: torch.nn.Module) -> tuple[object, ...]:
    """Traverse common wrapper chains (DDP/compile) to inspect raw model attrs."""
    to_visit: list[object] = [model]
    visited: set[int] = set()
    ordered: list[object] = []
    while to_visit:
        current = to_visit.pop(0)
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)
        ordered.append(current)
        for attr_name in ("module", "_orig_mod"):
            wrapped = getattr(current, attr_name, None)
            if wrapped is not None:
                to_visit.append(wrapped)
    return tuple(ordered)


def _clamp_probability(value: float) -> float:
    """Clamp probability-like values into [0, 1]."""
    return float(min(1.0, max(0.0, value)))


def _resolve_rollout_num_steps_from_model(model: torch.nn.Module) -> int:
    """
    Resolve rollout discretization from ``model.timestep_scale``.

    Falls back to 1000 when no usable scale is available.
    """
    for current in _iter_wrapped_modules(model):
        scale_value = getattr(current, "timestep_scale", None)
        if scale_value is not None:
            try:
                steps = int(round(float(scale_value)))
                return max(2, steps)
            except Exception:
                pass

    return 1000


def _resolve_model_memory_tokens(model: torch.nn.Module) -> int:
    """Resolve memory-token count from wrapped models when available."""
    for current in _iter_wrapped_modules(model):
        try:
            count = int(getattr(current, "num_memory_tokens", 0) or 0)
        except Exception:
            continue
        if count > 0:
            return count
    return 0


def _init_rollout_memory(
    *,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Initialize recurrent memory from the first wrapped model that supports it."""
    for current in _iter_wrapped_modules(model):
        try:
            count = int(getattr(current, "num_memory_tokens", 0) or 0)
        except Exception:
            continue
        if count <= 0:
            continue
        init_fn = getattr(current, "init_memory", None)
        if callable(init_fn):
            mem = init_fn(batch_size=batch_size, device=device, dtype=dtype)
            if mem is None:
                raise RuntimeError("Model memory is enabled but init_memory returned None.")
            return cast(torch.Tensor, mem)
    return None


def _predict_velocity(
    *,
    model: torch.nn.Module,
    zt: torch.Tensor,
    t: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    mem: torch.Tensor | None,
    use_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one model call with optional recurrent-memory state threading."""
    if not use_memory:
        return (
            cast(
                torch.Tensor,
                model(
                    zt=zt,
                    t=t,
                    z_cond=z_cond,
                    valid_mask=valid_mask,
                ),
            ),
            mem,
        )

    prediction = model(
        zt=zt,
        t=t,
        z_cond=z_cond,
        valid_mask=valid_mask,
        mem=mem,
        return_mem=True,
    )
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise RuntimeError(
            "Model with memory tokens must return (prediction, memory) when return_mem=True."
        )
    velocity, next_mem = prediction
    return cast(torch.Tensor, velocity), cast(torch.Tensor | None, next_mem)


def _predict_velocity_probe(
    *,
    model: torch.nn.Module,
    zt: torch.Tensor,
    t: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    mem: torch.Tensor | None,
    use_memory: bool,
) -> torch.Tensor:
    """
    Run one model call without mutating recurrent memory state.

    Used for provisional predictor states in multi-eval solvers
    (for example, Heun/UniPC) where only accepted states should
    advance memory.
    """
    if not use_memory:
        return cast(
            torch.Tensor,
            model(
                zt=zt,
                t=t,
                z_cond=z_cond,
                valid_mask=valid_mask,
            ),
        )
    prediction = model(
        zt=zt,
        t=t,
        z_cond=z_cond,
        valid_mask=valid_mask,
        mem=mem,
        return_mem=True,
    )
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise RuntimeError(
            "Model with memory tokens must return (prediction, memory) when return_mem=True."
        )
    velocity, _ = prediction
    return cast(torch.Tensor, velocity)


def _resolve_window_rollout_plan(
    *,
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
) -> tuple[list[int], int, int]:
    """Resolve deterministic window starts for rollout-overlap processing."""
    resolved_window_frames = max(1, int(window_frames))
    resolved_overlap_frames = min(
        max(0, int(overlap_frames)),
        max(0, resolved_window_frames - 1),
    )
    stride_frames = max(1, resolved_window_frames - resolved_overlap_frames)
    starts = _common_segment_starts(
        total_frames=total_frames,
        window_frames=resolved_window_frames,
        stride_frames=stride_frames,
    )
    return starts, resolved_window_frames, resolved_overlap_frames


def _slice_and_pad_tensor4d(
    *,
    tensor: torch.Tensor,
    start: int,
    end: int,
    window_frames: int,
) -> torch.Tensor:
    """Slice one ``[B,C,D,T]`` window and right-pad to ``window_frames``."""
    segment_len = int(end - start)
    window = tensor[..., start:end]
    if segment_len >= window_frames:
        return window
    pad_t = window_frames - segment_len
    return torch.cat([window, torch.zeros_like(window[..., :pad_t])], dim=-1)


def _slice_and_pad_mask2d(
    *,
    mask: torch.Tensor,
    start: int,
    end: int,
    window_frames: int,
) -> torch.Tensor:
    """Slice one ``[B,T]`` mask window and right-pad with ``False``."""
    segment_len = int(end - start)
    window = mask[:, start:end]
    if segment_len >= window_frames:
        return window
    pad_t = window_frames - segment_len
    pad = torch.zeros(
        (mask.shape[0], pad_t),
        device=mask.device,
        dtype=mask.dtype,
    )
    return torch.cat([window, pad], dim=1)


def _predict_velocity_windowed(
    *,
    model: torch.nn.Module,
    zt: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    t_value: float,
    t_dtype: torch.dtype,
    mem: torch.Tensor | None,
    use_memory: bool,
    window_frames: int,
    overlap_frames: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Evaluate model velocity over a full sequence via overlap-add windows.

    Memory semantics match normal chunk processing: one committed memory update
    per window via ``return_mem=True`` when memory tokens are enabled.
    """
    total_frames = int(zt.shape[-1])
    starts, resolved_window_frames, resolved_overlap_frames = _resolve_window_rollout_plan(
        total_frames=total_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
    )
    pred_sum = torch.zeros_like(zt)
    weight_sum = torch.zeros((total_frames,), device=zt.device, dtype=torch.float32)
    mem_state = mem
    t_batch = torch.tensor([float(t_value)], device=zt.device, dtype=t_dtype)

    for window_idx, start in enumerate(starts):
        end = min(start + resolved_window_frames, total_frames)
        segment_len = int(end - start)
        zt_w = _slice_and_pad_tensor4d(
            tensor=zt,
            start=start,
            end=end,
            window_frames=resolved_window_frames,
        )
        zc_w = _slice_and_pad_tensor4d(
            tensor=z_cond,
            start=start,
            end=end,
            window_frames=resolved_window_frames,
        )
        vm_w = _slice_and_pad_mask2d(
            mask=valid_mask,
            start=start,
            end=end,
            window_frames=resolved_window_frames,
        )
        pred_w, mem_state = _predict_velocity(
            model=model,
            zt=zt_w,
            t=t_batch,
            z_cond=zc_w,
            valid_mask=vm_w,
            mem=mem_state,
            use_memory=use_memory,
        )

        weight = _common_chunk_weight(
            chunk_length=segment_len,
            overlap_frames=resolved_overlap_frames,
            is_first=(window_idx == 0),
            is_last=(window_idx == len(starts) - 1),
            device=zt.device,
            dtype=torch.float32,
        )
        weight_f = weight.to(dtype=pred_w.dtype)
        pred_sum[..., start:end] = (
            pred_sum[..., start:end]
            + pred_w[..., :segment_len] * weight_f[None, None, None, :]
        )
        weight_sum[start:end] = weight_sum[start:end] + weight

    denom = torch.clamp(weight_sum, min=1e-8).to(dtype=pred_sum.dtype)
    prediction = pred_sum / denom[None, None, None, :]
    return prediction, mem_state


def _rollout_to_target_with_fixed_memory(
    *,
    model: torch.nn.Module,
    current: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    source_t: float,
    target_t: float,
    rollout_steps: int,
    sampler: str,
    t_dtype: torch.dtype,
    mem: torch.Tensor | None,
    use_memory: bool,
) -> torch.Tensor:
    """
    Integrate one rollout interval while keeping recurrent memory fixed.

    Memory state is intentionally not advanced during provisional or accepted
    integration substeps. Callers should commit memory once at interval end.
    """
    if rollout_steps <= 0 or source_t >= target_t:
        return current

    base_dt = (target_t - source_t) / float(rollout_steps)
    current_t = source_t
    previous_velocity: torch.Tensor | None = None
    for rollout_idx in range(rollout_steps):
        next_t = (
            target_t if rollout_idx == rollout_steps - 1 else min(target_t, current_t + base_dt)
        )
        dt = float(next_t - current_t)
        if dt <= 0.0:
            break

        t_curr = torch.tensor([current_t], device=current.device, dtype=t_dtype)
        velocity_curr = _predict_velocity_probe(
            model=model,
            zt=current,
            t=t_curr,
            z_cond=z_cond,
            valid_mask=valid_mask,
            mem=mem,
            use_memory=use_memory,
        )
        if sampler == "heun":
            euler_state = current + dt * velocity_curr
            t_next = torch.tensor([next_t], device=current.device, dtype=t_dtype)
            velocity_next = _predict_velocity_probe(
                model=model,
                zt=euler_state,
                t=t_next,
                z_cond=z_cond,
                valid_mask=valid_mask,
                mem=mem,
                use_memory=use_memory,
            )
            current = current + 0.5 * dt * (velocity_curr + velocity_next)
        elif sampler == "unipc":
            if previous_velocity is None:
                predictor_state = current + dt * velocity_curr
            else:
                predictor_state = current + dt * (
                    1.5 * velocity_curr - 0.5 * previous_velocity
                )
            t_next = torch.tensor([next_t], device=current.device, dtype=t_dtype)
            velocity_next = _predict_velocity_probe(
                model=model,
                zt=predictor_state,
                t=t_next,
                z_cond=z_cond,
                valid_mask=valid_mask,
                mem=mem,
                use_memory=use_memory,
            )
            if previous_velocity is None:
                current = current + 0.5 * dt * (velocity_curr + velocity_next)
            else:
                current = current + dt * (
                    (5.0 / 12.0) * velocity_next
                    + (2.0 / 3.0) * velocity_curr
                    - (1.0 / 12.0) * previous_velocity
                )
            previous_velocity = velocity_curr
        else:
            current = current + dt * velocity_curr
        current_t = next_t

    return current


@dataclass(frozen=True)
class ScheduledSamplingPlan:
    """Per-sample rollout plan in discretized flow-time steps."""

    target_steps: torch.Tensor
    source_steps: torch.Tensor
    rollout_steps: torch.Tensor


@dataclass(frozen=True)
class ReflexFlowOptions:
    """Resolved ReflexFlow toggles and coefficient values."""

    enabled: bool
    alpha: float
    beta1: float
    beta2: float


@dataclass(frozen=True)
class ScheduledSamplingRolloutResult:
    """Scheduled-sampling rollout output plus optional ReflexFlow caches."""

    t: torch.Tensor
    zt: torch.Tensor
    reflexflow: ReflexFlowOptions
    reflex_clean_pred: torch.Tensor | None = None
    reflex_biased_pred: torch.Tensor | None = None


def resolve_reflexflow_options(
    *,
    training_config: Any,
    max_step_offset: int,
) -> ReflexFlowOptions:
    """
    Resolve ReflexFlow options.

    ReflexFlow is auto-enabled when scheduled sampling is active and the user
    did not explicitly provide ``scheduled_sampling_reflexflow``.
    """
    explicit_flag = getattr(training_config, "scheduled_sampling_reflexflow", None)
    enabled = bool(explicit_flag) if explicit_flag is not None else int(max_step_offset) > 0

    alpha = float(
        getattr(training_config, "scheduled_sampling_reflexflow_alpha", 1.0) or 0.0
    )
    beta1 = float(
        getattr(training_config, "scheduled_sampling_reflexflow_beta1", 10.0) or 0.0
    )
    beta2 = float(
        getattr(training_config, "scheduled_sampling_reflexflow_beta2", 1.0) or 0.0
    )
    return ReflexFlowOptions(
        enabled=enabled,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
    )


def resolve_scheduled_sampling_probability(
    *,
    max_step_offset: int,
    probability: float,
    prob_start: float | None,
    prob_end: float | None,
    ramp_steps: int,
    ramp_start_step: int,
    ramp_shape: str,
    global_step: int,
) -> float:
    """Resolve effective rollout probability at the current global step."""
    if int(max_step_offset) <= 0:
        return 0.0

    prob = _clamp_probability(float(probability))
    start_prob = prob if prob_start is None else _clamp_probability(float(prob_start))
    end_prob = prob if prob_end is None else _clamp_probability(float(prob_end))
    ramp_shape = str(ramp_shape).strip().lower()
    if ramp_shape not in _ALLOWED_RAMP_SHAPES:
        ramp_shape = "linear"

    ramp_steps = max(0, int(ramp_steps))
    ramp_start_step = max(0, int(ramp_start_step))
    if ramp_steps <= 0:
        return prob

    if int(global_step) < ramp_start_step:
        return start_prob

    progress = min(
        1.0,
        max(0.0, (float(global_step) - float(ramp_start_step)) / float(ramp_steps)),
    )
    if ramp_shape == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return _clamp_probability(start_prob + (end_prob - start_prob) * progress)


def build_rollout_plan(
    *,
    base_steps: torch.Tensor,
    max_step_offset: int,
    apply_probability: float,
    strategy: str,
    num_time_steps: int,
) -> ScheduledSamplingPlan:
    """Build per-sample source/target rollout steps for flow-matching rollout."""
    if base_steps.dim() != 1:
        raise ValueError(f"base_steps must be rank-1 [B], got {tuple(base_steps.shape)}")
    if base_steps.dtype not in {torch.int16, torch.int32, torch.int64, torch.uint8}:
        base_steps = base_steps.long()
    if int(num_time_steps) <= 1:
        raise ValueError("num_time_steps must be >= 2 for scheduled sampling.")

    base_steps = torch.clamp(base_steps.long(), min=0, max=int(num_time_steps) - 1)
    max_step_offset = max(0, int(max_step_offset))
    if max_step_offset == 0:
        zeros = torch.zeros_like(base_steps)
        return ScheduledSamplingPlan(
            target_steps=base_steps,
            source_steps=base_steps,
            rollout_steps=zeros,
        )

    batch_size = int(base_steps.shape[0])
    device = base_steps.device
    prob = _clamp_probability(float(apply_probability))
    if prob < 1.0:
        rollout_mask = torch.bernoulli(
            torch.full((batch_size,), prob, device=device, dtype=torch.float32)
        ).to(dtype=torch.bool)
    else:
        rollout_mask = torch.ones((batch_size,), device=device, dtype=torch.bool)

    strategy = str(strategy).strip().lower()
    if strategy not in _ALLOWED_ROLLOUT_STRATEGIES:
        raise ValueError(
            "scheduled_sampling_strategy must be one of: "
            "uniform, biased_early, biased_late"
        )

    if strategy == "uniform":
        sampled_offsets = torch.randint(
            0, max_step_offset + 1, (batch_size,), device=device
        )
    elif strategy == "biased_early":
        sampled_offsets = torch.round(
            torch.rand((batch_size,), device=device).pow(2) * max_step_offset
        ).long()
    else:  # biased_late
        sampled_offsets = torch.round(
            (1.0 - torch.rand((batch_size,), device=device).pow(2)) * max_step_offset
        ).long()

    sampled_offsets = sampled_offsets * rollout_mask
    source_steps = torch.clamp(base_steps - sampled_offsets, min=0)
    rollout_steps = base_steps - source_steps
    return ScheduledSamplingPlan(
        target_steps=base_steps,
        source_steps=source_steps,
        rollout_steps=rollout_steps,
    )


@torch.no_grad()
def apply_flow_matching_scheduled_sampling(
    *,
    model: torch.nn.Module,
    z1: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    t: torch.Tensor,
    zt: torch.Tensor,
    z0: torch.Tensor,
    training_config: Any,
    global_step: int,
    plan: ScheduledSamplingPlan | None = None,
    window_frames: int | None = None,
    overlap_frames: int = 0,
) -> ScheduledSamplingRolloutResult:
    """
    Roll out model-generated latents from a noisier source time to target time.

    Rollout endpoints follow the discretized ``target_steps`` in the rollout plan.
    ``t`` is updated only for samples that actually roll out.

    When ``window_frames`` is provided, rollout is performed in overlap-add
    windows and recurrent memory is committed once per window at interval end.
    """
    if t.dim() != 1:
        raise ValueError(f"t must be [B], got {tuple(t.shape)}")
    if z1.shape != z0.shape or z1.shape != zt.shape:
        raise ValueError(
            "z1, z0, and zt must have identical shape [B,C,D,T]. "
            f"Got z1={tuple(z1.shape)} z0={tuple(z0.shape)} zt={tuple(zt.shape)}"
        )
    if z_cond.dim() != 4:
        raise ValueError(f"z_cond must be [B,C,D,T], got {tuple(z_cond.shape)}")
    if valid_mask.dim() != 2:
        raise ValueError(f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}")

    batch_size = int(t.shape[0])
    if (
        z1.shape[0] != batch_size
        or z_cond.shape[0] != batch_size
        or valid_mask.shape[0] != batch_size
    ):
        raise ValueError("Batch dimension mismatch across scheduled-sampling inputs.")

    max_step_offset = int(
        getattr(training_config, "scheduled_sampling_max_step_offset", 0) or 0
    )
    reflexflow = resolve_reflexflow_options(
        training_config=training_config,
        max_step_offset=max_step_offset,
    )
    # Prediction cache collection only makes sense when rollout is active.
    reflex_cache_enabled = bool(reflexflow.enabled and max_step_offset > 0)
    if max_step_offset <= 0 and plan is None:
        return ScheduledSamplingRolloutResult(
            t=t,
            zt=zt,
            reflexflow=reflexflow,
        )

    num_time_steps = _resolve_rollout_num_steps_from_model(model)
    time_step_denom = float(num_time_steps - 1)

    sampler = str(getattr(training_config, "scheduled_sampling_sampler", "heun")).strip().lower()
    if sampler not in _ALLOWED_SAMPLERS:
        raise ValueError("scheduled_sampling_sampler must be one of: euler, heun, unipc")

    if plan is None:
        effective_probability = resolve_scheduled_sampling_probability(
            max_step_offset=max_step_offset,
            probability=float(
                getattr(training_config, "scheduled_sampling_probability", 0.0) or 0.0
            ),
            prob_start=getattr(training_config, "scheduled_sampling_prob_start", None),
            prob_end=getattr(training_config, "scheduled_sampling_prob_end", None),
            ramp_steps=int(getattr(training_config, "scheduled_sampling_ramp_steps", 0) or 0),
            ramp_start_step=int(getattr(training_config, "scheduled_sampling_start_step", 0) or 0),
            ramp_shape=str(getattr(training_config, "scheduled_sampling_ramp_shape", "linear") or "linear"),
            global_step=int(global_step),
        )
        if effective_probability <= 0.0:
            return ScheduledSamplingRolloutResult(
                t=t,
                zt=zt,
                reflexflow=reflexflow,
            )

        base_steps = torch.clamp(
            torch.round(t.float() * time_step_denom).long(),
            min=0,
            max=num_time_steps - 1,
        )
        plan = build_rollout_plan(
            base_steps=base_steps,
            max_step_offset=max_step_offset,
            apply_probability=effective_probability,
            strategy=str(getattr(training_config, "scheduled_sampling_strategy", "uniform") or "uniform"),
            num_time_steps=num_time_steps,
        )

    if (
        int(plan.target_steps.shape[0]) != batch_size
        or int(plan.source_steps.shape[0]) != batch_size
        or int(plan.rollout_steps.shape[0]) != batch_size
    ):
        raise ValueError(
            "scheduled-sampling plan batch size mismatch: "
            f"expected {batch_size}, got target={int(plan.target_steps.shape[0])}, "
            f"source={int(plan.source_steps.shape[0])}, "
            f"rollout={int(plan.rollout_steps.shape[0])}"
        )

    if not bool(torch.any(plan.rollout_steps > 0).item()):
        return ScheduledSamplingRolloutResult(
            t=t,
            zt=zt,
            reflexflow=reflexflow,
        )

    device = zt.device
    dtype = zt.dtype
    new_zt = zt.clone()
    new_t = t.clone()
    use_memory = _resolve_model_memory_tokens(model) > 0
    rollout_window_frames = int(window_frames) if window_frames is not None else None
    use_windowed_rollout = rollout_window_frames is not None and rollout_window_frames > 0
    clean_preds = torch.zeros_like(zt) if reflex_cache_enabled else None
    biased_preds = torch.zeros_like(zt) if reflex_cache_enabled else None

    for idx in range(batch_size):
        rollout_steps = int(plan.rollout_steps[idx].item())

        target_t_original = float(t[idx].item())
        target_step = int(torch.clamp(plan.target_steps[idx], min=0, max=num_time_steps - 1).item())
        target_t = float(target_step) / time_step_denom
        source_step = int(torch.clamp(plan.source_steps[idx], min=0, max=num_time_steps - 1).item())
        source_t = float(source_step) / time_step_denom
        cond_i = z_cond[idx : idx + 1]
        mask_i = valid_mask[idx : idx + 1]
        clean_t_batch = torch.tensor([target_t_original], device=device, dtype=t.dtype)
        target_t_batch = torch.tensor([target_t], device=device, dtype=t.dtype)
        mem_i: torch.Tensor | None = (
            _init_rollout_memory(
                model=model,
                batch_size=1,
                device=device,
                dtype=dtype,
            )
            if use_memory
            else None
        )

        if reflex_cache_enabled and clean_preds is not None:
            if use_windowed_rollout:
                if rollout_window_frames is None:
                    raise RuntimeError("window_frames must be set for windowed rollout.")
                clean_mem: torch.Tensor | None = (
                    _init_rollout_memory(
                        model=model,
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                    if use_memory
                    else None
                )
                clean_pred, _ = _predict_velocity_windowed(
                    model=model,
                    zt=zt[idx : idx + 1],
                    z_cond=cond_i,
                    valid_mask=mask_i,
                    t_value=target_t_original,
                    t_dtype=t.dtype,
                    mem=clean_mem,
                    use_memory=use_memory,
                    window_frames=rollout_window_frames,
                    overlap_frames=int(overlap_frames),
                )
            else:
                clean_pred, _ = _predict_velocity(
                    model=model,
                    zt=zt[idx : idx + 1],
                    t=clean_t_batch,
                    z_cond=cond_i,
                    valid_mask=mask_i,
                    mem=mem_i,
                    use_memory=use_memory,
                )
            clean_preds[idx : idx + 1] = clean_pred.to(device=device, dtype=dtype)

        if rollout_steps <= 0 or source_t >= target_t:
            if (
                reflex_cache_enabled
                and clean_preds is not None
                and biased_preds is not None
            ):
                biased_preds[idx : idx + 1] = clean_preds[idx : idx + 1]
            continue

        if use_windowed_rollout:
            if rollout_window_frames is None:
                raise RuntimeError("window_frames must be set for windowed rollout.")
            z0_i = z0[idx : idx + 1]
            z1_i = z1[idx : idx + 1]
            total_frames = int(z0_i.shape[-1])
            (
                starts,
                resolved_window_frames,
                resolved_overlap_frames,
            ) = _resolve_window_rollout_plan(
                total_frames=total_frames,
                window_frames=rollout_window_frames,
                overlap_frames=int(overlap_frames),
            )
            rolled_sum = torch.zeros_like(z0_i)
            weight_sum = torch.zeros((total_frames,), device=device, dtype=torch.float32)
            biased_sum = (
                torch.zeros_like(z0_i)
                if reflex_cache_enabled and biased_preds is not None
                else None
            )
            mem_roll = mem_i

            for window_idx, start in enumerate(starts):
                end = min(start + resolved_window_frames, total_frames)
                segment_len = int(end - start)
                z0_w = _slice_and_pad_tensor4d(
                    tensor=z0_i,
                    start=start,
                    end=end,
                    window_frames=resolved_window_frames,
                )
                z1_w = _slice_and_pad_tensor4d(
                    tensor=z1_i,
                    start=start,
                    end=end,
                    window_frames=resolved_window_frames,
                )
                zc_w = _slice_and_pad_tensor4d(
                    tensor=cond_i,
                    start=start,
                    end=end,
                    window_frames=resolved_window_frames,
                )
                vm_w = _slice_and_pad_mask2d(
                    mask=mask_i,
                    start=start,
                    end=end,
                    window_frames=resolved_window_frames,
                )
                current_w = (1.0 - source_t) * z0_w + source_t * z1_w
                current_w = _rollout_to_target_with_fixed_memory(
                    model=model,
                    current=current_w,
                    z_cond=zc_w,
                    valid_mask=vm_w,
                    source_t=source_t,
                    target_t=target_t,
                    rollout_steps=rollout_steps,
                    sampler=sampler,
                    t_dtype=t.dtype,
                    mem=mem_roll,
                    use_memory=use_memory,
                )

                if use_memory:
                    biased_pred_w, mem_roll = _predict_velocity(
                        model=model,
                        zt=current_w,
                        t=target_t_batch,
                        z_cond=zc_w,
                        valid_mask=vm_w,
                        mem=mem_roll,
                        use_memory=True,
                    )
                else:
                    biased_pred_w, _ = _predict_velocity(
                        model=model,
                        zt=current_w,
                        t=target_t_batch,
                        z_cond=zc_w,
                        valid_mask=vm_w,
                        mem=None,
                        use_memory=False,
                    )

                weight = _common_chunk_weight(
                    chunk_length=segment_len,
                    overlap_frames=resolved_overlap_frames,
                    is_first=(window_idx == 0),
                    is_last=(window_idx == len(starts) - 1),
                    device=device,
                    dtype=torch.float32,
                )
                weight_f = weight.to(dtype=current_w.dtype)
                rolled_sum[..., start:end] = (
                    rolled_sum[..., start:end]
                    + current_w[..., :segment_len] * weight_f[None, None, None, :]
                )
                weight_sum[start:end] = weight_sum[start:end] + weight

                if biased_sum is not None:
                    biased_sum[..., start:end] = (
                        biased_sum[..., start:end]
                        + biased_pred_w[..., :segment_len] * weight_f[None, None, None, :]
                    )

            weight_denom = torch.clamp(weight_sum, min=1e-8).to(dtype=rolled_sum.dtype)
            rolled = rolled_sum / weight_denom[None, None, None, :]
            new_zt[idx : idx + 1] = rolled.to(device=device, dtype=dtype)
            new_t[idx] = target_t_batch[0]
            if biased_sum is not None and biased_preds is not None:
                biased_preds[idx : idx + 1] = (
                    biased_sum / weight_denom[None, None, None, :]
                ).to(device=device, dtype=dtype)
            continue

        current = (1.0 - source_t) * z0[idx : idx + 1] + source_t * z1[idx : idx + 1]

        base_dt = (target_t - source_t) / float(rollout_steps)
        current_t = source_t
        previous_velocity: torch.Tensor | None = None
        for rollout_idx in range(rollout_steps):
            next_t = (
                target_t
                if rollout_idx == rollout_steps - 1
                else min(target_t, current_t + base_dt)
            )
            dt = float(next_t - current_t)
            if dt <= 0.0:
                break

            t_curr = torch.tensor([current_t], device=device, dtype=t.dtype)
            velocity_curr, mem_i = _predict_velocity(
                model=model,
                zt=current,
                t=t_curr,
                z_cond=cond_i,
                valid_mask=mask_i,
                mem=mem_i,
                use_memory=use_memory,
            )
            if sampler == "heun":
                euler_state = current + dt * velocity_curr
                t_next = torch.tensor([next_t], device=device, dtype=t.dtype)
                velocity_next = _predict_velocity_probe(
                    model=model,
                    zt=euler_state,
                    t=t_next,
                    z_cond=cond_i,
                    valid_mask=mask_i,
                    mem=mem_i,
                    use_memory=use_memory,
                )
                current = current + 0.5 * dt * (velocity_curr + velocity_next)
            elif sampler == "unipc":
                if previous_velocity is None:
                    predictor_state = current + dt * velocity_curr
                else:
                    predictor_state = current + dt * (
                        1.5 * velocity_curr - 0.5 * previous_velocity
                    )
                t_next = torch.tensor([next_t], device=device, dtype=t.dtype)
                velocity_next = _predict_velocity_probe(
                    model=model,
                    zt=predictor_state,
                    t=t_next,
                    z_cond=cond_i,
                    valid_mask=mask_i,
                    mem=mem_i,
                    use_memory=use_memory,
                )
                if previous_velocity is None:
                    current = current + 0.5 * dt * (velocity_curr + velocity_next)
                else:
                    current = current + dt * (
                        (5.0 / 12.0) * velocity_next
                        + (2.0 / 3.0) * velocity_curr
                        - (1.0 / 12.0) * previous_velocity
                    )
                previous_velocity = velocity_curr
            else:
                current = current + dt * velocity_curr
            current_t = next_t

        if reflex_cache_enabled and biased_preds is not None:
            biased_pred, _ = _predict_velocity(
                model=model,
                zt=current,
                t=target_t_batch,
                z_cond=cond_i,
                valid_mask=mask_i,
                mem=mem_i,
                use_memory=use_memory,
            )
            biased_preds[idx : idx + 1] = biased_pred.to(device=device, dtype=dtype)

        new_zt[idx : idx + 1] = current.to(device=device, dtype=dtype)
        new_t[idx] = target_t_batch[0]

    return ScheduledSamplingRolloutResult(
        t=new_t,
        zt=new_zt,
        reflexflow=reflexflow,
        reflex_clean_pred=clean_preds.detach() if clean_preds is not None else None,
        reflex_biased_pred=biased_preds.detach() if biased_preds is not None else None,
    )


__all__ = [
    "ReflexFlowOptions",
    "ScheduledSamplingPlan",
    "ScheduledSamplingRolloutResult",
    "apply_flow_matching_scheduled_sampling",
    "build_rollout_plan",
    "resolve_reflexflow_options",
    "resolve_scheduled_sampling_probability",
]
