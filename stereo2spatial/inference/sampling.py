"""Waveform-patch sampling utilities used by the inference runner."""

from __future__ import annotations

import math
from typing import cast

import torch
from torchdiffeq import odeint

from stereo2spatial.common.windowing import (
    chunk_weight as _common_chunk_weight,
)
from stereo2spatial.common.windowing import (
    segment_starts as _common_segment_starts,
)
from stereo2spatial.modeling import SpatialDiT

SolverName = str
_CLEAN_PREDICTION_EPS = 1e-4
_INTEGRATION_T_END = 1.0 - _CLEAN_PREDICTION_EPS


def _resolve_time_grid(
    method: str,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ODE integration time points for fixed-step and adaptive solvers."""
    method = method.lower()
    fixed_step_methods = {
        "heun",
        "euler",
        "unipc",
        "midpoint",
        "rk4",
        "explicit_adams",
        "implicit_adams",
    }
    if method in fixed_step_methods:
        return torch.linspace(
            0.0, _INTEGRATION_T_END, num_steps + 1, device=device, dtype=dtype
        )
    return torch.tensor([0.0, _INTEGRATION_T_END], device=device, dtype=dtype)


def _segment_starts(
    total_frames: int,
    window_frames: int,
    stride_frames: int,
) -> list[int]:
    """Compatibility wrapper for shared segment start computation."""
    return _common_segment_starts(
        total_frames=total_frames,
        window_frames=window_frames,
        stride_frames=stride_frames,
    )


def _chunk_weight(
    chunk_length: int,
    overlap_frames: int,
    is_first: bool,
    is_last: bool,
    device: torch.device,
) -> torch.Tensor:
    """Compatibility wrapper for shared overlap-add weighting."""
    return _common_chunk_weight(
        chunk_length=chunk_length,
        overlap_frames=overlap_frames,
        is_first=is_first,
        is_last=is_last,
        device=device,
        dtype=torch.float32,
    )


def _clean_prediction_to_velocity(
    clean_prediction: torch.Tensor,
    z_state: torch.Tensor,
    t_value: float,
) -> torch.Tensor:
    """Convert a clean endpoint prediction into the rectified-flow velocity."""
    denom = max(1.0 - float(t_value), _CLEAN_PREDICTION_EPS)
    return (clean_prediction - z_state) / denom


def _phi_series(j: int, z: float, terms: int = 24) -> float:
    """Evaluate phi_j(z) by its Taylor series for small |z|."""
    total = 0.0
    z_power = 1.0
    for term_idx in range(terms):
        total += z_power / math.factorial(term_idx + j)
        z_power *= z
    return total


def _phi(j: int, z: float) -> float:
    """Evaluate exponential-integrator phi_j(z)."""
    if j <= 0:
        raise ValueError("j must be positive")
    if abs(z) < 1e-4:
        return _phi_series(j, z)
    remainder = sum((z**k) / math.factorial(k) for k in range(j))
    return (math.exp(z) - remainder) / (z**j)


def _res6s_tableau(step_size: float) -> tuple[list[float], list[list[float]], list[float]]:
    """Return the RES4LYF res_6s exponential RK tableau for one time step.

    Adapted from RES4LYF's ``beta/rk_coefficients_beta.py`` ``res_6s`` case.
    The first coefficient column is generated so each stage row sums to
    ``c_i * phi_1(-c_i h)`` and the final weights sum to ``phi_1(-h)``.
    """
    h = float(step_size)
    c1, c2, c3, c4, c5, c6 = 0.0, 0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0, 5.0 / 6.0
    c = [c1, c2, c3, c4, c5, c6]

    def phi_at(j: int, stage_index: int | None = None) -> float:
        if stage_index is None:
            stage_c = 1.0
        else:
            stage_c = c[stage_index]
            if stage_c == 0.0:
                return 0.0
        return _phi(j, -h * stage_c)

    a3_2 = (c3**2 / c2) * phi_at(2, 2)

    a4_2 = (c4**2 / c2) * phi_at(2, 3)
    a4_3 = (c4**2 * phi_at(2, 3) - a4_2 * c2) / c3

    a5_2 = 0.0
    a5_3 = (
        -c4 * c5**2 * phi_at(2, 4) + 2.0 * c5**3 * phi_at(3, 4)
    ) / (c3 * (c3 - c4))
    a5_4 = (
        -c3 * c5**2 * phi_at(2, 4) + 2.0 * c5**3 * phi_at(3, 4)
    ) / (c4 * (c4 - c3))

    a6_2 = 0.0
    a6_3 = (
        -c4 * c6**2 * phi_at(2, 5) + 2.0 * c6**3 * phi_at(3, 5)
    ) / (c3 * (c3 - c4))
    a6_4 = (
        -c3 * c6**2 * phi_at(2, 5) + 2.0 * c6**3 * phi_at(3, 5)
    ) / (c4 * (c4 - c3))
    a6_5 = (c6**2 * phi_at(2, 5) - a6_3 * c3 - a6_4 * c4) / c5

    b2 = b3 = b4 = 0.0
    b5 = (-c6 * phi_at(2) + 2.0 * phi_at(3)) / (c5 * (c5 - c6))
    b6 = (-c5 * phi_at(2) + 2.0 * phi_at(3)) / (c6 * (c6 - c5))

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, a3_2, 0.0, 0.0, 0.0, 0.0],
        [0.0, a4_2, a4_3, 0.0, 0.0, 0.0],
        [0.0, a5_2, a5_3, a5_4, 0.0, 0.0],
        [0.0, a6_2, a6_3, a6_4, a6_5, 0.0],
    ]
    b = [0.0, b2, b3, b4, b5, b6]

    for row_idx, stage_c in enumerate(c):
        a[row_idx][0] = stage_c * phi_at(1, row_idx) - sum(a[row_idx])
    b[0] = phi_at(1) - sum(b)
    return c, a, b


def _weighted_sum(weights: list[float], values: list[torch.Tensor]) -> torch.Tensor:
    result = torch.zeros_like(values[0])
    for weight, value in zip(weights, values):
        if weight != 0.0:
            result = result + float(weight) * value
    return result


@torch.inference_mode()
def _sample_chunk_signal(
    model: SpatialDiT,
    cond_chunk: torch.Tensor,
    valid_mask: torch.Tensor | None,
    z0_chunk: torch.Tensor,
    solver: SolverName,
    solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    mem: torch.Tensor | None,
    mix_style: torch.Tensor | None = None,
    amplitude_gain: torch.Tensor | None = None,
    one_step: bool = False,
    one_step_input: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample one waveform-patch chunk and optionally return updated memory tokens."""
    batch_size = cond_chunk.shape[0]
    if batch_size != 1:
        raise ValueError(f"Expected batch_size=1 for inference chunk, got {batch_size}")

    device = cond_chunk.device
    dtype = cond_chunk.dtype

    solver = solver.lower()
    if solver in {"midpoint", "midpoint_rk2", "midpoint-rk2", "rk2"}:
        solver = "midpoint_rk2"
    mem_fixed = mem
    conditioning_cache = None
    cache_builder = getattr(model, "build_conditioning_cache", None)
    if cache_builder is None and hasattr(model, "_orig_mod"):
        cache_builder = getattr(model._orig_mod, "build_conditioning_cache", None)
    if cache_builder is not None:
        conditioning_cache = cache_builder(
            z_cond=cond_chunk,
            valid_mask=valid_mask,
        )

    def predict_clean(t_value: float, z_state: torch.Tensor) -> torch.Tensor:
        """Evaluate the model clean waveform-patch prediction at scalar time `t_value`."""
        t_batch = torch.full(
            (batch_size,),
            float(t_value),
            device=z_state.device,
            dtype=z_state.dtype,
        )
        kwargs = {
            "zt": z_state,
            "t": t_batch,
            "z_cond": cond_chunk,
            "valid_mask": valid_mask,
        }
        if mix_style is not None:
            kwargs["mix_style"] = mix_style
        if amplitude_gain is not None:
            kwargs["amplitude_gain"] = amplitude_gain
        if mem_fixed is not None:
            kwargs["mem"] = mem_fixed
        if conditioning_cache is not None:
            kwargs["conditioning_cache"] = conditioning_cache
        return cast(torch.Tensor, model(**kwargs))

    def predict_velocity(t_value: float, z_state: torch.Tensor) -> torch.Tensor:
        clean_prediction = predict_clean(t_value, z_state)
        return _clean_prediction_to_velocity(clean_prediction, z_state, t_value)

    if one_step:
        input_mode = str(one_step_input).strip().lower()
        zt_fixed = cond_chunk.clone() if input_mode == "cond" else torch.zeros_like(z0_chunk)
        t_batch = torch.full((batch_size,), 1.0, device=device, dtype=dtype)
        kwargs: dict = {
            "zt": zt_fixed, "t": t_batch, "z_cond": cond_chunk,
            "valid_mask": valid_mask,
        }
        if mix_style is not None:
            kwargs["mix_style"] = mix_style
        if amplitude_gain is not None:
            kwargs["amplitude_gain"] = amplitude_gain
        if mem is not None:
            kwargs["mem"] = mem
            kwargs["return_mem"] = True
        if conditioning_cache is not None:
            kwargs["conditioning_cache"] = conditioning_cache
        if mem is not None:
            z1_chunk, mem_out = model(**kwargs)
            return z1_chunk, mem_out
        return cast(torch.Tensor, model(**kwargs)), None

    if solver == "heun":
        dt = _INTEGRATION_T_END / float(solver_steps)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            t1 = float(step_idx + 1) * dt

            v0 = predict_velocity(t0, z_state)
            z_euler = z_state + dt * v0
            v1 = predict_velocity(t1, z_euler)
            z_state = z_state + 0.5 * dt * (v0 + v1)

        z1_chunk = predict_clean(_INTEGRATION_T_END, z_state)
    elif solver == "unipc":
        dt = _INTEGRATION_T_END / float(solver_steps)
        z_state = z0_chunk
        previous_velocity: torch.Tensor | None = None
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            t1 = float(step_idx + 1) * dt

            v0 = predict_velocity(t0, z_state)
            if previous_velocity is None:
                z_pred = z_state + dt * v0
            else:
                z_pred = z_state + dt * (1.5 * v0 - 0.5 * previous_velocity)

            v1 = predict_velocity(t1, z_pred)
            if previous_velocity is None:
                z_state = z_state + 0.5 * dt * (v0 + v1)
            else:
                z_state = z_state + dt * (
                    (5.0 / 12.0) * v1
                    + (2.0 / 3.0) * v0
                    - (1.0 / 12.0) * previous_velocity
                )
            previous_velocity = v0

        z1_chunk = predict_clean(_INTEGRATION_T_END, z_state)
    elif solver == "euler":
        dt = _INTEGRATION_T_END / float(solver_steps)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            v0 = predict_velocity(t0, z_state)
            z_state = z_state + dt * v0

        z1_chunk = predict_clean(_INTEGRATION_T_END, z_state)
    elif solver == "midpoint_rk2":
        dt = _INTEGRATION_T_END / float(solver_steps)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            t_mid = min(t0 + 0.5 * dt, _INTEGRATION_T_END)

            v0 = predict_velocity(t0, z_state)
            z_mid = z_state + 0.5 * dt * v0
            v_mid = predict_velocity(t_mid, z_mid)
            z_state = z_state + dt * v_mid

        z1_chunk = predict_clean(_INTEGRATION_T_END, z_state)
    elif solver in {"res6s", "res_6s"}:
        dt = _INTEGRATION_T_END / float(solver_steps)
        c, a, b = _res6s_tableau(dt)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t_base = float(step_idx) * dt
            velocities: list[torch.Tensor] = []
            for stage_idx, stage_c in enumerate(c):
                if stage_idx == 0:
                    z_stage = z_state
                else:
                    z_stage = z_state + dt * _weighted_sum(
                        a[stage_idx][:stage_idx],
                        velocities,
                    )
                t_stage = min(t_base + stage_c * dt, _INTEGRATION_T_END)
                velocities.append(predict_velocity(t_stage, z_stage))
            z_state = z_state + dt * _weighted_sum(b, velocities)

        z1_chunk = predict_clean(_INTEGRATION_T_END, z_state)
    else:

        def velocity_field(
            t_scalar: torch.Tensor, z_state: torch.Tensor
        ) -> torch.Tensor:
            """`torchdiffeq` callback wrapping scalar time tensor -> model velocity."""
            return predict_velocity(float(t_scalar.item()), z_state)

        time_grid = _resolve_time_grid(
            method=solver,
            num_steps=solver_steps,
            device=device,
            dtype=dtype,
        )

        trajectory = odeint(
            func=velocity_field,
            y0=z0_chunk,
            t=time_grid,
            method=solver,
            rtol=solver_rtol,
            atol=solver_atol,
        )
        z_state = cast(torch.Tensor, trajectory[-1])
        z1_chunk = predict_clean(_INTEGRATION_T_END, z_state)

    if mem is None:
        return z1_chunk, None

    final_t = torch.full(
        (1,),
        _INTEGRATION_T_END,
        device=device,
        dtype=dtype,
    )
    final_kwargs = {
        "zt": z1_chunk,
        "t": final_t,
        "z_cond": cond_chunk,
        "valid_mask": valid_mask,
        "mem": mem,
        "return_mem": True,
    }
    if mix_style is not None:
        final_kwargs["mix_style"] = mix_style
    if amplitude_gain is not None:
        final_kwargs["amplitude_gain"] = amplitude_gain
    if conditioning_cache is not None:
        final_kwargs["conditioning_cache"] = conditioning_cache
    _, mem_out = model(**final_kwargs)
    return z1_chunk, mem_out


@torch.inference_mode()
def generate_spatial_signal(
    model: SpatialDiT,
    cond_signal: torch.Tensor,
    chunk_frames: int,
    overlap_frames: int,
    solver: SolverName,
    solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    seed: int,
    mix_style: torch.Tensor | None = None,
    amplitude_gain: torch.Tensor | None = None,
    one_step: bool = False,
    one_step_input: str = "zeros",
) -> torch.Tensor:
    """Sample target waveform patches from conditioning patches with overlap-add."""
    if cond_signal.dim() != 3:
        raise ValueError(f"cond_signal must be [C,P,T], got {tuple(cond_signal.shape)}")
    if cond_signal.shape[0] != model.cond_channels:
        raise ValueError(
            "cond_signal first dimension must match model.cond_channels "
            f"({cond_signal.shape[0]} != {model.cond_channels})"
        )
    if cond_signal.shape[1] != model.patch_size:
        raise ValueError(
            "cond_signal patch dimension must match model.patch_size "
            f"({cond_signal.shape[1]} != {model.patch_size})"
        )

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be > 0")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")
    if overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be smaller than chunk_frames")
    if solver_steps <= 0:
        raise ValueError("solver_steps must be > 0")

    cond_signal = cond_signal.contiguous()
    inference_dtype = cond_signal.dtype
    if mix_style is not None:
        if mix_style.dim() == 1:
            mix_style = mix_style.view(1, -1)
        expected_dim = int(getattr(model, "mix_style_dim", mix_style.shape[-1]))
        if mix_style.shape != (1, expected_dim):
            raise ValueError(
                "mix_style must be [K] or [1,K] with "
                f"K={expected_dim}, got {tuple(mix_style.shape)}"
            )
        mix_style = mix_style.to(device=cond_signal.device, dtype=inference_dtype)
    total_frames = cond_signal.shape[-1]
    stride_frames = chunk_frames - overlap_frames
    starts = _segment_starts(
        total_frames=total_frames,
        window_frames=chunk_frames,
        stride_frames=stride_frames,
    )

    target_channels = model.target_channels
    patch_size = model.patch_size

    assembled = torch.zeros(
        (target_channels, patch_size, total_frames), dtype=torch.float32
    )
    weight_sum = torch.zeros((total_frames,), dtype=torch.float32)

    device = cond_signal.device

    if one_step:
        z0_full = torch.zeros(
            (1, target_channels, patch_size, total_frames),
            device=device,
            dtype=inference_dtype,
        )
        generator = None
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        z0_full = torch.randn(
            (1, target_channels, patch_size, total_frames),
            device=device,
            dtype=inference_dtype,
            generator=generator,
        )

    mem = model.init_memory(batch_size=1, device=device, dtype=inference_dtype)

    for idx, start in enumerate(starts):
        end = start + chunk_frames
        segment_length = min(chunk_frames, total_frames - start)
        end = min(end, total_frames)

        cond_chunk = cond_signal[:, :, start:end]
        z0_chunk = z0_full[..., start:end]

        if segment_length < chunk_frames:
            pad_t = chunk_frames - segment_length
            pad_cond = torch.zeros(
                (model.cond_channels, patch_size, pad_t),
                device=device,
                dtype=inference_dtype,
            )
            if one_step:
                pad_z0 = torch.zeros(
                    (1, target_channels, patch_size, pad_t),
                    device=device, dtype=inference_dtype,
                )
            else:
                pad_z0 = torch.randn(
                    (1, target_channels, patch_size, pad_t),
                    device=device,
                    dtype=inference_dtype,
                    generator=generator,
                )
            cond_chunk = torch.cat([cond_chunk, pad_cond], dim=-1)
            z0_chunk = torch.cat([z0_chunk, pad_z0], dim=-1)

        cond_chunk = cond_chunk.unsqueeze(0)

        valid_mask = None
        if segment_length < chunk_frames:
            valid_mask = torch.zeros((1, chunk_frames), device=device, dtype=torch.bool)
            valid_mask[:, :segment_length] = True

        sample_kwargs = {
            "model": model,
            "cond_chunk": cond_chunk,
            "valid_mask": valid_mask,
            "z0_chunk": z0_chunk,
            "solver": solver,
            "solver_steps": solver_steps,
            "solver_rtol": solver_rtol,
            "solver_atol": solver_atol,
            "mem": mem,
            "one_step": one_step,
            "one_step_input": one_step_input,
        }
        if mix_style is not None:
            sample_kwargs["mix_style"] = mix_style
        if amplitude_gain is not None:
            sample_kwargs["amplitude_gain"] = amplitude_gain
        z1_chunk, mem = _sample_chunk_signal(**sample_kwargs)

        pred_chunk = z1_chunk[0, :, :, :segment_length].detach().cpu().float()

        w = _chunk_weight(
            chunk_length=segment_length,
            overlap_frames=overlap_frames,
            is_first=(idx == 0),
            is_last=(idx == len(starts) - 1),
            device=pred_chunk.device,
        )
        assembled[:, :, start:end] += pred_chunk * w[None, None, :]
        weight_sum[start:end] += w

    weight_sum = torch.clamp(weight_sum, min=1e-8)
    assembled = assembled / weight_sum[None, None, :]
    return assembled.contiguous()


def resolve_chunk_frames(
    cond_signal_frames: int,
    patch_fps: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[int, int]:
    """Resolve chunk and overlap frame counts from second-based inference settings."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")

    chunk_frames = max(1, int(round(chunk_seconds * patch_fps)))
    overlap_frames = int(round(overlap_seconds * patch_fps))
    overlap_frames = min(overlap_frames, max(0, chunk_frames - 1))

    if cond_signal_frames < chunk_frames:
        chunk_frames = cond_signal_frames
        overlap_frames = 0
    return chunk_frames, overlap_frames
