"""Training-aligned timestep-major sampling for long spatial signals.

Unlike the compatibility sampler, which solves one window completely before
moving to the next, this module keeps one song-level ODE state.  Every model
field evaluation sweeps all windows from left to right at the same diffusion
time, carrying memory only within that sweep.  The window predictions are then
overlap-added into one global field before the solver advances to its next
stage or timestep.
"""

from __future__ import annotations

from typing import cast

import torch
from torchdiffeq import odeint

from stereo2spatial.common.windowing import chunk_weight
from stereo2spatial.modeling import SpatialModel

from .solvers import (
    INTEGRATION_T_END,
    clean_prediction_to_velocity,
    res6s_tableau,
    weighted_sum,
)
from .windowing import extract_fixed_window, fixed_window_specs

SolverName = str


def _normalize_solver_name(solver: SolverName) -> str:
    """Normalize the aliases accepted by the regular inference sampler."""
    normalized = str(solver).strip().lower()
    if normalized == "res_6s":
        return "res6s"
    if normalized in {"midpoint", "midpoint-rk2", "rk2"}:
        return "midpoint_rk2"
    return normalized


def _resolve_time_grid(
    *,
    method: str,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the same float32 integration grid used by window-major sampling."""
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
            0.0,
            INTEGRATION_T_END,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )
    return torch.tensor(
        [0.0, INTEGRATION_T_END],
        device=device,
        dtype=torch.float32,
    )


def _validate_inputs(
    *,
    model: SpatialModel,
    cond_signal: torch.Tensor,
    chunk_frames: int,
    overlap_frames: int,
    solver_steps: int,
) -> None:
    """Validate public sampler inputs before allocating the global state."""
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
    if cond_signal.shape[-1] <= 0:
        raise ValueError("cond_signal must contain at least one frame")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be > 0")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")
    if overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be smaller than chunk_frames")
    if solver_steps <= 0:
        raise ValueError("solver_steps must be > 0")


@torch.inference_mode()
def generate_spatial_signal_timestep_major(
    model: SpatialModel,
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
    """Generate a signal by advancing every window together at each solver stage.

    Memory is initialized at the beginning of each global model evaluation,
    propagated from the first window to the last, and then discarded.  This
    matches the normal full-song training pass, where all windows share one
    sampled timestep and memory does not recur between diffusion timesteps.
    """
    _validate_inputs(
        model=model,
        cond_signal=cond_signal,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        solver_steps=solver_steps,
    )

    cond_signal = cond_signal.contiguous()
    device = cond_signal.device
    inference_dtype = cond_signal.dtype
    total_frames = int(cond_signal.shape[-1])
    target_channels = int(model.target_channels)
    patch_size = int(model.patch_size)
    solver_name = _normalize_solver_name(solver)

    if mix_style is not None:
        if mix_style.dim() == 1:
            mix_style = mix_style.view(1, -1)
        expected_dim = int(getattr(model, "mix_style_dim", mix_style.shape[-1]))
        if mix_style.shape != (1, expected_dim):
            raise ValueError(
                "mix_style must be [K] or [1,K] with "
                f"K={expected_dim}, got {tuple(mix_style.shape)}"
            )
        mix_style = mix_style.to(device=device, dtype=inference_dtype)

    specs = fixed_window_specs(
        total_frames=total_frames,
        window_frames=chunk_frames,
        overlap_frames=overlap_frames,
    )
    window_weights = [
        chunk_weight(
            chunk_length=spec.valid_frames,
            overlap_frames=overlap_frames,
            is_first=(index == 0),
            is_last=(index == len(specs) - 1),
            device=device,
            dtype=torch.float32,
        )
        for index, spec in enumerate(specs)
    ]

    def predict_clean_global(
        t_value: float,
        z_state: torch.Tensor,
    ) -> torch.Tensor:
        """Run one training-style memory sweep and stitch its clean prediction."""
        mem = model.init_memory(
            batch_size=1,
            device=device,
            dtype=inference_dtype,
        )
        assembled = torch.zeros(
            (1, target_channels, patch_size, total_frames),
            device=device,
            dtype=torch.float32,
        )
        weight_sum = torch.zeros(
            (total_frames,),
            device=device,
            dtype=torch.float32,
        )

        for index, spec in enumerate(specs):
            z_window = extract_fixed_window(z_state, spec, pad_value=0.0)
            cond_window = extract_fixed_window(cond_signal, spec, pad_value=0.0)
            valid_mask = torch.zeros(
                (1, chunk_frames),
                device=device,
                dtype=torch.bool,
            )
            valid_mask[:, : spec.valid_frames] = True
            t_batch = torch.full(
                (1,),
                float(t_value),
                device=device,
                dtype=torch.float32,
            )
            kwargs: dict[str, object] = {
                "zt": z_window,
                "t": t_batch,
                "z_cond": cond_window.unsqueeze(0),
                "valid_mask": valid_mask,
            }
            if mix_style is not None:
                kwargs["mix_style"] = mix_style
            if amplitude_gain is not None:
                kwargs["amplitude_gain"] = amplitude_gain
            if mem is not None:
                kwargs["mem"] = mem
                kwargs["return_mem"] = True
                model_output = model(**kwargs)
                if not isinstance(model_output, tuple) or len(model_output) != 2:
                    raise RuntimeError(
                        "return_mem model evaluation must return (prediction, mem)"
                    )
                clean_window, mem = model_output
            else:
                clean_window = cast(torch.Tensor, model(**kwargs))

            clean_valid = clean_window[..., : spec.valid_frames].float()
            weight = window_weights[index]
            assembled[..., spec.start_frame : spec.end_frame] += (
                clean_valid * weight[None, None, None, :]
            )
            weight_sum[spec.start_frame : spec.end_frame] += weight

        clean_global = assembled / weight_sum.clamp_min(1.0e-8)[None, None, None, :]
        return clean_global.to(dtype=z_state.dtype)

    def predict_velocity_global(
        t_value: float,
        z_state: torch.Tensor,
    ) -> torch.Tensor:
        clean_prediction = predict_clean_global(t_value, z_state)
        return clean_prediction_to_velocity(clean_prediction, z_state, t_value)

    if one_step:
        input_mode = str(one_step_input).strip().lower()
        if input_mode not in {"zeros", "cond"}:
            raise ValueError("one_step_input must be 'zeros' or 'cond'")
        if input_mode == "cond":
            if int(cond_signal.shape[0]) != target_channels:
                raise ValueError(
                    "one_step_input='cond' requires cond_channels == target_channels"
                )
            z_state = cond_signal.unsqueeze(0).clone()
        else:
            z_state = torch.zeros(
                (1, target_channels, patch_size, total_frames),
                device=device,
                dtype=inference_dtype,
            )
        return predict_clean_global(1.0, z_state)[0].detach().cpu().float().contiguous()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z_state = torch.randn(
        (1, target_channels, patch_size, total_frames),
        device=device,
        dtype=inference_dtype,
        generator=generator,
    )

    if solver_name == "heun":
        dt = INTEGRATION_T_END / float(solver_steps)
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            v0 = predict_velocity_global(t0, z_state)
            if step_idx == solver_steps - 1:
                # The terminal clean-to-velocity conversion is ill-conditioned.
                # Midpoint RK2 retains second order without probing t ~= 1.
                t_mid = t0 + 0.5 * dt
                z_mid = z_state + 0.5 * dt * v0
                v_mid = predict_velocity_global(t_mid, z_mid)
                z_state = z_state + dt * v_mid
            else:
                t1 = float(step_idx + 1) * dt
                z_euler = z_state + dt * v0
                v1 = predict_velocity_global(t1, z_euler)
                z_state = z_state + 0.5 * dt * (v0 + v1)
    elif solver_name == "unipc":
        dt = INTEGRATION_T_END / float(solver_steps)
        previous_velocity: torch.Tensor | None = None
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            t1 = float(step_idx + 1) * dt
            v0 = predict_velocity_global(t0, z_state)
            if previous_velocity is None:
                z_pred = z_state + dt * v0
            else:
                z_pred = z_state + dt * (1.5 * v0 - 0.5 * previous_velocity)
            v1 = predict_velocity_global(t1, z_pred)
            if previous_velocity is None:
                z_state = z_state + 0.5 * dt * (v0 + v1)
            else:
                z_state = z_state + dt * (
                    (5.0 / 12.0) * v1
                    + (2.0 / 3.0) * v0
                    - (1.0 / 12.0) * previous_velocity
                )
            previous_velocity = v0
    elif solver_name == "euler":
        dt = INTEGRATION_T_END / float(solver_steps)
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            z_state = z_state + dt * predict_velocity_global(t0, z_state)
    elif solver_name == "midpoint_rk2":
        dt = INTEGRATION_T_END / float(solver_steps)
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            t_mid = min(t0 + 0.5 * dt, INTEGRATION_T_END)
            v0 = predict_velocity_global(t0, z_state)
            z_mid = z_state + 0.5 * dt * v0
            v_mid = predict_velocity_global(t_mid, z_mid)
            z_state = z_state + dt * v_mid
    elif solver_name == "res6s":
        dt = INTEGRATION_T_END / float(solver_steps)
        c, a, b = res6s_tableau(dt)
        for step_idx in range(solver_steps):
            t_base = float(step_idx) * dt
            velocities: list[torch.Tensor] = []
            for stage_idx, stage_c in enumerate(c):
                if stage_idx == 0:
                    z_stage = z_state
                else:
                    z_stage = z_state + dt * weighted_sum(
                        a[stage_idx][:stage_idx],
                        velocities,
                    )
                t_stage = min(t_base + stage_c * dt, INTEGRATION_T_END)
                velocities.append(predict_velocity_global(t_stage, z_stage))
            z_state = z_state + dt * weighted_sum(b, velocities)
    else:

        def velocity_field(
            t_scalar: torch.Tensor,
            state: torch.Tensor,
        ) -> torch.Tensor:
            return predict_velocity_global(float(t_scalar.item()), state)

        time_grid = _resolve_time_grid(
            method=solver_name,
            num_steps=solver_steps,
            device=device,
        )
        trajectory = odeint(
            func=velocity_field,
            y0=z_state,
            t=time_grid,
            method=solver_name,
            rtol=solver_rtol,
            atol=solver_atol,
        )
        z_state = cast(torch.Tensor, trajectory[-1])

    final_clean = predict_clean_global(INTEGRATION_T_END, z_state)
    return final_clean[0].detach().cpu().float().contiguous()


__all__ = ["generate_spatial_signal_timestep_major"]
