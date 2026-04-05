"""Latent sampling utilities used by the inference runner."""

from __future__ import annotations

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
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    return torch.tensor([0.0, 1.0], device=device, dtype=dtype)


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


@torch.no_grad()
def _sample_chunk_latent(
    model: SpatialDiT,
    cond_chunk: torch.Tensor,
    valid_mask: torch.Tensor,
    z0_chunk: torch.Tensor,
    solver: SolverName,
    solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    mem: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample one latent chunk and optionally return updated memory tokens."""
    batch_size = cond_chunk.shape[0]
    if batch_size != 1:
        raise ValueError(f"Expected batch_size=1 for inference chunk, got {batch_size}")

    device = cond_chunk.device
    dtype = cond_chunk.dtype

    solver = solver.lower()
    mem_fixed = mem

    def predict_velocity(t_value: float, z_state: torch.Tensor) -> torch.Tensor:
        """Evaluate the model velocity field at scalar time `t_value`."""
        t_batch = torch.full(
            (batch_size,),
            float(t_value),
            device=z_state.device,
            dtype=z_state.dtype,
        )
        if mem_fixed is None:
            return cast(
                torch.Tensor,
                model(zt=z_state, t=t_batch, z_cond=cond_chunk, valid_mask=valid_mask),
            )
        return cast(
            torch.Tensor,
            model(
                zt=z_state,
                t=t_batch,
                z_cond=cond_chunk,
                valid_mask=valid_mask,
                mem=mem_fixed,
            ),
        )

    if solver == "heun":
        dt = 1.0 / float(solver_steps)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t0 = float(step_idx) * dt
            t1 = float(step_idx + 1) * dt

            v0 = predict_velocity(t0, z_state)
            z_euler = z_state + dt * v0
            v1 = predict_velocity(t1, z_euler)
            z_state = z_state + 0.5 * dt * (v0 + v1)

        z1_chunk = z_state
    elif solver == "unipc":
        dt = 1.0 / float(solver_steps)
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

        z1_chunk = z_state
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
        z1_chunk = cast(torch.Tensor, trajectory[-1])

    if mem is None:
        return z1_chunk, None

    ones = torch.ones((1,), device=device, dtype=dtype)
    _, mem_out = model(
        zt=z1_chunk,
        t=ones,
        z_cond=cond_chunk,
        valid_mask=valid_mask,
        mem=mem,
        return_mem=True,
    )
    return z1_chunk, mem_out


@torch.no_grad()
def generate_spatial_latent(
    model: SpatialDiT,
    cond_latent: torch.Tensor,
    chunk_frames: int,
    overlap_frames: int,
    solver: SolverName,
    solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    seed: int,
) -> torch.Tensor:
    """Sample target latents from conditioning latents with overlap-add chunking."""
    if cond_latent.dim() != 3:
        raise ValueError(f"cond_latent must be [C,D,T], got {tuple(cond_latent.shape)}")
    if cond_latent.shape[0] != 1:
        raise ValueError(
            "cond_latent first dimension must be 1 for current setup, "
            f"got {cond_latent.shape[0]}"
        )

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be > 0")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")
    if overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be smaller than chunk_frames")
    if solver_steps <= 0:
        raise ValueError("solver_steps must be > 0")

    cond_latent = cond_latent.contiguous().float()
    total_frames = cond_latent.shape[-1]
    stride_frames = chunk_frames - overlap_frames
    starts = _segment_starts(
        total_frames=total_frames,
        window_frames=chunk_frames,
        stride_frames=stride_frames,
    )

    target_channels = model.target_channels
    latent_dim = model.latent_dim

    assembled = torch.zeros(
        (target_channels, latent_dim, total_frames), dtype=torch.float32
    )
    weight_sum = torch.zeros((total_frames,), dtype=torch.float32)

    device = cond_latent.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    z0_full = torch.randn(
        (1, target_channels, latent_dim, total_frames),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    mem = model.init_memory(batch_size=1, device=device, dtype=torch.float32)

    for idx, start in enumerate(starts):
        end = start + chunk_frames
        segment_length = min(chunk_frames, total_frames - start)
        end = min(end, total_frames)

        cond_chunk = cond_latent[:, :, start:end]
        z0_chunk = z0_full[..., start:end]

        if segment_length < chunk_frames:
            pad_t = chunk_frames - segment_length
            pad_cond = torch.zeros(
                (1, cond_latent.shape[1], pad_t), device=device, dtype=torch.float32
            )
            pad_z0 = torch.randn(
                (1, target_channels, latent_dim, pad_t),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            cond_chunk = torch.cat([cond_chunk, pad_cond], dim=-1)
            z0_chunk = torch.cat([z0_chunk, pad_z0], dim=-1)

        cond_chunk = cond_chunk.unsqueeze(1)

        valid_mask = torch.zeros((1, chunk_frames), device=device, dtype=torch.bool)
        valid_mask[:, :segment_length] = True

        z1_chunk, mem = _sample_chunk_latent(
            model=model,
            cond_chunk=cond_chunk,
            valid_mask=valid_mask,
            z0_chunk=z0_chunk,
            solver=solver,
            solver_steps=solver_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            mem=mem,
        )

        pred_chunk = z1_chunk[0, :, :, :segment_length].detach().cpu()

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
    cond_latent_frames: int,
    latent_fps: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[int, int]:
    """Resolve chunk and overlap frame counts from second-based inference settings."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")

    chunk_frames = max(1, int(round(chunk_seconds * latent_fps)))
    overlap_frames = int(round(overlap_seconds * latent_fps))
    overlap_frames = min(overlap_frames, max(0, chunk_frames - 1))

    if cond_latent_frames < chunk_frames:
        chunk_frames = cond_latent_frames
        overlap_frames = 0
    return chunk_frames, overlap_frames
