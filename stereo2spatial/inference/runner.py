"""Top-level inference orchestration from waveform input to waveform output."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import torch

from stereo2spatial.codecs.ear_vae import (
    decode_channels_independent,
    get_default_device,
    load_vae,
    vae_encode,
)
from stereo2spatial.modeling import SpatialDiT
from stereo2spatial.training.config import TrainConfig

from .audio import read_audio_channels_first, write_audio_channels_first
from .checkpoint import load_model_weights, resolve_checkpoint_path
from .sampling import generate_spatial_latent, resolve_chunk_frames

RequestedSolverName = Literal[
    "auto",
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
]
ResolvedSolverName = Literal[
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
]
WeightsSource = Literal["auto", "ema", "student"]

_INFERENCE_SOLVERS = {
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
}


LATENT_FPS = 50


def _resolve_inference_solver(
    *,
    requested_solver: RequestedSolverName,
) -> ResolvedSolverName:
    """Resolve requested solver, supporting ``auto`` => default heun."""
    requested = str(requested_solver).strip().lower()
    if requested == "auto":
        requested = "heun"
    if requested not in _INFERENCE_SOLVERS:
        raise ValueError(
            "solver must be one of: auto, dopri5, heun, euler, unipc, "
            "midpoint, rk4, explicit_adams, implicit_adams"
        )
    return requested  # type: ignore[return-value]


class InferenceReport(TypedDict):
    """Structured metadata emitted by :func:`run_inference`."""

    input_audio_path: str
    output_audio_path: str
    checkpoint_path: str
    sample_rate: int
    input_channels: int
    input_samples: int
    conditioning_latent_shape: list[int]
    pred_latent_shape: list[int]
    decoded_shape: list[int]
    weights_source: str
    latent_fps: float
    chunk_seconds: float
    chunk_frames: int
    overlap_seconds: float
    overlap_frames: int
    solver: ResolvedSolverName
    solver_steps: int
    solver_rtol: float
    solver_atol: float
    seed: int
    device: str


@torch.no_grad()
def run_inference(
    config: TrainConfig,
    checkpoint: str | Path,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    vae_checkpoint_path: str | Path,
    vae_config_path: str | Path | None,
    sample_rate: int,
    chunk_seconds: float | None,
    overlap_seconds: float,
    solver: RequestedSolverName,
    solver_steps: int | None,
    solver_rtol: float,
    solver_atol: float,
    seed: int,
    device: str | None,
    encode_chunk_size_samples: int | None,
    encode_overlap_samples: int | None,
    decode_chunk_size_frames: int,
    decode_overlap_frames: int,
    disable_chunked_decode: bool,
    show_progress: bool,
    normalize_peak: bool,
    weights_source: WeightsSource = "auto",
) -> InferenceReport:
    """
    Run end-to-end inference from input waveform to rendered multichannel WAV.

    The output channel count is set by ``config.model.target_channels``.
    """
    run_device = torch.device(device) if device else get_default_device()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=checkpoint,
        output_dir=config.output_dir,
    )

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=getattr(config.model, "num_memory_tokens", 0),
    )
    used_weights_source = load_model_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        weights_source=weights_source,
    )
    model = model.to(device=run_device, dtype=torch.float32)
    model.eval()

    vae = load_vae(
        vae_checkpoint_path=vae_checkpoint_path,
        config_path=vae_config_path,
        device=run_device,
        torch_dtype=torch.float32,
    )

    input_path = Path(input_audio_path)
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, actual_sample_rate = read_audio_channels_first(
        audio_path=input_path,
        target_sample_rate=sample_rate,
    )
    if audio.shape[0] not in {1, 2}:
        raise ValueError(
            f"Input must be mono or stereo. Got channels={audio.shape[0]} for {input_path}"
        )

    cond_latent = vae_encode(
        vae=vae,
        audio=audio,
        sample_rate=actual_sample_rate,
        use_sample=False,
        use_chunked_encode=True,
        chunk_size_samples=encode_chunk_size_samples,
        overlap_samples=encode_overlap_samples,
        duplicate_mono_to_stereo=True,
        offload_latent_to_cpu=False,
        show_progress=show_progress,
        device=run_device,
    )
    if cond_latent.dim() != 2:
        raise ValueError(
            f"Expected encoded conditioning latent [D,T], got {tuple(cond_latent.shape)}"
        )
    cond_latent = cond_latent.unsqueeze(0).contiguous()

    target_chunk_seconds = (
        float(chunk_seconds)
        if chunk_seconds is not None
        else float(config.data.segment_seconds)
    )

    chunk_frames, overlap_frames = resolve_chunk_frames(
        cond_latent_frames=cond_latent.shape[-1],
        latent_fps=LATENT_FPS,
        chunk_seconds=target_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    resolved_solver = _resolve_inference_solver(
        requested_solver=solver,
    )
    resolved_solver_steps: int
    if solver_steps is None:
        resolved_solver_steps = 64
    else:
        resolved_solver_steps = max(1, int(solver_steps))

    pred_latent = generate_spatial_latent(
        model=model,
        cond_latent=cond_latent.to(run_device),
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        solver=resolved_solver,
        solver_steps=resolved_solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        seed=seed,
    )

    decoded = decode_channels_independent(
        vae=vae,
        channel_latents=pred_latent.to(run_device),
        use_chunked_decode=not disable_chunked_decode,
        chunk_size_frames=decode_chunk_size_frames,
        overlap_frames=decode_overlap_frames,
        offload_wav_to_cpu=True,
        reduction="mean",
        show_progress=show_progress,
        device=run_device,
    )
    decoded = decoded.float().cpu()

    if normalize_peak:
        peak = decoded.abs().amax().item()
        if peak > 1e-8:
            decoded = decoded / peak * 0.99

    write_audio_channels_first(
        audio_path=output_path,
        audio=decoded,
        sample_rate=actual_sample_rate,
    )

    report: InferenceReport = {
        "input_audio_path": str(input_path),
        "output_audio_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
        "sample_rate": int(actual_sample_rate),
        "input_channels": int(audio.shape[0]),
        "input_samples": int(audio.shape[-1]),
        "conditioning_latent_shape": [int(x) for x in cond_latent.shape],
        "pred_latent_shape": [int(x) for x in pred_latent.shape],
        "decoded_shape": [int(x) for x in decoded.shape],
        "weights_source": used_weights_source,
        "latent_fps": float(LATENT_FPS),
        "chunk_seconds": float(target_chunk_seconds),
        "chunk_frames": int(chunk_frames),
        "overlap_seconds": float(overlap_seconds),
        "overlap_frames": int(overlap_frames),
        "solver": resolved_solver,
        "solver_steps": int(resolved_solver_steps),
        "solver_rtol": float(solver_rtol),
        "solver_atol": float(solver_atol),
        "seed": int(seed),
        "device": str(run_device),
    }
    return report
