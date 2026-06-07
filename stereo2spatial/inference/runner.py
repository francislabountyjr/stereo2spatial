"""Top-level inference orchestration from stereo waveform to spatial waveform."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import torch

from stereo2spatial.common.amplitude_lift import (
    apply_amplitude_lift,
    compute_shared_rms_gain,
    undo_amplitude_lift,
)
from stereo2spatial.common.mix_style import (
    mix_style_active_names,
    mix_style_dict_to_vector,
    mix_style_preset_values,
)
from stereo2spatial.modeling import SpatialDiT
from stereo2spatial.training.config import TrainConfig

from .audio import read_audio_channels_first, write_audio_channels_first
from .checkpoint import load_model_weights, resolve_checkpoint_path
from .sampling import generate_spatial_signal, resolve_chunk_frames

RequestedSolverName = Literal[
    "auto",
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "res6s",
    "res_6s",
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
    "res6s",
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
    "res6s",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
}


def _default_device() -> torch.device:
    """Return the default inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_inference_solver(
    *,
    requested_solver: RequestedSolverName,
) -> ResolvedSolverName:
    """Resolve requested solver, supporting ``auto`` => default heun."""
    requested = str(requested_solver).strip().lower()
    if requested == "auto":
        requested = "heun"
    if requested == "res_6s":
        requested = "res6s"
    if requested not in _INFERENCE_SOLVERS:
        raise ValueError(
            "solver must be one of: auto, dopri5, heun, euler, unipc, "
            "res6s/res_6s, midpoint, rk4, explicit_adams, implicit_adams"
        )
    return requested  # type: ignore[return-value]


def _prepare_conditioning_audio(audio: torch.Tensor, cond_channels: int) -> torch.Tensor:
    """Map mono/stereo input audio to the model conditioning channel count."""
    if audio.dim() != 2:
        raise ValueError(f"audio must be [C,S], got {tuple(audio.shape)}")
    if audio.shape[0] not in {1, 2}:
        raise ValueError(f"Input must be mono or stereo, got channels={audio.shape[0]}")
    if cond_channels == 1:
        return audio.mean(dim=0, keepdim=True) if audio.shape[0] == 2 else audio
    if cond_channels == 2:
        return audio.expand(2, -1).contiguous() if audio.shape[0] == 1 else audio
    raise ValueError(f"Waveform inference expects cond_channels 1 or 2, got {cond_channels}")


def _patch_audio(audio: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int]:
    """Convert channel-first audio `[C,S]` to waveform patches `[C,P,T]`."""
    if audio.dim() != 2:
        raise ValueError(f"audio must be [C,S], got {tuple(audio.shape)}")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    sample_count = int(audio.shape[-1])
    frame_count = max(1, (sample_count + patch_size - 1) // patch_size)
    padded_samples = frame_count * patch_size
    if padded_samples != sample_count:
        pad = torch.zeros(
            (audio.shape[0], padded_samples - sample_count),
            dtype=audio.dtype,
            device=audio.device,
        )
        audio = torch.cat([audio, pad], dim=-1)
    patches = audio.reshape(audio.shape[0], frame_count, patch_size)
    return patches.permute(0, 2, 1).contiguous(), sample_count


def _unpatch_audio(patches: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Convert waveform patches `[C,P,T]` back to channel-first audio `[C,S]`."""
    if patches.dim() != 3:
        raise ValueError(f"patches must be [C,P,T], got {tuple(patches.shape)}")
    audio = patches.permute(0, 2, 1).reshape(patches.shape[0], -1)
    return audio[:, :sample_count].contiguous()


class InferenceReport(TypedDict):
    """Structured metadata emitted by :func:`run_inference`."""

    input_audio_path: str
    output_audio_path: str
    checkpoint_path: str
    sample_rate: int
    input_channels: int
    input_samples: int
    conditioning_signal_shape: list[int]
    pred_signal_shape: list[int]
    decoded_shape: list[int]
    weights_source: str
    patch_fps: float
    chunk_seconds: float
    chunk_frames: int
    overlap_seconds: float
    overlap_frames: int
    solver: ResolvedSolverName
    solver_steps: int
    solver_rtol: float
    solver_atol: float
    seed: int
    mix_style: list[float] | None
    mix_style_preset: str | None
    amplitude_lift_enabled: bool
    amplitude_lift_reference: str
    amplitude_lift_target_rms: float
    amplitude_lift_scale: float
    amplitude_lift_clip_value: float | None
    amplitude_lift_gain: float | None
    device: str


def _resolve_inference_mix_style(
    raw_mix_style: list[float] | dict[str, float] | None,
    mix_style_dim: int,
    target_channels: int,
    preset_name: str | None = None,
) -> torch.Tensor | None:
    """Return optional normalized mix-style tensor for inference."""
    if int(mix_style_dim) <= 0:
        return None
    if preset_name is not None:
        if raw_mix_style is not None:
            raise ValueError("mix_style and mix_style_preset cannot both be provided")
        raw_mix_style = mix_style_preset_values(preset_name)
    if raw_mix_style is None:
        return None
    if isinstance(raw_mix_style, dict):
        names: tuple[str, ...] | None = None
        if int(target_channels) == 2 and int(mix_style_dim) == 10:
            names = mix_style_active_names(layout_mode="binaural_stereo")
        elif int(target_channels) == 6 and int(mix_style_dim) == 11:
            names = mix_style_active_names(layout_mode="5_1_rear")
        values = mix_style_dict_to_vector(raw_mix_style, names=names)
    else:
        values = [float(value) for value in raw_mix_style]
    if len(values) != int(mix_style_dim):
        raise ValueError(
            f"mix_style must contain {int(mix_style_dim)} values, got {len(values)}"
        )
    return torch.tensor(values, dtype=torch.float32).view(1, -1)


@torch.no_grad()
def run_inference(
    config: TrainConfig,
    checkpoint: str | Path,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    sample_rate: int,
    chunk_seconds: float | None,
    overlap_seconds: float,
    solver: RequestedSolverName,
    solver_steps: int | None,
    solver_rtol: float,
    solver_atol: float,
    seed: int,
    device: str | None,
    show_progress: bool,
    normalize_peak: bool,
    mix_style: list[float] | dict[str, float] | None = None,
    mix_style_preset: str | None = None,
    weights_source: WeightsSource = "auto",
) -> InferenceReport:
    """
    Run end-to-end inference from input waveform to rendered multichannel WAV.

    The output channel count is set by ``config.model.target_channels``.
    """
    del show_progress
    run_device = torch.device(device) if device else _default_device()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=checkpoint,
        output_dir=config.output_dir,
    )

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        patch_size=config.model.patch_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=getattr(config.model, "num_memory_tokens", 0),
        mix_style_dim=getattr(config.model, "mix_style_dim", 0),
        waveform_level_depth=getattr(config.model, "waveform_level_depth", 0),
        waveform_micro_patch_size=getattr(
            config.model, "waveform_micro_patch_size", 16
        ),
        waveform_hidden_dim=getattr(config.model, "waveform_hidden_dim", 16),
        waveform_num_heads=getattr(config.model, "waveform_num_heads", None),
        waveform_mlp_ratio=getattr(config.model, "waveform_mlp_ratio", 2.0),
        activation_checkpointing=getattr(
            config.model, "activation_checkpointing", False
        ),
    )
    used_weights_source = load_model_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        weights_source=weights_source,
    )
    model = model.to(device=run_device, dtype=torch.float32)
    model.eval()

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

    conditioning_audio = _prepare_conditioning_audio(
        audio.float(),
        cond_channels=int(config.model.cond_channels),
    )
    amplitude_lift_gain = None
    if bool(getattr(config.data, "amplitude_lift_enabled", False)):
        lift_reference = str(
            getattr(config.data, "amplitude_lift_reference", "source")
        ).strip().lower()
        if lift_reference != "source":
            raise ValueError(
                "Inference amplitude lifting requires data.amplitude_lift_reference="
                "'source' because target audio is unavailable at inference time."
            )
        amplitude_lift_gain = compute_shared_rms_gain(
            conditioning_audio,
            target_rms=float(getattr(config.data, "amplitude_lift_target_rms", 0.33)),
            eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
        )
        conditioning_audio = apply_amplitude_lift(
            conditioning_audio,
            gain=amplitude_lift_gain,
            scale=float(getattr(config.data, "amplitude_lift_scale", 3.0)),
            clip_value=getattr(config.data, "amplitude_lift_clip_value", 4.0),
        )
    cond_signal, input_samples = _patch_audio(
        conditioning_audio,
        patch_size=int(config.model.patch_size),
    )
    mix_style_tensor = _resolve_inference_mix_style(
        raw_mix_style=mix_style,
        mix_style_dim=int(getattr(config.model, "mix_style_dim", 0)),
        target_channels=int(getattr(config.model, "target_channels", 0)),
        preset_name=mix_style_preset,
    )

    target_chunk_seconds = (
        float(chunk_seconds)
        if chunk_seconds is not None
        else float(config.data.segment_seconds)
    )
    patch_fps = float(actual_sample_rate) / float(config.model.patch_size)
    chunk_frames, overlap_frames = resolve_chunk_frames(
        cond_signal_frames=cond_signal.shape[-1],
        patch_fps=patch_fps,
        chunk_seconds=target_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    resolved_solver = _resolve_inference_solver(requested_solver=solver)
    resolved_solver_steps = 64 if solver_steps is None else max(1, int(solver_steps))

    pred_signal = generate_spatial_signal(
        model=model,
        cond_signal=cond_signal.to(run_device),
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        solver=resolved_solver,
        solver_steps=resolved_solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        seed=seed,
        mix_style=(
            mix_style_tensor.to(run_device) if mix_style_tensor is not None else None
        ),
    )

    decoded = _unpatch_audio(pred_signal.cpu().float(), sample_count=input_samples)
    if amplitude_lift_gain is not None:
        decoded = undo_amplitude_lift(
            decoded,
            gain=amplitude_lift_gain.cpu(),
            scale=float(getattr(config.data, "amplitude_lift_scale", 3.0)),
            eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
        )

    if normalize_peak:
        peak = decoded.abs().amax().item()
        if peak > 1e-8:
            decoded = decoded / peak * 0.99

    write_audio_channels_first(
        audio_path=output_path,
        audio=decoded,
        sample_rate=actual_sample_rate,
        channel_order=config.training.downmix_channel_order,
    )

    report: InferenceReport = {
        "input_audio_path": str(input_path),
        "output_audio_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
        "sample_rate": int(actual_sample_rate),
        "input_channels": int(audio.shape[0]),
        "input_samples": int(audio.shape[-1]),
        "conditioning_signal_shape": [int(x) for x in cond_signal.shape],
        "pred_signal_shape": [int(x) for x in pred_signal.shape],
        "decoded_shape": [int(x) for x in decoded.shape],
        "weights_source": used_weights_source,
        "patch_fps": float(patch_fps),
        "chunk_seconds": float(target_chunk_seconds),
        "chunk_frames": int(chunk_frames),
        "overlap_seconds": float(overlap_seconds),
        "overlap_frames": int(overlap_frames),
        "solver": resolved_solver,
        "solver_steps": int(resolved_solver_steps),
        "solver_rtol": float(solver_rtol),
        "solver_atol": float(solver_atol),
        "seed": int(seed),
        "mix_style": (
            [float(x) for x in mix_style_tensor.flatten().tolist()]
            if mix_style_tensor is not None
            else None
        ),
        "mix_style_preset": mix_style_preset,
        "amplitude_lift_enabled": bool(
            getattr(config.data, "amplitude_lift_enabled", False)
        ),
        "amplitude_lift_reference": str(
            getattr(config.data, "amplitude_lift_reference", "source")
        ),
        "amplitude_lift_target_rms": float(
            getattr(config.data, "amplitude_lift_target_rms", 0.33)
        ),
        "amplitude_lift_scale": float(getattr(config.data, "amplitude_lift_scale", 3.0)),
        "amplitude_lift_clip_value": getattr(
            config.data,
            "amplitude_lift_clip_value",
            4.0,
        ),
        "amplitude_lift_gain": (
            float(amplitude_lift_gain.flatten()[0].item())
            if amplitude_lift_gain is not None
            else None
        ),
        "device": str(run_device),
    }
    return report
