"""Top-level audio inference for waveform and v1-compatible latent models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch

from stereo2spatial.codecs.ear_vae import (
    decode_channels_independent,
    encode_channels_independent,
    load_vae,
    vae_encode,
)
from stereo2spatial.common.amplitude_lift import (
    amplitude_lift_log_gain,
    apply_amplitude_lift,
    resolve_amplitude_lift_gain,
    undo_amplitude_lift,
    undo_wavflow_output_lift,
    wavflow_source_transform,
)
from stereo2spatial.common.mix_style import (
    mix_style_active_names,
    mix_style_dict_to_vector,
    mix_style_preset_values,
)
from stereo2spatial.modeling import (
    SpatialModel,
    build_spatial_model,
    is_legacy_vae_model,
)
from stereo2spatial.training.config import TrainConfig

from .audio import read_audio_channels_first, write_audio_channels_first
from .checkpoint import load_model_weights, resolve_checkpoint_path
from .export_bundle import LEGACY_VAE_SAMPLE_RATE
from .sampling import generate_spatial_signal, resolve_chunk_frames
from .timestep_sampling import generate_spatial_signal_timestep_major

RequestedSolverName = Literal[
    "auto",
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "res6s",
    "res_6s",
    "midpoint",
    "midpoint_rk2",
    "midpoint-rk2",
    "rk2",
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
    "midpoint_rk2",
    "rk4",
    "explicit_adams",
    "implicit_adams",
]
WeightsSource = Literal["auto", "ema", "student"]
InferenceDTypeName = Literal["float32", "float16", "bfloat16", "auto"]
SamplingOrder = Literal["window_major", "timestep_major"]

_INFERENCE_SOLVERS = {
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "res6s",
    "midpoint_rk2",
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
    if requested in {"midpoint", "midpoint-rk2", "rk2"}:
        requested = "midpoint_rk2"
    if requested not in _INFERENCE_SOLVERS:
        raise ValueError(
            "solver must be one of: auto, dopri5, heun, euler, unipc, "
            "res6s/res_6s, midpoint/midpoint_rk2/rk2, rk4, "
            "explicit_adams, implicit_adams"
        )
    return requested  # type: ignore[return-value]


def _prepare_conditioning_audio(
    audio: torch.Tensor, cond_channels: int
) -> torch.Tensor:
    """Map mono/stereo input audio to the model conditioning channel count."""
    if audio.dim() != 2:
        raise ValueError(f"audio must be [C,S], got {tuple(audio.shape)}")
    if audio.shape[0] not in {1, 2}:
        raise ValueError(f"Input must be mono or stereo, got channels={audio.shape[0]}")
    if cond_channels == 1:
        return audio.mean(dim=0, keepdim=True) if audio.shape[0] == 2 else audio
    if cond_channels == 2:
        return audio.expand(2, -1).contiguous() if audio.shape[0] == 1 else audio
    raise ValueError(
        f"Waveform inference expects cond_channels 1 or 2, got {cond_channels}"
    )


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
    sampling_order: SamplingOrder
    seed: int
    mix_style: list[float] | None
    mix_style_preset: str | None
    amplitude_lift_enabled: bool
    amplitude_lift_mode: str
    amplitude_lift_reference: str
    amplitude_lift_target_rms: float
    amplitude_lift_scale: float
    amplitude_lift_clip_value: float | None
    amplitude_lift_output_lufs: float
    amplitude_lift_gain: float | None
    device: str
    compiled: bool
    compile_mode: str | None
    inference_dtype: str
    architecture: str
    representation: str


@dataclass
class InferenceSession:
    """Loaded inference model and immutable runtime metadata."""

    config: TrainConfig
    model: torch.nn.Module
    checkpoint_path: Path
    run_device: torch.device
    used_weights_source: str
    compiled: bool
    compile_mode: str | None
    run_dtype: torch.dtype = torch.float32
    vae: torch.nn.Module | None = None


def _resolve_inference_dtype(
    *,
    requested_dtype: InferenceDTypeName,
    device: torch.device,
) -> torch.dtype:
    """Resolve the tensor dtype used for inference model weights and activations."""
    requested = str(requested_dtype).strip().lower()
    if requested == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if requested in {"float32", "fp32"}:
        return torch.float32
    if requested in {"float16", "fp16", "half"}:
        if device.type != "cuda":
            raise ValueError("float16 inference is only supported on CUDA devices")
        return torch.float16
    if requested in {"bfloat16", "bf16"}:
        if device.type != "cuda":
            raise ValueError("bfloat16 inference is only supported on CUDA devices")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 inference is not supported by this CUDA device")
        return torch.bfloat16
    raise ValueError(
        "inference dtype must be one of: float32, float16, bfloat16, auto "
        f"(got {requested_dtype!r})"
    )


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


def _build_model_from_config(config: TrainConfig) -> SpatialModel:
    """Instantiate the inference model described by a resolved config."""
    return build_spatial_model(config.model)


def build_inference_session(
    config: TrainConfig,
    checkpoint: str | Path,
    device: str | None,
    weights_source: WeightsSource = "auto",
    compile_model: bool = False,
    compile_mode: str = "default",
    inference_dtype: InferenceDTypeName = "float32",
    vae_checkpoint_path: str | Path | None = None,
    vae_config_path: str | Path | None = None,
) -> InferenceSession:
    """Build, load, move, and optionally compile an inference model once."""
    run_device = torch.device(device) if device else _default_device()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=checkpoint,
        output_dir=config.output_dir,
    )
    legacy_vae = is_legacy_vae_model(config)
    if legacy_vae and vae_checkpoint_path is None:
        raise ValueError(
            "legacy_vae inference requires an EAR-VAE checkpoint. Pass "
            "vae_checkpoint_path or use an exported bundle containing vae/."
        )
    if vae_checkpoint_path is not None and not Path(vae_checkpoint_path).exists():
        raise FileNotFoundError(f"EAR-VAE checkpoint not found: {vae_checkpoint_path}")
    if vae_config_path is not None and not Path(vae_config_path).exists():
        raise FileNotFoundError(f"EAR-VAE config not found: {vae_config_path}")

    model = _build_model_from_config(config)
    used_weights_source = load_model_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        weights_source=weights_source,
    )
    run_dtype = _resolve_inference_dtype(
        requested_dtype=inference_dtype,
        device=run_device,
    )
    model = model.to(device=run_device, dtype=run_dtype)
    model.eval()
    vae: torch.nn.Module | None = None
    if legacy_vae:
        assert vae_checkpoint_path is not None
        vae = load_vae(
            vae_checkpoint_path=vae_checkpoint_path,
            config_path=vae_config_path,
            device=run_device,
            torch_dtype=torch.float32,
        )
    compiled = False
    resolved_compile_mode: str | None = None
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
        resolved_compile_mode = str(compile_mode).strip().lower()
        model = cast(
            SpatialModel,
            torch.compile(model, mode=resolved_compile_mode),
        )
        compiled = True

    return InferenceSession(
        config=config,
        model=model,
        checkpoint_path=checkpoint_path,
        run_device=run_device,
        used_weights_source=used_weights_source,
        compiled=compiled,
        compile_mode=resolved_compile_mode,
        run_dtype=run_dtype,
        vae=vae,
    )


@torch.inference_mode()
def run_inference_with_session(
    session: InferenceSession,
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
    show_progress: bool,
    normalize_peak: bool,
    mix_style: list[float] | dict[str, float] | None = None,
    mix_style_preset: str | None = None,
    encode_chunk_size_samples: int | None = None,
    encode_overlap_samples: int | None = None,
    decode_chunk_size_frames: int = 2048,
    decode_overlap_frames: int = 256,
    disable_chunked_decode: bool = False,
    sampling_order: SamplingOrder | str = "timestep_major",
) -> InferenceReport:
    """
    Run inference for one file using an already-loaded inference session.

    The output channel count is set by ``session.config.model.target_channels``.
    """
    config = session.config
    model = session.model
    run_device = session.run_device
    run_dtype = session.run_dtype
    checkpoint_path = session.checkpoint_path
    legacy_vae = is_legacy_vae_model(config)
    if legacy_vae and int(sample_rate) != LEGACY_VAE_SAMPLE_RATE:
        raise ValueError(
            "legacy_vae inference requires sample_rate=48000 because the "
            "EAR-VAE codec is fixed at 48 kHz."
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

    conditioning_audio = (
        audio.float()
        if legacy_vae
        else _prepare_conditioning_audio(
            audio.float(),
            cond_channels=int(config.model.cond_channels),
        )
    )
    amplitude_lift_gain = None
    amplitude_lift_log_gain_tensor: torch.Tensor | None = None
    amplitude_lift_mode = (
        str(getattr(config.data, "amplitude_lift_mode", "rms")).strip().lower()
    )
    _lift_scale = float(getattr(config.data, "amplitude_lift_scale", 3.0))
    _lift_clip = getattr(config.data, "amplitude_lift_clip_value", 4.0)
    _lift_power = float(getattr(config.data, "amplitude_lift_gain_power", 1.0))
    _lift_min = getattr(config.data, "amplitude_lift_gain_min_value", None)
    _lift_output_lufs = float(getattr(config.data, "amplitude_lift_output_lufs", -23.0))
    if bool(getattr(config.data, "amplitude_lift_enabled", False)):
        if legacy_vae:
            raise ValueError(
                "Amplitude lifting is waveform-only and cannot be used with "
                "legacy_vae latent inference."
            )
        lift_reference = (
            str(getattr(config.data, "amplitude_lift_reference", "source"))
            .strip()
            .lower()
        )
        if amplitude_lift_mode != "wavflow" and lift_reference != "source":
            raise ValueError(
                "Inference amplitude lifting requires data.amplitude_lift_reference="
                "'source' because target audio is unavailable at inference time."
            )
        if amplitude_lift_mode == "wavflow":
            conditioning_audio, amplitude_lift_gain = wavflow_source_transform(
                conditioning_audio,
                target_rms=float(
                    getattr(config.data, "amplitude_lift_target_rms", 0.33)
                ),
                scale=_lift_scale,
                peak_limit=float(
                    getattr(config.data, "amplitude_lift_peak_limit", 1.0)
                ),
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            )
            amplitude_lift_log_gain_tensor = torch.log(
                amplitude_lift_gain.clamp_min(
                    float(getattr(config.data, "amplitude_lift_eps", 1.0e-8))
                )
            ).to(run_device, dtype=run_dtype)
        else:
            amplitude_lift_gain = resolve_amplitude_lift_gain(
                conditioning_audio,
                mode=amplitude_lift_mode,
                target_rms=float(
                    getattr(config.data, "amplitude_lift_target_rms", 0.33)
                ),
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            )
            _lift_clip = None if amplitude_lift_mode == "scale" else _lift_clip
            conditioning_audio = apply_amplitude_lift(
                conditioning_audio,
                gain=amplitude_lift_gain,
                scale=_lift_scale,
                clip_value=_lift_clip,
                gain_power=_lift_power,
                gain_min_value=_lift_min,
            )
            amplitude_lift_log_gain_tensor = amplitude_lift_log_gain(
                amplitude_lift_gain,
                scale=_lift_scale,
                gain_clip_value=_lift_clip,
                gain_power=_lift_power,
                gain_min_value=_lift_min,
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            ).to(run_device, dtype=run_dtype)
    input_samples = int(conditioning_audio.shape[-1])
    if legacy_vae:
        if session.vae is None:
            raise RuntimeError("legacy_vae inference session is missing its VAE")
        if int(config.model.cond_channels) == 1:
            encoded = vae_encode(
                vae=session.vae,
                audio=conditioning_audio,
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
            if encoded.dim() != 2:
                raise ValueError(
                    "Expected EAR-VAE conditioning latent [D,T], "
                    f"got {tuple(encoded.shape)}"
                )
            cond_signal = encoded.unsqueeze(0).contiguous()
        else:
            cond_signal = encode_channels_independent(
                vae=session.vae,
                audio=_prepare_conditioning_audio(
                    conditioning_audio,
                    cond_channels=int(config.model.cond_channels),
                ),
                sample_rate=actual_sample_rate,
                use_sample=False,
                use_chunked_encode=True,
                chunk_size_samples=encode_chunk_size_samples,
                overlap_samples=encode_overlap_samples,
                offload_latent_to_cpu=False,
                show_progress=show_progress,
                device=run_device,
            )
        expected_latent_dim = int(
            getattr(config.model, "latent_dim", None) or config.model.patch_size
        )
        if int(cond_signal.shape[1]) != expected_latent_dim:
            raise ValueError(
                "EAR-VAE latent width does not match model.latent_dim: "
                f"{int(cond_signal.shape[1])} != {expected_latent_dim}"
            )
    else:
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
        else float(
            getattr(
                config.training,
                "window_seconds",
                config.data.segment_seconds,
            )
        )
    )
    if legacy_vae:
        latent_fps = getattr(config.data, "latent_fps", 50.0)
        patch_fps = (
            50.0 if str(latent_fps).strip().lower() == "auto" else float(latent_fps)
        )
    else:
        patch_fps = float(actual_sample_rate) / float(config.model.patch_size)
    # Training always presents a fixed-size padded window, including for short
    # clips. Ask the sampler for that same window size instead of shrinking the
    # model context to the input length; the selected sampler supplies the valid
    # mask and removes the padded tail during stitching.
    requested_chunk_frames = max(1, int(round(target_chunk_seconds * patch_fps)))
    chunk_frames, overlap_frames = resolve_chunk_frames(
        cond_signal_frames=max(int(cond_signal.shape[-1]), requested_chunk_frames),
        patch_fps=patch_fps,
        chunk_seconds=target_chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    resolved_solver = _resolve_inference_solver(requested_solver=solver)
    resolved_solver_steps = 64 if solver_steps is None else max(1, int(solver_steps))

    resolved_sampling_order = str(sampling_order).strip().lower().replace("-", "_")
    if resolved_sampling_order not in {"window_major", "timestep_major"}:
        raise ValueError("sampling_order must be 'window_major' or 'timestep_major'")
    sampler = (
        generate_spatial_signal_timestep_major
        if resolved_sampling_order == "timestep_major"
        else generate_spatial_signal
    )

    pred_signal = sampler(
        model=cast(SpatialModel, model),
        cond_signal=cond_signal.to(run_device, dtype=run_dtype),
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        solver=resolved_solver,
        solver_steps=resolved_solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        seed=seed,
        mix_style=(
            mix_style_tensor.to(run_device, dtype=run_dtype)
            if mix_style_tensor is not None
            else None
        ),
        amplitude_gain=amplitude_lift_log_gain_tensor,
        one_step=bool(getattr(config.training, "flow_one_step", False)),
        one_step_input=str(getattr(config.training, "flow_one_step_input", "zeros")),
    )

    if legacy_vae:
        assert session.vae is not None
        decoded = (
            decode_channels_independent(
                vae=session.vae,
                channel_latents=pred_signal.to(run_device, dtype=torch.float32),
                use_chunked_decode=not disable_chunked_decode,
                chunk_size_frames=decode_chunk_size_frames,
                overlap_frames=decode_overlap_frames,
                offload_wav_to_cpu=True,
                reduction="mean",
                show_progress=show_progress,
                device=run_device,
            )
            .float()
            .cpu()
        )
        decoded = decoded[:, :input_samples].contiguous()
    else:
        decoded = _unpatch_audio(pred_signal.cpu().float(), sample_count=input_samples)
    if amplitude_lift_gain is not None:
        if amplitude_lift_mode == "wavflow":
            decoded = undo_wavflow_output_lift(
                decoded,
                scale=_lift_scale,
                sample_rate=actual_sample_rate,
                target_lufs=_lift_output_lufs,
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            )
        else:
            decoded = undo_amplitude_lift(
                decoded,
                gain=amplitude_lift_gain.cpu(),
                scale=_lift_scale,
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
                clip_value=_lift_clip,
                gain_power=_lift_power,
                gain_min_value=_lift_min,
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
        "weights_source": session.used_weights_source,
        "patch_fps": float(patch_fps),
        "chunk_seconds": float(target_chunk_seconds),
        "chunk_frames": int(chunk_frames),
        "overlap_seconds": float(overlap_seconds),
        "overlap_frames": int(overlap_frames),
        "solver": resolved_solver,
        "solver_steps": int(resolved_solver_steps),
        "solver_rtol": float(solver_rtol),
        "solver_atol": float(solver_atol),
        "sampling_order": cast(SamplingOrder, resolved_sampling_order),
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
        "amplitude_lift_mode": str(getattr(config.data, "amplitude_lift_mode", "rms")),
        "amplitude_lift_reference": str(
            getattr(config.data, "amplitude_lift_reference", "source")
        ),
        "amplitude_lift_target_rms": float(
            getattr(config.data, "amplitude_lift_target_rms", 0.33)
        ),
        "amplitude_lift_scale": float(
            getattr(config.data, "amplitude_lift_scale", 3.0)
        ),
        "amplitude_lift_clip_value": getattr(
            config.data,
            "amplitude_lift_clip_value",
            4.0,
        ),
        "amplitude_lift_output_lufs": float(
            getattr(config.data, "amplitude_lift_output_lufs", -23.0)
        ),
        "amplitude_lift_gain": (
            float(amplitude_lift_gain.flatten()[0].item())
            if amplitude_lift_gain is not None
            else None
        ),
        "device": str(run_device),
        "compiled": bool(session.compiled),
        "compile_mode": session.compile_mode,
        "inference_dtype": str(run_dtype).replace("torch.", ""),
        "architecture": str(getattr(config.model, "architecture", "waveform")),
        "representation": "latent" if legacy_vae else "waveform",
    }
    return report


@torch.inference_mode()
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
    compile_model: bool = False,
    compile_mode: str = "default",
    inference_dtype: InferenceDTypeName = "float32",
    vae_checkpoint_path: str | Path | None = None,
    vae_config_path: str | Path | None = None,
    encode_chunk_size_samples: int | None = None,
    encode_overlap_samples: int | None = None,
    decode_chunk_size_frames: int = 2048,
    decode_overlap_frames: int = 256,
    disable_chunked_decode: bool = False,
    sampling_order: SamplingOrder | str = "timestep_major",
) -> InferenceReport:
    """
    Run end-to-end inference from input waveform to rendered multichannel WAV.

    This single-file convenience wrapper builds one inference session and then
    runs exactly one generation. Folder callers should build a session once and
    reuse it through :func:`run_inference_with_session`.
    """
    session = build_inference_session(
        config=config,
        checkpoint=checkpoint,
        device=device,
        weights_source=weights_source,
        compile_model=compile_model,
        compile_mode=compile_mode,
        inference_dtype=inference_dtype,
        vae_checkpoint_path=vae_checkpoint_path,
        vae_config_path=vae_config_path,
    )
    report: InferenceReport = run_inference_with_session(
        session=session,
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        solver=solver,
        solver_steps=solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        seed=seed,
        show_progress=show_progress,
        normalize_peak=normalize_peak,
        mix_style=mix_style,
        mix_style_preset=mix_style_preset,
        encode_chunk_size_samples=encode_chunk_size_samples,
        encode_overlap_samples=encode_overlap_samples,
        decode_chunk_size_frames=decode_chunk_size_frames,
        decode_overlap_frames=decode_overlap_frames,
        disable_chunked_decode=disable_chunked_decode,
        sampling_order=sampling_order,
    )
    return report
