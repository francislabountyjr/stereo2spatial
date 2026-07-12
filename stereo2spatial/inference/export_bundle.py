"""Helpers for exporting and consuming inference-ready model bundles."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from stereo2spatial.common.channel_layouts import (
    CHANNEL_ORDER_7_1_4,
    channel_labels_for_layout,
    channel_mask_for_order,
)
from stereo2spatial.common.checkpoints import try_detect_state_dict_architecture
from stereo2spatial.modeling.factory import (
    LEGACY_VAE_ARCHITECTURE,
    normalize_model_architecture,
)
from stereo2spatial.training.config import load_config
from stereo2spatial.training.config.types import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
)

EXPORT_BUNDLE_CONFIG_FILENAME = "config.json"
EXPORT_BUNDLE_WEIGHTS_FILENAME = "model.safetensors"
EXPORT_BUNDLE_VAE_DIRNAME = "vae"
EXPORT_BUNDLE_VAE_CONFIG_FILENAME = "ear_vae_v2.json"
EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME = "ear_vae_v2_48k.pyt"
EXPORT_BUNDLE_KIND = "stereo2spatial_inference_bundle"
EXPORT_BUNDLE_SCHEMA_VERSION = 1
DEFAULT_BUNDLE_CHUNK_SECONDS = 10.0
DEFAULT_BUNDLE_OVERLAP_SECONDS = 2.0
DEFAULT_BUNDLE_SOLVER = "auto"
DEFAULT_BUNDLE_SOLVER_STEPS = 64
DEFAULT_BUNDLE_SOLVER_RTOL = 1.0e-5
DEFAULT_BUNDLE_SOLVER_ATOL = 1.0e-5
DEFAULT_BUNDLE_SAMPLING_ORDER = "timestep_major"
DEFAULT_BUNDLE_SEED = 1337
LEGACY_VAE_SAMPLE_RATE = 48_000

DEFAULT_CHANNEL_ORDER_7_1_4 = CHANNEL_ORDER_7_1_4

_KNOWN_STATE_DICT_PREFIXES = ("_orig_mod.", "module.")
_EMA_FILENAME_PATTERN = re.compile(r"^custom_checkpoint_\d+\.pkl$")
_STEP_DIR_PATTERN = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class ExportBundleResult:
    """Summary of one exported inference bundle."""

    output_dir: Path
    checkpoint_path: Path
    weights_source: str
    config_path: Path
    vae_checkpoint_path: Path | None = None
    vae_config_path: Path | None = None


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        prefix_stripped = True
        while prefix_stripped:
            prefix_stripped = False
            for prefix in _KNOWN_STATE_DICT_PREFIXES:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    prefix_stripped = True
        normalized[normalized_key] = value.detach().cpu().contiguous()
    return normalized


def _load_ema_state_dict_from_checkpoint_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    candidates = sorted(
        path
        for path in checkpoint_path.glob("custom_checkpoint_*.pkl")
        if _EMA_FILENAME_PATTERN.match(path.name)
    )
    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        state_dict = payload.get("model")
        if isinstance(state_dict, dict):
            return {
                str(key): value
                for key, value in state_dict.items()
                if isinstance(value, torch.Tensor)
            }
    return None


def _load_state_dict_from_checkpoint_path(
    checkpoint_path: Path,
    weights_source: str,
) -> tuple[dict[str, torch.Tensor], str]:
    source = str(weights_source).strip().lower()
    if source not in {"auto", "ema", "student"}:
        raise ValueError("weights_source must be one of: auto, ema, student")

    if checkpoint_path.is_dir():
        if source in {"auto", "ema"}:
            ema_state = _load_ema_state_dict_from_checkpoint_dir(checkpoint_path)
            if ema_state is not None:
                return ema_state, "ema"
            if source == "ema":
                raise FileNotFoundError(
                    "Requested EMA weights but no EMA payload was found under "
                    f"{checkpoint_path}"
                )

        state_path = checkpoint_path / EXPORT_BUNDLE_WEIGHTS_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(
                f"Expected generator weights at {state_path}, but the file was missing."
            )
        state_dict = load_safetensors_file(str(state_path), device="cpu")
        return dict(state_dict), "student"

    if checkpoint_path.suffix.lower() == ".safetensors":
        state_dict = load_safetensors_file(str(checkpoint_path), device="cpu")
        return dict(state_dict), "student"

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        maybe_state_dict = payload["model_state_dict"]
    else:
        maybe_state_dict = payload
    if not isinstance(maybe_state_dict, dict):
        raise TypeError(
            f"Unsupported checkpoint payload type: {type(payload)} ({checkpoint_path})"
        )
    state_dict = {
        str(key): value
        for key, value in maybe_state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    return state_dict, "student"


def resolve_export_checkpoint_path(
    train_run_dir: str | Path,
    checkpoint: str | Path,
) -> Path:
    """Resolve ``latest`` or an explicit step/path for bundle export."""
    run_dir = Path(train_run_dir)
    checkpoint_value = str(checkpoint).strip()
    if not checkpoint_value:
        raise ValueError("checkpoint cannot be empty")

    if checkpoint_value.lower() == "latest":
        checkpoint_root = run_dir / "checkpoints"
        candidates = sorted(
            path for path in checkpoint_root.glob("step_*") if path.is_dir()
        )
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint directories were found under {checkpoint_root}"
            )
        return candidates[-1]

    direct_path = Path(checkpoint_value)
    if direct_path.exists():
        return direct_path

    relative_to_run = run_dir / checkpoint_value
    if relative_to_run.exists():
        return relative_to_run

    relative_to_checkpoint_root = run_dir / "checkpoints" / checkpoint_value
    if relative_to_checkpoint_root.exists():
        return relative_to_checkpoint_root

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_value}")


def resolve_inference_config_path(checkpoint: str | Path) -> Path | None:
    """Return a config path implied by a bundle/checkpoint path, when possible."""
    checkpoint_value = str(checkpoint).strip()
    if not checkpoint_value or checkpoint_value.lower() == "latest":
        return None

    path = Path(checkpoint_value)
    if not path.exists():
        return None

    candidates: list[Path] = []
    if path.is_file():
        if path.name == EXPORT_BUNDLE_CONFIG_FILENAME:
            candidates.append(path)
        candidates.append(path.parent / EXPORT_BUNDLE_CONFIG_FILENAME)
        if _STEP_DIR_PATTERN.match(path.parent.name):
            candidates.append(path.parent.parent.parent / "resolved_config.json")
    else:
        candidates.append(path / EXPORT_BUNDLE_CONFIG_FILENAME)
        candidates.append(path / "resolved_config.json")
        if _STEP_DIR_PATTERN.match(path.name):
            candidates.append(path.parent.parent / "resolved_config.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_channel_mask(channel_order: list[str]) -> int | None:
    return channel_mask_for_order(channel_order)


def _load_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def is_inference_bundle_payload(payload: dict[str, Any]) -> bool:
    if payload.get("model_type") == "spatial_dit":
        return True
    return payload.get("bundle_kind") == EXPORT_BUNDLE_KIND


def load_inference_bundle_payload(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(Path(path))
    if not is_inference_bundle_payload(payload):
        raise ValueError(f"Config is not an inference bundle: {path}")
    return payload


def resolve_bundle_vae_paths(
    checkpoint: str | Path,
) -> tuple[Path | None, Path | None]:
    """Return bundled EAR-VAE weight/config paths when present."""
    checkpoint_path = Path(checkpoint)
    bundle_root = (
        checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    )
    weights = (
        bundle_root / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME
    )
    config = bundle_root / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_CONFIG_FILENAME
    return (
        weights.resolve() if weights.exists() else None,
        config.resolve() if config.exists() else None,
    )


def build_train_config_from_bundle_payload(
    payload: dict[str, Any],
    *,
    bundle_root: Path | None = None,
) -> TrainConfig:
    """Build a minimal runtime-compatible config from bundle metadata."""
    if not is_inference_bundle_payload(payload):
        raise ValueError("payload is not an inference bundle config")

    model_raw = payload.get("model")
    audio_raw = payload.get("audio")
    data_raw = payload.get("data")
    if model_raw is None:
        model_raw = payload
    if audio_raw is None:
        audio_raw = payload
    if data_raw is None:
        data_raw = payload
    if not isinstance(model_raw, dict):
        raise TypeError("bundle config is missing model fields")
    if not isinstance(audio_raw, dict):
        raise TypeError("bundle config is missing audio fields")
    if not isinstance(data_raw, dict):
        raise TypeError("bundle config has invalid data fields")
    inference_raw = payload.get("inference", {})
    if not isinstance(inference_raw, dict):
        raise TypeError("bundle config has invalid inference fields")

    architecture_raw = model_raw.get("architecture", model_raw.get("model_variant"))
    if architecture_raw is None:
        architecture_raw = (
            LEGACY_VAE_ARCHITECTURE
            if model_raw.get("latent_dim") is not None
            and model_raw.get("patch_size") is None
            else "waveform"
        )
    architecture = normalize_model_architecture(architecture_raw)
    feature_size = (
        model_raw.get("latent_dim")
        if architecture == LEGACY_VAE_ARCHITECTURE
        else model_raw.get("patch_size")
    )
    if feature_size is None:
        raise KeyError(
            "legacy bundle is missing latent_dim"
            if architecture == LEGACY_VAE_ARCHITECTURE
            else "waveform bundle is missing patch_size"
        )
    resolved_feature_size = int(feature_size)
    model = ModelConfig(
        target_channels=int(model_raw["target_channels"]),
        cond_channels=int(model_raw["cond_channels"]),
        patch_size=resolved_feature_size,
        hidden_dim=int(model_raw["hidden_dim"]),
        num_layers=int(model_raw["num_layers"]),
        num_heads=int(model_raw["num_heads"]),
        mlp_ratio=float(model_raw["mlp_ratio"]),
        dropout=float(model_raw.get("dropout", 0.0)),
        timestep_embed_dim=int(model_raw["timestep_embed_dim"]),
        timestep_scale=float(model_raw["timestep_scale"]),
        max_period=float(model_raw["max_period"]),
        num_memory_tokens=int(model_raw.get("num_memory_tokens", 0)),
        mix_style_dim=int(model_raw.get("mix_style_dim", 0)),
        amplitude_gain_conditioning=bool(
            model_raw.get("amplitude_gain_conditioning", False)
        ),
        waveform_level_depth=int(model_raw.get("waveform_level_depth", 0)),
        waveform_micro_patch_size=int(model_raw.get("waveform_micro_patch_size", 16)),
        waveform_hidden_dim=int(model_raw.get("waveform_hidden_dim", 16)),
        waveform_num_heads=(
            int(model_raw["waveform_num_heads"])
            if model_raw.get("waveform_num_heads") is not None
            else None
        ),
        waveform_mlp_ratio=float(model_raw.get("waveform_mlp_ratio", 2.0)),
        final_output_kernel_size=int(model_raw.get("final_output_kernel_size", 7)),
        final_output_zero_init=bool(model_raw.get("final_output_zero_init", False)),
        rope_enabled=bool(model_raw.get("rope_enabled", True)),
        rope_theta=float(model_raw.get("rope_theta", 10000.0)),
        activation_checkpointing=bool(model_raw.get("activation_checkpointing", False)),
        architecture=architecture,
        latent_dim=(
            int(
                resolved_feature_size
                if model_raw.get("latent_dim") is None
                else model_raw["latent_dim"]
            )
            if architecture == LEGACY_VAE_ARCHITECTURE
            else None
        ),
    )

    latent_fps_raw = audio_raw.get("latent_fps", data_raw.get("latent_fps", 50.0))
    if latent_fps_raw is None:
        latent_fps_raw = 50.0
    resolved_latent_fps: float | str = (
        str(latent_fps_raw)
        if isinstance(latent_fps_raw, str)
        else float(latent_fps_raw)
    )

    runtime_chunk_seconds = float(
        inference_raw.get(
            "chunk_seconds",
            payload.get("chunk_seconds", DEFAULT_BUNDLE_CHUNK_SECONDS),
        )
    )
    runtime_overlap_seconds = float(
        inference_raw.get(
            "overlap_seconds",
            payload.get("overlap_seconds", DEFAULT_BUNDLE_OVERLAP_SECONDS),
        )
    )
    runtime_solver = str(
        inference_raw.get("solver", payload.get("solver", DEFAULT_BUNDLE_SOLVER))
    )
    runtime_solver_steps = int(
        inference_raw.get(
            "solver_steps",
            payload.get("solver_steps", DEFAULT_BUNDLE_SOLVER_STEPS),
        )
    )
    runtime_solver_rtol = float(
        inference_raw.get(
            "solver_rtol",
            payload.get("solver_rtol", DEFAULT_BUNDLE_SOLVER_RTOL),
        )
    )
    runtime_solver_atol = float(
        inference_raw.get(
            "solver_atol",
            payload.get("solver_atol", DEFAULT_BUNDLE_SOLVER_ATOL),
        )
    )
    runtime_seed = int(
        inference_raw.get("seed", payload.get("seed", DEFAULT_BUNDLE_SEED))
    )
    runtime_flow_one_step = bool(
        inference_raw.get(
            "flow_one_step",
            payload.get("flow_one_step", False),
        )
    )
    runtime_flow_one_step_input = str(
        inference_raw.get(
            "flow_one_step_input",
            payload.get("flow_one_step_input", "zeros"),
        )
    )

    data = DataConfig(
        dataset_root="",
        manifest_path="",
        sample_artifact_mode="bundle",
        segment_seconds=runtime_chunk_seconds,
        sequence_seconds=runtime_chunk_seconds,
        stride_seconds=runtime_chunk_seconds,
        sample_rate=int(audio_raw["sample_rate"]),
        training_sample_rate=(
            None
            if data_raw.get("training_sample_rate") is None
            else int(data_raw["training_sample_rate"])
        ),
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        batch_size=1,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        amplitude_lift_enabled=bool(data_raw.get("amplitude_lift_enabled", False)),
        amplitude_lift_mode=str(data_raw.get("amplitude_lift_mode", "rms")),
        amplitude_lift_reference=str(
            data_raw.get("amplitude_lift_reference", "source")
        ),
        amplitude_lift_target_rms=float(
            data_raw.get("amplitude_lift_target_rms", 0.33)
        ),
        amplitude_lift_scale=float(data_raw.get("amplitude_lift_scale", 3.0)),
        amplitude_lift_clip_value=(
            None
            if data_raw.get("amplitude_lift_clip_value", 4.0) is None
            else float(data_raw.get("amplitude_lift_clip_value", 4.0))
        ),
        amplitude_lift_gain_power=float(
            data_raw.get("amplitude_lift_gain_power", 1.0)
        ),
        amplitude_lift_gain_min_value=(
            None
            if data_raw.get("amplitude_lift_gain_min_value") is None
            else float(data_raw["amplitude_lift_gain_min_value"])
        ),
        amplitude_lift_waveform_clamp=bool(
            data_raw.get("amplitude_lift_waveform_clamp", True)
        ),
        amplitude_lift_peak_limit=float(data_raw.get("amplitude_lift_peak_limit", 1.0)),
        amplitude_lift_peak_rescale_min_rms=float(
            data_raw.get("amplitude_lift_peak_rescale_min_rms", 0.3)
        ),
        amplitude_lift_output_lufs=float(
            data_raw.get("amplitude_lift_output_lufs", -23.0)
        ),
        amplitude_lift_eps=float(data_raw.get("amplitude_lift_eps", 1.0e-8)),
        latent_fps=resolved_latent_fps,
    )

    training = TrainingConfig(
        max_steps=1,
        grad_accum_steps=1,
        mixed_precision="no",
        compile_model=False,
        compile_mode="default",
        resume_from_checkpoint=None,
        init_from_checkpoint=None,
        grad_clip_norm=1.0,
        log_every=1,
        checkpoint_every=1,
        max_checkpoints_to_keep=1,
        num_epochs_hint=1,
        window_seconds=runtime_chunk_seconds,
        overlap_seconds=runtime_overlap_seconds,
        sequence_seconds_choices=[runtime_chunk_seconds],
        randomize_sequence_per_batch=False,
        detach_memory=False,
        sequence_mode="full_song",
        tbptt_windows=0,
        full_song_max_seconds=None,
        require_batch_size_one_for_full_song=True,
        use_gan=False,
        gan_d_lr=1e-4,
        gan_d_beta1=0.0,
        gan_d_beta2=0.9,
        gan_d_base_channels=64,
        gan_d_num_layers=4,
        gan_d_fine_layers=3,
        gan_d_coarse_layers=4,
        gan_d_use_spectral_norm=True,
        gan_use_mask_channel=True,
        gan_ms_w_fine=0.5,
        gan_ms_w_coarse=0.5,
        gan_lambda_adv=0.0,
        gan_adv_warmup_steps=0,
        gan_r1_gamma=1.0,
        gan_r1_every=16,
        routing_kl_weight=0.0,
        routing_kl_temperature=1.0,
        routing_kl_eps=1e-6,
        corr_weight=0.0,
        corr_eps=1e-6,
        corr_offdiag_only=True,
        corr_use_correlation=True,
        run_validation=False,
        validation_dataset_root=None,
        validation_dataset_path=None,
        validation_steps=0,
        run_validation_generations=False,
        num_valid_generations=0,
        validation_generation_seed=runtime_seed,
        validation_generation_input_path=None,
        validation_generation_output_path=None,
        validation_generation_solver=runtime_solver,
        validation_generation_solver_steps=runtime_solver_steps,
        validation_generation_solver_rtol=runtime_solver_rtol,
        validation_generation_solver_atol=runtime_solver_atol,
        validation_generation_chunk_seconds=runtime_chunk_seconds,
        validation_generation_overlap_seconds=runtime_overlap_seconds,
        downmix_channel_order=(
            list(audio_raw["channel_order"])
            if isinstance(audio_raw.get("channel_order"), list)
            else None
        ),
        flow_one_step=runtime_flow_one_step,
        flow_one_step_input=runtime_flow_one_step_input,
    )

    optimizer = OptimizerConfig(
        type="adamw",
        lr=0.0,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        adamw_fused=False,
        adamw_foreach=False,
        muon_ns_steps=5,
        muon_nesterov=True,
    )
    scheduler = SchedulerConfig(
        type="cosine",
        warmup_steps=0,
        min_lr=0.0,
    )

    return TrainConfig(
        seed=runtime_seed,
        output_dir=str(bundle_root) if bundle_root is not None else "",
        data=data,
        model=model,
        training=training,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _build_runtime_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    architecture_raw = model_config.get(
        "architecture", model_config.get("model_variant")
    )
    if architecture_raw is None:
        architecture_raw = (
            LEGACY_VAE_ARCHITECTURE
            if model_config.get("latent_dim") is not None
            and model_config.get("patch_size") is None
            else "waveform"
        )
    architecture = normalize_model_architecture(architecture_raw)
    common = {
        "architecture": architecture,
        "target_channels": int(model_config["target_channels"]),
        "cond_channels": int(model_config["cond_channels"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "num_layers": int(model_config["num_layers"]),
        "num_heads": int(model_config["num_heads"]),
        "mlp_ratio": float(model_config["mlp_ratio"]),
        "dropout": float(model_config.get("dropout", 0.0)),
        "timestep_embed_dim": int(model_config["timestep_embed_dim"]),
        "timestep_scale": float(model_config["timestep_scale"]),
        "max_period": float(model_config["max_period"]),
        "num_memory_tokens": int(model_config.get("num_memory_tokens", 0)),
    }
    if architecture == LEGACY_VAE_ARCHITECTURE:
        latent_dim = model_config.get("latent_dim")
        if latent_dim is None:
            latent_dim = model_config.get("patch_size")
        if latent_dim is None:
            raise KeyError("legacy model config is missing latent_dim")
        common["latent_dim"] = int(latent_dim)
        return common
    return {
        **common,
        "patch_size": int(model_config["patch_size"]),
        "mix_style_dim": int(model_config.get("mix_style_dim", 0)),
        "amplitude_gain_conditioning": bool(
            model_config.get("amplitude_gain_conditioning", False)
        ),
        "waveform_level_depth": int(model_config.get("waveform_level_depth", 0)),
        "waveform_micro_patch_size": int(
            model_config.get("waveform_micro_patch_size", 16)
        ),
        "waveform_hidden_dim": int(model_config.get("waveform_hidden_dim", 16)),
        "waveform_num_heads": (
            int(model_config["waveform_num_heads"])
            if model_config.get("waveform_num_heads") is not None
            else None
        ),
        "waveform_mlp_ratio": float(model_config.get("waveform_mlp_ratio", 2.0)),
        "final_output_kernel_size": int(
            model_config.get("final_output_kernel_size", 7)
        ),
        "final_output_zero_init": bool(
            model_config.get("final_output_zero_init", False)
        ),
        "rope_enabled": bool(model_config.get("rope_enabled", True)),
        "rope_theta": float(model_config.get("rope_theta", 10000.0)),
        "activation_checkpointing": bool(
            model_config.get("activation_checkpointing", False)
        ),
    }


def _build_runtime_config(
    *,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    training_config: dict[str, Any],
    channel_layout_name: str,
    channel_order: list[str],
    sample_rate: int,
) -> dict[str, Any]:
    target_channels = int(model_config["target_channels"])
    runtime_config: dict[str, Any] = {
        "bundle_kind": EXPORT_BUNDLE_KIND,
        "bundle_schema_version": EXPORT_BUNDLE_SCHEMA_VERSION,
        "model_type": "spatial_dit",
        "architectures": [
            "LegacySpatialDiT"
            if model_config.get("architecture") == LEGACY_VAE_ARCHITECTURE
            else "SpatialDiT"
        ],
        "sample_rate": int(sample_rate),
        "output_kind": (
            "direct_binaural" if target_channels == 2 else "speaker_layout"
        ),
        "channel_layout": channel_layout_name,
        "channel_order": channel_order,
        "channel_mask": _resolve_channel_mask(channel_order),
        "amplitude_lift_enabled": bool(
            data_config.get("amplitude_lift_enabled", False)
        ),
        "training_sample_rate": data_config.get("training_sample_rate"),
        "amplitude_lift_mode": str(data_config.get("amplitude_lift_mode", "rms")),
        "amplitude_lift_reference": str(
            data_config.get("amplitude_lift_reference", "source")
        ),
        "amplitude_lift_target_rms": float(
            data_config.get("amplitude_lift_target_rms", 0.33)
        ),
        "amplitude_lift_scale": float(data_config.get("amplitude_lift_scale", 3.0)),
        "amplitude_lift_clip_value": data_config.get(
            "amplitude_lift_clip_value",
            4.0,
        ),
        "amplitude_lift_gain_power": float(
            data_config.get("amplitude_lift_gain_power", 1.0)
        ),
        "amplitude_lift_gain_min_value": data_config.get(
            "amplitude_lift_gain_min_value"
        ),
        "amplitude_lift_waveform_clamp": bool(
            data_config.get("amplitude_lift_waveform_clamp", True)
        ),
        "amplitude_lift_peak_limit": float(
            data_config.get("amplitude_lift_peak_limit", 1.0)
        ),
        "amplitude_lift_peak_rescale_min_rms": float(
            data_config.get("amplitude_lift_peak_rescale_min_rms", 0.3)
        ),
        "amplitude_lift_output_lufs": float(
            data_config.get("amplitude_lift_output_lufs", -23.0)
        ),
        "amplitude_lift_eps": float(data_config.get("amplitude_lift_eps", 1.0e-8)),
        "inference": {
            "sample_rate": int(sample_rate),
            "chunk_seconds": float(
                training_config.get(
                    "window_seconds",
                    data_config.get("segment_seconds", DEFAULT_BUNDLE_CHUNK_SECONDS),
                )
            ),
            "overlap_seconds": float(
                training_config.get("overlap_seconds", DEFAULT_BUNDLE_OVERLAP_SECONDS)
            ),
            "solver": str(
                training_config.get(
                    "validation_generation_solver", DEFAULT_BUNDLE_SOLVER
                )
            ),
            "solver_steps": int(
                training_config.get(
                    "validation_generation_solver_steps",
                    DEFAULT_BUNDLE_SOLVER_STEPS,
                )
            ),
            "solver_rtol": float(
                training_config.get(
                    "validation_generation_solver_rtol", DEFAULT_BUNDLE_SOLVER_RTOL
                )
            ),
            "solver_atol": float(
                training_config.get(
                    "validation_generation_solver_atol", DEFAULT_BUNDLE_SOLVER_ATOL
                )
            ),
            "sampling_order": DEFAULT_BUNDLE_SAMPLING_ORDER,
            "seed": int(
                training_config.get("validation_generation_seed", DEFAULT_BUNDLE_SEED)
            ),
            "flow_one_step": bool(training_config.get("flow_one_step", False)),
            "flow_one_step_input": str(
                training_config.get("flow_one_step_input", "zeros")
            ),
        },
        **model_config,
    }
    if model_config.get("architecture") == LEGACY_VAE_ARCHITECTURE:
        runtime_config["latent_fps"] = data_config.get("latent_fps", 50.0)
    return runtime_config


def _default_channel_layout_name(target_channels: int) -> str:
    if target_channels == 2:
        return "binaural"
    if target_channels == 6:
        return "5.1 rear"
    if target_channels == 12:
        return "7.1.4"
    return f"{target_channels}-channel"


def export_model_bundle(
    *,
    train_run_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    weights_source: str = "auto",
    channel_layout_name: str | None = None,
    channel_order: list[str] | None = None,
    sample_rate: int | None = None,
    include_vae: bool | None = None,
    vae_checkpoint_path: str | Path | None = None,
    vae_config_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> ExportBundleResult:
    """Export a training checkpoint into an inference-ready model bundle."""
    run_dir = Path(train_run_dir).resolve()
    checkpoint_path = resolve_export_checkpoint_path(run_dir, checkpoint).resolve()
    resolved_config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else run_dir / "resolved_config.json"
    )
    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"Resolved training config not found: {resolved_config_path}"
        )

    if resolved_config_path.suffix.lower() == ".json":
        with open(resolved_config_path, encoding="utf-8") as handle:
            training_config = json.load(handle)
    else:
        training_config = asdict(load_config(resolved_config_path))
    if not isinstance(training_config, dict):
        raise TypeError(
            f"Expected JSON object in resolved config: {resolved_config_path}"
        )

    model_config = training_config.get("model")
    data_config = training_config.get("data")
    if not isinstance(model_config, dict):
        raise TypeError("resolved_config.json is missing object section 'model'")
    if not isinstance(data_config, dict):
        raise TypeError("resolved_config.json is missing object section 'data'")
    raw_training_config = training_config.get("training")
    if not isinstance(raw_training_config, dict):
        raise TypeError("resolved_config.json is missing object section 'training'")

    target_channels = int(model_config["target_channels"])
    resolved_channel_layout_name = (
        _default_channel_layout_name(target_channels)
        if channel_layout_name is None
        else str(channel_layout_name).strip()
    )
    if not resolved_channel_layout_name:
        raise ValueError("channel_layout_name cannot be empty")
    resolved_channel_order = list(
        channel_labels_for_layout(resolved_channel_layout_name, target_channels)
        if channel_order is None
        else channel_order
    )
    if len(resolved_channel_order) != target_channels:
        raise ValueError(
            "channel_order length must match model.target_channels "
            f"({len(resolved_channel_order)} != {target_channels})"
        )

    state_dict, resolved_weights_source = _load_state_dict_from_checkpoint_path(
        checkpoint_path,
        weights_source=weights_source,
    )
    normalized_state_dict = _normalize_state_dict_keys(state_dict)

    runtime_model_config = _build_runtime_model_config(model_config)
    is_legacy = runtime_model_config["architecture"] == LEGACY_VAE_ARCHITECTURE
    configured_sample_rate = data_config.get("training_sample_rate")
    if configured_sample_rate is None:
        configured_sample_rate = data_config.get("sample_rate", 48_000)
    resolved_sample_rate = (
        int(configured_sample_rate) if sample_rate is None else int(sample_rate)
    )
    if is_legacy:
        if sample_rate is not None and resolved_sample_rate != LEGACY_VAE_SAMPLE_RATE:
            raise ValueError(
                "legacy_vae bundles require a 48000 Hz sample rate; "
                f"got {resolved_sample_rate}."
            )
        resolved_sample_rate = LEGACY_VAE_SAMPLE_RATE
    if resolved_sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    checkpoint_architecture = try_detect_state_dict_architecture(normalized_state_dict)
    if (
        checkpoint_architecture is not None
        and checkpoint_architecture != runtime_model_config["architecture"]
    ):
        raise ValueError(
            "Checkpoint architecture mismatch during export: "
            f"checkpoint={checkpoint_architecture} "
            f"config={runtime_model_config['architecture']}."
        )
    resolved_include_vae = is_legacy if include_vae is None else bool(include_vae)
    if resolved_include_vae and not is_legacy:
        raise ValueError("EAR-VAE assets can only be bundled for legacy_vae models")
    if resolved_include_vae and (
        vae_checkpoint_path is None or vae_config_path is None
    ):
        raise ValueError(
            "Exporting a legacy_vae bundle requires vae_checkpoint_path and "
            "vae_config_path (or set include_vae=False)."
        )

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    weights_output_path = output_path / EXPORT_BUNDLE_WEIGHTS_FILENAME
    save_safetensors_file(normalized_state_dict, str(weights_output_path))

    bundled_vae_checkpoint_path: Path | None = None
    bundled_vae_config_path: Path | None = None
    vae_dir = output_path / EXPORT_BUNDLE_VAE_DIRNAME
    if vae_dir.exists() and not resolved_include_vae:
        shutil.rmtree(vae_dir)
    if resolved_include_vae:
        assert vae_checkpoint_path is not None and vae_config_path is not None
        source_weights = Path(vae_checkpoint_path).resolve()
        source_config = Path(vae_config_path).resolve()
        if not source_weights.exists():
            raise FileNotFoundError(f"EAR-VAE checkpoint not found: {source_weights}")
        if not source_config.exists():
            raise FileNotFoundError(f"EAR-VAE config not found: {source_config}")
        vae_dir.mkdir(parents=True, exist_ok=True)
        bundled_vae_checkpoint_path = vae_dir / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME
        bundled_vae_config_path = vae_dir / EXPORT_BUNDLE_VAE_CONFIG_FILENAME
        shutil.copy2(source_weights, bundled_vae_checkpoint_path)
        shutil.copy2(source_config, bundled_vae_config_path)

    runtime_config_path = output_path / EXPORT_BUNDLE_CONFIG_FILENAME
    runtime_config = _build_runtime_config(
        model_config=runtime_model_config,
        data_config=data_config,
        training_config=raw_training_config,
        channel_layout_name=resolved_channel_layout_name,
        channel_order=resolved_channel_order,
        sample_rate=resolved_sample_rate,
    )
    runtime_config_path.write_text(
        json.dumps(runtime_config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return ExportBundleResult(
        output_dir=output_path,
        checkpoint_path=checkpoint_path,
        weights_source=resolved_weights_source,
        config_path=runtime_config_path,
        vae_checkpoint_path=bundled_vae_checkpoint_path,
        vae_config_path=bundled_vae_config_path,
    )
