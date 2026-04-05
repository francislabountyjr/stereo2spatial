"""Helpers for exporting and consuming inference-ready model bundles."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

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
DEFAULT_EAR_VAE_ROOT = Path(r"E:\Python\EAR_VAE")
DEFAULT_EAR_VAE_CONFIG_PATH = DEFAULT_EAR_VAE_ROOT / "config" / "ear_vae_v2.json"
DEFAULT_EAR_VAE_CHECKPOINT_PATH = (
    DEFAULT_EAR_VAE_ROOT / "pretrained_weight" / "ear_vae_v2_48k.pyt"
)
DEFAULT_BUNDLE_CHUNK_SECONDS = 10.0
DEFAULT_BUNDLE_OVERLAP_SECONDS = 2.0

DEFAULT_CHANNEL_ORDER_7_1_4 = [
    "FL",
    "FR",
    "FC",
    "LFE",
    "BL",
    "BR",
    "SL",
    "SR",
    "TFL",
    "TFR",
    "TBL",
    "TBR",
]

_KNOWN_STATE_DICT_PREFIXES = ("_orig_mod.", "module.")
_EMA_FILENAME_PATTERN = re.compile(r"^custom_checkpoint_\d+\.pkl$")
_STEP_DIR_PATTERN = re.compile(r"^step_(\d+)$")
_CHANNEL_MASK_BY_ORDER = {
    tuple(DEFAULT_CHANNEL_ORDER_7_1_4): 0x2D63F,
}


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
            "Unsupported checkpoint payload type: "
            f"{type(payload)} ({checkpoint_path})"
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
    return _CHANNEL_MASK_BY_ORDER.get(tuple(channel_order))


def _load_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def is_inference_bundle_payload(payload: dict[str, Any]) -> bool:
    if payload.get("model_type") == "spatial_dit":
        return True
    return payload.get("bundle_kind") == "stereo2spatial_inference_bundle"


def load_inference_bundle_payload(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(Path(path))
    if not is_inference_bundle_payload(payload):
        raise ValueError(f"Config is not an inference bundle: {path}")
    return payload


def resolve_bundle_vae_paths(
    checkpoint: str | Path,
) -> tuple[Path | None, Path | None]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        bundle_root = checkpoint_path.parent
    else:
        bundle_root = checkpoint_path
    checkpoint_candidate = (
        bundle_root / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME
    )
    config_candidate = (
        bundle_root / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_CONFIG_FILENAME
    )
    resolved_checkpoint = (
        checkpoint_candidate.resolve() if checkpoint_candidate.exists() else None
    )
    resolved_config = config_candidate.resolve() if config_candidate.exists() else None
    return resolved_checkpoint, resolved_config


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
    if model_raw is None:
        model_raw = payload
    if audio_raw is None:
        audio_raw = payload
    if not isinstance(model_raw, dict):
        raise TypeError("bundle config is missing model fields")
    if not isinstance(audio_raw, dict):
        raise TypeError("bundle config is missing audio fields")

    model = ModelConfig(
        target_channels=int(model_raw["target_channels"]),
        cond_channels=int(model_raw["cond_channels"]),
        latent_dim=int(model_raw["latent_dim"]),
        hidden_dim=int(model_raw["hidden_dim"]),
        num_layers=int(model_raw["num_layers"]),
        num_heads=int(model_raw["num_heads"]),
        mlp_ratio=float(model_raw["mlp_ratio"]),
        dropout=float(model_raw.get("dropout", 0.0)),
        timestep_embed_dim=int(model_raw["timestep_embed_dim"]),
        timestep_scale=float(model_raw["timestep_scale"]),
        max_period=float(model_raw["max_period"]),
        num_memory_tokens=int(model_raw.get("num_memory_tokens", 0)),
    )

    data = DataConfig(
        dataset_root="",
        manifest_path="",
        sample_artifact_mode="bundle",
        segment_seconds=DEFAULT_BUNDLE_CHUNK_SECONDS,
        sequence_seconds=DEFAULT_BUNDLE_CHUNK_SECONDS,
        stride_seconds=DEFAULT_BUNDLE_CHUNK_SECONDS,
        latent_fps=audio_raw["latent_fps"],
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
        window_seconds=DEFAULT_BUNDLE_CHUNK_SECONDS,
        overlap_seconds=DEFAULT_BUNDLE_OVERLAP_SECONDS,
        sequence_seconds_choices=[DEFAULT_BUNDLE_CHUNK_SECONDS],
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
        validation_generation_seed=0,
        validation_generation_input_path=None,
        validation_generation_output_path=None,
        validation_generation_vae_checkpoint_path=None,
        validation_generation_vae_config_path=None,
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
    )
    scheduler = SchedulerConfig(
        type="cosine",
        warmup_steps=0,
        min_lr=0.0,
    )

    return TrainConfig(
        seed=0,
        output_dir=str(bundle_root) if bundle_root is not None else "",
        data=data,
        model=model,
        training=training,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _build_runtime_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_channels": int(model_config["target_channels"]),
        "cond_channels": int(model_config["cond_channels"]),
        "latent_dim": int(model_config["latent_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "num_layers": int(model_config["num_layers"]),
        "num_heads": int(model_config["num_heads"]),
        "mlp_ratio": float(model_config["mlp_ratio"]),
        "timestep_embed_dim": int(model_config["timestep_embed_dim"]),
        "timestep_scale": float(model_config["timestep_scale"]),
        "max_period": float(model_config["max_period"]),
        "num_memory_tokens": int(model_config.get("num_memory_tokens", 0)),
    }


def _build_runtime_config(
    *,
    model_config: dict[str, Any],
    channel_layout_name: str,
    channel_order: list[str],
    sample_rate: int,
    latent_fps: float | str,
) -> dict[str, Any]:
    return {
        "model_type": "spatial_dit",
        "architectures": ["SpatialDiT"],
        "sample_rate": int(sample_rate),
        "latent_fps": latent_fps,
        "channel_layout": channel_layout_name,
        "channel_order": channel_order,
        **model_config,
    }


def _resolve_export_vae_source_paths(
    *,
    include_vae: bool,
    ear_vae_root: str | Path | None,
    vae_checkpoint_path: str | Path | None,
    vae_config_path: str | Path | None,
) -> tuple[Path | None, Path | None]:
    if not include_vae:
        return None, None

    if vae_checkpoint_path is not None:
        resolved_checkpoint = Path(vae_checkpoint_path).resolve()
    else:
        root = (
            Path(ear_vae_root).resolve()
            if ear_vae_root is not None
            else DEFAULT_EAR_VAE_ROOT.resolve()
        )
        resolved_checkpoint = (root / "pretrained_weight" / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME).resolve()

    if vae_config_path is not None:
        resolved_config = Path(vae_config_path).resolve()
    else:
        root = (
            Path(ear_vae_root).resolve()
            if ear_vae_root is not None
            else DEFAULT_EAR_VAE_ROOT.resolve()
        )
        resolved_config = (root / "config" / EXPORT_BUNDLE_VAE_CONFIG_FILENAME).resolve()

    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"EAR-VAE checkpoint not found: {resolved_checkpoint}")
    if not resolved_config.exists():
        raise FileNotFoundError(f"EAR-VAE config not found: {resolved_config}")
    return resolved_checkpoint, resolved_config


def export_model_bundle(
    *,
    train_run_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    weights_source: str = "auto",
    channel_layout_name: str = "7.1.4",
    channel_order: list[str] | None = None,
    sample_rate: int = 48000,
    include_vae: bool = True,
    ear_vae_root: str | Path | None = None,
    vae_checkpoint_path: str | Path | None = None,
    vae_config_path: str | Path | None = None,
) -> ExportBundleResult:
    """Export a training checkpoint into an inference-ready model bundle."""
    run_dir = Path(train_run_dir).resolve()
    checkpoint_path = resolve_export_checkpoint_path(run_dir, checkpoint).resolve()
    resolved_config_path = run_dir / "resolved_config.json"
    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"Resolved training config not found: {resolved_config_path}"
        )

    with open(resolved_config_path, encoding="utf-8") as handle:
        training_config = json.load(handle)
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

    resolved_channel_order = list(
        DEFAULT_CHANNEL_ORDER_7_1_4 if channel_order is None else channel_order
    )
    target_channels = int(model_config["target_channels"])
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

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    weights_output_path = output_path / EXPORT_BUNDLE_WEIGHTS_FILENAME
    save_safetensors_file(normalized_state_dict, str(weights_output_path))

    exported_vae_checkpoint_source, exported_vae_config_source = (
        _resolve_export_vae_source_paths(
            include_vae=include_vae,
            ear_vae_root=ear_vae_root,
            vae_checkpoint_path=vae_checkpoint_path,
            vae_config_path=vae_config_path,
        )
    )
    bundled_vae_checkpoint_path: Path | None = None
    bundled_vae_config_path: Path | None = None
    vae_dir = output_path / EXPORT_BUNDLE_VAE_DIRNAME
    if vae_dir.exists():
        shutil.rmtree(vae_dir)
    if exported_vae_checkpoint_source is not None and exported_vae_config_source is not None:
        vae_dir.mkdir(parents=True, exist_ok=True)
        bundled_vae_checkpoint_path = (
            vae_dir / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME
        )
        bundled_vae_config_path = vae_dir / EXPORT_BUNDLE_VAE_CONFIG_FILENAME
        shutil.copy2(exported_vae_checkpoint_source, bundled_vae_checkpoint_path)
        shutil.copy2(exported_vae_config_source, bundled_vae_config_path)

    runtime_config_path = output_path / EXPORT_BUNDLE_CONFIG_FILENAME
    runtime_config = _build_runtime_config(
        model_config=_build_runtime_model_config(model_config),
        channel_layout_name=channel_layout_name,
        channel_order=resolved_channel_order,
        sample_rate=sample_rate,
        latent_fps=data_config["latent_fps"],
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
