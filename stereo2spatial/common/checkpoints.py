"""Shared checkpoint-state helpers used by both train and infer paths."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

LEGACY_CHECKPOINT_ARCHITECTURE = "legacy_vae"
WAVEFORM_CHECKPOINT_ARCHITECTURE = "waveform"
_STATE_PREFIXES = ("module.", "_orig_mod.")


def _normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in _STATE_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def detect_state_dict_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    """Classify legacy-VAE vs waveform checkpoints from stable head signatures."""
    keys = {_normalize_checkpoint_key(key) for key in state_dict}
    has_legacy = "final_proj.weight" in keys and "final_norm.weight" in keys
    has_waveform = (
        "final_output.conv.weight" in keys
        and "final_output.adaLN_modulation.1.weight" in keys
    )
    if has_legacy == has_waveform:
        raise ValueError(
            "Unable to determine checkpoint architecture: expected exactly one of "
            "legacy final_proj/final_norm or waveform final_output signatures."
        )
    return (
        LEGACY_CHECKPOINT_ARCHITECTURE
        if has_legacy
        else WAVEFORM_CHECKPOINT_ARCHITECTURE
    )


def try_detect_state_dict_architecture(
    state_dict: dict[str, torch.Tensor],
) -> str | None:
    """Best-effort classifier used for generic test/helper modules."""
    keys = {_normalize_checkpoint_key(key) for key in state_dict}
    has_legacy = "final_proj.weight" in keys and "final_norm.weight" in keys
    has_waveform = (
        "final_output.conv.weight" in keys
        and "final_output.adaLN_modulation.1.weight" in keys
    )
    if has_legacy and has_waveform:
        raise ValueError(
            "Checkpoint contains both legacy and waveform architecture signatures."
        )
    if not has_legacy and not has_waveform:
        return None
    return (
        LEGACY_CHECKPOINT_ARCHITECTURE
        if has_legacy
        else WAVEFORM_CHECKPOINT_ARCHITECTURE
    )


def validate_state_dict_architecture(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> str:
    """Fail early with a clear error when config and checkpoint families differ."""
    checkpoint_architecture = try_detect_state_dict_architecture(state_dict)
    model_architecture = try_detect_state_dict_architecture(dict(model.state_dict()))
    if checkpoint_architecture is None or model_architecture is None:
        return checkpoint_architecture or model_architecture or "unknown"
    if checkpoint_architecture != model_architecture:
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"checkpoint={checkpoint_architecture} model={model_architecture}. "
            "Use the matching model.architecture/config (legacy_vae or waveform)."
        )
    return checkpoint_architecture


def adapt_state_dict_keys_for_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip common wrapper prefixes until keys match the target model."""
    expected_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    if loaded_keys == expected_keys:
        return state_dict

    prefixes = _STATE_PREFIXES
    adapted = state_dict
    for _ in range(4):
        changed = False
        next_adapted: dict[str, torch.Tensor] = {}
        for key, value in adapted.items():
            next_key = key
            for prefix in prefixes:
                if next_key.startswith(prefix):
                    next_key = next_key[len(prefix) :]
                    changed = True
                    break
            next_adapted[next_key] = value
        if not changed:
            break
        adapted = next_adapted
        if set(adapted.keys()) == expected_keys:
            return adapted

    return state_dict


def load_safetensors_state_dict_file(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """Load a state-dict from a single ``.safetensors`` file on CPU."""
    try:
        from safetensors.torch import load_file as load_safetensors_file
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Missing dependency: safetensors. Install with `pip install safetensors`."
        ) from error

    return cast(
        dict[str, torch.Tensor],
        load_safetensors_file(str(checkpoint_path), device="cpu"),
    )


def load_safetensors_state_dict_from_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Load ``model.safetensors`` from a checkpoint directory when present."""
    model_safetensors_path = checkpoint_path / "model.safetensors"
    if not model_safetensors_path.exists():
        return None
    return load_safetensors_state_dict_file(model_safetensors_path)
