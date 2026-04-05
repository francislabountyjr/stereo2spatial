"""Shared checkpoint-state helpers used by both train and infer paths."""

from __future__ import annotations

from pathlib import Path

import torch


def adapt_state_dict_keys_for_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip common wrappers (for example ``module.``) when needed."""
    expected_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    if loaded_keys == expected_keys:
        return state_dict

    for prefix in ("_orig_mod.", "module."):
        if not any(key.startswith(prefix) for key in loaded_keys):
            continue
        adapted = {
            (key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }
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

    return load_safetensors_file(str(checkpoint_path), device="cpu")


def load_safetensors_state_dict_from_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Load ``model.safetensors`` from a checkpoint directory when present."""
    model_safetensors_path = checkpoint_path / "model.safetensors"
    if not model_safetensors_path.exists():
        return None
    return load_safetensors_state_dict_file(model_safetensors_path)
