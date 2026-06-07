"""Shared checkpoint-state helpers used by both train and infer paths."""

from __future__ import annotations

from pathlib import Path

import torch


def adapt_state_dict_keys_for_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip common wrapper prefixes until keys match the target model."""
    expected_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    if loaded_keys == expected_keys:
        return state_dict

    prefixes = ("module.", "_orig_mod.")
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

    return load_safetensors_file(str(checkpoint_path), device="cpu")


def load_safetensors_state_dict_from_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Load ``model.safetensors`` from a checkpoint directory when present."""
    model_safetensors_path = checkpoint_path / "model.safetensors"
    if not model_safetensors_path.exists():
        return None
    return load_safetensors_state_dict_file(model_safetensors_path)
