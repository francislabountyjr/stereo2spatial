"""Checkpoint resolution and model-weight loading for inference."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from accelerate import load_checkpoint_in_model

from stereo2spatial.common.checkpoints import (
    adapt_state_dict_keys_for_model,
    load_safetensors_state_dict_file,
    load_safetensors_state_dict_from_dir,
)


def resolve_checkpoint_path(
    checkpoint: str | Path,
    output_dir: str | Path,
) -> Path:
    """Resolve explicit or ``latest`` checkpoint input into a concrete path."""
    checkpoint_value = str(checkpoint).strip()
    if not checkpoint_value:
        raise ValueError("checkpoint cannot be empty")

    if checkpoint_value.lower() != "latest":
        path = Path(checkpoint_value)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoint_root = Path(output_dir) / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"No checkpoint root found: {checkpoint_root}")
    candidates = sorted(
        path for path in checkpoint_root.glob("step_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_root}")
    return candidates[-1]


def _load_state_dict_from_checkpoint_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Return a safetensors model state-dict when present in checkpoint dir."""
    return load_safetensors_state_dict_from_dir(checkpoint_path)


def _load_ema_state_dict_from_checkpoint_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Return EMA model state from ``custom_checkpoint_*.pkl`` when available."""
    candidates = sorted(
        path
        for path in checkpoint_path.glob("custom_checkpoint_*.pkl")
        if re.match(r"^custom_checkpoint_\d+\.pkl$", path.name)
    )
    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        model_state = payload.get("model")
        if isinstance(model_state, dict):
            return model_state
    return None


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    weights_source: str = "auto",
) -> str:
    """
    Load model weights from checkpoint and return resolved source.

    Returns:
        ``"ema"`` when EMA weights were loaded, otherwise ``"student"``.
    """
    source = str(weights_source).strip().lower()
    if source not in {"auto", "ema", "student"}:
        raise ValueError("weights_source must be one of: auto, ema, student")

    if checkpoint_path.is_dir():
        if source in {"auto", "ema"}:
            ema_state = _load_ema_state_dict_from_checkpoint_dir(checkpoint_path)
            if ema_state is not None:
                ema_state = adapt_state_dict_keys_for_model(model, ema_state)
                model.load_state_dict(ema_state, strict=True)
                return "ema"
            if source == "ema":
                raise FileNotFoundError(
                    "Requested EMA inference weights, but no EMA checkpoint was found "
                    f"under: {checkpoint_path}"
                )

        state_dict = _load_state_dict_from_checkpoint_dir(checkpoint_path)
        if state_dict is not None:
            state_dict = adapt_state_dict_keys_for_model(model, state_dict)
            model.load_state_dict(state_dict, strict=True)
            return "student"
        load_checkpoint_in_model(
            model=model,
            checkpoint=str(checkpoint_path),
            strict=True,
        )
        return "student"

    if source == "ema":
        raise ValueError(
            "EMA inference weights are only available when loading from an "
            "Accelerate checkpoint directory."
        )
    if checkpoint_path.suffix.lower() == ".safetensors":
        state_dict = load_safetensors_state_dict_file(checkpoint_path)
        state_dict = adapt_state_dict_keys_for_model(model, state_dict)
        model.load_state_dict(state_dict, strict=True)
        return "student"
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(
            f"Unsupported checkpoint payload type: {type(payload)} ({checkpoint_path})"
        )
    state_dict = adapt_state_dict_keys_for_model(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    return "student"
