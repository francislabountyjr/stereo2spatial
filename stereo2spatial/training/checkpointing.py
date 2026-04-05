"""Checkpoint save/load and resume/init helpers for training."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator, load_checkpoint_in_model

from stereo2spatial.common.checkpoints import (
    adapt_state_dict_keys_for_model,
    load_safetensors_state_dict_file,
    load_safetensors_state_dict_from_dir,
)

TRAINER_STATE_FILENAME = "trainer_state.json"
GAN_DISCRIMINATOR_STATE_FILENAME = "discriminator_state.pt"
GAN_DISCRIMINATOR_OPTIMIZER_FILENAME = "discriminator_optimizer.pt"


def _checkpoint_root(output_dir: Path) -> Path:
    """Return the root directory that stores numbered training checkpoints."""
    return output_dir / "checkpoints"


def _checkpoint_dir_for_step(output_dir: Path, step: int) -> Path:
    """Return the checkpoint directory path for a specific global step."""
    return _checkpoint_root(output_dir) / f"step_{step:07d}"


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the most recent checkpoint directory under ``output_dir``."""
    checkpoint_root = _checkpoint_root(output_dir)
    if not checkpoint_root.exists():
        return None
    candidates = sorted(
        path for path in checkpoint_root.glob("step_*") if path.is_dir()
    )
    if not candidates:
        return None
    return candidates[-1]


def _save_trainer_state(
    checkpoint_dir: Path,
    global_step: int,
    epoch: int,
    batches_seen_in_epoch: int,
) -> None:
    """Persist scalar trainer progress metadata beside model/optimizer state."""
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batches_seen_in_epoch": int(batches_seen_in_epoch),
    }
    with open(checkpoint_dir / TRAINER_STATE_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _load_trainer_state(checkpoint_dir: Path) -> tuple[int, int, int]:
    """Load persisted trainer progress metadata from a checkpoint directory."""
    state_path = checkpoint_dir / TRAINER_STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing {TRAINER_STATE_FILENAME} in checkpoint: {checkpoint_dir}"
        )
    with open(state_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return (
        int(payload["global_step"]),
        int(payload["epoch"]),
        int(payload["batches_seen_in_epoch"]),
    )


def _cleanup_old_checkpoints(
    output_dir: Path,
    max_to_keep: int,
    accelerator: Accelerator,
) -> None:
    """Delete oldest checkpoint directories beyond the retention budget."""
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    checkpoint_root = _checkpoint_root(output_dir)
    if not checkpoint_root.exists():
        accelerator.wait_for_everyone()
        return
    checkpoints = sorted(
        path for path in checkpoint_root.glob("step_*") if path.is_dir()
    )
    extra = len(checkpoints) - max_to_keep
    for idx in range(max(0, extra)):
        shutil.rmtree(checkpoints[idx], ignore_errors=True)
    accelerator.wait_for_everyone()


def _save_checkpoint(
    output_dir: Path,
    accelerator: Accelerator,
    global_step: int,
    epoch: int,
    batches_seen_in_epoch: int,
    max_to_keep: int,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save full accelerator state and optional GAN state for one step."""
    checkpoint_dir = _checkpoint_dir_for_step(output_dir, global_step)
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        if discriminator is not None and discriminator_optimizer is not None:
            raw_discriminator = accelerator.unwrap_model(discriminator)
            torch.save(
                raw_discriminator.state_dict(),
                checkpoint_dir / GAN_DISCRIMINATOR_STATE_FILENAME,
            )
            torch.save(
                discriminator_optimizer.state_dict(),
                checkpoint_dir / GAN_DISCRIMINATOR_OPTIMIZER_FILENAME,
            )
        _save_trainer_state(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            epoch=epoch,
            batches_seen_in_epoch=batches_seen_in_epoch,
        )
    accelerator.wait_for_everyone()
    _cleanup_old_checkpoints(
        output_dir=output_dir,
        max_to_keep=max_to_keep,
        accelerator=accelerator,
    )
    return checkpoint_dir


def _load_gan_state_if_available(
    checkpoint_path: Path,
    discriminator: torch.nn.Module,
    discriminator_optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> bool:
    """Restore discriminator + optimizer GAN state when files are present."""
    discriminator_state_path = checkpoint_path / GAN_DISCRIMINATOR_STATE_FILENAME
    discriminator_optimizer_path = (
        checkpoint_path / GAN_DISCRIMINATOR_OPTIMIZER_FILENAME
    )
    if (not discriminator_state_path.exists()) or (
        not discriminator_optimizer_path.exists()
    ):
        return False

    disc_state = torch.load(discriminator_state_path, map_location="cpu")
    opt_state = torch.load(discriminator_optimizer_path, map_location="cpu")
    try:
        raw_discriminator = accelerator.unwrap_model(discriminator)
        raw_discriminator.load_state_dict(disc_state, strict=True)
        discriminator_optimizer.load_state_dict(opt_state)
    except Exception as error:
        accelerator.print(
            "[gan_checkpoint_warning] "
            f"Failed to restore discriminator state from {checkpoint_path}: {error}. "
            "Continuing with a freshly initialized discriminator."
        )
        return False
    return True


def _load_discriminator_weights_if_available(
    checkpoint_path: Path,
    discriminator: torch.nn.Module,
    accelerator: Accelerator,
) -> bool:
    """Load discriminator weights from a checkpoint directory or state file."""
    if checkpoint_path.is_dir():
        discriminator_state_path = checkpoint_path / GAN_DISCRIMINATOR_STATE_FILENAME
    elif checkpoint_path.name == GAN_DISCRIMINATOR_STATE_FILENAME:
        discriminator_state_path = checkpoint_path
    else:
        return False

    if not discriminator_state_path.exists():
        return False

    disc_state = torch.load(discriminator_state_path, map_location="cpu")
    try:
        raw_discriminator = accelerator.unwrap_model(discriminator)
        raw_discriminator.load_state_dict(disc_state, strict=True)
    except Exception as error:
        accelerator.print(
            "[gan_checkpoint_warning] "
            f"Failed to load discriminator weights from {discriminator_state_path}: {error}. "
            "Continuing with a freshly initialized discriminator."
        )
        return False
    return True


def _resolve_checkpoint_reference(
    *,
    checkpoint_value: str | None,
    output_dir: Path,
    field_name: str,
    missing_label: str,
) -> Path | None:
    """Resolve checkpoint strings (explicit path or ``latest``) into concrete paths."""
    if checkpoint_value is None:
        return None
    value = checkpoint_value.strip()
    if not value:
        return None
    if value.lower() == "latest":
        latest = _find_latest_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(
                f"{field_name}='latest' but no checkpoints found under {output_dir}"
            )
        return latest

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{missing_label} checkpoint not found: {path}")
    return path


def _resolve_resume_checkpoint(
    resume_from_checkpoint: str | None,
    output_dir: Path,
) -> Path | None:
    """Resolve ``training.resume_from_checkpoint`` into a concrete path."""
    return _resolve_checkpoint_reference(
        checkpoint_value=resume_from_checkpoint,
        output_dir=output_dir,
        field_name="resume_from_checkpoint",
        missing_label="Resume",
    )


def _resolve_model_init_checkpoint(
    init_from_checkpoint: str | None,
    output_dir: Path,
) -> Path | None:
    """Resolve ``training.init_from_checkpoint`` into a concrete path."""
    return _resolve_checkpoint_reference(
        checkpoint_value=init_from_checkpoint,
        output_dir=output_dir,
        field_name="init_from_checkpoint",
        missing_label="Init",
    )


def _load_state_dict_from_checkpoint_dir(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Load a model state dict from an Accelerate checkpoint directory, if present."""
    return load_safetensors_state_dict_from_dir(checkpoint_path)


def _checkpoint_has_ema_state(checkpoint_path: Path) -> bool:
    """Return whether an Accelerate checkpoint directory includes EMA payload."""
    if not checkpoint_path.is_dir():
        return False
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
            return True
    return False


def _load_model_weights_only(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load model weights from directory, safetensors file, or torch payload."""
    if checkpoint_path.is_dir():
        state_dict = _load_state_dict_from_checkpoint_dir(checkpoint_path)
        if state_dict is not None:
            state_dict = adapt_state_dict_keys_for_model(model, state_dict)
            model.load_state_dict(state_dict, strict=True)
            return
        load_checkpoint_in_model(
            model=model,
            checkpoint=str(checkpoint_path),
            strict=True,
        )
        return

    if checkpoint_path.suffix.lower() == ".safetensors":
        state_dict = load_safetensors_state_dict_file(checkpoint_path)
        state_dict = adapt_state_dict_keys_for_model(model, state_dict)
        model.load_state_dict(state_dict, strict=True)
        return

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


def _load_resume_state(
    accelerator: Accelerator,
    output_dir: Path,
    resume_from_checkpoint: str | None,
    resolved_checkpoint_path: Path | None = None,
) -> tuple[int, int, int, Path | None]:
    """Load full training resume state and return scalar progress metadata."""
    checkpoint_path = resolved_checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = _resolve_resume_checkpoint(
            resume_from_checkpoint=resume_from_checkpoint,
            output_dir=output_dir,
        )
    if checkpoint_path is None:
        return 0, 0, 0, None

    accelerator.print(f"Resuming from checkpoint: {checkpoint_path}")
    accelerator.load_state(str(checkpoint_path))
    global_step, epoch, batches_seen_in_epoch = _load_trainer_state(checkpoint_path)
    return global_step, epoch, batches_seen_in_epoch, checkpoint_path
