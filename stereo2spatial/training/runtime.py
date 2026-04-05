"""Runtime helpers shared by the training loop."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import LatentSongDataset

try:
    _DATALOADER_INIT_PARAMS = set(inspect.signature(DataLoader.__init__).parameters)
except (TypeError, ValueError):
    _DATALOADER_INIT_PARAMS = set()
_SUPPORTS_DATALOADER_IN_ORDER = "in_order" in _DATALOADER_INIT_PARAMS
_SUPPORTS_PIN_MEMORY_DEVICE = "pin_memory_device" in _DATALOADER_INIT_PARAMS


def _worker_init_fn(_: int) -> None:
    """Set deterministic low-thread worker behavior for dataloader subprocesses."""
    # Keep worker CPU usage predictable; avoids heavy thread oversubscription
    # when many workers do tensor slicing/conversion at once.
    torch.set_num_threads(1)


def _disable_inductor_cudagraphs_if_possible() -> None:
    """
    Best-effort: disable Inductor cudagraphs for this process.
    Needed because this trainer runs multiple forwards per step before backward.
    """
    try:
        from torch._inductor import config as inductor_config
    except Exception:
        return

    try:
        triton_cfg = getattr(inductor_config, "triton", None)
        if triton_cfg is not None:
            if hasattr(triton_cfg, "cudagraphs"):
                triton_cfg.cudagraphs = False
            # PyTorch 2.10 can still route through cudagraph_trees even if
            # cudagraphs=False; disable both to avoid buffer overwrite issues
            # when multiple forwards happen before backward.
            if hasattr(triton_cfg, "cudagraph_trees"):
                triton_cfg.cudagraph_trees = False
    except Exception:
        pass

    try:
        cuda_cfg = getattr(inductor_config, "cuda", None)
        if cuda_cfg is not None and hasattr(cuda_cfg, "cudagraphs"):
            cuda_cfg.cudagraphs = False
    except Exception:
        pass


def _create_dataloader(
    dataset: LatentSongDataset,
    config: TrainConfig,
    drop_last: bool | None = None,
    for_training: bool = False,
) -> DataLoader:
    """Create a dataloader with runtime-compatible worker and pin-memory settings."""
    num_workers = config.data.num_workers
    persistent_workers = config.data.persistent_workers and num_workers > 0
    if for_training and persistent_workers:
        # Training mutates dataset epoch schedule via dataset.set_epoch(...).
        # With persistent workers, worker-local dataset replicas go stale and can
        # desync from sampler indices, causing IndexError near epoch boundaries.
        persistent_workers = False
    resolved_drop_last = config.data.drop_last if drop_last is None else bool(drop_last)
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": config.data.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": resolved_drop_last,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(config.data.prefetch_factor)
        dataloader_kwargs["worker_init_fn"] = _worker_init_fn
    if (
        _SUPPORTS_PIN_MEMORY_DEVICE
        and config.data.pin_memory
        and torch.cuda.is_available()
    ):
        dataloader_kwargs["pin_memory_device"] = "cuda"
    if _SUPPORTS_DATALOADER_IN_ORDER and num_workers > 0:
        dataloader_kwargs["in_order"] = False

    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def _lr_for_step(step: int, config: TrainConfig) -> float:
    """Compute warmup + scheduler learning rate for the provided global step."""
    base_lr = config.optimizer.lr
    min_lr = config.scheduler.min_lr
    warmup_steps = config.scheduler.warmup_steps
    max_steps = max(config.training.max_steps, 1)
    scheduler_type = config.scheduler.type.lower()

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if scheduler_type == "constant":
        return base_lr

    # cosine
    progress_num = max(0, step - warmup_steps)
    progress_den = max(1, max_steps - warmup_steps)
    progress = min(1.0, float(progress_num) / float(progress_den))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply scalar learning rate to all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = lr
