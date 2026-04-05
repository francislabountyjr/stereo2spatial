"""Validation rules for optimizer/scheduler/mixed-precision sections."""

from __future__ import annotations

from ..types import TrainConfig
from .common import require_non_negative, require_positive


def validate_optimizer_section(config: TrainConfig) -> None:
    """Validate optimizer family and its core numeric parameters."""
    optimizer_type = config.optimizer.type.strip().lower()
    if optimizer_type not in {"adamw", "adam"}:
        raise ValueError("optimizer.type must be one of: adamw, adam")
    require_positive(config.optimizer.lr, "optimizer.lr")
    require_positive(config.optimizer.eps, "optimizer.eps")
    if (
        optimizer_type == "adamw"
        and config.optimizer.adamw_fused
        and config.optimizer.adamw_foreach
    ):
        raise ValueError(
            "optimizer.adamw_fused and optimizer.adamw_foreach cannot both be true"
        )


def validate_scheduler_and_precision(config: TrainConfig) -> None:
    """Validate scheduler settings and mixed-precision mode selection."""
    require_non_negative(config.scheduler.warmup_steps, "scheduler.warmup_steps")

    scheduler_type = config.scheduler.type.lower()
    if scheduler_type not in {"cosine", "constant"}:
        raise ValueError("scheduler.type must be one of: cosine, constant")

    mixed_precision = config.training.mixed_precision.lower()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be one of: no, fp16, bf16")
