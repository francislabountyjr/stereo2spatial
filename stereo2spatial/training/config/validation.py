"""Validation rules for :class:`TrainConfig`."""

from __future__ import annotations

from .types import TrainConfig
from .validators import (
    optional_str,
    validate_data_section,
    validate_model_section,
    validate_optimizer_section,
    validate_scheduler_and_precision,
    validate_training_aux_losses,
    validate_training_gan,
    validate_training_schedule,
    validate_validation_paths,
)

__all__ = ["optional_str", "validate_config"]


def validate_config(config: TrainConfig) -> None:
    """Validate semantic constraints for a fully parsed config object."""
    sequence_mode = config.training.sequence_mode.strip().lower()

    validate_data_section(config)
    validate_model_section(config)
    validate_training_schedule(config, sequence_mode=sequence_mode)
    validate_training_gan(config)
    validate_training_aux_losses(config)
    validate_optimizer_section(config)
    validate_scheduler_and_precision(config)
    validate_validation_paths(config)
