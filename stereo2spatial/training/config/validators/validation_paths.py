"""Validation rules for optional validation dataset/generation paths."""

from __future__ import annotations

from ..types import TrainConfig
from .common import optional_str


def validate_validation_paths(config: TrainConfig) -> None:
    """Validate optional dataset and path requirements for validation modes."""
    if config.training.run_validation or config.training.run_validation_generations:
        if config.training.validation_steps <= 0:
            raise ValueError(
                "training.validation_steps must be > 0 when validation is enabled"
            )

    if config.training.run_validation:
        if optional_str(config.training.validation_dataset_root) is None:
            raise ValueError(
                "training.validation_dataset_root is required when run_validation=true"
            )
        if optional_str(config.training.validation_dataset_path) is None:
            raise ValueError(
                "training.validation_dataset_path is required when run_validation=true"
            )

    if config.training.run_validation_generations:
        if config.training.num_valid_generations <= 0:
            raise ValueError(
                "training.num_valid_generations must be > 0 when run_validation_generations=true"
            )
        if optional_str(config.training.validation_generation_input_path) is None:
            raise ValueError(
                "training.validation_generation_input_path is required "
                "when run_validation_generations=true"
            )
        if optional_str(config.training.validation_generation_output_path) is None:
            raise ValueError(
                "training.validation_generation_output_path is required "
                "when run_validation_generations=true"
            )
        if optional_str(config.training.validation_generation_vae_checkpoint_path) is None:
            raise ValueError(
                "training.validation_generation_vae_checkpoint_path is required "
                "when run_validation_generations=true"
            )
