"""Validation rules for optional validation dataset/generation paths."""

from __future__ import annotations

from stereo2spatial.modeling.factory import is_legacy_vae_model

from ..types import TrainConfig
from .common import optional_str

_VALIDATION_GENERATION_SOLVERS = {
    "auto",
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "res6s",
    "res_6s",
    "midpoint",
    "midpoint_rk2",
    "midpoint-rk2",
    "rk2",
    "rk4",
    "explicit_adams",
    "implicit_adams",
}


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
        if (
            is_legacy_vae_model(config)
            and optional_str(config.training.validation_generation_vae_checkpoint_path)
            is None
        ):
            raise ValueError(
                "training.validation_generation_vae_checkpoint_path is required "
                "for legacy_vae generation validation"
            )
        if (
            bool(getattr(config.data, "amplitude_lift_enabled", False))
            and str(getattr(config.data, "amplitude_lift_mode", "rms")).strip().lower()
            != "wavflow"
            and str(getattr(config.data, "amplitude_lift_reference", "source"))
            .strip()
            .lower()
            != "source"
        ):
            raise ValueError(
                "training.run_validation_generations=true requires "
                "data.amplitude_lift_reference='source' when amplitude lifting is "
                "enabled because generation inputs have no target signal."
            )
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

        solver = str(config.training.validation_generation_solver).strip().lower()
        if solver not in _VALIDATION_GENERATION_SOLVERS:
            raise ValueError(
                "training.validation_generation_solver must be one of: "
                + ", ".join(sorted(_VALIDATION_GENERATION_SOLVERS))
            )
        if config.training.validation_generation_solver_steps <= 0:
            raise ValueError(
                "training.validation_generation_solver_steps must be > 0 "
                "when run_validation_generations=true"
            )
        if config.training.validation_generation_solver_rtol <= 0:
            raise ValueError(
                "training.validation_generation_solver_rtol must be > 0 "
                "when run_validation_generations=true"
            )
        if config.training.validation_generation_solver_atol <= 0:
            raise ValueError(
                "training.validation_generation_solver_atol must be > 0 "
                "when run_validation_generations=true"
            )
        if (
            config.training.validation_generation_chunk_seconds is not None
            and config.training.validation_generation_chunk_seconds <= 0
        ):
            raise ValueError(
                "training.validation_generation_chunk_seconds must be > 0 when provided"
            )
        if config.training.validation_generation_overlap_seconds < 0:
            raise ValueError(
                "training.validation_generation_overlap_seconds must be >= 0"
            )
