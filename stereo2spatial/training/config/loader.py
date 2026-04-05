"""Config loading and typed coercion from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .sections import (
    build_data_config,
    build_model_config,
    build_optimizer_config,
    build_scheduler_config,
    build_training_config,
    require_key,
)
from .types import (
    TrainConfig,
)
from .validation import validate_config


def _require_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required top-level mapping section."""
    section = require_key(raw, key)
    if not isinstance(section, dict):
        raise TypeError(f"Config section {key!r} must be a mapping.")
    return section


def load_config(config_path: str | Path) -> TrainConfig:
    """Load, coerce, and validate a YAML config into :class:`TrainConfig`."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level mapping in config: {path}")

    data_raw = _require_section(raw, "data")
    model_raw = _require_section(raw, "model")
    training_raw = _require_section(raw, "training")
    optimizer_raw = _require_section(raw, "optimizer")
    scheduler_raw = _require_section(raw, "scheduler")

    config = TrainConfig(
        seed=int(require_key(raw, "seed")),
        output_dir=str(require_key(raw, "output_dir")),
        data=build_data_config(data_raw),
        model=build_model_config(model_raw),
        training=build_training_config(training_raw, data_raw),
        optimizer=build_optimizer_config(optimizer_raw),
        scheduler=build_scheduler_config(scheduler_raw),
    )
    validate_config(config)
    return config
