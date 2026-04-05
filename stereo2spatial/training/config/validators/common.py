"""Small shared utility validators."""

from __future__ import annotations

from typing import Any


def optional_str(value: Any) -> str | None:
    """Normalize optional string-like config values."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def require_positive(value: int | float, field_name: str) -> None:
    """Require ``value > 0`` for a config field."""
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def require_non_negative(value: int | float, field_name: str) -> None:
    """Require ``value >= 0`` for a config field."""
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def require_probability(value: float, field_name: str) -> None:
    """Require a probability value in the inclusive range ``[0, 1]``."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
