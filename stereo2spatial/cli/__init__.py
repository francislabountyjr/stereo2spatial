"""Command-line entrypoints for stereo2spatial."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load CLI callables so ``python -m`` entrypoints stay warning-free."""
    if name in {"build_infer_parser", "infer_main"}:
        from . import infer

        return infer.build_parser if name == "build_infer_parser" else infer.main
    if name in {"build_train_parser", "train_main"}:
        from . import train

        return train.build_parser if name == "build_train_parser" else train.main
    raise AttributeError(name)

__all__ = [
    "build_infer_parser",
    "build_train_parser",
    "infer_main",
    "train_main",
]
