"""Runtime helpers for model components."""

from __future__ import annotations

import torch


def is_compiling_runtime() -> bool:
    """Return True when executing inside a `torch.compile` capture/runtime context."""
    try:
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and hasattr(compiler, "is_compiling"):
            return bool(compiler.is_compiling())
    except Exception:
        pass
    try:
        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False

