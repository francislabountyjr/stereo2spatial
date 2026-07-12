"""Inference-only scaled-dot-product attention backend controls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Literal

import torch

SDPABackendName = Literal["auto", "flash", "efficient", "math"]


@contextmanager
def sdpa_backend_context(backend: SDPABackendName) -> Iterator[None]:
    """Temporarily constrain PyTorch SDPA backend selection for inference."""
    name = str(backend).strip().lower()
    if name == "auto":
        with nullcontext():
            yield
        return
    if not hasattr(torch.nn, "attention"):
        raise RuntimeError("This PyTorch build does not expose torch.nn.attention")
    attention = torch.nn.attention
    if not hasattr(attention, "sdpa_kernel") or not hasattr(attention, "SDPBackend"):
        raise RuntimeError("This PyTorch build does not expose SDPA backend controls")
    backend_enum = attention.SDPBackend
    mapping = {
        "flash": backend_enum.FLASH_ATTENTION,
        "efficient": backend_enum.EFFICIENT_ATTENTION,
        "math": backend_enum.MATH,
    }
    if name not in mapping:
        raise ValueError("SDPA backend must be one of: auto, flash, efficient, math")
    with attention.sdpa_kernel(mapping[name]):
        yield


__all__ = ["SDPABackendName", "sdpa_backend_context"]
