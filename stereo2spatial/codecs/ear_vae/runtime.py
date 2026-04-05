"""Runtime helpers for EAR-VAE integration."""

from __future__ import annotations

import logging
import os
import subprocess

import torch

DeviceLike = str | torch.device

LOGGER = logging.getLogger(__name__)
DEBUG_MAX_CUDA_VRAM_ENV = "MAX_CUDA_VRAM"


def _device_type(device: DeviceLike) -> str:
    """Return normalized device backend name (for example ``cuda`` or ``cpu``)."""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def _empty_cache(device: DeviceLike) -> None:
    """Clear accelerator memory cache for CUDA, XPU, or MPS."""
    device_type = _device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif (
        device_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def get_gpu_memory_gb() -> float:
    """
    Return total accelerator memory in GB for CUDA/XPU/MPS, else 0.

    Set MAX_CUDA_VRAM to simulate VRAM during testing.
    """
    debug_vram = os.environ.get(DEBUG_MAX_CUDA_VRAM_ENV)
    if debug_vram is not None:
        try:
            simulated_gb = float(debug_vram)
            LOGGER.warning(
                "DEBUG MODE: Simulating GPU memory as %.1fGB via %s",
                simulated_gb,
                DEBUG_MAX_CUDA_VRAM_ENV,
            )
            return simulated_gb
        except ValueError:
            LOGGER.warning(
                "Invalid %s value: %s; ignoring",
                DEBUG_MAX_CUDA_VRAM_ENV,
                debug_vram,
            )

    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_properties(0).total_memory / (1024**3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                total_system_bytes = int(result.stdout.strip())
                return (total_system_bytes / (1024**3)) * 0.75
            except Exception:
                return 8.0
        return 0.0
    except Exception as error:
        LOGGER.warning("Failed to detect GPU memory: %s", error)
        return 0.0
