"""Exponential moving average (EMA) utilities for teacher-model training."""

from __future__ import annotations

import copy
from typing import Literal, cast

import torch

from stereo2spatial.common.checkpoints import adapt_state_dict_keys_for_model

EMAStorageDevice = Literal["accelerator", "cpu"]


def _normalize_ema_storage_device(device: str) -> EMAStorageDevice:
    normalized = str(device).strip().lower()
    if normalized not in {"accelerator", "cpu"}:
        raise ValueError(
            "EMA storage device must be one of: accelerator, cpu "
            f"(got {device!r})"
        )
    return cast(EMAStorageDevice, normalized)


class EMATeacher:
    """Maintain a frozen EMA copy of a student model for teacher forwards."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        *,
        storage_device: str = "accelerator",
        cpu_only: bool = False,
    ) -> None:
        self.decay = float(decay)
        self.storage_device: EMAStorageDevice = _normalize_ema_storage_device(
            storage_device
        )
        self.cpu_only = bool(cpu_only)
        if self.cpu_only:
            self.storage_device = "cpu"
        self.model = copy.deepcopy(model)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._accelerator_device: torch.device | None = None

    @torch.no_grad()
    def configure_runtime(
        self,
        *,
        accelerator_device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Configure where EMA should live during training updates."""
        self._accelerator_device = torch.device(accelerator_device)
        target_device = (
            self._accelerator_device
            if self.storage_device == "accelerator"
            else torch.device("cpu")
        )
        self.to(device=target_device, dtype=dtype)

    @torch.no_grad()
    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Move EMA teacher model to a target device/dtype."""
        if dtype is None:
            self.model.to(device=device, non_blocking=non_blocking)
            return
        self.model.to(device=device, dtype=dtype, non_blocking=non_blocking)

    @torch.no_grad()
    def pin_memory(self) -> None:
        """Pin EMA tensors when stored on CPU for faster non-blocking transfers."""
        if self.storage_device != "cpu" or self.cpu_only:
            return
        if torch.backends.mps.is_available():
            return

        for parameter in self.model.parameters():
            if parameter.device.type == "cpu":
                parameter.data = parameter.data.pin_memory()
        for buffer in self.model.buffers():
            if buffer.device.type == "cpu":
                buffer.data = buffer.data.pin_memory()

    @torch.no_grad()
    def copy_from(self, model: torch.nn.Module) -> None:
        """Hard-copy student weights into EMA weights."""
        model_state_dict = adapt_state_dict_keys_for_model(
            self.model,
            model.state_dict(),
        )
        self.model.load_state_dict(model_state_dict, strict=True)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """EMA-update teacher weights from current student weights."""
        should_restore_to_cpu = False
        if (
            self.storage_device == "cpu"
            and not self.cpu_only
            and self._accelerator_device is not None
        ):
            # Keep EMA in CPU RAM between steps but update it on accelerator.
            self.to(device=self._accelerator_device, non_blocking=True)
            should_restore_to_cpu = True

        one_minus_decay = 1.0 - float(self.decay)
        try:
            ema_params = list(self.model.parameters())
            student_params = list(model.parameters())
            if len(ema_params) != len(student_params):
                raise RuntimeError("EMA/student parameter count mismatch.")
            for ema_param, student_param in zip(ema_params, student_params):
                source = student_param.detach().to(
                    device=ema_param.device,
                    dtype=ema_param.dtype,
                )
                ema_param.mul_(self.decay).add_(source, alpha=one_minus_decay)

            ema_buffers = list(self.model.buffers())
            student_buffers = list(model.buffers())
            if len(ema_buffers) != len(student_buffers):
                raise RuntimeError("EMA/student buffer count mismatch.")
            for ema_buffer, student_buffer in zip(ema_buffers, student_buffers):
                ema_buffer.copy_(
                    student_buffer.detach().to(
                        device=ema_buffer.device,
                        dtype=ema_buffer.dtype,
                    )
                )
        finally:
            if should_restore_to_cpu:
                self.to(device=torch.device("cpu"), non_blocking=True)

    def state_dict(self) -> dict[str, object]:
        """Return state payload compatible with Accelerate checkpointing."""
        return {
            "decay": float(self.decay),
            "storage_device": self.storage_device,
            "cpu_only": bool(self.cpu_only),
            "model": self.model.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore EMA state payload from Accelerate checkpointing."""
        decay_value = state_dict["decay"]
        if not isinstance(decay_value, (int, float)):
            raise TypeError("EMA decay must be numeric.")
        self.decay = float(decay_value)
        storage_device = state_dict.get("storage_device")
        if isinstance(storage_device, str):
            self.storage_device = _normalize_ema_storage_device(storage_device)
        cpu_only = state_dict.get("cpu_only")
        if cpu_only is not None:
            self.cpu_only = bool(cpu_only)
        if self.cpu_only:
            self.storage_device = "cpu"
        model_state = state_dict["model"]
        if not isinstance(model_state, dict):
            raise TypeError("EMA model state must be a state-dict mapping.")
        self.model.load_state_dict(model_state, strict=True)


__all__ = ["EMATeacher"]
