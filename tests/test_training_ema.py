from __future__ import annotations

from typing import Any, TypeVar, overload

import torch

from stereo2spatial.training.ema import EMATeacher


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer("running_stat", torch.ones(1, dtype=torch.float32))


class _OrigModPrefixedStateDictModel(torch.nn.Module):
    def __init__(self, inner: _TinyModel) -> None:
        super().__init__()
        self.inner = inner

    _TStateDict = TypeVar("_TStateDict", bound=dict[str, Any])

    @overload
    def state_dict(
        self,
        *,
        destination: _TStateDict,
        prefix: str = ...,
        keep_vars: bool = ...,
    ) -> _TStateDict: ...

    @overload
    def state_dict(
        self,
        *,
        prefix: str = ...,
        keep_vars: bool = ...,
    ) -> dict[str, Any]: ...

    def state_dict(
        self,
        *,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        inner_state = self.inner.state_dict(prefix="", keep_vars=keep_vars)
        target = destination if destination is not None else {}
        target.update(
            {f"{prefix}_orig_mod.{key}": value for key, value in inner_state.items()}
        )
        return target


class _TrackingEMATeacher(EMATeacher):
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        *,
        storage_device: str = "accelerator",
        cpu_only: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            decay=decay,
            storage_device=storage_device,
            cpu_only=cpu_only,
        )
        self.to_calls: list[tuple[str, bool]] = []

    @torch.no_grad()
    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> None:
        self.to_calls.append((str(device), non_blocking))
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)


@torch.no_grad()
def _fill_model(model: torch.nn.Module, value: float) -> None:
    for parameter in model.parameters():
        parameter.fill_(value)
    for buffer in model.buffers():
        buffer.fill_(value)


def test_ema_teacher_cpu_only_update_keeps_weights_on_cpu() -> None:
    student = _TinyModel()
    ema_teacher = EMATeacher(
        model=student,
        decay=0.5,
        storage_device="cpu",
        cpu_only=True,
    )
    ema_teacher.configure_runtime(
        accelerator_device=torch.device("cpu"),
        dtype=torch.float32,
    )

    _fill_model(student, 2.0)
    ema_teacher.copy_from(student)
    _fill_model(student, 4.0)
    ema_teacher.update(student)

    for parameter in ema_teacher.model.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 3.0))
        assert parameter.device.type == "cpu"
    for buffer in ema_teacher.model.buffers():
        assert torch.allclose(buffer, torch.full_like(buffer, 4.0))
        assert buffer.device.type == "cpu"


def test_ema_teacher_cpu_storage_hops_for_update_when_not_cpu_only() -> None:
    student = _TinyModel()
    ema_teacher = _TrackingEMATeacher(
        model=student,
        decay=0.9,
        storage_device="cpu",
        cpu_only=False,
    )
    ema_teacher.configure_runtime(
        accelerator_device=torch.device("cpu"),
        dtype=torch.float32,
    )

    ema_teacher.to_calls.clear()
    ema_teacher.update(student)

    assert len(ema_teacher.to_calls) == 2
    assert ema_teacher.to_calls[0] == ("cpu", True)
    assert ema_teacher.to_calls[1] == ("cpu", True)


def test_ema_teacher_cpu_only_does_not_hop_during_update() -> None:
    student = _TinyModel()
    ema_teacher = _TrackingEMATeacher(
        model=student,
        decay=0.9,
        storage_device="cpu",
        cpu_only=True,
    )
    ema_teacher.configure_runtime(
        accelerator_device=torch.device("cpu"),
        dtype=torch.float32,
    )

    ema_teacher.to_calls.clear()
    ema_teacher.update(student)

    assert ema_teacher.to_calls == []


def test_ema_teacher_copy_from_accepts_orig_mod_prefixed_state_dict_keys() -> None:
    student = _TinyModel()
    prefixed_student = _OrigModPrefixedStateDictModel(student)
    ema_teacher = EMATeacher(model=student, decay=0.9)

    _fill_model(student, 5.0)
    ema_teacher.copy_from(prefixed_student)

    for parameter in ema_teacher.model.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 5.0))
    for buffer in ema_teacher.model.buffers():
        assert torch.allclose(buffer, torch.full_like(buffer, 5.0))
