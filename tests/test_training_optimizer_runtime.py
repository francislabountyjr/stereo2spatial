from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from stereo2spatial.training.config import OptimizerConfig, TrainConfig
from stereo2spatial.training.optimizer import (
    build_optimizer,
    build_optimizer_param_groups,
)
from stereo2spatial.training.runtime import _apply_lr, _lr_for_step


class _TinyOptimizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.mem_init = torch.nn.Parameter(torch.ones(4, 4))
        self.frozen = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)


def _optimizer_config(**overrides: Any) -> OptimizerConfig:
    payload: dict[str, Any] = {
        "type": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "adamw_fused": False,
        "adamw_foreach": False,
    }
    payload.update(overrides)
    return OptimizerConfig(**payload)


def _runtime_config(
    *,
    scheduler_type: str,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
    warmup_steps: int = 0,
    max_steps: int = 100,
) -> TrainConfig:
    return cast(
        TrainConfig,
        SimpleNamespace(
            optimizer=SimpleNamespace(lr=base_lr),
            scheduler=SimpleNamespace(
                type=scheduler_type, min_lr=min_lr, warmup_steps=warmup_steps
            ),
            training=SimpleNamespace(max_steps=max_steps),
        ),
    )


def test_build_optimizer_param_groups_split_decay_and_no_decay() -> None:
    model = _TinyOptimizerModel()

    groups = build_optimizer_param_groups(
        model=cast(Any, model),
        weight_decay=0.1,
    )

    decay_group, no_decay_group = groups
    decay_ids = {id(param) for param in decay_group["params"]}
    no_decay_ids = {id(param) for param in no_decay_group["params"]}

    assert decay_group["weight_decay"] == pytest.approx(0.1)
    assert no_decay_group["weight_decay"] == pytest.approx(0.0)

    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids
    assert id(model.norm.weight) in no_decay_ids
    assert id(model.norm.bias) in no_decay_ids
    assert id(model.mem_init) in no_decay_ids
    assert id(model.frozen) not in decay_ids | no_decay_ids


def test_build_optimizer_rejects_unknown_optimizer_type() -> None:
    model = _TinyOptimizerModel()
    config = _optimizer_config(type="sgd")

    with pytest.raises(ValueError, match="Unsupported optimizer.type"):
        build_optimizer(model=cast(Any, model), optimizer_config=config)


def test_build_optimizer_falls_back_when_fused_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyOptimizerModel()
    config = _optimizer_config(adamw_fused=True)

    calls: list[dict[str, Any]] = []

    class _SentinelOptimizer:
        pass

    def _fake_adamw(_: Any, **kwargs: Any) -> _SentinelOptimizer:
        calls.append(dict(kwargs))
        if kwargs.get("fused") is True:
            raise RuntimeError("fused not supported")
        return _SentinelOptimizer()

    monkeypatch.setattr(torch.optim, "AdamW", _fake_adamw)

    optimizer = build_optimizer(model=cast(Any, model), optimizer_config=config)

    assert isinstance(optimizer, _SentinelOptimizer)
    assert len(calls) == 2
    assert calls[0].get("fused") is True
    assert "fused" not in calls[1]


def test_lr_for_step_applies_warmup_and_cosine_decay() -> None:
    config = _runtime_config(
        scheduler_type="cosine",
        base_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=4,
        max_steps=10,
    )

    assert _lr_for_step(0, config) == pytest.approx(2.5e-4)
    assert _lr_for_step(3, config) == pytest.approx(1e-3)
    assert _lr_for_step(10, config) == pytest.approx(1e-5)
    assert _lr_for_step(15, config) == pytest.approx(1e-5)


def test_lr_for_step_constant_scheduler_stays_at_base_after_warmup() -> None:
    config = _runtime_config(
        scheduler_type="constant",
        base_lr=2e-4,
        min_lr=1e-6,
        warmup_steps=2,
        max_steps=100,
    )

    assert _lr_for_step(0, config) == pytest.approx(1e-4)
    assert _lr_for_step(1, config) == pytest.approx(2e-4)
    assert _lr_for_step(99, config) == pytest.approx(2e-4)


def test_apply_lr_sets_all_optimizer_param_groups() -> None:
    param_a = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    optimizer = torch.optim.SGD(
        [
            {"params": [param_a], "lr": 0.3},
            {"params": [param_b], "lr": 0.5},
        ],
        lr=0.1,
    )

    _apply_lr(optimizer, 0.0123)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0123)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.0123)
