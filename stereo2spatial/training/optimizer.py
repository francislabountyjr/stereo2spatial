"""Optimizer construction utilities for training."""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import OptimizerConfig


def _is_no_decay_param(name: str, param: torch.nn.Parameter) -> bool:
    """Return True when a parameter should live in the no-weight-decay group."""
    normalized_name = name.lower()
    if normalized_name.endswith(".bias"):
        return True
    if "norm" in normalized_name or "rmsnorm" in normalized_name:
        return True
    if "mem_init" in normalized_name:
        return True
    if param.ndim == 1:
        return True
    return False


def _is_muon_param(name: str, param: torch.nn.Parameter) -> bool:
    """Return True when a parameter is suitable for Muon orthogonalized updates."""
    normalized_name = name.lower()
    if param.ndim < 2:
        return False
    if _is_no_decay_param(name, param):
        return False
    excluded_fragments = (
        "target_in",
        "cond_in",
        "waveform_in",
        "waveform_cond_in",
        "waveform_out",
        "time_mlp",
        "mix_style_mlp",
        "time_mod",
        "cond_mod",
        "semantic_mod",
        "embed",
        "embedding",
        "final",
        "head",
        "output",
        "classifier",
    )
    return not any(fragment in normalized_name for fragment in excluded_fragments)


def build_optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Create decayed and non-decayed parameter groups."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay_param(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_muon_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Create Muon and AdamW fallback parameter groups."""
    muon_params: list[torch.nn.Parameter] = []
    adamw_decay_params: list[torch.nn.Parameter] = []
    adamw_no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_param(name, param):
            muon_params.append(param)
        elif _is_no_decay_param(name, param):
            adamw_no_decay_params.append(param)
        else:
            adamw_decay_params.append(param)

    return [
        {"params": muon_params, "weight_decay": float(weight_decay), "use_muon": True},
        {
            "params": adamw_decay_params,
            "weight_decay": float(weight_decay),
            "use_muon": False,
        },
        {"params": adamw_no_decay_params, "weight_decay": 0.0, "use_muon": False},
    ]


def _orthogonalize_newton_schulz(
    update: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Approximate the zeroth power / polar factor of a matrix update.

    This is the Newton-Schulz quintic iteration used by Muon-style optimizers.
    """
    if update.dim() < 2:
        return update
    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.transpose(0, 1)

    work_dtype = torch.bfloat16 if matrix.is_cuda else torch.float32
    x = matrix.to(dtype=work_dtype)
    x = x / x.norm().clamp_min(float(eps))

    a = 3.4445
    b = -4.7750
    c = 2.0315
    for _ in range(max(1, int(steps))):
        xx_t = x @ x.transpose(0, 1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.reshape(original_shape).to(dtype=update.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer with AdamW fallback groups.

    Groups marked ``use_muon=True`` receive momentum plus Newton-Schulz
    orthogonalized matrix updates. Other groups use AdamW updates.
    """

    def __init__(
        self,
        params: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(x) for x in betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "nesterov": bool(nesterov),
            "use_muon": False,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if bool(group.get("use_muon", False)):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1 = float(group["betas"][0])
        weight_decay = float(group.get("weight_decay", 0.0))
        ns_steps = int(group.get("ns_steps", 5))
        nesterov = bool(group.get("nesterov", True))

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")
            if not torch.isfinite(grad).all():
                continue

            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)

            momentum = state["momentum_buffer"]
            momentum.lerp_(grad, 1.0 - beta1)
            update = grad.lerp(momentum, beta1) if nesterov else momentum
            update = _orthogonalize_newton_schulz(update, steps=ns_steps)
            if not torch.isfinite(update).all():
                continue

            fan_out = max(1, int(update.reshape(update.shape[0], -1).shape[0]))
            fan_in = max(1, int(update.reshape(update.shape[0], -1).shape[1]))
            update_scale = math.sqrt(max(1.0, fan_out / fan_in))

            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update, alpha=-lr * update_scale)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = (float(x) for x in group["betas"])
        eps = float(group["eps"])
        weight_decay = float(group.get("weight_decay", 0.0))

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("AdamW fallback does not support sparse gradients.")
            if not torch.isfinite(grad).all():
                continue

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = int(state["step"])

            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)


def build_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Instantiate a configured optimizer for the model."""
    optimizer_type = str(getattr(optimizer_config, "type", "adamw")).strip().lower()

    if optimizer_type == "muon":
        optimizer_groups = build_muon_param_groups(
            model=model,
            weight_decay=float(optimizer_config.weight_decay),
        )
        return Muon(
            optimizer_groups,
            lr=float(optimizer_config.lr),
            betas=(float(optimizer_config.beta1), float(optimizer_config.beta2)),
            eps=float(optimizer_config.eps),
            weight_decay=0.0,
            ns_steps=int(getattr(optimizer_config, "muon_ns_steps", 5)),
            nesterov=bool(getattr(optimizer_config, "muon_nesterov", True)),
        )

    optimizer_groups = build_optimizer_param_groups(
        model=model,
        weight_decay=float(optimizer_config.weight_decay),
    )

    if optimizer_type == "adamw":
        adamw_kwargs: dict[str, Any] = {
            "lr": optimizer_config.lr,
            "betas": (optimizer_config.beta1, optimizer_config.beta2),
            "eps": optimizer_config.eps,
            # Weight decay is handled per parameter group.
            "weight_decay": 0.0,
        }

        if bool(getattr(optimizer_config, "adamw_fused", False)):
            adamw_kwargs["fused"] = True
        elif bool(getattr(optimizer_config, "adamw_foreach", False)):
            adamw_kwargs["foreach"] = True

        try:
            return torch.optim.AdamW(optimizer_groups, **adamw_kwargs)
        except (TypeError, ValueError, RuntimeError) as error:
            unsupported_option: str | None = None
            if "fused" in adamw_kwargs:
                unsupported_option = "fused"
            elif "foreach" in adamw_kwargs:
                unsupported_option = "foreach"
            if unsupported_option is None:
                raise

            print(
                "[optimizer_warning] "
                f"AdamW option {unsupported_option}=true is not supported in this runtime "
                f"({error}). Falling back to {unsupported_option}=false."
            )
            adamw_kwargs.pop(unsupported_option, None)
            return torch.optim.AdamW(optimizer_groups, **adamw_kwargs)

    if optimizer_type == "adam":
        return torch.optim.Adam(
            optimizer_groups,
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            weight_decay=0.0,
        )

    raise ValueError(
        "Unsupported optimizer.type: "
        f"{optimizer_type!r}. Supported values are: adamw, adam, muon."
    )


__all__ = [
    "Muon",
    "build_muon_param_groups",
    "build_optimizer",
    "build_optimizer_param_groups",
]
