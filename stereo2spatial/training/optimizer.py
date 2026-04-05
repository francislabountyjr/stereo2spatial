"""Optimizer construction utilities for training."""

from __future__ import annotations

from typing import Any

import torch

from stereo2spatial.modeling import SpatialDiT

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


def build_optimizer_param_groups(
    model: SpatialDiT,
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


def build_optimizer(
    model: SpatialDiT,
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Instantiate a configured optimizer for the model."""
    optimizer_type = str(getattr(optimizer_config, "type", "adamw")).strip().lower()
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
        f"{optimizer_type!r}. Supported values are: adamw, adam."
    )

