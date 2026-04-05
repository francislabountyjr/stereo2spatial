"""Running-loss accumulation helpers for the training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from accelerate import Accelerator


@dataclass(frozen=True)
class ReducedAverages:
    """Step-window averaged losses reduced across distributed workers."""

    avg_loss: float
    avg_d_loss: float | None
    avg_adv_loss: float | None
    avg_route_loss: float | None
    avg_corr_loss: float | None


@dataclass
class RunningLossState:
    """Mutable accumulators for local (pre-reduction) running losses."""

    loss_sum: torch.Tensor
    count: torch.Tensor
    d_loss_sum: torch.Tensor
    adv_loss_sum: torch.Tensor
    route_loss_sum: torch.Tensor
    corr_loss_sum: torch.Tensor

    @classmethod
    def create(cls: type[RunningLossState], device: torch.device) -> RunningLossState:
        """Create zero-initialized running-loss accumulators on a target device."""
        return cls(
            loss_sum=torch.zeros((), device=device, dtype=torch.float64),
            count=torch.zeros((), device=device, dtype=torch.long),
            d_loss_sum=torch.zeros((), device=device, dtype=torch.float64),
            adv_loss_sum=torch.zeros((), device=device, dtype=torch.float64),
            route_loss_sum=torch.zeros((), device=device, dtype=torch.float64),
            corr_loss_sum=torch.zeros((), device=device, dtype=torch.float64),
        )


def update_running_losses(
    *,
    state: RunningLossState,
    loss: torch.Tensor,
    use_gan: bool,
    loss_d_step: torch.Tensor | None,
    loss_adv_step: torch.Tensor | None,
    loss_route_step: torch.Tensor | None,
    loss_corr_step: torch.Tensor | None,
) -> None:
    """Accumulate one synchronized-step loss tuple into running state."""
    state.loss_sum += loss.detach().to(dtype=torch.float64)
    state.count += 1
    if use_gan and loss_d_step is not None and loss_adv_step is not None:
        state.d_loss_sum += loss_d_step.to(dtype=torch.float64)
        state.adv_loss_sum += loss_adv_step.to(dtype=torch.float64)
    if loss_route_step is not None:
        state.route_loss_sum += loss_route_step.to(dtype=torch.float64)
    if loss_corr_step is not None:
        state.corr_loss_sum += loss_corr_step.to(dtype=torch.float64)


def compute_reduced_averages(
    *,
    accelerator: Accelerator,
    state: RunningLossState,
    use_gan: bool,
    use_channel_aux_losses: bool,
) -> ReducedAverages:
    """All-reduce running sums and convert them into scalar averages."""
    total_loss_sum = accelerator.reduce(state.loss_sum, reduction="sum")
    total_loss_count = accelerator.reduce(state.count, reduction="sum")
    denom = torch.clamp(total_loss_count.to(dtype=torch.float64), min=1.0)
    avg_loss = float((total_loss_sum / denom).item())

    avg_d_loss: float | None = None
    avg_adv_loss: float | None = None
    if use_gan:
        total_d_loss_sum = accelerator.reduce(state.d_loss_sum, reduction="sum")
        total_adv_loss_sum = accelerator.reduce(state.adv_loss_sum, reduction="sum")
        avg_d_loss = float((total_d_loss_sum / denom).item())
        avg_adv_loss = float((total_adv_loss_sum / denom).item())

    avg_route_loss: float | None = None
    avg_corr_loss: float | None = None
    if use_channel_aux_losses:
        total_route_loss_sum = accelerator.reduce(state.route_loss_sum, reduction="sum")
        total_corr_loss_sum = accelerator.reduce(state.corr_loss_sum, reduction="sum")
        avg_route_loss = float((total_route_loss_sum / denom).item())
        avg_corr_loss = float((total_corr_loss_sum / denom).item())

    return ReducedAverages(
        avg_loss=avg_loss,
        avg_d_loss=avg_d_loss,
        avg_adv_loss=avg_adv_loss,
        avg_route_loss=avg_route_loss,
        avg_corr_loss=avg_corr_loss,
    )


def reset_running_losses(
    *,
    state: RunningLossState,
    use_gan: bool,
    use_channel_aux_losses: bool,
) -> None:
    """Zero running accumulators after a logging/reporting checkpoint."""
    state.loss_sum.zero_()
    state.count.zero_()
    if use_gan:
        state.d_loss_sum.zero_()
        state.adv_loss_sum.zero_()
    if use_channel_aux_losses:
        state.route_loss_sum.zero_()
        state.corr_loss_sum.zero_()
