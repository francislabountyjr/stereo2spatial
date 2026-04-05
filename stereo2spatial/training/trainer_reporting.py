"""Progress-bar and log-message formatting helpers for training."""

from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator


def _log_main(
    *,
    accelerator: Accelerator,
    progress_bar: Any,
    message: str,
) -> None:
    """Emit a message from the main process, using tqdm when available."""
    if not accelerator.is_main_process:
        return
    if progress_bar is not None:
        progress_bar.write(message)
    else:
        print(message)


def _build_step_postfix(
    *,
    optimizer: torch.optim.Optimizer,
    step_loss_value: float,
    t_eff: int,
    num_windows: int,
    use_gan: bool,
    loss_d_step: torch.Tensor | None,
    loss_adv_step: torch.Tensor | None,
    gan_lambda_adv_step: float,
    loss_route_step: torch.Tensor | None,
    loss_corr_step: torch.Tensor | None,
    latest_avg_loss: float | None,
    latest_avg_d_loss: float | None,
    latest_avg_adv_loss: float | None,
    latest_avg_route_loss: float | None,
    latest_avg_corr_loss: float | None,
) -> dict[str, str]:
    """Build tqdm postfix for per-step updates."""
    postfix: dict[str, str] = {
        "loss_step": f"{step_loss_value:.6f}",
        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        "T": str(t_eff),
        "win": str(num_windows),
    }
    if use_gan and loss_d_step is not None and loss_adv_step is not None:
        postfix["d_step"] = f"{float(loss_d_step.item()):.6f}"
        postfix["adv_step"] = f"{float(loss_adv_step.item()):.6f}"
        postfix["lam_adv"] = f"{gan_lambda_adv_step:.2e}"
    if loss_route_step is not None:
        postfix["route_step"] = f"{float(loss_route_step.item()):.6f}"
    if loss_corr_step is not None:
        postfix["corr_step"] = f"{float(loss_corr_step.item()):.6f}"
    if latest_avg_loss is not None:
        postfix["loss_avg"] = f"{latest_avg_loss:.6f}"
    if use_gan and latest_avg_d_loss is not None:
        postfix["d_avg"] = f"{latest_avg_d_loss:.6f}"
    if use_gan and latest_avg_adv_loss is not None:
        postfix["adv_avg"] = f"{latest_avg_adv_loss:.6f}"
    if latest_avg_route_loss is not None:
        postfix["route_avg"] = f"{latest_avg_route_loss:.6f}"
    if latest_avg_corr_loss is not None:
        postfix["corr_avg"] = f"{latest_avg_corr_loss:.6f}"
    return postfix


def _build_log_postfix(
    *,
    optimizer: torch.optim.Optimizer,
    step_loss_value: float,
    avg_loss: float,
    t_eff: int,
    num_windows: int,
    use_gan: bool,
    loss_d_step: torch.Tensor | None,
    loss_adv_step: torch.Tensor | None,
    gan_lambda_adv_step: float,
    loss_route_step: torch.Tensor | None,
    loss_corr_step: torch.Tensor | None,
    avg_d_loss: float | None,
    avg_adv_loss: float | None,
    avg_route_loss: float | None,
    avg_corr_loss: float | None,
) -> dict[str, str]:
    """Build tqdm postfix for log intervals (uses running averages)."""
    log_postfix: dict[str, str] = {
        "loss_step": f"{step_loss_value:.6f}",
        "loss_avg": f"{avg_loss:.6f}",
        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        "T": str(t_eff),
        "win": str(num_windows),
    }
    if use_gan and loss_d_step is not None and loss_adv_step is not None:
        log_postfix["d_step"] = f"{float(loss_d_step.item()):.6f}"
        log_postfix["adv_step"] = f"{float(loss_adv_step.item()):.6f}"
        log_postfix["lam_adv"] = f"{gan_lambda_adv_step:.2e}"
    if loss_route_step is not None:
        log_postfix["route_step"] = f"{float(loss_route_step.item()):.6f}"
    if loss_corr_step is not None:
        log_postfix["corr_step"] = f"{float(loss_corr_step.item()):.6f}"
    if avg_d_loss is not None:
        log_postfix["d_avg"] = f"{avg_d_loss:.6f}"
    if avg_adv_loss is not None:
        log_postfix["adv_avg"] = f"{avg_adv_loss:.6f}"
    if avg_route_loss is not None:
        log_postfix["route_avg"] = f"{avg_route_loss:.6f}"
    if avg_corr_loss is not None:
        log_postfix["corr_avg"] = f"{avg_corr_loss:.6f}"
    return log_postfix


def _build_step_message(
    *,
    global_step: int,
    epoch: int,
    batches_seen_in_epoch: int,
    total_batches_this_epoch: int,
    step_loss_value: float,
    avg_loss: float,
    optimizer: torch.optim.Optimizer,
    t_eff: int,
    num_windows: int,
    use_gan: bool,
    loss_d_step: torch.Tensor | None,
    loss_adv_step: torch.Tensor | None,
    gan_lambda_adv_step: float,
    avg_d_loss: float | None,
    avg_adv_loss: float | None,
    loss_route_step: torch.Tensor | None,
    loss_corr_step: torch.Tensor | None,
    avg_route_loss: float | None,
    avg_corr_loss: float | None,
    cond_stereo: int,
    cond_mono: int,
    cond_downmix: int,
) -> str:
    """Render the single-line periodic training log message."""
    message = (
        f"step={global_step} "
        f"epoch={epoch} "
        f"batch={batches_seen_in_epoch}/{total_batches_this_epoch} "
        f"loss_step={step_loss_value:.6f} "
        f"loss_avg={avg_loss:.6f} "
        f"lr={optimizer.param_groups[0]['lr']:.7f} "
        f"T_eff={t_eff} "
        f"windows={num_windows} "
    )
    if use_gan and loss_d_step is not None and loss_adv_step is not None:
        message += (
            f"d_step={float(loss_d_step.item()):.6f} "
            f"adv_step={float(loss_adv_step.item()):.6f} "
            f"lam_adv={gan_lambda_adv_step:.2e} "
        )
    if use_gan and avg_d_loss is not None and avg_adv_loss is not None:
        message += f"d_avg={avg_d_loss:.6f} " f"adv_avg={avg_adv_loss:.6f} "
    if loss_route_step is not None:
        message += f"route_step={float(loss_route_step.item()):.6f} "
    if loss_corr_step is not None:
        message += f"corr_step={float(loss_corr_step.item()):.6f} "
    if avg_route_loss is not None and avg_corr_loss is not None:
        message += f"route_avg={avg_route_loss:.6f} " f"corr_avg={avg_corr_loss:.6f} "
    message += (
        "cond(stereo/mono/downmix)="
        f"{cond_stereo}/{cond_mono}/{cond_downmix}"
    )
    return message
