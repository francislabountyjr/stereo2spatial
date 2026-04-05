"""Single-batch optimization step helpers for trainer orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from accelerate import Accelerator

from .config import TrainConfig
from .gan_training import compute_channel_aux_losses, run_gan_step
from .losses import (
    _compute_batch_flow_matching_loss,
    _compute_full_song_flow_matching_loss,
)
from .runtime import _apply_lr, _lr_for_step
from .sequence_plan import SequenceTrainingPlan
from .trainer_settings import TrainerRuntimeSettings


@dataclass(frozen=True)
class TrainingStepResult:
    """Outputs produced by one trainer batch update attempt."""

    loss: torch.Tensor
    t_eff: int
    num_windows: int
    loss_d_step: torch.Tensor | None
    loss_adv_step: torch.Tensor | None
    loss_route_step: torch.Tensor | None
    loss_corr_step: torch.Tensor | None
    gan_lambda_adv_step: float


def _run_training_step(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    discriminator: torch.nn.Module | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    batch: dict[str, torch.Tensor],
    sequence_plan: SequenceTrainingPlan,
    global_step: int,
    seed: int,
    settings: TrainerRuntimeSettings,
    grad_clip_norm: float,
    config: TrainConfig,
) -> TrainingStepResult:
    """Execute forward/backward and optimizer-step logic for one batch."""
    with accelerator.accumulate(model):
        tbptt_backward_done_in_loss = (
            sequence_plan.sequence_mode == "full_song"
            and sequence_plan.tbptt_windows > 0
        )
        loss_d_step: torch.Tensor | None = None
        loss_adv_step: torch.Tensor | None = None
        loss_route_step: torch.Tensor | None = None
        loss_corr_step: torch.Tensor | None = None
        gan_lambda_adv_step = 0.0
        collect_aux = settings.use_gan or settings.use_channel_aux_losses
        if sequence_plan.sequence_mode == "full_song":
            loss_fm, t_eff, num_windows, gan_aux = _compute_full_song_flow_matching_loss(
                accelerator=accelerator,
                model=model,
                batch=batch,
                window_frames=sequence_plan.window_frames,
                overlap_frames=sequence_plan.overlap_frames,
                detach_memory=sequence_plan.detach_memory,
                tbptt_windows=sequence_plan.tbptt_windows,
                global_step=global_step,
                window_metadata=sequence_plan.window_metadata,
                collect_gan_aux=collect_aux,
                scheduled_sampling_config=config.training,
                enable_scheduled_sampling=True,
                routing_kl_weight=settings.routing_kl_weight,
                routing_kl_temperature=settings.routing_kl_temperature,
                routing_kl_eps=settings.routing_kl_eps,
                corr_weight=settings.corr_weight,
                corr_eps=settings.corr_eps,
                corr_offdiag_only=settings.corr_offdiag_only,
                corr_use_correlation=settings.corr_use_correlation,
            )
        else:
            loss_fm, t_eff, num_windows, gan_aux = _compute_batch_flow_matching_loss(
                accelerator=accelerator,
                model=model,
                batch=batch,
                seq_choices_frames=sequence_plan.seq_choices_frames,
                max_choice_frames=sequence_plan.max_choice_frames,
                window_frames=sequence_plan.window_frames,
                overlap_frames=sequence_plan.overlap_frames,
                randomize_per_batch=sequence_plan.randomize_per_batch,
                detach_memory=sequence_plan.detach_memory,
                global_step=global_step,
                seed=seed,
                window_metadata=sequence_plan.window_metadata,
                collect_gan_aux=collect_aux,
                scheduled_sampling_config=config.training,
                enable_scheduled_sampling=True,
                routing_kl_weight=settings.routing_kl_weight,
                routing_kl_temperature=settings.routing_kl_temperature,
                routing_kl_eps=settings.routing_kl_eps,
                corr_weight=settings.corr_weight,
                corr_eps=settings.corr_eps,
                corr_offdiag_only=settings.corr_offdiag_only,
                corr_use_correlation=settings.corr_use_correlation,
            )

        if settings.use_channel_aux_losses:
            loss_route_step, loss_corr_step = compute_channel_aux_losses(
                gan_aux=gan_aux,
                routing_kl_weight=settings.routing_kl_weight,
                routing_kl_temperature=settings.routing_kl_temperature,
                routing_kl_eps=settings.routing_kl_eps,
                corr_weight=settings.corr_weight,
                corr_eps=settings.corr_eps,
                corr_offdiag_only=settings.corr_offdiag_only,
                corr_use_correlation=settings.corr_use_correlation,
            )

        if settings.use_gan:
            if discriminator is None or discriminator_optimizer is None:
                raise RuntimeError(
                    "GAN enabled but discriminator/discriminator_optimizer are missing."
                )
            loss, loss_d_step, loss_adv_step, gan_lambda_adv_step = run_gan_step(
                accelerator=accelerator,
                discriminator=discriminator,
                gan_aux=gan_aux,
                gan_use_mask_channel=settings.gan_use_mask_channel,
                global_step=global_step,
                gan_ms_w_fine=settings.gan_ms_w_fine,
                gan_ms_w_coarse=settings.gan_ms_w_coarse,
                gan_r1_gamma=settings.gan_r1_gamma,
                gan_r1_every=settings.gan_r1_every,
                gan_lambda_adv_max=settings.gan_lambda_adv_max,
                gan_adv_warmup_steps=settings.gan_adv_warmup_steps,
                loss_fm=loss_fm,
            )
        else:
            loss = loss_fm
            if not tbptt_backward_done_in_loss:
                accelerator.backward(loss)

        if accelerator.sync_gradients and grad_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if settings.use_gan and discriminator is not None:
                accelerator.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)

        if accelerator.sync_gradients:
            lr = _lr_for_step(global_step, config)
            _apply_lr(optimizer, lr)

        if accelerator.sync_gradients:
            if settings.use_gan:
                if discriminator_optimizer is None:
                    raise RuntimeError(
                        "GAN enabled but discriminator optimizer is missing."
                    )
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad(set_to_none=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return TrainingStepResult(
        loss=loss,
        t_eff=t_eff,
        num_windows=num_windows,
        loss_d_step=loss_d_step,
        loss_adv_step=loss_adv_step,
        loss_route_step=loss_route_step,
        loss_corr_step=loss_corr_step,
        gan_lambda_adv_step=gan_lambda_adv_step,
    )
