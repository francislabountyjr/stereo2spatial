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
    skipped_step: bool = False
    skip_reason: str | None = None
    grad_norm: torch.Tensor | None = None


def _zero_optimizers(
    *,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer | None,
) -> None:
    """Clear pending gradients after a protected skipped step."""
    optimizer.zero_grad(set_to_none=True)
    if discriminator_optimizer is not None:
        discriminator_optimizer.zero_grad(set_to_none=True)


def _make_skipped_result(
    *,
    loss: torch.Tensor,
    t_eff: int,
    num_windows: int,
    loss_d_step: torch.Tensor | None = None,
    loss_adv_step: torch.Tensor | None = None,
    loss_route_step: torch.Tensor | None = None,
    loss_corr_step: torch.Tensor | None = None,
    gan_lambda_adv_step: float = 0.0,
    skip_reason: str,
    grad_norm: torch.Tensor | None = None,
) -> TrainingStepResult:
    """Return a standardized result for a batch whose optimizer update was skipped."""
    return TrainingStepResult(
        loss=loss.detach(),
        t_eff=t_eff,
        num_windows=num_windows,
        loss_d_step=loss_d_step.detach() if loss_d_step is not None else None,
        loss_adv_step=loss_adv_step.detach() if loss_adv_step is not None else None,
        loss_route_step=loss_route_step.detach()
        if loss_route_step is not None
        else None,
        loss_corr_step=loss_corr_step.detach() if loss_corr_step is not None else None,
        gan_lambda_adv_step=gan_lambda_adv_step,
        skipped_step=True,
        skip_reason=skip_reason,
        grad_norm=grad_norm.detach() if grad_norm is not None else None,
    )


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
            (
                loss_fm,
                t_eff,
                num_windows,
                gan_aux,
            ) = _compute_full_song_flow_matching_loss(
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
                downmix_consistency_weight=settings.downmix_consistency_weight,
                downmix_consistency_loss=settings.downmix_consistency_loss,
                downmix_channel_order=settings.downmix_channel_order,
                mrstft_loss_weight=settings.mrstft_loss_weight,
                mrstft_fft_sizes=settings.mrstft_fft_sizes,
                mrstft_hop_lengths=settings.mrstft_hop_lengths,
                mrstft_win_lengths=settings.mrstft_win_lengths,
                mrstft_sc_weight=settings.mrstft_sc_weight,
                mrstft_log_mag_weight=settings.mrstft_log_mag_weight,
                mrstft_eps=settings.mrstft_eps,
                waveform_mse_loss_weight=settings.waveform_mse_loss_weight,
                waveform_l1_loss_weight=settings.waveform_l1_loss_weight,
                waveform_charbonnier_loss_weight=(
                    settings.waveform_charbonnier_loss_weight
                ),
                waveform_charbonnier_eps=settings.waveform_charbonnier_eps,
                perceptual_loss_weight=settings.perceptual_loss_weight,
                perceptual_sample_rate=settings.perceptual_sample_rate,
                perceptual_n_fft=settings.perceptual_n_fft,
                perceptual_hop_length=settings.perceptual_hop_length,
                perceptual_win_length=settings.perceptual_win_length,
                perceptual_n_mels=settings.perceptual_n_mels,
                perceptual_f_min=settings.perceptual_f_min,
                perceptual_f_max=settings.perceptual_f_max,
                perceptual_band_weight=settings.perceptual_band_weight,
                perceptual_band_low_hz=settings.perceptual_band_low_hz,
                perceptual_band_high_hz=settings.perceptual_band_high_hz,
                perceptual_eps=settings.perceptual_eps,
                binaural_ild_loss_weight=settings.binaural_ild_loss_weight,
                binaural_ipd_loss_weight=settings.binaural_ipd_loss_weight,
                binaural_ccf_loss_weight=settings.binaural_ccf_loss_weight,
                binaural_frame_ild_loss_weight=(
                    settings.binaural_frame_ild_loss_weight
                ),
                binaural_frame_ild_frame_size=(
                    settings.binaural_frame_ild_frame_size
                ),
                binaural_frame_ild_hop_size=settings.binaural_frame_ild_hop_size,
                binaural_frame_ild_silence_threshold=(
                    settings.binaural_frame_ild_silence_threshold
                ),
                binaural_frame_ild_max_weight=(
                    settings.binaural_frame_ild_max_weight
                ),
                binaural_mid_side_loss_weight=settings.binaural_mid_side_loss_weight,
                binaural_mid_side_loss_type=settings.binaural_mid_side_loss_type,
                binaural_mid_side_mid_weight=settings.binaural_mid_side_mid_weight,
                binaural_mid_side_side_weight=settings.binaural_mid_side_side_weight,
                binaural_mid_side_charbonnier_eps=(
                    settings.binaural_mid_side_charbonnier_eps
                ),
                binaural_loss_warmup_steps=settings.binaural_loss_warmup_steps,
                binaural_sample_rate=settings.binaural_sample_rate,
                binaural_loss_eps=settings.binaural_loss_eps,
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
                downmix_consistency_weight=settings.downmix_consistency_weight,
                downmix_consistency_loss=settings.downmix_consistency_loss,
                downmix_channel_order=settings.downmix_channel_order,
                mrstft_loss_weight=settings.mrstft_loss_weight,
                mrstft_fft_sizes=settings.mrstft_fft_sizes,
                mrstft_hop_lengths=settings.mrstft_hop_lengths,
                mrstft_win_lengths=settings.mrstft_win_lengths,
                mrstft_sc_weight=settings.mrstft_sc_weight,
                mrstft_log_mag_weight=settings.mrstft_log_mag_weight,
                mrstft_eps=settings.mrstft_eps,
                waveform_mse_loss_weight=settings.waveform_mse_loss_weight,
                waveform_l1_loss_weight=settings.waveform_l1_loss_weight,
                waveform_charbonnier_loss_weight=(
                    settings.waveform_charbonnier_loss_weight
                ),
                waveform_charbonnier_eps=settings.waveform_charbonnier_eps,
                perceptual_loss_weight=settings.perceptual_loss_weight,
                perceptual_sample_rate=settings.perceptual_sample_rate,
                perceptual_n_fft=settings.perceptual_n_fft,
                perceptual_hop_length=settings.perceptual_hop_length,
                perceptual_win_length=settings.perceptual_win_length,
                perceptual_n_mels=settings.perceptual_n_mels,
                perceptual_f_min=settings.perceptual_f_min,
                perceptual_f_max=settings.perceptual_f_max,
                perceptual_band_weight=settings.perceptual_band_weight,
                perceptual_band_low_hz=settings.perceptual_band_low_hz,
                perceptual_band_high_hz=settings.perceptual_band_high_hz,
                perceptual_eps=settings.perceptual_eps,
                binaural_ild_loss_weight=settings.binaural_ild_loss_weight,
                binaural_ipd_loss_weight=settings.binaural_ipd_loss_weight,
                binaural_ccf_loss_weight=settings.binaural_ccf_loss_weight,
                binaural_frame_ild_loss_weight=(
                    settings.binaural_frame_ild_loss_weight
                ),
                binaural_frame_ild_frame_size=(
                    settings.binaural_frame_ild_frame_size
                ),
                binaural_frame_ild_hop_size=settings.binaural_frame_ild_hop_size,
                binaural_frame_ild_silence_threshold=(
                    settings.binaural_frame_ild_silence_threshold
                ),
                binaural_frame_ild_max_weight=(
                    settings.binaural_frame_ild_max_weight
                ),
                binaural_mid_side_loss_weight=settings.binaural_mid_side_loss_weight,
                binaural_mid_side_loss_type=settings.binaural_mid_side_loss_type,
                binaural_mid_side_mid_weight=settings.binaural_mid_side_mid_weight,
                binaural_mid_side_side_weight=settings.binaural_mid_side_side_weight,
                binaural_mid_side_charbonnier_eps=(
                    settings.binaural_mid_side_charbonnier_eps
                ),
                binaural_loss_warmup_steps=settings.binaural_loss_warmup_steps,
                binaural_sample_rate=settings.binaural_sample_rate,
                binaural_loss_eps=settings.binaural_loss_eps,
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
            if not torch.isfinite(loss.detach()).all():
                _zero_optimizers(
                    optimizer=optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                )
                return _make_skipped_result(
                    loss=loss,
                    t_eff=t_eff,
                    num_windows=num_windows,
                    loss_route_step=loss_route_step,
                    loss_corr_step=loss_corr_step,
                    skip_reason="nonfinite_loss",
                )
            if not tbptt_backward_done_in_loss:
                accelerator.backward(loss)

        grad_norm: torch.Tensor | None = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if settings.use_gan and discriminator is not None:
                accelerator.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
            if grad_norm is not None and not torch.isfinite(grad_norm.detach()).all():
                _zero_optimizers(
                    optimizer=optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                )
                return _make_skipped_result(
                    loss=loss,
                    t_eff=t_eff,
                    num_windows=num_windows,
                    loss_d_step=loss_d_step,
                    loss_adv_step=loss_adv_step,
                    loss_route_step=loss_route_step,
                    loss_corr_step=loss_corr_step,
                    gan_lambda_adv_step=gan_lambda_adv_step,
                    skip_reason="nonfinite_grad_norm",
                    grad_norm=grad_norm,
                )

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
        grad_norm=grad_norm,
    )
