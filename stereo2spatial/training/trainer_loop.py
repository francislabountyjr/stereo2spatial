"""Core epoch/batch loop for distributed training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from .checkpointing import _save_checkpoint
from .config import TrainConfig
from .dataset import ConditioningSource, LatentSongDataset
from .ema import EMATeacher
from .sequence_plan import SequenceTrainingPlan
from .trainer_metrics import (
    RunningLossState,
    compute_reduced_averages,
    reset_running_losses,
    update_running_losses,
)
from .trainer_reporting import (
    _build_log_postfix,
    _build_step_message,
    _build_step_postfix,
    _log_main,
)
from .trainer_settings import TrainerRuntimeSettings
from .trainer_step import _run_training_step
from .validation import _run_generation_validation, _run_latent_validation


@dataclass
class LatestAverages:
    """Most recent reduced moving averages used for reporting/postfixes."""

    avg_loss: float | None = None
    avg_d_loss: float | None = None
    avg_adv_loss: float | None = None
    avg_route_loss: float | None = None
    avg_corr_loss: float | None = None


def _create_progress_bar(
    *,
    accelerator: Accelerator,
    max_steps: int,
    initial_global_step: int,
) -> Any:
    """Create a main-process tqdm progress bar when tqdm is available."""
    if not accelerator.is_main_process or tqdm is None:
        return None
    return tqdm(
        total=max_steps,
        initial=min(initial_global_step, max_steps),
        desc="train",
        dynamic_ncols=True,
        mininterval=0.5,
    )


def _resolve_epoch_iterator(
    *,
    accelerator: Accelerator,
    dataloader: DataLoader,
    resume_batches_seen: int,
    total_batches_this_epoch: int,
) -> tuple[Any, int, int, bool, str | None]:
    """Resolve dataloader iteration state for resumed epochs."""
    if resume_batches_seen <= 0:
        return dataloader, 0, resume_batches_seen, False, None
    if resume_batches_seen >= total_batches_this_epoch:
        return dataloader, 0, 0, True, None
    return (
        accelerator.skip_first_batches(dataloader, resume_batches_seen),
        resume_batches_seen,
        0,
        False,
        (
            "Skipping already-seen batches for resumed epoch: "
            f"{resume_batches_seen}/{total_batches_this_epoch}"
        ),
    )


def _compute_conditioning_counts(
    *,
    batch: dict[str, torch.Tensor],
    accelerator: Accelerator,
) -> tuple[int, int, int]:
    """Return reduced conditioning-source counts for reporting."""
    cond_counts = torch.bincount(
        batch["conditioning_source"].detach().to(batch["target_latent"].device),
        minlength=3,
    ).to(dtype=torch.long)
    cond_counts = accelerator.reduce(cond_counts, reduction="sum")
    return (
        int(cond_counts[ConditioningSource.STEREO]),
        int(cond_counts[ConditioningSource.MONO]),
        int(cond_counts[ConditioningSource.DOWNMIX]),
    )


def _should_run_validation(
    *,
    settings: TrainerRuntimeSettings,
    global_step: int,
) -> bool:
    """Return True when the current step matches validation cadence."""
    return settings.validation_steps > 0 and global_step % settings.validation_steps == 0


def _update_latest_averages(
    *,
    latest: LatestAverages,
    avg_loss: float,
    avg_d_loss: float | None,
    avg_adv_loss: float | None,
    avg_route_loss: float | None,
    avg_corr_loss: float | None,
) -> None:
    """Mutate latest average state from newly reduced values."""
    latest.avg_loss = avg_loss
    if avg_d_loss is not None:
        latest.avg_d_loss = avg_d_loss
    if avg_adv_loss is not None:
        latest.avg_adv_loss = avg_adv_loss
    if avg_route_loss is not None:
        latest.avg_route_loss = avg_route_loss
    if avg_corr_loss is not None:
        latest.avg_corr_loss = avg_corr_loss


def run_training_loop(
    *,
    accelerator: Accelerator,
    config: TrainConfig,
    output_dir: Path,
    dataset: LatentSongDataset,
    dataloader: DataLoader,
    validation_dataloader: DataLoader | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    discriminator: torch.nn.Module | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    ema_teacher: EMATeacher | None,
    settings: TrainerRuntimeSettings,
    sequence_plan: SequenceTrainingPlan,
    initial_global_step: int,
    initial_epoch: int,
    initial_resume_batches_seen: int,
) -> None:
    """Execute the long-running train loop, including periodic checkpointing."""
    progress_bar = _create_progress_bar(
        accelerator=accelerator,
        max_steps=config.training.max_steps,
        initial_global_step=initial_global_step,
    )
    latest = LatestAverages()
    running_losses = RunningLossState.create(device=accelerator.device)

    global_step = int(initial_global_step)
    epoch = int(initial_epoch)
    resume_batches_seen = int(initial_resume_batches_seen)

    while global_step < config.training.max_steps:
        dataset.set_epoch(epoch)
        total_batches_this_epoch = len(dataloader)
        if total_batches_this_epoch == 0:
            raise RuntimeError(
                "Dataloader is empty. Decrease batch size or set data.drop_last=false."
            )

        (
            epoch_iterator,
            batches_seen_in_epoch,
            resume_batches_seen,
            skip_epoch,
            resume_log_message,
        ) = _resolve_epoch_iterator(
            accelerator=accelerator,
            dataloader=dataloader,
            resume_batches_seen=resume_batches_seen,
            total_batches_this_epoch=total_batches_this_epoch,
        )
        if skip_epoch:
            epoch += 1
            continue
        if resume_log_message is not None:
            _log_main(
                accelerator=accelerator,
                progress_bar=progress_bar,
                message=resume_log_message,
            )

        for batch in epoch_iterator:
            batches_seen_in_epoch += 1
            if global_step >= config.training.max_steps:
                break

            step_result = _run_training_step(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                batch=batch,
                sequence_plan=sequence_plan,
                global_step=global_step,
                seed=config.seed,
                settings=settings,
                grad_clip_norm=config.training.grad_clip_norm,
                config=config,
            )
            loss = step_result.loss
            t_eff = step_result.t_eff
            num_windows = step_result.num_windows
            loss_d_step = step_result.loss_d_step
            loss_adv_step = step_result.loss_adv_step
            loss_route_step = step_result.loss_route_step
            loss_corr_step = step_result.loss_corr_step
            gan_lambda_adv_step = step_result.gan_lambda_adv_step

            if accelerator.sync_gradients:
                if ema_teacher is not None:
                    ema_teacher.update(accelerator.unwrap_model(model))
                global_step += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                update_running_losses(
                    state=running_losses,
                    loss=loss,
                    use_gan=settings.use_gan,
                    loss_d_step=loss_d_step,
                    loss_adv_step=loss_adv_step,
                    loss_route_step=loss_route_step,
                    loss_corr_step=loss_corr_step,
                )
                if progress_bar is not None:
                    step_loss_value = float(loss.detach().item())
                    progress_bar.set_postfix(
                        _build_step_postfix(
                            optimizer=optimizer,
                            step_loss_value=step_loss_value,
                            t_eff=t_eff,
                            num_windows=num_windows,
                            use_gan=settings.use_gan,
                            loss_d_step=loss_d_step,
                            loss_adv_step=loss_adv_step,
                            gan_lambda_adv_step=gan_lambda_adv_step,
                            loss_route_step=loss_route_step,
                            loss_corr_step=loss_corr_step,
                            latest_avg_loss=latest.avg_loss,
                            latest_avg_d_loss=latest.avg_d_loss,
                            latest_avg_adv_loss=latest.avg_adv_loss,
                            latest_avg_route_loss=latest.avg_route_loss,
                            latest_avg_corr_loss=latest.avg_corr_loss,
                        ),
                        refresh=False,
                    )

                if global_step % config.training.log_every == 0:
                    step_loss_value = float(loss.detach().item())
                    reduced_averages = compute_reduced_averages(
                        accelerator=accelerator,
                        state=running_losses,
                        use_gan=settings.use_gan,
                        use_channel_aux_losses=settings.use_channel_aux_losses,
                    )
                    avg_loss = reduced_averages.avg_loss
                    avg_d_loss = reduced_averages.avg_d_loss
                    avg_adv_loss = reduced_averages.avg_adv_loss
                    avg_route_loss = reduced_averages.avg_route_loss
                    avg_corr_loss = reduced_averages.avg_corr_loss
                    _update_latest_averages(
                        latest=latest,
                        avg_loss=avg_loss,
                        avg_d_loss=avg_d_loss,
                        avg_adv_loss=avg_adv_loss,
                        avg_route_loss=avg_route_loss,
                        avg_corr_loss=avg_corr_loss,
                    )
                    if progress_bar is not None:
                        progress_bar.set_postfix(
                            _build_log_postfix(
                                optimizer=optimizer,
                                step_loss_value=step_loss_value,
                                avg_loss=avg_loss,
                                t_eff=t_eff,
                                num_windows=num_windows,
                                use_gan=settings.use_gan,
                                loss_d_step=loss_d_step,
                                loss_adv_step=loss_adv_step,
                                gan_lambda_adv_step=gan_lambda_adv_step,
                                loss_route_step=loss_route_step,
                                loss_corr_step=loss_corr_step,
                                avg_d_loss=avg_d_loss,
                                avg_adv_loss=avg_adv_loss,
                                avg_route_loss=avg_route_loss,
                                avg_corr_loss=avg_corr_loss,
                            ),
                            refresh=False,
                        )

                    cond_stereo, cond_mono, cond_downmix = _compute_conditioning_counts(
                        batch=batch,
                        accelerator=accelerator,
                    )

                    message = _build_step_message(
                        global_step=global_step,
                        epoch=epoch,
                        batches_seen_in_epoch=batches_seen_in_epoch,
                        total_batches_this_epoch=total_batches_this_epoch,
                        step_loss_value=step_loss_value,
                        avg_loss=avg_loss,
                        optimizer=optimizer,
                        t_eff=t_eff,
                        num_windows=num_windows,
                        use_gan=settings.use_gan,
                        loss_d_step=loss_d_step,
                        loss_adv_step=loss_adv_step,
                        gan_lambda_adv_step=gan_lambda_adv_step,
                        avg_d_loss=avg_d_loss,
                        avg_adv_loss=avg_adv_loss,
                        loss_route_step=loss_route_step,
                        loss_corr_step=loss_corr_step,
                        avg_route_loss=avg_route_loss,
                        avg_corr_loss=avg_corr_loss,
                        cond_stereo=cond_stereo,
                        cond_mono=cond_mono,
                        cond_downmix=cond_downmix,
                    )
                    _log_main(
                        accelerator=accelerator,
                        progress_bar=progress_bar,
                        message=message,
                    )
                    reset_running_losses(
                        state=running_losses,
                        use_gan=settings.use_gan,
                        use_channel_aux_losses=settings.use_channel_aux_losses,
                    )

                if global_step % config.training.checkpoint_every == 0:
                    ckpt_dir = _save_checkpoint(
                        output_dir=output_dir,
                        accelerator=accelerator,
                        global_step=global_step,
                        epoch=epoch,
                        batches_seen_in_epoch=batches_seen_in_epoch,
                        max_to_keep=config.training.max_checkpoints_to_keep,
                        discriminator=discriminator,
                        discriminator_optimizer=discriminator_optimizer,
                    )
                    _log_main(
                        accelerator=accelerator,
                        progress_bar=progress_bar,
                        message=f"checkpoint={ckpt_dir}",
                    )

                should_run_validation = _should_run_validation(
                    settings=settings,
                    global_step=global_step,
                )
                if (
                    should_run_validation
                    and settings.run_validation
                    and validation_dataloader is not None
                ):
                    val_loss, val_batches = _run_latent_validation(
                        accelerator=accelerator,
                        model=model,
                        dataloader=validation_dataloader,
                        config=config,
                        seq_choices_frames=sequence_plan.seq_choices_frames,
                        max_choice_frames=sequence_plan.max_choice_frames,
                        window_frames=sequence_plan.window_frames,
                        overlap_frames=sequence_plan.overlap_frames,
                        detach_memory=sequence_plan.detach_memory,
                        global_step=global_step,
                        window_metadata=sequence_plan.window_metadata,
                    )
                    _log_main(
                        accelerator=accelerator,
                        progress_bar=progress_bar,
                        message=(
                            f"validation step={global_step} "
                            f"val_loss={val_loss:.6f} "
                            f"batches={val_batches}"
                        ),
                    )

                if should_run_validation and settings.run_validation_generations:
                    accelerator.wait_for_everyone()
                    generated_count = 0
                    generation_error_count = 0
                    if accelerator.is_main_process:
                        generated_count, generation_error_count = (
                            _run_generation_validation(
                                accelerator=accelerator,
                                model=model,
                                config=config,
                                global_step=global_step,
                            )
                        )
                        _log_main(
                            accelerator=accelerator,
                            progress_bar=progress_bar,
                            message=(
                                f"validation_generations step={global_step} "
                                f"generated={generated_count} "
                                f"errors={generation_error_count}"
                            ),
                        )
                    accelerator.wait_for_everyone()

                if global_step >= config.training.max_steps:
                    break

        epoch += 1

    final_dir = _save_checkpoint(
        output_dir=output_dir,
        accelerator=accelerator,
        global_step=global_step,
        epoch=epoch,
        batches_seen_in_epoch=0,
        max_to_keep=config.training.max_checkpoints_to_keep,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
    )
    _log_main(
        accelerator=accelerator,
        progress_bar=progress_bar,
        message=f"Training complete. final_checkpoint={final_dir}",
    )
    if progress_bar is not None:
        progress_bar.close()
