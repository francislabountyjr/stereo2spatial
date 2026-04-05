"""Top-level distributed training orchestration for stereo2spatial."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .checkpointing import (
    _checkpoint_has_ema_state,
    _load_model_weights_only,
    _load_resume_state,
    _resolve_model_init_checkpoint,
    _resolve_resume_checkpoint,
)
from .components import build_training_components
from .config import TrainConfig
from .ema import EMATeacher
from .runtime import (
    _create_dataloader,
    _disable_inductor_cudagraphs_if_possible,
)
from .sequence_plan import build_sequence_training_plan
from .trainer_gan import _prepare_discriminator_components
from .trainer_logging import log_training_setup
from .trainer_loop import run_training_loop
from .trainer_settings import resolve_trainer_runtime_settings
from .validation import _build_validation_dataset


def _write_resolved_config(output_dir: Path, config: TrainConfig) -> None:
    """Persist fully-resolved config for reproducibility."""
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=True)


def _resolve_checkpoint_inputs(
    *,
    config: TrainConfig,
    resume_from_checkpoint: str | None,
    init_from_checkpoint: str | None,
    accelerator: Accelerator,
) -> tuple[str | None, str | None]:
    """Resolve CLI overrides against config defaults for resume/init checkpoints."""
    resume_value = (
        resume_from_checkpoint
        if resume_from_checkpoint is not None
        else config.training.resume_from_checkpoint
    )
    init_value = (
        init_from_checkpoint
        if init_from_checkpoint is not None
        else getattr(config.training, "init_from_checkpoint", None)
    )
    if resume_value is not None and str(resume_value).strip():
        if init_value is not None and str(init_value).strip():
            if accelerator.is_main_process:
                print(
                    "[resume_precedence] resume_from_checkpoint is set; "
                    "ignoring init_from_checkpoint."
                )
            init_value = None
    return resume_value, init_value


def _maybe_compile_model(
    *,
    model: torch.nn.Module,
    config: TrainConfig,
    accelerator: Accelerator,
) -> tuple[torch.nn.Module, bool, str, bool]:
    """Compile model when requested and supported by runtime."""
    compile_requested = bool(getattr(config.training, "compile_model", False))
    compile_mode = (
        str(getattr(config.training, "compile_mode", "default")).strip().lower()
    )
    model_was_compiled = False
    if not compile_requested:
        return model, compile_requested, compile_mode, model_was_compiled

    if hasattr(torch, "compile"):
        try:
            # This training loop runs multiple model forwards (windowed chunks)
            # before a single backward pass. Disable cudagraphs to avoid
            # output-buffer overwrite errors across those forwards.
            _disable_inductor_cudagraphs_if_possible()
            model = cast(torch.nn.Module, torch.compile(model, mode=compile_mode))
            model_was_compiled = True
        except Exception as error:
            if accelerator.is_main_process:
                print(
                    "[compile_warning] "
                    f"torch.compile failed ({error}). Continuing without compile."
                )
    elif accelerator.is_main_process:
        print(
            "[compile_warning] torch.compile is unavailable in this PyTorch build. "
            "Continuing without compile."
        )

    return model, compile_requested, compile_mode, model_was_compiled


def _maybe_build_ema_teacher(
    *,
    config: TrainConfig,
    model: torch.nn.Module,
) -> EMATeacher | None:
    """Build EMA teacher when requested by config."""
    if not bool(getattr(config.training, "use_ema", False)):
        return None
    decay = float(getattr(config.training, "ema_decay", 0.999))
    return EMATeacher(
        model=model,
        decay=decay,
        storage_device=str(getattr(config.training, "ema_device", "accelerator")),
        cpu_only=bool(getattr(config.training, "ema_cpu_only", False)),
    )


def train(
    config: TrainConfig,
    resume_from_checkpoint: str | None = None,
    init_from_checkpoint: str | None = None,
) -> None:
    """Run distributed training until `config.training.max_steps` is reached."""
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.grad_accum_steps,
        mixed_precision=config.training.mixed_precision.lower(),
    )

    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_resolved_config(output_dir=output_dir, config=config)
    accelerator.wait_for_everyone()

    set_seed(config.seed, device_specific=False)
    resume_value, init_value = _resolve_checkpoint_inputs(
        config=config,
        resume_from_checkpoint=resume_from_checkpoint,
        init_from_checkpoint=init_from_checkpoint,
        accelerator=accelerator,
    )
    resume_checkpoint_path = _resolve_resume_checkpoint(
        resume_from_checkpoint=resume_value,
        output_dir=output_dir,
    )
    init_checkpoint_path = _resolve_model_init_checkpoint(
        init_from_checkpoint=init_value,
        output_dir=output_dir,
    )

    dataset, built_model, optimizer = build_training_components(config)
    model: torch.nn.Module = built_model
    ema_teacher = _maybe_build_ema_teacher(
        config=config,
        model=built_model,
    )
    if init_checkpoint_path is not None:
        _load_model_weights_only(model=model, checkpoint_path=init_checkpoint_path)
        if accelerator.is_main_process:
            print(f"Initialized model weights from checkpoint: {init_checkpoint_path}")
        if ema_teacher is not None:
            ema_teacher.copy_from(model)

    model, compile_requested, compile_mode, model_was_compiled = _maybe_compile_model(
        model=model,
        config=config,
        accelerator=accelerator,
    )

    dataset.set_epoch(0)
    dataloader = _create_dataloader(dataset, config, for_training=True)
    validation_dataloader: DataLoader | None = None

    settings = resolve_trainer_runtime_settings(config)

    if settings.run_validation:
        validation_dataset = _build_validation_dataset(
            config=config,
            training_dataset=dataset,
        )
        validation_dataset.set_epoch(0)
        validation_dataloader = _create_dataloader(
            validation_dataset,
            config,
            drop_last=False,
            for_training=False,
        )

    if validation_dataloader is None:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model, optimizer, dataloader, validation_dataloader = accelerator.prepare(
            model, optimizer, dataloader, validation_dataloader
        )
    register_for_checkpointing = getattr(accelerator, "register_for_checkpointing", None)
    ema_registered_for_checkpointing = False
    resume_missing_ema_state = False
    if ema_teacher is not None:
        raw_model = accelerator.unwrap_model(model)
        reference_param = next(raw_model.parameters())
        ema_teacher.configure_runtime(
            accelerator_device=reference_param.device,
            dtype=reference_param.dtype,
        )
        if ema_teacher.storage_device == "cpu" and not ema_teacher.cpu_only:
            try:
                ema_teacher.pin_memory()
            except Exception as error:
                if accelerator.is_main_process:
                    print(
                        "[ema_warning] "
                        f"Failed to pin EMA teacher to CPU: {error}"
                    )
        ema_teacher.copy_from(raw_model)
        should_register_ema_pre_resume = True
        if resume_checkpoint_path is not None:
            has_ema_state = _checkpoint_has_ema_state(resume_checkpoint_path)
            if not has_ema_state:
                should_register_ema_pre_resume = False
                resume_missing_ema_state = True
                if accelerator.is_main_process:
                    print(
                        "[ema_resume] Resuming from checkpoint without EMA state; "
                        "initializing EMA from resumed student weights."
                    )
        if callable(register_for_checkpointing) and should_register_ema_pre_resume:
            register_for_checkpointing(ema_teacher)
            ema_registered_for_checkpointing = True
    model.train()

    global_step, epoch, resume_batches_seen, resumed_ckpt = _load_resume_state(
        accelerator=accelerator,
        output_dir=output_dir,
        resume_from_checkpoint=resume_value,
        resolved_checkpoint_path=resume_checkpoint_path,
    )
    if ema_teacher is not None and resumed_ckpt is not None and resume_missing_ema_state:
        ema_teacher.copy_from(accelerator.unwrap_model(model))
    if (
        ema_teacher is not None
        and callable(register_for_checkpointing)
        and not ema_registered_for_checkpointing
    ):
        register_for_checkpointing(ema_teacher)
        ema_registered_for_checkpointing = True

    discriminator, discriminator_optimizer = _prepare_discriminator_components(
        accelerator=accelerator,
        config=config,
        settings=settings,
        resumed_ckpt=resumed_ckpt,
        init_checkpoint_path=init_checkpoint_path,
    )

    if resumed_ckpt is not None and accelerator.is_main_process:
        print(
            f"Resumed: step={global_step} epoch={epoch} "
            f"batches_seen_in_epoch={resume_batches_seen}"
        )

    log_training_setup(
        accelerator=accelerator,
        config=config,
        dataset=dataset,
        dataloader=dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        settings=settings,
        compile_requested=compile_requested,
        compile_mode=compile_mode,
        model_was_compiled=model_was_compiled,
        resumed_checkpoint=resumed_ckpt,
        init_checkpoint=init_checkpoint_path,
        discriminator_optimizer=discriminator_optimizer,
    )

    sequence_plan = build_sequence_training_plan(config=config, dataset=dataset)
    run_training_loop(
        accelerator=accelerator,
        config=config,
        output_dir=output_dir,
        dataset=dataset,
        dataloader=dataloader,
        validation_dataloader=validation_dataloader,
        model=model,
        optimizer=optimizer,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        ema_teacher=ema_teacher,
        settings=settings,
        sequence_plan=sequence_plan,
        initial_global_step=global_step,
        initial_epoch=epoch,
        initial_resume_batches_seen=resume_batches_seen,
    )
