"""Validation helpers for latent-loss and audio generation checks."""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import LatentSongDataset
from .losses import _compute_batch_flow_matching_loss

_AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a"}


def _build_validation_dataset(
    config: TrainConfig,
    training_dataset: LatentSongDataset,
) -> LatentSongDataset:
    """Construct validation dataset mirroring training sequence/window semantics."""
    validation_dataset_root = config.training.validation_dataset_root
    validation_dataset_path = config.training.validation_dataset_path
    if validation_dataset_root is None or validation_dataset_path is None:
        raise ValueError(
            "Validation dataset root/path are required when run_validation is enabled."
        )

    return LatentSongDataset(
        dataset_root=validation_dataset_root,
        manifest_path=validation_dataset_path,
        sample_artifact_mode=config.data.sample_artifact_mode,
        segment_seconds=config.data.segment_seconds,
        latent_fps=config.data.latent_fps,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=config.data.cache_size,
        shuffle_segments_within_epoch=False,
        seed=config.seed + 100_000,
        sequence_seconds=training_dataset.sequence_seconds,
        stride_seconds=training_dataset.stride_seconds,
        sequence_mode=training_dataset.sequence_mode,
        full_song_max_seconds=training_dataset.full_song_max_seconds,
    )


@torch.no_grad()
def _run_latent_validation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainConfig,
    seq_choices_frames: list[int],
    max_choice_frames: int,
    window_frames: int,
    overlap_frames: int,
    detach_memory: bool,
    global_step: int,
    window_metadata: dict[int, tuple[list[int], list[torch.Tensor]]] | None = None,
) -> tuple[float, int]:
    """Evaluate mean latent validation loss across the validation dataloader."""
    was_training = model.training
    model.eval()
    try:
        local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
        local_batch_count = torch.zeros((), device=accelerator.device, dtype=torch.long)

        for val_batch_idx, batch in enumerate(dataloader):
            loss, _, _, _ = _compute_batch_flow_matching_loss(
                accelerator=accelerator,
                model=model,
                batch=batch,
                seq_choices_frames=seq_choices_frames,
                max_choice_frames=max_choice_frames,
                window_frames=window_frames,
                overlap_frames=overlap_frames,
                randomize_per_batch=False,
                detach_memory=detach_memory,
                global_step=global_step + val_batch_idx,
                seed=config.seed + 1_000_000,
                window_metadata=window_metadata,
                force_full_sequence=True,
                scheduled_sampling_config=config.training,
                enable_scheduled_sampling=False,
            )
            local_loss_sum += loss.detach().to(dtype=torch.float64)
            local_batch_count += 1

        total_loss_sum = accelerator.reduce(local_loss_sum, reduction="sum")
        total_batch_count = accelerator.reduce(local_batch_count, reduction="sum")
        total_batches = int(total_batch_count.item())
        if total_batches <= 0:
            raise RuntimeError("Validation dataloader is empty.")

        return float((total_loss_sum / float(total_batches)).item()), total_batches
    finally:
        if was_training:
            model.train()


def _list_validation_audio_files(input_root: Path) -> list[Path]:
    """Recursively list supported audio files for generation validation."""
    if not input_root.exists():
        raise FileNotFoundError(
            f"Validation generation input path not found: {input_root}"
        )
    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Validation generation input path must be a directory: {input_root}"
        )

    files = [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
    ]
    return files


@torch.no_grad()
def _run_generation_validation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: TrainConfig,
    global_step: int,
) -> tuple[int, int]:
    """Run periodic audio generation validation and return success/error counts."""
    from stereo2spatial.codecs.ear_vae import (
        decode_channels_independent,
        load_vae,
        vae_encode,
    )
    from stereo2spatial.inference import (
        generate_spatial_latent,
        read_audio_channels_first,
        resolve_chunk_frames,
    )
    from stereo2spatial.inference.audio import write_audio_channels_first

    input_root = Path(config.training.validation_generation_input_path or "")
    output_root = Path(config.training.validation_generation_output_path or "")
    vae_checkpoint_path = config.training.validation_generation_vae_checkpoint_path
    vae_config_path = config.training.validation_generation_vae_config_path
    if vae_checkpoint_path is None:
        raise ValueError(
            "training.validation_generation_vae_checkpoint_path must be set for "
            "run_validation_generations=true"
        )

    audio_paths = _list_validation_audio_files(input_root)
    if not audio_paths:
        return 0, 0

    raw_model = accelerator.unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    try:
        model_device: torch.device
        try:
            model_device = next(raw_model.parameters()).device
        except StopIteration:
            model_device = accelerator.device

        vae = load_vae(
            vae_checkpoint_path=vae_checkpoint_path,
            config_path=vae_config_path,
            device=model_device,
            torch_dtype=torch.float32,
        )

        sample_rate = 48_000
        chunk_seconds = float(config.data.segment_seconds)
        overlap_seconds = 0.5
        solver = "dopri5"
        solver_steps = 64
        solver_rtol = 1e-5
        solver_atol = 1e-5

        generated_count = 0
        error_count = 0
        step_root = output_root / f"step_{global_step:07d}"
        step_root.mkdir(parents=True, exist_ok=True)

        for generation_idx in range(int(config.training.num_valid_generations)):
            seed = int(config.training.validation_generation_seed) + generation_idx
            seed_root = step_root / f"seed_{seed}"

            for input_audio_path in audio_paths:
                try:
                    audio, actual_sample_rate = read_audio_channels_first(
                        audio_path=input_audio_path,
                        target_sample_rate=sample_rate,
                    )
                    if audio.shape[0] not in {1, 2}:
                        raise ValueError(
                            f"Input must be mono or stereo. Got channels={audio.shape[0]}"
                        )

                    cond_latent = vae_encode(
                        vae=vae,
                        audio=audio,
                        sample_rate=actual_sample_rate,
                        use_sample=False,
                        use_chunked_encode=True,
                        chunk_size_samples=None,
                        overlap_samples=None,
                        duplicate_mono_to_stereo=True,
                        offload_latent_to_cpu=False,
                        show_progress=False,
                        device=model_device,
                    )
                    if cond_latent.dim() != 2:
                        raise ValueError(
                            "Expected conditioning latent [D,T], "
                            f"got {tuple(cond_latent.shape)}"
                        )

                    cond_latent = cond_latent.unsqueeze(0).contiguous()  # [1,D,T]
                    latent_fps = (
                        float(actual_sample_rate)
                        * float(cond_latent.shape[-1])
                        / float(audio.shape[-1])
                    )
                    chunk_frames, overlap_frames = resolve_chunk_frames(
                        cond_latent_frames=cond_latent.shape[-1],
                        latent_fps=latent_fps,
                        chunk_seconds=chunk_seconds,
                        overlap_seconds=overlap_seconds,
                    )

                    pred_latent = generate_spatial_latent(
                        model=raw_model,
                        cond_latent=cond_latent.to(model_device),
                        chunk_frames=chunk_frames,
                        overlap_frames=overlap_frames,
                        solver=solver,
                        solver_steps=solver_steps,
                        solver_rtol=solver_rtol,
                        solver_atol=solver_atol,
                        seed=seed,
                    )
                    decoded = decode_channels_independent(
                        vae=vae,
                        channel_latents=pred_latent.to(model_device),
                        use_chunked_decode=True,
                        chunk_size_frames=2048,
                        overlap_frames=256,
                        offload_wav_to_cpu=True,
                        reduction="mean",
                        show_progress=False,
                        device=model_device,
                    ).float()

                    rel_path = input_audio_path.relative_to(input_root)
                    output_audio_path = (seed_root / rel_path).with_suffix(".wav")
                    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                    write_audio_channels_first(
                        audio_path=output_audio_path,
                        audio=decoded.cpu(),
                        sample_rate=actual_sample_rate,
                    )
                    generated_count += 1
                except Exception as error:
                    error_count += 1
                    print(
                        "[validation_generation_error] "
                        f"step={global_step} input={input_audio_path} seed={seed} error={error}"
                    )

        return generated_count, error_count
    finally:
        if was_training:
            raw_model.train()
