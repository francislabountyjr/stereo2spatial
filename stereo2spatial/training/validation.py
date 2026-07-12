"""Validation helpers for signal-loss and audio generation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from stereo2spatial.modeling import SpatialModel, is_legacy_vae_model

from .config import TrainConfig
from .dataset import WaveformSongDataset
from .ema import EMATeacher
from .latent_dataset import LatentSongDataset
from .losses import _compute_batch_flow_matching_loss

_AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a"}


def _build_validation_dataset(
    config: TrainConfig,
    training_dataset: WaveformSongDataset | LatentSongDataset,
) -> WaveformSongDataset | LatentSongDataset:
    """Construct validation dataset mirroring training sequence/window semantics."""
    validation_dataset_root = config.training.validation_dataset_root
    validation_dataset_path = config.training.validation_dataset_path
    if validation_dataset_root is None or validation_dataset_path is None:
        raise ValueError(
            "Validation dataset root/path are required when run_validation is enabled."
        )

    if is_legacy_vae_model(config):
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
            shuffle_segments_within_song=False,
            seed=config.seed + 100_000,
            sample_exclusion_path=cast(Any, config.data.sample_exclusion_path),
            sequence_seconds=training_dataset.sequence_seconds,
            stride_seconds=training_dataset.stride_seconds,
            sequence_mode=training_dataset.sequence_mode,
            full_song_max_seconds=training_dataset.full_song_max_seconds,
        )

    effective_sample_rate = int(
        getattr(config.data, "training_sample_rate", None) or config.data.sample_rate
    )
    return WaveformSongDataset(
        dataset_root=validation_dataset_root,
        manifest_path=validation_dataset_path,
        sample_artifact_mode=config.data.sample_artifact_mode,
        segment_seconds=config.data.segment_seconds,
        patch_fps=float(effective_sample_rate) / float(config.model.patch_size),
        patch_size=config.model.patch_size,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=config.data.cache_size,
        shuffle_segments_within_epoch=False,
        shuffle_segments_within_song=False,
        seed=config.seed + 100_000,
        materialize_cached_signals=config.data.materialize_cached_signals,
        sequence_seconds=training_dataset.sequence_seconds,
        stride_seconds=training_dataset.stride_seconds,
        sample_rate=config.data.sample_rate,
        training_sample_rate=effective_sample_rate,
        sequence_mode=training_dataset.sequence_mode,
        full_song_max_seconds=training_dataset.full_song_max_seconds,
        amplitude_lift_enabled=config.data.amplitude_lift_enabled,
        amplitude_lift_mode=config.data.amplitude_lift_mode,
        amplitude_lift_reference=config.data.amplitude_lift_reference,
        amplitude_lift_target_rms=config.data.amplitude_lift_target_rms,
        amplitude_lift_scale=config.data.amplitude_lift_scale,
        amplitude_lift_clip_value=config.data.amplitude_lift_clip_value,
        amplitude_lift_gain_power=getattr(
            config.data, "amplitude_lift_gain_power", 1.0
        ),
        amplitude_lift_gain_min_value=getattr(
            config.data, "amplitude_lift_gain_min_value", None
        ),
        amplitude_lift_waveform_clamp=config.data.amplitude_lift_waveform_clamp,
        amplitude_lift_peak_limit=config.data.amplitude_lift_peak_limit,
        amplitude_lift_peak_rescale_min_rms=(
            config.data.amplitude_lift_peak_rescale_min_rms
        ),
        amplitude_lift_eps=config.data.amplitude_lift_eps,
        min_source_rms=getattr(config.data, "min_source_rms", None),
    )


@torch.no_grad()
def _run_signal_validation(
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
    """Evaluate mean signal validation loss across the validation dataloader."""
    was_training = model.training
    effective_sample_rate = int(
        getattr(config.data, "training_sample_rate", None) or config.data.sample_rate
    )
    model.eval()
    try:
        local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
        local_batch_count = torch.zeros((), device=accelerator.device, dtype=torch.long)

        for val_batch_idx, batch in enumerate(dataloader):
            loss, _, _, _, _ = _compute_batch_flow_matching_loss(
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
                downmix_consistency_weight=config.training.downmix_consistency_weight,
                downmix_consistency_loss=config.training.downmix_consistency_loss,
                downmix_channel_order=config.training.downmix_channel_order,
                mrstft_loss_weight=config.training.mrstft_loss_weight,
                mrstft_fft_sizes=config.training.mrstft_fft_sizes,
                mrstft_hop_lengths=config.training.mrstft_hop_lengths,
                mrstft_win_lengths=config.training.mrstft_win_lengths,
                mrstft_sc_weight=config.training.mrstft_sc_weight,
                mrstft_log_mag_weight=config.training.mrstft_log_mag_weight,
                mrstft_eps=config.training.mrstft_eps,
                waveform_mse_loss_weight=config.training.waveform_mse_loss_weight,
                waveform_l1_loss_weight=config.training.waveform_l1_loss_weight,
                waveform_charbonnier_loss_weight=(
                    config.training.waveform_charbonnier_loss_weight
                ),
                waveform_charbonnier_eps=config.training.waveform_charbonnier_eps,
                x_pred_v_loss_weight=config.training.x_pred_v_loss_weight,
                perceptual_loss_weight=config.training.perceptual_loss_weight,
                perceptual_sample_rate=effective_sample_rate,
                perceptual_n_fft=config.training.perceptual_n_fft,
                perceptual_hop_length=config.training.perceptual_hop_length,
                perceptual_win_length=config.training.perceptual_win_length,
                perceptual_n_mels=config.training.perceptual_n_mels,
                perceptual_f_min=config.training.perceptual_f_min,
                perceptual_f_max=config.training.perceptual_f_max,
                perceptual_band_weight=config.training.perceptual_band_weight,
                perceptual_band_low_hz=config.training.perceptual_band_low_hz,
                perceptual_band_high_hz=config.training.perceptual_band_high_hz,
                perceptual_eps=config.training.perceptual_eps,
                binaural_ild_loss_weight=config.training.binaural_ild_loss_weight,
                binaural_ipd_loss_weight=config.training.binaural_ipd_loss_weight,
                binaural_ccf_loss_weight=config.training.binaural_ccf_loss_weight,
                binaural_frame_ild_loss_weight=(
                    config.training.binaural_frame_ild_loss_weight
                ),
                binaural_frame_ild_frame_size=(
                    config.training.binaural_frame_ild_frame_size
                ),
                binaural_frame_ild_hop_size=(
                    config.training.binaural_frame_ild_hop_size
                ),
                binaural_frame_ild_silence_threshold=(
                    config.training.binaural_frame_ild_silence_threshold
                ),
                binaural_frame_ild_max_weight=(
                    config.training.binaural_frame_ild_max_weight
                ),
                binaural_mid_side_loss_weight=(
                    config.training.binaural_mid_side_loss_weight
                ),
                binaural_mid_side_loss_type=(
                    config.training.binaural_mid_side_loss_type
                ),
                binaural_mid_side_mid_weight=(
                    config.training.binaural_mid_side_mid_weight
                ),
                binaural_mid_side_side_weight=(
                    config.training.binaural_mid_side_side_weight
                ),
                binaural_mid_side_charbonnier_eps=(
                    config.training.binaural_mid_side_charbonnier_eps
                ),
                binaural_loss_warmup_steps=config.training.binaural_loss_warmup_steps,
                binaural_sample_rate=effective_sample_rate,
                binaural_loss_eps=config.training.binaural_loss_eps,
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
    ema_teacher: EMATeacher | None = None,
) -> tuple[int, int]:
    """Run periodic audio generation validation and return success/error counts."""
    from stereo2spatial.codecs.ear_vae import (
        decode_channels_independent,
        encode_channels_independent,
        load_vae,
        vae_encode,
    )
    from stereo2spatial.common.amplitude_lift import (
        amplitude_lift_log_gain,
        apply_amplitude_lift,
        resolve_amplitude_lift_gain,
        undo_amplitude_lift,
        undo_wavflow_output_lift,
        wavflow_source_transform,
    )
    from stereo2spatial.inference.audio import (
        read_audio_channels_first,
        write_audio_channels_first,
    )
    from stereo2spatial.inference.runner import (
        RequestedSolverName,
        _patch_audio,
        _prepare_conditioning_audio,
        _resolve_inference_solver,
        _unpatch_audio,
    )
    from stereo2spatial.inference.sampling import resolve_chunk_frames
    from stereo2spatial.inference.timestep_sampling import (
        generate_spatial_signal_timestep_major,
    )

    input_root = Path(config.training.validation_generation_input_path or "")
    output_root = Path(config.training.validation_generation_output_path or "")
    audio_paths = _list_validation_audio_files(input_root)
    if not audio_paths:
        return 0, 0

    raw_model = cast(
        SpatialModel,
        ema_teacher.model
        if ema_teacher is not None
        else accelerator.unwrap_model(model),
    )
    was_training = raw_model.training
    raw_model.eval()
    try:
        try:
            model_device = next(raw_model.parameters()).device
        except StopIteration:
            model_device = accelerator.device

        legacy_vae = is_legacy_vae_model(config)
        vae: torch.nn.Module | None = None
        if legacy_vae:
            vae_checkpoint_path = (
                config.training.validation_generation_vae_checkpoint_path
            )
            if vae_checkpoint_path is None:
                raise ValueError(
                    "validation_generation_vae_checkpoint_path is required for "
                    "legacy_vae validation generation"
                )
            vae = load_vae(
                vae_checkpoint_path=vae_checkpoint_path,
                config_path=config.training.validation_generation_vae_config_path,
                device=model_device,
                torch_dtype=torch.float32,
            )

        sample_rate = int(
            getattr(config.data, "training_sample_rate", None)
            or config.data.sample_rate
        )
        chunk_seconds = (
            float(config.training.validation_generation_chunk_seconds)
            if config.training.validation_generation_chunk_seconds is not None
            else float(config.data.segment_seconds)
        )
        overlap_seconds = float(config.training.validation_generation_overlap_seconds)
        solver = _resolve_inference_solver(
            requested_solver=cast(
                RequestedSolverName,
                config.training.validation_generation_solver,
            )
        )
        solver_steps = int(config.training.validation_generation_solver_steps)
        solver_rtol = float(config.training.validation_generation_solver_rtol)
        solver_atol = float(config.training.validation_generation_solver_atol)

        generated_count = 0
        error_count = 0
        step_root = output_root / f"step_{global_step:07d}"
        step_root.mkdir(parents=True, exist_ok=True)
        output_suffix = ".flac" if int(config.model.target_channels) == 2 else ".wav"

        for generation_idx in range(int(config.training.num_valid_generations)):
            seed = int(config.training.validation_generation_seed) + generation_idx
            seed_root = step_root / f"seed_{seed}"

            for input_audio_path in audio_paths:
                try:
                    audio, actual_sample_rate = read_audio_channels_first(
                        audio_path=input_audio_path,
                        target_sample_rate=sample_rate,
                    )
                    conditioning_audio = (
                        audio.float()
                        if legacy_vae
                        else _prepare_conditioning_audio(
                            audio.float(),
                            cond_channels=int(config.model.cond_channels),
                        )
                    )
                    amplitude_lift_gain = None
                    amplitude_lift_log_gain_tensor: torch.Tensor | None = None
                    amplitude_lift_mode = (
                        str(getattr(config.data, "amplitude_lift_mode", "rms"))
                        .strip()
                        .lower()
                    )
                    _lift_scale = float(
                        getattr(config.data, "amplitude_lift_scale", 3.0)
                    )
                    _lift_clip = getattr(config.data, "amplitude_lift_clip_value", 4.0)
                    _lift_power = float(
                        getattr(config.data, "amplitude_lift_gain_power", 1.0)
                    )
                    _lift_min = getattr(
                        config.data, "amplitude_lift_gain_min_value", None
                    )
                    _lift_output_lufs = float(
                        getattr(config.data, "amplitude_lift_output_lufs", -23.0)
                    )
                    if bool(getattr(config.data, "amplitude_lift_enabled", False)):
                        lift_reference = (
                            str(
                                getattr(
                                    config.data, "amplitude_lift_reference", "source"
                                )
                            )
                            .strip()
                            .lower()
                        )
                        if (
                            amplitude_lift_mode != "wavflow"
                            and lift_reference != "source"
                        ):
                            raise ValueError(
                                "Validation generation amplitude lifting requires "
                                "data.amplitude_lift_reference='source' because "
                                "target audio is unavailable."
                            )
                        if amplitude_lift_mode == "wavflow":
                            (
                                conditioning_audio,
                                amplitude_lift_gain,
                            ) = wavflow_source_transform(
                                conditioning_audio,
                                target_rms=float(
                                    getattr(
                                        config.data,
                                        "amplitude_lift_target_rms",
                                        0.33,
                                    )
                                ),
                                scale=_lift_scale,
                                peak_limit=float(
                                    getattr(
                                        config.data,
                                        "amplitude_lift_peak_limit",
                                        1.0,
                                    )
                                ),
                                eps=float(
                                    getattr(
                                        config.data,
                                        "amplitude_lift_eps",
                                        1.0e-8,
                                    )
                                ),
                            )
                            amplitude_lift_log_gain_tensor = torch.log(
                                amplitude_lift_gain.clamp_min(
                                    float(
                                        getattr(
                                            config.data,
                                            "amplitude_lift_eps",
                                            1.0e-8,
                                        )
                                    )
                                )
                            ).to(model_device)
                        else:
                            amplitude_lift_gain = resolve_amplitude_lift_gain(
                                conditioning_audio,
                                mode=amplitude_lift_mode,
                                target_rms=float(
                                    getattr(
                                        config.data,
                                        "amplitude_lift_target_rms",
                                        0.33,
                                    )
                                ),
                                eps=float(
                                    getattr(config.data, "amplitude_lift_eps", 1.0e-8)
                                ),
                            )
                            _lift_clip = (
                                None if amplitude_lift_mode == "scale" else _lift_clip
                            )
                            conditioning_audio = apply_amplitude_lift(
                                conditioning_audio,
                                gain=amplitude_lift_gain,
                                scale=_lift_scale,
                                clip_value=_lift_clip,
                                gain_power=_lift_power,
                                gain_min_value=_lift_min,
                            )
                            amplitude_lift_log_gain_tensor = amplitude_lift_log_gain(
                                amplitude_lift_gain,
                                scale=_lift_scale,
                                gain_clip_value=_lift_clip,
                                gain_power=_lift_power,
                                gain_min_value=_lift_min,
                                eps=float(
                                    getattr(config.data, "amplitude_lift_eps", 1.0e-8)
                                ),
                            ).to(model_device)
                    input_samples = int(conditioning_audio.shape[-1])
                    if legacy_vae:
                        if vae is None:
                            raise RuntimeError("legacy validation VAE was not loaded")
                        if int(config.model.cond_channels) == 1:
                            encoded = vae_encode(
                                vae=vae,
                                audio=conditioning_audio,
                                sample_rate=actual_sample_rate,
                                use_sample=False,
                                use_chunked_encode=True,
                                duplicate_mono_to_stereo=True,
                                offload_latent_to_cpu=False,
                                device=model_device,
                            )
                            cond_signal = encoded.unsqueeze(0).contiguous()
                        else:
                            cond_signal = encode_channels_independent(
                                vae=vae,
                                audio=_prepare_conditioning_audio(
                                    conditioning_audio,
                                    cond_channels=int(config.model.cond_channels),
                                ),
                                sample_rate=actual_sample_rate,
                                use_sample=False,
                                use_chunked_encode=True,
                                offload_latent_to_cpu=False,
                                device=model_device,
                            )
                        latent_fps = getattr(config.data, "latent_fps", 50.0)
                        patch_fps = (
                            50.0
                            if str(latent_fps).strip().lower() == "auto"
                            else float(latent_fps)
                        )
                    else:
                        cond_signal, input_samples = _patch_audio(
                            conditioning_audio,
                            patch_size=int(config.model.patch_size),
                        )
                        patch_fps = float(actual_sample_rate) / float(
                            config.model.patch_size
                        )
                    requested_chunk_frames = max(
                        1,
                        int(round(chunk_seconds * patch_fps)),
                    )
                    chunk_frames, overlap_frames = resolve_chunk_frames(
                        cond_signal_frames=max(
                            int(cond_signal.shape[-1]),
                            requested_chunk_frames,
                        ),
                        patch_fps=patch_fps,
                        chunk_seconds=chunk_seconds,
                        overlap_seconds=overlap_seconds,
                    )

                    pred_signal = generate_spatial_signal_timestep_major(
                        model=raw_model,
                        cond_signal=cond_signal.to(model_device),
                        chunk_frames=chunk_frames,
                        overlap_frames=overlap_frames,
                        solver=solver,
                        solver_steps=solver_steps,
                        solver_rtol=solver_rtol,
                        solver_atol=solver_atol,
                        seed=seed,
                        amplitude_gain=amplitude_lift_log_gain_tensor,
                        one_step=bool(getattr(config.training, "flow_one_step", False)),
                        one_step_input=str(
                            getattr(config.training, "flow_one_step_input", "zeros")
                        ),
                    )
                    if legacy_vae:
                        assert vae is not None
                        decoded = decode_channels_independent(
                            vae=vae,
                            channel_latents=pred_signal.to(
                                model_device, dtype=torch.float32
                            ),
                            use_chunked_decode=True,
                            chunk_size_frames=2048,
                            overlap_frames=256,
                            offload_wav_to_cpu=True,
                            reduction="mean",
                            device=model_device,
                        )[:, :input_samples]
                    else:
                        decoded = _unpatch_audio(
                            pred_signal.cpu().float(),
                            sample_count=input_samples,
                        )
                    if amplitude_lift_gain is not None:
                        if amplitude_lift_mode == "wavflow":
                            decoded = undo_wavflow_output_lift(
                                decoded,
                                scale=_lift_scale,
                                sample_rate=actual_sample_rate,
                                target_lufs=_lift_output_lufs,
                                eps=float(
                                    getattr(config.data, "amplitude_lift_eps", 1.0e-8)
                                ),
                            )
                        else:
                            decoded = undo_amplitude_lift(
                                decoded,
                                gain=amplitude_lift_gain.cpu(),
                                scale=_lift_scale,
                                eps=float(
                                    getattr(config.data, "amplitude_lift_eps", 1.0e-8)
                                ),
                                clip_value=_lift_clip,
                                gain_power=_lift_power,
                                gain_min_value=_lift_min,
                            )

                    rel_path = input_audio_path.relative_to(input_root)
                    output_audio_path = (seed_root / rel_path).with_suffix(
                        output_suffix
                    )
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
