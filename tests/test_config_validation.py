from __future__ import annotations

import pytest

from stereo2spatial.training.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
    validate_config,
)


def _valid_config() -> TrainConfig:
    return TrainConfig(
        seed=1337,
        output_dir="runs/test",
        data=DataConfig(
            dataset_root="dataset",
            manifest_path="dataset/manifest.jsonl",
            sample_artifact_mode="bundle",
            segment_seconds=10.0,
            sequence_seconds=10.0,
            stride_seconds=5.0,
            sample_rate=48_000,
            mono_probability=0.1,
            downmix_probability=0.1,
            cache_size=8,
            shuffle_segments_within_epoch=True,
            batch_size=1,
            num_workers=0,
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        ),
        model=ModelConfig(
            target_channels=12,
            cond_channels=2,
            patch_size=128,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
            timestep_embed_dim=256,
            timestep_scale=1000.0,
            max_period=10000.0,
            num_memory_tokens=0,
        ),
        training=TrainingConfig(
            max_steps=100,
            grad_accum_steps=1,
            mixed_precision="no",
            compile_model=False,
            compile_mode="default",
            resume_from_checkpoint=None,
            init_from_checkpoint=None,
            grad_clip_norm=1.0,
            log_every=10,
            checkpoint_every=50,
            max_checkpoints_to_keep=3,
            num_epochs_hint=1,
            window_seconds=8.0,
            overlap_seconds=2.0,
            sequence_seconds_choices=[10.0],
            randomize_sequence_per_batch=True,
            detach_memory=True,
            sequence_mode="strided_crops",
            tbptt_windows=0,
            full_song_max_seconds=None,
            require_batch_size_one_for_full_song=True,
            use_gan=False,
            gan_d_lr=1e-4,
            gan_d_beta1=0.0,
            gan_d_beta2=0.9,
            gan_d_base_channels=64,
            gan_d_num_layers=4,
            gan_d_fine_layers=3,
            gan_d_coarse_layers=4,
            gan_d_use_spectral_norm=True,
            gan_use_mask_channel=True,
            gan_ms_w_fine=1.0,
            gan_ms_w_coarse=0.5,
            gan_lambda_adv=1e-3,
            gan_adv_warmup_steps=0,
            gan_r1_gamma=10.0,
            gan_r1_every=16,
            routing_kl_weight=0.0,
            routing_kl_temperature=1.0,
            routing_kl_eps=1e-6,
            corr_weight=0.0,
            corr_eps=1e-6,
            corr_offdiag_only=True,
            corr_use_correlation=True,
            run_validation=False,
            validation_dataset_root=None,
            validation_dataset_path=None,
            validation_steps=0,
            run_validation_generations=False,
            num_valid_generations=1,
            validation_generation_seed=1337,
            validation_generation_input_path=None,
            validation_generation_output_path=None,
        ),
        optimizer=OptimizerConfig(
            type="adamw",
            lr=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            adamw_fused=False,
            adamw_foreach=False,
        ),
        scheduler=SchedulerConfig(
            type="cosine",
            warmup_steps=10,
            min_lr=1e-6,
        ),
    )


def test_validate_config_accepts_valid_payload() -> None:
    config = _valid_config()
    validate_config(config)


def test_validate_config_accepts_song_local_batch_mode() -> None:
    config = _valid_config()
    config.data.batch_mode = "song_local"
    validate_config(config)


def test_validate_config_accepts_flac_artifact_mode() -> None:
    config = _valid_config()
    config.data.sample_artifact_mode = "flac"
    validate_config(config)


def test_validate_config_rejects_unknown_batch_mode() -> None:
    config = _valid_config()
    config.data.batch_mode = "random_song"

    with pytest.raises(ValueError, match="batch_mode"):
        validate_config(config)


def test_validate_config_rejects_invalid_sample_rate() -> None:
    config = _valid_config()
    config.data.sample_rate = 0

    with pytest.raises(ValueError, match="sample_rate"):
        validate_config(config)


def test_validate_config_rejects_training_sample_rate_above_sample_rate() -> None:
    config = _valid_config()
    config.data.training_sample_rate = 96_000

    with pytest.raises(ValueError, match="training_sample_rate"):
        validate_config(config)


def test_validate_config_rejects_invalid_source_resample_augmentation() -> None:
    config = _valid_config()
    config.data.source_resample_aug_enabled = True
    config.data.source_resample_aug_probability = 0.5
    config.data.source_resample_aug_rates = []

    with pytest.raises(ValueError, match="source_resample_augmentation.rates"):
        validate_config(config)


def test_validate_config_rejects_source_resample_weight_mismatch() -> None:
    config = _valid_config()
    config.data.source_resample_aug_enabled = True
    config.data.source_resample_aug_probability = 0.5
    config.data.source_resample_aug_rates = [44_100, 32_000]
    config.data.source_resample_aug_weights = [1.0]

    with pytest.raises(ValueError, match="source_resample_augmentation.weights"):
        validate_config(config)


def test_validate_config_rejects_invalid_source_codec_name() -> None:
    config = _valid_config()
    config.data.source_codec_aug_enabled = True
    config.data.source_codec_aug_probability = 0.2
    config.data.source_codec_aug_codecs = ["mp2"]
    config.data.source_codec_aug_codec_weights = [1.0]
    config.data.source_codec_aug_bitrates = {"mp2": [128]}

    with pytest.raises(ValueError, match="source_codec_augmentation codecs"):
        validate_config(config)


def test_validate_config_rejects_invalid_source_codec_backend() -> None:
    config = _valid_config()
    config.data.source_codec_aug_backend = "shell"

    with pytest.raises(ValueError, match="source_codec_augmentation.backend"):
        validate_config(config)


def test_validate_config_rejects_invalid_waveform_micro_patch_size() -> None:
    config = _valid_config()
    config.model.waveform_level_depth = 1
    config.model.waveform_micro_patch_size = 30

    with pytest.raises(ValueError, match="waveform_micro_patch_size"):
        validate_config(config)


def test_validate_config_rejects_invalid_model_num_heads() -> None:
    config = _valid_config()
    config.model.num_heads = 7

    with pytest.raises(ValueError, match="model.num_heads"):
        validate_config(config)


def test_validate_config_rejects_invalid_waveform_num_heads() -> None:
    config = _valid_config()
    config.model.waveform_level_depth = 1
    config.model.waveform_num_heads = 7

    with pytest.raises(ValueError, match="waveform_num_heads"):
        validate_config(config)


def test_validate_config_rejects_invalid_scheduled_sampling_sampler() -> None:
    config = _valid_config()
    config.training.scheduled_sampling_sampler = "rk4"

    with pytest.raises(ValueError, match="scheduled_sampling_sampler"):
        validate_config(config)


def test_validate_config_accepts_unipc_scheduled_sampling_sampler() -> None:
    config = _valid_config()
    config.training.scheduled_sampling_sampler = "unipc"
    validate_config(config)


def test_validate_config_accepts_res6s_validation_generation_solver() -> None:
    config = _valid_config()
    config.training.validation_steps = 1
    config.training.run_validation_generations = True
    config.training.num_valid_generations = 1
    config.training.validation_generation_input_path = "dataset/validation_audio"
    config.training.validation_generation_output_path = "runs/test/validation_audio"
    config.training.validation_generation_solver = "res6s"
    config.training.validation_generation_solver_steps = 20
    config.training.validation_generation_chunk_seconds = 12.0
    config.training.validation_generation_overlap_seconds = 4.0
    validate_config(config)


@pytest.mark.parametrize("solver", ["midpoint", "midpoint_rk2", "midpoint-rk2", "rk2"])
def test_validate_config_accepts_midpoint_rk2_validation_generation_solver(
    solver: str,
) -> None:
    config = _valid_config()
    config.training.validation_steps = 1
    config.training.run_validation_generations = True
    config.training.num_valid_generations = 1
    config.training.validation_generation_input_path = "dataset/validation_audio"
    config.training.validation_generation_output_path = "runs/test/validation_audio"
    config.training.validation_generation_solver = solver
    config.training.validation_generation_solver_steps = 20
    config.training.validation_generation_chunk_seconds = 12.0
    config.training.validation_generation_overlap_seconds = 4.0
    validate_config(config)


def test_validate_config_rejects_invalid_validation_generation_solver() -> None:
    config = _valid_config()
    config.training.validation_steps = 1
    config.training.run_validation_generations = True
    config.training.num_valid_generations = 1
    config.training.validation_generation_input_path = "dataset/validation_audio"
    config.training.validation_generation_output_path = "runs/test/validation_audio"
    config.training.validation_generation_solver = "bad_solver"

    with pytest.raises(ValueError, match="validation_generation_solver"):
        validate_config(config)


def test_validate_config_rejects_target_lift_for_validation_generations() -> None:
    config = _valid_config()
    config.data.amplitude_lift_enabled = True
    config.data.amplitude_lift_reference = "target"
    config.training.validation_steps = 1
    config.training.run_validation_generations = True
    config.training.num_valid_generations = 1
    config.training.validation_generation_input_path = "dataset/validation_audio"
    config.training.validation_generation_output_path = "runs/test/validation_audio"

    with pytest.raises(ValueError, match="amplitude_lift_reference"):
        validate_config(config)


def test_validate_config_allows_wavflow_lift_for_validation_generations() -> None:
    config = _valid_config()
    config.data.amplitude_lift_enabled = True
    config.data.amplitude_lift_mode = "wavflow"
    config.data.amplitude_lift_reference = "target"
    config.training.validation_steps = 1
    config.training.run_validation_generations = True
    config.training.num_valid_generations = 1
    config.training.validation_generation_input_path = "dataset/validation_audio"
    config.training.validation_generation_output_path = "runs/test/validation_audio"

    validate_config(config)


def test_validate_config_rejects_invalid_amplitude_lift_clip_value() -> None:
    config = _valid_config()
    config.data.amplitude_lift_clip_value = 0.0

    with pytest.raises(ValueError, match="amplitude_lift_clip_value"):
        validate_config(config)


def test_validate_config_rejects_invalid_amplitude_lift_mode() -> None:
    config = _valid_config()
    config.data.amplitude_lift_mode = "peak"

    with pytest.raises(ValueError, match="amplitude_lift_mode"):
        validate_config(config)


def test_validate_config_allows_unclipped_amplitude_lift() -> None:
    config = _valid_config()
    config.data.amplitude_lift_enabled = True
    config.data.amplitude_lift_clip_value = None

    validate_config(config)


def test_validate_config_rejects_invalid_min_source_rms() -> None:
    config = _valid_config()
    config.data.min_source_rms = 0.0

    with pytest.raises(ValueError, match="min_source_rms"):
        validate_config(config)


def test_validate_config_allows_min_source_rms() -> None:
    config = _valid_config()
    config.data.min_source_rms = 0.04

    validate_config(config)


def test_validate_config_rejects_invalid_scheduled_sampling_probability() -> None:
    config = _valid_config()
    config.training.scheduled_sampling_probability = 1.2

    with pytest.raises(ValueError, match="scheduled_sampling_probability"):
        validate_config(config)


def test_validate_config_rejects_negative_reflexflow_beta1() -> None:
    config = _valid_config()
    config.training.scheduled_sampling_reflexflow_beta1 = -0.1

    with pytest.raises(ValueError, match="scheduled_sampling_reflexflow_beta1"):
        validate_config(config)


def test_validate_config_rejects_invalid_flow_timestep_sampling() -> None:
    config = _valid_config()
    config.training.flow_timestep_sampling = "gaussian"

    with pytest.raises(ValueError, match="flow_timestep_sampling"):
        validate_config(config)


def test_validate_config_rejects_invalid_flow_loss_weighting() -> None:
    config = _valid_config()
    config.training.flow_loss_weighting = "snr"

    with pytest.raises(ValueError, match="flow_loss_weighting"):
        validate_config(config)


def test_validate_config_rejects_mismatched_mrstft_resolution_lists() -> None:
    config = _valid_config()
    config.training.mrstft_loss_weight = 0.05
    config.training.mrstft_fft_sizes = [256, 512]
    config.training.mrstft_hop_lengths = [64]
    config.training.mrstft_win_lengths = [256, 512]

    with pytest.raises(ValueError, match="mrstft_fft_sizes"):
        validate_config(config)


def test_validate_config_rejects_all_zero_x_prediction_weights() -> None:
    config = _valid_config()
    config.training.waveform_mse_loss_weight = 0.0
    config.training.waveform_l1_loss_weight = 0.0
    config.training.waveform_charbonnier_loss_weight = 0.0
    config.training.x_pred_v_loss_weight = 0.0

    with pytest.raises(ValueError, match="waveform/x-prediction"):
        validate_config(config)


def test_validate_config_allows_velocity_only_x_prediction_loss() -> None:
    config = _valid_config()
    config.training.waveform_mse_loss_weight = 0.0
    config.training.waveform_l1_loss_weight = 0.0
    config.training.waveform_charbonnier_loss_weight = 0.0
    config.training.x_pred_v_loss_weight = 1.0

    validate_config(config)


def test_validate_config_rejects_invalid_waveform_charbonnier_eps() -> None:
    config = _valid_config()
    config.training.waveform_charbonnier_eps = 0.0

    with pytest.raises(ValueError, match="waveform_charbonnier_eps"):
        validate_config(config)


def test_validate_config_rejects_invalid_perceptual_fft_window() -> None:
    config = _valid_config()
    config.training.perceptual_loss_weight = 0.03
    config.training.perceptual_n_fft = 256
    config.training.perceptual_win_length = 512

    with pytest.raises(ValueError, match="perceptual_win_length"):
        validate_config(config)


def test_validate_config_rejects_invalid_perceptual_band() -> None:
    config = _valid_config()
    config.training.perceptual_loss_weight = 0.03
    config.training.perceptual_band_low_hz = 8000.0
    config.training.perceptual_band_high_hz = 150.0

    with pytest.raises(ValueError, match="perceptual_band_high_hz"):
        validate_config(config)


def test_validate_config_rejects_negative_binaural_loss_weight() -> None:
    config = _valid_config()
    config.training.binaural_ild_loss_weight = -0.01

    with pytest.raises(ValueError, match="binaural_ild_loss_weight"):
        validate_config(config)


def test_validate_config_rejects_invalid_binaural_loss_eps() -> None:
    config = _valid_config()
    config.training.binaural_loss_eps = 0.0

    with pytest.raises(ValueError, match="binaural_loss_eps"):
        validate_config(config)


def test_validate_config_rejects_invalid_binaural_mid_side_loss_type() -> None:
    config = _valid_config()
    config.training.binaural_mid_side_loss_type = "huber"

    with pytest.raises(ValueError, match="binaural_mid_side_loss_type"):
        validate_config(config)


def test_validate_config_rejects_invalid_binaural_frame_ild_size() -> None:
    config = _valid_config()
    config.training.binaural_frame_ild_frame_size = 0

    with pytest.raises(ValueError, match="binaural_frame_ild_frame_size"):
        validate_config(config)


def test_validate_config_rejects_invalid_ema_decay() -> None:
    config = _valid_config()
    config.training.use_ema = True
    config.training.ema_decay = 1.0

    with pytest.raises(ValueError, match="ema_decay"):
        validate_config(config)


def test_validate_config_rejects_invalid_ema_device() -> None:
    config = _valid_config()
    config.training.use_ema = True
    config.training.ema_device = "nvme"

    with pytest.raises(ValueError, match="ema_device"):
        validate_config(config)


def test_validate_config_rejects_ema_cpu_only_with_non_cpu_device() -> None:
    config = _valid_config()
    config.training.use_ema = True
    config.training.ema_device = "accelerator"
    config.training.ema_cpu_only = True

    with pytest.raises(ValueError, match="ema_cpu_only"):
        validate_config(config)
