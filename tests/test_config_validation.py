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
            latent_fps="auto",
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
            cond_channels=1,
            latent_dim=128,
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
            validation_generation_vae_checkpoint_path=None,
            validation_generation_vae_config_path=None,
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


def test_validate_config_rejects_invalid_latent_fps_string() -> None:
    config = _valid_config()
    config.data.latent_fps = "bad-value"

    with pytest.raises(ValueError, match="latent_fps"):
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
