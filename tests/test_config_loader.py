from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from stereo2spatial.training.config import load_config


def _base_config_payload() -> dict[str, Any]:
    return {
        "seed": 1337,
        "output_dir": "runs/test",
        "data": {
            "dataset_root": "dataset",
            "manifest_path": "dataset/manifest.jsonl",
            "sample_artifact_mode": "bundle",
            "segment_seconds": 10.0,
            "sequence_seconds": 10.0,
            "stride_seconds": 5.0,
            "sample_rate": 48_000,
            "mono_probability": 0.1,
            "downmix_probability": 0.1,
            "cache_size": 8,
            "shuffle_segments_within_epoch": True,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        },
        "model": {
            "target_channels": 12,
            "cond_channels": 2,
            "patch_size": 128,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "timestep_embed_dim": 256,
            "timestep_scale": 1000.0,
            "max_period": 10000.0,
            "waveform_level_depth": 2,
            "waveform_micro_patch_size": 16,
            "waveform_hidden_dim": 16,
            "waveform_num_heads": 8,
            "waveform_mlp_ratio": 2.0,
            "activation_checkpointing": True,
        },
        "training": {
            "max_steps": 100,
            "grad_accum_steps": 1,
            "mixed_precision": "no",
            "grad_clip_norm": 1.0,
            "log_every": 10,
            "checkpoint_every": 50,
            "max_checkpoints_to_keep": 3,
            "num_epochs_hint": 1,
            "window_seconds": 8.0,
            "overlap_seconds": 2.0,
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 10,
            "min_lr": 1e-6,
        },
    }


def test_load_config_reads_scheduled_sampling_fields(tmp_path: Path) -> None:
    payload = _base_config_payload()
    payload["training"] = {
        **payload["training"],
        "scheduled_sampling_max_step_offset": 6,
        "scheduled_sampling_probability": 0.7,
        "scheduled_sampling_prob_start": 0.1,
        "scheduled_sampling_prob_end": 0.8,
        "scheduled_sampling_ramp_steps": 500,
        "scheduled_sampling_start_step": 100,
        "scheduled_sampling_ramp_shape": "cosine",
        "scheduled_sampling_strategy": "biased_late",
        "scheduled_sampling_sampler": "euler",
        "scheduled_sampling_reflexflow": False,
        "scheduled_sampling_reflexflow_alpha": 0.75,
        "scheduled_sampling_reflexflow_beta1": 12.0,
        "scheduled_sampling_reflexflow_beta2": 1.5,
        "flow_timestep_sampling": "beta",
        "flow_fast_schedule": True,
        "flow_logit_mean": -0.2,
        "flow_logit_std": 0.8,
        "flow_beta_alpha": 2.0,
        "flow_beta_beta": 3.0,
        "flow_custom_timesteps": [0.2, 0.5, 0.9],
        "flow_schedule_shift": 2.5,
        "flow_schedule_auto_shift": True,
        "flow_schedule_base_seq_len": 128,
        "flow_schedule_max_seq_len": 2048,
        "flow_schedule_base_shift": 0.4,
        "flow_schedule_max_shift": 1.4,
        "flow_loss_weighting": "cosmap",
        "use_ema": True,
        "ema_decay": 0.995,
        "ema_device": "cpu",
        "ema_cpu_only": True,
        "mrstft_loss_weight": 0.05,
        "mrstft_fft_sizes": [256, 512],
        "mrstft_hop_lengths": [64, 128],
        "mrstft_win_lengths": [256, 512],
        "mrstft_sc_weight": 0.75,
        "mrstft_log_mag_weight": 0.5,
        "mrstft_eps": 1e-6,
        "waveform_mse_loss_weight": 0.25,
        "waveform_l1_loss_weight": 0.5,
        "waveform_charbonnier_loss_weight": 0.75,
        "waveform_charbonnier_eps": 1e-4,
        "perceptual_loss_weight": 0.03,
        "perceptual_n_fft": 512,
        "perceptual_hop_length": 128,
        "perceptual_win_length": 512,
        "perceptual_n_mels": 64,
        "perceptual_f_min": 30.0,
        "perceptual_f_max": 12000.0,
        "perceptual_band_weight": 1.25,
        "perceptual_band_low_hz": 150.0,
        "perceptual_band_high_hz": 8000.0,
        "perceptual_eps": 1e-5,
        "binaural_ild_loss_weight": 0.03,
        "binaural_ipd_loss_weight": 0.01,
        "binaural_ccf_loss_weight": 0.02,
        "binaural_frame_ild_loss_weight": 0.005,
        "binaural_frame_ild_frame_size": 1024,
        "binaural_frame_ild_hop_size": 512,
        "binaural_frame_ild_silence_threshold": 2e-4,
        "binaural_frame_ild_max_weight": 3.5,
        "binaural_mid_side_loss_weight": 0.02,
        "binaural_mid_side_loss_type": "l1",
        "binaural_mid_side_mid_weight": 0.1,
        "binaural_mid_side_side_weight": 0.9,
        "binaural_mid_side_charbonnier_eps": 2e-3,
        "binaural_loss_warmup_steps": 20000,
        "binaural_loss_eps": 1e-6,
        "validation_generation_solver": "res6s",
        "validation_generation_solver_steps": 20,
        "validation_generation_solver_rtol": 2e-5,
        "validation_generation_solver_atol": 3e-5,
        "validation_generation_chunk_seconds": 12.0,
        "validation_generation_overlap_seconds": 4.0,
    }
    payload["data"]["source_resample_augmentation"] = {
        "enabled": True,
        "probability": 0.15,
        "rates": [44_100],
        "weights": [1.0],
    }
    payload["data"]["source_codec_augmentation"] = {
        "enabled": True,
        "probability": 0.2,
        "start_step": 50_000,
        "full_strength_step": 150_000,
        "backend": "auto",
        "ffmpeg_path": "ffmpeg",
        "max_chunk_seconds": 12.0,
        "align_max_lag": 8192,
        "timeout_seconds": 20.0,
        "codecs": {
            "mp3": {"weight": 0.4, "bitrates": [128, 192]},
            "aac": {"weight": 0.35, "bitrates": [128, 192]},
            "opus": {"weight": 0.25, "bitrates": [96, 160]},
        },
    }
    config_path = tmp_path / "config_ss.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.training.scheduled_sampling_max_step_offset == 6
    assert config.training.scheduled_sampling_probability == pytest.approx(0.7)
    assert config.training.scheduled_sampling_prob_start == pytest.approx(0.1)
    assert config.training.scheduled_sampling_prob_end == pytest.approx(0.8)
    assert config.training.scheduled_sampling_ramp_steps == 500
    assert config.training.scheduled_sampling_start_step == 100
    assert config.training.scheduled_sampling_ramp_shape == "cosine"
    assert config.training.scheduled_sampling_strategy == "biased_late"
    assert config.training.scheduled_sampling_sampler == "euler"
    assert config.training.scheduled_sampling_reflexflow is False
    assert config.training.scheduled_sampling_reflexflow_alpha == pytest.approx(0.75)
    assert config.training.scheduled_sampling_reflexflow_beta1 == pytest.approx(12.0)
    assert config.training.scheduled_sampling_reflexflow_beta2 == pytest.approx(1.5)
    assert config.training.flow_timestep_sampling == "beta"
    assert config.training.flow_fast_schedule is True
    assert config.training.flow_logit_mean == pytest.approx(-0.2)
    assert config.training.flow_logit_std == pytest.approx(0.8)
    assert config.training.flow_beta_alpha == pytest.approx(2.0)
    assert config.training.flow_beta_beta == pytest.approx(3.0)
    assert config.training.flow_custom_timesteps == pytest.approx([0.2, 0.5, 0.9])
    assert config.training.flow_schedule_shift == pytest.approx(2.5)
    assert config.training.flow_schedule_auto_shift is True
    assert config.training.flow_schedule_base_seq_len == 128
    assert config.training.flow_schedule_max_seq_len == 2048
    assert config.training.flow_schedule_base_shift == pytest.approx(0.4)
    assert config.training.flow_schedule_max_shift == pytest.approx(1.4)
    assert config.training.flow_loss_weighting == "cosmap"
    assert config.training.use_ema is True
    assert config.training.ema_decay == pytest.approx(0.995)
    assert config.training.ema_device == "cpu"
    assert config.training.ema_cpu_only is True
    assert config.training.mrstft_loss_weight == pytest.approx(0.05)
    assert config.training.mrstft_fft_sizes == [256, 512]
    assert config.training.mrstft_hop_lengths == [64, 128]
    assert config.training.mrstft_win_lengths == [256, 512]
    assert config.training.mrstft_sc_weight == pytest.approx(0.75)
    assert config.training.mrstft_log_mag_weight == pytest.approx(0.5)
    assert config.training.mrstft_eps == pytest.approx(1e-6)
    assert config.training.waveform_mse_loss_weight == pytest.approx(0.25)
    assert config.training.waveform_l1_loss_weight == pytest.approx(0.5)
    assert config.training.waveform_charbonnier_loss_weight == pytest.approx(0.75)
    assert config.training.waveform_charbonnier_eps == pytest.approx(1e-4)
    assert config.training.perceptual_loss_weight == pytest.approx(0.03)
    assert config.training.perceptual_n_fft == 512
    assert config.training.perceptual_hop_length == 128
    assert config.training.perceptual_win_length == 512
    assert config.training.perceptual_n_mels == 64
    assert config.training.perceptual_f_min == pytest.approx(30.0)
    assert config.training.perceptual_f_max == pytest.approx(12000.0)
    assert config.training.perceptual_band_weight == pytest.approx(1.25)
    assert config.training.perceptual_band_low_hz == pytest.approx(150.0)
    assert config.training.perceptual_band_high_hz == pytest.approx(8000.0)
    assert config.training.perceptual_eps == pytest.approx(1e-5)
    assert config.training.binaural_ild_loss_weight == pytest.approx(0.03)
    assert config.training.binaural_ipd_loss_weight == pytest.approx(0.01)
    assert config.training.binaural_ccf_loss_weight == pytest.approx(0.02)
    assert config.training.binaural_frame_ild_loss_weight == pytest.approx(0.005)
    assert config.training.binaural_frame_ild_frame_size == 1024
    assert config.training.binaural_frame_ild_hop_size == 512
    assert config.training.binaural_frame_ild_silence_threshold == pytest.approx(2e-4)
    assert config.training.binaural_frame_ild_max_weight == pytest.approx(3.5)
    assert config.training.binaural_mid_side_loss_weight == pytest.approx(0.02)
    assert config.training.binaural_mid_side_loss_type == "l1"
    assert config.training.binaural_mid_side_mid_weight == pytest.approx(0.1)
    assert config.training.binaural_mid_side_side_weight == pytest.approx(0.9)
    assert config.training.binaural_mid_side_charbonnier_eps == pytest.approx(2e-3)
    assert config.training.binaural_loss_warmup_steps == 20000
    assert config.training.binaural_loss_eps == pytest.approx(1e-6)
    assert config.training.validation_generation_solver == "res6s"
    assert config.training.validation_generation_solver_steps == 20
    assert config.training.validation_generation_solver_rtol == pytest.approx(2e-5)
    assert config.training.validation_generation_solver_atol == pytest.approx(3e-5)
    assert config.training.validation_generation_chunk_seconds == pytest.approx(12.0)
    assert config.training.validation_generation_overlap_seconds == pytest.approx(4.0)
    assert config.data.source_resample_aug_enabled is True
    assert config.data.source_resample_aug_probability == pytest.approx(0.15)
    assert config.data.source_resample_aug_rates == [44_100]
    assert config.data.source_resample_aug_weights == pytest.approx([1.0])
    assert config.data.source_codec_aug_enabled is True
    assert config.data.source_codec_aug_probability == pytest.approx(0.2)
    assert config.data.source_codec_aug_start_step == 50_000
    assert config.data.source_codec_aug_full_strength_step == 150_000
    assert config.data.source_codec_aug_backend == "auto"
    assert config.data.source_codec_aug_ffmpeg_path == "ffmpeg"
    assert config.data.source_codec_aug_max_chunk_seconds == pytest.approx(12.0)
    assert config.data.source_codec_aug_align_max_lag == 8192
    assert config.data.source_codec_aug_timeout_seconds == pytest.approx(20.0)
    codec_weights = dict(
        zip(
            config.data.source_codec_aug_codecs or [],
            config.data.source_codec_aug_codec_weights or [],
        )
    )
    assert codec_weights == pytest.approx({"mp3": 0.4, "aac": 0.35, "opus": 0.25})
    assert config.data.source_codec_aug_bitrates == {
        "mp3": [128, 192],
        "aac": [128, 192],
        "opus": [96, 160],
    }
    assert config.model.waveform_level_depth == 2
    assert config.model.waveform_micro_patch_size == 16
    assert config.model.waveform_hidden_dim == 16
    assert config.model.waveform_num_heads == 8
    assert config.model.waveform_mlp_ratio == pytest.approx(2.0)
    assert config.model.activation_checkpointing is True


def test_load_config_reads_multiple_dataset_specs(tmp_path: Path) -> None:
    payload = _base_config_payload()
    payload["data"].pop("dataset_root")
    payload["data"].pop("manifest_path")
    payload["data"]["datasets"] = [
        {
            "dataset_root": "dataset/ssd",
            "manifest_path": "dataset/ssd/manifest.jsonl",
        },
        {
            "dataset_root": "dataset/hdd",
            "manifest_path": "dataset/hdd/manifest.jsonl",
        },
    ]
    config_path = tmp_path / "config_multi_data.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.data.dataset_root == ["dataset/ssd", "dataset/hdd"]
    assert config.data.manifest_path == [
        "dataset/ssd/manifest.jsonl",
        "dataset/hdd/manifest.jsonl",
    ]


def test_load_config_reads_data_batch_mode(tmp_path: Path) -> None:
    payload = _base_config_payload()
    payload["data"]["batch_mode"] = "song_local"
    payload["data"]["materialize_cached_signals"] = True
    payload["data"]["shuffle_segments_within_song"] = False
    config_path = tmp_path / "config_batch_mode.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.data.batch_mode == "song_local"
    assert config.data.materialize_cached_signals is True
    assert config.data.shuffle_segments_within_song is False


def test_load_config_requires_mapping_sections(tmp_path: Path) -> None:
    payload = _base_config_payload()
    payload["training"] = ["not", "a", "mapping"]
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="training"):
        load_config(config_path)
