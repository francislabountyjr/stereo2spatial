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
            "latent_fps": "auto",
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
            "cond_channels": 1,
            "latent_dim": 128,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "timestep_embed_dim": 256,
            "timestep_scale": 1000.0,
            "max_period": 10000.0,
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


def test_load_config_requires_mapping_sections(tmp_path: Path) -> None:
    payload = _base_config_payload()
    payload["training"] = ["not", "a", "mapping"]
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="training"):
        load_config(config_path)
