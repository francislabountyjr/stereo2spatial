from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from stereo2spatial.cli.infer import (
    _load_runtime_config_and_bundle_payload,
    _resolve_cli_vae_paths,
    resolve_cli_config_path,
)
from stereo2spatial.inference.export_bundle import (
    EXPORT_BUNDLE_CONFIG_FILENAME,
    EXPORT_BUNDLE_VAE_CONFIG_FILENAME,
    EXPORT_BUNDLE_VAE_DIRNAME,
    EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME,
    EXPORT_BUNDLE_WEIGHTS_FILENAME,
    export_model_bundle,
    resolve_inference_config_path,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)


def _resolved_config_payload(
    *, output_dir: str, target_channels: int
) -> dict[str, object]:
    return {
        "seed": 1337,
        "output_dir": output_dir,
        "data": {
            "dataset_root": "dataset_qc_subset",
            "manifest_path": "dataset_qc_subset/manifest.jsonl",
            "sample_artifact_mode": "bundle",
            "segment_seconds": 10.0,
            "sequence_seconds": 10.0,
            "stride_seconds": 10.0,
            "latent_fps": 50,
            "mono_probability": 0.05,
            "downmix_probability": 0.05,
            "cache_size": 16,
            "shuffle_segments_within_epoch": True,
            "batch_size": 1,
            "num_workers": 0,
            "prefetch_factor": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        },
        "model": {
            "target_channels": target_channels,
            "cond_channels": 1,
            "latent_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "timestep_embed_dim": 128,
            "timestep_scale": 1000.0,
            "max_period": 10000.0,
            "num_memory_tokens": 0,
        },
        "training": {
            "max_steps": 1000,
            "grad_accum_steps": 1,
            "mixed_precision": "no",
            "compile_model": False,
            "compile_mode": "default",
            "resume_from_checkpoint": None,
            "init_from_checkpoint": None,
            "grad_clip_norm": 1.0,
            "log_every": 10,
            "checkpoint_every": 100,
            "max_checkpoints_to_keep": 2,
            "num_epochs_hint": 1,
            "window_seconds": 10.0,
            "overlap_seconds": 2.0,
            "sequence_seconds_choices": [10.0],
            "randomize_sequence_per_batch": False,
            "detach_memory": False,
            "sequence_mode": "full_song",
            "tbptt_windows": 0,
            "full_song_max_seconds": 10.0,
            "require_batch_size_one_for_full_song": True,
            "use_gan": False,
            "gan_d_lr": 1e-4,
            "gan_d_beta1": 0.0,
            "gan_d_beta2": 0.9,
            "gan_d_base_channels": 64,
            "gan_d_num_layers": 4,
            "gan_d_fine_layers": 3,
            "gan_d_coarse_layers": 4,
            "gan_d_use_spectral_norm": True,
            "gan_use_mask_channel": True,
            "gan_ms_w_fine": 0.5,
            "gan_ms_w_coarse": 0.5,
            "gan_lambda_adv": 0.0,
            "gan_adv_warmup_steps": 0,
            "gan_r1_gamma": 1.0,
            "gan_r1_every": 16,
            "routing_kl_weight": 0.015,
            "routing_kl_temperature": 1.1,
            "routing_kl_eps": 1e-6,
            "corr_weight": 0.008,
            "corr_eps": 1e-6,
            "corr_offdiag_only": True,
            "corr_use_correlation": True,
            "run_validation": False,
            "validation_dataset_root": None,
            "validation_dataset_path": None,
            "validation_steps": 100,
            "run_validation_generations": False,
            "num_valid_generations": 0,
            "validation_generation_seed": 1337,
            "validation_generation_input_path": None,
            "validation_generation_output_path": None,
            "validation_generation_vae_checkpoint_path": None,
            "validation_generation_vae_config_path": None,
            "use_ema": False,
            "ema_decay": 0.999,
            "ema_device": "accelerator",
            "ema_cpu_only": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.99,
            "eps": 1e-8,
            "adamw_fused": False,
            "adamw_foreach": False,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 10,
            "min_lr": 1e-6,
        },
    }


def _write_training_run(
    tmp_path: Path,
    *,
    target_channels: int,
    student_state: dict[str, torch.Tensor],
    ema_state: dict[str, torch.Tensor] | None = None,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints" / "step_0000001"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = _resolved_config_payload(
        output_dir=str(run_dir.as_posix()),
        target_channels=target_channels,
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    save_safetensors_file(student_state, str(checkpoint_dir / "model.safetensors"))
    if ema_state is not None:
        torch.save(
            {"decay": 0.999, "model": ema_state},
            checkpoint_dir / "custom_checkpoint_0.pkl",
        )
    return run_dir, checkpoint_dir


def _write_dummy_vae_assets(tmp_path: Path) -> tuple[Path, Path]:
    vae_checkpoint_path = tmp_path / "ear_vae_v2_48k.pyt"
    vae_config_path = tmp_path / "ear_vae_v2.json"
    torch.save({"state_dict": {"dummy": torch.ones(1)}}, vae_checkpoint_path)
    vae_config_path.write_text(
        json.dumps(
            {
                "encoder": {"channels": 2},
                "decoder": {"channels": 2},
                "transformer": None,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return vae_checkpoint_path, vae_config_path


def test_export_model_bundle_prefers_ema_state_when_available(tmp_path: Path) -> None:
    student_model = _TinyModel()
    ema_model = _TinyModel()
    for parameter in student_model.parameters():
        parameter.data.zero_()
    for parameter in ema_model.parameters():
        parameter.data.fill_(1.0)

    student_state = {
        f"_orig_mod.{name}": tensor.detach().clone()
        for name, tensor in student_model.state_dict().items()
    }
    ema_state = {
        f"_orig_mod.{name}": tensor.detach().clone()
        for name, tensor in ema_model.state_dict().items()
    }
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state=student_state,
        ema_state=ema_state,
    )
    vae_checkpoint_path, vae_config_path = _write_dummy_vae_assets(tmp_path)

    output_dir = tmp_path / "bundle"
    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
        vae_checkpoint_path=vae_checkpoint_path,
        vae_config_path=vae_config_path,
    )

    exported_state = load_safetensors_file(
        str(output_dir / EXPORT_BUNDLE_WEIGHTS_FILENAME),
        device="cpu",
    )
    assert torch.allclose(exported_state["linear.weight"], ema_model.linear.weight)
    assert torch.allclose(exported_state["linear.bias"], ema_model.linear.bias)


def test_cli_bundle_helpers_resolve_config_and_bundled_vae(tmp_path: Path) -> None:
    run_dir, checkpoint_dir = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    vae_checkpoint_path, vae_config_path = _write_dummy_vae_assets(tmp_path)
    bundle_dir = tmp_path / "bundle"
    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=bundle_dir,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
        vae_checkpoint_path=vae_checkpoint_path,
        vae_config_path=vae_config_path,
    )

    assert resolve_inference_config_path(bundle_dir) == (
        bundle_dir / EXPORT_BUNDLE_CONFIG_FILENAME
    )
    assert resolve_inference_config_path(checkpoint_dir) == (
        run_dir / "resolved_config.json"
    )

    resolved_cli_path = resolve_cli_config_path(
        config=None,
        checkpoint=str(bundle_dir),
    )
    assert resolved_cli_path == str(bundle_dir / EXPORT_BUNDLE_CONFIG_FILENAME)

    config, bundle_payload = _load_runtime_config_and_bundle_payload(resolved_cli_path)
    assert bundle_payload is not None
    assert config.model.target_channels == 2

    resolved_vae_checkpoint, resolved_vae_config = _resolve_cli_vae_paths(
        checkpoint=str(bundle_dir),
        resolved_config_path=resolved_cli_path,
        vae_checkpoint_path=None,
        vae_config_path=None,
    )
    assert resolved_vae_checkpoint == (
        bundle_dir / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME
    )
    assert resolved_vae_config == (
        bundle_dir / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_CONFIG_FILENAME
    )


def test_export_model_bundle_validates_channel_order_length(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    vae_checkpoint_path, vae_config_path = _write_dummy_vae_assets(tmp_path)

    with pytest.raises(ValueError, match="channel_order length"):
        export_model_bundle(
            train_run_dir=run_dir,
            checkpoint="latest",
            output_dir=tmp_path / "bundle",
            channel_layout_name="stereo",
            channel_order=["FL", "FR", "FC"],
            vae_checkpoint_path=vae_checkpoint_path,
            vae_config_path=vae_config_path,
        )
