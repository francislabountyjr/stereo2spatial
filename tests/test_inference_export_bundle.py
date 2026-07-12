from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

import stereo2spatial.cli.infer as infer_cli
from stereo2spatial.cli.infer import (
    _iter_input_audio_paths,
    _load_runtime_config_and_bundle_payload,
    _resolve_output_audio_path,
    _resolve_report_json_path,
    _resolve_runtime_arg,
    resolve_cli_config_path,
)
from stereo2spatial.inference.export_bundle import (
    EXPORT_BUNDLE_CONFIG_FILENAME,
    EXPORT_BUNDLE_KIND,
    EXPORT_BUNDLE_SCHEMA_VERSION,
    EXPORT_BUNDLE_VAE_CONFIG_FILENAME,
    EXPORT_BUNDLE_VAE_DIRNAME,
    EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME,
    EXPORT_BUNDLE_WEIGHTS_FILENAME,
    build_train_config_from_bundle_payload,
    export_model_bundle,
    load_inference_bundle_payload,
    resolve_inference_config_path,
)
from stereo2spatial.inference.offline_batch import DynamicInferenceJob
from stereo2spatial.training.config import load_config


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
            "sample_rate": 48_000,
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
            "cond_channels": 2,
            "patch_size": 1024,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "timestep_embed_dim": 128,
            "timestep_scale": 1000.0,
            "max_period": 10000.0,
            "num_memory_tokens": 0,
            "waveform_level_depth": 2,
            "waveform_micro_patch_size": 16,
            "waveform_hidden_dim": 16,
            "waveform_num_heads": 4,
            "waveform_mlp_ratio": 2.0,
            "final_output_kernel_size": 9,
            "final_output_zero_init": True,
            "rope_enabled": True,
            "rope_theta": 20000.0,
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


def _rewrite_run_as_legacy(run_dir: Path, *, latent_fps: float = 50.0) -> None:
    config_path = run_dir / "resolved_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model = payload["model"]
    model["architecture"] = "legacy_vae"
    model["cond_channels"] = 1
    model["latent_dim"] = 64
    model.pop("patch_size", None)
    payload["data"]["latent_fps"] = latent_fps
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


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

    output_dir = tmp_path / "bundle"
    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
    )

    exported_state = load_safetensors_file(
        str(output_dir / EXPORT_BUNDLE_WEIGHTS_FILENAME),
        device="cpu",
    )
    assert torch.allclose(exported_state["linear.weight"], ema_model.linear.weight)
    assert torch.allclose(exported_state["linear.bias"], ema_model.linear.bias)


def test_export_model_bundle_accepts_explicit_training_yaml(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    override = _resolved_config_payload(
        output_dir=str(run_dir.as_posix()),
        target_channels=6,
    )
    config_path = tmp_path / "export_config.yaml"
    config_path.write_text(yaml.safe_dump(override), encoding="utf-8")
    output_dir = tmp_path / "bundle"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        config_path=config_path,
    )

    payload = load_inference_bundle_payload(output_dir / "config.json")
    assert payload["target_channels"] == 6
    assert payload["channel_layout"] == "5.1 rear"


def test_legacy_export_round_trips_latent_fps(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    _rewrite_run_as_legacy(run_dir, latent_fps=37.5)
    output_dir = tmp_path / "bundle"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
        include_vae=False,
    )

    payload = load_inference_bundle_payload(output_dir / EXPORT_BUNDLE_CONFIG_FILENAME)
    assert payload["latent_fps"] == 37.5
    config = build_train_config_from_bundle_payload(payload, bundle_root=output_dir)
    assert config.data.latent_fps == 37.5

    legacy_payload = dict(payload)
    legacy_payload.pop("bundle_kind")
    legacy_payload.pop("bundle_schema_version")
    legacy_payload.pop("architecture")
    legacy_payload.pop("output_kind")
    legacy_config = build_train_config_from_bundle_payload(legacy_payload)
    assert legacy_config.model.architecture == "legacy_vae"
    assert legacy_config.model.latent_dim == 64


def test_export_emits_versioned_waveform_contract_and_round_trips_runtime_fields(
    tmp_path: Path,
) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    config_path = run_dir / "resolved_config.json"
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    resolved["model"].update(
        {
            "architecture": "waveform",
            "dropout": 0.125,
            "mix_style_dim": 10,
            "amplitude_gain_conditioning": True,
        }
    )
    resolved["data"].update(
        {
            "amplitude_lift_enabled": True,
            "amplitude_lift_mode": "wavflow",
            "amplitude_lift_gain_power": 1.75,
            "amplitude_lift_gain_min_value": 0.25,
        }
    )
    resolved["training"].update(
        {
            "validation_generation_seed": 4242,
            "flow_one_step": True,
            "flow_one_step_input": "cond",
        }
    )
    config_path.write_text(json.dumps(resolved), encoding="utf-8")
    output_dir = tmp_path / "bundle"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
    )

    payload = load_inference_bundle_payload(output_dir / "config.json")
    assert payload["bundle_kind"] == EXPORT_BUNDLE_KIND
    assert payload["bundle_schema_version"] == EXPORT_BUNDLE_SCHEMA_VERSION
    assert payload["architecture"] == "waveform"
    assert payload["architectures"] == ["SpatialDiT"]
    assert payload["output_kind"] == "direct_binaural"
    assert payload["channel_layout"] == "binaural"
    assert payload["channel_order"] == ["FL", "FR"]
    assert payload["channel_mask"] == 0x3
    assert payload["dropout"] == pytest.approx(0.125)
    assert payload["mix_style_dim"] == 10
    assert "mix_style_presets" not in payload
    assert "mix_style_names" not in payload
    assert payload["amplitude_gain_conditioning"] is True
    assert payload["amplitude_lift_gain_power"] == pytest.approx(1.75)
    assert payload["amplitude_lift_gain_min_value"] == pytest.approx(0.25)
    assert payload["inference"]["sampling_order"] == "timestep_major"
    assert payload["inference"]["seed"] == 4242
    assert payload["inference"]["flow_one_step"] is True
    assert payload["inference"]["flow_one_step_input"] == "cond"
    assert not (output_dir / EXPORT_BUNDLE_VAE_DIRNAME).exists()

    runtime = build_train_config_from_bundle_payload(payload, bundle_root=output_dir)
    assert runtime.seed == 4242
    assert runtime.model.architecture == "waveform"
    assert runtime.model.dropout == pytest.approx(0.125)
    assert runtime.model.amplitude_gain_conditioning is True
    assert runtime.data.amplitude_lift_gain_power == pytest.approx(1.75)
    assert runtime.data.amplitude_lift_gain_min_value == pytest.approx(0.25)
    assert runtime.training.validation_generation_seed == 4242
    assert runtime.training.flow_one_step is True
    assert runtime.training.flow_one_step_input == "cond"


@pytest.mark.parametrize(
    ("target_channels", "expected_layout", "expected_order"),
    [
        (6, "5.1 rear", ["FL", "FR", "FC", "LFE", "BL", "BR"]),
        (
            12,
            "7.1.4",
            [
                "FL",
                "FR",
                "FC",
                "LFE",
                "BL",
                "BR",
                "SL",
                "SR",
                "TFL",
                "TFR",
                "TBL",
                "TBR",
            ],
        ),
    ],
)
def test_export_resolves_default_speaker_layout_from_target_channels(
    tmp_path: Path,
    target_channels: int,
    expected_layout: str,
    expected_order: list[str],
) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=target_channels,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    output_dir = tmp_path / f"bundle_{target_channels}"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
    )

    payload = load_inference_bundle_payload(output_dir / "config.json")
    assert payload["output_kind"] == "speaker_layout"
    assert payload["channel_layout"] == expected_layout
    assert payload["channel_order"] == expected_order


def test_legacy_export_keeps_fixed_vae_bundle_contract(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=12,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    _rewrite_run_as_legacy(run_dir)
    vae_weights = tmp_path / "source_vae.pyt"
    vae_config = tmp_path / "source_vae.json"
    vae_weights.write_bytes(b"weights")
    vae_config.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "legacy_bundle"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        vae_checkpoint_path=vae_weights,
        vae_config_path=vae_config,
    )

    payload = load_inference_bundle_payload(output_dir / "config.json")
    assert payload["bundle_kind"] == EXPORT_BUNDLE_KIND
    assert payload["bundle_schema_version"] == EXPORT_BUNDLE_SCHEMA_VERSION
    assert payload["architecture"] == "legacy_vae"
    assert payload["architectures"] == ["LegacySpatialDiT"]
    assert payload["output_kind"] == "speaker_layout"
    assert payload["channel_layout"] == "7.1.4"
    assert payload["latent_dim"] == 64
    assert "patch_size" not in payload
    assert payload["sample_rate"] == 48_000
    assert (
        output_dir / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_WEIGHTS_FILENAME
    ).read_bytes() == b"weights"
    assert (
        output_dir / EXPORT_BUNDLE_VAE_DIRNAME / EXPORT_BUNDLE_VAE_CONFIG_FILENAME
    ).read_text(encoding="utf-8") == "{}"


def test_export_preserves_training_aligned_inference_recommendations(
    tmp_path: Path,
) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    config_path = run_dir / "resolved_config.json"
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    resolved["data"]["training_sample_rate"] = 24_000
    resolved["training"].update(
        {
            "window_seconds": 7.5,
            "overlap_seconds": 1.25,
            "validation_generation_solver": "midpoint_rk2",
            "validation_generation_solver_steps": 48,
            "validation_generation_solver_rtol": 2.0e-5,
            "validation_generation_solver_atol": 3.0e-5,
        }
    )
    config_path.write_text(json.dumps(resolved), encoding="utf-8")
    output_dir = tmp_path / "bundle"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
    )

    payload = load_inference_bundle_payload(output_dir / "config.json")
    assert payload["sample_rate"] == 24_000
    assert payload["inference"] == {
        "sample_rate": 24_000,
        "chunk_seconds": 7.5,
        "overlap_seconds": 1.25,
        "solver": "midpoint_rk2",
        "solver_steps": 48,
        "solver_rtol": 2.0e-5,
        "solver_atol": 3.0e-5,
        "sampling_order": "timestep_major",
        "seed": 1337,
        "flow_one_step": False,
        "flow_one_step_input": "zeros",
    }
    runtime = build_train_config_from_bundle_payload(payload)
    assert runtime.data.segment_seconds == pytest.approx(7.5)
    assert runtime.training.window_seconds == pytest.approx(7.5)
    assert runtime.training.overlap_seconds == pytest.approx(1.25)
    assert runtime.training.validation_generation_solver == "midpoint_rk2"
    assert runtime.training.validation_generation_solver_steps == 48
    assert runtime.training.validation_generation_solver_rtol == pytest.approx(2.0e-5)
    assert runtime.training.validation_generation_solver_atol == pytest.approx(3.0e-5)


def test_export_sample_rate_override_and_legacy_48k_constraint(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={"linear.weight": torch.zeros((1, 1))},
    )
    waveform_output = tmp_path / "waveform_bundle"
    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=waveform_output,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
        sample_rate=32_000,
    )
    assert (
        load_inference_bundle_payload(waveform_output / "config.json")["sample_rate"]
        == 32_000
    )

    _rewrite_run_as_legacy(run_dir)
    with pytest.raises(ValueError, match="require a 48000 Hz sample rate"):
        export_model_bundle(
            train_run_dir=run_dir,
            checkpoint="latest",
            output_dir=tmp_path / "legacy_bundle",
            channel_layout_name="stereo",
            channel_order=["FL", "FR"],
            sample_rate=44_100,
            include_vae=False,
        )


@pytest.mark.parametrize(
    ("config_architecture", "state_architecture"),
    [("waveform", "legacy_vae"), ("legacy_vae", "waveform")],
)
def test_export_rejects_checkpoint_architecture_mismatch(
    tmp_path: Path,
    config_architecture: str,
    state_architecture: str,
) -> None:
    state = (
        {
            "final_proj.weight": torch.zeros((1, 1)),
            "final_norm.weight": torch.zeros(1),
        }
        if state_architecture == "legacy_vae"
        else {
            "final_output.conv.weight": torch.zeros((1, 1, 1)),
            "final_output.adaLN_modulation.1.weight": torch.zeros((1, 1)),
        }
    )
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state=state,
    )
    if config_architecture == "legacy_vae":
        _rewrite_run_as_legacy(run_dir)

    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        export_model_bundle(
            train_run_dir=run_dir,
            checkpoint="latest",
            output_dir=tmp_path / "bundle",
            channel_layout_name="stereo",
            channel_order=["FL", "FR"],
            include_vae=False,
        )


def test_cli_bundle_helpers_resolve_config(tmp_path: Path) -> None:
    run_dir, checkpoint_dir = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    bundle_dir = tmp_path / "bundle"
    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=bundle_dir,
        channel_layout_name="stereo",
        channel_order=["FL", "FR"],
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
    assert config.model.waveform_level_depth == 2
    assert config.model.waveform_micro_patch_size == 16
    assert config.model.waveform_hidden_dim == 16
    assert config.model.waveform_num_heads == 4
    assert config.model.final_output_kernel_size == 9
    assert config.model.final_output_zero_init is True
    assert config.model.rope_enabled is True
    assert config.model.rope_theta == pytest.approx(20000.0)


def test_cli_runtime_arg_prefers_explicit_then_nested_bundle_recommendation() -> None:
    payload = {
        "sample_rate": 48_000,
        "inference": {"sample_rate": 24_000, "chunk_seconds": 7.5},
    }
    assert (
        _resolve_runtime_arg(
            explicit_value=32_000,
            bundle_payload=payload,
            section_name="inference",
            key="sample_rate",
            fallback=16_000,
        )
        == 32_000
    )
    assert (
        _resolve_runtime_arg(
            explicit_value=None,
            bundle_payload=payload,
            section_name="inference",
            key="sample_rate",
            fallback=16_000,
        )
        == 24_000
    )
    assert (
        _resolve_runtime_arg(
            explicit_value=None,
            bundle_payload={"sample_rate": 48_000},
            section_name="inference",
            key="sample_rate",
            fallback=16_000,
        )
        == 48_000
    )


def test_infer_cli_discovers_audio_files_recursively(tmp_path: Path) -> None:
    input_dir = tmp_path / "validation"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    (input_dir / "b.flac").write_bytes(b"fake")
    (nested_dir / "a.wav").write_bytes(b"fake")
    (nested_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    discovered = _iter_input_audio_paths(input_dir)

    assert discovered == [input_dir / "b.flac", nested_dir / "a.wav"]


def test_infer_cli_preserves_folder_layout_for_outputs_and_reports(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "validation"
    input_path = input_dir / "artist" / "song.wav"
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "reports"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"fake")

    headphone_output_path = _resolve_output_audio_path(
        input_audio_path=input_path,
        input_root_path=input_dir,
        output_root_path=output_dir,
        target_channels=2,
    )
    spatial_output_path = _resolve_output_audio_path(
        input_audio_path=input_path,
        input_root_path=input_dir,
        output_root_path=output_dir,
        target_channels=12,
    )
    report_path = _resolve_report_json_path(
        input_audio_path=input_path,
        input_root_path=input_dir,
        report_root_path=report_dir,
    )

    assert headphone_output_path == output_dir / "artist" / "song.flac"
    assert spatial_output_path == output_dir / "artist" / "song.wav"
    assert report_path == report_dir / "artist" / "song.json"


def test_infer_cli_keeps_explicit_single_file_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "custom_output.flac"
    report_path = tmp_path / "report.json"
    input_path.write_bytes(b"fake")

    assert _iter_input_audio_paths(input_path) == [input_path]
    assert (
        _resolve_output_audio_path(
            input_audio_path=input_path,
            input_root_path=input_path,
            output_root_path=output_path,
            target_channels=12,
        )
        == output_path
    )
    assert (
        _resolve_report_json_path(
            input_audio_path=input_path,
            input_root_path=input_path,
            report_root_path=report_path,
        )
        == report_path
    )


def test_infer_cli_folder_mode_reuses_session_without_device_arg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"
    checkpoint_path = tmp_path / "checkpoint"
    input_dir.mkdir()
    checkpoint_path.mkdir()
    (input_dir / "a.wav").write_bytes(b"fake")
    (input_dir / "b.flac").write_bytes(b"fake")
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    config_path = run_dir / "resolved_config.json"
    train_config = load_config(config_path)
    train_config.data.training_sample_rate = 24_000

    class _Session:
        run_device = "cuda"
        compile_mode = "default"

    calls: list[dict[str, object]] = []

    def fake_load_config(_path: Path) -> object:
        return train_config, None

    def fake_build_session(**kwargs: object) -> object:
        assert kwargs["device"] == "cuda"
        assert kwargs["compile_model"] is True
        return _Session()

    def fake_run_with_session(**kwargs: object) -> dict[str, object]:
        assert "device" not in kwargs
        calls.append(kwargs)
        output_path = Path(cast(str | Path, kwargs["output_audio_path"]))
        return {
            "config_path": str(config_path),
            "input_audio_path": str(kwargs["input_audio_path"]),
            "output_audio_path": str(output_path),
            "checkpoint_path": str(checkpoint_path),
            "weights_source": "student",
            "device": "cuda",
            "input_channels": 2,
            "sample_rate": 48000,
            "conditioning_signal_shape": [2, 200, 1],
            "pred_signal_shape": [2, 200, 1],
            "decoded_shape": [2, 200],
            "patch_fps": 240.0,
            "chunk_frames": 1,
            "overlap_frames": 0,
            "solver": "res6s",
            "seed": 1337,
            "mix_style_preset": None,
            "mix_style": None,
            "compiled": True,
            "compile_mode": "default",
        }

    monkeypatch.setattr(
        infer_cli,
        "_load_runtime_config_and_bundle_payload",
        fake_load_config,
    )
    monkeypatch.setattr(infer_cli, "build_inference_session", fake_build_session)
    monkeypatch.setattr(infer_cli, "run_inference_with_session", fake_run_with_session)
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--input-audio",
            str(input_dir),
            "--output-audio",
            str(output_dir),
            "--report-json",
            str(report_dir),
            "--device",
            "cuda",
            "--solver",
            "res6s",
            "--solver-steps",
            "20",
            "--weights-source",
            "student",
            "--compile-model",
        ],
    )

    infer_cli.main()

    assert len(calls) == 2
    assert all(call["sample_rate"] == 24_000 for call in calls)
    assert all(call["chunk_seconds"] == 10.0 for call in calls)
    assert all(call["overlap_seconds"] == 2.0 for call in calls)
    assert all(call["solver"] == "res6s" for call in calls)
    assert all(call["solver_steps"] == 20 for call in calls)
    assert (report_dir / "a.json").exists()
    assert (report_dir / "b.json").exists()


def test_infer_cli_dynamic_folder_mode_dispatches_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"
    checkpoint_path = tmp_path / "checkpoint"
    input_dir.mkdir()
    checkpoint_path.mkdir()
    (input_dir / "a.wav").write_bytes(b"fake")
    (input_dir / "nested").mkdir()
    (input_dir / "nested" / "b.flac").write_bytes(b"fake")
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    config_path = run_dir / "resolved_config.json"
    train_config = load_config(config_path)

    class _Session:
        run_device = "cuda"
        compile_mode = None

    captured: dict[str, object] = {}

    def fake_load_config(_path: Path) -> object:
        return train_config, None

    def fake_build_session(**kwargs: object) -> object:
        captured["session_kwargs"] = kwargs
        return _Session()

    def fake_dynamic_folder(**kwargs: object) -> object:
        captured["dynamic_kwargs"] = kwargs
        jobs = list(cast(Iterable[DynamicInferenceJob], kwargs["jobs"]))
        reports = []
        for job in jobs:
            reports.append(
                {
                    "output_audio_path": str(job.output_audio_path),
                    "input_audio_path": str(job.input_audio_path),
                }
            )
        return SimpleNamespace(
            reports=reports,
            errors=[],
            stats=SimpleNamespace(
                scheduler=SimpleNamespace(
                    model_batches=3,
                    model_queries=6,
                    max_observed_batch_size=2,
                    completed_controllers=4,
                )
            ),
        )

    monkeypatch.setattr(
        infer_cli,
        "_load_runtime_config_and_bundle_payload",
        fake_load_config,
    )
    monkeypatch.setattr(infer_cli, "build_inference_session", fake_build_session)
    monkeypatch.setattr(infer_cli, "run_dynamic_folder_inference", fake_dynamic_folder)
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--input-audio",
            str(input_dir),
            "--output-audio",
            str(output_dir),
            "--report-json",
            str(report_dir),
            "--device",
            "cuda",
            "--solver",
            "midpoint_rk2",
            "--solver-steps",
            "20",
            "--weights-source",
            "student",
            "--dynamic-batching",
            "--max-batch-size",
            "2",
            "--max-active-requests",
            "3",
            "--encode-chunk-size-samples",
            "48000",
            "--encode-overlap-samples",
            "4800",
            "--decode-chunk-size-frames",
            "1024",
            "--decode-overlap-frames",
            "128",
            "--disable-chunked-decode",
        ],
    )

    infer_cli.main()

    dynamic_kwargs = cast(dict[str, object], captured["dynamic_kwargs"])
    jobs = list(cast(Iterable[DynamicInferenceJob], dynamic_kwargs["jobs"]))
    assert len(jobs) == 2
    assert jobs[0].output_audio_path == output_dir / "a.flac"
    assert jobs[1].output_audio_path == output_dir / "nested" / "b.flac"
    assert dynamic_kwargs["max_batch_size"] == 2
    assert dynamic_kwargs["max_active_requests"] == 3
    assert dynamic_kwargs["solver"] == "midpoint_rk2"
    assert dynamic_kwargs["encode_chunk_size_samples"] == 48000
    assert dynamic_kwargs["encode_overlap_samples"] == 4800
    assert dynamic_kwargs["decode_chunk_size_frames"] == 1024
    assert dynamic_kwargs["decode_overlap_frames"] == 128
    assert dynamic_kwargs["disable_chunked_decode"] is True
    assert (report_dir / "a.json").exists()
    assert (report_dir / "nested" / "b.json").exists()


def test_infer_cli_dynamic_folder_mode_skips_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    checkpoint_path = tmp_path / "checkpoint"
    input_dir.mkdir()
    checkpoint_path.mkdir()
    output_dir.mkdir()
    (input_dir / "a.wav").write_bytes(b"fake")
    (input_dir / "b.wav").write_bytes(b"fake")
    (output_dir / "a.flac").write_bytes(b"done")
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    config_path = run_dir / "resolved_config.json"
    train_config = load_config(config_path)

    class _Session:
        run_device = "cuda"
        compile_mode = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        infer_cli,
        "_load_runtime_config_and_bundle_payload",
        lambda _path: (train_config, None),
    )
    monkeypatch.setattr(
        infer_cli,
        "build_inference_session",
        lambda **kwargs: _Session(),
    )

    def fake_dynamic_folder(**kwargs: object) -> object:
        captured["dynamic_kwargs"] = kwargs
        jobs = list(cast(Iterable[DynamicInferenceJob], kwargs["jobs"]))
        return SimpleNamespace(
            reports=[
                {
                    "output_audio_path": str(job.output_audio_path),
                    "input_audio_path": str(job.input_audio_path),
                }
                for job in jobs
            ],
            errors=[],
            stats=SimpleNamespace(
                scheduler=SimpleNamespace(
                    model_batches=1,
                    model_queries=1,
                    max_observed_batch_size=1,
                    completed_controllers=1,
                )
            ),
        )

    monkeypatch.setattr(infer_cli, "run_dynamic_folder_inference", fake_dynamic_folder)
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--input-audio",
            str(input_dir),
            "--output-audio",
            str(output_dir),
            "--dynamic-batching",
            "--weights-source",
            "student",
        ],
    )

    infer_cli.main()

    dynamic_kwargs = cast(dict[str, object], captured["dynamic_kwargs"])
    jobs = list(cast(Iterable[DynamicInferenceJob], dynamic_kwargs["jobs"]))
    assert len(jobs) == 1
    assert jobs[0].input_audio_path == input_dir / "b.wav"
    assert jobs[0].output_audio_path == output_dir / "b.flac"


def test_infer_cli_force_overwrite_keeps_existing_outputs_in_work_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    checkpoint_path = tmp_path / "checkpoint"
    input_dir.mkdir()
    checkpoint_path.mkdir()
    output_dir.mkdir()
    (input_dir / "a.wav").write_bytes(b"fake")
    (output_dir / "a.flac").write_bytes(b"done")
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    config_path = run_dir / "resolved_config.json"
    train_config = load_config(config_path)

    class _Session:
        run_device = "cuda"
        compile_mode = None

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        infer_cli,
        "_load_runtime_config_and_bundle_payload",
        lambda _path: (train_config, None),
    )
    monkeypatch.setattr(
        infer_cli,
        "build_inference_session",
        lambda **kwargs: _Session(),
    )

    def fake_dynamic_folder(**kwargs: object) -> object:
        captured["dynamic_kwargs"] = kwargs
        jobs = list(cast(Iterable[DynamicInferenceJob], kwargs["jobs"]))
        return SimpleNamespace(
            reports=[
                {
                    "output_audio_path": str(job.output_audio_path),
                    "input_audio_path": str(job.input_audio_path),
                }
                for job in jobs
            ],
            errors=[],
            stats=SimpleNamespace(
                scheduler=SimpleNamespace(
                    model_batches=1,
                    model_queries=1,
                    max_observed_batch_size=1,
                    completed_controllers=1,
                )
            ),
        )

    monkeypatch.setattr(infer_cli, "run_dynamic_folder_inference", fake_dynamic_folder)
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--input-audio",
            str(input_dir),
            "--output-audio",
            str(output_dir),
            "--dynamic-batching",
            "--force-overwrite",
            "--weights-source",
            "student",
        ],
    )

    infer_cli.main()

    dynamic_kwargs = cast(dict[str, object], captured["dynamic_kwargs"])
    jobs = list(cast(Iterable[DynamicInferenceJob], dynamic_kwargs["jobs"]))
    assert len(jobs) == 1
    assert jobs[0].input_audio_path == input_dir / "a.wav"


def test_export_model_bundle_resolves_5_1_rear_layout_metadata(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=6,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )
    output_dir = tmp_path / "bundle_5_1"

    export_model_bundle(
        train_run_dir=run_dir,
        checkpoint="latest",
        output_dir=output_dir,
        channel_layout_name="5.1 rear",
    )

    payload = json.loads((output_dir / EXPORT_BUNDLE_CONFIG_FILENAME).read_text())
    assert payload["channel_order"] == ["FL", "FR", "FC", "LFE", "BL", "BR"]
    assert payload["channel_mask"] == 0x3F

    config, _ = _load_runtime_config_and_bundle_payload(
        output_dir / EXPORT_BUNDLE_CONFIG_FILENAME
    )
    assert config.training.downmix_channel_order == [
        "FL",
        "FR",
        "FC",
        "LFE",
        "BL",
        "BR",
    ]


def test_export_model_bundle_validates_channel_order_length(tmp_path: Path) -> None:
    run_dir, _ = _write_training_run(
        tmp_path,
        target_channels=2,
        student_state={
            "_orig_mod.linear.weight": torch.zeros((3, 4)),
            "_orig_mod.linear.bias": torch.zeros((3,)),
        },
    )

    with pytest.raises(ValueError, match="channel_order length"):
        export_model_bundle(
            train_run_dir=run_dir,
            checkpoint="latest",
            output_dir=tmp_path / "bundle",
            channel_layout_name="stereo",
            channel_order=["FL", "FR", "FC"],
        )
