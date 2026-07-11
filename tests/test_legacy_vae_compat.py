from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open

from stereo2spatial.common.checkpoints import (
    detect_state_dict_architecture,
    validate_state_dict_architecture,
)
from stereo2spatial.inference.export_bundle import (
    build_train_config_from_bundle_payload,
    load_inference_bundle_payload,
    resolve_bundle_vae_paths,
)
from stereo2spatial.inference.runner import (
    InferenceSession,
    run_inference_with_session,
)
from stereo2spatial.modeling import (
    LegacySpatialDiT,
    SpatialDiT,
    build_spatial_model,
)
from stereo2spatial.training.config import load_config

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CHECKPOINT = (
    ROOT
    / "runs"
    / "train_with_gan_stage_2"
    / "checkpoints"
    / "step_0040000"
    / "model.safetensors"
)
WAVEFORM_CHECKPOINT = ROOT / "checkpoints" / "step_0630000" / "model.safetensors"
LEGACY_BUNDLE = (
    ROOT / "runs" / "train_with_gan_stage_2" / "exported_bundle_step_0040000"
)


def _normalized_checkpoint_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key.removeprefix("_orig_mod.").removeprefix("module."): tuple(
                handle.get_slice(key).get_shape()
            )
            for key in handle.keys()
        }


def _assert_meta_model_matches_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path
) -> None:
    expected = {key: tuple(tensor.shape) for key, tensor in model.state_dict().items()}
    assert _normalized_checkpoint_shapes(checkpoint_path) == expected


def test_legacy_velocity_head_is_adapted_to_current_clean_prediction_contract() -> None:
    model = LegacySpatialDiT(
        target_channels=1,
        cond_channels=1,
        latent_dim=2,
        hidden_dim=8,
        num_layers=0,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        timestep_embed_dim=8,
        timestep_scale=1000.0,
        max_period=10000.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.final_proj.bias.copy_(torch.tensor([2.0, -4.0]))

    zt = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    cond = torch.zeros_like(zt)
    t = torch.tensor([0.25])
    clean = model(zt=zt, t=t, z_cond=cond)
    expected_velocity = torch.tensor([[[[2.0, 2.0], [-4.0, -4.0]]]])
    assert torch.allclose(clean, zt + 0.75 * expected_velocity)

    clean.sum().backward()
    assert model.final_proj.bias.grad is not None


def test_model_factory_selects_both_architectures() -> None:
    legacy = load_config(ROOT / "configs" / "train_legacy_vae.yaml")
    waveform = load_config(ROOT / "configs" / "train.yaml")
    with torch.device("meta"):
        assert isinstance(build_spatial_model(legacy.model), LegacySpatialDiT)
        assert isinstance(build_spatial_model(waveform.model), SpatialDiT)


@pytest.mark.skipif(
    not LEGACY_BUNDLE.exists(), reason="local legacy bundle unavailable"
)
def test_historical_flat_bundle_auto_detects_legacy_and_resolves_vae() -> None:
    payload = load_inference_bundle_payload(LEGACY_BUNDLE / "config.json")
    config = build_train_config_from_bundle_payload(
        payload,
        bundle_root=LEGACY_BUNDLE,
    )
    assert config.model.architecture == "legacy_vae"
    assert config.model.latent_dim == 64
    assert config.model.patch_size == 64
    vae_checkpoint, vae_config = resolve_bundle_vae_paths(LEGACY_BUNDLE)
    assert vae_checkpoint is not None and vae_checkpoint.exists()
    assert vae_config is not None and vae_config.exists()


@pytest.mark.skipif(
    not LEGACY_CHECKPOINT.exists(), reason="local legacy checkpoint unavailable"
)
def test_provided_legacy_checkpoint_exactly_matches_legacy_architecture() -> None:
    config = load_config(
        ROOT / "runs" / "train_with_gan_stage_2" / "resolved_config.json"
    )
    with torch.device("meta"):
        model = build_spatial_model(config.model)
    _assert_meta_model_matches_checkpoint(model, LEGACY_CHECKPOINT)
    assert detect_state_dict_architecture(dict(model.state_dict())) == "legacy_vae"


@pytest.mark.skipif(
    not WAVEFORM_CHECKPOINT.exists(), reason="local waveform checkpoint unavailable"
)
def test_provided_waveform_checkpoint_exactly_matches_current_architecture() -> None:
    config = load_config(ROOT / "configs" / "train_headphone_virtualizer.yaml")
    with torch.device("meta"):
        model = build_spatial_model(config.model)
    _assert_meta_model_matches_checkpoint(model, WAVEFORM_CHECKPOINT)
    assert detect_state_dict_architecture(dict(model.state_dict())) == "waveform"


def test_checkpoint_architecture_mismatch_has_actionable_error() -> None:
    legacy = LegacySpatialDiT(
        target_channels=1,
        cond_channels=1,
        latent_dim=2,
        hidden_dim=8,
        num_layers=0,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        timestep_embed_dim=8,
        timestep_scale=1000.0,
        max_period=10000.0,
    )
    waveform = SpatialDiT(
        target_channels=1,
        cond_channels=1,
        patch_size=2,
        hidden_dim=8,
        num_layers=0,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        timestep_embed_dim=8,
        timestep_scale=1000.0,
        max_period=10000.0,
        rope_enabled=False,
    )
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        validate_state_dict_architecture(legacy, dict(waveform.state_dict()))


@pytest.mark.parametrize(
    ("sampling_order", "expected_order"),
    [
        ("window_major", "window_major"),
        ("timestep_major", "timestep_major"),
        (None, "timestep_major"),
    ],
)
def test_legacy_inference_uses_vae_codec_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sampling_order: str | None,
    expected_order: str,
) -> None:
    payload = {
        "model_type": "spatial_dit",
        "sample_rate": 48_000,
        "latent_fps": 50,
        "target_channels": 2,
        "cond_channels": 1,
        "latent_dim": 3,
        "hidden_dim": 8,
        "num_layers": 0,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "timestep_embed_dim": 8,
        "timestep_scale": 1000.0,
        "max_period": 10000.0,
        "num_memory_tokens": 0,
    }
    config = build_train_config_from_bundle_payload(payload)
    model = build_spatial_model(config.model)
    fake_vae = torch.nn.Identity()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "stereo2spatial.inference.runner.read_audio_channels_first",
        lambda **_: (torch.ones(2, 960), 48_000),
    )

    def fake_encode(**kwargs: object) -> torch.Tensor:
        captured["encode_audio_shape"] = tuple(getattr(kwargs["audio"], "shape"))
        return torch.ones(3, 1)

    def fake_generate(order: str, **kwargs: object) -> torch.Tensor:
        cond = kwargs["cond_signal"]
        captured["sampling_order"] = order
        captured["cond_shape"] = tuple(getattr(cond, "shape"))
        captured["chunk_frames"] = kwargs["chunk_frames"]
        return torch.ones(2, 3, 1)

    def fake_decode(**kwargs: object) -> torch.Tensor:
        captured["latent_shape"] = tuple(getattr(kwargs["channel_latents"], "shape"))
        return torch.full((2, 960), 0.25)

    monkeypatch.setattr("stereo2spatial.inference.runner.vae_encode", fake_encode)
    monkeypatch.setattr(
        "stereo2spatial.inference.runner.generate_spatial_signal",
        lambda **kwargs: fake_generate("window_major", **kwargs),
    )
    monkeypatch.setattr(
        "stereo2spatial.inference.runner.generate_spatial_signal_timestep_major",
        lambda **kwargs: fake_generate("timestep_major", **kwargs),
    )
    monkeypatch.setattr(
        "stereo2spatial.inference.runner.decode_channels_independent", fake_decode
    )
    monkeypatch.setattr(
        "stereo2spatial.inference.runner.write_audio_channels_first",
        lambda **kwargs: captured.update(written_shape=tuple(kwargs["audio"].shape)),
    )

    session = InferenceSession(
        config=config,
        model=model,
        checkpoint_path=tmp_path / "model.safetensors",
        run_device=torch.device("cpu"),
        used_weights_source="student",
        compiled=False,
        compile_mode=None,
        vae=fake_vae,
    )
    inference_kwargs: dict[str, Any] = {
        "session": session,
        "input_audio_path": tmp_path / "input.wav",
        "output_audio_path": tmp_path / "output.wav",
        "sample_rate": 48_000,
        "chunk_seconds": 1.0,
        "overlap_seconds": 0.0,
        "solver": "euler",
        "solver_steps": 1,
        "solver_rtol": 1e-5,
        "solver_atol": 1e-5,
        "seed": 1,
        "show_progress": False,
        "normalize_peak": False,
    }
    if sampling_order is not None:
        inference_kwargs["sampling_order"] = sampling_order
    report = run_inference_with_session(**inference_kwargs)

    assert captured == {
        "encode_audio_shape": (2, 960),
        "sampling_order": expected_order,
        "cond_shape": (1, 3, 1),
        "chunk_frames": 50,
        "latent_shape": (2, 3, 1),
        "written_shape": (2, 960),
    }
    assert report["architecture"] == "legacy_vae"
    assert report["representation"] == "latent"
    assert report["patch_fps"] == 50.0
    assert report["sampling_order"] == expected_order

    with pytest.raises(ValueError, match="sample_rate=48000"):
        run_inference_with_session(
            session=session,
            input_audio_path=tmp_path / "input.wav",
            output_audio_path=tmp_path / "bad-rate.wav",
            sample_rate=44_100,
            chunk_seconds=None,
            overlap_seconds=0.0,
            solver="euler",
            solver_steps=1,
            solver_rtol=1e-5,
            solver_atol=1e-5,
            seed=1,
            show_progress=False,
            normalize_peak=False,
        )
