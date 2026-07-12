from __future__ import annotations

from pathlib import Path

import pytest
import torch

from stereo2spatial.inference.audio import (
    read_audio_channels_first,
    write_audio_channels_first,
)
from stereo2spatial.inference.runner import (
    RequestedSolverName,
    _patch_audio,
    _prepare_conditioning_audio,
    _resolve_inference_mix_style,
    _resolve_inference_solver,
    _unpatch_audio,
    run_inference,
)
from stereo2spatial.modeling import SpatialDiT
from stereo2spatial.training.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
)


def _tiny_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        seed=123,
        output_dir=str(tmp_path / "run"),
        data=DataConfig(
            dataset_root="dataset",
            manifest_path="dataset/manifest.jsonl",
            sample_artifact_mode="bundle",
            segment_seconds=0.01,
            sequence_seconds=0.01,
            stride_seconds=0.01,
            sample_rate=8000,
            mono_probability=0.0,
            downmix_probability=0.0,
            cache_size=1,
            shuffle_segments_within_epoch=False,
            batch_size=1,
            num_workers=0,
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        ),
        model=ModelConfig(
            target_channels=2,
            cond_channels=2,
            patch_size=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            mlp_ratio=1.0,
            dropout=0.0,
            timestep_embed_dim=16,
            timestep_scale=1000.0,
            max_period=10000.0,
            num_memory_tokens=0,
        ),
        training=TrainingConfig(
            max_steps=1,
            grad_accum_steps=1,
            mixed_precision="no",
            compile_model=False,
            compile_mode="default",
            resume_from_checkpoint=None,
            init_from_checkpoint=None,
            grad_clip_norm=1.0,
            log_every=1,
            checkpoint_every=1,
            max_checkpoints_to_keep=1,
            num_epochs_hint=1,
            window_seconds=0.01,
            overlap_seconds=0.0,
            sequence_seconds_choices=[0.01],
            randomize_sequence_per_batch=False,
            detach_memory=False,
            sequence_mode="strided_crops",
            tbptt_windows=0,
            full_song_max_seconds=None,
            require_batch_size_one_for_full_song=True,
            use_gan=False,
            gan_d_lr=1e-4,
            gan_d_beta1=0.0,
            gan_d_beta2=0.9,
            gan_d_base_channels=8,
            gan_d_num_layers=1,
            gan_d_fine_layers=1,
            gan_d_coarse_layers=1,
            gan_d_use_spectral_norm=False,
            gan_use_mask_channel=True,
            gan_ms_w_fine=0.0,
            gan_ms_w_coarse=0.0,
            gan_lambda_adv=0.0,
            gan_adv_warmup_steps=0,
            gan_r1_gamma=1.0,
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
            num_valid_generations=0,
            validation_generation_seed=123,
            validation_generation_input_path=None,
            validation_generation_output_path=None,
            downmix_consistency_weight=0.0,
            downmix_consistency_loss="mse",
        ),
        optimizer=OptimizerConfig(
            type="adamw",
            lr=1e-4,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.99,
            eps=1e-8,
            adamw_fused=False,
            adamw_foreach=False,
        ),
        scheduler=SchedulerConfig(
            type="constant",
            warmup_steps=0,
            min_lr=0.0,
        ),
    )


def test_patch_audio_round_trips_with_padding_trim() -> None:
    audio = torch.arange(20, dtype=torch.float32).reshape(2, 10)

    patches, sample_count = _patch_audio(audio, patch_size=4)
    round_trip = _unpatch_audio(patches, sample_count=sample_count)

    assert sample_count == 10
    assert patches.shape == (2, 4, 3)
    assert torch.equal(round_trip, audio)


def test_prepare_conditioning_audio_maps_mono_and_stereo() -> None:
    mono = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    stereo = torch.stack([torch.zeros(5), torch.ones(5)], dim=0)

    assert torch.equal(
        _prepare_conditioning_audio(mono, cond_channels=2), mono.expand(2, -1)
    )
    assert torch.equal(_prepare_conditioning_audio(mono, cond_channels=1), mono)
    assert torch.equal(
        _prepare_conditioning_audio(stereo, cond_channels=1),
        torch.full((1, 5), 0.5),
    )


def test_resolve_inference_solver_accepts_res6s_aliases() -> None:
    assert _resolve_inference_solver(requested_solver="res6s") == "res6s"
    assert _resolve_inference_solver(requested_solver="res_6s") == "res6s"


@pytest.mark.parametrize("solver", ["midpoint", "midpoint_rk2", "midpoint-rk2", "rk2"])
def test_resolve_inference_solver_accepts_midpoint_rk2_aliases(
    solver: RequestedSolverName,
) -> None:
    assert _resolve_inference_solver(requested_solver=solver) == "midpoint_rk2"


def test_resolve_inference_mix_style_uses_headphone_active_order() -> None:
    mix_style = _resolve_inference_mix_style(
        raw_mix_style=None,
        mix_style_dim=10,
        target_channels=2,
        preset_name="intimate",
    )

    assert mix_style is not None
    assert mix_style.shape == (1, 10)
    assert mix_style.flatten().tolist() == pytest.approx(
        [0.78, 0.82, 0.30, 0.22, 0.20, 0.78, 0.25, 0.22, 0.25, 0.18]
    )


def test_resolve_inference_mix_style_rejects_preset_and_json() -> None:
    with pytest.raises(ValueError, match="cannot both be provided"):
        _resolve_inference_mix_style(
            raw_mix_style={"center_focus": 0.5},
            mix_style_dim=10,
            target_channels=2,
            preset_name="balanced",
        )


@pytest.mark.parametrize("sampling_order", ["window_major", "timestep_major"])
def test_run_inference_writes_direct_waveform_output(
    tmp_path: Path,
    sampling_order: str,
) -> None:
    config = _tiny_config(tmp_path)
    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        patch_size=config.model.patch_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=config.model.num_memory_tokens,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    input_audio_path = tmp_path / "input.wav"
    output_audio_path = tmp_path / f"output_{sampling_order}.wav"
    input_audio = torch.randn(2, 24)
    write_audio_channels_first(
        audio_path=input_audio_path,
        audio=input_audio,
        sample_rate=config.data.sample_rate,
    )

    report = run_inference(
        config=config,
        checkpoint=checkpoint_path,
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        sample_rate=config.data.sample_rate,
        chunk_seconds=0.002,
        overlap_seconds=0.0,
        solver="euler",
        solver_steps=1,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
        device="cpu",
        show_progress=False,
        normalize_peak=False,
        weights_source="student",
        sampling_order=sampling_order,
    )

    output_audio, output_sample_rate = read_audio_channels_first(
        output_audio_path,
        target_sample_rate=config.data.sample_rate,
    )
    assert output_sample_rate == config.data.sample_rate
    assert output_audio.shape == (config.model.target_channels, input_audio.shape[-1])
    assert report["conditioning_signal_shape"] == [2, 8, 3]
    assert report["pred_signal_shape"] == [2, 8, 3]
    assert report["decoded_shape"] == [2, 24]
    assert report["patch_fps"] == 1000.0
    assert report["sampling_order"] == sampling_order


def test_run_inference_reports_amplitude_lift_gain(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    config.data.amplitude_lift_enabled = True
    config.data.amplitude_lift_reference = "source"
    config.data.amplitude_lift_target_rms = 0.5
    config.data.amplitude_lift_scale = 2.0
    config.data.amplitude_lift_clip_value = 4.0

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        patch_size=config.model.patch_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=config.model.num_memory_tokens,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    input_audio_path = tmp_path / "input.wav"
    output_audio_path = tmp_path / "output.wav"
    input_audio = torch.full((2, 24), 0.25)
    write_audio_channels_first(
        audio_path=input_audio_path,
        audio=input_audio,
        sample_rate=config.data.sample_rate,
    )

    report = run_inference(
        config=config,
        checkpoint=checkpoint_path,
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        sample_rate=config.data.sample_rate,
        chunk_seconds=0.002,
        overlap_seconds=0.0,
        solver="euler",
        solver_steps=1,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
        device="cpu",
        show_progress=False,
        normalize_peak=False,
        weights_source="student",
    )

    assert report["amplitude_lift_enabled"] is True
    assert report["amplitude_lift_mode"] == "rms"
    assert report["amplitude_lift_reference"] == "source"
    assert report["amplitude_lift_target_rms"] == 0.5
    assert report["amplitude_lift_scale"] == 2.0
    assert report["amplitude_lift_clip_value"] == 4.0
    assert report["amplitude_lift_gain"] == pytest.approx(2.0)


def test_run_inference_reports_scale_only_amplitude_lift(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    config.data.amplitude_lift_enabled = True
    config.data.amplitude_lift_mode = "scale"
    config.data.amplitude_lift_reference = "source"
    config.data.amplitude_lift_scale = 3.0
    config.data.amplitude_lift_clip_value = 0.1

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        patch_size=config.model.patch_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=config.model.num_memory_tokens,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    input_audio_path = tmp_path / "input.wav"
    output_audio_path = tmp_path / "output.wav"
    write_audio_channels_first(
        audio_path=input_audio_path,
        audio=torch.full((2, 24), 0.25),
        sample_rate=config.data.sample_rate,
    )

    report = run_inference(
        config=config,
        checkpoint=checkpoint_path,
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        sample_rate=config.data.sample_rate,
        chunk_seconds=0.002,
        overlap_seconds=0.0,
        solver="euler",
        solver_steps=1,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
        device="cpu",
        show_progress=False,
        normalize_peak=False,
        weights_source="student",
    )

    assert report["amplitude_lift_enabled"] is True
    assert report["amplitude_lift_mode"] == "scale"
    assert report["amplitude_lift_scale"] == 3.0
    assert report["amplitude_lift_clip_value"] == 0.1
    assert report["amplitude_lift_gain"] == pytest.approx(1.0)


def test_run_inference_supports_wavflow_amplitude_lift(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    config.data.amplitude_lift_enabled = True
    config.data.amplitude_lift_mode = "wavflow"
    config.data.amplitude_lift_reference = "target"
    config.data.amplitude_lift_target_rms = 0.33
    config.data.amplitude_lift_scale = 3.0
    config.data.amplitude_lift_output_lufs = -23.0

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        patch_size=config.model.patch_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=config.model.num_memory_tokens,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    input_audio_path = tmp_path / "input.wav"
    output_audio_path = tmp_path / "output.wav"
    write_audio_channels_first(
        audio_path=input_audio_path,
        audio=torch.full((2, 24), 0.25),
        sample_rate=config.data.sample_rate,
    )

    report = run_inference(
        config=config,
        checkpoint=checkpoint_path,
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        sample_rate=config.data.sample_rate,
        chunk_seconds=0.002,
        overlap_seconds=0.0,
        solver="euler",
        solver_steps=1,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
        device="cpu",
        show_progress=False,
        normalize_peak=False,
        weights_source="student",
    )

    assert report["amplitude_lift_enabled"] is True
    assert report["amplitude_lift_mode"] == "wavflow"
    assert report["amplitude_lift_reference"] == "target"
    assert report["amplitude_lift_target_rms"] == 0.33
    assert report["amplitude_lift_scale"] == 3.0
    assert report["amplitude_lift_output_lufs"] == -23.0
    assert report["amplitude_lift_gain"] == pytest.approx(3.96)
