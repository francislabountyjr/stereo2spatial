from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import stereo2spatial.training.dataset as dataset_module
from scripts.data.normalize_mix_style import normalize_dataset_mix_style
from scripts.data.preprocess_dataset import downmix_multichannel_to_stereo_ac3
from stereo2spatial.common.mix_style import (
    MIX_STYLE_NAMES,
    compute_mix_style_raw,
    mix_style_active_names,
)
from stereo2spatial.modeling import SpatialDiT
from stereo2spatial.training.dataset import WaveformSongDataset
from stereo2spatial.training.losses_windowed import apply_mix_style_dropout

CHANNELS_7_1_4 = [
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
]


def test_compute_mix_style_raw_returns_complete_finite_vector() -> None:
    samples = 2048
    audio = torch.zeros(len(CHANNELS_7_1_4), samples)
    audio[0] = torch.sin(torch.linspace(0.0, 20.0, samples))
    audio[1] = torch.sin(torch.linspace(0.2, 20.2, samples))
    audio[2] = torch.sin(torch.linspace(0.0, 40.0, samples)) * 0.5
    audio[4:] = torch.randn(len(CHANNELS_7_1_4) - 4, samples) * 0.02

    raw = compute_mix_style_raw(
        audio=audio,
        sample_rate=48_000,
        channel_labels=CHANNELS_7_1_4,
    )

    assert list(raw.keys()) == list(MIX_STYLE_NAMES)
    assert all(torch.isfinite(torch.tensor(value)).item() for value in raw.values())


def test_compute_mix_style_raw_handles_binaural_stereo_surrogates() -> None:
    samples = 4096
    mono = torch.sin(torch.linspace(0.0, 80.0, samples))
    wide = torch.stack([mono, -mono], dim=0)
    narrow = torch.stack([mono, mono], dim=0)

    wide_raw = compute_mix_style_raw(
        audio=wide,
        sample_rate=48_000,
        channel_labels=["FL", "FR"],
    )
    narrow_raw = compute_mix_style_raw(
        audio=narrow,
        sample_rate=48_000,
        channel_labels=["FL", "FR"],
    )

    assert list(wide_raw.keys()) == list(MIX_STYLE_NAMES)
    assert all(torch.isfinite(torch.tensor(value)).item() for value in wide_raw.values())
    assert wide_raw["front_width"] > narrow_raw["front_width"]
    assert wide_raw["surround_amount"] > narrow_raw["surround_amount"]


def test_binaural_preprocess_should_not_use_target_as_downmix_placeholder() -> None:
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stereo_source = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    source_downmix = stereo_source.clone()
    target_downmix = downmix_multichannel_to_stereo_ac3(
        multichannel_audio=target,
        channel_labels=["FL", "FR"],
    )

    assert torch.allclose(target_downmix, target)
    assert torch.allclose(source_downmix, stereo_source)
    assert not torch.allclose(source_downmix, target)


def test_normalize_mix_style_updates_manifest_and_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    raw = {name: float(index + 1) for index, name in enumerate(MIX_STYLE_NAMES)}
    metadata = {
        "sample_rate": 48_000,
        "input_samples": 16,
        "mix_style_raw": raw,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    torch.save(
        {
            "target_signal": torch.zeros(12, 16),
            "source_stereo_signal": torch.zeros(2, 16),
            "source_mono_signal": torch.zeros(2, 16),
            "source_downmix_signal": torch.zeros(2, 16),
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [12, 16],
                "mix_style_raw": raw,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = normalize_dataset_mix_style(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        stats_path=dataset_root / "mix_style_stats.json",
        lower_percentile=1.0,
        upper_percentile=99.0,
        dry_run=False,
    )

    assert summary["normalized"] == 1
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(updated_manifest["mix_style"].keys()) == list(MIX_STYLE_NAMES)

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )
    sample = dataset[0]
    assert sample["mix_style"].shape == (len(MIX_STYLE_NAMES),)
    assert sample["target_downmix_signal"].shape == (2, 4, 4)
    assert torch.all((sample["mix_style"] >= 0.0) & (sample["mix_style"] <= 1.0))


def test_waveform_dataset_omits_downmix_batch_key_when_artifact_is_absent(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.zeros(2, 16),
            "source_stereo_signal": torch.ones(2, 16),
            "source_mono_signal": torch.full((2, 16), 0.5),
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [2, 16],
                "mix_style": [0.5] * len(MIX_STYLE_NAMES),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )
    sample = dataset[0]

    assert sample["target_signal"].shape == (2, 4, 4)
    assert sample["cond_signal"].shape == (2, 4, 4)
    assert sample["valid_mask"].tolist() == [True, True, True, True]
    assert "target_downmix_signal" not in sample


def test_waveform_dataset_allows_no_mono_artifact_when_mono_probability_is_zero(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.zeros(2, 16),
            "source_stereo_signal": torch.ones(2, 16),
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [2, 16],
                "mix_style": [0.5] * len(MIX_STYLE_NAMES),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )
    sample = dataset[0]

    assert sample["cond_signal"].shape == (2, 4, 4)
    assert "target_downmix_signal" not in sample


def test_normalize_mix_style_omits_layout_inactive_controls(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    active_names = mix_style_active_names(
        inactive_names=["rear_depth", "height_amount"]
    )
    raw = {name: float(index + 1) for index, name in enumerate(MIX_STYLE_NAMES)}
    metadata = {
        "sample_rate": 48_000,
        "input_samples": 16,
        "mix_style_raw": raw,
        "mix_style_names": list(active_names),
        "mix_style_inactive_names": ["rear_depth", "height_amount"],
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    torch.save(
        {
            "target_signal": torch.zeros(6, 16),
            "source_stereo_signal": torch.zeros(2, 16),
            "source_mono_signal": torch.zeros(2, 16),
            "source_downmix_signal": torch.zeros(2, 16),
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [6, 16],
                "mix_style_raw": raw,
                "mix_style_names": list(active_names),
                "mix_style_inactive_names": ["rear_depth", "height_amount"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    normalize_dataset_mix_style(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        stats_path=dataset_root / "mix_style_stats.json",
        lower_percentile=1.0,
        upper_percentile=99.0,
        dry_run=False,
    )

    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["mix_style_names"] == list(active_names)
    assert len(updated_manifest["mix_style_vector"]) == len(active_names)
    assert "rear_depth" not in updated_manifest["mix_style"]
    assert "height_amount" not in updated_manifest["mix_style"]

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )
    assert dataset[0]["mix_style"].shape == (len(active_names),)


def test_waveform_dataset_loads_multiple_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "ssd_dataset", tmp_path / "hdd_dataset"]
    manifest_paths: list[Path] = []
    for index, root in enumerate(roots, start=1):
        sample_dir = root / "samples" / f"sample_{index}"
        sample_dir.mkdir(parents=True)
        torch.save(
            {
                "target_signal": torch.full((12, 16), float(index)),
                "source_stereo_signal": torch.full((2, 16), float(index)),
                "source_mono_signal": torch.full((2, 16), float(index)),
                "source_downmix_signal": torch.full((2, 16), float(index)),
            },
            sample_dir / "sample_bundle.pt",
        )
        manifest_path = root / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "stream_hash": f"sample_{index}",
                    "sample_dir": f"samples/sample_{index}",
                    "target_signal_shape": [12, 16],
                    "mix_style": [0.5] * len(MIX_STYLE_NAMES),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_paths.append(manifest_path)

    dataset = WaveformSongDataset(
        dataset_root=roots,
        manifest_path=manifest_paths,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )

    assert dataset.describe()["num_datasets"] == 2
    assert dataset.describe()["num_songs"] == 2
    assert len(dataset) == 2
    assert torch.all(dataset[0]["target_signal"] == 1.0)
    assert torch.all(dataset[1]["target_signal"] == 2.0)


def test_waveform_dataset_source_resample_augmentation_is_source_only(
    tmp_path: Path,
) -> None:
    if dataset_module.torchaudio is None:
        pytest.skip("torchaudio is required for source resample augmentation")

    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    samples = 64
    source_mono = torch.where(
        torch.arange(samples) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    source = torch.stack([source_mono, -source_mono], dim=0)
    target = torch.stack(
        [
            torch.linspace(-0.5, 0.5, samples),
            torch.linspace(0.5, -0.5, samples),
        ],
        dim=0,
    )
    torch.save(
        {
            "target_signal": target,
            "source_stereo_signal": source,
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [2, samples],
                "mix_style": [0.5] * len(MIX_STYLE_NAMES),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=4.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=4.0,
        stride_seconds=4.0,
        source_resample_aug_enabled=True,
        source_resample_aug_probability=1.0,
        source_resample_aug_rates=[4_000],
        source_resample_aug_sample_rate=8_000,
    )
    sample = dataset[0]
    expected_target = target.reshape(2, 16, 4).permute(0, 2, 1).contiguous()
    expected_source = source.reshape(2, 16, 4).permute(0, 2, 1).contiguous()

    assert sample["target_signal"].shape == expected_target.shape
    assert sample["cond_signal"].shape == expected_source.shape
    assert torch.allclose(sample["target_signal"], expected_target)
    assert not torch.allclose(sample["cond_signal"], expected_source)


def test_waveform_dataset_source_resample_runs_before_amplitude_lift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_inputs: list[torch.Tensor] = []

    def _fake_resample(
        waveform: torch.Tensor,
        *,
        orig_freq: int,
        new_freq: int,
    ) -> torch.Tensor:
        del orig_freq, new_freq
        recorded_inputs.append(waveform.detach().clone())
        return waveform

    monkeypatch.setattr(
        dataset_module,
        "torchaudio",
        SimpleNamespace(functional=SimpleNamespace(resample=_fake_resample)),
    )

    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    samples = 64
    source = torch.full((2, samples), 0.1, dtype=torch.float32)
    target = torch.full((2, samples), 0.2, dtype=torch.float32)
    torch.save(
        {
            "target_signal": target,
            "source_stereo_signal": source,
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [2, samples],
                "signal_rms": {
                    "source_stereo_signal": 0.1,
                    "target_signal": 0.2,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=4.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=4.0,
        stride_seconds=4.0,
        amplitude_lift_enabled=True,
        amplitude_lift_reference="source",
        amplitude_lift_target_rms=0.33,
        amplitude_lift_scale=3.0,
        source_resample_aug_enabled=True,
        source_resample_aug_probability=1.0,
        source_resample_aug_rates=[4_000],
        source_resample_aug_sample_rate=8_000,
    )

    sample = dataset[0]

    assert recorded_inputs
    assert recorded_inputs[0].abs().amax().item() == pytest.approx(0.1)
    assert sample["cond_signal"].abs().amax().item() == pytest.approx(0.99)


def test_waveform_dataset_augmentation_rng_streams_are_independent(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.zeros(2, 16),
            "source_stereo_signal": torch.zeros(2, 16),
        },
        sample_dir / "sample_bundle.pt",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [2, 16],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=4.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )

    assert dataset._augmentation_rng(0, stream=1).random() != (
        dataset._augmentation_rng(0, stream=2).random()
    )


def test_spatial_dit_mix_style_changes_output() -> None:
    torch.manual_seed(0)
    model = SpatialDiT(
        target_channels=2,
        cond_channels=2,
        patch_size=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        timestep_embed_dim=16,
        timestep_scale=1000.0,
        max_period=10000.0,
        num_memory_tokens=0,
        mix_style_dim=len(MIX_STYLE_NAMES),
    )
    assert isinstance(model.mix_style_mlp, torch.nn.Sequential)
    output_layer = model.mix_style_mlp[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    with torch.no_grad():
        output_layer.weight.fill_(0.01)
        output_layer.bias.fill_(0.05)
    zt = torch.randn(1, 2, 4, 3)
    zc = torch.randn(1, 2, 4, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    t = torch.tensor([0.5])

    low = model(zt=zt, t=t, z_cond=zc, valid_mask=mask, mix_style=torch.zeros(1, 13))
    high = model(zt=zt, t=t, z_cond=zc, valid_mask=mask, mix_style=torch.ones(1, 13))
    missing = model(zt=zt, t=t, z_cond=zc, valid_mask=mask)
    masked = model(
        zt=zt,
        t=t,
        z_cond=zc,
        valid_mask=mask,
        mix_style=torch.ones(1, 13),
        mix_style_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert low.shape == zt.shape
    assert not torch.allclose(low, high)
    assert not torch.allclose(high, missing)
    assert torch.allclose(missing, masked)


def test_mix_style_dropout_marks_style_missing_only_in_training() -> None:
    style = torch.zeros(4, len(MIX_STYLE_NAMES))
    config = type("Config", (), {"mix_style_dropout_probability": 1.0})()
    model = torch.nn.Linear(1, 1)
    model.train()

    dropped_style, dropped_mask = apply_mix_style_dropout(
        mix_style=style,
        training_config=config,
        model=model,
    )

    assert dropped_style is style
    assert dropped_mask is not None
    assert dropped_mask.tolist() == [False, False, False, False]

    model.eval()
    unchanged_style, unchanged_mask = apply_mix_style_dropout(
        mix_style=style,
        training_config=config,
        model=model,
    )

    assert unchanged_style is style
    assert unchanged_mask is None
