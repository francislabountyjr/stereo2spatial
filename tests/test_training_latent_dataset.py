from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from stereo2spatial.modeling import LegacySpatialDiT
from stereo2spatial.training.components import build_training_components
from stereo2spatial.training.config import load_config
from stereo2spatial.training.dataset_types import ConditioningSource
from stereo2spatial.training.latent_dataset import LatentSongDataset
from stereo2spatial.training.latent_dataset_io import (
    SAMPLE_BUNDLE_FILENAME,
    SOURCE_DOWNMIX_LATENT_FILENAME,
    SOURCE_MONO_LATENT_FILENAME,
    SOURCE_STEREO_LATENT_FILENAME,
    TARGET_LATENT_FILENAME,
    _load_latents_from_sample,
)


def _latent_payload(
    *,
    frames: int,
    target_channels: int = 2,
    latent_dim: int = 3,
    offset: float = 0.0,
) -> dict[str, torch.Tensor]:
    target = torch.arange(
        target_channels * latent_dim * frames,
        dtype=torch.float64,
    ).reshape(target_channels, latent_dim, frames)
    return {
        "target_latent": target + offset,
        # Old source artifacts commonly stored [D,T] and relied on loader normalization.
        "source_stereo_latent": torch.full(
            (latent_dim, frames), 10.0 + offset, dtype=torch.float16
        ),
        "source_mono_latent": torch.full(
            (latent_dim, frames), 20.0 + offset, dtype=torch.float64
        ),
        "source_downmix_latent": torch.full(
            (1, latent_dim, frames), 30.0 + offset, dtype=torch.float32
        ),
    }


def _write_sample(
    root: Path,
    *,
    name: str,
    stream_hash: str,
    mode: str,
    frames: int,
    offset: float = 0.0,
    write_metadata: bool = True,
) -> tuple[Path, dict[str, object]]:
    sample_dir = root / "samples" / name[:1] / name
    sample_dir.mkdir(parents=True)
    payload = _latent_payload(frames=frames, offset=offset)
    if mode == "bundle":
        torch.save(payload, sample_dir / SAMPLE_BUNDLE_FILENAME)
    elif mode == "split":
        filenames = {
            "target_latent": TARGET_LATENT_FILENAME,
            "source_stereo_latent": SOURCE_STEREO_LATENT_FILENAME,
            "source_mono_latent": SOURCE_MONO_LATENT_FILENAME,
            "source_downmix_latent": SOURCE_DOWNMIX_LATENT_FILENAME,
        }
        for key, filename in filenames.items():
            torch.save(payload[key], sample_dir / filename)
    else:
        raise ValueError(mode)

    if write_metadata:
        (sample_dir / "metadata.json").write_text(
            json.dumps({"sample_rate": 100, "input_samples": frames * 2}),
            encoding="utf-8",
        )

    # Include an obsolete root prefix and Windows separators to exercise current
    # sample-dir normalization back to <configured root>/samples/...
    manifest_row: dict[str, object] = {
        "stream_hash": stream_hash,
        "sample_dir": f"obsolete\\dataset\\samples\\{name[:1]}\\{name}",
        "target_latent_shape": [2, 3, frames],
    }
    if not write_metadata:
        manifest_row.update({"sample_rate": 100, "input_samples": frames * 2})
    return sample_dir, manifest_row


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _dataset(
    root: Path,
    manifest: Path | str,
    **overrides: object,
) -> LatentSongDataset:
    kwargs: dict[str, object] = {
        "dataset_root": root,
        "manifest_path": manifest,
        "sample_artifact_mode": "bundle",
        "segment_seconds": 2.0,
        "latent_fps": 2.0,
        "mono_probability": 0.0,
        "downmix_probability": 0.0,
        "cache_size": 1,
        "shuffle_segments_within_epoch": False,
        "shuffle_segments_within_song": False,
        "seed": 17,
        "sequence_seconds": 2.0,
        "stride_seconds": 2.0,
    }
    kwargs.update(overrides)
    return LatentSongDataset(**kwargs)  # type: ignore[arg-type]


def test_bundle_dataset_emits_generic_keys_and_pads_mono_conditioning(
    tmp_path: Path,
) -> None:
    _, row = _write_sample(
        tmp_path,
        name="song-a",
        stream_hash="hash-a",
        mode="bundle",
        frames=3,
    )
    manifest = _write_manifest(tmp_path, [row])
    dataset = _dataset(tmp_path, manifest, mono_probability=1.0)

    sample = dataset[0]

    assert set(sample) == {
        "target_signal",
        "cond_signal",
        "valid_mask",
        "song_index",
        "start_frame",
        "conditioning_source",
    }
    assert sample["target_signal"].shape == (2, 3, 4)
    assert sample["cond_signal"].shape == (1, 3, 4)
    assert sample["target_signal"].dtype == torch.float32
    assert sample["cond_signal"].dtype == torch.float32
    assert sample["valid_mask"].tolist() == [True, True, True, False]
    assert torch.all(sample["cond_signal"][..., :3] == 20.0)
    assert torch.all(sample["cond_signal"][..., 3] == 0.0)
    assert int(sample["conditioning_source"].item()) == int(ConditioningSource.MONO)


def test_split_artifacts_are_normalized_and_dataset_full_song_is_capped(
    tmp_path: Path,
) -> None:
    sample_dir, row = _write_sample(
        tmp_path,
        name="song-b",
        stream_hash="hash-b",
        mode="split",
        frames=7,
    )
    manifest = _write_manifest(tmp_path, [row])

    loaded = _load_latents_from_sample(sample_dir, sample_artifact_mode="split")
    assert loaded["target_latent"].shape == (2, 3, 7)
    assert loaded["source_stereo_latent"].shape == (1, 3, 7)
    assert all(value.dtype == torch.float32 for value in loaded.values())
    assert all(value.is_contiguous() for value in loaded.values())

    dataset = _dataset(
        tmp_path,
        manifest,
        sample_artifact_mode="split",
        latent_fps=2.0,
        sequence_mode="full_song",
        full_song_max_seconds=2.0,
    )
    sample = dataset[0]
    assert sample["target_signal"].shape == (2, 3, 4)
    assert sample["cond_signal"].shape == (1, 3, 4)
    assert sample["valid_mask"].tolist() == [True] * 4
    assert int(sample["start_frame"].item()) == 0


def test_multi_dataset_manifest_normalization_exclusion_and_auto_fps(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "dataset-a"
    root_b = tmp_path / "dataset-b"
    root_a.mkdir()
    root_b.mkdir()
    _, row_a = _write_sample(
        root_a,
        name="song-a",
        stream_hash="hash-a",
        mode="bundle",
        frames=5,
    )
    sample_b, row_b = _write_sample(
        root_b,
        name="song-b",
        stream_hash="hash-b",
        mode="bundle",
        frames=6,
        offset=100.0,
        write_metadata=False,
    )
    _write_manifest(root_a, [row_a])
    _write_manifest(root_b, [row_b])
    exclusions = tmp_path / "exclude.json"
    exclusions.write_text(
        json.dumps({"exclude_stream_hashes": ["hash-a"]}),
        encoding="utf-8",
    )

    dataset = LatentSongDataset(
        dataset_root=[root_a, root_b],
        manifest_path=["manifest.jsonl", "manifest.jsonl"],
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        latent_fps="auto",
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        shuffle_segments_within_song=False,
        sample_exclusion_path=exclusions,
        seed=4,
        sequence_seconds=1.0,
        stride_seconds=1.0,
    )

    assert len(dataset._songs) == 1
    assert dataset._songs[0].sample_dir == sample_b
    assert dataset.resolved_latent_fps == pytest.approx(50.0)
    assert dataset.resolved_patch_fps == pytest.approx(50.0)
    dataset.set_global_step(12345)

    description = dataset.describe()
    assert description["sample_domain"] == "vae_latent"
    assert description["num_datasets"] == 2
    assert description["num_songs"] == 1
    assert description["excluded_song_count"] == 1
    assert description["resolved_patch_fps"] == pytest.approx(50.0)


def test_epoch_planner_honors_within_song_shuffle_flag(tmp_path: Path) -> None:
    _, row = _write_sample(
        tmp_path,
        name="song-c",
        stream_hash="hash-c",
        mode="bundle",
        frames=12,
    )
    manifest = _write_manifest(tmp_path, [row])
    ordered = _dataset(
        tmp_path,
        manifest,
        latent_fps=1.0,
        sequence_seconds=4.0,
        stride_seconds=2.0,
        shuffle_segments_within_song=False,
        seed=123,
    )
    shuffled = _dataset(
        tmp_path,
        manifest,
        latent_fps=1.0,
        sequence_seconds=4.0,
        stride_seconds=2.0,
        shuffle_segments_within_song=True,
        seed=123,
    )

    ordered_starts = [segment.start_frame for segment in ordered._segments]
    shuffled_starts = [segment.start_frame for segment in shuffled._segments]
    assert ordered_starts == sorted(ordered_starts)
    assert sorted(shuffled_starts) == ordered_starts
    assert shuffled_starts != ordered_starts

    shuffled.set_epoch(3)
    first = list(shuffled._segments)
    shuffled.set_epoch(3)
    assert shuffled._segments == first


def test_dataset_rejects_mismatched_root_and_manifest_lists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same number"):
        LatentSongDataset(
            dataset_root=[tmp_path, tmp_path / "other"],
            manifest_path=[tmp_path / "manifest.jsonl"],
            sample_artifact_mode="bundle",
            segment_seconds=1.0,
            latent_fps=50.0,
            mono_probability=0.0,
            downmix_probability=0.0,
            cache_size=0,
            shuffle_segments_within_epoch=False,
            seed=1,
        )


def test_legacy_training_components_complete_backward_and_optimizer_step(
    tmp_path: Path,
) -> None:
    _, row = _write_sample(
        tmp_path,
        name="song-train",
        stream_hash="hash-train",
        mode="bundle",
        frames=4,
    )
    manifest = _write_manifest(tmp_path, [row])
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "train_legacy_vae.yaml"
    )
    config.data.dataset_root = str(tmp_path)
    config.data.manifest_path = str(manifest)
    config.data.segment_seconds = 0.08
    config.data.sequence_seconds = 0.08
    config.data.stride_seconds = 0.08
    config.data.latent_fps = 50.0
    config.data.num_workers = 0
    config.data.batch_size = 1
    config.model.target_channels = 2
    config.model.cond_channels = 1
    config.model.patch_size = 3
    config.model.latent_dim = 3
    config.model.hidden_dim = 8
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.timestep_embed_dim = 8
    config.model.num_memory_tokens = 0
    config.training.sequence_seconds_choices = [0.08]
    config.training.sequence_mode = "strided_crops"
    config.training.window_seconds = 0.08
    config.training.overlap_seconds = 0.0
    config.optimizer.adamw_fused = False

    dataset, model, optimizer = build_training_components(config)
    assert isinstance(model, LegacySpatialDiT)
    sample = dataset[0]
    target = sample["target_signal"].unsqueeze(0)
    conditioning = sample["cond_signal"].unsqueeze(0)
    valid_mask = sample["valid_mask"].unsqueeze(0)
    t = torch.tensor([0.4])
    noise = torch.randn_like(target)
    zt = (1.0 - t[:, None, None, None]) * noise + t[:, None, None, None] * target

    prediction = model(
        zt=zt,
        t=t,
        z_cond=conditioning,
        valid_mask=valid_mask,
    )
    loss = (prediction - target).square().mean()
    loss.backward()
    assert model.final_proj.weight.grad is not None
    optimizer.step()
