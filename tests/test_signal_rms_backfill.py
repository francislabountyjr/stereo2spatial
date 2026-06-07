from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.data.backfill_signal_rms import backfill_signal_rms
from stereo2spatial.training.dataset_io import SAMPLE_BUNDLE_FILENAME


def test_backfill_signal_rms_updates_metadata_and_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.full((2, 8), 4.0),
            "source_stereo_signal": torch.full((2, 8), 3.0),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps({"sample_rate": 48_000, "input_samples": 8}),
        encoding="utf-8",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "stream_hash": "a",
                "sample_dir": "samples/a",
                "target_signal_shape": [2, 8],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = backfill_signal_rms(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        signal_keys=("target_signal", "source_stereo_signal"),
        force=False,
        dry_run=False,
        log_every=0,
    )

    assert summary == {
        "manifest_records": 1,
        "updated": 1,
        "skipped_existing": 0,
        "errors": 0,
    }
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metadata["signal_rms"]["target_signal"] == pytest.approx(4.0)
    assert metadata["signal_rms"]["source_stereo_signal"] == pytest.approx(3.0)
    assert manifest["signal_rms"]["target_signal"] == pytest.approx(4.0)
    assert manifest["signal_rms"]["source_stereo_signal"] == pytest.approx(3.0)
