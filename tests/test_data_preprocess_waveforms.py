from __future__ import annotations

import json

import pytest
import soundfile as sf
import torch

from scripts.data.convert_dataset_to_flac import (
    _convert_one,
    _matching_lengths_or_raise,
)
from scripts.data.find_invalid_samples import _scan_sample
from scripts.data.preprocess_dataset import (
    SOURCE_DOWNMIX_SIGNAL_FILENAME,
    SOURCE_MONO_SIGNAL_FILENAME,
    SOURCE_STEREO_SIGNAL_FILENAME,
    SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
    TARGET_SIGNAL_FILENAME,
    TARGET_SIGNAL_FLAC_FILENAME,
    align_signal_lengths,
    duplicate_mono_to_stereo_width,
    flac_sample_artifacts_exist,
    split_sample_artifacts_exist,
)


def test_align_signal_lengths_trims_to_shortest_sample_count() -> None:
    target = torch.zeros(12, 5)
    stereo = torch.zeros(2, 4)
    mono = torch.zeros(2, 6)
    downmix = torch.zeros(2, 7)

    aligned = align_signal_lengths(
        target_signal=target,
        source_stereo_signal=stereo,
        source_mono_signal=mono,
        source_downmix_signal=downmix,
    )

    assert [tensor.shape[-1] for tensor in aligned] == [4, 4, 4, 4]


def test_duplicate_mono_to_stereo_width_matches_default_condition_channels() -> None:
    mono = torch.arange(6, dtype=torch.float32).unsqueeze(0)

    stereo_width = duplicate_mono_to_stereo_width(mono)

    assert stereo_width.shape == (2, 6)
    assert torch.equal(stereo_width[0], mono[0])
    assert torch.equal(stereo_width[1], mono[0])


def test_split_sample_artifacts_exist_uses_signal_filenames(tmp_path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in [
        TARGET_SIGNAL_FILENAME,
        SOURCE_STEREO_SIGNAL_FILENAME,
        SOURCE_MONO_SIGNAL_FILENAME,
        SOURCE_DOWNMIX_SIGNAL_FILENAME,
    ]:
        (sample_dir / filename).write_bytes(b"x")
    (sample_dir / "metadata.json").write_text(
        json.dumps({"stream_hash": "abc123"}),
        encoding="utf-8",
    )

    assert split_sample_artifacts_exist(
        sample_dir,
        save_source_mono=True,
        save_source_downmix=True,
    )


def test_split_sample_artifacts_exist_allows_optional_mono_and_downmix(
    tmp_path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in [
        TARGET_SIGNAL_FILENAME,
        SOURCE_STEREO_SIGNAL_FILENAME,
    ]:
        (sample_dir / filename).write_bytes(b"x")
    (sample_dir / "metadata.json").write_text(
        json.dumps({"stream_hash": "abc123"}),
        encoding="utf-8",
    )

    assert split_sample_artifacts_exist(
        sample_dir,
        save_source_mono=False,
        save_source_downmix=False,
    )


def test_flac_sample_artifacts_exist_allows_optional_mono_and_downmix(tmp_path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in [
        TARGET_SIGNAL_FLAC_FILENAME,
        SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
    ]:
        (sample_dir / filename).write_bytes(b"x")
    (sample_dir / "metadata.json").write_text(
        json.dumps({"stream_hash": "abc123"}),
        encoding="utf-8",
    )

    assert flac_sample_artifacts_exist(
        sample_dir,
        save_source_mono=False,
        save_source_downmix=False,
    )


def test_convert_rejects_signal_length_mismatch(tmp_path) -> None:
    signals = {
        "target_signal": torch.zeros(2, 12),
        "source_stereo_signal": torch.zeros(2, 10),
    }

    with pytest.raises(RuntimeError, match="length mismatch"):
        _matching_lengths_or_raise(signals, tmp_path / "sample")


def test_invalid_sample_scan_flags_zero_byte_flac(tmp_path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "stream_hash": "abc123",
                "source_path": str(tmp_path / "source.wav"),
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / TARGET_SIGNAL_FLAC_FILENAME).write_bytes(b"")
    sf.write(
        str(sample_dir / SOURCE_STEREO_SIGNAL_FLAC_FILENAME),
        torch.zeros(16, 2).numpy(),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )

    rows = _scan_sample(str(sample_dir))

    assert any("target_signal_flac_invalid" in row.tensor_name for row in rows)


def test_invalid_sample_scan_flags_flac_length_mismatch(tmp_path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "stream_hash": "abc123",
                "source_path": str(tmp_path / "source.wav"),
            }
        ),
        encoding="utf-8",
    )
    sf.write(
        str(sample_dir / TARGET_SIGNAL_FLAC_FILENAME),
        torch.zeros(16, 2).numpy(),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    sf.write(
        str(sample_dir / SOURCE_STEREO_SIGNAL_FLAC_FILENAME),
        torch.zeros(12, 2).numpy(),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )

    rows = _scan_sample(str(sample_dir))

    assert any(row.tensor_name == "length_mismatch" for row in rows)


def test_convert_to_flac_resumes_existing_valid_output_without_source_tensors(
    tmp_path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    sample_dir = input_root / "samples" / "ab" / "cd" / "abcd"
    output_sample_dir = output_root / "samples" / "ab" / "cd" / "abcd"
    sample_dir.mkdir(parents=True)
    output_sample_dir.mkdir(parents=True)
    metadata = {
        "stream_hash": "abcd",
        "source_path": str(tmp_path / "source.wav"),
        "sample_rate": 48_000,
        "target_channels": 2,
        "target_channel_labels": ["FL", "FR"],
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for filename in [TARGET_SIGNAL_FLAC_FILENAME, SOURCE_STEREO_SIGNAL_FLAC_FILENAME]:
        sf.write(
            str(output_sample_dir / filename),
            torch.zeros(16, 2).numpy(),
            48_000,
            format="FLAC",
            subtype="PCM_24",
        )

    result = _convert_one(
        str(sample_dir),
        str(input_root),
        str(output_root),
        False,
    )

    assert result.ok is True
    assert result.bytes_in == 0
    assert result.manifest_record is not None
    assert (output_sample_dir / "metadata.json").exists()
