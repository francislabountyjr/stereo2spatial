from __future__ import annotations

import struct
from pathlib import Path

import pytest
import soundfile as sf
import torch

from stereo2spatial.inference.audio import write_audio_channels_first


def _read_fmt_chunk(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        if handle.read(4) != b"RIFF":
            raise AssertionError("missing RIFF header")
        handle.seek(8)
        if handle.read(4) != b"WAVE":
            raise AssertionError("missing WAVE signature")
        handle.seek(12)

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break

            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                fmt_chunk = handle.read(chunk_size)
                if len(fmt_chunk) != chunk_size:
                    raise AssertionError("truncated fmt chunk")

                info = {
                    "chunk_size": chunk_size,
                    "format_tag": struct.unpack_from("<H", fmt_chunk, 0)[0],
                    "channels": struct.unpack_from("<H", fmt_chunk, 2)[0],
                    "sample_rate": struct.unpack_from("<I", fmt_chunk, 4)[0],
                    "block_align": struct.unpack_from("<H", fmt_chunk, 12)[0],
                    "bits_per_sample": struct.unpack_from("<H", fmt_chunk, 14)[0],
                }
                if chunk_size >= 40:
                    info["channel_mask"] = struct.unpack_from("<I", fmt_chunk, 20)[0]
                return info

            handle.seek(chunk_size + (chunk_size % 2), 1)

    raise AssertionError("fmt chunk not found")


def test_write_audio_channels_first_sets_7_1_4_channel_mask(tmp_path: Path) -> None:
    output_path = tmp_path / "render_7_1_4.wav"
    audio = torch.zeros((12, 128), dtype=torch.float32)

    write_audio_channels_first(
        audio_path=output_path,
        audio=audio,
        sample_rate=48_000,
    )

    fmt_chunk = _read_fmt_chunk(output_path)
    assert fmt_chunk == {
        "chunk_size": 40,
        "format_tag": 0xFFFE,
        "channels": 12,
        "sample_rate": 48_000,
        "block_align": 48,
        "bits_per_sample": 32,
        "channel_mask": 0x2D63F,
    }


def test_write_audio_channels_first_sets_5_1_rear_channel_mask(tmp_path: Path) -> None:
    output_path = tmp_path / "render_5_1_rear.wav"
    audio = torch.zeros((6, 128), dtype=torch.float32)

    write_audio_channels_first(
        audio_path=output_path,
        audio=audio,
        sample_rate=48_000,
    )

    fmt_chunk = _read_fmt_chunk(output_path)
    assert fmt_chunk == {
        "chunk_size": 40,
        "format_tag": 0xFFFE,
        "channels": 6,
        "sample_rate": 48_000,
        "block_align": 24,
        "bits_per_sample": 32,
        "channel_mask": 0x3F,
    }


def test_write_audio_channels_first_accepts_explicit_5_1_side_order(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "render_5_1_side.wav"
    audio = torch.zeros((6, 128), dtype=torch.float32)

    write_audio_channels_first(
        audio_path=output_path,
        audio=audio,
        sample_rate=48_000,
        channel_order=["FL", "FR", "FC", "LFE", "SL", "SR"],
    )

    fmt_chunk = _read_fmt_chunk(output_path)
    assert fmt_chunk["channel_mask"] == 0x60F


def test_write_audio_channels_first_keeps_standard_wav_for_stereo(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "render_stereo.wav"
    audio = torch.zeros((2, 128), dtype=torch.float32)

    write_audio_channels_first(
        audio_path=output_path,
        audio=audio,
        sample_rate=48_000,
    )

    fmt_chunk = _read_fmt_chunk(output_path)
    assert fmt_chunk == {
        "chunk_size": 16,
        "format_tag": 0x3,
        "channels": 2,
        "sample_rate": 48_000,
        "block_align": 8,
        "bits_per_sample": 32,
    }


def test_write_audio_channels_first_keeps_standard_wav_for_explicit_stereo_order(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "render_binaural.wav"
    audio = torch.zeros((2, 128), dtype=torch.float32)

    write_audio_channels_first(
        audio_path=output_path,
        audio=audio,
        sample_rate=48_000,
        channel_order=["FL", "FR"],
    )

    fmt_chunk = _read_fmt_chunk(output_path)
    assert fmt_chunk["format_tag"] == 0x3
    assert fmt_chunk["channels"] == 2
    assert "channel_mask" not in fmt_chunk


def test_write_audio_channels_first_writes_stereo_flac_pcm24(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "render_binaural.flac"
    audio = torch.zeros((2, 128), dtype=torch.float32)

    write_audio_channels_first(
        audio_path=output_path,
        audio=audio,
        sample_rate=48_000,
    )

    info = sf.info(str(output_path))
    assert info.format == "FLAC"
    assert info.subtype == "PCM_24"
    assert info.channels == 2
    assert info.samplerate == 48_000


def test_write_audio_channels_first_rejects_multichannel_flac_with_mask(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "render_5_1.flac"
    audio = torch.zeros((6, 128), dtype=torch.float32)

    with pytest.raises(ValueError, match="Use .wav for multichannel"):
        write_audio_channels_first(
            audio_path=output_path,
            audio=audio,
            sample_rate=48_000,
        )
