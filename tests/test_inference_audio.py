from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch

import stereo2spatial.inference.audio as audio_io
from stereo2spatial.inference.audio import (
    read_audio_channels_first,
    write_audio_channels_first,
)


def _raise_missing_ffprobe(*_args: object, **_kwargs: object) -> object:
    raise FileNotFoundError("ffprobe")


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


def test_read_audio_channels_first_falls_back_to_torchaudio_for_m4a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.m4a"
    input_path.write_bytes(b"fake m4a")
    expected_audio = torch.tensor([[0.0, 0.25, 0.5], [0.5, 0.25, 0.0]])
    loaded_paths: list[str] = []

    def fake_sf_read(*_args: object, **_kwargs: object) -> object:
        raise sf.SoundFileError("Format not recognised")

    class _FakeTorchaudio:
        @staticmethod
        def load(path: str) -> tuple[torch.Tensor, int]:
            loaded_paths.append(path)
            return expected_audio.clone(), 48_000

    monkeypatch.setattr(audio_io.sf, "read", fake_sf_read)
    monkeypatch.setattr(audio_io, "torchaudio", _FakeTorchaudio)
    monkeypatch.setattr(audio_io.subprocess, "run", _raise_missing_ffprobe)

    audio, sample_rate = read_audio_channels_first(
        audio_path=input_path,
        target_sample_rate=48_000,
    )

    assert loaded_paths == [str(input_path)]
    assert sample_rate == 48_000
    assert torch.equal(audio, expected_audio)


def test_read_audio_channels_first_reports_decoder_failures_after_soundfile_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.m4a"
    input_path.write_bytes(b"fake m4a")

    def fake_sf_read(*_args: object, **_kwargs: object) -> object:
        raise sf.SoundFileError("Format not recognised")

    monkeypatch.setattr(audio_io.sf, "read", fake_sf_read)
    monkeypatch.setattr(audio_io, "torchaudio", None)
    monkeypatch.setattr(audio_io.subprocess, "run", _raise_missing_ffprobe)

    with pytest.raises(RuntimeError, match="Tried soundfile, ffmpeg, and torchaudio"):
        read_audio_channels_first(
            audio_path=input_path,
            target_sample_rate=48_000,
        )


def test_read_audio_channels_first_uses_ffmpeg_for_m4a_after_soundfile_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.m4a"
    input_path.write_bytes(b"fake m4a")
    expected_audio = torch.tensor([[0.0, 0.25, 0.5], [0.5, 0.25, 0.0]])
    ffmpeg_payload = (
        expected_audio.t()
        .contiguous()
        .numpy()
        .astype("float32", copy=False)
        .tobytes()
    )
    commands: list[list[str]] = []

    def fake_sf_read(*_args: object, **_kwargs: object) -> object:
        raise sf.SoundFileError("Format not recognised")

    class _FakeTorchaudio:
        @staticmethod
        def load(_path: str) -> tuple[torch.Tensor, int]:
            raise AssertionError("torchaudio should not be used before ffmpeg for m4a")

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0,
                stdout='{"streams": [{"channels": 2}]}',
                stderr="",
            )
        if command[0] == "ffmpeg":
            return SimpleNamespace(returncode=0, stdout=ffmpeg_payload, stderr=b"")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(audio_io.sf, "read", fake_sf_read)
    monkeypatch.setattr(audio_io, "torchaudio", _FakeTorchaudio)
    monkeypatch.setattr(audio_io.subprocess, "run", fake_run)

    audio, sample_rate = read_audio_channels_first(
        audio_path=input_path,
        target_sample_rate=48_000,
    )

    assert [command[0] for command in commands] == ["ffprobe", "ffmpeg"]
    assert commands[1][commands[1].index("-ar") + 1] == "48000"
    assert sample_rate == 48_000
    assert torch.equal(audio, expected_audio)


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
