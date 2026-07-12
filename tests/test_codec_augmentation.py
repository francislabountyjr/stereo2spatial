from __future__ import annotations

import shutil

import pytest
import torch

from stereo2spatial.common.codec_augmentation import codec_roundtrip


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_codec_roundtrip_preserves_shape_and_alignment() -> None:
    sample_rate = 48_000
    samples = sample_rate // 2
    waveform = torch.zeros(2, samples, dtype=torch.float32)
    waveform[:, 1000] = 0.75

    decoded = codec_roundtrip(
        waveform,
        sample_rate=sample_rate,
        codec="mp3",
        bitrate_kbps=128,
        backend="ffmpeg",
        align=True,
        align_max_lag=4096,
        timeout_seconds=20.0,
    )

    assert decoded.shape == waveform.shape
    peak_index = int(decoded.abs().mean(dim=0).argmax().item())
    assert abs(peak_index - 1000) <= 2
    assert torch.isfinite(decoded).all()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_auto_codec_roundtrip_falls_back_to_available_backend() -> None:
    sample_rate = 48_000
    time = torch.arange(sample_rate // 4, dtype=torch.float32) / sample_rate
    waveform = torch.stack(
        [
            0.25 * torch.sin(2.0 * torch.pi * 440.0 * time),
            0.25 * torch.sin(2.0 * torch.pi * 660.0 * time),
        ],
        dim=0,
    )

    decoded = codec_roundtrip(
        waveform,
        sample_rate=sample_rate,
        codec="opus",
        bitrate_kbps=96,
        backend="auto",
        align=True,
        align_max_lag=4096,
        timeout_seconds=20.0,
    )

    assert decoded.shape == waveform.shape
    assert torch.isfinite(decoded).all()
