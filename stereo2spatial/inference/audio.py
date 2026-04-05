"""Audio I/O helpers for inference entrypoints."""

from __future__ import annotations

import struct
from pathlib import Path

import torch

try:
    import soundfile as sf
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing dependency: soundfile. Install with `pip install soundfile`."
    ) from error

try:
    import torchaudio
except ModuleNotFoundError:
    torchaudio = None

_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_CHANNEL_MASK_7_1_4 = 0x2D63F
_CHANNEL_MASK_BY_COUNT = {
    12: _CHANNEL_MASK_7_1_4,
}


def read_audio_channels_first(
    audio_path: Path,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    """Read an audio file as [channels, samples], resampling when needed."""
    data, source_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data).transpose(0, 1).contiguous()
    if source_sample_rate == target_sample_rate:
        return audio, target_sample_rate

    if torchaudio is None:
        raise RuntimeError(
            f"Sample-rate mismatch for {audio_path} "
            f"({source_sample_rate} != {target_sample_rate}) and torchaudio is missing."
        )
    audio = torchaudio.functional.resample(
        waveform=audio,
        orig_freq=source_sample_rate,
        new_freq=target_sample_rate,
    ).contiguous()
    return audio, target_sample_rate


def _patch_wavex_channel_mask(audio_path: Path, channel_mask: int) -> None:
    """Patch the WAVEX fmt chunk to carry an explicit speaker mask."""
    with audio_path.open("r+b") as handle:
        if handle.read(4) != b"RIFF":
            raise ValueError(f"Expected RIFF header when patching {audio_path}")
        handle.seek(8)
        if handle.read(4) != b"WAVE":
            raise ValueError(f"Expected WAVE header when patching {audio_path}")
        handle.seek(12)

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break

            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                if chunk_size < 40:
                    raise ValueError(
                        f"Expected WAVEX fmt chunk for {audio_path}, got size={chunk_size}"
                    )

                fmt_chunk_start = handle.tell()
                fmt_chunk = handle.read(40)
                if len(fmt_chunk) < 40:
                    raise ValueError(f"Truncated fmt chunk in {audio_path}")

                format_tag = struct.unpack_from("<H", fmt_chunk, 0)[0]
                if format_tag != _WAVE_FORMAT_EXTENSIBLE:
                    raise ValueError(
                        f"Expected WAVEX format tag for {audio_path}, got 0x{format_tag:04X}"
                    )

                handle.seek(fmt_chunk_start + 20)
                handle.write(struct.pack("<I", channel_mask))
                return

            handle.seek(chunk_size + (chunk_size % 2), 1)

    raise ValueError(f"fmt chunk not found in {audio_path}")


def write_audio_channels_first(
    audio_path: Path,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Write [channels, samples] audio, preserving 7.1.4 speaker metadata."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError(
            f"Expected audio shaped [channels, samples], got {tuple(audio.shape)}"
        )

    channel_mask = _CHANNEL_MASK_BY_COUNT.get(int(audio.shape[0]))
    write_kwargs: dict[str, str] = {"subtype": "FLOAT"}
    if channel_mask is not None:
        write_kwargs["format"] = "WAVEX"

    sf.write(
        str(audio_path),
        audio.detach().cpu().transpose(0, 1).contiguous().numpy(),
        sample_rate,
        **write_kwargs,
    )
    if channel_mask is not None:
        _patch_wavex_channel_mask(audio_path=audio_path, channel_mask=channel_mask)
