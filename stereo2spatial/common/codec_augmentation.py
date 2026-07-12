"""Source-audio codec roundtrip augmentation helpers."""

from __future__ import annotations

import subprocess
from io import BytesIO
from typing import Literal, cast

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional dependency path
    torchaudio = None

CodecName = Literal["mp3", "aac", "opus"]
CodecBackend = Literal["auto", "torchaudio", "ffmpeg"]

_ENCODE_FORMAT_BY_CODEC: dict[str, str] = {
    "mp3": "mp3",
    "aac": "adts",
    "opus": "ogg",
}
_DECODE_FORMAT_BY_CODEC: dict[str, str] = {
    "mp3": "mp3",
    "aac": "aac",
    "opus": "ogg",
}
_FFMPEG_ENCODER_BY_CODEC: dict[str, str] = {
    "mp3": "libmp3lame",
    "aac": "aac",
    "opus": "libopus",
}
_TORCHAUDIO_ENCODING_BY_CODEC: dict[str, str | None] = {
    "mp3": None,
    "aac": "aac",
    "opus": "libopus",
}


def _normalize_codec(codec: str) -> CodecName:
    normalized = str(codec).strip().lower()
    if normalized not in {"mp3", "aac", "opus"}:
        raise ValueError("codec must be one of: mp3, aac, opus")
    return normalized  # type: ignore[return-value]


def _normalize_backend(backend: str) -> CodecBackend:
    normalized = str(backend).strip().lower()
    if normalized not in {"auto", "torchaudio", "ffmpeg"}:
        raise ValueError("codec backend must be one of: auto, torchaudio, ffmpeg")
    return normalized  # type: ignore[return-value]


def _trim_or_pad(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    """Return waveform with exactly ``samples`` samples."""
    if waveform.shape[-1] > samples:
        return waveform[..., :samples].contiguous()
    if waveform.shape[-1] == samples:
        return waveform.contiguous()
    pad = torch.zeros(
        (*waveform.shape[:-1], samples - waveform.shape[-1]),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return torch.cat([waveform, pad], dim=-1).contiguous()


def _best_alignment_lag(
    *,
    reference: torch.Tensor,
    decoded: torch.Tensor,
    max_lag: int,
) -> int:
    """Return sample lag where positive means decoded is delayed vs reference."""
    if max_lag <= 0:
        return 0
    samples = min(int(reference.numel()), int(decoded.numel()))
    if samples <= 1:
        return 0

    ref = reference[:samples].float()
    dec = decoded[:samples].float()
    ref = ref - ref.mean()
    dec = dec - dec.mean()
    ref_energy = float(ref.square().mean().item())
    dec_energy = float(dec.square().mean().item())
    if ref_energy <= 1e-12 or dec_energy <= 1e-12:
        return 0

    n_corr = int(ref.numel() + dec.numel() - 1)
    n_fft = 1 << int(n_corr - 1).bit_length()
    corr = torch.fft.irfft(
        torch.fft.rfft(dec, n_fft) * torch.conj(torch.fft.rfft(ref, n_fft)),
        n_fft,
    )
    bounded = min(int(max_lag), samples - 1)
    values = []
    for lag in range(-bounded, bounded + 1):
        index = lag if lag >= 0 else n_fft + lag
        values.append(corr[index])
    return int(torch.stack(values).argmax().item()) - bounded


def _align_decoded_to_reference(
    *,
    reference: torch.Tensor,
    decoded: torch.Tensor,
    max_lag: int,
) -> torch.Tensor:
    """Shift decoded waveform to match reference timing, then trim/pad length."""
    target_samples = int(reference.shape[-1])
    if decoded.numel() == 0:
        return torch.zeros_like(reference)

    ref_mono = reference.float().mean(dim=0)
    dec_mono = decoded.float().mean(dim=0)
    lag = _best_alignment_lag(
        reference=ref_mono,
        decoded=dec_mono,
        max_lag=max_lag,
    )
    if lag > 0:
        decoded = decoded[..., lag:]
    elif lag < 0:
        pad = torch.zeros(
            (decoded.shape[0], -lag),
            dtype=decoded.dtype,
            device=decoded.device,
        )
        decoded = torch.cat([pad, decoded], dim=-1)
    return _trim_or_pad(decoded, target_samples)


def _decode_interleaved_f32le(
    payload: bytes,
    *,
    channels: int,
) -> torch.Tensor:
    """Decode interleaved float32 PCM bytes to channel-first tensor."""
    if not payload:
        return torch.empty((channels, 0), dtype=torch.float32)
    flat = torch.frombuffer(bytearray(payload), dtype=torch.float32)
    frame_count = int(flat.numel()) // int(channels)
    if frame_count <= 0:
        return torch.empty((channels, 0), dtype=torch.float32)
    flat = flat[: frame_count * int(channels)]
    return flat.reshape(frame_count, int(channels)).t().contiguous()


def _roundtrip_torchaudio(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    codec: CodecName,
    bitrate_kbps: int,
) -> torch.Tensor:
    """Encode/decode through torchaudio using an in-memory buffer."""
    if torchaudio is None:
        raise RuntimeError("torchaudio is not available")

    encoded = BytesIO()
    save_kwargs: dict[str, object] = {
        "uri": encoded,
        "src": waveform.detach().cpu().float(),
        "sample_rate": int(sample_rate),
        "format": _ENCODE_FORMAT_BY_CODEC[codec],
        "compression": int(bitrate_kbps) * 1000,
    }
    encoding = _TORCHAUDIO_ENCODING_BY_CODEC[codec]
    if encoding is not None:
        save_kwargs["encoding"] = encoding
    try:
        torchaudio.save(**save_kwargs)
    except TypeError:
        save_kwargs.pop("compression", None)
        torchaudio.save(**save_kwargs)

    encoded.seek(0)
    decoded, decoded_sample_rate = torchaudio.load(
        encoded,
        format=_DECODE_FORMAT_BY_CODEC[codec],
    )
    if int(decoded_sample_rate) != int(sample_rate):
        decoded = torchaudio.functional.resample(
            waveform=decoded.float(),
            orig_freq=int(decoded_sample_rate),
            new_freq=int(sample_rate),
        )
    return cast(torch.Tensor, decoded.float())


def _roundtrip_ffmpeg(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    codec: CodecName,
    bitrate_kbps: int,
    ffmpeg_path: str,
    timeout_seconds: float,
) -> torch.Tensor:
    """Encode/decode through ffmpeg pipes without temporary files."""
    channels = int(waveform.shape[0])
    pcm = (
        waveform.detach()
        .cpu()
        .float()
        .clamp(-1.0, 1.0)
        .t()
        .contiguous()
        .numpy()
        .astype("float32", copy=False)
        .tobytes()
    )
    encode_format = _ENCODE_FORMAT_BY_CODEC[codec]
    decode_format = _DECODE_FORMAT_BY_CODEC[codec]
    encoder = _FFMPEG_ENCODER_BY_CODEC[codec]
    encode_cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-f",
        "f32le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-c:a",
        encoder,
        "-b:a",
        f"{int(bitrate_kbps)}k",
    ]
    if codec == "opus":
        encode_cmd += ["-vbr", "on"]
    encode_cmd += ["-f", encode_format, "pipe:1"]

    encoded = subprocess.run(
        encode_cmd,
        input=pcm,
        capture_output=True,
        timeout=float(timeout_seconds),
        check=False,
    )
    if encoded.returncode != 0:
        message = encoded.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg encode failed for {codec}: {message}")

    decode_cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-f",
        decode_format,
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        str(channels),
        "pipe:1",
    ]
    decoded = subprocess.run(
        decode_cmd,
        input=encoded.stdout,
        capture_output=True,
        timeout=float(timeout_seconds),
        check=False,
    )
    if decoded.returncode != 0:
        message = decoded.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed for {codec}: {message}")
    return _decode_interleaved_f32le(decoded.stdout, channels=channels)


def codec_roundtrip(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    codec: str,
    bitrate_kbps: int,
    backend: str = "auto",
    ffmpeg_path: str = "ffmpeg",
    align: bool = True,
    align_max_lag: int = 8192,
    timeout_seconds: float = 20.0,
) -> torch.Tensor:
    """
    Return a source waveform after lossy encode/decode.

    The returned tensor is channel-first, aligned back to the input waveform, and
    trimmed/padded to the original sample count. This keeps paired training from
    seeing codec encoder delay as a source/target timing error.
    """
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [C,S], got {tuple(waveform.shape)}")
    normalized_codec = _normalize_codec(codec)
    normalized_backend = _normalize_backend(backend)
    input_waveform = waveform.detach().cpu().float().contiguous()

    errors: list[Exception] = []
    decoded: torch.Tensor | None = None
    if normalized_backend in {"auto", "torchaudio"}:
        try:
            decoded = _roundtrip_torchaudio(
                input_waveform,
                sample_rate=int(sample_rate),
                codec=normalized_codec,
                bitrate_kbps=int(bitrate_kbps),
            )
        except Exception as error:
            errors.append(error)
            if normalized_backend == "torchaudio":
                raise

    if decoded is None:
        try:
            decoded = _roundtrip_ffmpeg(
                input_waveform,
                sample_rate=int(sample_rate),
                codec=normalized_codec,
                bitrate_kbps=int(bitrate_kbps),
                ffmpeg_path=ffmpeg_path,
                timeout_seconds=float(timeout_seconds),
            )
        except Exception as error:
            errors.append(error)
            raise RuntimeError(
                "codec augmentation failed with "
                f"backend={normalized_backend!r}, codec={normalized_codec!r}"
            ) from error

    if align:
        decoded = _align_decoded_to_reference(
            reference=input_waveform,
            decoded=decoded.float(),
            max_lag=int(align_max_lag),
        )
    else:
        decoded = _trim_or_pad(decoded.float(), int(input_waveform.shape[-1]))
    return decoded.to(device=waveform.device, dtype=waveform.dtype)
