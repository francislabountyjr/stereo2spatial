"""EAR-VAE codec integration used by stereo2spatial pipelines."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch

from .runtime import _empty_cache, get_gpu_memory_gb

DeviceLike = str | torch.device
LatentsLike = torch.Tensor | Sequence[torch.Tensor]
MonoReductionMode = Literal["mean", "left", "right"]

DEFAULT_ENCODE_CHUNK_SEC_LOW_VRAM = 15.0
DEFAULT_ENCODE_CHUNK_SEC_HIGH_VRAM = 30.0
DEFAULT_ENCODE_OVERLAP_SEC = 2.0

DEFAULT_DECODE_CHUNK_FRAMES = 2048
DEFAULT_DECODE_OVERLAP_FRAMES = 256


def _import_ear_vae_class() -> type[torch.nn.Module]:
    """Import EAR-VAE class with actionable dependency error messaging."""
    try:
        from stereo2spatial.vendor.ear_vae.model.ear_vae import EAR_VAE
    except ModuleNotFoundError as error:
        if error.name in {"alias_free_torch", "dac"}:
            raise ModuleNotFoundError(
                "Missing EAR_VAE dependency. Install with:\n"
                "  pip install descript-audio-codec alias-free-torch"
            ) from error
        raise
    return EAR_VAE


def _as_device(device: DeviceLike) -> torch.device:
    """Coerce string/device-like input into a concrete ``torch.device``."""
    return device if isinstance(device, torch.device) else torch.device(device)


def get_default_device() -> torch.device:
    """Pick the best available accelerator device with CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device(vae: torch.nn.Module, device: DeviceLike | None) -> torch.device:
    """Resolve runtime device from explicit argument or model parameters."""
    if device is not None:
        return _as_device(device)
    try:
        return next(vae.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _step_iterator(count: int, desc: str, show_progress: bool) -> Iterable[int]:
    """Return a plain range or tqdm-wrapped iterator for chunk loops."""
    if show_progress:
        try:
            from tqdm.auto import tqdm

            return cast(Iterable[int], tqdm(range(count), desc=desc))
        except Exception:
            pass
    return range(count)


def _resolve_config_path(
    checkpoint_path: Path, config_path: str | Path | None
) -> Path:
    """Resolve EAR-VAE config path from explicit path or checkpoint-adjacent defaults."""
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"VAE config not found: {path}")
        return path

    candidates = [
        checkpoint_path.parent.parent / "config" / "ear_vae_v2.json",
        checkpoint_path.parent.parent / "config" / "model_config.json",
        checkpoint_path.with_suffix(".json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not infer VAE config path. Provide config_path explicitly."
    )


def _extract_state_dict(raw_state: Any) -> dict[str, torch.Tensor]:
    """Extract a flat state dict from common checkpoint payload structures."""
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        maybe_nested = raw_state["state_dict"]
        if isinstance(maybe_nested, dict):
            return maybe_nested
    if isinstance(raw_state, dict):
        return raw_state
    raise TypeError(
        f"Unsupported checkpoint format: expected dict-like state_dict, got {type(raw_state)}"
    )


def _normalize_audio_input(
    audio: torch.Tensor, duplicate_mono_to_stereo: bool
) -> tuple[torch.Tensor, bool]:
    """Normalize audio input shape to ``[B, C, S]`` and validate channel count."""
    if audio.dim() not in {2, 3}:
        raise ValueError(
            f"audio must have shape [C, S] or [B, C, S], got {tuple(audio.shape)}"
        )

    input_was_2d = audio.dim() == 2
    if input_was_2d:
        audio = audio.unsqueeze(0)

    channels = audio.shape[1]
    if channels == 2:
        return audio, input_was_2d
    if channels == 1 and duplicate_mono_to_stereo:
        return audio.repeat(1, 2, 1), input_was_2d
    raise ValueError(
        "EAR_VAE expects stereo input [B, 2, S]. "
        f"Got channels={channels}. If mono, set duplicate_mono_to_stereo=True."
    )


def _normalize_latent_input(pred_latents: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize latent input shape to contiguous ``[B, D, T]``."""
    if pred_latents.dim() not in {2, 3}:
        raise ValueError(
            f"pred_latents must have shape [D, T] or [B, D, T], got {tuple(pred_latents.shape)}"
        )

    input_was_2d = pred_latents.dim() == 2
    if input_was_2d:
        pred_latents = pred_latents.unsqueeze(0)

    return pred_latents.contiguous(), input_was_2d


def default_encode_chunk_size_samples(sample_rate: int = 48000) -> int:
    """Choose a conservative encode chunk size based on available accelerator memory."""
    return int(
        round(
            (
                DEFAULT_ENCODE_CHUNK_SEC_LOW_VRAM
                if get_gpu_memory_gb() <= 8.0
                else DEFAULT_ENCODE_CHUNK_SEC_HIGH_VRAM
            )
            * sample_rate
        )
    )


def default_encode_overlap_samples(sample_rate: int = 48000) -> int:
    """Default encode overlap in samples for chunked encode mode."""
    return int(round(DEFAULT_ENCODE_OVERLAP_SEC * sample_rate))


def load_vae(
    vae_checkpoint_path: str | Path,
    device: DeviceLike | None = None,
    config_path: str | Path | None = None,
    torch_dtype: torch.dtype = torch.float32,
    strict: bool = True,
    disable_transformer_if_missing: bool = True,
) -> torch.nn.Module:
    """
    Load EAR_VAE from checkpoint and config.

    If checkpoint lacks transformer weights but config enables transformer,
    transformer is disabled automatically when disable_transformer_if_missing=True.
    """
    checkpoint_path = Path(vae_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

    resolved_config_path = _resolve_config_path(checkpoint_path, config_path)
    model_device = _as_device(device) if device is not None else get_default_device()

    with open(resolved_config_path, encoding="utf-8") as handle:
        model_config = json.load(handle)

    raw_state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(raw_state)

    config_for_load = copy.deepcopy(model_config)
    checkpoint_has_transformer = any(
        key.startswith("transformers.") for key in state_dict
    )
    config_has_transformer = config_for_load.get("transformer") is not None
    if (
        disable_transformer_if_missing
        and config_has_transformer
        and not checkpoint_has_transformer
    ):
        config_for_load["transformer"] = None

    EAR_VAE = _import_ear_vae_class()
    vae = EAR_VAE(model_config=config_for_load)
    vae.load_state_dict(state_dict, strict=strict)
    vae = vae.to(device=model_device, dtype=torch_dtype)
    vae.eval()

    # Keep resolved metadata on the module for downstream logging/debug.
    setattr(vae, "loaded_checkpoint_path", str(checkpoint_path))
    setattr(vae, "loaded_config_path", str(resolved_config_path))
    setattr(vae, "loaded_model_config", config_for_load)
    return vae


def _encode_full(
    vae: torch.nn.Module, audio_bcs: torch.Tensor, use_sample: bool
) -> torch.Tensor:
    """Run one full-window EAR-VAE encode call under ``torch.no_grad``."""
    encode_fn = getattr(vae, "encode", None)
    if not callable(encode_fn):
        raise TypeError("EAR_VAE module does not expose a callable encode method")
    with torch.no_grad():
        return cast(torch.Tensor, encode_fn(audio_bcs, use_sample=use_sample))


def _decode_full(vae: torch.nn.Module, latents_bdt: torch.Tensor) -> torch.Tensor:
    """Run one full-window EAR-VAE decode call under ``torch.no_grad``."""
    decode_fn = getattr(vae, "decode", None)
    if not callable(decode_fn):
        raise TypeError("EAR_VAE module does not expose a callable decode method")
    with torch.no_grad():
        return cast(torch.Tensor, decode_fn(latents_bdt))


def chunked_encode(
    vae: torch.nn.Module,
    audio: torch.Tensor,
    sample_rate: int = 48000,
    chunk_size_samples: int | None = None,
    overlap_samples: int | None = None,
    use_sample: bool = False,
    duplicate_mono_to_stereo: bool = True,
    offload_latent_to_cpu: bool = True,
    show_progress: bool = False,
    device: DeviceLike | None = None,
) -> torch.Tensor:
    """
    Chunked overlap-discard encode.

    Input shape:
    - [C, S] or [B, C, S]

    Returns model-format latents:
    - [D, T] or [B, D, T]
    """
    model_device = _resolve_device(vae, device)
    audio_bcs, input_was_2d = _normalize_audio_input(audio, duplicate_mono_to_stereo)
    audio_bcs = audio_bcs.to(device=model_device, dtype=torch.float32)

    if chunk_size_samples is None:
        chunk_size_samples = default_encode_chunk_size_samples(sample_rate)
    if overlap_samples is None:
        overlap_samples = default_encode_overlap_samples(sample_rate)

    total_samples = audio_bcs.shape[-1]
    if total_samples <= chunk_size_samples:
        latents = _encode_full(vae, audio_bcs, use_sample=use_sample)
        if offload_latent_to_cpu:
            latents = latents.cpu()
        return latents.squeeze(0) if input_was_2d else latents

    stride = chunk_size_samples - 2 * overlap_samples
    if stride <= 0:
        raise ValueError(
            f"chunk_size_samples={chunk_size_samples} must be > 2*overlap_samples={overlap_samples}"
        )

    num_steps = math.ceil(total_samples / stride)
    latent_chunks: list[torch.Tensor] = []
    downsample_factor: float | None = None

    for step in _step_iterator(num_steps, "Encoding chunks", show_progress):
        core_start = step * stride
        core_end = min(core_start + stride, total_samples)
        win_start = max(0, core_start - overlap_samples)
        win_end = min(total_samples, core_end + overlap_samples)

        audio_chunk = audio_bcs[:, :, win_start:win_end]
        latent_chunk = _encode_full(vae, audio_chunk, use_sample=use_sample)

        if downsample_factor is None:
            downsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

        added_start = core_start - win_start
        trim_start = int(round(added_start / downsample_factor))
        added_end = win_end - core_end
        trim_end = int(round(added_end / downsample_factor))

        latent_len = latent_chunk.shape[-1]
        end_idx = latent_len - trim_end if trim_end > 0 else latent_len
        latent_core = latent_chunk[:, :, trim_start:end_idx]

        if offload_latent_to_cpu:
            latent_core = latent_core.cpu()
        latent_chunks.append(latent_core)

        del audio_chunk, latent_chunk

    latents = torch.cat(latent_chunks, dim=-1)
    _empty_cache(model_device)
    return latents.squeeze(0) if input_was_2d else latents


def chunked_decode(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    chunk_size_frames: int = DEFAULT_DECODE_CHUNK_FRAMES,
    overlap_frames: int = DEFAULT_DECODE_OVERLAP_FRAMES,
    offload_wav_to_cpu: bool = True,
    show_progress: bool = False,
    device: DeviceLike | None = None,
) -> torch.Tensor:
    """
    Chunked overlap-discard decode.

    Input model-format latent shape:
    - [D, T] or [B, D, T]

    Returns:
    - [2, S] or [B, 2, S]
    """
    model_device = _resolve_device(vae, device)

    if latents.dim() not in {2, 3}:
        raise ValueError(
            f"latents must have shape [D, T] or [B, D, T], got {tuple(latents.shape)}"
        )

    input_was_2d = latents.dim() == 2
    if input_was_2d:
        latents = latents.unsqueeze(0)

    latents = latents.to(device=model_device, dtype=torch.float32)
    total_frames = latents.shape[-1]

    if total_frames <= chunk_size_frames:
        audio = _decode_full(vae, latents)
        if offload_wav_to_cpu:
            audio = audio.cpu()
        return audio.squeeze(0) if input_was_2d else audio

    stride = chunk_size_frames - 2 * overlap_frames
    if stride <= 0:
        raise ValueError(
            f"chunk_size_frames={chunk_size_frames} must be > 2*overlap_frames={overlap_frames}"
        )

    num_steps = math.ceil(total_frames / stride)
    audio_chunks: list[torch.Tensor] = []
    upsample_factor: float | None = None

    for step in _step_iterator(num_steps, "Decoding chunks", show_progress):
        core_start = step * stride
        core_end = min(core_start + stride, total_frames)
        win_start = max(0, core_start - overlap_frames)
        win_end = min(total_frames, core_end + overlap_frames)

        latent_chunk = latents[:, :, win_start:win_end]
        audio_chunk = _decode_full(vae, latent_chunk)

        if upsample_factor is None:
            upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

        added_start = core_start - win_start
        trim_start = int(round(added_start * upsample_factor))
        added_end = win_end - core_end
        trim_end = int(round(added_end * upsample_factor))

        audio_len = audio_chunk.shape[-1]
        end_idx = audio_len - trim_end if trim_end > 0 else audio_len
        audio_core = audio_chunk[:, :, trim_start:end_idx]

        if offload_wav_to_cpu:
            audio_core = audio_core.cpu()
        audio_chunks.append(audio_core)

        del latent_chunk, audio_chunk

    audio = torch.cat(audio_chunks, dim=-1)
    _empty_cache(model_device)
    return audio.squeeze(0) if input_was_2d else audio


def vae_encode(
    vae: torch.nn.Module,
    audio: torch.Tensor,
    dtype: torch.dtype | None = None,
    sample_rate: int = 48000,
    use_sample: bool = False,
    use_chunked_encode: bool = True,
    chunk_size_samples: int | None = None,
    overlap_samples: int | None = None,
    duplicate_mono_to_stereo: bool = True,
    offload_latent_to_cpu: bool = True,
    show_progress: bool = False,
    device: DeviceLike | None = None,
) -> torch.Tensor:
    """
    Encode audio to latents.

    Input shape:
    - [C, S] or [B, C, S]

    Output shape:
    - [D, T] or [B, D, T]
    """
    if use_chunked_encode:
        latents_bdt = chunked_encode(
            vae=vae,
            audio=audio,
            sample_rate=sample_rate,
            chunk_size_samples=chunk_size_samples,
            overlap_samples=overlap_samples,
            use_sample=use_sample,
            duplicate_mono_to_stereo=duplicate_mono_to_stereo,
            offload_latent_to_cpu=offload_latent_to_cpu,
            show_progress=show_progress,
            device=device,
        )
    else:
        model_device = _resolve_device(vae, device)
        audio_bcs, input_was_2d = _normalize_audio_input(
            audio, duplicate_mono_to_stereo
        )
        audio_bcs = audio_bcs.to(device=model_device, dtype=torch.float32)
        latents_bdt = _encode_full(vae, audio_bcs, use_sample=use_sample)
        if offload_latent_to_cpu:
            latents_bdt = latents_bdt.cpu()
        latents_bdt = latents_bdt.squeeze(0) if input_was_2d else latents_bdt

    if dtype is not None:
        latents_bdt = latents_bdt.to(dtype)
    return latents_bdt


def vae_decode(
    vae: torch.nn.Module,
    pred_latents: torch.Tensor,
    use_chunked_decode: bool = True,
    chunk_size_frames: int = DEFAULT_DECODE_CHUNK_FRAMES,
    overlap_frames: int = DEFAULT_DECODE_OVERLAP_FRAMES,
    offload_wav_to_cpu: bool = True,
    normalize_audio: bool = False,
    return_cpu_list: bool = True,
    show_progress: bool = False,
    device: DeviceLike | None = None,
) -> list[torch.Tensor] | torch.Tensor:
    """
    Decode latents to stereo audio.

    Input shape:
    - [D, T] or [B, D, T]

    Output:
    - list of [2, S] tensors when return_cpu_list=True
    - [2, S] or [B, 2, S] tensor otherwise
    """
    latents_bdt, input_was_2d = _normalize_latent_input(pred_latents)

    if use_chunked_decode:
        pred_wavs = chunked_decode(
            vae=vae,
            latents=latents_bdt,
            chunk_size_frames=chunk_size_frames,
            overlap_frames=overlap_frames,
            offload_wav_to_cpu=offload_wav_to_cpu,
            show_progress=show_progress,
            device=device,
        )
    else:
        model_device = _resolve_device(vae, device)
        latents_bdt = latents_bdt.to(device=model_device, dtype=torch.float32)
        pred_wavs = _decode_full(vae, latents_bdt)
        if offload_wav_to_cpu:
            pred_wavs = pred_wavs.cpu()

    if pred_wavs.dtype != torch.float32:
        pred_wavs = pred_wavs.float()

    if normalize_audio:
        std = torch.std(pred_wavs, dim=[1, 2], keepdim=True) * 5.0
        std = torch.clamp(std, min=1.0)
        pred_wavs = pred_wavs / std

    if not return_cpu_list:
        return pred_wavs.squeeze(0) if input_was_2d else pred_wavs

    pred_wavs_cpu = pred_wavs.detach().cpu()
    if input_was_2d:
        return [pred_wavs_cpu.squeeze(0).float()]
    return [pred_wavs_cpu[i].float() for i in range(pred_wavs_cpu.shape[0])]


# Convenience aliases.
def tiled_encode(
    vae: torch.nn.Module,
    audio: torch.Tensor,
    device: DeviceLike | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    offload_latent_to_cpu: bool = True,
    show_progress: bool = False,
) -> torch.Tensor:
    """Backward-compatible alias for `chunked_encode`."""
    return chunked_encode(
        vae=vae,
        audio=audio,
        chunk_size_samples=chunk_size,
        overlap_samples=overlap,
        offload_latent_to_cpu=offload_latent_to_cpu,
        show_progress=show_progress,
        device=device,
    )


def tiled_decode(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    device: DeviceLike | None = None,
    chunk_size: int = DEFAULT_DECODE_CHUNK_FRAMES,
    overlap: int = DEFAULT_DECODE_OVERLAP_FRAMES,
    offload_wav_to_cpu: bool = True,
    show_progress: bool = False,
) -> torch.Tensor:
    """Backward-compatible alias for `chunked_decode`."""
    return chunked_decode(
        vae=vae,
        latents=latents,
        chunk_size_frames=chunk_size,
        overlap_frames=overlap,
        offload_wav_to_cpu=offload_wav_to_cpu,
        show_progress=show_progress,
        device=device,
    )


def _reduce_stereo_to_mono(
    stereo_audio: torch.Tensor, mode: MonoReductionMode
) -> torch.Tensor:
    """Reduce decoded stereo audio to mono using the selected reduction mode."""
    if stereo_audio.dim() != 2 or stereo_audio.shape[0] != 2:
        raise ValueError(
            f"Expected stereo [2, S] tensor for reduction, got {tuple(stereo_audio.shape)}"
        )
    if mode == "mean":
        return stereo_audio.mean(dim=0, keepdim=True)
    if mode == "left":
        return stereo_audio[0:1, :]
    if mode == "right":
        return stereo_audio[1:2, :]
    raise ValueError("mode must be one of: 'mean', 'left', 'right'")


def encode_channels_independent(
    vae: torch.nn.Module,
    audio: torch.Tensor,
    sample_rate: int = 48000,
    use_sample: bool = False,
    use_chunked_encode: bool = True,
    chunk_size_samples: int | None = None,
    overlap_samples: int | None = None,
    offload_latent_to_cpu: bool = True,
    show_progress: bool = False,
    device: DeviceLike | None = None,
) -> torch.Tensor:
    """
    Encode each input channel independently by duplicating mono channel to stereo.

    Input:
    - audio: [C, S]

    Returns:
    - stacked latents: [C, D, T]
    """
    if audio.dim() != 2:
        raise ValueError(
            f"audio must have shape [C, S] for per-channel encode, got {tuple(audio.shape)}"
        )

    channel_latents: list[torch.Tensor] = []
    for channel_idx in range(audio.shape[0]):
        mono = audio[channel_idx : channel_idx + 1, :]
        latents = vae_encode(
            vae=vae,
            audio=mono,
            sample_rate=sample_rate,
            use_sample=use_sample,
            use_chunked_encode=use_chunked_encode,
            chunk_size_samples=chunk_size_samples,
            overlap_samples=overlap_samples,
            duplicate_mono_to_stereo=True,
            offload_latent_to_cpu=offload_latent_to_cpu,
            show_progress=show_progress,
            device=device,
        )
        channel_latents.append(latents)

    shape_set = {tuple(lat.shape) for lat in channel_latents}
    if len(shape_set) != 1:
        raise RuntimeError(f"Per-channel latent shapes do not match: {shape_set}")

    return torch.stack(channel_latents, dim=0)


def decode_channels_independent(
    vae: torch.nn.Module,
    channel_latents: LatentsLike,
    use_chunked_decode: bool = True,
    chunk_size_frames: int = DEFAULT_DECODE_CHUNK_FRAMES,
    overlap_frames: int = DEFAULT_DECODE_OVERLAP_FRAMES,
    offload_wav_to_cpu: bool = True,
    reduction: MonoReductionMode = "mean",
    show_progress: bool = False,
    device: DeviceLike | None = None,
) -> torch.Tensor:
    """
    Decode per-channel latents independently, then reduce decoded stereo to mono per channel.

    Input:
    - channel_latents: [C, D, T] tensor or list of [D, T] tensors

    Returns:
    - reconstructed channels: [C, S]
    """
    if isinstance(channel_latents, torch.Tensor):
        if channel_latents.dim() != 3:
            raise ValueError(
                "channel_latents tensor must have shape [C, D, T], "
                f"got {tuple(channel_latents.shape)}"
            )
        latent_list = [channel_latents[i] for i in range(channel_latents.shape[0])]
    else:
        latent_list = list(channel_latents)
        if not latent_list:
            raise ValueError("channel_latents list is empty")

    reconstructed: list[torch.Tensor] = []
    for latents in latent_list:
        decoded_list = vae_decode(
            vae=vae,
            pred_latents=latents,
            use_chunked_decode=use_chunked_decode,
            chunk_size_frames=chunk_size_frames,
            overlap_frames=overlap_frames,
            offload_wav_to_cpu=offload_wav_to_cpu,
            normalize_audio=False,
            return_cpu_list=True,
            show_progress=show_progress,
            device=device,
        )
        stereo = decoded_list[0]  # [2, S]
        mono = _reduce_stereo_to_mono(stereo, mode=reduction)
        reconstructed.append(mono)

    lengths = [item.shape[-1] for item in reconstructed]
    min_length = min(lengths)
    reconstructed = [item[:, :min_length] for item in reconstructed]
    return torch.cat(reconstructed, dim=0)  # [C, S]
