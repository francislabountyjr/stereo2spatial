"""I/O and tensor-shaping helpers used by the signal-song dataset."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch

from stereo2spatial.common.amplitude_lift import (
    apply_amplitude_lift,
    compute_shared_rms_gain,
)
from stereo2spatial.common.mix_style import mix_style_dict_to_vector

from .dataset_types import SongRecord

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - dependency is required by inference/preprocess
    sf = None

TARGET_SIGNAL_FILENAME = "target_signal.pt"
SOURCE_STEREO_SIGNAL_FILENAME = "source_stereo_signal.pt"
SOURCE_MONO_SIGNAL_FILENAME = "source_mono_signal.pt"
SOURCE_DOWNMIX_SIGNAL_FILENAME = "source_downmix_signal.pt"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"
METADATA_FILENAME = "metadata.json"
TARGET_SIGNAL_FLAC_FILENAME = "target_signal.flac"
SOURCE_STEREO_SIGNAL_FLAC_FILENAME = "source_stereo_signal.flac"
SOURCE_MONO_SIGNAL_FLAC_FILENAME = "source_mono_signal.flac"
SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME = "source_downmix_signal.flac"

_TORCH_LOAD_PARAM_NAMES: set[str]
try:
    _TORCH_LOAD_PARAM_NAMES = set(inspect.signature(torch.load).parameters)
except (TypeError, ValueError):
    _TORCH_LOAD_PARAM_NAMES = set()
_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in _TORCH_LOAD_PARAM_NAMES
_TORCH_LOAD_SUPPORTS_MMAP = "mmap" in _TORCH_LOAD_PARAM_NAMES


def _patch_audio(signal: torch.Tensor, patch_size: int, name: str) -> torch.Tensor:
    """Convert continuous waveform `[C,S]` to patches `[C,P,T]`."""
    if signal.dim() != 2:
        raise ValueError(f"{name} must have shape [C,S], got {tuple(signal.shape)}")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    sample_count = int(signal.shape[-1])
    frame_count = max(1, (sample_count + patch_size - 1) // patch_size)
    padded_samples = frame_count * patch_size
    if padded_samples != sample_count:
        pad = torch.zeros(
            (signal.shape[0], padded_samples - sample_count),
            dtype=signal.dtype,
            device=signal.device,
        )
        signal = torch.cat([signal, pad], dim=-1)
    return signal.reshape(signal.shape[0], frame_count, patch_size).permute(0, 2, 1).contiguous()


def _to_cpt(signal: torch.Tensor, name: str, patch_size: int) -> torch.Tensor:
    """Normalize signal tensor shape to ``[C, P, T]`` contiguous layout."""
    if signal.dim() == 2:
        return _patch_audio(signal.contiguous(), patch_size=patch_size, name=name)
    if signal.dim() == 3:
        return signal.contiguous()
    raise ValueError(
        f"{name} must have shape [C,S] or [C,P,T], got {tuple(signal.shape)}"
    )


def _slice_with_right_pad(
    signal_cdt: torch.Tensor,
    start_frame: int,
    num_valid_frames: int,
    window_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice ``signal_cdt`` and right-pad to a fixed window with a validity mask."""
    if num_valid_frames <= 0:
        raise ValueError(f"num_valid_frames must be > 0, got {num_valid_frames}")
    if num_valid_frames > window_frames:
        raise ValueError(
            f"num_valid_frames={num_valid_frames} cannot exceed window_frames={window_frames}"
        )

    total_frames = signal_cdt.shape[-1]
    end_frame = min(total_frames, start_frame + num_valid_frames)
    clipped = signal_cdt[..., start_frame:end_frame]
    valid = int(clipped.shape[-1])
    if valid == window_frames:
        mask = torch.ones(window_frames, dtype=torch.bool)
        return clipped, mask

    if valid <= 0:
        raise RuntimeError(
            f"Invalid slice window: start={start_frame}, window={window_frames}, total={total_frames}"
        )

    padded = torch.zeros(
        (*signal_cdt.shape[:-1], window_frames),
        dtype=signal_cdt.dtype,
        device=signal_cdt.device,
    )
    padded[..., :valid] = clipped
    mask = torch.zeros(window_frames, dtype=torch.bool, device=signal_cdt.device)
    mask[:valid] = True
    return padded, mask


def _torch_load_cpu(path: Path) -> Any:
    """Load a torch payload on CPU with compatibility fallbacks."""
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if _TORCH_LOAD_SUPPORTS_MMAP:
        load_kwargs["mmap"] = True

    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except Exception:
            pass

    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        return torch.load(path, **load_kwargs)


def _try_read_metadata(sample_dir: Path) -> dict[str, Any]:
    """Read optional sample metadata and return normalized runtime fields."""
    metadata_path = sample_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {
            "sample_rate": None,
            "input_samples": None,
            "mix_style": None,
            "signal_rms": None,
        }
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"sample_rate": None, "input_samples": None, "mix_style": None}

    sample_rate = metadata.get("sample_rate")
    input_samples = metadata.get("input_samples")
    mix_style = metadata.get("mix_style")
    mix_style_vector = metadata.get("mix_style_vector")
    mix_style_names = metadata.get("mix_style_names")
    signal_rms = metadata.get("signal_rms")
    return {
        "sample_rate": int(sample_rate) if isinstance(sample_rate, int) else None,
        "input_samples": int(input_samples) if isinstance(input_samples, int) else None,
        "mix_style": mix_style if isinstance(mix_style, dict) else None,
        "mix_style_vector": mix_style_vector if isinstance(mix_style_vector, list) else None,
        "mix_style_names": mix_style_names if isinstance(mix_style_names, list) else None,
        "signal_rms": signal_rms if isinstance(signal_rms, dict) else None,
    }


def _coerce_mix_style(payload: dict[str, Any], metadata: dict[str, Any]) -> tuple[float, ...] | None:
    """Read the normalized style vector from manifest first, then metadata."""
    for source in (payload.get("mix_style_vector"), metadata.get("mix_style_vector")):
        if isinstance(source, list):
            return tuple(float(x) for x in source)
    for values, names in (
        (payload.get("mix_style"), payload.get("mix_style_names")),
        (metadata.get("mix_style"), metadata.get("mix_style_names")),
    ):
        if isinstance(values, list):
            return tuple(float(x) for x in values)
        if isinstance(values, dict):
            resolved_names = (
                [str(name) for name in names] if isinstance(names, list) else None
            )
            return tuple(mix_style_dict_to_vector(values, names=resolved_names))
    return None


def _coerce_signal_rms(
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, float] | None:
    """Read precomputed full-song RMS values from manifest or metadata."""
    for source in (payload.get("signal_rms"), metadata.get("signal_rms")):
        if not isinstance(source, dict):
            continue
        values: dict[str, float] = {}
        for key, value in source.items():
            try:
                resolved = float(value)
            except (TypeError, ValueError):
                continue
            if resolved > 0:
                values[str(key)] = resolved
        if values:
            return values
    return None


def _load_manifest_records(
    *,
    dataset_root: Path,
    manifest_path: Path,
    patch_size: int,
) -> list[SongRecord]:
    """Load manifest JSONL rows into typed song records."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    songs: list[SongRecord] = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_dir_raw = payload.get("sample_dir")
            if not sample_dir_raw:
                raise KeyError(f"manifest line {line_idx}: missing key 'sample_dir'")
            sample_dir = dataset_root / Path(sample_dir_raw)
            target_shape = payload.get("target_signal_shape")
            if not isinstance(target_shape, list) or len(target_shape) not in {2, 3}:
                raise ValueError(
                    f"manifest line {line_idx}: invalid target_signal_shape={target_shape!r}"
                )
            target_channels = int(target_shape[0])
            if len(target_shape) == 2:
                target_samples = int(target_shape[1])
                target_frames = max(1, (target_samples + int(patch_size) - 1) // int(patch_size))
            else:
                target_frames = int(target_shape[2])
            metadata = _try_read_metadata(sample_dir)
            mix_style = _coerce_mix_style(payload, metadata)
            signal_rms = _coerce_signal_rms(payload, metadata)
            songs.append(
                SongRecord(
                    stream_hash=str(payload.get("stream_hash", sample_dir.name)),
                    sample_dir=sample_dir,
                    target_frames=target_frames,
                    target_channels=target_channels,
                    sample_rate=metadata.get("sample_rate"),
                    input_samples=metadata.get("input_samples"),
                    mix_style=mix_style,
                    signal_rms=signal_rms,
                )
            )
    return songs


def _apply_amplitude_lift_to_signals(
    signals: dict[str, torch.Tensor],
    *,
    reference: str,
    target_rms: float,
    scale: float,
    clip_value: float | None,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Apply one shared RMS lift gain to all raw sample signals before patching."""
    normalized_reference = str(reference).strip().lower()
    reference_key = (
        "target_signal" if normalized_reference == "target" else "source_stereo_signal"
    )
    if reference_key not in signals:
        raise KeyError(f"Amplitude lifting reference signal missing: {reference_key}")
    gain = compute_shared_rms_gain(
        signals[reference_key].float(),
        target_rms=float(target_rms),
        eps=float(eps),
    )
    return {
        key: apply_amplitude_lift(
            value.float(),
            gain=gain,
            scale=float(scale),
            clip_value=clip_value,
        )
        for key, value in signals.items()
    }


def _load_signals_from_sample(
    sample_dir: Path,
    patch_size: int,
    *,
    amplitude_lift_enabled: bool = False,
    amplitude_lift_reference: str = "source",
    amplitude_lift_target_rms: float = 0.33,
    amplitude_lift_scale: float = 3.0,
    amplitude_lift_clip_value: float | None = 4.0,
    amplitude_lift_eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Load and normalize signal tensors from one sample directory."""
    signals = _load_raw_signals_from_sample(sample_dir)

    if amplitude_lift_enabled:
        signals = _apply_amplitude_lift_to_signals(
            signals,
            reference=amplitude_lift_reference,
            target_rms=amplitude_lift_target_rms,
            scale=amplitude_lift_scale,
            clip_value=amplitude_lift_clip_value,
            eps=amplitude_lift_eps,
        )

    normalized: dict[str, torch.Tensor] = {}
    for key, value in signals.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")
        tensor = value if value.dtype == torch.float32 else value.float()
        normalized[key] = _to_cpt(tensor, key, patch_size=patch_size)
    return normalized


def _load_raw_signals_from_sample(sample_dir: Path) -> dict[str, torch.Tensor]:
    """Load signal tensors without patching or amplitude lifting."""
    bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
    if bundle_path.exists():
        payload = _torch_load_cpu(bundle_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid bundle payload type: {type(payload)}")
        signals = {
            "target_signal": payload["target_signal"],
            "source_stereo_signal": payload["source_stereo_signal"],
        }
        if "source_mono_signal" in payload:
            signals["source_mono_signal"] = payload["source_mono_signal"]
        if "source_downmix_signal" in payload:
            signals["source_downmix_signal"] = payload["source_downmix_signal"]
    else:
        flac_paths = {
            "target_signal": sample_dir / TARGET_SIGNAL_FLAC_FILENAME,
            "source_stereo_signal": sample_dir / SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
        }
        if all(path.exists() for path in flac_paths.values()):
            signals = {
                key: _read_flac_signal(path)
                for key, path in flac_paths.items()
            }
            source_mono_flac = sample_dir / SOURCE_MONO_SIGNAL_FLAC_FILENAME
            if source_mono_flac.exists():
                signals["source_mono_signal"] = _read_flac_signal(source_mono_flac)
            source_downmix_flac = sample_dir / SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME
            if source_downmix_flac.exists():
                signals["source_downmix_signal"] = _read_flac_signal(source_downmix_flac)
        else:
            required_split_paths = {
                "target_signal": sample_dir / TARGET_SIGNAL_FILENAME,
                "source_stereo_signal": sample_dir / SOURCE_STEREO_SIGNAL_FILENAME,
            }
            missing = [
                str(path) for path in required_split_paths.values() if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "Missing signal artifacts and no bundle/flac artifacts found:\n  - "
                    + "\n  - ".join(missing)
                )
            split_paths = dict(required_split_paths)
            source_mono_path = sample_dir / SOURCE_MONO_SIGNAL_FILENAME
            if source_mono_path.exists():
                split_paths["source_mono_signal"] = source_mono_path
            source_downmix_path = sample_dir / SOURCE_DOWNMIX_SIGNAL_FILENAME
            if source_downmix_path.exists():
                split_paths["source_downmix_signal"] = source_downmix_path
            signals = {key: _torch_load_cpu(path) for key, path in split_paths.items()}

    for key, value in signals.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")
    return signals


def _read_flac_signal(path: Path) -> torch.Tensor:
    """Read a FLAC signal artifact into `[C,S]` float32 tensor layout."""
    if sf is None:
        raise RuntimeError("FLAC dataset artifacts require soundfile to be installed.")
    audio, _sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return torch.from_numpy(audio.T.copy()).contiguous()
