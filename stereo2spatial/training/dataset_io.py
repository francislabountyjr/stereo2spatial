"""I/O and tensor-shaping helpers used by the latent-song dataset."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch

from .dataset_types import SongRecord

TARGET_LATENT_FILENAME = "target_latent.pt"
SOURCE_STEREO_LATENT_FILENAME = "source_stereo_latent.pt"
SOURCE_MONO_LATENT_FILENAME = "source_mono_latent.pt"
SOURCE_DOWNMIX_LATENT_FILENAME = "source_downmix_latent.pt"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"
METADATA_FILENAME = "metadata.json"

_TORCH_LOAD_PARAM_NAMES: set[str]
try:
    _TORCH_LOAD_PARAM_NAMES = set(inspect.signature(torch.load).parameters)
except (TypeError, ValueError):
    _TORCH_LOAD_PARAM_NAMES = set()
_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in _TORCH_LOAD_PARAM_NAMES
_TORCH_LOAD_SUPPORTS_MMAP = "mmap" in _TORCH_LOAD_PARAM_NAMES


def _to_cdt(latent: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize latent tensor shape to ``[C, D, T]`` contiguous layout."""
    if latent.dim() == 2:
        return latent.unsqueeze(0).contiguous()
    if latent.dim() == 3:
        return latent.contiguous()
    raise ValueError(
        f"{name} must have shape [D,T] or [C,D,T], got {tuple(latent.shape)}"
    )


def _slice_with_right_pad(
    latent_cdt: torch.Tensor,
    start_frame: int,
    num_valid_frames: int,
    window_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice ``latent_cdt`` and right-pad to a fixed window with a validity mask."""
    if num_valid_frames <= 0:
        raise ValueError(f"num_valid_frames must be > 0, got {num_valid_frames}")
    if num_valid_frames > window_frames:
        raise ValueError(
            f"num_valid_frames={num_valid_frames} cannot exceed window_frames={window_frames}"
        )

    total_frames = latent_cdt.shape[-1]
    end_frame = min(total_frames, start_frame + num_valid_frames)
    clipped = latent_cdt[..., start_frame:end_frame]
    valid = int(clipped.shape[-1])
    if valid == window_frames:
        mask = torch.ones(window_frames, dtype=torch.bool)
        return clipped, mask

    if valid <= 0:
        raise RuntimeError(
            f"Invalid slice window: start={start_frame}, window={window_frames}, total={total_frames}"
        )

    padded = torch.zeros(
        (*latent_cdt.shape[:-1], window_frames),
        dtype=latent_cdt.dtype,
        device=latent_cdt.device,
    )
    padded[..., :valid] = clipped
    mask = torch.zeros(window_frames, dtype=torch.bool, device=latent_cdt.device)
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


def _try_read_metadata(sample_dir: Path) -> dict[str, int | None]:
    """Read optional sample metadata and return normalized numeric fields."""
    metadata_path = sample_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {"sample_rate": None, "input_samples": None}
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"sample_rate": None, "input_samples": None}

    sample_rate = metadata.get("sample_rate")
    input_samples = metadata.get("input_samples")
    return {
        "sample_rate": int(sample_rate) if isinstance(sample_rate, int) else None,
        "input_samples": int(input_samples) if isinstance(input_samples, int) else None,
    }


def _load_manifest_records(
    *,
    dataset_root: Path,
    manifest_path: Path,
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
            target_shape = payload.get("target_latent_shape")
            if not isinstance(target_shape, list) or len(target_shape) != 3:
                raise ValueError(
                    f"manifest line {line_idx}: invalid target_latent_shape={target_shape!r}"
                )
            target_channels = int(target_shape[0])
            target_frames = int(target_shape[2])
            metadata = _try_read_metadata(sample_dir)
            songs.append(
                SongRecord(
                    stream_hash=str(payload.get("stream_hash", sample_dir.name)),
                    sample_dir=sample_dir,
                    target_frames=target_frames,
                    target_channels=target_channels,
                    sample_rate=metadata.get("sample_rate"),
                    input_samples=metadata.get("input_samples"),
                )
            )
    return songs


def _load_latents_from_sample(sample_dir: Path) -> dict[str, torch.Tensor]:
    """Load and normalize latent tensors from one sample directory."""
    bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
    if bundle_path.exists():
        payload = _torch_load_cpu(bundle_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid bundle payload type: {type(payload)}")
        latents = {
            "target_latent": payload["target_latent"],
            "source_stereo_latent": payload["source_stereo_latent"],
            "source_mono_latent": payload["source_mono_latent"],
            "source_downmix_latent": payload["source_downmix_latent"],
        }
    else:
        split_paths = {
            "target_latent": sample_dir / TARGET_LATENT_FILENAME,
            "source_stereo_latent": sample_dir / SOURCE_STEREO_LATENT_FILENAME,
            "source_mono_latent": sample_dir / SOURCE_MONO_LATENT_FILENAME,
            "source_downmix_latent": sample_dir / SOURCE_DOWNMIX_LATENT_FILENAME,
        }
        missing = [str(path) for path in split_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing latent artifacts and no bundle found:\n  - "
                + "\n  - ".join(missing)
            )
        latents = {key: _torch_load_cpu(path) for key, path in split_paths.items()}

    normalized: dict[str, torch.Tensor] = {}
    for key, value in latents.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")
        tensor = value if value.dtype == torch.float32 else value.float()
        normalized[key] = _to_cdt(tensor, key)
    return normalized
