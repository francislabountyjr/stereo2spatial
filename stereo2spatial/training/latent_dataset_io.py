"""I/O helpers for the legacy precomputed-latent dataset backend."""

from __future__ import annotations

import csv
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

_LATENT_KEYS = (
    "target_latent",
    "source_stereo_latent",
    "source_mono_latent",
    "source_downmix_latent",
)
_SPLIT_FILENAMES = {
    "target_latent": TARGET_LATENT_FILENAME,
    "source_stereo_latent": SOURCE_STEREO_LATENT_FILENAME,
    "source_mono_latent": SOURCE_MONO_LATENT_FILENAME,
    "source_downmix_latent": SOURCE_DOWNMIX_LATENT_FILENAME,
}

try:
    _TORCH_LOAD_PARAM_NAMES = set(inspect.signature(torch.load).parameters)
except (TypeError, ValueError):
    _TORCH_LOAD_PARAM_NAMES = set()
_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in _TORCH_LOAD_PARAM_NAMES
_TORCH_LOAD_SUPPORTS_MMAP = "mmap" in _TORCH_LOAD_PARAM_NAMES


def _to_cdt(latent: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize one latent tensor to contiguous ``[C,D,T]`` float32 layout."""
    if latent.dim() == 2:
        latent = latent.unsqueeze(0)
    elif latent.dim() != 3:
        raise ValueError(
            f"{name} must have shape [D,T] or [C,D,T], got {tuple(latent.shape)}"
        )
    return latent.float().contiguous()


def _slice_with_right_pad(
    latent_cdt: torch.Tensor,
    start_frame: int,
    num_valid_frames: int,
    window_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a latent sequence and right-pad it to a fixed frame count."""
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if num_valid_frames <= 0:
        raise ValueError(f"num_valid_frames must be > 0, got {num_valid_frames}")
    if window_frames <= 0:
        raise ValueError(f"window_frames must be > 0, got {window_frames}")
    if num_valid_frames > window_frames:
        raise ValueError(
            f"num_valid_frames={num_valid_frames} cannot exceed "
            f"window_frames={window_frames}"
        )

    total_frames = int(latent_cdt.shape[-1])
    end_frame = min(total_frames, int(start_frame) + int(num_valid_frames))
    clipped = latent_cdt[..., int(start_frame) : end_frame]
    valid = int(clipped.shape[-1])
    if valid <= 0:
        raise RuntimeError(
            "Invalid latent slice window: "
            f"start={start_frame}, requested={num_valid_frames}, total={total_frames}"
        )

    if valid == int(window_frames):
        return clipped.contiguous(), torch.ones(
            int(window_frames), dtype=torch.bool, device=latent_cdt.device
        )

    padded = torch.zeros(
        (*latent_cdt.shape[:-1], int(window_frames)),
        dtype=latent_cdt.dtype,
        device=latent_cdt.device,
    )
    padded[..., :valid] = clipped
    mask = torch.zeros(int(window_frames), dtype=torch.bool, device=latent_cdt.device)
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
    """Read optional sample timing metadata used by automatic FPS resolution."""
    metadata_path = sample_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {"sample_rate": None, "input_samples": None}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"sample_rate": None, "input_samples": None}
    if not isinstance(payload, dict):
        return {"sample_rate": None, "input_samples": None}

    sample_rate = payload.get("sample_rate")
    input_samples = payload.get("input_samples")
    return {
        "sample_rate": int(sample_rate) if isinstance(sample_rate, int) else None,
        "input_samples": (
            int(input_samples) if isinstance(input_samples, int) else None
        ),
    }


def _resolve_manifest_sample_dir(dataset_root: Path, sample_dir_raw: Any) -> Path:
    """Resolve old Windows/POSIX manifest paths under the configured root."""
    raw = str(sample_dir_raw).strip()
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    lower_parts = [part.lower() for part in parts]
    if "samples" in lower_parts:
        samples_index = lower_parts.index("samples")
        return dataset_root.joinpath(*parts[samples_index:])

    path = Path(normalized)
    if path.is_absolute():
        return path
    return dataset_root.joinpath(*parts)


def _target_shape(payload: dict[str, Any], line_index: int) -> tuple[int, int]:
    """Return target channel/frame counts from an old latent manifest row."""
    shape = payload.get("target_latent_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) not in {2, 3}:
        raise ValueError(
            f"manifest line {line_index}: invalid target_latent_shape={shape!r}"
        )
    if len(shape) == 3:
        target_channels = int(shape[0])
    else:
        target_channels = int(payload.get("target_channels", 1))
    target_frames = int(shape[-1])
    if target_channels <= 0 or target_frames <= 0:
        raise ValueError(
            f"manifest line {line_index}: target latent dimensions must be > 0"
        )
    return target_channels, target_frames


def _load_manifest_records(
    *,
    dataset_root: Path,
    manifest_path: Path,
) -> list[SongRecord]:
    """Load one old latent manifest into current shared song records."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    songs: list[SongRecord] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on manifest line {line_index}: {manifest_path}"
                ) from error
            if not isinstance(payload, dict):
                raise TypeError(f"manifest line {line_index}: expected a JSON object")

            sample_dir_raw = payload.get("sample_dir")
            if not sample_dir_raw:
                raise KeyError(f"manifest line {line_index}: missing key 'sample_dir'")
            sample_dir = _resolve_manifest_sample_dir(dataset_root, sample_dir_raw)
            target_channels, target_frames = _target_shape(payload, line_index)
            metadata = _try_read_metadata(sample_dir)

            manifest_sample_rate = payload.get("sample_rate")
            manifest_input_samples = payload.get("input_samples")
            sample_rate = metadata["sample_rate"]
            input_samples = metadata["input_samples"]
            if sample_rate is None and isinstance(manifest_sample_rate, int):
                sample_rate = int(manifest_sample_rate)
            if input_samples is None and isinstance(manifest_input_samples, int):
                input_samples = int(manifest_input_samples)

            songs.append(
                SongRecord(
                    stream_hash=str(payload.get("stream_hash", sample_dir.name)),
                    sample_dir=sample_dir,
                    target_frames=target_frames,
                    target_channels=target_channels,
                    sample_rate=sample_rate,
                    input_samples=input_samples,
                )
            )
    return songs


def _path_list_or_none(
    value: str | Path | list[str | Path] | None,
) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if not isinstance(value, list):
        raise TypeError("sample exclusion path must be a path or list of paths")
    return [Path(item) for item in value]


def _load_sample_exclusion_keys(
    paths: str | Path | list[str | Path] | None,
) -> tuple[set[str], set[str]]:
    """Load current JSON/CSV/text sample exclusion formats."""
    stream_hashes: set[str] = set()
    sample_dirs: set[str] = set()
    for path in _path_list_or_none(paths):
        if not path.exists():
            raise FileNotFoundError(f"Sample exclusion file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                stream_hashes.update(str(item).strip().lower() for item in payload)
            elif isinstance(payload, dict):
                for key in (
                    "exclude_stream_hashes",
                    "excluded_stream_hashes",
                    "stream_hashes",
                ):
                    values = payload.get(key)
                    if isinstance(values, list):
                        stream_hashes.update(
                            str(item).strip().lower() for item in values
                        )
                for key in (
                    "exclude_sample_dirs",
                    "excluded_sample_dirs",
                    "sample_dirs",
                ):
                    values = payload.get(key)
                    if isinstance(values, list):
                        sample_dirs.update(
                            str(item).replace("\\", "/").strip().lower()
                            for item in values
                        )
            else:
                raise TypeError(f"Unsupported JSON exclusion payload in {path}")
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    stream_hash = str(row.get("stream_hash") or "").strip().lower()
                    sample_dir = (
                        str(row.get("sample_dir") or "")
                        .replace("\\", "/")
                        .strip()
                        .lower()
                    )
                    if stream_hash:
                        stream_hashes.add(stream_hash)
                    if sample_dir:
                        sample_dirs.add(sample_dir)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if item and not item.startswith("#"):
                    stream_hashes.add(item.lower())

    stream_hashes.discard("")
    sample_dirs.discard("")
    return stream_hashes, sample_dirs


def _filter_songs_by_sample_exclusion(
    songs: list[SongRecord],
    *,
    sample_exclusion_path: str | Path | list[str | Path] | None,
) -> tuple[list[SongRecord], int]:
    """Drop old latent songs selected by current exclusion-file semantics."""
    stream_hashes, sample_dirs = _load_sample_exclusion_keys(sample_exclusion_path)
    if not stream_hashes and not sample_dirs:
        return songs, 0

    kept: list[SongRecord] = []
    dropped = 0
    for song in songs:
        stream_hash = song.stream_hash.strip().lower()
        sample_dir = str(song.sample_dir).replace("\\", "/").lower()
        sample_dir_name = song.sample_dir.name.strip().lower()
        excluded_by_dir = any(
            sample_dir == excluded or sample_dir.endswith(f"/{excluded}")
            for excluded in sample_dirs
        )
        if (
            stream_hash in stream_hashes
            or sample_dir_name in stream_hashes
            or excluded_by_dir
        ):
            dropped += 1
        else:
            kept.append(song)
    return kept, dropped


def _load_bundle_latents(bundle_path: Path) -> dict[str, Any]:
    payload = _torch_load_cpu(bundle_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid bundle payload type: {type(payload)}")
    missing = [key for key in _LATENT_KEYS if key not in payload]
    if missing:
        raise KeyError(
            f"Latent bundle {bundle_path} is missing keys: {', '.join(missing)}"
        )
    return {key: payload[key] for key in _LATENT_KEYS}


def _load_split_latents(sample_dir: Path) -> dict[str, Any]:
    return {
        key: _torch_load_cpu(sample_dir / filename)
        for key, filename in _SPLIT_FILENAMES.items()
    }


def _load_latents_from_sample(
    sample_dir: Path,
    *,
    sample_artifact_mode: str = "bundle",
) -> dict[str, torch.Tensor]:
    """Load legacy bundle/split artifacts and normalize every latent tensor."""
    mode = str(sample_artifact_mode).strip().lower()
    if mode not in {"bundle", "split"}:
        raise ValueError("sample_artifact_mode must be one of: bundle, split")

    bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
    split_paths = {
        key: sample_dir / filename for key, filename in _SPLIT_FILENAMES.items()
    }
    split_complete = all(path.exists() for path in split_paths.values())

    latents: dict[str, Any]
    if mode == "bundle" and bundle_path.exists():
        latents = _load_bundle_latents(bundle_path)
    elif mode == "split" and split_complete:
        latents = _load_split_latents(sample_dir)
    elif bundle_path.exists():
        latents = _load_bundle_latents(bundle_path)
    elif split_complete:
        latents = _load_split_latents(sample_dir)
    else:
        missing = [str(path) for path in split_paths.values() if not path.exists()]
        raise FileNotFoundError(
            f"Missing latent artifacts under {sample_dir}; expected {bundle_path} "
            "or all split files:\n  - " + "\n  - ".join(missing)
        )

    normalized: dict[str, torch.Tensor] = {}
    for key, value in latents.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")
        normalized[key] = _to_cdt(value, key)
    return normalized


__all__ = [
    "METADATA_FILENAME",
    "SAMPLE_BUNDLE_FILENAME",
    "SOURCE_DOWNMIX_LATENT_FILENAME",
    "SOURCE_MONO_LATENT_FILENAME",
    "SOURCE_STEREO_LATENT_FILENAME",
    "TARGET_LATENT_FILENAME",
]
