#!/usr/bin/env python3
"""Find and optionally remove dataset samples containing non-finite signal tensors."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

MANIFEST_FILENAME = "manifest.jsonl"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"
TARGET_SIGNAL_FILENAME = "target_signal.pt"
SOURCE_STEREO_SIGNAL_FILENAME = "source_stereo_signal.pt"
SOURCE_MONO_SIGNAL_FILENAME = "source_mono_signal.pt"
SOURCE_DOWNMIX_SIGNAL_FILENAME = "source_downmix_signal.pt"
TARGET_SIGNAL_FLAC_FILENAME = "target_signal.flac"
SOURCE_STEREO_SIGNAL_FLAC_FILENAME = "source_stereo_signal.flac"
SOURCE_MONO_SIGNAL_FLAC_FILENAME = "source_mono_signal.flac"
SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME = "source_downmix_signal.flac"
_SIGNAL_FILENAMES = {
    "target_signal": TARGET_SIGNAL_FILENAME,
    "source_stereo_signal": SOURCE_STEREO_SIGNAL_FILENAME,
    "source_mono_signal": SOURCE_MONO_SIGNAL_FILENAME,
    "source_downmix_signal": SOURCE_DOWNMIX_SIGNAL_FILENAME,
}
_FLAC_SIGNAL_FILENAMES = {
    "target_signal": TARGET_SIGNAL_FLAC_FILENAME,
    "source_stereo_signal": SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
    "source_mono_signal": SOURCE_MONO_SIGNAL_FLAC_FILENAME,
    "source_downmix_signal": SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME,
}
_REQUIRED_SIGNALS = {"target_signal", "source_stereo_signal"}


@dataclass(frozen=True)
class InvalidTensor:
    sample_dir: str
    stream_hash: str
    source_path: str
    tensor_name: str
    shape: tuple[int, ...]
    bad_count: int
    total_count: int


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def _read_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload: object = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items()}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_signal_items(sample_dir: Path) -> dict[str, torch.Tensor]:
    bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
    if bundle_path.exists():
        payload = _torch_load_cpu(bundle_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid bundle payload type: {type(payload)}")
        return {
            key: value
            for key, value in payload.items()
            if isinstance(value, torch.Tensor)
        }

    flac_paths = {
        key: sample_dir / filename for key, filename in _FLAC_SIGNAL_FILENAMES.items()
    }
    existing_flac = {key: path for key, path in flac_paths.items() if path.exists()}
    if existing_flac:
        return {key: _read_flac_signal(path) for key, path in existing_flac.items()}

    paths = {key: sample_dir / filename for key, filename in _SIGNAL_FILENAMES.items()}
    return {key: _torch_load_cpu(path) for key, path in paths.items() if path.exists()}


def _read_flac_signal(path: Path) -> torch.Tensor:
    audio, _sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return torch.from_numpy(audio.T.copy()).contiguous()


def _invalid_row(
    *,
    sample_dir: Path,
    metadata: dict[str, Any],
    tensor_name: str,
    shape: tuple[int, ...] = (),
    bad_count: int = 0,
    total_count: int = 0,
) -> InvalidTensor:
    stream_hash = str(metadata.get("stream_hash") or sample_dir.name)
    source_path = str(
        metadata.get("source_path") or metadata.get("source_relpath") or ""
    )
    return InvalidTensor(
        sample_dir=str(sample_dir),
        stream_hash=stream_hash,
        source_path=source_path,
        tensor_name=tensor_name,
        shape=shape,
        bad_count=bad_count,
        total_count=total_count,
    )


def _probe_flac_signal(path: Path) -> tuple[int, int]:
    """Return `(channels, frames)` after a fast FLAC integrity probe."""
    if path.stat().st_size <= 0:
        raise RuntimeError("file is empty")
    with sf.SoundFile(str(path), mode="r") as handle:
        channels = int(handle.channels)
        frames = int(len(handle))
        if channels <= 0:
            raise RuntimeError("file reports zero channels")
        if frames <= 0:
            raise RuntimeError("file reports zero frames")
        handle.seek(frames - 1)
        tail = handle.read(1, dtype="float32", always_2d=True)
        if getattr(tail, "shape", (0,))[0] != 1:
            raise RuntimeError("file could not decode final frame")
    return channels, frames


def _scan_flac_sample(
    *,
    sample_dir: Path,
    metadata: dict[str, Any],
    deep_scan: bool,
) -> list[InvalidTensor]:
    invalid: list[InvalidTensor] = []
    lengths: dict[str, int] = {}
    existing = {
        key: sample_dir / filename
        for key, filename in _FLAC_SIGNAL_FILENAMES.items()
        if (sample_dir / filename).exists()
    }

    for key in sorted(_REQUIRED_SIGNALS - set(existing)):
        invalid.append(
            _invalid_row(
                sample_dir=sample_dir,
                metadata=metadata,
                tensor_name=f"{key}_missing",
            )
        )

    for key, path in existing.items():
        try:
            channels, frames = _probe_flac_signal(path)
        except Exception as error:
            invalid.append(
                _invalid_row(
                    sample_dir=sample_dir,
                    metadata=metadata,
                    tensor_name=f"{key}_flac_invalid:{type(error).__name__}",
                    shape=(0,),
                )
            )
            continue
        lengths[key] = frames
        if frames <= 0:
            invalid.append(
                _invalid_row(
                    sample_dir=sample_dir,
                    metadata=metadata,
                    tensor_name=f"{key}_zero_length",
                    shape=(channels, frames),
                )
            )

    if _REQUIRED_SIGNALS.issubset(lengths) and len(set(lengths.values())) > 1:
        invalid.append(
            _invalid_row(
                sample_dir=sample_dir,
                metadata=metadata,
                tensor_name="length_mismatch",
                shape=tuple(lengths[key] for key in sorted(lengths)),
            )
        )

    if deep_scan and not invalid:
        for key, path in existing.items():
            try:
                tensor = _read_flac_signal(path)
            except Exception as error:
                invalid.append(
                    _invalid_row(
                        sample_dir=sample_dir,
                        metadata=metadata,
                        tensor_name=f"{key}_flac_decode:{type(error).__name__}",
                    )
                )
                continue
            finite = torch.isfinite(tensor.float())
            if not bool(finite.all()):
                invalid.append(
                    _invalid_row(
                        sample_dir=sample_dir,
                        metadata=metadata,
                        tensor_name=key,
                        shape=tuple(int(x) for x in tensor.shape),
                        bad_count=int((~finite).sum().item()),
                        total_count=int(tensor.numel()),
                    )
                )
    return invalid


def _scan_sample(
    sample_dir_text: str, deep_flac_scan: bool = False
) -> list[InvalidTensor]:
    sample_dir = Path(sample_dir_text)
    metadata = _read_metadata(sample_dir)
    invalid: list[InvalidTensor] = []

    has_flac = any(
        (sample_dir / filename).exists() for filename in _FLAC_SIGNAL_FILENAMES.values()
    )
    if has_flac:
        return _scan_flac_sample(
            sample_dir=sample_dir,
            metadata=metadata,
            deep_scan=bool(deep_flac_scan),
        )

    try:
        signals = _load_signal_items(sample_dir)
    except Exception as error:
        return [
            _invalid_row(
                sample_dir=sample_dir,
                metadata=metadata,
                tensor_name=f"sample_load_error:{type(error).__name__}",
            )
        ]
    for key in sorted(_REQUIRED_SIGNALS - set(signals)):
        invalid.append(
            _invalid_row(
                sample_dir=sample_dir,
                metadata=metadata,
                tensor_name=f"{key}_missing",
            )
        )
    lengths = {
        key: int(tensor.shape[-1])
        for key, tensor in signals.items()
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 1
    }
    if _REQUIRED_SIGNALS.issubset(lengths) and len(set(lengths.values())) > 1:
        invalid.append(
            _invalid_row(
                sample_dir=sample_dir,
                metadata=metadata,
                tensor_name="length_mismatch",
                shape=tuple(lengths[key] for key in sorted(lengths)),
            )
        )
    for key, tensor in signals.items():
        if int(tensor.numel()) <= 0 or int(tensor.shape[-1]) <= 0:
            invalid.append(
                _invalid_row(
                    sample_dir=sample_dir,
                    metadata=metadata,
                    tensor_name=f"{key}_zero_length",
                    shape=tuple(int(x) for x in tensor.shape),
                )
            )
            continue
        finite = torch.isfinite(tensor.float())
        if bool(finite.all()):
            continue
        invalid.append(
            _invalid_row(
                sample_dir=sample_dir,
                metadata=metadata,
                tensor_name=key,
                shape=tuple(int(x) for x in tensor.shape),
                bad_count=int((~finite).sum().item()),
                total_count=int(tensor.numel()),
            )
        )
    return invalid


def _iter_sample_dirs(dataset_root: Path) -> list[Path]:
    samples_root = dataset_root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_root}")
    out: list[Path] = []
    for metadata_path in samples_root.rglob("metadata.json"):
        sample_dir = metadata_path.parent
        if (
            (sample_dir / SAMPLE_BUNDLE_FILENAME).exists()
            or (sample_dir / TARGET_SIGNAL_FILENAME).exists()
            or (sample_dir / TARGET_SIGNAL_FLAC_FILENAME).exists()
        ):
            out.append(sample_dir)
    return sorted(out)


def _write_report(path: Path, rows: list[InvalidTensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_dir",
                "stream_hash",
                "source_path",
                "tensor_name",
                "shape",
                "bad_count",
                "total_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_dir": row.sample_dir,
                    "stream_hash": row.stream_hash,
                    "source_path": row.source_path,
                    "tensor_name": row.tensor_name,
                    "shape": "x".join(str(x) for x in row.shape),
                    "bad_count": row.bad_count,
                    "total_count": row.total_count,
                }
            )


def _prune_manifest(dataset_root: Path, removed_hashes: set[str]) -> dict[str, int]:
    manifest_path = dataset_root / MANIFEST_FILENAME
    stats = {"exists": int(manifest_path.exists()), "kept": 0, "removed": 0}
    if not manifest_path.exists() or not removed_hashes:
        return stats

    kept_lines: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                stats["kept"] += 1
                continue
            stream_hash = str(payload.get("stream_hash", "")).strip().lower()
            if stream_hash in removed_hashes:
                stats["removed"] += 1
                continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")
            stats["kept"] += 1

    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept_lines)
    tmp_path.replace(manifest_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan dataset signal tensors for NaN/Inf values."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-csv", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--deep-flac-scan",
        action="store_true",
        help="Decode full FLAC files to check finite samples. Default is a fast header/tail integrity scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve(strict=False)
    sample_dirs = _iter_sample_dirs(dataset_root)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: max(0, int(args.max_samples))]

    print(f"Scanning samples: {len(sample_dirs)}")
    invalid: list[InvalidTensor] = []
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _scan_sample, str(sample_dir), bool(args.deep_flac_scan)
            ): sample_dir
            for sample_dir in sample_dirs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            rows = future.result()
            invalid.extend(rows)
            if args.log_every > 0 and index % int(args.log_every) == 0:
                print(
                    f"progress scanned={index}/{len(sample_dirs)} invalid={len(invalid)}"
                )

    invalid_dirs = {Path(row.sample_dir) for row in invalid}
    invalid_hashes = {
        row.stream_hash.strip().lower() for row in invalid if row.stream_hash
    }
    print(f"INVALID TENSORS: {len(invalid)}")
    print(f"INVALID SAMPLE DIRS: {len(invalid_dirs)}")
    for row in invalid[:50]:
        print(
            f"  - {row.sample_dir} {row.tensor_name} "
            f"bad={row.bad_count}/{row.total_count} source={row.source_path}"
        )
    if len(invalid) > 50:
        print(f"  ... {len(invalid) - 50} more omitted")

    if args.report_csv:
        report_path = Path(args.report_csv).resolve(strict=False)
        _write_report(report_path, invalid)
        print(f"report_csv={report_path}")

    if not args.apply:
        print(
            "DRY RUN ONLY: pass --apply to delete invalid sample dirs and prune manifest."
        )
        return

    for sample_dir in invalid_dirs:
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
    manifest_stats = _prune_manifest(dataset_root, invalid_hashes)
    print(f"deleted_sample_dirs={len(invalid_dirs)}")
    print(f"manifest_prune={manifest_stats}")


if __name__ == "__main__":
    main()
