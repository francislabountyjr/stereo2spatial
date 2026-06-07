#!/usr/bin/env python3
"""Convert waveform dataset signal artifacts to lossless FLAC PCM_24."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from stereo2spatial.training.dataset_io import _load_raw_signals_from_sample

MANIFEST_FILENAME = "manifest.jsonl"
METADATA_FILENAME = "metadata.json"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"
TARGET_SIGNAL_FILENAME = "target_signal.pt"
SOURCE_STEREO_SIGNAL_FILENAME = "source_stereo_signal.pt"
SOURCE_MONO_SIGNAL_FILENAME = "source_mono_signal.pt"
SOURCE_DOWNMIX_SIGNAL_FILENAME = "source_downmix_signal.pt"
TARGET_SIGNAL_FLAC_FILENAME = "target_signal.flac"
SOURCE_STEREO_SIGNAL_FLAC_FILENAME = "source_stereo_signal.flac"
SOURCE_MONO_SIGNAL_FLAC_FILENAME = "source_mono_signal.flac"
SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME = "source_downmix_signal.flac"


@dataclass(frozen=True)
class ConvertResult:
    ok: bool
    sample_dir: str
    output_sample_dir: str
    stream_hash: str
    source_path: str
    bytes_in: int
    bytes_out: int
    manifest_record: dict[str, Any] | None
    error: str | None = None


def _sample_dirs(dataset_root: Path) -> list[Path]:
    samples_root = dataset_root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_root}")
    return sorted(path.parent for path in samples_root.rglob(METADATA_FILENAME))


def _read_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_flac(path: Path, signal: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        sf.write(
            str(tmp_path),
            signal.float().t().cpu().numpy(),
            int(sample_rate),
            format="FLAC",
            subtype="PCM_24",
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _probe_flac_signal(path: Path) -> tuple[int, int]:
    """Return `(channels, frames)` after a fast FLAC integrity probe."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{path} is empty")
    with sf.SoundFile(str(path), mode="r") as handle:
        channels = int(handle.channels)
        frames = int(len(handle))
        if channels <= 0:
            raise RuntimeError(f"{path} reports zero channels")
        if frames <= 0:
            raise RuntimeError(f"{path} reports zero frames")
        handle.seek(frames - 1)
        tail = handle.read(1, dtype="float32", always_2d=True)
        if getattr(tail, "shape", (0,))[0] != 1:
            raise RuntimeError(f"{path} could not decode final frame")
    return channels, frames


def _try_existing_flac_result(
    *,
    sample_dir: Path,
    output_root: Path,
    output_sample_dir: Path,
    metadata: dict[str, Any],
    source_path: str,
) -> ConvertResult | None:
    """Return a valid existing conversion result without re-reading source tensors."""
    required_paths = {
        "target_signal": output_sample_dir / TARGET_SIGNAL_FLAC_FILENAME,
        "source_stereo_signal": output_sample_dir / SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
    }
    if any(not path.exists() for path in required_paths.values()):
        return None

    flac_paths = dict(required_paths)
    optional_paths = {
        "source_mono_signal": output_sample_dir / SOURCE_MONO_SIGNAL_FLAC_FILENAME,
        "source_downmix_signal": output_sample_dir
        / SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME,
    }
    flac_paths.update(
        {key: path for key, path in optional_paths.items() if path.exists()}
    )

    probes: dict[str, tuple[int, int]] = {}
    try:
        for key, path in flac_paths.items():
            probes[key] = _probe_flac_signal(path)
    except Exception:
        return None

    required_lengths = {key: probes[key][1] for key in required_paths}
    if len(set(required_lengths.values())) != 1:
        return None
    all_lengths = {key: frames for key, (_channels, frames) in probes.items()}
    if len(set(all_lengths.values())) != 1:
        return None

    files = {
        "target_signal": TARGET_SIGNAL_FLAC_FILENAME,
        "source_stereo_signal": SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
        "metadata": METADATA_FILENAME,
    }
    if "source_mono_signal" in probes:
        files["source_mono_signal"] = SOURCE_MONO_SIGNAL_FLAC_FILENAME
    if "source_downmix_signal" in probes:
        files["source_downmix_signal"] = SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME

    metadata = dict(metadata)
    metadata["sample_artifact_mode"] = "flac"
    metadata["signal_codec"] = "flac_pcm24"
    metadata["files"] = files
    metadata["target_signal_shape"] = [
        int(probes["target_signal"][0]),
        int(probes["target_signal"][1]),
    ]
    metadata["source_stereo_signal_shape"] = [
        int(probes["source_stereo_signal"][0]),
        int(probes["source_stereo_signal"][1]),
    ]
    if "source_mono_signal" in probes:
        metadata["source_mono_signal_shape"] = [
            int(probes["source_mono_signal"][0]),
            int(probes["source_mono_signal"][1]),
        ]
    if "source_downmix_signal" in probes:
        metadata["source_downmix_signal_shape"] = [
            int(probes["source_downmix_signal"][0]),
            int(probes["source_downmix_signal"][1]),
        ]
    metadata_path = output_sample_dir / METADATA_FILENAME
    _write_json_atomic(metadata_path, metadata)
    manifest_record = _manifest_record_from_metadata(
        output_root=output_root,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    return ConvertResult(
        ok=True,
        sample_dir=str(sample_dir),
        output_sample_dir=str(output_sample_dir),
        stream_hash=str(metadata.get("stream_hash") or sample_dir.name),
        source_path=source_path,
        bytes_in=0,
        bytes_out=_dir_size(output_sample_dir),
        manifest_record=manifest_record,
    )


def _remove_known_output_artifacts(output_sample_dir: Path) -> None:
    for filename in (
        TARGET_SIGNAL_FLAC_FILENAME,
        SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
        SOURCE_MONO_SIGNAL_FLAC_FILENAME,
        SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME,
        METADATA_FILENAME,
    ):
        path = output_sample_dir / filename
        if path.exists():
            path.unlink()
    for tmp_path in output_sample_dir.glob("*.tmp-*"):
        if tmp_path.is_file():
            tmp_path.unlink()


def _finite_or_raise(signals: dict[str, torch.Tensor], sample_dir: Path) -> None:
    for key, signal in signals.items():
        finite = torch.isfinite(signal.float())
        if bool(finite.all()):
            continue
        bad = int((~finite).sum().item())
        raise RuntimeError(
            f"{sample_dir} {key} contains non-finite samples: "
            f"bad={bad}/{int(signal.numel())}"
        )


def _matching_lengths_or_raise(
    signals: dict[str, torch.Tensor],
    sample_dir: Path,
) -> None:
    lengths = {
        key: int(signal.shape[-1])
        for key, signal in signals.items()
        if isinstance(signal, torch.Tensor) and signal.dim() >= 1
    }
    required = {"target_signal", "source_stereo_signal"}
    if not required.issubset(lengths):
        missing = sorted(required - set(lengths))
        raise RuntimeError(f"{sample_dir} missing required signals: {missing}")
    if len(set(lengths.values())) > 1:
        raise RuntimeError(f"{sample_dir} signal length mismatch: {lengths}")


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.glob("*") if item.is_file())


def _manifest_record_from_metadata(
    *,
    output_root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "created_utc": metadata.get("created_utc"),
        "stream_hash": metadata.get("stream_hash"),
        "signal_layout": metadata.get("signal_layout", "c_s"),
        "sample_dir": str(metadata_path.parent.relative_to(output_root)),
        "source_relpath": metadata.get("source_relpath"),
        "target_channels": metadata.get("target_channels"),
        "target_channel_labels": metadata.get("target_channel_labels"),
        "target_channel_mask": metadata.get("target_channel_mask"),
        "target_is_binaural_stereo": metadata.get("target_is_binaural_stereo"),
        "mix_style_layout_mode": metadata.get("mix_style_layout_mode"),
        "mix_style_names": metadata.get("mix_style_names"),
        "mix_style_inactive_names": metadata.get("mix_style_inactive_names"),
        "target_signal_shape": metadata.get("target_signal_shape"),
        "source_stereo_signal_shape": metadata.get("source_stereo_signal_shape"),
        "source_mono_saved": metadata.get("source_mono_saved"),
        "source_downmix_saved": metadata.get("source_downmix_saved"),
        "mix_style_raw": metadata.get("mix_style_raw"),
        "mix_style_vector": metadata.get("mix_style_vector"),
        "mix_style": metadata.get("mix_style"),
        "signal_rms": metadata.get("signal_rms"),
        "sample_artifact_mode": "flac",
        "dead_channel_indices": metadata.get("dead_channel_indices", []),
        "duplicate_channel_pairs": metadata.get("duplicate_channel_pairs", []),
    }
    if "source_downmix_signal_shape" in metadata:
        record["source_downmix_signal_shape"] = metadata["source_downmix_signal_shape"]
    if "source_mono_signal_shape" in metadata:
        record["source_mono_signal_shape"] = metadata["source_mono_signal_shape"]
    return {key: value for key, value in record.items() if value is not None}


def _convert_one(
    sample_dir_text: str,
    input_root_text: str,
    output_root_text: str,
    delete_source_tensors: bool,
) -> ConvertResult:
    sample_dir = Path(sample_dir_text)
    input_root = Path(input_root_text)
    output_root = Path(output_root_text)
    metadata: dict[str, Any] = {}
    try:
        metadata = _read_metadata(sample_dir)
        source_path = str(
            metadata.get("source_path") or metadata.get("source_relpath") or ""
        )
        relative = sample_dir.relative_to(input_root)
        output_sample_dir = output_root / relative
        output_sample_dir.mkdir(parents=True, exist_ok=True)

        existing_result = _try_existing_flac_result(
            sample_dir=sample_dir,
            output_root=output_root,
            output_sample_dir=output_sample_dir,
            metadata=metadata,
            source_path=source_path,
        )
        if existing_result is not None:
            return existing_result

        _remove_known_output_artifacts(output_sample_dir)
        signals = _load_raw_signals_from_sample(sample_dir)
        _finite_or_raise(signals, sample_dir)
        _matching_lengths_or_raise(signals, sample_dir)
        sample_rate = int(metadata.get("sample_rate") or 48000)

        _write_flac(
            output_sample_dir / TARGET_SIGNAL_FLAC_FILENAME,
            signals["target_signal"],
            sample_rate,
        )
        _write_flac(
            output_sample_dir / SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
            signals["source_stereo_signal"],
            sample_rate,
        )
        files = {
            "target_signal": TARGET_SIGNAL_FLAC_FILENAME,
            "source_stereo_signal": SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
            "metadata": METADATA_FILENAME,
        }
        if "source_mono_signal" in signals:
            _write_flac(
                output_sample_dir / SOURCE_MONO_SIGNAL_FLAC_FILENAME,
                signals["source_mono_signal"],
                sample_rate,
            )
            files["source_mono_signal"] = SOURCE_MONO_SIGNAL_FLAC_FILENAME
        if "source_downmix_signal" in signals:
            _write_flac(
                output_sample_dir / SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME,
                signals["source_downmix_signal"],
                sample_rate,
            )
            files["source_downmix_signal"] = SOURCE_DOWNMIX_SIGNAL_FLAC_FILENAME

        metadata = dict(metadata)
        metadata["sample_artifact_mode"] = "flac"
        metadata["signal_codec"] = "flac_pcm24"
        metadata["files"] = files
        metadata["target_signal_shape"] = [
            int(x) for x in signals["target_signal"].shape
        ]
        metadata["source_stereo_signal_shape"] = [
            int(x) for x in signals["source_stereo_signal"].shape
        ]
        if "source_mono_signal" in signals:
            metadata["source_mono_signal_shape"] = [
                int(x) for x in signals["source_mono_signal"].shape
            ]
        if "source_downmix_signal" in signals:
            metadata["source_downmix_signal_shape"] = [
                int(x) for x in signals["source_downmix_signal"].shape
            ]
        metadata_path = output_sample_dir / METADATA_FILENAME
        _write_json_atomic(metadata_path, metadata)
        manifest_record = _manifest_record_from_metadata(
            output_root=output_root,
            metadata_path=metadata_path,
            metadata=metadata,
        )

        bytes_in = _dir_size(sample_dir)
        bytes_out = _dir_size(output_sample_dir)

        if delete_source_tensors:
            for filename in (
                SAMPLE_BUNDLE_FILENAME,
                TARGET_SIGNAL_FILENAME,
                SOURCE_STEREO_SIGNAL_FILENAME,
                SOURCE_MONO_SIGNAL_FILENAME,
                SOURCE_DOWNMIX_SIGNAL_FILENAME,
            ):
                path = sample_dir / filename
                if path.exists():
                    path.unlink()
            if input_root == output_root:
                (sample_dir / METADATA_FILENAME).write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )

        return ConvertResult(
            ok=True,
            sample_dir=str(sample_dir),
            output_sample_dir=str(output_sample_dir),
            stream_hash=str(metadata.get("stream_hash") or sample_dir.name),
            source_path=source_path,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            manifest_record=manifest_record,
        )
    except Exception as error:
        source_path = str(
            metadata.get("source_path") or metadata.get("source_relpath") or ""
        )
        stream_hash = str(metadata.get("stream_hash") or sample_dir.name)
        return ConvertResult(
            ok=False,
            sample_dir=str(sample_dir),
            output_sample_dir="",
            stream_hash=stream_hash,
            source_path=source_path,
            bytes_in=0,
            bytes_out=0,
            manifest_record=None,
            error=str(error),
        )


def _write_failed_repair_csv(path: Path, errors: list[ConvertResult]) -> int:
    """Write a QC-style repair queue for failed conversion rows with source paths."""
    repairable = [
        item
        for item in errors
        if item.source_path
        and Path(item.source_path).is_absolute()
        and Path(item.source_path).exists()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "label",
                "duration_s",
                "stream_hash",
                "sample_dir",
                "error",
            ],
        )
        writer.writeheader()
        for item in repairable:
            writer.writerow(
                {
                    "path": item.source_path,
                    "label": "OK",
                    "duration_s": "",
                    "stream_hash": item.stream_hash,
                    "sample_dir": item.sample_dir,
                    "error": item.error or "",
                }
            )
    return len(repairable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing waveform dataset from .pt tensor artifacts to "
            "lossless FLAC PCM_24 artifacts."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--delete-source-tensors",
        action="store_true",
        help="Delete .pt tensor artifacts after successful conversion. Use only for in-place conversion.",
    )
    parser.add_argument(
        "--failed-repair-csv",
        default=None,
        help=(
            "QC-style CSV written for samples that could not be converted. "
            "Defaults to <output-root>/failed_conversion_repair_rows.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve(strict=False)
    output_root = Path(args.output_root).resolve(strict=False)
    sample_dirs = _sample_dirs(input_root)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: max(0, int(args.max_samples))]
    output_root.mkdir(parents=True, exist_ok=True)

    for filename in ("mix_style_stats.json",):
        source = input_root / filename
        if source.exists() and input_root != output_root:
            shutil.copy2(source, output_root / filename)

    print(f"Converting samples: {len(sample_dirs)}")
    results: list[ConvertResult] = []
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _convert_one,
                str(sample_dir),
                str(input_root),
                str(output_root),
                bool(args.delete_source_tensors),
            ): sample_dir
            for sample_dir in sample_dirs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if args.log_every > 0 and index % int(args.log_every) == 0:
                ok = sum(1 for item in results if item.ok)
                total_in = sum(item.bytes_in for item in results if item.ok)
                total_out = sum(item.bytes_out for item in results if item.ok)
                ratio = total_out / total_in if total_in else 0.0
                print(
                    f"progress converted={index}/{len(sample_dirs)} "
                    f"ok={ok} errors={len(results)-ok} ratio={ratio:.3f}"
                )

    ok_results = [
        item for item in results if item.ok and item.manifest_record is not None
    ]
    errors = [item for item in results if not item.ok]
    failed_repair_csv = (
        Path(args.failed_repair_csv).resolve(strict=False)
        if args.failed_repair_csv
        else output_root / "failed_conversion_repair_rows.csv"
    )
    repairable_errors = _write_failed_repair_csv(failed_repair_csv, errors)
    manifest_path = output_root / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in sorted(ok_results, key=lambda item: str(item.output_sample_dir)):
            handle.write(
                json.dumps(result.manifest_record, ensure_ascii=True, sort_keys=True)
                + "\n"
            )

    total_in = sum(item.bytes_in for item in ok_results)
    total_out = sum(item.bytes_out for item in ok_results)
    ratio = total_out / total_in if total_in else 0.0
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "samples_total": len(sample_dirs),
        "samples_ok": len(ok_results),
        "samples_error": len(errors),
        "bytes_in": total_in,
        "bytes_out": total_out,
        "ratio": ratio,
        "manifest": str(manifest_path),
        "failed_repair_csv": str(failed_repair_csv),
        "repairable_errors": repairable_errors,
        "errors": [
            {
                "sample_dir": item.sample_dir,
                "stream_hash": item.stream_hash,
                "source_path": item.source_path,
                "error": item.error,
            }
            for item in errors[:100]
        ],
    }
    (output_root / "flac_conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(
        f"CONVERT COMPLETE ok={len(ok_results)} errors={len(errors)} ratio={ratio:.3f}"
    )
    print(f"manifest={manifest_path}")
    print(f"summary={output_root / 'flac_conversion_summary.json'}")
    if errors:
        print(f"failed_repair_csv={failed_repair_csv}")
        print(f"repairable_errors={repairable_errors}/{len(errors)}")


if __name__ == "__main__":
    main()
