"""Backfill full-song signal RMS metadata for efficient chunked training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stereo2spatial.training.dataset_io import (  # noqa: E402
    METADATA_FILENAME,
    _load_raw_signals_from_sample,
)

MANIFEST_FILENAME = "manifest.jsonl"
RMS_FIELD = "signal_rms"
DEFAULT_SIGNAL_KEYS = ("target_signal", "source_stereo_signal")


def _load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"manifest line {line_idx}: expected JSON object")
            records.append(payload)
    return records


def _write_manifest_atomic(manifest_path: Path, records: list[dict[str, Any]]) -> None:
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    os.replace(tmp_path, manifest_path)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _sample_dir(dataset_root: Path, record: dict[str, Any]) -> Path:
    raw = record.get("sample_dir")
    if not raw:
        raise KeyError("manifest record is missing sample_dir")
    path = Path(str(raw))
    return path if path.is_absolute() else dataset_root / path


def _valid_existing_rms(value: Any, keys: tuple[str, ...]) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    resolved: dict[str, float] = {}
    for key in keys:
        try:
            rms = float(value[key])
        except (KeyError, TypeError, ValueError):
            return None
        if rms <= 0:
            return None
        resolved[key] = rms
    return resolved


def _compute_signal_rms(
    sample_dir: Path,
    keys: tuple[str, ...],
) -> dict[str, float]:
    signals = _load_raw_signals_from_sample(sample_dir)
    missing = [key for key in keys if key not in signals]
    if missing:
        raise KeyError(f"sample is missing signal(s): {', '.join(missing)}")

    rms_values: dict[str, float] = {}
    for key in keys:
        tensor = signals[key]
        if tensor.dim() != 2:
            raise ValueError(f"{key} must be raw [C,S], got {tuple(tensor.shape)}")
        rms = tensor.float().pow(2).mean().sqrt()
        if not torch.isfinite(rms).item() or float(rms.item()) <= 0:
            raise ValueError(f"{key} has invalid RMS: {float(rms.item())}")
        rms_values[key] = float(rms.item())
    return rms_values


def _process_record(
    *,
    dataset_root: str,
    index: int,
    record: dict[str, Any],
    signal_keys: tuple[str, ...],
    force: bool,
) -> dict[str, Any]:
    sample_dir = _sample_dir(Path(dataset_root), record)
    metadata_path = sample_dir / METADATA_FILENAME
    try:
        metadata = _load_json_object(metadata_path)
        existing = _valid_existing_rms(metadata.get(RMS_FIELD), signal_keys)
        if existing is not None and not force:
            return {
                "index": index,
                "status": "skipped_existing",
                "metadata_path": str(metadata_path),
                "rms": dict(existing),
                "error": None,
            }
        rms_values = _compute_signal_rms(sample_dir, signal_keys)
        return {
            "index": index,
            "status": "updated",
            "metadata_path": str(metadata_path),
            "rms": dict(rms_values),
            "error": None,
        }
    except Exception as error:
        return {
            "index": index,
            "status": "error",
            "metadata_path": str(metadata_path),
            "rms": None,
            "error": str(error),
        }


def _iter_results(
    *,
    dataset_root: Path,
    records: list[dict[str, Any]],
    signal_keys: tuple[str, ...],
    force: bool,
    workers: int,
) -> Iterator[dict[str, Any]]:
    if workers <= 1:
        for index, record in enumerate(records, start=1):
            yield _process_record(
                dataset_root=str(dataset_root),
                index=index,
                record=record,
                signal_keys=signal_keys,
                force=force,
            )
        return

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        pending: set[Future[dict[str, Any]]] = set()
        next_index = 1
        max_pending = max(1, int(workers) * 2)

        def submit_until_full() -> None:
            nonlocal next_index
            while next_index <= len(records) and len(pending) < max_pending:
                record = records[next_index - 1]
                pending.add(
                    executor.submit(
                        _process_record,
                        dataset_root=str(dataset_root),
                        index=next_index,
                        record=record,
                        signal_keys=signal_keys,
                        force=force,
                    )
                )
                next_index += 1

        try:
            submit_until_full()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                submit_until_full()
        except KeyboardInterrupt:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise


def backfill_signal_rms(
    *,
    dataset_root: Path,
    manifest_path: Path,
    signal_keys: tuple[str, ...],
    force: bool,
    dry_run: bool,
    log_every: int,
    max_samples: int | None = None,
    workers: int = 1,
) -> dict[str, int]:
    records = _load_manifest_records(manifest_path)
    if max_samples is not None:
        records = records[: int(max_samples)]
    updated = 0
    skipped_existing = 0
    errors = 0

    for seen, result in enumerate(
        _iter_results(
            dataset_root=dataset_root,
            records=records,
            signal_keys=signal_keys,
            force=force,
            workers=max(1, int(workers)),
        ),
        start=1,
    ):
        index = int(result["index"])
        record = records[index - 1]
        status = str(result["status"])
        if status == "error":
            errors += 1
            print(
                f"[error] index={index} "
                f"metadata={result.get('metadata_path')} "
                f"error={result.get('error')}",
                flush=True,
            )
        else:
            rms_values = result["rms"]
            if not isinstance(rms_values, dict):
                errors += 1
                continue
            record[RMS_FIELD] = dict(rms_values)
            if status == "skipped_existing":
                skipped_existing += 1
            else:
                updated += 1
                if not dry_run:
                    metadata_path = Path(str(result["metadata_path"]))
                    metadata = _load_json_object(metadata_path)
                    metadata[RMS_FIELD] = dict(rms_values)
                    _write_json_atomic(metadata_path, metadata)

        if log_every > 0 and (seen % log_every == 0 or seen == len(records)):
            print(
                "progress "
                f"rows={seen}/{len(records)} "
                f"updated={updated} "
                f"skipped_existing={skipped_existing} "
                f"errors={errors}",
                flush=True,
            )

    if not dry_run:
        _write_manifest_atomic(manifest_path, records)

    return {
        "manifest_records": len(records),
        "updated": updated,
        "skipped_existing": skipped_existing,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-song RMS values for source/target tensors and store them "
            "in metadata.json plus manifest.jsonl. This lets training compute "
            "amplitude-lift gains without scanning full waveforms per crop."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-file", default=MANIFEST_FILENAME)
    parser.add_argument(
        "--signal-key",
        action="append",
        default=None,
        help=(
            "Signal key to backfill. Repeatable. Defaults to target_signal and "
            "source_stereo_signal."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional leading sample limit for timing/testing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).resolve(strict=False)
    manifest_path = Path(args.manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    signal_keys = (
        tuple(str(item) for item in args.signal_key)
        if args.signal_key
        else DEFAULT_SIGNAL_KEYS
    )
    summary = backfill_signal_rms(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        signal_keys=signal_keys,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        log_every=int(args.log_every),
        max_samples=args.max_samples,
        workers=int(args.workers),
    )
    print("Signal RMS backfill summary:")
    for key, value in summary.items():
        print(f"  - {key}={value}")
    print(f"  - dataset_root={dataset_root}")
    print(f"  - manifest={manifest_path}")
    print(f"  - signal_keys={list(signal_keys)}")
    if args.dry_run:
        print("  - dry_run=True")


if __name__ == "__main__":
    main()
