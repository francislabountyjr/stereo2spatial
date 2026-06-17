"""Scan dataset samples for WavFlow-style amplitude-lift clipping risk."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stereo2spatial.common.amplitude_lift import (  # noqa: E402
    wavflow_source_peak_stats,
    wavflow_target_clip_stats,
)
from stereo2spatial.training.dataset_io import (  # noqa: E402
    _load_raw_signals_from_sample,
)

MANIFEST_FILENAME = "manifest.jsonl"


def _load_manifest_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"manifest line {line_idx}: expected JSON object")
            records.append(payload)
    return records


def _sample_dir(dataset_root: Path, record: dict[str, Any]) -> Path:
    raw = record.get("sample_dir")
    if not raw:
        raise KeyError("manifest record is missing sample_dir")
    path = Path(str(raw))
    return path if path.is_absolute() else dataset_root / path


def _process_record(
    *,
    dataset_root: str,
    index: int,
    record: dict[str, Any],
    target_rms: float,
    peak_limit: float,
    peak_rescale_min_rms: float,
    eps: float,
) -> dict[str, Any]:
    sample_dir = _sample_dir(Path(dataset_root), record)
    stream_hash = str(record.get("stream_hash") or sample_dir.name)
    try:
        signals = _load_raw_signals_from_sample(sample_dir)
        target_stats = wavflow_target_clip_stats(
            signals["target_signal"],
            target_rms=target_rms,
            peak_limit=peak_limit,
            peak_rescale_min_rms=peak_rescale_min_rms,
            eps=eps,
        )
        source_stats = wavflow_source_peak_stats(
            signals["source_stereo_signal"],
            target_rms=target_rms,
            peak_limit=peak_limit,
            eps=eps,
        )
        return {
            "index": index,
            "status": "ok",
            "stream_hash": stream_hash,
            "sample_dir": str(sample_dir),
            "error": "",
            **{f"target_{key}": value for key, value in target_stats.items()},
            **{f"source_{key}": value for key, value in source_stats.items()},
        }
    except Exception as error:
        return {
            "index": index,
            "status": "error",
            "stream_hash": stream_hash,
            "sample_dir": str(sample_dir),
            "error": str(error),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "status",
        "stream_hash",
        "sample_dir",
        "error",
        "target_rms_gain",
        "target_peak_after_rms",
        "target_rms_after_gain",
        "target_peak_rescale_applied",
        "target_peak_rescale_scale",
        "target_pre_clamp_peak",
        "target_would_clip",
        "target_clipped_samples",
        "source_rms_gain",
        "source_peak_after_rms",
        "source_peak_rescale_required",
        "source_peak_rescale_scale",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-file", default=MANIFEST_FILENAME)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-rms", type=float, default=0.33)
    parser.add_argument("--peak-limit", type=float, default=1.0)
    parser.add_argument("--peak-rescale-min-rms", type=float, default=0.3)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    records = _load_manifest_records(manifest_path)
    if args.max_samples is not None:
        records = records[: max(0, int(args.max_samples))]

    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else dataset_root / "wavflow_clipping_scan.csv"
    )
    summary_json = (
        Path(args.summary_json)
        if args.summary_json is not None
        else output_csv.with_suffix(".summary.json")
    )

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(
                _process_record,
                dataset_root=str(dataset_root),
                index=index,
                record=record,
                target_rms=float(args.target_rms),
                peak_limit=float(args.peak_limit),
                peak_rescale_min_rms=float(args.peak_rescale_min_rms),
                eps=float(args.eps),
            )
            for index, record in enumerate(records)
        ]
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done == total or done % 250 == 0:
                print(f"progress scanned={done}/{total}", flush=True)

    rows.sort(key=lambda item: int(item.get("index", 0)))
    _write_csv(output_csv, rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") == "error"]
    target_clip_rows = [
        row for row in ok_rows if str(row.get("target_would_clip")).lower() == "true"
    ]
    source_rescale_rows = [
        row
        for row in ok_rows
        if str(row.get("source_peak_rescale_required")).lower() == "true"
    ]
    summary = {
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "output_csv": str(output_csv),
        "records": len(records),
        "ok": len(ok_rows),
        "errors": len(error_rows),
        "target_would_clip": len(target_clip_rows),
        "source_peak_rescale_required": len(source_rescale_rows),
        "target_rms": float(args.target_rms),
        "peak_limit": float(args.peak_limit),
        "peak_rescale_min_rms": float(args.peak_rescale_min_rms),
    }
    _write_json(summary_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
