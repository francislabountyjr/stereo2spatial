#!/usr/bin/env python3
"""Repair dataset samples whose saved signal tensors contain NaN or Inf values."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from find_invalid_samples import (
    InvalidTensor,
    _iter_sample_dirs,
    _prune_manifest,
    _scan_sample,
    _write_report,
)
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass(frozen=True)
class RepairCandidate:
    sample_dir: Path
    stream_hash: str
    source_path: Path


def _sample_dir_from_stream_hash(dataset_root: Path, stream_hash: str) -> Path:
    clean = stream_hash.strip().lower()
    return dataset_root / "samples" / clean[:2] / clean[2:4] / clean


def _scan_invalid(
    *,
    dataset_root: Path,
    workers: int,
    log_every: int,
    max_samples: int | None,
) -> list[InvalidTensor]:
    try:
        sample_dirs = _iter_sample_dirs(dataset_root)
    except FileNotFoundError:
        sample_dirs = []
    if max_samples is not None:
        sample_dirs = sample_dirs[: max(0, int(max_samples))]
    print(f"Scanning samples: {len(sample_dirs)}")

    invalid: list[InvalidTensor] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_scan_sample, str(sample_dir)): sample_dir
            for sample_dir in sample_dirs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            invalid.extend(future.result())
            if log_every > 0 and index % int(log_every) == 0:
                print(f"progress scanned={index}/{len(sample_dirs)} invalid={len(invalid)}")
    return invalid


def _repair_candidates(rows: list[InvalidTensor]) -> tuple[list[RepairCandidate], list[InvalidTensor]]:
    by_dir: dict[Path, RepairCandidate] = {}
    missing: list[InvalidTensor] = []
    for row in rows:
        source_text = row.source_path.strip()
        source_path = Path(source_text) if source_text else Path()
        if not source_text or not source_path.is_absolute() or not source_path.exists():
            missing.append(row)
            continue
        sample_dir = Path(row.sample_dir)
        by_dir[sample_dir] = RepairCandidate(
            sample_dir=sample_dir,
            stream_hash=row.stream_hash.strip().lower() or sample_dir.name.lower(),
            source_path=source_path,
        )
    return sorted(by_dir.values(), key=lambda item: str(item.sample_dir)), missing


def _write_repair_qc_csv(path: Path, candidates: list[RepairCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "duration_s"])
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "path": str(candidate.source_path),
                    "label": "OK",
                    "duration_s": "",
                }
            )


def _read_conversion_repair_candidates(
    *,
    dataset_root: Path,
    repair_csv: Path,
) -> tuple[list[RepairCandidate], int]:
    """Read converter-produced failed_conversion_repair_rows.csv if present."""
    if not repair_csv.exists():
        return [], 0
    candidates: list[RepairCandidate] = []
    skipped = 0
    with repair_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_text = (row.get("path") or "").strip()
            stream_hash = (row.get("stream_hash") or "").strip().lower()
            sample_dir_text = (row.get("sample_dir") or "").strip()
            source_path = Path(source_text) if source_text else Path()
            if not source_text or not source_path.is_absolute() or not source_path.exists():
                skipped += 1
                continue
            if stream_hash:
                sample_dir = _sample_dir_from_stream_hash(dataset_root, stream_hash)
            elif sample_dir_text:
                sample_dir = dataset_root / Path(sample_dir_text)
                stream_hash = sample_dir.name.lower()
            else:
                skipped += 1
                continue
            candidates.append(
                RepairCandidate(
                    sample_dir=sample_dir,
                    stream_hash=stream_hash,
                    source_path=source_path,
                )
            )
    return candidates, skipped


def _dedupe_candidates(candidates: list[RepairCandidate]) -> list[RepairCandidate]:
    by_source: dict[str, RepairCandidate] = {}
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate.source_path)))
        by_source[key] = candidate
    return sorted(by_source.values(), key=lambda item: str(item.source_path))


def _safe_delete_sample_dirs(dataset_root: Path, candidates: list[RepairCandidate]) -> int:
    samples_root = dataset_root / "samples"
    deleted = 0
    for candidate in candidates:
        sample_dir = candidate.sample_dir.resolve(strict=False)
        try:
            relative = sample_dir.relative_to(samples_root.resolve(strict=False))
        except ValueError as error:
            raise RuntimeError(f"Refusing to delete outside samples root: {sample_dir}") from error
        if str(relative) in {"", "."}:
            raise RuntimeError("Refusing to delete samples root.")
        if sample_dir.exists():
            import shutil

            shutil.rmtree(sample_dir)
            deleted += 1
    return deleted


def _build_preprocess_command(
    *,
    args: argparse.Namespace,
    repair_csv: Path,
    dataset_root: Path,
) -> list[str]:
    script = Path(__file__).resolve().parent / "preprocess_dataset_parallel.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--qc-csv",
        str(repair_csv),
        "--dataset-root",
        str(dataset_root),
        "--workers",
        str(args.render_workers),
        "--target-output-layout",
        args.target_output_layout,
        "--target-input-layout",
        args.target_input_layout,
        "--sample-artifact-mode",
        args.sample_artifact_mode,
        "--signal-dtype",
        args.signal_dtype,
        "--sample-rate",
        str(args.sample_rate),
        "--cavernize-timeout-sec",
        str(args.cavernize_timeout_sec),
        "--stream-hash-algorithm",
        args.stream_hash_algorithm,
        "--extensions",
        args.extensions,
        "--retry-failed",
    ]
    if args.skip_source_mono:
        command.append("--skip-source-mono")
    if args.allow_dead_channels:
        command.append("--allow-dead-channels")
    if args.allow_duplicate_channels:
        command.append("--allow-duplicate-channels")
    if args.keep_renders:
        command.append("--keep-renders")
    if args.cavernize_exe:
        command += ["--cavernize-exe", args.cavernize_exe]
    if args.ffmpeg_exe:
        command += ["--ffmpeg-exe", args.ffmpeg_exe]
    for item in args.extra_arg:
        command += ["--extra-arg", item]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a waveform dataset for non-finite signal tensors and repair "
            "invalid samples by rerendering their source files."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workers", type=int, default=4, help="Scanner workers.")
    parser.add_argument(
        "--render-workers",
        type=int,
        default=None,
        help="Parallel render workers. Defaults to --workers.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--report-csv", default=None)
    parser.add_argument("--repair-csv", default=None)
    parser.add_argument(
        "--conversion-repair-csv",
        default=None,
        help=(
            "Optional converter-produced repair queue. Defaults to "
            "<dataset-root>/failed_conversion_repair_rows.csv when present."
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-missing-sources", action="store_true")
    parser.add_argument("--target-output-layout", default="Headphone Virtualizer")
    parser.add_argument("--target-input-layout", default="2.0")
    parser.add_argument("--sample-artifact-mode", choices=["bundle", "split", "flac"], default="bundle")
    parser.add_argument("--signal-dtype", default="float32")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--skip-source-mono", action="store_true", default=False)
    parser.add_argument("--allow-dead-channels", action="store_true")
    parser.add_argument("--allow-duplicate-channels", action="store_true")
    parser.add_argument("--keep-renders", action="store_true")
    parser.add_argument("--cavernize-timeout-sec", type=float, default=600.0)
    parser.add_argument("--stream-hash-algorithm", default="sha256")
    parser.add_argument(
        "--extensions",
        default=".wav,.flac,.m4a,.mp3,.ogg,.opus,.aif,.aiff",
    )
    parser.add_argument("--cavernize-exe", default=None)
    parser.add_argument("--ffmpeg-exe", default=None)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra single argument passed through to preprocess_dataset_parallel.py.",
    )
    args = parser.parse_args()
    if args.render_workers is None:
        args.render_workers = args.workers
    return args


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve(strict=False)
    repair_root = dataset_root / "_invalid_sample_repair"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = repair_root / "runs" / run_id
    report_csv = (
        Path(args.report_csv).resolve(strict=False)
        if args.report_csv
        else run_root / "invalid_samples.csv"
    )
    repair_csv = (
        Path(args.repair_csv).resolve(strict=False)
        if args.repair_csv
        else run_root / "repair_rows.csv"
    )
    conversion_repair_csv = (
        Path(args.conversion_repair_csv).resolve(strict=False)
        if args.conversion_repair_csv
        else dataset_root / "failed_conversion_repair_rows.csv"
    )

    invalid = _scan_invalid(
        dataset_root=dataset_root,
        workers=int(args.workers),
        log_every=int(args.log_every),
        max_samples=args.max_samples,
    )
    _write_report(report_csv, invalid)
    candidates, missing = _repair_candidates(invalid)
    queued_candidates, queued_missing = _read_conversion_repair_candidates(
        dataset_root=dataset_root,
        repair_csv=conversion_repair_csv,
    )
    candidates = _dedupe_candidates([*candidates, *queued_candidates])

    print(f"INVALID TENSORS: {len(invalid)}")
    print(f"INVALID/QUEUED SAMPLE DIRS REPAIRABLE: {len(candidates)}")
    print(f"INVALID ROWS WITH MISSING SOURCES: {len(missing)}")
    print(f"CONVERSION QUEUE: {conversion_repair_csv if conversion_repair_csv.exists() else 'none'}")
    print(f"CONVERSION QUEUE ROWS REPAIRABLE: {len(queued_candidates)}")
    print(f"CONVERSION QUEUE ROWS MISSING SOURCES: {queued_missing}")
    print(f"report_csv={report_csv}")

    if missing and not args.allow_missing_sources:
        for row in missing[:20]:
            print(f"  missing-source: {row.sample_dir} source={row.source_path!r}")
        raise RuntimeError(
            "Some invalid samples do not have an existing absolute source_path. "
            "Pass --allow-missing-sources to repair the others anyway."
        )

    if not candidates:
        print("No repairable invalid samples found.")
        return

    _write_repair_qc_csv(repair_csv, candidates)
    print(f"repair_csv={repair_csv}")
    if not args.apply:
        print("DRY RUN ONLY: pass --apply to delete invalid samples and rerender them.")
        return

    deleted = _safe_delete_sample_dirs(dataset_root, candidates)
    hashes = {candidate.stream_hash for candidate in candidates if candidate.stream_hash}
    manifest_stats = _prune_manifest(dataset_root, hashes)
    print(f"deleted_invalid_sample_dirs={deleted}")
    print(f"manifest_prune={manifest_stats}")

    command = _build_preprocess_command(
        args=args,
        repair_csv=repair_csv,
        dataset_root=dataset_root,
    )
    print("Launching repair preprocess:")
    print("  " + " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Repair preprocess failed with returncode={completed.returncode}. "
            f"repair_csv={repair_csv}"
        )

    if conversion_repair_csv.exists() and queued_candidates:
        consumed = conversion_repair_csv.with_suffix(
            conversion_repair_csv.suffix + ".consumed"
        )
        try:
            conversion_repair_csv.replace(consumed)
            print(f"consumed_conversion_repair_csv={consumed}")
        except OSError:
            pass

    print("Repair preprocess completed. Re-scan recommended before resuming training.")


if __name__ == "__main__":
    main()
