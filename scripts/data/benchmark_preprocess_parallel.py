"""Benchmark preprocessing throughput for independent track-level workers.

This script is intentionally separate from ``preprocess_dataset.py``. It creates
small QC CSV shards, runs the existing preprocessor in parallel over disposable
dataset/render roots, records timing, and deletes large temporary outputs by
default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_WORKERS = "1,2,4,6,8"
DEFAULT_ALLOWED_LABELS = {"OK", "FRONTFOCUS"}
DEFAULT_BENCH_ROOT = "_parallel_preprocess_bench"


@dataclass(frozen=True)
class WorkerRun:
    index: int
    shard_csv: Path
    dataset_root: Path
    render_root: Path
    log_file: Path
    num_rows: int


def _is_admin_windows() -> bool | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def _parse_worker_levels(raw: str) -> list[int]:
    levels: list[int] = []
    for part in raw.replace(",", " ").split():
        clean = part.strip()
        if not clean:
            continue
        value = int(clean)
        if value <= 0:
            raise ValueError("worker levels must be positive integers")
        if value not in levels:
            levels.append(value)
    if not levels:
        raise ValueError("At least one worker level is required")
    return levels


def _coerce_duration(row: dict[str, str]) -> float:
    try:
        return float(row.get("duration_s") or 0.0)
    except ValueError:
        return 0.0


def _load_candidate_rows(
    qc_csv: Path,
    *,
    allowed_labels: set[str],
    min_duration: float | None,
    max_duration: float | None,
) -> tuple[list[str], list[dict[str, str]]]:
    with qc_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"QC CSV has no header: {qc_csv}")
        fieldnames = list(reader.fieldnames)
        if "path" not in fieldnames:
            raise ValueError("QC CSV must include a 'path' column")

        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            if label not in allowed_labels:
                continue
            path = Path(row.get("path", "").strip()).resolve(strict=False)
            if not path.exists():
                continue
            duration = _coerce_duration(row)
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            key = os.path.normcase(os.path.abspath(str(path)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))
    return fieldnames, rows


def _select_rows(
    rows: list[dict[str, str]],
    *,
    samples: int,
    selection: str,
    seed: int,
    start_offset: int,
) -> list[dict[str, str]]:
    if start_offset < 0:
        raise ValueError("start_offset must be >= 0")
    if samples <= 0:
        raise ValueError("samples must be > 0")
    if start_offset >= len(rows):
        raise ValueError(
            f"start_offset={start_offset} leaves no candidate rows ({len(rows)} total)"
        )

    available = rows[start_offset:]
    selection = selection.strip().lower()
    if selection == "first":
        ordered = available
    elif selection == "shortest":
        ordered = sorted(available, key=_coerce_duration)
    elif selection == "random":
        ordered = list(available)
        random.Random(seed).shuffle(ordered)
    else:
        raise ValueError("selection must be one of: first, shortest, random")

    if len(ordered) < samples:
        raise ValueError(f"Need {samples} rows, only found {len(ordered)}")
    return [dict(row) for row in ordered[:samples]]


def _write_shard(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _count_manifest_rows(dataset_root: Path) -> int:
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.exists():
        return 0
    with manifest.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _safe_rmtree(path: Path, bench_root: Path) -> None:
    resolved = path.resolve(strict=False)
    root = bench_root.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to delete outside benchmark root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _build_worker_command(
    *,
    python_exe: str,
    preprocess_script: Path,
    worker: WorkerRun,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_exe,
        str(preprocess_script),
        "--qc-csv",
        str(worker.shard_csv),
        "--dataset-root",
        str(worker.dataset_root),
        "--render-root",
        str(worker.render_root),
        "--target-output-layout",
        args.target_output_layout,
        "--target-input-layout",
        args.target_input_layout,
        "--sample-artifact-mode",
        args.sample_artifact_mode,
        "--sleep-between-renders-sec",
        str(args.sleep_between_renders_sec),
        "--cavernize-timeout-sec",
        str(args.cavernize_timeout_sec),
        "--hash-index-backend",
        "sqlite",
        "--signal-dtype",
        args.signal_dtype,
    ]
    if args.cavernize_exe:
        command += ["--cavernize-exe", args.cavernize_exe]
    if args.ffmpeg_exe:
        command += ["--ffmpeg-exe", args.ffmpeg_exe]
    if args.skip_source_mono:
        command.append("--skip-source-mono")
    if args.allow_dead_channels:
        command.append("--allow-dead-channels")
    if args.allow_duplicate_channels:
        command.append("--allow-duplicate-channels")
    for item in args.extra_arg:
        command.append(item)
    return command


def _run_level(
    *,
    level: int,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    bench_root: Path,
    results_dir: Path,
    preprocess_script: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    level_root = bench_root / f"workers_{level}"
    shards_dir = results_dir / f"workers_{level}_shards"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if level_root.exists():
        _safe_rmtree(level_root, bench_root)
    level_root.mkdir(parents=True, exist_ok=True)

    shards: list[list[dict[str, str]]] = [[] for _ in range(level)]
    for idx, row in enumerate(rows):
        shards[idx % level].append(row)

    workers: list[WorkerRun] = []
    for worker_idx, shard_rows in enumerate(shards):
        shard_csv = shards_dir / f"worker_{worker_idx:02d}.csv"
        _write_shard(shard_csv, fieldnames, shard_rows)
        worker_root = level_root / f"worker_{worker_idx:02d}"
        workers.append(
            WorkerRun(
                index=worker_idx,
                shard_csv=shard_csv,
                dataset_root=worker_root / "dataset",
                render_root=worker_root / "renders",
                log_file=logs_dir / f"workers_{level}_worker_{worker_idx:02d}.log",
                num_rows=len(shard_rows),
            )
        )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    start = time.perf_counter()
    processes: list[tuple[WorkerRun, subprocess.Popen[str], Any]] = []
    for worker in workers:
        command = _build_worker_command(
            python_exe=sys.executable,
            preprocess_script=preprocess_script,
            worker=worker,
            args=args,
        )
        worker.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = worker.log_file.open("w", encoding="utf-8", errors="replace")
        log_handle.write("COMMAND:\n")
        log_handle.write(json.dumps(command, ensure_ascii=True) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path.cwd()),
            env=env,
        )
        processes.append((worker, process, log_handle))

    worker_results: list[dict[str, Any]] = []
    for worker, process, log_handle in processes:
        returncode = process.wait()
        log_handle.flush()
        log_handle.close()
        worker_results.append(
            {
                "worker_index": worker.index,
                "rows": worker.num_rows,
                "returncode": returncode,
                "samples_written": _count_manifest_rows(worker.dataset_root),
                "dataset_root": str(worker.dataset_root),
                "render_root": str(worker.render_root),
                "log_file": str(worker.log_file),
            }
        )

    elapsed = time.perf_counter() - start
    samples_written = sum(int(item["samples_written"]) for item in worker_results)
    ok = all(int(item["returncode"]) == 0 for item in worker_results)
    result = {
        "workers": level,
        "rows_requested": len(rows),
        "samples_written": samples_written,
        "elapsed_seconds": elapsed,
        "samples_per_hour": (samples_written / elapsed * 3600.0) if elapsed > 0 else 0.0,
        "ok": ok,
        "workers_detail": worker_results,
    }

    if ok and not args.keep_workdirs:
        _safe_rmtree(level_root, bench_root)
        result["workdirs_deleted"] = True
    else:
        result["workdirs_deleted"] = False
    return result


def _write_summary_files(results_dir: Path, summary: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "workers",
                "rows_requested",
                "samples_written",
                "elapsed_seconds",
                "samples_per_hour",
                "ok",
                "workdirs_deleted",
            ],
        )
        writer.writeheader()
        for result in summary["results"]:
            writer.writerow(
                {
                    key: result.get(key)
                    for key in [
                        "workers",
                        "rows_requested",
                        "samples_written",
                        "elapsed_seconds",
                        "samples_per_hour",
                        "ok",
                        "workdirs_deleted",
                    ]
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark track-level parallelism for scripts/data/preprocess_dataset.py "
            "using disposable dataset roots."
        )
    )
    parser.add_argument("--qc-csv", default="dataset_paths.csv")
    parser.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", default=DEFAULT_WORKERS)
    parser.add_argument(
        "--selection",
        choices=["first", "shortest", "random"],
        default="first",
        help="Which eligible QC rows to benchmark.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--target-output-layout", default="Headphone Virtualizer")
    parser.add_argument("--target-input-layout", default="2.0")
    parser.add_argument("--sample-artifact-mode", choices=["bundle", "split", "flac"], default="bundle")
    parser.add_argument("--signal-dtype", default="float32")
    parser.add_argument("--sleep-between-renders-sec", type=float, default=0.0)
    parser.add_argument("--cavernize-timeout-sec", type=float, default=600.0)
    parser.add_argument("--skip-source-mono", action="store_true", default=True)
    parser.add_argument("--save-source-mono", dest="skip_source_mono", action="store_false")
    parser.add_argument("--allow-dead-channels", action="store_true")
    parser.add_argument("--allow-duplicate-channels", action="store_true")
    parser.add_argument("--cavernize-exe", default=None)
    parser.add_argument("--ffmpeg-exe", default=None)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Create selected_rows.csv and summary skeleton without running preprocess workers.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra single argument passed through to preprocess_dataset.py. Repeat as needed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    admin = _is_admin_windows()
    if admin is False:
        print(
            "WARNING: this process is not elevated. If Cavernize requires admin, "
            "rerun from an Administrator PowerShell."
        )

    qc_csv = Path(args.qc_csv).resolve(strict=False)
    bench_root = Path(args.bench_root).resolve(strict=False)
    results_dir = bench_root / "results" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    preprocess_script = repo_root / "scripts" / "data" / "preprocess_dataset.py"
    worker_levels = _parse_worker_levels(args.workers)

    fieldnames, candidates = _load_candidate_rows(
        qc_csv,
        allowed_labels=DEFAULT_ALLOWED_LABELS,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    rows = _select_rows(
        candidates,
        samples=int(args.samples),
        selection=str(args.selection),
        seed=int(args.seed),
        start_offset=int(args.start_offset),
    )

    bench_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = results_dir / "selected_rows.csv"
    _write_shard(selected_csv, fieldnames, rows)

    durations = [_coerce_duration(row) for row in rows]
    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "admin": admin,
        "qc_csv": str(qc_csv),
        "bench_root": str(bench_root),
        "selected_csv": str(selected_csv),
        "selection": args.selection,
        "samples": int(args.samples),
        "durations_seconds": durations,
        "duration_total_seconds": sum(durations),
        "worker_levels": worker_levels,
        "target_output_layout": args.target_output_layout,
        "target_input_layout": args.target_input_layout,
        "skip_source_mono": bool(args.skip_source_mono),
        "results": [],
    }
    _write_summary_files(results_dir, summary)

    print(f"Benchmark results dir: {results_dir}")
    print(f"Selected rows: {selected_csv}")
    print(
        "Selected duration total: "
        f"{sum(durations):.1f}s across {len(rows)} samples"
    )
    if args.plan_only:
        print("Plan-only mode: not running preprocess workers.")
        print(results_dir / "summary.json")
        return

    for level in worker_levels:
        print(f"\n=== Running worker level: {level} ===")
        result = _run_level(
            level=level,
            rows=rows,
            fieldnames=fieldnames,
            bench_root=bench_root,
            results_dir=results_dir,
            preprocess_script=preprocess_script,
            args=args,
        )
        summary["results"].append(result)
        _write_summary_files(results_dir, summary)
        print(
            f"workers={level} ok={result['ok']} "
            f"samples={result['samples_written']}/{result['rows_requested']} "
            f"elapsed={result['elapsed_seconds']:.1f}s "
            f"throughput={result['samples_per_hour']:.2f} samples/hour"
        )

    print("\nSummary:")
    print(results_dir / "summary.csv")
    print(results_dir / "summary.json")


if __name__ == "__main__":
    main()
