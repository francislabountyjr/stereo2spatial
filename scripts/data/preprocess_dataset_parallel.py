"""Run dataset preprocessing with track-level parallel workers.

Each worker receives a disjoint QC CSV shard and writes to the same dataset root,
but uses isolated render roots, manifests, and hash databases. After workers
finish, this script merges the worker manifests into the dataset manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_LABELS = {"OK", "FRONTFOCUS"}
DEFAULT_EXTENSIONS = ".wav,.flac,.m4a,.mp3,.ogg,.opus,.aif,.aiff"
DEFAULT_PARALLEL_DIRNAME = "_parallel_preprocess"
DEFAULT_MANIFEST_FILENAME = "manifest.jsonl"
DEFAULT_STREAM_HASH_DB_FILENAME = "_processed_stream_hashes.sqlite3"
DEFAULT_PROGRESS_INTERVAL_SEC = 10.0
STREAM_HASH_RE = re.compile(r"^[0-9a-f]{16,}$")


@dataclass(frozen=True)
class WorkerPlan:
    index: int
    shard_csv: Path
    manifest_file: Path
    stream_hash_file: Path
    failed_hash_file: Path
    render_root: Path
    log_file: Path
    num_rows: int


@dataclass
class WorkerProgress:
    log_offset: int = 0
    rows_started: int = 0
    rows_total: int = 0
    logged_samples_ok: int = 0
    samples_ok: int = 0
    skipped: int = 0
    errors: int = 0


def _is_admin_windows() -> bool | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def _normalize_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _resolve_qc_path(raw_path: str, qc_csv: Path) -> Path:
    candidate = Path(raw_path.strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    from_qc_parent = (qc_csv.parent / candidate).resolve(strict=False)
    if from_qc_parent.exists():
        return from_qc_parent
    return candidate.resolve(strict=False)


def _normalize_extensions(raw: str) -> set[str]:
    extensions: set[str] = set()
    for part in raw.replace(";", ",").split(","):
        clean = part.strip().lower()
        if not clean:
            continue
        if not clean.startswith("."):
            clean = "." + clean
        extensions.add(clean)
    if not extensions:
        raise ValueError("At least one extension is required")
    return extensions


def _coerce_duration(row: dict[str, str]) -> float:
    try:
        return float(row.get("duration_s") or 0.0)
    except ValueError:
        return 0.0


def _load_qc_rows(
    qc_csv: Path,
    *,
    extensions: set[str],
    allowed_labels: set[str],
    min_duration: float | None,
    max_duration: float | None,
) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    if not qc_csv.exists():
        raise FileNotFoundError(f"QC CSV not found: {qc_csv}")

    stats = {
        "rows_total": 0,
        "rows_unique_paths": 0,
        "rows_selected": 0,
        "rows_overwritten": 0,
        "rows_disallowed_label": 0,
        "rows_missing_path": 0,
        "rows_missing_file": 0,
        "rows_bad_extension": 0,
        "rows_duration_filtered": 0,
    }
    latest_by_path: dict[str, dict[str, Any]] = {}

    with qc_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"QC CSV has no header: {qc_csv}")
        fieldnames = list(reader.fieldnames)
        field_lookup = {name.strip().lower(): name for name in fieldnames if name}
        path_field = field_lookup.get("path")
        label_field = field_lookup.get("label")
        if not path_field:
            raise ValueError("QC CSV must include a 'path' column")
        if not label_field:
            raise ValueError("QC CSV must include a 'label' column")

        for row in reader:
            stats["rows_total"] += 1
            raw_path = (row.get(path_field) or "").strip()
            if not raw_path:
                stats["rows_missing_path"] += 1
                continue
            resolved_path = _resolve_qc_path(raw_path, qc_csv)
            path_key = _normalize_path_key(resolved_path)
            if path_key in latest_by_path:
                stats["rows_overwritten"] += 1
            latest_by_path[path_key] = {
                "row": dict(row),
                "path": resolved_path,
                "path_field": path_field,
                "label_field": label_field,
            }

    stats["rows_unique_paths"] = len(latest_by_path)

    selected: list[dict[str, str]] = []
    for path_key in sorted(latest_by_path.keys()):
        item = latest_by_path[path_key]
        row = dict(item["row"])
        path = Path(item["path"])
        path_field = str(item["path_field"])
        label_field = str(item["label_field"])
        label = (row.get(label_field) or "").strip().upper()

        if label not in allowed_labels:
            stats["rows_disallowed_label"] += 1
            continue
        if path.suffix.lower() not in extensions:
            stats["rows_bad_extension"] += 1
            continue
        if not path.exists():
            stats["rows_missing_file"] += 1
            continue
        duration = _coerce_duration(row)
        if min_duration is not None and duration < min_duration:
            stats["rows_duration_filtered"] += 1
            continue
        if max_duration is not None and duration > max_duration:
            stats["rows_duration_filtered"] += 1
            continue

        row[path_field] = str(path)
        row[label_field] = label
        selected.append(row)

    stats["rows_selected"] = len(selected)
    return fieldnames, selected, stats


def _select_rows(
    rows: list[dict[str, str]],
    *,
    max_files: int | None,
    selection: str,
    seed: int,
    start_offset: int,
) -> list[dict[str, str]]:
    if start_offset < 0:
        raise ValueError("start_offset must be >= 0")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be > 0 when provided")
    if start_offset >= len(rows):
        return []

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

    if max_files is not None:
        ordered = ordered[:max_files]
    return [dict(row) for row in ordered]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {error}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def _metadata_artifacts_exist(metadata_path: Path, metadata: dict[str, Any]) -> bool:
    files = metadata.get("files")
    if not isinstance(files, dict):
        return False
    sample_dir = metadata_path.parent
    for value in files.values():
        if not value:
            continue
        candidate = sample_dir / str(value)
        if not candidate.exists():
            return False
    return True


def _iter_hash_sample_dirs(dataset_root: Path) -> list[Path]:
    """Return candidate hash-named sample directories under dataset_root/samples."""
    samples_root = dataset_root / "samples"
    if not samples_root.exists():
        return []
    sample_dirs: list[Path] = []
    for first_level in samples_root.iterdir():
        if not first_level.is_dir():
            continue
        for second_level in first_level.iterdir():
            if not second_level.is_dir():
                continue
            for sample_dir in second_level.iterdir():
                if sample_dir.is_dir() and STREAM_HASH_RE.fullmatch(sample_dir.name.lower()):
                    sample_dirs.append(sample_dir)
    return sample_dirs


def _metadata_source_path_keys(metadata: dict[str, Any]) -> set[str]:
    """Return normalized absolute source-path keys recorded in sample metadata."""
    keys: set[str] = set()
    for key in ("source_path", "source_relpath"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value.strip())
            if path.is_absolute():
                keys.add(_normalize_path_key(path))
    return keys


def _scan_completed_samples(
    *,
    dataset_root: Path,
    cleanup_incomplete: bool,
) -> tuple[dict[str, int], set[str]]:
    """Scan sample dirs, optionally remove incomplete dirs, and return completed sources."""
    samples_root = dataset_root / "samples"
    stats = {
        "sample_dirs": 0,
        "complete_samples": 0,
        "invalid_metadata": 0,
        "incomplete_samples": 0,
        "deleted_incomplete_samples": 0,
        "completed_source_paths": 0,
    }
    completed_source_keys: set[str] = set()
    for sample_dir in _iter_hash_sample_dirs(dataset_root):
        stats["sample_dirs"] += 1
        metadata_path = sample_dir / "metadata.json"
        delete_reason: str | None = None
        if not metadata_path.exists():
            stats["incomplete_samples"] += 1
            delete_reason = "missing_metadata"
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = None
                stats["invalid_metadata"] += 1
                delete_reason = "invalid_metadata"
            if metadata is not None:
                if not isinstance(metadata, dict) or not _metadata_artifacts_exist(
                    metadata_path, metadata
                ):
                    stats["incomplete_samples"] += 1
                    delete_reason = "missing_artifacts"
                else:
                    stats["complete_samples"] += 1
                    completed_source_keys.update(_metadata_source_path_keys(metadata))

        if delete_reason is not None and cleanup_incomplete:
            _safe_rmtree(sample_dir, samples_root)
            stats["deleted_incomplete_samples"] += 1

    stats["completed_source_paths"] = len(completed_source_keys)
    return stats, completed_source_keys


def _manifest_record_from_metadata(
    *,
    dataset_root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "created_utc": metadata.get("created_utc"),
        "stream_hash": metadata.get("stream_hash"),
        "signal_layout": metadata.get("signal_layout"),
        "sample_dir": str(metadata_path.parent.relative_to(dataset_root)),
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
        "sample_artifact_mode": metadata.get("sample_artifact_mode"),
        "dead_channel_indices": metadata.get("dead_channel_indices", []),
        "duplicate_channel_pairs": metadata.get("duplicate_channel_pairs", []),
    }
    if "source_downmix_signal_shape" in metadata:
        record["source_downmix_signal_shape"] = metadata["source_downmix_signal_shape"]
    if "source_mono_signal_shape" in metadata:
        record["source_mono_signal_shape"] = metadata["source_mono_signal_shape"]
    return record


def _manifest_key(record: dict[str, Any]) -> str:
    stream_hash = str(record.get("stream_hash") or "").strip().lower()
    if stream_hash:
        return f"hash:{stream_hash}"
    sample_dir = str(record.get("sample_dir") or "").strip()
    return f"sample:{sample_dir}"


def _merge_manifests(
    *,
    main_manifest: Path,
    worker_manifests: list[Path],
) -> dict[str, int]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    existing_count = 0
    added_count = 0
    duplicate_count = 0

    for record in _read_manifest(main_manifest):
        key = _manifest_key(record)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        existing_count += 1
        merged.append(record)

    for manifest in worker_manifests:
        for record in _read_manifest(manifest):
            key = _manifest_key(record)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            added_count += 1
            merged.append(record)

    main_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = main_manifest.with_suffix(main_manifest.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(tmp_path, main_manifest)
    return {
        "existing_records": existing_count,
        "added_records": added_count,
        "duplicate_records": duplicate_count,
        "total_records": len(merged),
    }


def _repair_manifest_from_sample_metadata(
    *,
    dataset_root: Path,
    main_manifest: Path,
) -> dict[str, int]:
    records = _read_manifest(main_manifest)
    seen = {_manifest_key(record) for record in records}
    scanned = 0
    incomplete = 0
    invalid_metadata = 0
    added = 0

    samples_root = dataset_root / "samples"
    if not samples_root.exists():
        return {
            "scanned_metadata": 0,
            "incomplete_samples": 0,
            "invalid_metadata": 0,
            "added_records": 0,
            "total_records": len(records),
        }

    for metadata_path in samples_root.rglob("metadata.json"):
        scanned += 1
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_metadata += 1
            continue
        if not isinstance(metadata, dict):
            invalid_metadata += 1
            continue
        if not _metadata_artifacts_exist(metadata_path, metadata):
            incomplete += 1
            continue
        record = _manifest_record_from_metadata(
            dataset_root=dataset_root,
            metadata_path=metadata_path,
            metadata=metadata,
        )
        key = _manifest_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        added += 1

    if added > 0:
        tmp_path = main_manifest.with_suffix(main_manifest.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                )
        os.replace(tmp_path, main_manifest)

    return {
        "scanned_metadata": scanned,
        "incomplete_samples": incomplete,
        "invalid_metadata": invalid_metadata,
        "added_records": added,
        "total_records": len(records),
    }


def _refresh_stream_hash_db(dataset_root: Path, manifest: Path) -> dict[str, int]:
    records = _read_manifest(manifest)
    db_path = dataset_root / DEFAULT_STREAM_HASH_DB_FILENAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_hash_index (
                stream_hash TEXT PRIMARY KEY,
                source_path TEXT NOT NULL
            )
            """
        )
        inserted = 0
        for record in records:
            stream_hash = str(record.get("stream_hash") or "").strip().lower()
            if not stream_hash:
                continue
            source_path = str(record.get("source_relpath") or record.get("sample_dir") or "")
            cursor = connection.execute(
                "INSERT OR IGNORE INTO stream_hash_index (stream_hash, source_path) VALUES (?, ?)",
                (stream_hash, source_path),
            )
            inserted += int(cursor.rowcount)
        connection.commit()
        count = connection.execute("SELECT COUNT(*) FROM stream_hash_index").fetchone()[0]
    finally:
        connection.close()
    return {"db_path": str(db_path), "inserted": inserted, "total_hashes": int(count)}


def _filter_completed_source_rows(
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    completed_source_keys: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Remove rows whose absolute path already appears in completed sample metadata."""
    if not rows or not completed_source_keys:
        return rows, {
            "input_rows": len(rows),
            "skipped_completed_sources": 0,
            "remaining_rows": len(rows),
        }
    field_lookup = {name.strip().lower(): name for name in fieldnames if name}
    path_field = field_lookup.get("path")
    if path_field is None:
        return rows, {
            "input_rows": len(rows),
            "skipped_completed_sources": 0,
            "remaining_rows": len(rows),
        }
    filtered: list[dict[str, str]] = []
    skipped = 0
    for row in rows:
        raw_path = (row.get(path_field) or "").strip()
        if raw_path and _normalize_path_key(Path(raw_path)) in completed_source_keys:
            skipped += 1
            continue
        filtered.append(row)
    return filtered, {
        "input_rows": len(rows),
        "skipped_completed_sources": skipped,
        "remaining_rows": len(filtered),
    }


def _count_manifest_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _safe_rmtree(path: Path, required_parent: Path) -> None:
    resolved = path.resolve(strict=False)
    parent = required_parent.resolve(strict=False)
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"Refusing to delete outside expected directory: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _create_worker_plans(
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    workers: int,
    run_root: Path,
) -> list[WorkerPlan]:
    if workers <= 0:
        raise ValueError("workers must be > 0")
    if not rows:
        return []

    actual_workers = min(workers, len(rows))
    shards: list[list[dict[str, str]]] = [[] for _ in range(actual_workers)]
    for row_index, row in enumerate(rows):
        shards[row_index % actual_workers].append(row)

    plans: list[WorkerPlan] = []
    for worker_index, shard_rows in enumerate(shards):
        shard_csv = run_root / "shards" / f"worker_{worker_index:02d}.csv"
        _write_csv(shard_csv, fieldnames, shard_rows)
        plans.append(
            WorkerPlan(
                index=worker_index,
                shard_csv=shard_csv,
                manifest_file=run_root / "manifests" / f"manifest_worker_{worker_index:02d}.jsonl",
                stream_hash_file=run_root / "hashes" / f"stream_worker_{worker_index:02d}.sqlite3",
                failed_hash_file=run_root / "hashes" / f"failed_worker_{worker_index:02d}.sqlite3",
                render_root=run_root / "renders" / f"worker_{worker_index:02d}",
                log_file=run_root / "logs" / f"worker_{worker_index:02d}.log",
                num_rows=len(shard_rows),
            )
        )
    return plans


def _build_worker_command(
    *,
    python_exe: str,
    preprocess_script: Path,
    plan: WorkerPlan,
    dataset_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_exe,
        "-u",
        str(preprocess_script),
        "--qc-csv",
        str(plan.shard_csv),
        "--dataset-root",
        str(dataset_root),
        "--render-root",
        str(plan.render_root),
        "--manifest-file",
        str(plan.manifest_file),
        "--stream-hash-file",
        str(plan.stream_hash_file),
        "--failed-hash-file",
        str(plan.failed_hash_file),
        "--hash-index-backend",
        "sqlite",
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
        "--mono-reduction",
        args.mono_reduction,
        "--dead-channel-threshold",
        str(args.dead_channel_threshold),
        "--sleep-between-renders-sec",
        str(args.sleep_between_renders_sec),
        "--cavernize-timeout-sec",
        str(args.cavernize_timeout_sec),
        "--stream-hash-algorithm",
        args.stream_hash_algorithm,
        "--extensions",
        args.extensions,
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
    if args.keep_renders:
        command.append("--keep-renders")
    if args.retry_failed:
        command.append("--retry-failed")
    for item in args.extra_arg:
        command.append(item)
    return command


_ROW_RE = re.compile(r"^\[(\d+)/(\d+)\]")


def _scan_progress_log(plan: WorkerPlan, progress: WorkerProgress) -> None:
    if plan.log_file.exists():
        with plan.log_file.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(progress.log_offset)
            for line in handle:
                match = _ROW_RE.match(line)
                if match:
                    progress.rows_started = max(
                        progress.rows_started, int(match.group(1))
                    )
                    progress.rows_total = max(progress.rows_total, int(match.group(2)))
                if "  - OK sample " in line:
                    progress.logged_samples_ok += 1
                elif "  - SKIP error " in line:
                    progress.errors += 1
                    progress.skipped += 1
                elif "  - SKIP " in line:
                    progress.skipped += 1
            progress.log_offset = handle.tell()

    manifest_rows = _count_manifest_rows(plan.manifest_file)
    progress.samples_ok = max(progress.logged_samples_ok, manifest_rows)
    progress.rows_started = max(
        progress.rows_started,
        min(plan.num_rows, progress.samples_ok + progress.skipped),
    )


def _print_progress(
    *,
    plans: list[WorkerPlan],
    progress_by_worker: dict[int, WorkerProgress],
    running: int,
    elapsed: float,
) -> None:
    rows_started = sum(item.rows_started for item in progress_by_worker.values())
    rows_total = sum(plan.num_rows for plan in plans)
    samples_ok = sum(item.samples_ok for item in progress_by_worker.values())
    skipped = sum(item.skipped for item in progress_by_worker.values())
    errors = sum(item.errors for item in progress_by_worker.values())
    print(
        "progress "
        f"rows_started={rows_started}/{rows_total} "
        f"ok={samples_ok} skip={skipped} errors={errors} "
        f"workers_running={running} elapsed={elapsed:.0f}s",
        flush=True,
    )


def _run_workers(
    *,
    plans: list[WorkerPlan],
    dataset_root: Path,
    preprocess_script: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    processes: list[tuple[WorkerPlan, subprocess.Popen[str], Any]] = []
    progress_by_worker = {plan.index: WorkerProgress(rows_total=plan.num_rows) for plan in plans}
    start = time.perf_counter()
    for plan in plans:
        command = _build_worker_command(
            python_exe=sys.executable,
            preprocess_script=preprocess_script,
            plan=plan,
            dataset_root=dataset_root,
            args=args,
        )
        plan.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = plan.log_file.open("w", encoding="utf-8", errors="replace")
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
        processes.append((plan, process, log_handle))

    results: list[dict[str, Any]] = []
    pending = set(range(len(processes)))
    next_progress = time.perf_counter() + max(float(args.progress_interval_sec), 0.5)
    try:
        while pending:
            now = time.perf_counter()
            finished: list[int] = []
            for process_index in list(pending):
                plan, process, log_handle = processes[process_index]
                if process.poll() is None:
                    continue
                _scan_progress_log(plan, progress_by_worker[plan.index])
                log_handle.flush()
                log_handle.close()
                results.append(
                    {
                        "worker_index": plan.index,
                        "rows": plan.num_rows,
                        "returncode": int(process.returncode),
                        "manifest_file": str(plan.manifest_file),
                        "samples_written": _count_manifest_rows(plan.manifest_file),
                        "stream_hash_file": str(plan.stream_hash_file),
                        "failed_hash_file": str(plan.failed_hash_file),
                        "render_root": str(plan.render_root),
                        "log_file": str(plan.log_file),
                    }
                )
                finished.append(process_index)
            for process_index in finished:
                pending.remove(process_index)

            if not args.no_progress and (now >= next_progress or not pending):
                for process_index in pending:
                    plan, _, _ = processes[process_index]
                    _scan_progress_log(plan, progress_by_worker[plan.index])
                _print_progress(
                    plans=plans,
                    progress_by_worker=progress_by_worker,
                    running=len(pending),
                    elapsed=now - start,
                )
                next_progress = now + max(float(args.progress_interval_sec), 0.5)

            if pending:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating worker processes ...", flush=True)
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.perf_counter() + 10.0
        for _, process, _ in processes:
            while process.poll() is None and time.perf_counter() < deadline:
                time.sleep(0.2)
            if process.poll() is None:
                process.kill()
        for _, _, log_handle in processes:
            try:
                log_handle.flush()
                log_handle.close()
            except OSError:
                pass
        raise

    elapsed = time.perf_counter() - start
    results.sort(key=lambda item: int(item["worker_index"]))
    return results, elapsed


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/data/preprocess_dataset.py in parallel over disjoint "
            "track-level QC shards."
        )
    )
    parser.add_argument("--qc-csv", default="dataset_paths.csv")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parallel-dirname", default=DEFAULT_PARALLEL_DIRNAME)
    parser.add_argument("--manifest-file", default=DEFAULT_MANIFEST_FILENAME)
    parser.add_argument("--target-output-layout", default="Headphone Virtualizer")
    parser.add_argument("--target-input-layout", default="2.0")
    parser.add_argument("--sample-artifact-mode", choices=["bundle", "split", "flac"], default="bundle")
    parser.add_argument("--signal-dtype", default="float32")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--mono-reduction", choices=["mean", "left", "right"], default="mean")
    parser.add_argument("--skip-source-mono", action="store_true", default=False)
    parser.add_argument("--allow-dead-channels", action="store_true")
    parser.add_argument("--allow-duplicate-channels", action="store_true")
    parser.add_argument("--dead-channel-threshold", type=float, default=1e-6)
    parser.add_argument("--sleep-between-renders-sec", type=float, default=0.0)
    parser.add_argument("--cavernize-timeout-sec", type=float, default=600.0)
    parser.add_argument("--stream-hash-algorithm", default="sha256")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-renders", action="store_true")
    parser.add_argument("--cleanup-worker-renders", action="store_true", default=True)
    parser.add_argument("--keep-worker-renders", dest="cleanup_worker_renders", action="store_false")
    parser.add_argument("--cavernize-exe", default=None)
    parser.add_argument("--ffmpeg-exe", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--selection", choices=["first", "shortest", "random"], default="first")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--progress-interval-sec", type=float, default=DEFAULT_PROGRESS_INTERVAL_SEC)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run startup cleanup/manifest repair/hash refresh, then exit before "
            "creating worker shards."
        ),
    )
    parser.add_argument(
        "--cleanup-incomplete-samples",
        action="store_true",
        default=True,
        help="Delete corrupt or incomplete sample dirs found during startup scan.",
    )
    parser.add_argument(
        "--keep-incomplete-samples",
        dest="cleanup_incomplete_samples",
        action="store_false",
        help="Keep corrupt/incomplete sample dirs instead of deleting them.",
    )
    parser.add_argument(
        "--filter-completed-sources",
        action="store_true",
        default=True,
        help=(
            "Before sharding, skip QC rows whose absolute source path already "
            "appears in completed sample metadata."
        ),
    )
    parser.add_argument(
        "--no-filter-completed-sources",
        dest="filter_completed_sources",
        action="store_false",
        help="Do not filter QC rows by completed sample metadata before sharding.",
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
    dataset_root = Path(args.dataset_root).resolve(strict=False)
    parallel_root = dataset_root / args.parallel_dirname
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = parallel_root / "runs" / run_id
    preprocess_script = repo_root / "scripts" / "data" / "preprocess_dataset.py"
    main_manifest = (
        Path(args.manifest_file)
        if Path(args.manifest_file).is_absolute()
        else dataset_root / args.manifest_file
    )

    extensions = _normalize_extensions(args.extensions)
    dataset_root.mkdir(parents=True, exist_ok=True)

    preflight_scan_stats, completed_source_keys = _scan_completed_samples(
        dataset_root=dataset_root,
        cleanup_incomplete=bool(args.cleanup_incomplete_samples),
    )
    preflight_repair_stats = _repair_manifest_from_sample_metadata(
        dataset_root=dataset_root,
        main_manifest=main_manifest,
    )
    preflight_hash_stats = _refresh_stream_hash_db(dataset_root, main_manifest)

    preflight_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "manifest_file": str(main_manifest),
        "cleanup_incomplete_samples": bool(args.cleanup_incomplete_samples),
        "scan": preflight_scan_stats,
        "manifest_repair": preflight_repair_stats,
        "stream_hash_index": preflight_hash_stats,
    }
    preflight_summary_path = parallel_root / "last_preflight_summary.json"
    _write_summary(preflight_summary_path, preflight_summary)

    print("Preflight dataset resume scan:")
    print(f"  - complete_samples={preflight_scan_stats['complete_samples']}")
    print(f"  - invalid_metadata={preflight_scan_stats['invalid_metadata']}")
    print(f"  - incomplete_samples={preflight_scan_stats['incomplete_samples']}")
    print(
        "  - deleted_incomplete_samples="
        f"{preflight_scan_stats['deleted_incomplete_samples']}"
    )
    print(
        "  - repaired_manifest_records="
        f"{preflight_repair_stats['added_records']}"
    )
    print(f"  - manifest_total={preflight_repair_stats['total_records']}")
    print(f"  - hash_index_total={preflight_hash_stats['total_hashes']}")
    print(f"  - summary={preflight_summary_path}")
    if args.preflight_only:
        print("Preflight-only mode: not running preprocess workers.")
        return

    fieldnames, candidates, qc_stats = _load_qc_rows(
        qc_csv,
        extensions=extensions,
        allowed_labels=DEFAULT_ALLOWED_LABELS,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    rows = _select_rows(
        candidates,
        max_files=args.max_files,
        selection=args.selection,
        seed=args.seed,
        start_offset=args.start_offset,
    )
    filter_stats = {
        "input_rows": len(rows),
        "skipped_completed_sources": 0,
        "remaining_rows": len(rows),
    }
    if args.filter_completed_sources:
        rows, filter_stats = _filter_completed_source_rows(
            rows=rows,
            fieldnames=fieldnames,
            completed_source_keys=completed_source_keys,
        )
    if not rows:
        print("No remaining eligible QC rows selected after preflight filtering.")
        print(f"  - selected_before_filter={filter_stats['input_rows']}")
        print(
            "  - skipped_completed_sources="
            f"{filter_stats['skipped_completed_sources']}"
        )
        return

    run_root.mkdir(parents=True, exist_ok=True)
    selected_csv = run_root / "selected_rows.csv"
    _write_csv(selected_csv, fieldnames, rows)
    plans = _create_worker_plans(
        rows=rows,
        fieldnames=fieldnames,
        workers=int(args.workers),
        run_root=run_root,
    )

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "admin": admin,
        "qc_csv": str(qc_csv),
        "dataset_root": str(dataset_root),
        "parallel_root": str(parallel_root),
        "run_root": str(run_root),
        "run_id": run_id,
        "selected_csv": str(selected_csv),
        "manifest_file": str(main_manifest),
        "workers_requested": int(args.workers),
        "workers_started": len(plans),
        "rows_selected": len(rows),
        "qc_stats": qc_stats,
        "target_output_layout": args.target_output_layout,
        "target_input_layout": args.target_input_layout,
        "skip_source_mono": bool(args.skip_source_mono),
        "plan_only": bool(args.plan_only),
        "preflight": preflight_summary,
        "completed_source_filter": filter_stats,
        "workers": [
            {
                "worker_index": plan.index,
                "rows": plan.num_rows,
                "shard_csv": str(plan.shard_csv),
                "manifest_file": str(plan.manifest_file),
                "log_file": str(plan.log_file),
            }
            for plan in plans
        ],
    }
    summary_path = run_root / "summary.json"
    _write_summary(summary_path, summary)

    print(f"Parallel preprocessing dir: {parallel_root}")
    print(f"Run dir: {run_root}")
    print(f"Selected rows: {len(rows)}")
    print(
        "Skipped completed sources: "
        f"{filter_stats['skipped_completed_sources']}"
    )
    print(f"Workers: {len(plans)}")
    print(f"Selected rows CSV: {selected_csv}")
    print(f"Summary: {summary_path}")
    if args.plan_only:
        print("Plan-only mode: not running preprocess workers.")
        return

    worker_results, elapsed = _run_workers(
        plans=plans,
        dataset_root=dataset_root,
        preprocess_script=preprocess_script,
        args=args,
    )
    ok = all(int(result["returncode"]) == 0 for result in worker_results)
    samples_written = sum(int(result["samples_written"]) for result in worker_results)
    summary["elapsed_seconds"] = elapsed
    summary["samples_written_worker_manifests"] = samples_written
    summary["samples_per_hour_worker_manifests"] = (
        samples_written / elapsed * 3600.0 if elapsed > 0 else 0.0
    )
    summary["ok"] = ok
    summary["worker_results"] = worker_results

    if not ok:
        _write_summary(summary_path, summary)
        print("\nOne or more workers failed. Not merging manifests.")
        for result in worker_results:
            if int(result["returncode"]) != 0:
                print(
                    f"  - worker={result['worker_index']} "
                    f"returncode={result['returncode']} log={result['log_file']}"
                )
        raise SystemExit(1)

    merge_stats = _merge_manifests(
        main_manifest=main_manifest,
        worker_manifests=[plan.manifest_file for plan in plans],
    )
    repair_stats = _repair_manifest_from_sample_metadata(
        dataset_root=dataset_root,
        main_manifest=main_manifest,
    )
    hash_stats = _refresh_stream_hash_db(dataset_root, main_manifest)
    summary["manifest_merge"] = merge_stats
    summary["manifest_repair"] = repair_stats
    summary["stream_hash_index"] = hash_stats

    if args.cleanup_worker_renders and not args.keep_renders:
        renders_root = run_root / "renders"
        if renders_root.exists():
            _safe_rmtree(renders_root, run_root)
            summary["worker_renders_deleted"] = True
    else:
        summary["worker_renders_deleted"] = False

    _write_summary(summary_path, summary)

    print("\nParallel preprocessing summary:")
    print(f"  - ok={ok}")
    print(f"  - elapsed={elapsed:.1f}s")
    print(f"  - worker_manifest_samples={samples_written}")
    print(f"  - throughput={summary['samples_per_hour_worker_manifests']:.2f} samples/hour")
    print(f"  - manifest_added={merge_stats['added_records']}")
    print(f"  - repaired_existing_samples={repair_stats['added_records']}")
    print(f"  - manifest_total={repair_stats['total_records']}")
    print(f"  - duplicate_manifest_records={merge_stats['duplicate_records']}")
    print(f"  - manifest={main_manifest}")
    print(f"  - summary={summary_path}")


if __name__ == "__main__":
    main()
