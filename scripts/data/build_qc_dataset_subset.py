#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.jsonl"
TARGET_LATENT_FILENAME = "target_latent.pt"
SOURCE_STEREO_LATENT_FILENAME = "source_stereo_latent.pt"
SOURCE_MONO_LATENT_FILENAME = "source_mono_latent.pt"
SOURCE_DOWNMIX_LATENT_FILENAME = "source_downmix_latent.pt"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"


@dataclass(frozen=True)
class SampleRecord:
    stream_hash: str
    sample_dir: Path
    metadata: dict[str, Any]


def strip_optional_quotes(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def normalize_path_key(path: str | Path) -> str:
    raw = str(path).strip()
    return os.path.normcase(os.path.abspath(raw))


def resolve_path_from_text(raw_path: str, reference_dir: Path) -> Path:
    cleaned = strip_optional_quotes(raw_path)
    candidate = Path(cleaned).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (reference_dir / candidate).resolve(strict=False)


def path_is_under_prefix(path_key: str, prefix_key: str) -> bool:
    prefix = prefix_key if prefix_key.endswith(os.sep) else prefix_key + os.sep
    return path_key == prefix_key or path_key.startswith(prefix)


def load_path_list(txt_path: Path) -> list[Path]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Path-list file not found: {txt_path}")

    items: list[Path] = []
    seen: set[str] = set()
    with open(txt_path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            resolved = resolve_path_from_text(stripped, txt_path.parent)
            key = normalize_path_key(resolved)
            if key in seen:
                continue
            seen.add(key)
            items.append(resolved)
    return items


def write_csv_atomic(
    csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def write_manifest_atomic(manifest_path: Path, records: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    os.replace(tmp_path, manifest_path)


def filter_qc_rows(
    input_qc_csv: Path,
    keep_prefix_keys: list[str],
    avoid_keys: set[str],
) -> tuple[list[str], list[dict[str, str]], list[str], dict[str, int]]:
    with open(input_qc_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"QC CSV has no header row: {input_qc_csv}")

        lookup = {
            field.strip().lower(): field for field in fieldnames if field is not None
        }
        path_field = lookup.get("path")
        if not path_field:
            raise ValueError(f"QC CSV missing required column 'path': {input_qc_csv}")

        selected_rows: list[dict[str, str]] = []
        selected_keys_ordered: list[str] = []
        selected_key_seen: set[str] = set()
        stats = {
            "rows_total": 0,
            "rows_selected": 0,
            "rows_missing_path": 0,
            "rows_avoided": 0,
            "rows_prefix_miss": 0,
        }

        for row in reader:
            stats["rows_total"] += 1
            raw_path = (row.get(path_field) or "").strip()
            if not raw_path:
                stats["rows_missing_path"] += 1
                continue

            resolved = resolve_path_from_text(raw_path, input_qc_csv.parent)
            path_key = normalize_path_key(resolved)
            if path_key in avoid_keys:
                stats["rows_avoided"] += 1
                continue
            if not any(
                path_is_under_prefix(path_key, keep_key) for keep_key in keep_prefix_keys
            ):
                stats["rows_prefix_miss"] += 1
                continue

            selected_rows.append(row)
            stats["rows_selected"] += 1
            if path_key not in selected_key_seen:
                selected_key_seen.add(path_key)
                selected_keys_ordered.append(path_key)

    return fieldnames, selected_rows, selected_keys_ordered, stats


def split_sample_artifacts_exist(sample_dir: Path) -> bool:
    required = [
        sample_dir / TARGET_LATENT_FILENAME,
        sample_dir / SOURCE_STEREO_LATENT_FILENAME,
        sample_dir / SOURCE_MONO_LATENT_FILENAME,
        sample_dir / SOURCE_DOWNMIX_LATENT_FILENAME,
        sample_dir / METADATA_FILENAME,
    ]
    return all(path.exists() for path in required)


def bundled_sample_artifacts_exist(sample_dir: Path) -> bool:
    required = [
        sample_dir / SAMPLE_BUNDLE_FILENAME,
        sample_dir / METADATA_FILENAME,
    ]
    return all(path.exists() for path in required)


def metadata_declared_files_exist(sample_dir: Path, metadata: dict[str, Any]) -> bool:
    files = metadata.get("files")
    if not isinstance(files, dict) or not files:
        return False

    for value in files.values():
        if not isinstance(value, str) or not value.strip():
            return False
        if not (sample_dir / value).exists():
            return False
    return True


def sample_is_complete(sample_dir: Path, metadata: dict[str, Any]) -> bool:
    if metadata_declared_files_exist(sample_dir, metadata):
        return True
    return split_sample_artifacts_exist(sample_dir) or bundled_sample_artifacts_exist(
        sample_dir
    )


def destination_sample_is_complete(sample_dir: Path) -> bool:
    metadata_path = sample_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    return sample_is_complete(sample_dir, metadata)


def metadata_source_key(metadata: dict[str, Any]) -> str | None:
    source_path = metadata.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        return normalize_path_key(source_path.strip())

    source_relpath = metadata.get("source_relpath")
    if isinstance(source_relpath, str) and source_relpath.strip():
        candidate = Path(source_relpath.strip())
        if candidate.is_absolute():
            return normalize_path_key(candidate)
    return None


def build_source_index_for_selected_keys(
    input_dataset_root: Path,
    selected_keys: set[str],
) -> tuple[dict[str, list[SampleRecord]], dict[str, int]]:
    samples_root = input_dataset_root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"Dataset samples directory not found: {samples_root}")

    source_index: dict[str, list[SampleRecord]] = {}
    stats = {
        "metadata_files_scanned": 0,
        "metadata_invalid": 0,
        "metadata_missing_source": 0,
        "samples_incomplete": 0,
        "samples_indexed": 0,
    }

    for metadata_path in samples_root.rglob(METADATA_FILENAME):
        stats["metadata_files_scanned"] += 1
        sample_dir = metadata_path.parent

        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            stats["metadata_invalid"] += 1
            continue

        if not isinstance(metadata, dict):
            stats["metadata_invalid"] += 1
            continue
        if not sample_is_complete(sample_dir, metadata):
            stats["samples_incomplete"] += 1
            continue

        source_key = metadata_source_key(metadata)
        if not source_key:
            stats["metadata_missing_source"] += 1
            continue
        if source_key not in selected_keys:
            continue

        stream_hash_raw = metadata.get("stream_hash")
        stream_hash = (
            str(stream_hash_raw).strip().lower()
            if isinstance(stream_hash_raw, str) and stream_hash_raw.strip()
            else sample_dir.name
        )
        source_index.setdefault(source_key, []).append(
            SampleRecord(
                stream_hash=stream_hash,
                sample_dir=sample_dir,
                metadata=metadata,
            )
        )
        stats["samples_indexed"] += 1

    for records in source_index.values():
        records.sort(key=lambda record: record.stream_hash)

    return source_index, stats


def destination_sample_dir(
    input_dataset_root: Path,
    output_dataset_root: Path,
    record: SampleRecord,
) -> Path:
    try:
        relative = record.sample_dir.relative_to(input_dataset_root)
    except ValueError:
        relative = (
            Path("samples")
            / record.stream_hash[:2]
            / record.stream_hash[2:4]
            / record.stream_hash
        )
    return output_dataset_root / relative


def select_and_copy_samples(
    selected_keys_ordered: list[str],
    source_index: dict[str, list[SampleRecord]],
    input_dataset_root: Path,
    output_dataset_root: Path,
    overwrite_existing: bool,
    dry_run: bool,
) -> tuple[list[SampleRecord], dict[str, int]]:
    chosen: list[SampleRecord] = []
    chosen_hashes: set[str] = set()

    stats = {
        "selected_source_keys": len(selected_keys_ordered),
        "source_keys_with_no_sample": 0,
        "sample_records_selected": 0,
        "sample_records_copied": 0,
        "sample_records_skipped_existing": 0,
    }

    for source_key in selected_keys_ordered:
        candidates = source_index.get(source_key)
        if not candidates:
            stats["source_keys_with_no_sample"] += 1
            continue

        for record in candidates:
            if record.stream_hash in chosen_hashes:
                continue
            chosen_hashes.add(record.stream_hash)
            chosen.append(record)
            stats["sample_records_selected"] += 1

            destination = destination_sample_dir(
                input_dataset_root=input_dataset_root,
                output_dataset_root=output_dataset_root,
                record=record,
            )
            if destination.exists():
                if not overwrite_existing and destination_sample_is_complete(destination):
                    stats["sample_records_skipped_existing"] += 1
                    continue
                if not dry_run:
                    shutil.rmtree(destination)

            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(record.sample_dir, destination)
            stats["sample_records_copied"] += 1

    return chosen, stats


def filter_ready_records_for_manifest(
    selected_records: list[SampleRecord],
    input_dataset_root: Path,
    output_dataset_root: Path,
) -> tuple[list[SampleRecord], dict[str, int]]:
    ready: list[SampleRecord] = []
    stats = {"records_checked": 0, "records_ready": 0, "records_dropped": 0}

    for record in selected_records:
        stats["records_checked"] += 1
        destination = destination_sample_dir(
            input_dataset_root=input_dataset_root,
            output_dataset_root=output_dataset_root,
            record=record,
        )
        if destination_sample_is_complete(destination):
            ready.append(record)
            stats["records_ready"] += 1
        else:
            stats["records_dropped"] += 1

    return ready, stats


def load_manifest_map(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    entries: dict[str, dict[str, Any]] = {}
    stats = {"manifest_lines": 0, "manifest_bad_lines": 0, "manifest_loaded": 0}

    if not manifest_path.exists():
        return entries, stats

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["manifest_lines"] += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                stats["manifest_bad_lines"] += 1
                continue
            if not isinstance(payload, dict):
                stats["manifest_bad_lines"] += 1
                continue
            stream_hash_raw = payload.get("stream_hash")
            if not isinstance(stream_hash_raw, str) or not stream_hash_raw.strip():
                stats["manifest_bad_lines"] += 1
                continue
            stream_hash = stream_hash_raw.strip().lower()
            if stream_hash in entries:
                continue
            entries[stream_hash] = payload
            stats["manifest_loaded"] += 1

    return entries, stats


def build_fallback_manifest_record(
    record: SampleRecord, input_dataset_root: Path
) -> dict[str, Any]:
    metadata = record.metadata
    try:
        sample_dir_relative = str(record.sample_dir.relative_to(input_dataset_root))
    except ValueError:
        sample_dir_relative = str(
            Path("samples")
            / record.stream_hash[:2]
            / record.stream_hash[2:4]
            / record.stream_hash
        )

    payload = {
        "created_utc": metadata.get("created_utc"),
        "stream_hash": record.stream_hash,
        "latent_layout": metadata.get("latent_layout", "c_d_t"),
        "sample_dir": sample_dir_relative,
        "source_relpath": metadata.get("source_relpath", metadata.get("source_path", "")),
        "target_channels": metadata.get("target_channels"),
        "target_latent_shape": metadata.get("target_latent_shape"),
        "source_stereo_latent_shape": metadata.get("source_stereo_latent_shape"),
        "source_mono_latent_shape": metadata.get("source_mono_latent_shape"),
        "source_downmix_latent_shape": metadata.get("source_downmix_latent_shape"),
        "sample_artifact_mode": metadata.get("sample_artifact_mode"),
        "dead_channel_indices": metadata.get("dead_channel_indices"),
        "duplicate_channel_pairs": metadata.get("duplicate_channel_pairs"),
    }
    return payload


def build_output_manifest_records(
    selected_records: list[SampleRecord],
    manifest_map: dict[str, dict[str, Any]],
    input_dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {"from_existing_manifest": 0, "from_metadata_fallback": 0}

    for record in selected_records:
        existing = manifest_map.get(record.stream_hash)
        if existing is not None:
            records.append(existing)
            stats["from_existing_manifest"] += 1
            continue

        records.append(
            build_fallback_manifest_record(
                record=record,
                input_dataset_root=input_dataset_root,
            )
        )
        stats["from_metadata_fallback"] += 1

    return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a QC+dataset subset from keep-prefix and avoid-path text files."
        )
    )
    parser.add_argument(
        "keep_prefixes_txt",
        help="Text file with one keep-path prefix per line.",
    )
    parser.add_argument(
        "input_qc_csv",
        help="Existing QC CSV path.",
    )
    parser.add_argument(
        "output_qc_csv",
        help="Output QC CSV path.",
    )
    parser.add_argument(
        "input_dataset_root",
        help="Existing dataset root (must contain samples/).",
    )
    parser.add_argument(
        "output_dataset_root",
        help="Output dataset root to write subset samples + manifest.jsonl.",
    )
    parser.add_argument(
        "avoid_paths_txt",
        help="Text file with one explicit file path to avoid per line.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite sample directories already present in output dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print what would be written/copied without changing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    keep_prefixes_txt = Path(args.keep_prefixes_txt).expanduser().resolve(strict=False)
    input_qc_csv = Path(args.input_qc_csv).expanduser().resolve(strict=False)
    output_qc_csv = Path(args.output_qc_csv).expanduser().resolve(strict=False)
    input_dataset_root = Path(args.input_dataset_root).expanduser().resolve(strict=False)
    output_dataset_root = Path(args.output_dataset_root).expanduser().resolve(
        strict=False
    )
    avoid_paths_txt = Path(args.avoid_paths_txt).expanduser().resolve(strict=False)

    if not input_qc_csv.exists():
        raise FileNotFoundError(f"Input QC CSV not found: {input_qc_csv}")
    if not input_dataset_root.exists():
        raise FileNotFoundError(f"Input dataset root not found: {input_dataset_root}")
    if normalize_path_key(input_qc_csv) == normalize_path_key(output_qc_csv):
        raise ValueError("input_qc_csv and output_qc_csv must be different paths.")
    if normalize_path_key(input_dataset_root) == normalize_path_key(output_dataset_root):
        raise ValueError(
            "input_dataset_root and output_dataset_root must be different paths."
        )

    keep_prefixes = load_path_list(keep_prefixes_txt)
    avoid_paths = load_path_list(avoid_paths_txt)
    keep_prefix_keys = [normalize_path_key(path) for path in keep_prefixes]
    avoid_keys = {normalize_path_key(path) for path in avoid_paths}

    if not keep_prefix_keys:
        raise ValueError(f"No keep prefixes found in: {keep_prefixes_txt}")

    fieldnames, selected_rows, selected_keys_ordered, qc_stats = filter_qc_rows(
        input_qc_csv=input_qc_csv,
        keep_prefix_keys=keep_prefix_keys,
        avoid_keys=avoid_keys,
    )

    print("QC FILTER")
    print(f"  keep_prefix_count={len(keep_prefix_keys)}")
    print(f"  avoid_path_count={len(avoid_keys)}")
    print(f"  rows_total={qc_stats['rows_total']}")
    print(f"  rows_selected={qc_stats['rows_selected']}")
    print(f"  rows_missing_path={qc_stats['rows_missing_path']}")
    print(f"  rows_avoided={qc_stats['rows_avoided']}")
    print(f"  rows_prefix_miss={qc_stats['rows_prefix_miss']}")
    print(f"  selected_unique_paths={len(selected_keys_ordered)}")

    if args.dry_run:
        print(f"[dry-run] QC CSV would be written: {output_qc_csv}")
    else:
        write_csv_atomic(output_qc_csv, selected_rows, fieldnames)
        print(f"QC CSV written: {output_qc_csv}")

    source_index, index_stats = build_source_index_for_selected_keys(
        input_dataset_root=input_dataset_root,
        selected_keys=set(selected_keys_ordered),
    )

    print("DATASET INDEX")
    print(f"  metadata_files_scanned={index_stats['metadata_files_scanned']}")
    print(f"  metadata_invalid={index_stats['metadata_invalid']}")
    print(f"  metadata_missing_source={index_stats['metadata_missing_source']}")
    print(f"  samples_incomplete={index_stats['samples_incomplete']}")
    print(f"  matching_samples_indexed={index_stats['samples_indexed']}")

    selected_records, copy_stats = select_and_copy_samples(
        selected_keys_ordered=selected_keys_ordered,
        source_index=source_index,
        input_dataset_root=input_dataset_root,
        output_dataset_root=output_dataset_root,
        overwrite_existing=bool(args.overwrite_existing),
        dry_run=bool(args.dry_run),
    )

    print("DATASET COPY")
    print(f"  selected_source_keys={copy_stats['selected_source_keys']}")
    print(f"  source_keys_with_no_sample={copy_stats['source_keys_with_no_sample']}")
    print(f"  sample_records_selected={copy_stats['sample_records_selected']}")
    print(f"  sample_records_copied={copy_stats['sample_records_copied']}")
    print(
        "  sample_records_skipped_existing="
        f"{copy_stats['sample_records_skipped_existing']}"
    )

    ready_records, ready_stats = filter_ready_records_for_manifest(
        selected_records=selected_records,
        input_dataset_root=input_dataset_root,
        output_dataset_root=output_dataset_root,
    )
    print("MANIFEST ELIGIBILITY")
    print(f"  records_checked={ready_stats['records_checked']}")
    print(f"  records_ready={ready_stats['records_ready']}")
    print(f"  records_dropped={ready_stats['records_dropped']}")

    input_manifest_path = input_dataset_root / MANIFEST_FILENAME
    output_manifest_path = output_dataset_root / MANIFEST_FILENAME
    manifest_map, manifest_map_stats = load_manifest_map(input_manifest_path)
    output_manifest_records, output_manifest_stats = build_output_manifest_records(
        selected_records=ready_records,
        manifest_map=manifest_map,
        input_dataset_root=input_dataset_root,
    )

    if args.dry_run:
        print(f"[dry-run] Manifest would be written: {output_manifest_path}")
    else:
        write_manifest_atomic(output_manifest_path, output_manifest_records)
        print(f"Manifest written: {output_manifest_path}")

    print("MANIFEST")
    print(f"  input_manifest={input_manifest_path}")
    print(f"  input_manifest_lines={manifest_map_stats['manifest_lines']}")
    print(f"  input_manifest_loaded={manifest_map_stats['manifest_loaded']}")
    print(f"  input_manifest_bad_lines={manifest_map_stats['manifest_bad_lines']}")
    print(f"  records_from_existing_manifest={output_manifest_stats['from_existing_manifest']}")
    print(
        "  records_from_metadata_fallback="
        f"{output_manifest_stats['from_metadata_fallback']}"
    )
    print(f"  output_manifest_records={len(output_manifest_records)}")


if __name__ == "__main__":
    main()
