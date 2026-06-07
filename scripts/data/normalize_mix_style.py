"""Normalize raw mix-style metadata into training conditioning vectors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stereo2spatial.common.mix_style import (
    MIX_STYLE_NAMES,
    mix_style_active_names,
    mix_style_dict_to_vector,
    normalize_mix_style_raw,
)

MANIFEST_FILENAME = "manifest.jsonl"
METADATA_FILENAME = "metadata.json"
STATS_FILENAME = "mix_style_stats.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8") as handle:
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
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    os.replace(tmp_path, manifest_path)


def _sample_dir(dataset_root: Path, record: dict[str, Any]) -> Path:
    raw = record.get("sample_dir")
    if not raw:
        raise KeyError("manifest record is missing sample_dir")
    path = Path(str(raw))
    return path if path.is_absolute() else dataset_root / path


def _resolve_mix_style_names(record: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, ...]:
    for source in (metadata.get("mix_style_names"), record.get("mix_style_names")):
        if isinstance(source, list) and source:
            return tuple(str(name) for name in source)
    inactive = metadata.get("mix_style_inactive_names", record.get("mix_style_inactive_names"))
    inactive_names = [str(name) for name in inactive] if isinstance(inactive, list) else None
    return mix_style_active_names(
        layout_mode=metadata.get("mix_style_layout_mode", record.get("mix_style_layout_mode")),
        inactive_names=inactive_names,
    )


def _finite_raw_values(raw: dict[str, Any], names: tuple[str, ...]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for name in names:
        try:
            value = float(raw[name])
        except (KeyError, TypeError, ValueError):
            return None
        if not torch.isfinite(torch.tensor(value)).item():
            return None
        values[name] = value
    return values


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, float(q) / 100.0).item())


def _build_stats(
    raw_rows: list[dict[str, float]],
    names: tuple[str, ...],
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        values = [row[name] for row in raw_rows]
        lower = _percentile(values, lower_percentile)
        upper = _percentile(values, upper_percentile)
        if upper <= lower:
            upper = lower + 1.0
        stats[name] = {
            "p01": lower,
            "p50": _percentile(values, 50.0),
            "p99": upper,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return stats


def normalize_dataset_mix_style(
    *,
    dataset_root: Path,
    manifest_path: Path,
    stats_path: Path,
    lower_percentile: float,
    upper_percentile: float,
    dry_run: bool,
) -> dict[str, int]:
    """Normalize every manifest sample that has complete ``mix_style_raw``."""
    records = _load_manifest_records(manifest_path)
    raw_by_index: dict[int, dict[str, float]] = {}
    metadata_by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    names_by_index: dict[int, tuple[str, ...]] = {}
    skipped = 0

    for index, record in enumerate(records):
        metadata_path = _sample_dir(dataset_root, record) / METADATA_FILENAME
        metadata = _load_json_object(metadata_path)
        raw = metadata.get("mix_style_raw", record.get("mix_style_raw"))
        names = _resolve_mix_style_names(record, metadata)
        raw_values = _finite_raw_values(raw, names) if isinstance(raw, dict) else None
        if raw_values is None:
            skipped += 1
            continue
        raw_by_index[index] = raw_values
        metadata_by_index[index] = (metadata_path, metadata)
        names_by_index[index] = names

    if not raw_by_index:
        raise RuntimeError("No samples with complete mix_style_raw metadata were found.")

    unique_names = {names for names in names_by_index.values()}
    if len(unique_names) != 1:
        raise RuntimeError(
            "Mix-style normalization expects one active-name set per dataset. "
            f"Found {len(unique_names)} sets; split layouts into separate datasets."
        )
    active_names = next(iter(unique_names))

    stats = _build_stats(
        list(raw_by_index.values()),
        active_names,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    stats_payload: dict[str, Any] = {
        "version": 1,
        "names": list(active_names),
        "all_names": list(MIX_STYLE_NAMES),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "sample_count": len(raw_by_index),
        "features": stats,
    }

    for index, raw_values in raw_by_index.items():
        names = names_by_index[index]
        normalized = normalize_mix_style_raw(raw_values, stats, names=names)
        vector = mix_style_dict_to_vector(normalized, names=names)
        records[index]["mix_style"] = normalized
        records[index]["mix_style_vector"] = vector
        records[index]["mix_style_names"] = list(names)
        records[index]["mix_style_raw"] = raw_values

        metadata_path, metadata = metadata_by_index[index]
        metadata["mix_style"] = normalized
        metadata["mix_style_vector"] = vector
        metadata["mix_style_names"] = list(names)
        metadata["mix_style_stats_file"] = str(stats_path.name)
        metadata["mix_style_raw"] = raw_values
        if not dry_run:
            _write_json_atomic(metadata_path, metadata)

    if not dry_run:
        _write_json_atomic(stats_path, stats_payload)
        _write_manifest_atomic(manifest_path, records)

    return {
        "manifest_records": len(records),
        "normalized": len(raw_by_index),
        "skipped_missing_raw": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize raw target-derived mix-style controls and update "
            "metadata.json plus manifest.jsonl for training."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-file", default=MANIFEST_FILENAME)
    parser.add_argument("--stats-file", default=STATS_FILENAME)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    stats_path = Path(args.stats_file)
    if not stats_path.is_absolute():
        stats_path = dataset_root / stats_path
    summary = normalize_dataset_mix_style(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        stats_path=stats_path,
        lower_percentile=float(args.lower_percentile),
        upper_percentile=float(args.upper_percentile),
        dry_run=bool(args.dry_run),
    )
    print("Mix-style normalization summary:")
    for key, value in summary.items():
        print(f"  - {key}={value}")
    print(f"  - stats={stats_path}")
    print(f"  - manifest={manifest_path}")
    if args.dry_run:
        print("  - dry_run=True")


if __name__ == "__main__":
    main()
