#!/usr/bin/env python3
"""
Update QC labels for all tracks that live under a specific album directory.

Examples:
  python update_qc_album_label.py "E:\\Downloads\\Atmos Maybe\\Luv Is Rage 2"
  python update_qc_album_label.py "E:\\Downloads\\Atmos Maybe\\Luv Is Rage 2" --label FRONTFOCUS
  python update_qc_album_label.py "E:\\Downloads\\Atmos Maybe\\Luv Is Rage 2" --qc-csv atmos_filter_report.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LABEL = "OK"
DEFAULT_QC_CSV = Path(__file__).with_name("atmos_filter_report.csv")
REQUIRED_COLUMNS = ("path", "label")


@dataclass(frozen=True)
class UpdateSummary:
    matched_rows: int
    updated_rows: int
    skipped_missing_path_rows: int


def normalize_path_key(raw_path: str) -> str:
    """Normalize a path for case-insensitive prefix comparisons."""
    return os.path.normcase(os.path.abspath(raw_path.strip()))


def path_is_under_album(track_path: str, album_dir: str) -> bool:
    """Return True when a track path belongs to an album directory prefix."""
    track_key = normalize_path_key(track_path)
    album_key = normalize_path_key(album_dir)
    album_prefix = album_key if album_key.endswith(os.sep) else album_key + os.sep
    return track_key == album_key or track_key.startswith(album_prefix)


def write_csv_atomic(
    csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    """Write a CSV through a temporary file, then atomically replace."""
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def _require_columns(fieldnames: Iterable[str], csv_path: Path) -> None:
    normalized = set(fieldnames)
    for column in REQUIRED_COLUMNS:
        if column not in normalized:
            raise ValueError(f"CSV missing required column: {column!r} ({csv_path})")


def _apply_label_updates(
    *,
    rows: list[dict[str, str]],
    album_dir: str,
    target_label: str,
) -> UpdateSummary:
    updated = 0
    matched = 0
    skipped_missing_path = 0
    for row in rows:
        track_path = (row.get("path") or "").strip()
        if not track_path:
            skipped_missing_path += 1
            continue
        if not path_is_under_album(track_path, album_dir):
            continue

        matched += 1
        old_label = (row.get("label") or "").strip()
        if old_label == target_label:
            continue

        row["label"] = target_label
        updated += 1

    return UpdateSummary(
        matched_rows=matched,
        updated_rows=updated,
        skipped_missing_path_rows=skipped_missing_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Update labels in a QC CSV for tracks whose path starts with an album directory."
        )
    )
    parser.add_argument(
        "album_dir",
        help="Album directory path prefix to match against CSV 'path'.",
    )
    parser.add_argument(
        "--qc-csv",
        default=str(DEFAULT_QC_CSV),
        help="QC CSV file path. Defaults to atmos_filter_report.csv next to this script.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"New label to apply to matching rows (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would change; do not write the CSV.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    qc_csv_path = Path(args.qc_csv).expanduser().resolve(strict=False)
    if not qc_csv_path.exists():
        raise FileNotFoundError(f"QC CSV not found: {qc_csv_path}")

    with open(qc_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header row: {qc_csv_path}")
        _require_columns(fieldnames, qc_csv_path)
        rows = list(reader)

    target_label = str(args.label).strip()
    summary = _apply_label_updates(
        rows=rows,
        album_dir=args.album_dir,
        target_label=target_label,
    )

    print(f"QC CSV: {qc_csv_path}")
    print(f"Album prefix: {normalize_path_key(args.album_dir)}")
    print(f"Matched rows: {summary.matched_rows}")
    print(f"Rows updated: {summary.updated_rows}")
    print(f"Rows skipped (missing path): {summary.skipped_missing_path_rows}")

    if args.dry_run:
        print("Dry run enabled; no file changes written.")
        return

    if summary.updated_rows == 0:
        print("No updates needed.")
        return

    write_csv_atomic(qc_csv_path, rows, fieldnames)
    print(f"Updated CSV written: {qc_csv_path}")


if __name__ == "__main__":
    main()
