#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Iterable

METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.jsonl"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"
TARGET_LATENT_FILENAME = "target_latent.pt"
SOURCE_STEREO_LATENT_FILENAME = "source_stereo_latent.pt"
SOURCE_MONO_LATENT_FILENAME = "source_mono_latent.pt"
SOURCE_DOWNMIX_LATENT_FILENAME = "source_downmix_latent.pt"

DRIVE_ABS_RE = re.compile(r"^[a-zA-Z]:([\\/]|$)")


@dataclass(frozen=True)
class MatchPattern:
    raw: str
    normalized: str
    is_absolute_prefix: bool
    absolute_prefix_cf: str | None
    parts_cf: tuple[str, ...]


@dataclass
class CandidateSample:
    sample_dir: Path
    stream_hash: str
    source_path: str
    matched_patterns: set[str] = field(default_factory=set)


def strip_optional_quotes(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def canonical_path_text(raw: str) -> str:
    cleaned = strip_optional_quotes(raw).replace("/", "\\")
    return str(PureWindowsPath(cleaned))


def canonical_for_prefix(raw: str) -> str:
    text = canonical_path_text(raw)
    if len(text) > 3 and text.endswith("\\"):
        text = text.rstrip("\\")
    return text.casefold()


def path_parts_casefold(raw: str) -> tuple[str, ...]:
    text = canonical_path_text(raw)
    parts: list[str] = []
    for part in PureWindowsPath(text).parts:
        if not part or part == "." or part in {"\\", "/"}:
            continue
        part_cf = part.casefold()
        if re.fullmatch(r"[a-z]:\\", part_cf):
            part_cf = part_cf[:2]
        parts.append(part_cf)
    return tuple(parts)


def is_absolute_prefix_pattern(normalized: str) -> bool:
    if normalized.startswith("\\\\"):
        return True
    return bool(DRIVE_ABS_RE.match(normalized))


def build_pattern(raw_line: str) -> MatchPattern | None:
    cleaned = strip_optional_quotes(raw_line)
    if not cleaned:
        return None

    normalized = canonical_path_text(cleaned)
    abs_prefix = is_absolute_prefix_pattern(normalized)
    if abs_prefix:
        return MatchPattern(
            raw=cleaned,
            normalized=normalized,
            is_absolute_prefix=True,
            absolute_prefix_cf=canonical_for_prefix(normalized),
            parts_cf=(),
        )

    parts = path_parts_casefold(normalized)
    if not parts:
        return None
    return MatchPattern(
        raw=cleaned,
        normalized=normalized,
        is_absolute_prefix=False,
        absolute_prefix_cf=None,
        parts_cf=parts,
    )


def load_patterns(txt_path: Path) -> list[MatchPattern]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {txt_path}")

    patterns: list[MatchPattern] = []
    seen: set[tuple[str, bool]] = set()
    with open(txt_path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pattern = build_pattern(stripped)
            if pattern is None:
                continue
            dedupe_key = (pattern.normalized.casefold(), pattern.is_absolute_prefix)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            patterns.append(pattern)
    return patterns


def path_has_subpath(source_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    pattern_len = len(pattern_parts)
    if pattern_len == 0 or pattern_len > len(source_parts):
        return False
    max_start = len(source_parts) - pattern_len
    for index in range(max_start + 1):
        if source_parts[index : index + pattern_len] == pattern_parts:
            return True
    return False


def pattern_matches_source(pattern: MatchPattern, source_path: str) -> bool:
    if pattern.is_absolute_prefix:
        source_cf = canonical_for_prefix(source_path)
        prefix_cf = pattern.absolute_prefix_cf
        if prefix_cf is None:
            return False
        return source_cf == prefix_cf or source_cf.startswith(prefix_cf + "\\")

    source_parts = path_parts_casefold(source_path)
    return path_has_subpath(source_parts, pattern.parts_cf)


def iter_metadata_files(samples_root: Path) -> Iterable[Path]:
    if not samples_root.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_root}")
    yield from samples_root.rglob(METADATA_FILENAME)


def sample_dir_is_loadable(sample_dir: Path) -> bool:
    bundle = sample_dir / SAMPLE_BUNDLE_FILENAME
    if bundle.exists():
        return True

    split_required = [
        sample_dir / TARGET_LATENT_FILENAME,
        sample_dir / SOURCE_STEREO_LATENT_FILENAME,
        sample_dir / SOURCE_MONO_LATENT_FILENAME,
        sample_dir / SOURCE_DOWNMIX_LATENT_FILENAME,
    ]
    return all(path.exists() for path in split_required)


def sync_manifest_file(
    dataset_root: Path,
    removed_stream_hashes: set[str],
    apply_changes: bool,
) -> dict[str, int]:
    manifest_path = dataset_root / MANIFEST_FILENAME
    stats = {
        "manifest_exists": int(manifest_path.exists()),
        "lines_total": 0,
        "lines_kept": 0,
        "lines_invalid": 0,
        "lines_removed_by_hash": 0,
        "lines_removed_missing_or_unloadable": 0,
    }
    if not manifest_path.exists():
        return stats

    kept_lines: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["lines_total"] += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                stats["lines_invalid"] += 1
                kept_lines.append(line)
                continue
            if not isinstance(payload, dict):
                stats["lines_invalid"] += 1
                kept_lines.append(line)
                continue

            stream_hash_raw = payload.get("stream_hash")
            stream_hash = (
                str(stream_hash_raw).strip().lower()
                if isinstance(stream_hash_raw, str) and stream_hash_raw.strip()
                else None
            )
            if stream_hash is not None and stream_hash in removed_stream_hashes:
                stats["lines_removed_by_hash"] += 1
                continue

            sample_dir_raw = payload.get("sample_dir")
            if isinstance(sample_dir_raw, str) and sample_dir_raw.strip():
                sample_dir = dataset_root / Path(sample_dir_raw)
                if (not sample_dir.exists()) or (not sample_dir_is_loadable(sample_dir)):
                    stats["lines_removed_missing_or_unloadable"] += 1
                    continue

            kept_lines.append(line if line.endswith("\n") else line + "\n")

    stats["lines_kept"] = len(kept_lines)
    total_removed = (
        stats["lines_removed_by_hash"] + stats["lines_removed_missing_or_unloadable"]
    )

    if apply_changes and total_removed > 0:
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.writelines(kept_lines)
        tmp_path.replace(manifest_path)

    return stats


def write_report_csv(report_csv: Path, candidates: list[CandidateSample]) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_dir", "stream_hash", "source_path", "matched_patterns"],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "sample_dir": str(candidate.sample_dir),
                    "stream_hash": candidate.stream_hash,
                    "source_path": candidate.source_path,
                    "matched_patterns": " | ".join(sorted(candidate.matched_patterns)),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete dataset samples whose metadata.source_path matches any absolute "
            "path-prefix or subpath pattern from a text file."
        )
    )
    parser.add_argument(
        "paths_txt",
        help="Text file with one absolute path or subpath pattern per line.",
    )
    parser.add_argument(
        "dataset_root",
        help="Dataset root containing samples/.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched sample directories. Without this, script is dry-run.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Allow apply-mode deletion even if some patterns match zero samples.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Optional path to write a CSV report of matched samples.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=200,
        help="Maximum number of matched samples to print (default: 200).",
    )
    parser.add_argument(
        "--skip-manifest-update",
        action="store_true",
        help=(
            "Do not update dataset_root/manifest.jsonl. "
            "By default, deleted and stale entries are pruned."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths_txt = Path(args.paths_txt).expanduser().resolve(strict=False)
    dataset_root = Path(args.dataset_root).expanduser().resolve(strict=False)
    samples_root = dataset_root / "samples"
    report_csv = (
        Path(args.report_csv).expanduser().resolve(strict=False)
        if args.report_csv
        else None
    )

    patterns = load_patterns(paths_txt)
    if not patterns:
        raise ValueError(f"No valid patterns loaded from: {paths_txt}")

    pattern_match_counts = {pattern.raw: 0 for pattern in patterns}

    candidates_by_dir: dict[Path, CandidateSample] = {}
    stats = {
        "metadata_scanned": 0,
        "metadata_invalid": 0,
        "metadata_missing_source_path": 0,
    }

    for metadata_path in iter_metadata_files(samples_root):
        stats["metadata_scanned"] += 1
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

        source_path = metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            stats["metadata_missing_source_path"] += 1
            continue

        matched_raw_patterns: list[str] = []
        for pattern in patterns:
            if pattern_matches_source(pattern, source_path):
                matched_raw_patterns.append(pattern.raw)

        if not matched_raw_patterns:
            continue

        stream_hash_raw = metadata.get("stream_hash")
        stream_hash = (
            str(stream_hash_raw).strip().lower()
            if isinstance(stream_hash_raw, str) and stream_hash_raw.strip()
            else sample_dir.name
        )

        candidate = candidates_by_dir.get(sample_dir)
        if candidate is None:
            candidate = CandidateSample(
                sample_dir=sample_dir,
                stream_hash=stream_hash,
                source_path=source_path,
            )
            candidates_by_dir[sample_dir] = candidate

        for raw_pattern in matched_raw_patterns:
            candidate.matched_patterns.add(raw_pattern)
            pattern_match_counts[raw_pattern] += 1

    candidates = sorted(candidates_by_dir.values(), key=lambda item: str(item.sample_dir))
    unmatched_patterns = [raw for raw, count in pattern_match_counts.items() if count == 0]

    print("DELETE DATASET SAMPLES")
    print(f"  dataset_root={dataset_root}")
    print(f"  samples_root={samples_root}")
    print(f"  paths_txt={paths_txt}")
    print(f"  patterns_loaded={len(patterns)}")
    print(f"  metadata_scanned={stats['metadata_scanned']}")
    print(f"  metadata_invalid={stats['metadata_invalid']}")
    print(f"  metadata_missing_source_path={stats['metadata_missing_source_path']}")
    print(f"  matched_sample_dirs={len(candidates)}")
    print(f"  unmatched_patterns={len(unmatched_patterns)}")

    if unmatched_patterns:
        print("UNMATCHED PATTERNS")
        for raw in unmatched_patterns:
            print(f"  - {raw}")

    if candidates:
        print("MATCHED SAMPLES")
        for index, candidate in enumerate(candidates):
            if index >= max(0, args.print_limit):
                remaining = len(candidates) - args.print_limit
                if remaining > 0:
                    print(f"  ... {remaining} more omitted")
                break
            pattern_text = " | ".join(sorted(candidate.matched_patterns))
            print(
                f"  - {candidate.sample_dir} :: {candidate.source_path} "
                f"(patterns: {pattern_text})"
            )

    if report_csv is not None:
        write_report_csv(report_csv, candidates)
        print(f"  report_csv_written={report_csv}")

    candidate_hashes = {candidate.stream_hash for candidate in candidates}
    manifest_stats = None
    if not args.skip_manifest_update:
        manifest_stats = sync_manifest_file(
            dataset_root=dataset_root,
            removed_stream_hashes=candidate_hashes,
            apply_changes=False,
        )
        if manifest_stats["manifest_exists"]:
            print("MANIFEST SYNC (preview)")
            print(f"  manifest_path={dataset_root / MANIFEST_FILENAME}")
            print(f"  lines_total={manifest_stats['lines_total']}")
            print(f"  lines_kept={manifest_stats['lines_kept']}")
            print(f"  lines_invalid={manifest_stats['lines_invalid']}")
            print(f"  lines_removed_by_hash={manifest_stats['lines_removed_by_hash']}")
            print(
                "  lines_removed_missing_or_unloadable="
                f"{manifest_stats['lines_removed_missing_or_unloadable']}"
            )

    if not args.apply:
        print(
            "DRY RUN ONLY: pass --apply to delete matched sample directories "
            "and write manifest updates."
        )
        return

    if unmatched_patterns and not args.allow_unmatched:
        raise RuntimeError(
            "Refusing to delete because some patterns matched zero samples. "
            "Fix patterns or pass --allow-unmatched to override."
        )

    deleted = 0
    for candidate in candidates:
        try:
            relative = candidate.sample_dir.relative_to(samples_root)
        except ValueError as error:
            raise RuntimeError(
                f"Refusing to delete path outside samples root: {candidate.sample_dir}"
            ) from error

        if str(relative) in {"", "."}:
            raise RuntimeError("Refusing to delete the samples root directory.")

        if candidate.sample_dir.exists():
            shutil.rmtree(candidate.sample_dir)
            deleted += 1

    print(f"DELETE COMPLETE: deleted_sample_dirs={deleted}")

    if not args.skip_manifest_update:
        manifest_stats = sync_manifest_file(
            dataset_root=dataset_root,
            removed_stream_hashes=candidate_hashes,
            apply_changes=True,
        )
        if manifest_stats["manifest_exists"]:
            print("MANIFEST SYNC (applied)")
            print(f"  manifest_path={dataset_root / MANIFEST_FILENAME}")
            print(f"  lines_total={manifest_stats['lines_total']}")
            print(f"  lines_kept={manifest_stats['lines_kept']}")
            print(f"  lines_invalid={manifest_stats['lines_invalid']}")
            print(f"  lines_removed_by_hash={manifest_stats['lines_removed_by_hash']}")
            print(
                "  lines_removed_missing_or_unloadable="
                f"{manifest_stats['lines_removed_missing_or_unloadable']}"
            )


if __name__ == "__main__":
    main()
