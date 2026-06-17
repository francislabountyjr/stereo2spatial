"""Build a training exclusion list from a WavFlow clipping scan CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _counterfactual_target_stats(
    row: dict[str, str],
    *,
    peak_limit: float,
    peak_rescale_min_rms: float,
) -> tuple[bool, bool, float, float]:
    """Return ``(rescaled, would_clamp, effective_rms, pre_clamp_peak)``."""
    peak_after_rms = _as_float(row.get("target_peak_after_rms"))
    rms_after_gain = _as_float(row.get("target_rms_after_gain"))
    if peak_after_rms <= 0.0 or rms_after_gain <= 0.0:
        return False, True, 0.0, peak_after_rms
    if peak_after_rms <= peak_limit:
        return False, False, rms_after_gain, peak_after_rms
    candidate_scale = peak_limit / max(peak_after_rms, 1.0e-12)
    effective_rms = rms_after_gain * candidate_scale
    if effective_rms > peak_rescale_min_rms:
        return True, False, effective_rms, peak_limit
    return False, True, rms_after_gain, peak_after_rms


def _counterfactual_source_rms(
    row: dict[str, str],
    *,
    target_rms: float,
    peak_limit: float,
) -> float:
    """Return source effective RMS after RMS normalize and peak rescale."""
    peak_after_rms = _as_float(row.get("source_peak_after_rms"))
    if peak_after_rms <= peak_limit:
        return target_rms
    return target_rms * peak_limit / max(peak_after_rms, 1.0e-12)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _write_text(path: Path, hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(hashes) + ("\n" if hashes else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-txt", default=None)
    parser.add_argument("--peak-limit", type=float, required=True)
    parser.add_argument("--peak-rescale-min-rms", type=float, default=0.3)
    parser.add_argument("--target-rms", type=float, default=0.33)
    parser.add_argument("--min-effective-target-rms", type=float, default=None)
    parser.add_argument("--min-effective-source-rms", type=float, default=None)
    parser.add_argument(
        "--allow-target-clamp",
        action="store_true",
        help="Do not exclude rows that would still hard-clamp after counterfactual peak handling.",
    )
    parser.add_argument(
        "--keep-errors",
        action="store_true",
        help="Do not include scan error rows in the exclusion list.",
    )
    args = parser.parse_args()

    scan_csv = Path(args.scan_csv)
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else scan_csv.with_name(
            f"{scan_csv.stem}_exclude_peak{args.peak_limit:g}_rms{args.peak_rescale_min_rms:g}.json"
        )
    )
    output_txt = (
        Path(args.output_txt)
        if args.output_txt is not None
        else output_json.with_suffix(".txt")
    )
    min_effective_target_rms = (
        float(args.min_effective_target_rms)
        if args.min_effective_target_rms is not None
        else float(args.peak_rescale_min_rms)
    )
    min_effective_source_rms = (
        float(args.min_effective_source_rms)
        if args.min_effective_source_rms is not None
        else None
    )

    excluded: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {
        "scan_error": 0,
        "target_would_clamp": 0,
        "target_effective_rms_below_min": 0,
        "source_effective_rms_below_min": 0,
    }
    total = 0
    ok = 0
    for row in _load_rows(scan_csv):
        total += 1
        stream_hash = str(row.get("stream_hash") or "").strip().lower()
        if not stream_hash:
            continue
        reasons: list[str] = []
        if row.get("status") != "ok":
            if not args.keep_errors:
                reasons.append("scan_error")
                counts["scan_error"] += 1
        else:
            ok += 1
            (
                _rescaled,
                would_clamp,
                effective_target_rms,
                pre_clamp_peak,
            ) = _counterfactual_target_stats(
                row,
                peak_limit=float(args.peak_limit),
                peak_rescale_min_rms=float(args.peak_rescale_min_rms),
            )
            if would_clamp and not args.allow_target_clamp:
                reasons.append("target_would_clamp")
                counts["target_would_clamp"] += 1
            if effective_target_rms < min_effective_target_rms:
                reasons.append("target_effective_rms_below_min")
                counts["target_effective_rms_below_min"] += 1
            if min_effective_source_rms is not None:
                effective_source_rms = _counterfactual_source_rms(
                    row,
                    target_rms=float(args.target_rms),
                    peak_limit=float(args.peak_limit),
                )
                if effective_source_rms < min_effective_source_rms:
                    reasons.append("source_effective_rms_below_min")
                    counts["source_effective_rms_below_min"] += 1
            else:
                effective_source_rms = None
            if reasons:
                excluded[stream_hash] = {
                    "reasons": sorted(set(reasons)),
                    "sample_dir": row.get("sample_dir", ""),
                    "target_peak_after_rms": _as_float(
                        row.get("target_peak_after_rms")
                    ),
                    "target_effective_rms": effective_target_rms,
                    "target_pre_clamp_peak": pre_clamp_peak,
                    "source_peak_after_rms": _as_float(
                        row.get("source_peak_after_rms")
                    ),
                    "source_effective_rms": effective_source_rms,
                }

    hashes = sorted(excluded)
    payload = {
        "kind": "stereo2spatial_sample_exclusion",
        "source_scan_csv": str(scan_csv),
        "exclude_stream_hashes": hashes,
        "excluded_samples": excluded,
        "settings": {
            "peak_limit": float(args.peak_limit),
            "peak_rescale_min_rms": float(args.peak_rescale_min_rms),
            "target_rms": float(args.target_rms),
            "min_effective_target_rms": min_effective_target_rms,
            "min_effective_source_rms": min_effective_source_rms,
            "allow_target_clamp": bool(args.allow_target_clamp),
            "keep_errors": bool(args.keep_errors),
        },
        "summary": {
            "rows": total,
            "ok": ok,
            "excluded": len(hashes),
            "kept": max(0, total - len(hashes)),
            "reason_counts": counts,
        },
    }
    _write_json(output_json, payload)
    _write_text(output_txt, hashes)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=True))
    print(f"wrote_json={output_json}")
    print(f"wrote_txt={output_txt}")


if __name__ == "__main__":
    main()
