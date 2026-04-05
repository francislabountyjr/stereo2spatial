#!/usr/bin/env python3
"""
detect_upmix.py

Recursively scans a folder for Atmos-ish files and labels tracks as:
- UPMIX: channels largely reconstructible from stereo L/R (+ small shifts)
- DUPLICATE: dominant shared component + non-front tracks stereo-mid strongly
- LOWDIM: "stereo-ish Atmos" (low effective rank) *with active non-front energy*
- FRONTFOCUS: low effective rank but non-front energy is very low (often legit/intimate mixes)
- OK: everything else

Resume mode:
- If output CSV exists, it loads prior results and SKIPs tracks already processed
  (unless file_size/file_mtime changed, in which case it reprocesses and updates the row).

Autosave:
- Writes the CSV every N processed tracks (default: 10) so you can cancel safely.

Requires: ffmpeg + ffprobe on PATH.

Example:
  python detect_upmix.py "E:\Downloads\Atmos" -o atmos_filter_report.csv
  python detect_upmix.py "E:\Downloads\Atmos" -o atmos_filter_report.csv --save-every 25

Options:
  --segments 3
  --rank-cutoff 3.4
  --lowdim-nonfront-min -14
  --rank-agg max|median|p75|mean
  --no-skip   (force reprocess everything even if already in CSV)
"""

import os
import json
import csv
import math
import argparse
import subprocess
from typing import List, Dict, Optional

import numpy as np

# -------------------- Default thresholds --------------------
# Stereo upmix detector
UPMIX_MEDIAN_RESID_T1 = 0.10
UPMIX_EFF_RANK_T1 = 2.8

UPMIX_MEDIAN_RESID_T2 = 0.14
UPMIX_EFF_RANK_T2 = 3.2
UPMIX_NONFRONT_DB_T2 = -8.0

# Duplicate/smeary detector (shared mono-ish component + mid-tracking in non-front)
DUP_SHARED_T = 0.65
DUP_MIDCORR_T = 0.78
DUP_EFF_RANK_T = 3.2

# Stereo-ish Atmos detector
DEFAULT_RANK_CUTOFF = 3.4
DEFAULT_LOWDIM_NONFRONT_MIN_DB = -14.0
MAX_RESID_SAMPLES = 300_000

# CSV fields
CSV_FIELDS = [
    "path",
    "channels",
    "channel_layout",
    "codec",
    "duration_s",
    "segments",
    "file_size",
    "file_mtime",
    "median_resid",
    "eff_rank",
    "eff_rank_median",
    "nonfront_energy_db",
    "shared_ratio",
    "midcorr_nonfront",
    "likely_upmix",
    "likely_duplicate",
    "likely_lowdim",
    "label",
    "note",
]


# -------------------- tiny utils --------------------


def norm_path_key(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def safe_int(x, default=""):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def db10(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-12))


# -------------------- ffmpeg / ffprobe helpers --------------------


def ffprobe_audio_info(path: str) -> Dict:
    """
    Returns dict with channels, channel_layout, codec_name, duration_s (may be None).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,channel_layout,codec_name:format=duration",
        "-of",
        "json",
        path,
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8"))

    stream = data["streams"][0]
    duration_s = None
    try:
        if "format" in data and "duration" in data["format"]:
            duration_s = float(data["format"]["duration"])
    except Exception:
        duration_s = None

    return {
        "channels": int(stream.get("channels", 2)),
        "channel_layout": stream.get("channel_layout", ""),
        "codec_name": stream.get("codec_name", ""),
        "duration_s": duration_s,
    }


def decode_segment_float32(
    path: str, start_s: float, dur_s: float, sr: int, channels: int
) -> np.ndarray:
    """
    Returns audio as np.ndarray shape (C, N) float32, decoded by ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start_s),
        "-t",
        str(dur_s),
        "-i",
        path,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size == 0:
        return np.zeros((channels, 0), dtype=np.float32)
    audio = audio.reshape(-1, channels).T  # (C, N)
    return audio


# -------------------- signal features --------------------


def shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    """Zero-padded shift. Positive shift delays (moves right)."""
    if shift == 0:
        return x.copy()
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        s = -shift
        y[:-s] = x[s:]
    return y


def build_stereo_basis(L: np.ndarray, R: np.ndarray, shifts: List[int]) -> np.ndarray:
    cols = []
    for sh in shifts:
        cols.append(shift_signal(L, sh))
        cols.append(shift_signal(R, sh))
    return np.stack(cols, axis=1)  # (N, K)


def decimate_for_residual(audio: np.ndarray, max_samples: int) -> np.ndarray:
    """Evenly decimate along time axis for residual fitting only."""
    if max_samples <= 0 or audio.shape[1] <= max_samples:
        return audio
    step = max(1, int(math.ceil(audio.shape[1] / float(max_samples))))
    return audio[:, ::step]


def residual_ratio_from_stereo(c: np.ndarray, basis: np.ndarray) -> float:
    """Least squares fit c ≈ basis @ w. Return residual_energy / signal_energy."""
    if c.size == 0:
        return 1.0
    # Solve a tiny KxK system and keep arithmetic in float32.
    B = np.asarray(basis, dtype=np.float32)
    y = np.asarray(c, dtype=np.float32)
    gram = B.T @ B
    rhs = B.T @ y

    # Small ridge term for stability with correlated shifted channels.
    ridge = np.float32(1e-6) * (
        np.trace(gram) / max(1, gram.shape[0]) + np.float32(1e-12)
    )
    gram = gram.copy()
    gram.flat[:: gram.shape[0] + 1] += ridge

    try:
        w = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(gram) @ rhs

    pred = B @ w
    num = float(np.mean((y - pred) ** 2))
    den = float(np.mean(y**2)) + 1e-12
    return num / den


def effective_rank_participation_ratio(X: np.ndarray) -> float:
    """Participation ratio of covariance eigenvalues."""
    if X.shape[1] == 0:
        return 0.0
    cov = (X @ X.T) / (X.shape[1] + 1e-12)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    s1 = float(np.sum(eig)) + 1e-12
    s2 = float(np.sum(eig**2)) + 1e-12
    return (s1 * s1) / s2


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.sqrt(np.mean(a * a)) * np.sqrt(np.mean(b * b)) + 1e-12
    return float(np.mean(a * b) / den)


def shared_component_ratio(audio: np.ndarray) -> float:
    """Largest eigenvalue / sum eigenvalues of normalized channel covariance."""
    if audio.shape[1] == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio**2, axis=1, keepdims=True) + 1e-12)
    Xn = audio / rms
    cov = (Xn @ Xn.T) / (Xn.shape[1] + 1e-12)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    return float(np.max(eig) / (np.sum(eig) + 1e-12))


def nonfront_mid_corr(audio: np.ndarray) -> float:
    """Average |corr(nonfront_channel, mid(L,R))|."""
    C, N = audio.shape
    if C < 3 or N == 0:
        return 0.0
    L = audio[0]
    R = audio[1]
    mid = 0.5 * (L + R)
    cors = []
    for ch in range(2, C):
        cors.append(abs(corrcoef(audio[ch], mid)))
    return float(np.mean(cors)) if cors else 0.0


def compute_features(audio: np.ndarray) -> Dict[str, float]:
    """Compute features for one decoded segment."""
    C, N = audio.shape
    total_energy = float(np.mean(audio**2)) if N else 0.0
    if total_energy < 1e-8:
        return {
            "median_resid": 1.0,
            "eff_rank": 0.0,
            "nonfront_energy_db": -999.0,
            "shared_ratio": 0.0,
            "midcorr_nonfront": 0.0,
            "silent": 1.0,
        }

    resid_audio = decimate_for_residual(audio, MAX_RESID_SAMPLES)

    # Assume first two channels are FL, FR
    L = resid_audio[0]
    R = resid_audio[1] if C > 1 else resid_audio[0]

    # Stereo basis with small time shifts (±2ms, ±1ms, 0 at 48k)
    shifts = [-96, -48, 0, 48, 96]
    basis = build_stereo_basis(L, R, shifts)

    resid = []
    for ch in range(2, C):
        resid.append(residual_ratio_from_stereo(resid_audio[ch], basis))
    median_resid = float(np.median(resid)) if resid else 0.0

    # Effective rank (normalize per-channel RMS first)
    rms = np.sqrt(np.mean(audio**2, axis=1, keepdims=True) + 1e-12)
    Xn = audio / rms
    eff_rank = float(effective_rank_participation_ratio(Xn))

    # Non-front energy relative to front energy (mean-square)
    front_e = float(np.mean(audio[:2] ** 2)) if C >= 2 else float(np.mean(audio**2))
    nonfront_e = float(np.mean(audio[2:] ** 2)) if C > 2 else 0.0
    nonfront_energy_db = db10(nonfront_e / (front_e + 1e-12)) if C > 2 else -999.0

    shared_ratio = float(shared_component_ratio(audio))
    midcorr_nonfront = float(nonfront_mid_corr(audio))

    return {
        "median_resid": median_resid,
        "eff_rank": eff_rank,
        "nonfront_energy_db": nonfront_energy_db,
        "shared_ratio": shared_ratio,
        "midcorr_nonfront": midcorr_nonfront,
        "silent": 0.0,
    }


def aggregate(values: List[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "max":
        return float(np.max(values))
    if mode == "min":
        return float(np.min(values))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "p75":
        return float(np.percentile(values, 75))
    if mode == "p25":
        return float(np.percentile(values, 25))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def choose_segment_starts(
    duration_s: Optional[float], base_start_s: float, dur_s: float, segments: int
) -> List[float]:
    if duration_s is None or duration_s <= 0:
        return [base_start_s]

    max_start = max(0.0, duration_s - dur_s)
    if segments <= 1:
        return [min(base_start_s, max_start)]

    first = min(base_start_s, max_start)
    if max_start == first:
        return [first] * segments

    starts = np.linspace(first, max_start, segments)
    return [float(s) for s in starts]


# -------------------- resume CSV helpers --------------------


def load_existing_csv(csv_path: str) -> Dict:
    """Loads existing CSV rows and an index by normalized absolute path."""
    if not os.path.exists(csv_path):
        return {"rows": [], "index": {}, "has_size": False, "has_mtime": False}

    rows = []
    index = {}
    has_size = False
    has_mtime = False

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            has_size = "file_size" in reader.fieldnames
            has_mtime = "file_mtime" in reader.fieldnames

        for row in reader:
            rows.append(row)
            key = norm_path_key(row.get("path", ""))
            if key:
                index[key] = len(rows) - 1  # last wins

    return {"rows": rows, "index": index, "has_size": has_size, "has_mtime": has_mtime}


def upsert_row(rows: List[Dict], index: Dict[str, int], row: Dict):
    key = norm_path_key(row["path"])
    if key in index:
        rows[index[key]] = row
    else:
        index[key] = len(rows)
        rows.append(row)


def write_csv_atomic(rows: List[Dict], out_csv: str):
    """Write CSV to temp then atomically replace."""
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in CSV_FIELDS}
            w.writerow(out)
    os.replace(tmp, out_csv)


# -------------------- analysis --------------------


def analyze_file(
    path: str,
    base_start_s: float,
    dur_s: float,
    sr: int,
    segments: int,
    rank_cutoff: float,
    lowdim_nonfront_min_db: float,
    rank_agg: str,
) -> Dict:

    info = ffprobe_audio_info(path)
    C = info["channels"]

    starts = choose_segment_starts(info["duration_s"], base_start_s, dur_s, segments)

    seg_feats = []
    for s in starts:
        audio = decode_segment_float32(path, start_s=s, dur_s=dur_s, sr=sr, channels=C)
        seg_feats.append(compute_features(audio))

    median_resid = aggregate([f["median_resid"] for f in seg_feats], "median")
    eff_rank = aggregate([f["eff_rank"] for f in seg_feats], rank_agg)
    eff_rank_median = aggregate([f["eff_rank"] for f in seg_feats], "median")
    nonfront_db = aggregate([f["nonfront_energy_db"] for f in seg_feats], "max")
    shared_ratio = aggregate([f["shared_ratio"] for f in seg_feats], "median")
    midcorr_nonfront = aggregate([f["midcorr_nonfront"] for f in seg_feats], "max")

    likely_upmix = (
        median_resid < UPMIX_MEDIAN_RESID_T1 and eff_rank < UPMIX_EFF_RANK_T1
    ) or (
        median_resid < UPMIX_MEDIAN_RESID_T2
        and eff_rank < UPMIX_EFF_RANK_T2
        and nonfront_db < UPMIX_NONFRONT_DB_T2
    )

    likely_duplicate = (
        shared_ratio > DUP_SHARED_T
        and midcorr_nonfront > DUP_MIDCORR_T
        and eff_rank < DUP_EFF_RANK_T
    )

    likely_lowdim = eff_rank < rank_cutoff and nonfront_db > lowdim_nonfront_min_db

    if aggregate([f["silent"] for f in seg_feats], "mean") > 0.5:
        label = "SILENT"
        note = "silent/empty decode"
    elif likely_upmix:
        label = "UPMIX"
        note = ""
    elif likely_duplicate:
        label = "DUPLICATE"
        note = ""
    elif likely_lowdim:
        label = "LOWDIM"
        note = ""
    elif eff_rank < rank_cutoff:
        label = "FRONTFOCUS"
        note = ""
    else:
        label = "OK"
        note = ""

    st = os.stat(path)
    file_size = int(st.st_size)
    file_mtime = int(st.st_mtime)

    return {
        "path": os.path.abspath(path),
        "channels": C,
        "channel_layout": info["channel_layout"],
        "codec": info["codec_name"],
        "duration_s": info["duration_s"] if info["duration_s"] is not None else "",
        "segments": segments,
        "file_size": file_size,
        "file_mtime": file_mtime,
        "median_resid": median_resid,
        "eff_rank": eff_rank,
        "eff_rank_median": eff_rank_median,
        "nonfront_energy_db": nonfront_db,
        "shared_ratio": shared_ratio,
        "midcorr_nonfront": midcorr_nonfront,
        "likely_upmix": bool(likely_upmix),
        "likely_duplicate": bool(likely_duplicate),
        "likely_lowdim": bool(likely_lowdim),
        "label": label,
        "note": note,
    }


def scan_folder(
    folder: str,
    out_csv: str,
    base_start_s: float,
    dur_s: float,
    sr: int,
    segments: int,
    rank_cutoff: float,
    lowdim_nonfront_min_db: float,
    rank_agg: str,
    no_skip: bool,
    save_every: int,
    exts=(".m4a", ".mp4", ".m4b", ".m4v"),
):
    folder = os.path.abspath(folder)

    existing = load_existing_csv(out_csv)
    rows = existing["rows"]
    index = existing["index"]
    has_size = existing["has_size"]
    has_mtime = existing["has_mtime"]

    processed = 0
    skipped = 0
    updated = 0
    new = 0
    errors = 0

    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith(exts):
                continue

            path = os.path.abspath(os.path.join(root, fn))
            key = norm_path_key(path)

            # skip check
            if (not no_skip) and (key in index):
                old = rows[index[key]]
                if has_size and has_mtime:
                    try:
                        old_size = int(float(old.get("file_size", -1)))
                        old_mtime = int(float(old.get("file_mtime", -1)))
                        st = os.stat(path)
                        if old_size == int(st.st_size) and old_mtime == int(
                            st.st_mtime
                        ):
                            skipped += 1
                            print(f"SKIP      {old.get('label',''):<10} {fn}")
                            continue
                    except Exception:
                        pass
                else:
                    skipped += 1
                    print(f"SKIP      {old.get('label',''):<10} {fn}")
                    continue

            try:
                r = analyze_file(
                    path=path,
                    base_start_s=base_start_s,
                    dur_s=dur_s,
                    sr=sr,
                    segments=segments,
                    rank_cutoff=rank_cutoff,
                    lowdim_nonfront_min_db=lowdim_nonfront_min_db,
                    rank_agg=rank_agg,
                )

                if key in index:
                    updated += 1
                else:
                    new += 1

                upsert_row(rows, index, r)
                processed += 1

                print(
                    f"{r['label']:<9} "
                    f"resid={r['median_resid']:.3f}  "
                    f"rank={r['eff_rank']:.2f} (med={r['eff_rank_median']:.2f})  "
                    f"nonfront={r['nonfront_energy_db']:.1f}dB  "
                    f"shared={r['shared_ratio']:.2f}  "
                    f"midcorr={r['midcorr_nonfront']:.2f}  "
                    f"{fn}"
                )

            except subprocess.CalledProcessError as e:
                errors += 1
                err_row = {
                    "path": path,
                    "channels": "",
                    "channel_layout": "",
                    "codec": "",
                    "duration_s": "",
                    "segments": segments,
                    "file_size": safe_int(os.path.getsize(path), ""),
                    "file_mtime": safe_int(os.path.getmtime(path), ""),
                    "median_resid": "",
                    "eff_rank": "",
                    "eff_rank_median": "",
                    "nonfront_energy_db": "",
                    "shared_ratio": "",
                    "midcorr_nonfront": "",
                    "likely_upmix": False,
                    "likely_duplicate": False,
                    "likely_lowdim": False,
                    "label": "ERROR",
                    "note": f"ffmpeg/ffprobe error: {e}",
                }
                upsert_row(rows, index, err_row)
                processed += 1  # counts as "processed work" for autosave
                print(f"ERROR     {fn}  ({e})")

            # autosave every N processed tracks
            if save_every > 0 and (processed % save_every == 0):
                write_csv_atomic(rows, out_csv)
                print(
                    f"Autosaved -> {out_csv}  (processed={processed}, skipped={skipped})"
                )

    # final write
    write_csv_atomic(rows, out_csv)

    print(f"\nWrote: {out_csv}")
    print(
        f"Processed: {processed} | New: {new} | Updated: {updated} | Skipped: {skipped} | Errors: {errors}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Detect stereo-ish Atmos with recursion + resume CSV + autosave."
    )
    p.add_argument("folder", help="Folder to scan recursively")
    p.add_argument(
        "-o", "--out", default="atmos_filter_report.csv", help="Output CSV path"
    )

    p.add_argument(
        "--start", type=float, default=30.0, help="Base segment start (seconds)"
    )
    p.add_argument("--dur", type=float, default=60.0, help="Segment duration (seconds)")
    p.add_argument("--sr", type=int, default=48000, help="Decode sample rate")

    p.add_argument(
        "--segments", type=int, default=3, help="How many segments to sample per track"
    )
    p.add_argument(
        "--rank-agg",
        choices=["max", "median", "p75", "mean"],
        default="max",
        help="How to aggregate eff_rank across segments",
    )

    p.add_argument(
        "--rank-cutoff",
        type=float,
        default=DEFAULT_RANK_CUTOFF,
        help="Rank cutoff used for LOWDIM/FRONTFOCUS split",
    )
    p.add_argument(
        "--lowdim-nonfront-min",
        type=float,
        default=DEFAULT_LOWDIM_NONFRONT_MIN_DB,
        help="Require nonfront energy above this (dB) to call LOWDIM",
    )

    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable resume behavior and reprocess everything",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Autosave CSV every N processed tracks (0 disables)",
    )

    args = p.parse_args()

    scan_folder(
        folder=args.folder,
        out_csv=args.out,
        base_start_s=args.start,
        dur_s=args.dur,
        sr=args.sr,
        segments=max(1, args.segments),
        rank_cutoff=args.rank_cutoff,
        lowdim_nonfront_min_db=args.lowdim_nonfront_min,
        rank_agg=args.rank_agg,
        no_skip=args.no_skip,
        save_every=max(0, args.save_every),
    )


if __name__ == "__main__":
    main()
