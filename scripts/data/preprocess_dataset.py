import argparse
import csv
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atmos.atmos_utils import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_CAVERNIZE_EXE,
    DEFAULT_EXTENSIONS,
    DEFAULT_FFMPEG_EXE,
    DEFAULT_FORCE_24_BIT,
    DEFAULT_SKIP_IF_EXISTS_OVER_BYTES,
    DEFAULT_STREAM_HASH_ALGORITHM,
    DEFAULT_STREAM_HASH_FILENAME,
    DEFAULT_TARGET_INPUT_LAYOUT,
    DEFAULT_TARGET_OUTPUT_LAYOUT,
    append_hash_record,
    compute_stream_hash,
    is_big_enough,
    iter_media_files,
    layout_to_suffix,
    load_hash_index,
    mirrored_out_dir,
    normalize_extensions,
    resolve_executable,
    run_cavernize,
)
from stereo2spatial.codecs.ear_vae import (
    encode_channels_independent,
    get_default_device,
    load_vae,
    vae_encode,
)

try:
    import soundfile as sf
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing dependency: soundfile. Install with `pip install soundfile`."
    ) from error

try:
    import torchaudio
except ModuleNotFoundError:
    torchaudio = None

DEFAULT_INPUT_ROOT = r"D:\atmos_sources"
DEFAULT_DATASET_ROOT = r"D:\stereo2spatial_dataset"
DEFAULT_MANIFEST_FILENAME = "manifest.jsonl"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DEAD_CHANNEL_THRESHOLD = 1e-6
DEFAULT_MONO_REDUCTION = "mean"
DEFAULT_LATENT_DTYPE = "float32"
DEFAULT_KEEP_RENDERS = False
DEFAULT_ALLOW_DEAD_CHANNELS = False
DEFAULT_SHOW_PROGRESS = False
DEFAULT_SLEEP_BETWEEN_RENDERS_SEC = 0.25
DEFAULT_ALLOW_DUPLICATE_CHANNELS = False
DEFAULT_QC_ALLOWED_LABELS = {"OK", "FRONTFOCUS"}
DEFAULT_HASH_INDEX_BACKEND = "sqlite"
DEFAULT_STREAM_HASH_DB_FILENAME = "_processed_stream_hashes.sqlite3"
DEFAULT_FAILED_STREAM_HASH_DB_FILENAME = "_failed_stream_hashes.sqlite3"
DEFAULT_SAMPLE_ARTIFACT_MODE = "bundle"

TARGET_LATENT_FILENAME = "target_latent.pt"
SOURCE_STEREO_LATENT_FILENAME = "source_stereo_latent.pt"
SOURCE_MONO_LATENT_FILENAME = "source_mono_latent.pt"
SOURCE_DOWNMIX_LATENT_FILENAME = "source_downmix_latent.pt"
METADATA_FILENAME = "metadata.json"
SAMPLE_BUNDLE_FILENAME = "sample_bundle.pt"
AC3_MATRIX_VERSION = "ac3_fixed_v1"

SQRT_HALF = 1.0 / math.sqrt(2.0)

AC3_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "FL": (1.0, 0.0),
    "FR": (0.0, 1.0),
    "FC": (SQRT_HALF, SQRT_HALF),
    "LFE": (0.5, 0.5),
    "LFE2": (0.5, 0.5),
    "BL": (SQRT_HALF, 0.0),
    "BR": (0.0, SQRT_HALF),
    "SL": (SQRT_HALF, 0.0),
    "SR": (0.0, SQRT_HALF),
    "BC": (0.5, 0.5),
    "FLC": (SQRT_HALF, 0.0),
    "FRC": (0.0, SQRT_HALF),
    "TFL": (0.5, 0.0),
    "TFR": (0.0, 0.5),
    "TBL": (0.5, 0.0),
    "TBR": (0.0, 0.5),
    "TFC": (0.3535533905932738, 0.3535533905932738),
    "TC": (0.3535533905932738, 0.3535533905932738),
    "TBC": (0.3535533905932738, 0.3535533905932738),
}

LAYOUT_CHANNELS: dict[str, list[str]] = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"],
    "3.1": ["FL", "FR", "FC", "LFE"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "6.1(back)": ["FL", "FR", "FC", "LFE", "BL", "BR", "BC"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.0(front)": ["FL", "FR", "FC", "FLC", "FRC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
    "7.1.2": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR", "TFL", "TFR"],
    "7.1.4": [
        "FL",
        "FR",
        "FC",
        "LFE",
        "BL",
        "BR",
        "SL",
        "SR",
        "TFL",
        "TFR",
        "TBL",
        "TBR",
    ],
}

CHANNEL_COUNT_FALLBACKS: dict[int, list[str]] = {
    1: ["FC"],
    2: ["FL", "FR"],
    3: ["FL", "FR", "FC"],
    4: ["FL", "FR", "BL", "BR"],
    5: ["FL", "FR", "FC", "SL", "SR"],
    6: ["FL", "FR", "FC", "LFE", "SL", "SR"],
    8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    10: ["FL", "FR", "FC", "LFE", "SL", "SR", "TFL", "TFR", "TBL", "TBR"],
    12: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR", "TFL", "TFR", "TBL", "TBR"],
}


def parse_torch_dtype(raw_dtype: str) -> torch.dtype:
    normalized = raw_dtype.strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported latent dtype: {raw_dtype!r}")


def normalize_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def resolve_qc_source_path(raw_path: str, qc_csv_path: Path) -> Path:
    candidate = Path(raw_path.strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    from_qc_parent = (qc_csv_path.parent / candidate).resolve(strict=False)
    if from_qc_parent.exists():
        return from_qc_parent

    from_cwd = candidate.resolve(strict=False)
    if from_cwd.exists():
        return from_cwd
    return from_qc_parent


def load_qc_selected_media_files(
    qc_csv_path: Path,
    extensions: set[str],
) -> tuple[list[Path], dict[str, int]]:
    if not qc_csv_path.exists():
        raise FileNotFoundError(f"QC CSV not found: {qc_csv_path}")

    stats = {
        "rows_total": 0,
        "rows_unique_paths": 0,
        "rows_selected": 0,
        "rows_overwritten": 0,
        "rows_disallowed_label": 0,
        "rows_missing_path": 0,
        "rows_missing_file": 0,
        "rows_bad_extension": 0,
    }

    latest_by_path: dict[str, tuple[Path, str]] = {}

    with open(qc_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"QC CSV has no header row: {qc_csv_path}")

        field_lookup = {
            field.strip().lower(): field
            for field in reader.fieldnames
            if field is not None
        }
        path_field = field_lookup.get("path")
        label_field = field_lookup.get("label")
        if not path_field:
            raise ValueError(
                f"QC CSV is missing required column: 'path' ({qc_csv_path})"
            )
        if not label_field:
            raise ValueError(
                f"QC CSV is missing required column: 'label' ({qc_csv_path})"
            )

        for row in reader:
            stats["rows_total"] += 1

            raw_path = (row.get(path_field) or "").strip()
            if not raw_path:
                stats["rows_missing_path"] += 1
                continue

            resolved_path = resolve_qc_source_path(
                raw_path=raw_path,
                qc_csv_path=qc_csv_path,
            )
            label = (row.get(label_field) or "").strip().upper()
            path_key = normalize_path_key(resolved_path)
            if path_key in latest_by_path:
                stats["rows_overwritten"] += 1
            latest_by_path[path_key] = (resolved_path, label)

    stats["rows_unique_paths"] = len(latest_by_path)

    selected: list[Path] = []
    for path_key in sorted(latest_by_path.keys()):
        candidate_path, label = latest_by_path[path_key]

        if label not in DEFAULT_QC_ALLOWED_LABELS:
            stats["rows_disallowed_label"] += 1
            continue
        if candidate_path.suffix.lower() not in extensions:
            stats["rows_bad_extension"] += 1
            continue
        if not candidate_path.exists():
            stats["rows_missing_file"] += 1
            continue

        selected.append(candidate_path)

    stats["rows_selected"] = len(selected)
    return selected, stats


def normalize_stream_hash(raw_hash: str) -> str:
    candidate = raw_hash.strip().lower()
    if not re.fullmatch(r"[0-9a-f]+", candidate):
        raise ValueError(f"Invalid stream hash: {raw_hash!r}")
    return candidate


def resolve_hash_index_backend(raw_backend: str, hash_file: Path) -> str:
    backend = raw_backend.strip().lower()
    if backend not in {"auto", "sqlite", "tsv"}:
        raise ValueError(f"Unsupported hash-index backend: {raw_backend!r}")
    if backend != "auto":
        return backend
    if hash_file.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return "sqlite"
    if hash_file.suffix.lower() in {".tsv", ".txt"}:
        return "tsv"
    return "sqlite"


class TsvHashIndexStore:
    def __init__(self, hash_file: Path):
        self.hash_file = hash_file
        self._index = load_hash_index(hash_file)

    def size(self) -> int:
        return len(self._index)

    def get_source(self, stream_hash: str) -> Optional[str]:
        return self._index.get(stream_hash)

    def record(self, stream_hash: str, in_file: Path) -> None:
        if stream_hash in self._index:
            return
        append_hash_record(
            hash_file=self.hash_file,
            stream_hash=stream_hash,
            in_file=in_file,
        )
        self._index[stream_hash] = str(in_file)

    def close(self) -> None:
        return None


class SQLiteHashIndexStore:
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_file))
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_hash_index (
                stream_hash TEXT PRIMARY KEY,
                source_path TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def size(self) -> int:
        cursor = self.connection.execute("SELECT COUNT(*) FROM stream_hash_index")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def get_source(self, stream_hash: str) -> Optional[str]:
        cursor = self.connection.execute(
            "SELECT source_path FROM stream_hash_index WHERE stream_hash = ?",
            (stream_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return str(row[0])

    def record(self, stream_hash: str, in_file: Path) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO stream_hash_index (stream_hash, source_path) VALUES (?, ?)",
            (stream_hash, str(in_file)),
        )
        self.connection.commit()

    def import_tsv_if_empty(self, tsv_file: Path) -> int:
        if not tsv_file.exists():
            return 0
        if self.size() != 0:
            return 0

        cursor = self.connection.cursor()
        total_before = self.connection.total_changes
        batch: list[tuple[str, str]] = []

        with open(tsv_file, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                columns = stripped.split("\t", 1)
                raw_hash = columns[0].strip()
                if not re.fullmatch(r"[0-9a-fA-F]+", raw_hash):
                    continue
                stream_hash = raw_hash.lower()
                source = columns[1].strip() if len(columns) > 1 else ""
                batch.append((stream_hash, source))

                if len(batch) >= 1000:
                    cursor.executemany(
                        (
                            "INSERT OR IGNORE INTO stream_hash_index "
                            "(stream_hash, source_path) VALUES (?, ?)"
                        ),
                        batch,
                    )
                    batch.clear()

        if batch:
            cursor.executemany(
                (
                    "INSERT OR IGNORE INTO stream_hash_index "
                    "(stream_hash, source_path) VALUES (?, ?)"
                ),
                batch,
            )

        self.connection.commit()
        inserted = self.connection.total_changes - total_before
        return int(inserted)

    def close(self) -> None:
        self.connection.close()


def open_hash_index_store(
    backend: str,
    hash_file: Path,
    legacy_tsv_file: Optional[Path],
) -> tuple[SQLiteHashIndexStore | TsvHashIndexStore, str, int]:
    resolved_backend = resolve_hash_index_backend(backend, hash_file)
    if resolved_backend == "sqlite":
        store = SQLiteHashIndexStore(hash_file)
        imported = 0
        if legacy_tsv_file is not None:
            imported = store.import_tsv_if_empty(legacy_tsv_file)
        return store, resolved_backend, imported
    return TsvHashIndexStore(hash_file), resolved_backend, 0


class SQLiteFailureStore:
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_file))
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS failed_stream_hashes (
                stream_hash TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                fail_stage TEXT NOT NULL,
                fail_reason TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def size(self) -> int:
        cursor = self.connection.execute("SELECT COUNT(*) FROM failed_stream_hashes")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def get(self, stream_hash: str) -> Optional[dict[str, str]]:
        cursor = self.connection.execute(
            (
                "SELECT source_path, fail_stage, fail_reason, updated_utc "
                "FROM failed_stream_hashes WHERE stream_hash = ?"
            ),
            (stream_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "source_path": str(row[0]),
            "fail_stage": str(row[1]),
            "fail_reason": str(row[2]),
            "updated_utc": str(row[3]),
        }

    def record(
        self,
        stream_hash: str,
        in_file: Path,
        fail_stage: str,
        fail_reason: str,
    ) -> None:
        updated_utc = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            (
                "INSERT INTO failed_stream_hashes "
                "(stream_hash, source_path, fail_stage, fail_reason, updated_utc) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(stream_hash) DO UPDATE SET "
                "source_path=excluded.source_path, "
                "fail_stage=excluded.fail_stage, "
                "fail_reason=excluded.fail_reason, "
                "updated_utc=excluded.updated_utc"
            ),
            (
                stream_hash,
                str(in_file),
                str(fail_stage),
                str(fail_reason),
                updated_utc,
            ),
        )
        self.connection.commit()

    def remove(self, stream_hash: str) -> None:
        self.connection.execute(
            "DELETE FROM failed_stream_hashes WHERE stream_hash = ?",
            (stream_hash,),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def read_audio_channels_first(
    audio_path: Path,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    data, source_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data).transpose(0, 1).contiguous()
    if source_sample_rate == target_sample_rate:
        return audio, target_sample_rate

    if torchaudio is None:
        raise RuntimeError(
            f"Audio sample-rate mismatch for {audio_path} "
            f"({source_sample_rate} != {target_sample_rate}) and torchaudio is missing."
        )
    audio = torchaudio.functional.resample(
        waveform=audio, orig_freq=source_sample_rate, new_freq=target_sample_rate
    ).contiguous()
    return audio, target_sample_rate


def _layout_key_candidates(layout: str) -> list[str]:
    key = layout.strip().lower()
    condensed = key.replace(" ", "")
    return [
        key,
        condensed,
        condensed.replace("_", "."),
    ]


def channel_labels_for_layout(layout: str, num_channels: int) -> list[str]:
    for candidate in _layout_key_candidates(layout):
        if candidate in LAYOUT_CHANNELS:
            labels = LAYOUT_CHANNELS[candidate]
            if len(labels) == num_channels:
                return labels
            print(
                "  - LAYOUT WARN configured layout/channel mismatch: "
                f"layout={layout!r} implies {len(labels)} channels, "
                f"render has {num_channels}"
            )
            break

    if "+" in layout:
        split_labels = [part.strip().upper() for part in layout.split("+")]
        if len(split_labels) == num_channels and all(split_labels):
            return split_labels

    fallback = CHANNEL_COUNT_FALLBACKS.get(num_channels)
    if fallback is not None:
        return fallback
    return [f"C{i}" for i in range(num_channels)]


def downmix_multichannel_to_stereo_ac3(
    multichannel_audio: torch.Tensor, channel_labels: list[str]
) -> torch.Tensor:
    if multichannel_audio.dim() != 2:
        raise ValueError(
            "Expected multichannel audio shaped [C, S], "
            f"got {tuple(multichannel_audio.shape)}"
        )
    if multichannel_audio.shape[0] != len(channel_labels):
        raise ValueError(
            "Channel label count does not match audio channels: "
            f"{len(channel_labels)} vs {multichannel_audio.shape[0]}"
        )

    samples = multichannel_audio.shape[1]
    stereo = torch.zeros(
        (2, samples), dtype=multichannel_audio.dtype, device=multichannel_audio.device
    )
    for idx, label in enumerate(channel_labels):
        l_coeff, r_coeff = AC3_COEFFICIENTS.get(label.upper(), (0.5, 0.5))
        if l_coeff:
            stereo[0] += multichannel_audio[idx] * l_coeff
        if r_coeff:
            stereo[1] += multichannel_audio[idx] * r_coeff
    return stereo


def reduce_stereo_to_mono(stereo_audio: torch.Tensor, mode: str) -> torch.Tensor:
    if stereo_audio.dim() != 2 or stereo_audio.shape[0] != 2:
        raise ValueError(
            f"Expected stereo [2, S] tensor for mono reduction, got {tuple(stereo_audio.shape)}"
        )
    if mode == "mean":
        return stereo_audio.mean(dim=0, keepdim=True)
    if mode == "left":
        return stereo_audio[0:1, :]
    if mode == "right":
        return stereo_audio[1:2, :]
    raise ValueError("mono reduction must be one of: mean, left, right")


def dead_channel_indices(audio: torch.Tensor, threshold: float) -> list[int]:
    if audio.dim() != 2:
        raise ValueError(
            f"Expected [C, S] tensor for dead-channel check, got {tuple(audio.shape)}"
        )
    channel_peak = audio.abs().amax(dim=1)
    return [idx for idx, peak in enumerate(channel_peak.tolist()) if peak <= threshold]


def duplicate_channel_pairs(audio: torch.Tensor) -> list[tuple[int, int]]:
    if audio.dim() != 2:
        raise ValueError(
            f"Expected [C, S] tensor for duplicate-channel check, got {tuple(audio.shape)}"
        )

    duplicate_pairs: list[tuple[int, int]] = []
    channels = int(audio.shape[0])

    for left in range(channels):
        left_channel = audio[left].contiguous()
        for right in range(left + 1, channels):
            if torch.equal(left_channel, audio[right]):
                duplicate_pairs.append((left, right))

    return duplicate_pairs


def sample_dir_from_stream_hash(dataset_root: Path, stream_hash: str) -> Path:
    return dataset_root / "samples" / stream_hash[:2] / stream_hash[2:4] / stream_hash


def source_relpath_for_file(in_file: Path, input_root: Optional[Path]) -> str:
    if input_root is None:
        return str(in_file)
    try:
        return str(in_file.relative_to(input_root))
    except ValueError:
        return str(in_file)


def render_dir_for_file(
    in_file: Path,
    stream_hash: str,
    render_root: Path,
    input_root: Optional[Path],
) -> Path:
    if input_root is not None:
        try:
            return mirrored_out_dir(input_root, render_root, in_file)
        except ValueError:
            pass
    return render_root / "_by_hash" / stream_hash[:2] / stream_hash[2:4] / stream_hash


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


def all_latent_artifacts_exist(sample_dir: Path, sample_artifact_mode: str) -> bool:
    split_exists = split_sample_artifacts_exist(sample_dir)
    bundled_exists = bundled_sample_artifacts_exist(sample_dir)

    if sample_artifact_mode not in {"split", "bundle"}:
        raise ValueError(f"Unsupported sample artifact mode: {sample_artifact_mode!r}")
    return split_exists or bundled_exists


def append_manifest_record(manifest_file: Path, payload: dict) -> None:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def align_latent_lengths(
    target_latent: torch.Tensor,
    source_stereo_latent: torch.Tensor,
    source_mono_latent: torch.Tensor,
    source_downmix_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [
        target_latent.shape[-1],
        source_stereo_latent.shape[-1],
        source_mono_latent.shape[-1],
        source_downmix_latent.shape[-1],
    ]
    min_length = min(lengths)
    if min_length <= 0:
        raise RuntimeError(f"Invalid latent lengths (min={min_length}): {lengths}")
    if len(set(lengths)) == 1:
        return (
            target_latent,
            source_stereo_latent,
            source_mono_latent,
            source_downmix_latent,
        )
    print(f"  - LATENT ALIGN: trimming lengths {lengths} -> {min_length}")
    return (
        target_latent[..., :min_length],
        source_stereo_latent[..., :min_length],
        source_mono_latent[..., :min_length],
        source_downmix_latent[..., :min_length],
    )


def ensure_cdt_latent(latent: torch.Tensor, name: str) -> torch.Tensor:
    if latent.dim() == 2:
        return latent.unsqueeze(0).contiguous()
    if latent.dim() == 3:
        return latent.contiguous()
    raise ValueError(
        f"{name} must have shape [D, T] or [C, D, T], got {tuple(latent.shape)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render Atmos sources and build a resumable latent dataset for stereo->spatial modeling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cavernize-exe", default=DEFAULT_CAVERNIZE_EXE)
    parser.add_argument("--ffmpeg-exe", default=DEFAULT_FFMPEG_EXE)
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Root folder to scan when --qc-csv is not provided. "
            "Ignored when --qc-csv is provided."
        ),
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--render-root",
        default=None,
        help="Optional render workspace root. Defaults to <dataset-root>/_renders.",
    )
    parser.add_argument(
        "--target-output-layout",
        default=DEFAULT_TARGET_OUTPUT_LAYOUT,
        help=(
            "Cavernize output layout for multichannel target renders "
            "(for example: 5.1, 7.1, 5.1.2, 7.1.4)."
        ),
    )
    parser.add_argument(
        "--target-input-layout",
        default=DEFAULT_TARGET_INPUT_LAYOUT,
        help=(
            "Cavernize output layout for input-reference renders "
            "(typically 2.0 stereo)."
        ),
    )
    parser.add_argument("--audio-format", default=DEFAULT_AUDIO_FORMAT)
    parser.add_argument(
        "--force-24-bit", action="store_true", default=DEFAULT_FORCE_24_BIT
    )
    parser.add_argument(
        "--skip-if-exists-over-bytes",
        type=int,
        default=DEFAULT_SKIP_IF_EXISTS_OVER_BYTES,
        help="Render size threshold for considering existing render files valid.",
    )
    parser.add_argument(
        "--sleep-between-renders-sec",
        type=float,
        default=DEFAULT_SLEEP_BETWEEN_RENDERS_SEC,
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated media extensions.",
    )
    parser.add_argument(
        "--stream-hash-file",
        default=None,
        help=(
            "Path to dataset stream-hash index file. "
            "Defaults depend on backend: "
            f"sqlite -> <dataset-root>/{DEFAULT_STREAM_HASH_DB_FILENAME}, "
            f"tsv -> <dataset-root>/{DEFAULT_STREAM_HASH_FILENAME}."
        ),
    )
    parser.add_argument(
        "--failed-hash-file",
        default=None,
        help=(
            "Path to persistent failed-stream index file. "
            f"Defaults to <dataset-root>/{DEFAULT_FAILED_STREAM_HASH_DB_FILENAME}."
        ),
    )
    parser.add_argument(
        "--hash-index-backend",
        choices=["auto", "sqlite", "tsv"],
        default=DEFAULT_HASH_INDEX_BACKEND,
        help=(
            "Stream-hash index backend. "
            "Use sqlite for large datasets to avoid loading all hashes into RAM."
        ),
    )
    parser.add_argument(
        "--stream-hash-algorithm",
        default=DEFAULT_STREAM_HASH_ALGORITHM,
        help="Hash algorithm for ffmpeg streamhash (for example: sha256, md5).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry stream hashes that were previously recorded as failures. "
            "By default, known failures are skipped."
        ),
    )
    parser.add_argument("--manifest-file", default=DEFAULT_MANIFEST_FILENAME)
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate used for latent encoding.",
    )
    parser.add_argument(
        "--mono-reduction",
        choices=["mean", "left", "right"],
        default=DEFAULT_MONO_REDUCTION,
    )
    parser.add_argument(
        "--dead-channel-threshold",
        type=float,
        default=DEFAULT_DEAD_CHANNEL_THRESHOLD,
        help="Absolute peak threshold at or below which a channel is considered dead.",
    )
    parser.add_argument(
        "--allow-dead-channels",
        action="store_true",
        default=DEFAULT_ALLOW_DEAD_CHANNELS,
        help="Allow target samples containing dead/silent channels.",
    )
    parser.add_argument(
        "--allow-duplicate-channels",
        action="store_true",
        default=DEFAULT_ALLOW_DUPLICATE_CHANNELS,
        help="Allow target renders containing any identical channel pairs.",
    )
    parser.add_argument(
        "--sample-artifact-mode",
        choices=["split", "bundle"],
        default=DEFAULT_SAMPLE_ARTIFACT_MODE,
        help=(
            "How to persist per-sample artifacts: "
            "'split' writes 4 latent files + metadata.json, "
            "'bundle' writes 1 bundled latent file + metadata.json."
        ),
    )
    parser.add_argument(
        "--keep-renders",
        action="store_true",
        default=DEFAULT_KEEP_RENDERS,
        help="Persist rendered WAV files instead of deleting them after latent export.",
    )
    parser.add_argument(
        "--vae-checkpoint-path",
        required=True,
        help="Path to EAR_VAE checkpoint.",
    )
    parser.add_argument(
        "--vae-config-path",
        default=None,
        help="Optional path to VAE config JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for encoding (e.g., cuda, cpu). Defaults to auto.",
    )
    parser.add_argument(
        "--latent-dtype",
        default=DEFAULT_LATENT_DTYPE,
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Saved latent dtype on disk.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use stochastic sampling in VAE encoding.",
    )
    parser.add_argument(
        "--disable-chunked-encode",
        action="store_true",
        help="Disable chunked VAE encode path.",
    )
    parser.add_argument(
        "--encode-chunk-size-samples",
        type=int,
        default=None,
        help="Override VAE encode chunk size.",
    )
    parser.add_argument(
        "--encode-overlap-samples",
        type=int,
        default=None,
        help="Override VAE encode overlap size.",
    )
    parser.add_argument(
        "--disable-offload-latent-to-cpu",
        action="store_true",
        help="Keep intermediate latents on accelerator during encoding.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=DEFAULT_SHOW_PROGRESS,
        help="Enable tqdm progress bars inside chunked encode.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of input files processed.",
    )
    parser.add_argument(
        "--qc-csv",
        default=None,
        help=(
            "Optional QC report CSV (for example from detect_upmix.py). "
            "When set, only rows labeled OK or FRONTFOCUS are considered for processing."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    qc_mode = bool(args.qc_csv)

    input_root = Path(args.input_root).resolve(strict=False)
    dataset_root = Path(args.dataset_root).resolve(strict=False)
    render_root = (
        Path(args.render_root) if args.render_root else dataset_root / "_renders"
    )
    manifest_file = (
        Path(args.manifest_file)
        if Path(args.manifest_file).is_absolute()
        else dataset_root / args.manifest_file
    )
    default_hash_tsv_file = dataset_root / DEFAULT_STREAM_HASH_FILENAME
    default_hash_db_file = dataset_root / DEFAULT_STREAM_HASH_DB_FILENAME
    if args.stream_hash_file:
        hash_file = Path(args.stream_hash_file)
    else:
        requested_backend = args.hash_index_backend.strip().lower()
        if requested_backend == "tsv":
            hash_file = default_hash_tsv_file
        else:
            hash_file = default_hash_db_file
    failed_hash_file = (
        Path(args.failed_hash_file)
        if args.failed_hash_file
        else dataset_root / DEFAULT_FAILED_STREAM_HASH_DB_FILENAME
    )

    extensions = normalize_extensions(args.extensions)
    output_layout_suffix = layout_to_suffix(args.target_output_layout)
    input_layout_suffix = layout_to_suffix(args.target_input_layout)
    latent_dtype = parse_torch_dtype(args.latent_dtype)
    encode_device = torch.device(args.device) if args.device else get_default_device()

    if not extensions:
        raise ValueError("At least one valid extension must be provided.")

    cavernize_exe = Path(args.cavernize_exe)
    if not cavernize_exe.exists():
        raise FileNotFoundError(f"Cavernize executable not found: {cavernize_exe}")

    ffmpeg_exe = resolve_executable(args.ffmpeg_exe)

    dataset_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)

    hash_backend = resolve_hash_index_backend(args.hash_index_backend, hash_file)
    legacy_tsv_file: Optional[Path] = None
    if hash_backend == "sqlite":
        candidate_legacy_tsv = default_hash_tsv_file
        if candidate_legacy_tsv.resolve(strict=False) != hash_file.resolve(
            strict=False
        ):
            legacy_tsv_file = candidate_legacy_tsv

    hash_store, hash_backend, imported_hashes = open_hash_index_store(
        backend=args.hash_index_backend,
        hash_file=hash_file,
        legacy_tsv_file=legacy_tsv_file,
    )
    loaded_hash_count = hash_store.size()
    print(
        f"Loaded {loaded_hash_count} known stream hashes from: "
        f"{hash_file} (backend={hash_backend})"
    )
    if imported_hashes > 0:
        print(
            f"Imported {imported_hashes} stream hashes from legacy TSV: {legacy_tsv_file}"
        )
    failure_store = SQLiteFailureStore(failed_hash_file)
    print(
        f"Loaded {failure_store.size()} known failed stream hashes from: "
        f"{failed_hash_file}"
    )
    if args.retry_failed:
        print("Retrying known failures is enabled (--retry-failed).")

    print(f"Loading VAE on device={encode_device} ...")
    vae = load_vae(
        vae_checkpoint_path=args.vae_checkpoint_path,
        config_path=args.vae_config_path,
        device=encode_device,
        torch_dtype=torch.float32,
    )
    print("VAE loaded.")

    scan_input_root: Optional[Path] = None if qc_mode else input_root
    if scan_input_root is not None and not scan_input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {scan_input_root}")

    qc_csv_path: Optional[Path] = None
    qc_stats: Optional[dict[str, int]] = None
    if args.qc_csv:
        qc_csv_path = Path(args.qc_csv).resolve(strict=False)
        media_files, qc_stats = load_qc_selected_media_files(
            qc_csv_path=qc_csv_path,
            extensions=extensions,
        )
        print(f"Loaded QC CSV: {qc_csv_path}")
        print(
            "QC filter (labels=OK,FRONTFOCUS): "
            f"rows_total={qc_stats['rows_total']} "
            f"unique_paths={qc_stats['rows_unique_paths']} "
            f"selected={qc_stats['rows_selected']} "
            f"disallowed_label={qc_stats['rows_disallowed_label']} "
            f"missing_path={qc_stats['rows_missing_path']} "
            f"missing_file={qc_stats['rows_missing_file']} "
            f"bad_extension={qc_stats['rows_bad_extension']} "
            f"overwritten={qc_stats['rows_overwritten']}"
        )
    else:
        assert scan_input_root is not None
        media_files = sorted(
            iter_media_files(scan_input_root, extensions), key=lambda p: str(p).lower()
        )
        print(f"Found {len(media_files)} media files under: {scan_input_root}")

    if args.max_files is not None:
        media_files = media_files[: args.max_files]
    if qc_csv_path is not None:
        print(f"Selected {len(media_files)} media files from QC CSV")

    use_chunked_encode = not args.disable_chunked_encode
    offload_latent_to_cpu = not args.disable_offload_latent_to_cpu

    counters = {
        "success": 0,
        "skip_existing": 0,
        "skip_duplicate": 0,
        "skip_known_failure": 0,
        "skip_dead_channels": 0,
        "skip_duplicate_channels": 0,
        "skip_hash_failure": 0,
        "skip_render_failure": 0,
        "skip_error": 0,
    }

    for index, in_file in enumerate(media_files, 1):
        print(f"\n[{index}/{len(media_files)}] {in_file}")
        target_render: Optional[Path] = None
        stereo_render: Optional[Path] = None
        stream_hash: Optional[str] = None

        try:
            ok_hash, stream_hash, hash_info = compute_stream_hash(
                ffmpeg_exe=ffmpeg_exe,
                in_file=in_file,
                hash_algorithm=args.stream_hash_algorithm,
            )
            if not ok_hash:
                counters["skip_hash_failure"] += 1
                print(f"  - SKIP hash_failure ({hash_info}) -> {in_file.name}")
                continue
            stream_hash = normalize_stream_hash(stream_hash)

            sample_dir = sample_dir_from_stream_hash(dataset_root, stream_hash)
            metadata_path = sample_dir / METADATA_FILENAME
            bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
            latent_paths = {
                "target": sample_dir / TARGET_LATENT_FILENAME,
                "source_stereo": sample_dir / SOURCE_STEREO_LATENT_FILENAME,
                "source_mono": sample_dir / SOURCE_MONO_LATENT_FILENAME,
                "source_downmix": sample_dir / SOURCE_DOWNMIX_LATENT_FILENAME,
            }

            known_source = hash_store.get_source(stream_hash)
            if all_latent_artifacts_exist(
                sample_dir=sample_dir,
                sample_artifact_mode=args.sample_artifact_mode,
            ):
                counters["skip_existing"] += 1
                print(
                    f"  - SKIP existing ({args.stream_hash_algorithm.upper()}={stream_hash})"
                )
                if known_source is None:
                    hash_store.record(stream_hash=stream_hash, in_file=in_file)
                failure_store.remove(stream_hash)
                continue

            if known_source is not None and known_source != str(in_file):
                counters["skip_duplicate"] += 1
                print(
                    f"  - SKIP duplicate ({args.stream_hash_algorithm.upper()}={stream_hash}) "
                    f"already from {known_source}"
                )
                continue

            known_failure = failure_store.get(stream_hash)
            if known_failure is not None and not args.retry_failed:
                counters["skip_known_failure"] += 1
                print(
                    "  - SKIP known_failure "
                    f"({known_failure['fail_stage']}: {known_failure['fail_reason']}) "
                    f"last={known_failure['updated_utc']}"
                )
                continue

            render_dir = render_dir_for_file(
                in_file=in_file,
                stream_hash=stream_hash,
                render_root=render_root,
                input_root=scan_input_root,
            )
            render_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = render_dir / "_logs"

            render_stem = f"{in_file.stem}__{stream_hash[:12]}"
            target_render = render_dir / f"{render_stem}__{output_layout_suffix}.wav"
            stereo_render = render_dir / f"{render_stem}__{input_layout_suffix}.wav"
            target_log = logs_dir / f"{render_stem}__{output_layout_suffix}.log.txt"
            stereo_log = logs_dir / f"{render_stem}__{input_layout_suffix}.log.txt"

            if is_big_enough(target_render, args.skip_if_exists_over_bytes):
                print(
                    f"  - {args.target_output_layout}: RENDER SKIP (exists) -> {target_render.name}"
                )
            else:
                ok_render, info = run_cavernize(
                    cavernize_exe=cavernize_exe,
                    in_file=in_file,
                    out_file=target_render,
                    target_layout=args.target_output_layout,
                    audio_format=args.audio_format,
                    force_24_bit=args.force_24_bit,
                    log_file=target_log,
                )
                if not ok_render:
                    counters["skip_render_failure"] += 1
                    print(f"  - SKIP render_failure target ({info}) log={target_log}")
                    failure_store.record(
                        stream_hash=stream_hash,
                        in_file=in_file,
                        fail_stage="render_target",
                        fail_reason=str(info),
                    )
                    continue
                print(
                    f"  - {args.target_output_layout}: RENDER OK -> {target_render.name}"
                )
                time.sleep(args.sleep_between_renders_sec)

            target_audio, _ = read_audio_channels_first(
                audio_path=target_render,
                target_sample_rate=args.sample_rate,
            )
            if target_audio.shape[0] < 2:
                raise RuntimeError(
                    f"Expected multichannel target render, got {target_audio.shape[0]} channels"
                )

            dead_channels = dead_channel_indices(
                audio=target_audio, threshold=args.dead_channel_threshold
            )
            if dead_channels and not args.allow_dead_channels:
                counters["skip_dead_channels"] += 1
                print(
                    "  - SKIP dead_channel(s) "
                    f"indices={dead_channels} threshold={args.dead_channel_threshold}"
                )
                failure_store.record(
                    stream_hash=stream_hash,
                    in_file=in_file,
                    fail_stage="dead_channels",
                    fail_reason=(
                        f"indices={dead_channels} "
                        f"threshold={args.dead_channel_threshold}"
                    ),
                )
                continue
            if dead_channels:
                print(
                    "  - DEAD CHANNELS ALLOWED "
                    f"indices={dead_channels} threshold={args.dead_channel_threshold}"
                )

            duplicate_pairs = duplicate_channel_pairs(target_audio)
            if duplicate_pairs and not args.allow_duplicate_channels:
                counters["skip_duplicate_channels"] += 1
                print("  - SKIP duplicate_channel(s) " f"pairs={duplicate_pairs}")
                failure_store.record(
                    stream_hash=stream_hash,
                    in_file=in_file,
                    fail_stage="duplicate_channels",
                    fail_reason=f"pairs={duplicate_pairs}",
                )
                continue
            if duplicate_pairs:
                print(f"  - DUPLICATE CHANNELS ALLOWED pairs={duplicate_pairs}")

            if is_big_enough(stereo_render, args.skip_if_exists_over_bytes):
                print(
                    f"  - {args.target_input_layout}: RENDER SKIP (exists) -> {stereo_render.name}"
                )
            else:
                ok_render, info = run_cavernize(
                    cavernize_exe=cavernize_exe,
                    in_file=in_file,
                    out_file=stereo_render,
                    target_layout=args.target_input_layout,
                    audio_format=args.audio_format,
                    force_24_bit=args.force_24_bit,
                    log_file=stereo_log,
                )
                if not ok_render:
                    counters["skip_render_failure"] += 1
                    print(f"  - SKIP render_failure stereo ({info}) log={stereo_log}")
                    failure_store.record(
                        stream_hash=stream_hash,
                        in_file=in_file,
                        fail_stage="render_stereo",
                        fail_reason=str(info),
                    )
                    continue
                print(
                    f"  - {args.target_input_layout}: RENDER OK -> {stereo_render.name}"
                )
                time.sleep(args.sleep_between_renders_sec)

            stereo_audio, _ = read_audio_channels_first(
                audio_path=stereo_render,
                target_sample_rate=args.sample_rate,
            )
            if stereo_audio.shape[0] != 2:
                raise RuntimeError(
                    f"Expected stereo render to have 2 channels, got {stereo_audio.shape[0]}"
                )

            channel_labels = channel_labels_for_layout(
                layout=args.target_output_layout, num_channels=target_audio.shape[0]
            )
            source_downmix_audio = downmix_multichannel_to_stereo_ac3(
                multichannel_audio=target_audio, channel_labels=channel_labels
            )
            peak = source_downmix_audio.abs().amax().item()
            if peak > 1.0:
                source_downmix_audio = source_downmix_audio / peak * 0.99

            common_samples = min(
                target_audio.shape[1],
                stereo_audio.shape[1],
                source_downmix_audio.shape[1],
            )
            target_audio = target_audio[:, :common_samples]
            stereo_audio = stereo_audio[:, :common_samples]
            source_downmix_audio = source_downmix_audio[:, :common_samples]
            source_mono_audio = reduce_stereo_to_mono(
                stereo_audio=stereo_audio, mode=args.mono_reduction
            )

            print("  - Encoding target (per-channel, stored as [C,D,T]) ...")
            target_channel_latents = encode_channels_independent(
                vae=vae,
                audio=target_audio,
                sample_rate=args.sample_rate,
                use_sample=args.use_sample,
                use_chunked_encode=use_chunked_encode,
                chunk_size_samples=args.encode_chunk_size_samples,
                overlap_samples=args.encode_overlap_samples,
                offload_latent_to_cpu=offload_latent_to_cpu,
                show_progress=args.show_progress,
                device=encode_device,
            )
            target_latent = ensure_cdt_latent(target_channel_latents, "target_latent")

            print("  - Encoding source stereo/mono/downmix ...")
            source_stereo_latent = vae_encode(
                vae=vae,
                audio=stereo_audio,
                sample_rate=args.sample_rate,
                use_sample=args.use_sample,
                use_chunked_encode=use_chunked_encode,
                chunk_size_samples=args.encode_chunk_size_samples,
                overlap_samples=args.encode_overlap_samples,
                offload_latent_to_cpu=offload_latent_to_cpu,
                show_progress=args.show_progress,
                device=encode_device,
            )
            source_mono_latent = vae_encode(
                vae=vae,
                audio=source_mono_audio,
                sample_rate=args.sample_rate,
                use_sample=args.use_sample,
                use_chunked_encode=use_chunked_encode,
                chunk_size_samples=args.encode_chunk_size_samples,
                overlap_samples=args.encode_overlap_samples,
                offload_latent_to_cpu=offload_latent_to_cpu,
                show_progress=args.show_progress,
                device=encode_device,
            )
            source_downmix_latent = vae_encode(
                vae=vae,
                audio=source_downmix_audio,
                sample_rate=args.sample_rate,
                use_sample=args.use_sample,
                use_chunked_encode=use_chunked_encode,
                chunk_size_samples=args.encode_chunk_size_samples,
                overlap_samples=args.encode_overlap_samples,
                offload_latent_to_cpu=offload_latent_to_cpu,
                show_progress=args.show_progress,
                device=encode_device,
            )
            source_stereo_latent = ensure_cdt_latent(
                source_stereo_latent, "source_stereo_latent"
            )
            source_mono_latent = ensure_cdt_latent(
                source_mono_latent, "source_mono_latent"
            )
            source_downmix_latent = ensure_cdt_latent(
                source_downmix_latent, "source_downmix_latent"
            )

            (
                target_latent,
                source_stereo_latent,
                source_mono_latent,
                source_downmix_latent,
            ) = align_latent_lengths(
                target_latent=target_latent,
                source_stereo_latent=source_stereo_latent,
                source_mono_latent=source_mono_latent,
                source_downmix_latent=source_downmix_latent,
            )

            sample_dir.mkdir(parents=True, exist_ok=True)
            target_latent_cpu = target_latent.to(
                dtype=latent_dtype, device="cpu"
            ).contiguous()
            source_stereo_latent_cpu = source_stereo_latent.to(
                dtype=latent_dtype, device="cpu"
            ).contiguous()
            source_mono_latent_cpu = source_mono_latent.to(
                dtype=latent_dtype, device="cpu"
            ).contiguous()
            source_downmix_latent_cpu = source_downmix_latent.to(
                dtype=latent_dtype, device="cpu"
            ).contiguous()

            if args.sample_artifact_mode == "split":
                torch.save(target_latent_cpu, latent_paths["target"])
                torch.save(source_stereo_latent_cpu, latent_paths["source_stereo"])
                torch.save(source_mono_latent_cpu, latent_paths["source_mono"])
                torch.save(source_downmix_latent_cpu, latent_paths["source_downmix"])
                artifact_files = {
                    "target_latent": TARGET_LATENT_FILENAME,
                    "source_stereo_latent": SOURCE_STEREO_LATENT_FILENAME,
                    "source_mono_latent": SOURCE_MONO_LATENT_FILENAME,
                    "source_downmix_latent": SOURCE_DOWNMIX_LATENT_FILENAME,
                    "metadata": METADATA_FILENAME,
                }
            elif args.sample_artifact_mode == "bundle":
                torch.save(
                    {
                        "target_latent": target_latent_cpu,
                        "source_stereo_latent": source_stereo_latent_cpu,
                        "source_mono_latent": source_mono_latent_cpu,
                        "source_downmix_latent": source_downmix_latent_cpu,
                    },
                    bundle_path,
                )
                artifact_files = {
                    "sample_bundle": SAMPLE_BUNDLE_FILENAME,
                    "metadata": METADATA_FILENAME,
                }
            else:
                raise ValueError(
                    f"Unsupported sample artifact mode: {args.sample_artifact_mode!r}"
                )

            metadata = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "stream_hash_algorithm": args.stream_hash_algorithm,
                "stream_hash": stream_hash,
                "latent_layout": "c_d_t",
                "target_latent_layout": "c_d_t",
                "source_latent_layout": "c_d_t",
                "source_path": str(in_file),
                "source_relpath": source_relpath_for_file(
                    in_file=in_file,
                    input_root=scan_input_root,
                ),
                "target_layout": args.target_output_layout,
                "source_layout": args.target_input_layout,
                "target_render_path": str(target_render) if args.keep_renders else None,
                "source_render_path": str(stereo_render) if args.keep_renders else None,
                "sample_rate": args.sample_rate,
                "mono_reduction": args.mono_reduction,
                "allow_dead_channels": args.allow_dead_channels,
                "allow_duplicate_channels": args.allow_duplicate_channels,
                "dead_channel_threshold": args.dead_channel_threshold,
                "dead_channel_indices": dead_channels,
                "duplicate_channel_pairs": [list(pair) for pair in duplicate_pairs],
                "target_channels": int(target_audio.shape[0]),
                "target_channel_layout_config": args.target_output_layout,
                "target_channel_labels": channel_labels,
                "ac3_matrix_version": AC3_MATRIX_VERSION,
                "input_samples": int(common_samples),
                "target_latent_shape": [int(x) for x in target_latent.shape],
                "source_stereo_latent_shape": [
                    int(x) for x in source_stereo_latent.shape
                ],
                "source_mono_latent_shape": [int(x) for x in source_mono_latent.shape],
                "source_downmix_latent_shape": [
                    int(x) for x in source_downmix_latent.shape
                ],
                "latent_dtype": str(latent_dtype).replace("torch.", ""),
                "sample_artifact_mode": args.sample_artifact_mode,
                "files": artifact_files,
            }
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=True)

            append_manifest_record(
                manifest_file=manifest_file,
                payload={
                    "created_utc": metadata["created_utc"],
                    "stream_hash": stream_hash,
                    "latent_layout": metadata["latent_layout"],
                    "sample_dir": str(sample_dir.relative_to(dataset_root)),
                    "source_relpath": metadata["source_relpath"],
                    "target_channels": metadata["target_channels"],
                    "target_latent_shape": metadata["target_latent_shape"],
                    "source_stereo_latent_shape": metadata[
                        "source_stereo_latent_shape"
                    ],
                    "source_mono_latent_shape": metadata["source_mono_latent_shape"],
                    "source_downmix_latent_shape": metadata[
                        "source_downmix_latent_shape"
                    ],
                    "sample_artifact_mode": metadata["sample_artifact_mode"],
                    "dead_channel_indices": dead_channels,
                    "duplicate_channel_pairs": metadata["duplicate_channel_pairs"],
                },
            )
            hash_store.record(stream_hash=stream_hash, in_file=in_file)
            failure_store.remove(stream_hash)

            counters["success"] += 1
            print(
                "  - OK sample "
                f"hash={stream_hash} target={tuple(target_latent.shape)} "
                f"source={tuple(source_stereo_latent.shape)}"
            )

        except Exception as error:
            counters["skip_error"] += 1
            print(f"  - SKIP error ({type(error).__name__}): {error}")
            if stream_hash is not None:
                failure_store.record(
                    stream_hash=stream_hash,
                    in_file=in_file,
                    fail_stage=f"exception_{type(error).__name__}",
                    fail_reason=str(error),
                )
            continue
        finally:
            if not args.keep_renders:
                if target_render is not None:
                    safe_unlink(target_render)
                if stereo_render is not None:
                    safe_unlink(stereo_render)

    print("\nDataset preprocessing summary:")
    print(f"  - success={counters['success']}")
    print(f"  - skip_existing={counters['skip_existing']}")
    print(f"  - skip_duplicate={counters['skip_duplicate']}")
    print(f"  - skip_known_failure={counters['skip_known_failure']}")
    print(f"  - skip_dead_channels={counters['skip_dead_channels']}")
    print(f"  - skip_duplicate_channels={counters['skip_duplicate_channels']}")
    print(f"  - skip_hash_failure={counters['skip_hash_failure']}")
    print(f"  - skip_render_failure={counters['skip_render_failure']}")
    print(f"  - skip_error={counters['skip_error']}")
    print(f"  - sample_artifact_mode={args.sample_artifact_mode}")
    print(f"  - hash_index_backend={hash_backend}")
    if qc_csv_path is not None and qc_stats is not None:
        print(f"  - qc_csv={qc_csv_path}")
        print(f"  - qc_rows_total={qc_stats['rows_total']}")
        print(f"  - qc_rows_selected={qc_stats['rows_selected']}")
        print(f"  - qc_rows_disallowed_label={qc_stats['rows_disallowed_label']}")
        print(f"  - qc_rows_missing_file={qc_stats['rows_missing_file']}")
        print(f"  - qc_rows_bad_extension={qc_stats['rows_bad_extension']}")
    print(f"  - manifest={manifest_file}")
    print(f"  - hash_index={hash_file}")
    print(f"  - failed_hash_index={failed_hash_file}")
    hash_store.close()
    failure_store.close()
    print("Done.")


if __name__ == "__main__":
    main()
