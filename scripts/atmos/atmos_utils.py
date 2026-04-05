"""Shared Atmos conversion utilities.

This module provides reusable helpers and defaults consumed by script entrypoints
such as ``scripts/atmos/convert_atmos.py`` and
``scripts/data/preprocess_dataset.py``.

Layout defaults are intentionally defined here, while runtime overrides are
provided by each script's CLI flags (for example:
``--target-output-layout`` and ``--target-input-layout``).
"""

import re
import subprocess
from pathlib import Path
from shutil import which
from typing import Iterator

DEFAULT_CAVERNIZE_EXE = r"C:\Program Files\VoidX\Cavernize\CavernizeGUI.exe"
DEFAULT_FFMPEG_EXE = "ffmpeg"
DEFAULT_AUDIO_FORMAT = "PCM_Float"
DEFAULT_FORCE_24_BIT = False
DEFAULT_SLEEP_BETWEEN_RUNS_SEC = 1.0
DEFAULT_SKIP_IF_EXISTS_OVER_BYTES = 1 * 1024 * 1024
DEFAULT_STREAM_HASH_FILENAME = "_processed_stream_hashes.tsv"
DEFAULT_STREAM_HASH_ALGORITHM = "sha256"

DEFAULT_TARGET_OUTPUT_LAYOUT = "7.1.4"
DEFAULT_TARGET_INPUT_LAYOUT = "2.0"

DEFAULT_EXTENSIONS = {
    ".mkv",
    ".mka",
    ".mp4",
    ".mov",
    ".m2ts",
    ".ac3",
    ".eac3",
    ".ec3",
    ".thd",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
}


def normalize_extensions(raw_extensions: str) -> set[str]:
    normalized: set[str] = set()
    for part in raw_extensions.split(","):
        clean = part.strip().lower()
        if not clean:
            continue
        normalized.add(clean if clean.startswith(".") else f".{clean}")
    return normalized


def layout_to_suffix(layout: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", layout.strip()).strip("_")
    if not suffix:
        raise ValueError(f"Invalid layout value: {layout!r}")
    return suffix.lower()


def is_big_enough(path: Path, threshold_bytes: int) -> bool:
    try:
        return path.exists() and path.stat().st_size > threshold_bytes
    except OSError:
        return False


def resolve_executable(executable: str) -> str:
    candidate = Path(executable)
    if candidate.exists():
        return str(candidate)
    resolved = which(executable)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Executable not found: {executable}")


def load_hash_index(hash_file: Path) -> dict[str, str]:
    hash_index: dict[str, str] = {}
    if not hash_file.exists():
        return hash_index
    with open(hash_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split("\t", 1)
            stream_hash = columns[0].strip().lower()
            if not re.fullmatch(r"[0-9a-f]+", stream_hash):
                continue
            source = columns[1].strip() if len(columns) > 1 else ""
            if stream_hash not in hash_index:
                hash_index[stream_hash] = source
    return hash_index


def append_hash_record(hash_file: Path, stream_hash: str, in_file: Path) -> None:
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_file, "a", encoding="utf-8") as handle:
        handle.write(f"{stream_hash}\t{in_file}\n")


def compute_stream_hash(
    ffmpeg_exe: str,
    in_file: Path,
    hash_algorithm: str,
) -> tuple[bool, str, str]:
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(in_file),
        "-map",
        "0:a:0",
        "-f",
        "streamhash",
        "-hash",
        hash_algorithm,
        "-",
    ]
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as error:
        return False, "", f"exception={type(error).__name__}"

    combined_output = "\n".join(
        part for part in (process.stdout, process.stderr) if part
    )
    if process.returncode != 0:
        return False, "", f"exit={process.returncode}"

    for line in combined_output.splitlines():
        stripped = line.strip()
        if "=" not in stripped:
            continue
        candidate = stripped.rsplit("=", 1)[-1].strip().lower()
        if re.fullmatch(r"[0-9a-f]+", candidate):
            return True, candidate, "ok"
    return False, "", "no_hash_output"


def iter_media_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def run_cavernize(
    cavernize_exe: Path,
    in_file: Path,
    out_file: Path,
    target_layout: str,
    audio_format: str,
    force_24_bit: bool,
    log_file: Path,
) -> tuple[bool, str]:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(cavernize_exe),
        "-i",
        str(in_file),
        "-t",
        target_layout,
        "-f",
        audio_format,
    ]
    if force_24_bit:
        command.append("-f24")
    command += ["-o", str(out_file)]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(command) + "\n\n")
        log_handle.flush()

        try:
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            ok = (
                process.returncode == 0
                and out_file.exists()
                and out_file.stat().st_size > 1024
            )
            return ok, f"exit={process.returncode}"
        except Exception as error:
            log_handle.write(f"\nEXCEPTION: {error!r}\n")
            return False, f"exception={type(error).__name__}"


def mirrored_out_dir(input_root: Path, output_root: Path, in_file: Path) -> Path:
    return output_root / in_file.parent.relative_to(input_root)


def print_failure_hint(
    layout: str,
    info: str,
    out_file: Path,
    log_file: Path,
) -> None:
    print(f"  - {layout}: fail_{info} -> {out_file.name}")
    print(f"    log: {log_file}")
    print(
        "    hint: if the log mentions 'file in use' or 'access denied', run this command"
    )
    print("          from an elevated Command Prompt.")
    print(
        "    hint: you can also set Cavernize compatibility to always run as administrator."
    )
