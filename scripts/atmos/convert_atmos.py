import argparse
import time
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atmos.atmos_utils import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_CAVERNIZE_EXE,
    DEFAULT_EXTENSIONS,
    DEFAULT_FFMPEG_EXE,
    DEFAULT_FORCE_24_BIT,
    DEFAULT_SLEEP_BETWEEN_RUNS_SEC,
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
    print_failure_hint,
    resolve_executable,
    run_cavernize,
)

DEFAULT_INPUT_ROOT = r"D:\atmos_sources"
DEFAULT_OUTPUT_ROOT = r"D:\renders"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert Atmos sources with Cavernize into two target layouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cavernize-exe", default=DEFAULT_CAVERNIZE_EXE)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--target-output-layout",
        default=DEFAULT_TARGET_OUTPUT_LAYOUT,
        help=(
            "Cavernize output layout for multichannel renders "
            "(for example: 5.1, 7.1, 5.1.2, 7.1.4)."
        ),
    )
    parser.add_argument(
        "--target-input-layout",
        default=DEFAULT_TARGET_INPUT_LAYOUT,
        help=(
            "Cavernize output layout for the input-reference render "
            "(typically 2.0 stereo)."
        ),
    )
    parser.add_argument("--audio-format", default=DEFAULT_AUDIO_FORMAT)
    parser.add_argument(
        "--force-24-bit", action="store_true", default=DEFAULT_FORCE_24_BIT
    )
    parser.add_argument(
        "--sleep-between-runs-sec",
        type=float,
        default=DEFAULT_SLEEP_BETWEEN_RUNS_SEC,
    )
    parser.add_argument(
        "--skip-if-exists-over-bytes",
        type=int,
        default=DEFAULT_SKIP_IF_EXISTS_OVER_BYTES,
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated list of media file extensions.",
    )
    parser.add_argument(
        "--skip-input-processing",
        action="store_true",
        help="Skip rendering the input layout output (useful when it already exists).",
    )
    parser.add_argument(
        "--enable-stream-hashing",
        action="store_true",
        help=(
            "Use ffmpeg streamhash to deduplicate by audio stream before processing. "
            "Duplicates are skipped."
        ),
    )
    parser.add_argument(
        "--ffmpeg-exe",
        default=DEFAULT_FFMPEG_EXE,
        help="Path/name of ffmpeg executable used for stream hashing.",
    )
    parser.add_argument(
        "--stream-hash-file",
        default=None,
        help=(
            "Path to processed stream hash index file. "
            "Defaults to <output-root>/"
            f"{DEFAULT_STREAM_HASH_FILENAME}."
        ),
    )
    parser.add_argument(
        "--stream-hash-algorithm",
        default=DEFAULT_STREAM_HASH_ALGORITHM,
        help="Hash algorithm passed to ffmpeg streamhash (for example: sha256, md5).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    cavernize_exe = Path(args.cavernize_exe)
    extensions = normalize_extensions(args.extensions)
    output_layout_suffix = layout_to_suffix(args.target_output_layout)
    input_layout_suffix = layout_to_suffix(args.target_input_layout)

    if not cavernize_exe.exists():
        raise FileNotFoundError(f"Cavernize executable not found: {cavernize_exe}")
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    if not extensions:
        raise ValueError("At least one valid extension must be provided.")

    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = ""
    hash_file: Path | None = None
    hash_index: dict[str, str] = {}
    duplicate_skips = 0
    hash_additions = 0
    hash_failures = 0

    if args.enable_stream_hashing:
        ffmpeg_exe = resolve_executable(args.ffmpeg_exe)
        hash_file = (
            Path(args.stream_hash_file)
            if args.stream_hash_file
            else output_root / DEFAULT_STREAM_HASH_FILENAME
        )
        hash_index = load_hash_index(hash_file)
        print(
            f"Stream hashing enabled ({args.stream_hash_algorithm.lower()}); "
            f"loaded {len(hash_index)} hashes from: {hash_file}"
        )

    media_files = sorted(iter_media_files(input_root, extensions), key=lambda p: str(p).lower())
    print(f"Found {len(media_files)} media files under: {input_root}")

    for index, in_file in enumerate(media_files, 1):
        stream_hash = ""

        output_dir = mirrored_out_dir(input_root, output_root, in_file)
        base_name = in_file.stem

        output_output_layout = output_dir / f"{base_name}__{output_layout_suffix}.wav"
        output_input_layout = output_dir / f"{base_name}__{input_layout_suffix}.wav"

        logs_dir = output_dir / "_logs"
        log_output_layout = logs_dir / f"{base_name}__{output_layout_suffix}.log.txt"
        log_input_layout = logs_dir / f"{base_name}__{input_layout_suffix}.log.txt"

        print(f"\n[{index}/{len(media_files)}] {in_file}")

        if args.enable_stream_hashing:
            ok_hash, stream_hash, hash_info = compute_stream_hash(
                ffmpeg_exe=ffmpeg_exe,
                in_file=in_file,
                hash_algorithm=args.stream_hash_algorithm,
            )
            if not ok_hash:
                hash_failures += 1
                print(f"  - HASH SKIP ({hash_info}) -> {in_file.name}")
                continue

            if stream_hash in hash_index:
                duplicate_skips += 1
                prior_source = hash_index.get(stream_hash, "")
                print(
                    f"  - DUPLICATE SKIP ({args.stream_hash_algorithm.upper()}={stream_hash}) "
                    + (
                        f"already processed from: {prior_source}"
                        if prior_source
                        else "already present in hash index"
                    )
                )
                continue

        file_processed_successfully = True

        if is_big_enough(output_output_layout, args.skip_if_exists_over_bytes):
            print(
                f"  - {args.target_output_layout}: SKIP (exists) -> {output_output_layout.name}"
            )
        else:
            ok, info = run_cavernize(
                cavernize_exe=cavernize_exe,
                in_file=in_file,
                out_file=output_output_layout,
                target_layout=args.target_output_layout,
                audio_format=args.audio_format,
                force_24_bit=args.force_24_bit,
                log_file=log_output_layout,
            )
            if ok:
                print(
                    f"  - {args.target_output_layout}: ok -> {output_output_layout.name}"
                )
            else:
                file_processed_successfully = False
                print_failure_hint(
                    layout=args.target_output_layout,
                    info=info,
                    out_file=output_output_layout,
                    log_file=log_output_layout,
                )
            time.sleep(args.sleep_between_runs_sec)

        if args.skip_input_processing:
            print(f"  - {args.target_input_layout}: SKIP (--skip-input-processing)")
        elif is_big_enough(output_input_layout, args.skip_if_exists_over_bytes):
            print(
                f"  - {args.target_input_layout}: SKIP (exists) -> {output_input_layout.name}"
            )
        else:
            ok, info = run_cavernize(
                cavernize_exe=cavernize_exe,
                in_file=in_file,
                out_file=output_input_layout,
                target_layout=args.target_input_layout,
                audio_format=args.audio_format,
                force_24_bit=args.force_24_bit,
                log_file=log_input_layout,
            )
            if ok:
                print(
                    f"  - {args.target_input_layout}: ok -> {output_input_layout.name}"
                )
            else:
                file_processed_successfully = False
                print_failure_hint(
                    layout=args.target_input_layout,
                    info=info,
                    out_file=output_input_layout,
                    log_file=log_input_layout,
                )
            time.sleep(args.sleep_between_runs_sec)

        if args.enable_stream_hashing and stream_hash:
            if file_processed_successfully:
                append_hash_record(hash_file=hash_file, stream_hash=stream_hash, in_file=in_file)
                hash_index[stream_hash] = str(in_file)
                hash_additions += 1
                print(f"  - HASH ADDED ({args.stream_hash_algorithm.upper()}={stream_hash})")
            else:
                print("  - HASH NOT ADDED (processing failed; keeping index unchanged)")

    if args.enable_stream_hashing:
        print(
            "\nStream hash summary: "
            f"added={hash_additions}, "
            f"duplicate_skips={duplicate_skips}, "
            f"hash_failures={hash_failures}, "
            f"known_hashes={len(hash_index)}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
