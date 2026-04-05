from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stereo2spatial.inference.export_bundle import (  # noqa: E402
    DEFAULT_CHANNEL_ORDER_7_1_4,
    export_model_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a stereo2spatial training checkpoint into an inference-ready "
            "model bundle suitable for Hugging Face upload and downstream inference."
        )
    )
    parser.add_argument(
        "--train-run-dir",
        type=Path,
        required=True,
        help="Training run directory that contains resolved_config.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help=(
            "Checkpoint selector: 'latest', a step directory name, or an explicit path "
            "to a checkpoint directory/file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to populate with the exported inference bundle.",
    )
    parser.add_argument(
        "--weights-source",
        default="auto",
        choices=("auto", "ema", "student"),
        help="Which checkpoint weights to export. 'auto' prefers EMA when available.",
    )
    parser.add_argument(
        "--channel-layout-name",
        default="7.1.4",
        help="Human-readable output layout name for bundle metadata.",
    )
    parser.add_argument(
        "--channel-order",
        nargs="+",
        default=DEFAULT_CHANNEL_ORDER_7_1_4,
        help="Ordered channel labels for the exported multichannel waveform layout.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Nominal audio sample rate recorded in the bundle metadata.",
    )
    parser.add_argument(
        "--vae-checkpoint-path",
        type=Path,
        default=None,
        help="Optional EAR-VAE checkpoint to copy into the bundle.",
    )
    parser.add_argument(
        "--vae-config-path",
        type=Path,
        default=None,
        help="Optional EAR-VAE config JSON to copy into the bundle.",
    )
    parser.add_argument(
        "--ear-vae-root",
        type=Path,
        default=None,
        help=(
            "Optional EAR_VAE repo root used to resolve bundled EAR-VAE assets when "
            "--vae-checkpoint-path/--vae-config-path are not provided."
        ),
    )
    parser.add_argument(
        "--exclude-vae",
        action="store_true",
        help="Do not bundle EAR-VAE assets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_model_bundle(
        train_run_dir=args.train_run_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        weights_source=args.weights_source,
        channel_layout_name=args.channel_layout_name,
        channel_order=list(args.channel_order),
        sample_rate=args.sample_rate,
        include_vae=not args.exclude_vae,
        ear_vae_root=args.ear_vae_root,
        vae_checkpoint_path=args.vae_checkpoint_path,
        vae_config_path=args.vae_config_path,
    )

    print("Export complete:")
    print(f"  - output_dir={result.output_dir}")
    print(f"  - checkpoint_path={result.checkpoint_path}")
    print(f"  - weights_source={result.weights_source}")
    print(f"  - config_path={result.config_path}")
    if result.vae_checkpoint_path is not None:
        print(f"  - vae_checkpoint_path={result.vae_checkpoint_path}")
    if result.vae_config_path is not None:
        print(f"  - vae_config_path={result.vae_config_path}")


if __name__ == "__main__":
    main()
