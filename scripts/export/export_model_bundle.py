from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stereo2spatial.inference.export_bundle import (  # noqa: E402
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
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional resolved JSON or training YAML config. Defaults to "
            "<train-run-dir>/resolved_config.json."
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
        default=None,
        help=(
            "Human-readable output layout name for bundle metadata. Defaults from "
            "the target channel count: 2=binaural, 6=5.1 rear, 12=7.1.4."
        ),
    )
    parser.add_argument(
        "--channel-order",
        nargs="+",
        default=None,
        help="Ordered channel labels for the exported multichannel waveform layout.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=(
            "Recommended inference sample rate. Defaults to data.training_sample_rate "
            "when set, otherwise data.sample_rate. Legacy EAR-VAE bundles are fixed "
            "at 48000 Hz."
        ),
    )
    parser.add_argument(
        "--include-vae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Bundle EAR-VAE assets. Defaults to true for legacy_vae and false "
            "for waveform models."
        ),
    )
    parser.add_argument("--vae-checkpoint-path", type=Path, default=None)
    parser.add_argument("--vae-config-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_model_bundle(
        train_run_dir=args.train_run_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        weights_source=args.weights_source,
        channel_layout_name=args.channel_layout_name,
        channel_order=(
            list(args.channel_order) if args.channel_order is not None else None
        ),
        sample_rate=args.sample_rate,
        include_vae=args.include_vae,
        vae_checkpoint_path=args.vae_checkpoint_path,
        vae_config_path=args.vae_config_path,
        config_path=args.config,
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
