"""Training CLI for stereo2spatial."""

from __future__ import annotations

import argparse

from stereo2spatial.training.config import load_config
from stereo2spatial.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=("Train the stereo-to-spatial latent transformer model.")
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to YAML training config.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Optional checkpoint directory to resume from. "
            "Use 'latest' to pick the newest checkpoint under output_dir/checkpoints."
        ),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Optional model-only initialization checkpoint (weights only). "
            "Does not restore optimizer/global_step/RNG state."
        ),
    )
    return parser


def main() -> None:
    """Parse CLI arguments, load config, and launch training."""
    args = build_parser().parse_args()
    config = load_config(args.config)
    train(
        config,
        resume_from_checkpoint=args.resume_from,
        init_from_checkpoint=args.init_from,
    )
