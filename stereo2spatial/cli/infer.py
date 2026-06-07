"""Inference CLI for stereo2spatial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from stereo2spatial.inference import run_inference
from stereo2spatial.inference.export_bundle import (
    DEFAULT_BUNDLE_OVERLAP_SECONDS,
    build_train_config_from_bundle_payload,
    load_inference_bundle_payload,
    resolve_inference_config_path,
)
from stereo2spatial.inference.runner import RequestedSolverName, WeightsSource
from stereo2spatial.training.config import TrainConfig, load_config

SOLVER_CHOICES = (
    "auto",
    "dopri5",
    "heun",
    "euler",
    "unipc",
    "res6s",
    "res_6s",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
)


def _safe_print(message: str) -> None:
    """Print paths safely on Windows consoles that are not UTF-8."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="replace").decode(encoding))


def resolve_cli_config_path(
    *,
    config: str | None,
    checkpoint: str,
    default_config: str = "configs/train.yaml",
) -> str:
    """Resolve CLI config path."""
    if config is not None and str(config).strip():
        return str(config)

    inferred = resolve_inference_config_path(checkpoint)
    if inferred is not None:
        return str(inferred)
    return default_config


def _load_runtime_config_and_bundle_payload(
    config_path: str | Path,
) -> tuple[TrainConfig, dict[str, Any] | None]:
    resolved_path = Path(config_path)
    try:
        bundle_payload = load_inference_bundle_payload(resolved_path)
    except Exception:
        return load_config(resolved_path), None
    return (
        build_train_config_from_bundle_payload(
            bundle_payload,
            bundle_root=resolved_path.parent,
        ),
        bundle_payload,
    )


def _resolve_runtime_arg(
    *,
    explicit_value: Any,
    bundle_payload: dict[str, Any] | None,
    section_name: str,
    key: str,
    fallback: Any,
) -> Any:
    if explicit_value is not None:
        return explicit_value
    if bundle_payload is not None:
        section = (
            bundle_payload if not section_name else bundle_payload.get(section_name)
        )
        if isinstance(section, dict) and key in section:
            return section[key]
    return fallback


def _parse_mix_style_json(raw: str | None) -> list[float] | dict[str, float] | None:
    """Parse optional normalized mix-style conditioning from CLI JSON."""
    if raw is None or not str(raw).strip():
        return None
    payload = json.loads(raw)
    if isinstance(payload, list):
        return [float(value) for value in payload]
    if isinstance(payload, dict):
        return {str(key): float(value) for key, value in payload.items()}
    raise TypeError("--mix-style-json must be a JSON list or object")


def _add_model_and_io_args(parser: argparse.ArgumentParser) -> None:
    """Register model/checkpoint/input-output CLI arguments."""
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to training config used to build the model. If omitted, infer.py "
            "will try to resolve config from an exported bundle or checkpoint path "
            "before falling back to configs/train.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint path. Accepts an exported bundle directory, an Accelerate "
            "checkpoint directory (step_XXXXXXX), a .pt/.pth state-dict file, a "
            ".safetensors state-dict file, or 'latest' to use the newest checkpoint "
            "in output_dir/checkpoints."
        ),
    )
    parser.add_argument(
        "--input-audio",
        required=True,
        help="Path to mono or stereo input audio file.",
    )
    parser.add_argument(
        "--output-audio",
        required=True,
        help="Path to output spatial multichannel WAV file.",
    )


def _add_sampler_args(parser: argparse.ArgumentParser) -> None:
    """Register waveform-patch sampler and solver-related CLI arguments."""
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=(
            "Target sample rate for input load/resample and output write. Defaults to "
            "bundle sample_rate when available, otherwise 48000."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help=(
            "Inference chunk length in seconds. "
            "Defaults to data.segment_seconds from the config."
        ),
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=None,
        help=(
            "Chunk overlap in seconds for waveform-patch stitching. Defaults to 2.0."
        ),
    )
    parser.add_argument(
        "--solver",
        default=None,
        choices=SOLVER_CHOICES,
        help=("Sampler for waveform trajectory integration. " "'auto' selects heun."),
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=None,
        help=(
            "Step count for fixed-step solvers "
            "(heun/euler/unipc/res6s/midpoint/rk4/adams). "
            "Ignored by adaptive solvers like dopri5. "
            "Defaults to 64."
        ),
    )
    parser.add_argument("--solver-rtol", type=float, default=None)
    parser.add_argument("--solver-atol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--mix-style-json",
        default=None,
        help=(
            "Optional normalized mix-style conditioning as a JSON list in model "
            "order or a JSON object keyed by mix-style name. Defaults to omitted."
        ),
    )


def _add_runtime_and_reporting_args(parser: argparse.ArgumentParser) -> None:
    """Register runtime device and report-output CLI arguments."""
    parser.add_argument(
        "--weights-source",
        default="auto",
        choices=("auto", "ema", "student"),
        help=(
            "Which checkpoint weights to use: auto prefers EMA when present, "
            "ema requires EMA state, student uses model weights."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for inference (for example: cuda, cpu). Defaults to auto.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars where supported.",
    )
    parser.add_argument(
        "--normalize-peak",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override peak normalization. Defaults to false.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write an inference report JSON.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the inference CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate spatial multichannel WAV from mono/stereo audio "
            "using model.target_channels."
        )
    )
    _add_model_and_io_args(parser)
    _add_sampler_args(parser)
    _add_runtime_and_reporting_args(parser)
    return parser


def main() -> None:
    """Parse CLI arguments, run inference, and print/write the run report."""
    args = build_parser().parse_args()
    resolved_config_path = resolve_cli_config_path(
        config=args.config,
        checkpoint=args.checkpoint,
    )
    config, bundle_payload = _load_runtime_config_and_bundle_payload(
        resolved_config_path
    )
    sample_rate = int(
        _resolve_runtime_arg(
            explicit_value=args.sample_rate,
            bundle_payload=bundle_payload,
            section_name="",
            key="sample_rate",
            fallback=48000,
        )
    )
    chunk_seconds = args.chunk_seconds
    overlap_seconds = (
        float(args.overlap_seconds)
        if args.overlap_seconds is not None
        else DEFAULT_BUNDLE_OVERLAP_SECONDS
    )
    solver = cast(
        RequestedSolverName,
        str(args.solver) if args.solver is not None else "auto",
    )
    solver_steps = args.solver_steps
    solver_rtol = float(args.solver_rtol) if args.solver_rtol is not None else 1e-5
    solver_atol = float(args.solver_atol) if args.solver_atol is not None else 1e-5
    normalize_peak = (
        bool(args.normalize_peak) if args.normalize_peak is not None else False
    )
    mix_style = _parse_mix_style_json(args.mix_style_json)
    report = run_inference(
        config=config,
        checkpoint=args.checkpoint,
        input_audio_path=args.input_audio,
        output_audio_path=args.output_audio,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        solver=solver,
        solver_steps=solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        seed=args.seed,
        device=args.device,
        show_progress=args.show_progress,
        normalize_peak=normalize_peak,
        mix_style=mix_style,
        weights_source=cast(WeightsSource, args.weights_source),
    )
    report_values = dict(report)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)

    _safe_print("Inference complete:")
    for key in [
        "config_path",
        "input_audio_path",
        "output_audio_path",
        "checkpoint_path",
        "weights_source",
        "device",
        "input_channels",
        "sample_rate",
        "conditioning_signal_shape",
        "pred_signal_shape",
        "decoded_shape",
        "patch_fps",
        "chunk_frames",
        "overlap_frames",
        "solver",
        "seed",
        "mix_style",
    ]:
        if key == "config_path":
            _safe_print(f"  - config_path={resolved_config_path}")
        else:
            _safe_print(f"  - {key}={report_values[key]}")

    if args.report_json:
        _safe_print(f"  - report_json={report_path}")


if __name__ == "__main__":
    main()
