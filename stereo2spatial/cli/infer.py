"""Inference CLI for stereo2spatial."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from stereo2spatial.common.mix_style import (
    mix_style_preset_description,
    mix_style_preset_names,
    mix_style_preset_values,
)
from stereo2spatial.inference.cuda_graphs import (
    CudaGraphModelRunner,
    parse_cuda_graph_buckets,
)
from stereo2spatial.inference.export_bundle import (
    DEFAULT_BUNDLE_OVERLAP_SECONDS,
    DEFAULT_BUNDLE_SOLVER,
    DEFAULT_BUNDLE_SOLVER_ATOL,
    DEFAULT_BUNDLE_SOLVER_RTOL,
    DEFAULT_BUNDLE_SOLVER_STEPS,
    LEGACY_VAE_SAMPLE_RATE,
    build_train_config_from_bundle_payload,
    load_inference_bundle_payload,
    resolve_bundle_vae_paths,
    resolve_inference_config_path,
)
from stereo2spatial.inference.offline_batch import (
    DynamicInferenceJob,
    run_dynamic_folder_inference,
)
from stereo2spatial.inference.runner import (
    InferenceDTypeName,
    RequestedSolverName,
    WeightsSource,
    build_inference_session,
    run_inference_with_session,
)
from stereo2spatial.inference.sdpa import SDPABackendName, sdpa_backend_context
from stereo2spatial.modeling import is_legacy_vae_model
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
    "midpoint_rk2",
    "midpoint-rk2",
    "rk2",
    "rk4",
    "explicit_adams",
    "implicit_adams",
)
MIX_STYLE_PRESET_CHOICES = mix_style_preset_names()
_AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a"}
COMPILE_MODE_CHOICES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
INFERENCE_DTYPE_CHOICES = ("float32", "float16", "bfloat16", "auto")
SDPA_BACKEND_CHOICES = ("auto", "flash", "efficient", "math")


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
        # Older bundles stored runtime values at the top level. Keep accepting
        # those values while new bundles group their recommendations together.
        if key in bundle_payload:
            return bundle_payload[key]
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


def _iter_input_audio_paths(input_audio_path: Path) -> list[Path]:
    """Resolve a single input file or recursively discover audio files in a folder."""
    if input_audio_path.is_file():
        if input_audio_path.suffix.lower() not in _AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported input audio file: {input_audio_path}")
        return [input_audio_path]
    if not input_audio_path.is_dir():
        raise FileNotFoundError(f"Input audio path does not exist: {input_audio_path}")

    paths = sorted(
        path
        for path in input_audio_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported audio files found in: {input_audio_path}")
    return paths


def _default_output_suffix(target_channels: int) -> str:
    """Choose the default generated-audio container for folder-mode inference."""
    return ".flac" if int(target_channels) == 2 else ".wav"


def _resolve_output_audio_path(
    *,
    input_audio_path: Path,
    input_root_path: Path,
    output_root_path: Path,
    target_channels: int,
) -> Path:
    """Resolve one output path while preserving relative layout for folder input."""
    if input_root_path.is_file():
        return output_root_path
    relative_path = input_audio_path.relative_to(input_root_path)
    suffix = _default_output_suffix(target_channels)
    return (output_root_path / relative_path).with_suffix(suffix)


def _resolve_report_json_path(
    *,
    input_audio_path: Path,
    input_root_path: Path,
    report_root_path: Path | None,
) -> Path | None:
    """Resolve one report path while preserving relative layout for folder input."""
    if report_root_path is None:
        return None
    if input_root_path.is_file():
        return report_root_path
    relative_path = input_audio_path.relative_to(input_root_path)
    return (report_root_path / relative_path).with_suffix(".json")


def _print_mix_style_presets() -> None:
    """Print available mix-style presets and their normalized knob values."""
    _safe_print("Mix-style presets:")
    for name in MIX_STYLE_PRESET_CHOICES:
        _safe_print(f"  - {name}: {mix_style_preset_description(name)}")
        values = mix_style_preset_values(name)
        knobs = ", ".join(f"{key}={value:.2f}" for key, value in values.items())
        _safe_print(f"    {knobs}")


def _write_report_json_atomic(
    report_json_path: Path,
    report: Mapping[str, object],
) -> None:
    """Write a report JSON without exposing partially-written final files."""
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_json_path.with_name(
        f".{report_json_path.stem}.tmp-{uuid4().hex}{report_json_path.suffix}"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2, ensure_ascii=True)
        tmp_path.replace(report_json_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


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
        default=None,
        help=(
            "Checkpoint path. Accepts an exported bundle directory, an Accelerate "
            "checkpoint directory (step_XXXXXXX), a .pt/.pth state-dict file, a "
            ".safetensors state-dict file, or 'latest' to use the newest checkpoint "
            "in output_dir/checkpoints."
        ),
    )
    parser.add_argument(
        "--input-audio",
        default=None,
        help="Path to mono/stereo input audio file, or a folder of audio files.",
    )
    parser.add_argument(
        "--output-audio",
        default=None,
        help=(
            "Path to output audio file. When --input-audio is a folder, this is an "
            "output folder and the input tree is preserved."
        ),
    )


def _add_vae_args(parser: argparse.ArgumentParser) -> None:
    """Register optional EAR-VAE paths and chunk controls for legacy models."""
    parser.add_argument(
        "--vae-checkpoint-path",
        default=None,
        help=(
            "EAR-VAE checkpoint for legacy_vae models. Defaults to the vae/ asset "
            "inside an exported bundle when present."
        ),
    )
    parser.add_argument(
        "--vae-config-path",
        default=None,
        help="Optional EAR-VAE JSON config path.",
    )
    parser.add_argument(
        "--encode-chunk-size-samples",
        type=int,
        default=None,
        help=(
            "Legacy EAR-VAE encode chunk length in audio samples. Defaults to a "
            "VRAM-aware codec value."
        ),
    )
    parser.add_argument(
        "--encode-overlap-samples",
        type=int,
        default=None,
        help="Legacy EAR-VAE encode overlap in audio samples.",
    )
    parser.add_argument(
        "--decode-chunk-size-frames",
        type=int,
        default=2048,
        help="Legacy EAR-VAE decode chunk length in latent frames.",
    )
    parser.add_argument(
        "--decode-overlap-frames",
        type=int,
        default=256,
        help="Legacy EAR-VAE decode overlap in latent frames.",
    )
    parser.add_argument(
        "--disable-chunked-decode",
        action="store_true",
        help=(
            "Decode legacy EAR-VAE latents in one pass. This can substantially "
            "increase VRAM use for long or high-channel-count outputs."
        ),
    )


def _add_sampler_args(parser: argparse.ArgumentParser) -> None:
    """Register representation-neutral sampler and solver CLI arguments."""
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=(
            "Target sample rate for input load/resample and output write. Defaults to "
            "the bundle recommendation, then data.training_sample_rate when set, "
            "otherwise data.sample_rate. Legacy EAR-VAE inference is fixed at 48000."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help=(
            "Inference chunk length in seconds. "
            "Defaults to the bundle recommendation or training.window_seconds."
        ),
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=None,
        help=(
            "Chunk overlap in seconds for waveform/latent stitching. Defaults to the "
            "bundle recommendation or training.overlap_seconds."
        ),
    )
    parser.add_argument(
        "--solver",
        default=None,
        choices=SOLVER_CHOICES,
        help=(
            "Sampler for flow-trajectory integration. Defaults to the bundle or "
            "validation-generation recommendation; 'auto' selects heun."
        ),
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=None,
        help=(
            "Step count for fixed-step solvers "
            "(heun/euler/unipc/res6s/midpoint_rk2/rk4/adams). "
            "Ignored by adaptive solvers like dopri5. "
            "Defaults to the bundle or validation-generation recommendation."
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
    parser.add_argument(
        "--mix-style-preset",
        default=None,
        choices=MIX_STYLE_PRESET_CHOICES,
        help=(
            "Named mix-style preset to use for conditioning. Mutually exclusive "
            "with --mix-style-json."
        ),
    )
    parser.add_argument(
        "--list-mix-style-presets",
        action="store_true",
        help="Print available mix-style preset names, descriptions, and values.",
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
        "--compile-model",
        action="store_true",
        help=(
            "Compile the inference model with torch.compile after loading weights. "
            "This is usually worth it for folder inference or long songs."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=COMPILE_MODE_CHOICES,
        help="torch.compile mode used when --compile-model is set.",
    )
    parser.add_argument(
        "--inference-dtype",
        default="float32",
        choices=INFERENCE_DTYPE_CHOICES,
        help=(
            "Model/input dtype for inference. Use bfloat16 on CUDA for bf16-trained "
            "checkpoints when inference is memory-bandwidth-bound."
        ),
    )
    parser.add_argument(
        "--sdpa-backend",
        default="auto",
        choices=SDPA_BACKEND_CHOICES,
        help=(
            "Scaled-dot-product attention backend for inference. auto lets PyTorch "
            "choose; flash/efficient/math can be used for benchmarking."
        ),
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars where supported.",
    )
    parser.add_argument(
        "--dynamic-batching",
        action="store_true",
        help=(
            "Use dynamic folder batching across songs. Supports "
            "euler/heun/midpoint_rk2/res6s and does not support flow_one_step "
            "configs."
        ),
    )
    parser.add_argument(
        "--sampling-order",
        default="timestep-major",
        choices=("window-major", "timestep-major"),
        help=(
            "Long-sequence sampling order. timestep-major is the training-aligned "
            "default for sequential and dynamic inference; window-major retains "
            "the previous compatibility behavior."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum model-query batch size when --dynamic-batching is enabled.",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help=(
            "Use CUDA Graph replay for dynamic-batched model calls. Requires CUDA, "
            "fixed window shapes, and --dynamic-batching."
        ),
    )
    parser.add_argument(
        "--cuda-graph-buckets",
        default=None,
        help=(
            "Comma-separated CUDA Graph batch buckets, for example 1,2,4,8. "
            "Defaults to --max-batch-size when --cuda-graphs is set."
        ),
    )
    parser.add_argument(
        "--max-active-requests",
        type=int,
        default=None,
        help=(
            "Maximum decoded/patched songs active at once for --dynamic-batching. "
            "Defaults to --max-batch-size and is the primary timestep-major VRAM "
            "control for long songs."
        ),
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help=(
            "Worker count for dynamic folder decode/resample/conditioning prep. "
            "Legacy VAE models require 1. Only used with --dynamic-batching."
        ),
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=1,
        help=(
            "Worker count for dynamic folder decode/unpatch/normalization/audio "
            "writes. Legacy VAE models require 1. Only used with --dynamic-batching."
        ),
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help=(
            "Regenerate outputs even when the destination audio file already exists. "
            "By default existing outputs are skipped so folder inference can resume."
        ),
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
        help=(
            "Optional path to write an inference report JSON. When --input-audio is "
            "a folder, this is a report folder and the input tree is preserved."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the inference CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate spatial audio from mono/stereo input using "
            "model.target_channels."
        )
    )
    _add_model_and_io_args(parser)
    _add_vae_args(parser)
    _add_sampler_args(parser)
    _add_runtime_and_reporting_args(parser)
    return parser


def main() -> None:
    """Parse CLI arguments, run inference, and print/write the run report."""
    args = build_parser().parse_args()
    if args.list_mix_style_presets:
        _print_mix_style_presets()
        return
    if args.checkpoint is None or args.input_audio is None or args.output_audio is None:
        raise SystemExit(
            "--checkpoint, --input-audio, and --output-audio are required unless "
            "--list-mix-style-presets is used."
        )
    if args.mix_style_json is not None and args.mix_style_preset is not None:
        raise SystemExit("--mix-style-json and --mix-style-preset cannot both be used.")
    resolved_config_path = resolve_cli_config_path(
        config=args.config,
        checkpoint=args.checkpoint,
    )
    config, bundle_payload = _load_runtime_config_and_bundle_payload(
        resolved_config_path
    )
    bundled_vae_checkpoint, bundled_vae_config = resolve_bundle_vae_paths(
        args.checkpoint
    )
    resolved_vae_checkpoint = (
        Path(args.vae_checkpoint_path)
        if args.vae_checkpoint_path is not None
        else bundled_vae_checkpoint
    )
    resolved_vae_config = (
        Path(args.vae_config_path)
        if args.vae_config_path is not None
        else bundled_vae_config
    )
    legacy_vae = is_legacy_vae_model(config)
    configured_sample_rate = getattr(config.data, "training_sample_rate", None)
    if configured_sample_rate is None:
        configured_sample_rate = config.data.sample_rate
    sample_rate = int(
        _resolve_runtime_arg(
            explicit_value=args.sample_rate,
            bundle_payload=bundle_payload,
            section_name="inference",
            key="sample_rate",
            fallback=(LEGACY_VAE_SAMPLE_RATE if legacy_vae else configured_sample_rate),
        )
    )
    if legacy_vae and sample_rate != LEGACY_VAE_SAMPLE_RATE:
        raise SystemExit(
            "legacy_vae inference requires --sample-rate 48000 because the "
            "EAR-VAE codec is fixed at 48 kHz."
        )
    chunk_seconds = float(
        _resolve_runtime_arg(
            explicit_value=args.chunk_seconds,
            bundle_payload=bundle_payload,
            section_name="inference",
            key="chunk_seconds",
            fallback=getattr(
                config.training,
                "window_seconds",
                config.data.segment_seconds,
            ),
        )
    )
    overlap_seconds = float(
        _resolve_runtime_arg(
            explicit_value=args.overlap_seconds,
            bundle_payload=bundle_payload,
            section_name="inference",
            key="overlap_seconds",
            fallback=getattr(
                config.training,
                "overlap_seconds",
                DEFAULT_BUNDLE_OVERLAP_SECONDS,
            ),
        )
    )
    solver = cast(
        RequestedSolverName,
        str(
            _resolve_runtime_arg(
                explicit_value=args.solver,
                bundle_payload=bundle_payload,
                section_name="inference",
                key="solver",
                fallback=getattr(
                    config.training,
                    "validation_generation_solver",
                    DEFAULT_BUNDLE_SOLVER,
                ),
            )
        ),
    )
    solver_steps = int(
        _resolve_runtime_arg(
            explicit_value=args.solver_steps,
            bundle_payload=bundle_payload,
            section_name="inference",
            key="solver_steps",
            fallback=getattr(
                config.training,
                "validation_generation_solver_steps",
                DEFAULT_BUNDLE_SOLVER_STEPS,
            ),
        )
    )
    solver_rtol = float(
        _resolve_runtime_arg(
            explicit_value=args.solver_rtol,
            bundle_payload=bundle_payload,
            section_name="inference",
            key="solver_rtol",
            fallback=getattr(
                config.training,
                "validation_generation_solver_rtol",
                DEFAULT_BUNDLE_SOLVER_RTOL,
            ),
        )
    )
    solver_atol = float(
        _resolve_runtime_arg(
            explicit_value=args.solver_atol,
            bundle_payload=bundle_payload,
            section_name="inference",
            key="solver_atol",
            fallback=getattr(
                config.training,
                "validation_generation_solver_atol",
                DEFAULT_BUNDLE_SOLVER_ATOL,
            ),
        )
    )
    normalize_peak = (
        bool(args.normalize_peak) if args.normalize_peak is not None else False
    )
    mix_style = _parse_mix_style_json(args.mix_style_json)
    input_root_path = Path(args.input_audio)
    output_root_path = Path(args.output_audio)
    report_root_path = Path(args.report_json) if args.report_json else None
    input_audio_paths = _iter_input_audio_paths(input_root_path)
    folder_mode = input_root_path.is_dir()
    all_jobs = [
        DynamicInferenceJob(
            input_audio_path=input_audio_path,
            output_audio_path=_resolve_output_audio_path(
                input_audio_path=input_audio_path,
                input_root_path=input_root_path,
                output_root_path=output_root_path,
                target_channels=config.model.target_channels,
            ),
            report_json_path=_resolve_report_json_path(
                input_audio_path=input_audio_path,
                input_root_path=input_root_path,
                report_root_path=report_root_path,
            ),
        )
        for input_audio_path in input_audio_paths
    ]
    force_overwrite = bool(args.force_overwrite)
    skipped_jobs = [
        job
        for job in all_jobs
        if job.output_audio_path.exists() and not force_overwrite
    ]
    jobs_to_run = [
        job for job in all_jobs if force_overwrite or not job.output_audio_path.exists()
    ]
    if skipped_jobs:
        if folder_mode:
            _safe_print(f"Resume skip: existing_outputs={len(skipped_jobs)}")
        else:
            _safe_print(
                f"Resume skip: output exists: {skipped_jobs[0].output_audio_path}"
            )
            return
    if folder_mode and not jobs_to_run:
        _safe_print(
            "Inference folder complete: "
            f"generated=0 skipped={len(skipped_jobs)} errors=0 output_dir={output_root_path}"
        )
        if report_root_path is not None:
            _safe_print(f"  - report_dir={report_root_path}")
        return

    if args.dynamic_batching:
        if not folder_mode:
            raise SystemExit("--dynamic-batching currently requires folder input.")
        if bool(getattr(config.training, "flow_one_step", False)):
            raise SystemExit(
                "--dynamic-batching does not support training.flow_one_step=true."
            )
        if int(args.max_batch_size) <= 0:
            raise SystemExit("--max-batch-size must be > 0.")
        if args.max_active_requests is not None and int(args.max_active_requests) <= 0:
            raise SystemExit("--max-active-requests must be > 0 when provided.")
        if int(args.preprocess_workers) <= 0:
            raise SystemExit("--preprocess-workers must be > 0.")
        if int(args.postprocess_workers) <= 0:
            raise SystemExit("--postprocess-workers must be > 0.")
        if is_legacy_vae_model(config) and (
            int(args.preprocess_workers) != 1 or int(args.postprocess_workers) != 1
        ):
            raise SystemExit(
                "legacy_vae dynamic batching requires --preprocess-workers 1 "
                "and --postprocess-workers 1 because the VAE is shared."
            )
    elif args.cuda_graphs:
        raise SystemExit("--cuda-graphs requires --dynamic-batching.")

    if args.dynamic_batching:
        max_active_requests = (
            int(args.max_active_requests)
            if args.max_active_requests is not None
            else int(args.max_batch_size)
        )
        session = build_inference_session(
            config=config,
            checkpoint=args.checkpoint,
            device=args.device,
            weights_source=cast(WeightsSource, args.weights_source),
            compile_model=bool(args.compile_model),
            compile_mode=str(args.compile_mode),
            inference_dtype=cast(InferenceDTypeName, args.inference_dtype),
            vae_checkpoint_path=resolved_vae_checkpoint,
            vae_config_path=resolved_vae_config,
        )
        if args.compile_model:
            _safe_print(
                "Inference model compiled: "
                f"mode={session.compile_mode} device={session.run_device}"
            )
        cuda_graph_runner = None
        cuda_graph_buckets = None
        if args.cuda_graphs:
            if str(session.run_device).split(":", 1)[0] != "cuda":
                raise SystemExit("--cuda-graphs requires a CUDA inference device.")
            cuda_graph_buckets = parse_cuda_graph_buckets(
                args.cuda_graph_buckets,
                fallback_max_batch_size=int(args.max_batch_size),
            )
            cuda_graph_runner = CudaGraphModelRunner(
                session.model,
                bucket_sizes=cuda_graph_buckets,
                clone_outputs=False,
            )
        _safe_print(
            "Dynamic inference starting: "
            f"files={len(jobs_to_run)} skipped={len(skipped_jobs)} "
            f"max_batch_size={int(args.max_batch_size)} "
            f"max_active_requests={max_active_requests} "
            f"preprocess_workers={int(args.preprocess_workers)} "
            f"postprocess_workers={int(args.postprocess_workers)} "
            f"sampling_order={args.sampling_order} "
            f"cuda_graphs={bool(args.cuda_graphs)}"
            + (
                f" cuda_graph_buckets={','.join(str(size) for size in cuda_graph_buckets)}"
                if cuda_graph_buckets is not None
                else ""
            )
        )
        with sdpa_backend_context(cast(SDPABackendName, args.sdpa_backend)):
            dynamic_result = run_dynamic_folder_inference(
                session=session,
                jobs=jobs_to_run,
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                solver=solver,
                solver_steps=solver_steps,
                solver_rtol=solver_rtol,
                solver_atol=solver_atol,
                sampling_order=args.sampling_order,
                seed=args.seed,
                normalize_peak=normalize_peak,
                mix_style=mix_style,
                mix_style_preset=args.mix_style_preset,
                max_batch_size=int(args.max_batch_size),
                max_active_requests=max_active_requests,
                preprocess_workers=int(args.preprocess_workers),
                postprocess_workers=int(args.postprocess_workers),
                cuda_graph_runner=cuda_graph_runner,
                encode_chunk_size_samples=args.encode_chunk_size_samples,
                encode_overlap_samples=args.encode_overlap_samples,
                decode_chunk_size_frames=args.decode_chunk_size_frames,
                decode_overlap_frames=args.decode_overlap_frames,
                disable_chunked_decode=args.disable_chunked_decode,
            )
        report_path_by_output = {
            str(job.output_audio_path): job.report_json_path for job in jobs_to_run
        }
        for report in dynamic_result.reports:
            report_json_path = report_path_by_output.get(report["output_audio_path"])
            if report_json_path is not None:
                _write_report_json_atomic(report_json_path, report)
        for input_audio_path, exc in dynamic_result.errors:
            _safe_print(f"[infer_error] {input_audio_path}: {exc}")
        scheduler_stats = dynamic_result.stats.scheduler
        avg_batch_size = getattr(scheduler_stats, "average_batch_size", None)
        if avg_batch_size is None:
            avg_batch_size = (
                float(scheduler_stats.model_queries)
                / float(scheduler_stats.model_batches)
                if int(scheduler_stats.model_batches) > 0
                else 0.0
            )
        batch_hist = " ".join(
            f"{batch_size}:{count}"
            for batch_size, count in sorted(
                getattr(scheduler_stats, "batch_size_counts", {}).items()
            )
        )
        _safe_print(
            "Dynamic inference complete: "
            f"generated={len(dynamic_result.reports)} "
            f"skipped={len(skipped_jobs)} "
            f"errors={len(dynamic_result.errors)} output_dir={output_root_path}"
        )
        _safe_print(
            "  - scheduler="
            f"batches={scheduler_stats.model_batches} "
            f"queries={scheduler_stats.model_queries} "
            f"avg_batch={float(avg_batch_size):.2f} "
            f"max_batch={scheduler_stats.max_observed_batch_size} "
            f"windows={scheduler_stats.completed_controllers} "
            f"batch_hist=[{batch_hist}]"
        )
        if report_root_path is not None:
            _safe_print(f"  - report_dir={report_root_path}")
        if dynamic_result.errors:
            raise SystemExit(1)
        return

    session = build_inference_session(
        config=config,
        checkpoint=args.checkpoint,
        device=args.device,
        weights_source=cast(WeightsSource, args.weights_source),
        compile_model=bool(args.compile_model),
        compile_mode=str(args.compile_mode),
        inference_dtype=cast(InferenceDTypeName, args.inference_dtype),
        vae_checkpoint_path=resolved_vae_checkpoint,
        vae_config_path=resolved_vae_config,
    )
    if args.compile_model:
        _safe_print(
            "Inference model compiled: "
            f"mode={session.compile_mode} device={session.run_device}"
        )
    reports = []
    errors: list[tuple[Path, Exception]] = []
    total_inputs = len(jobs_to_run)
    with sdpa_backend_context(cast(SDPABackendName, args.sdpa_backend)):
        for index, job in enumerate(jobs_to_run, start=1):
            input_audio_path = job.input_audio_path
            output_audio_path = job.output_audio_path
            report_json_path = job.report_json_path
            if folder_mode:
                _safe_print(
                    f"[{index}/{total_inputs}] {input_audio_path} -> {output_audio_path}"
                )
            try:
                report = run_inference_with_session(
                    session=session,
                    input_audio_path=input_audio_path,
                    output_audio_path=output_audio_path,
                    sample_rate=sample_rate,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    solver=solver,
                    solver_steps=solver_steps,
                    solver_rtol=solver_rtol,
                    solver_atol=solver_atol,
                    seed=args.seed,
                    show_progress=args.show_progress,
                    normalize_peak=normalize_peak,
                    mix_style=mix_style,
                    mix_style_preset=args.mix_style_preset,
                    encode_chunk_size_samples=args.encode_chunk_size_samples,
                    encode_overlap_samples=args.encode_overlap_samples,
                    decode_chunk_size_frames=args.decode_chunk_size_frames,
                    decode_overlap_frames=args.decode_overlap_frames,
                    disable_chunked_decode=args.disable_chunked_decode,
                    sampling_order=args.sampling_order,
                )
            except Exception as exc:
                errors.append((input_audio_path, exc))
                _safe_print(f"[infer_error] {input_audio_path}: {exc}")
                continue

            reports.append(report)
            if report_json_path is not None:
                _write_report_json_atomic(report_json_path, report)

    if folder_mode:
        _safe_print(
            "Inference folder complete: "
            f"generated={len(reports)} skipped={len(skipped_jobs)} "
            f"errors={len(errors)} output_dir={output_root_path}"
        )
        if report_root_path is not None:
            _safe_print(f"  - report_dir={report_root_path}")
        if errors:
            raise SystemExit(1)
        return

    if errors:
        raise SystemExit(1)

    report_values = dict(reports[0])
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
        "sampling_order",
        "seed",
        "mix_style_preset",
        "mix_style",
        "compiled",
        "compile_mode",
        "inference_dtype",
    ]:
        if key == "config_path":
            _safe_print(f"  - config_path={resolved_config_path}")
        else:
            _safe_print(f"  - {key}={report_values[key]}")

    if report_root_path is not None:
        _safe_print(f"  - report_json={report_root_path}")


if __name__ == "__main__":
    main()
