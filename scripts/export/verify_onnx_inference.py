"""End-to-end A/B: ONNX reference host loop vs the package run_inference.

This validates the WHOLE pipeline (amplitude lift -> chunked flow-matching solver ->
overlap-add -> amplitude un-lift), not just a single forward, by rendering the same
input two ways and comparing the output waveforms:

  * reference: scripts.export.onnx_infer_reference.run_onnx_inference (numpy + ONNX)
  * baseline:  stereo2spatial.inference.run_inference (PyTorch)

Both are forced onto CPU with the same seed so the z0 noise is identical, making any
real discrepancy in the solver/chunking/overlap-add port show up as waveform error.
A synthetic stereo clip is generated if ``--input`` is not supplied.

Reports max abs diff and SNR(dB) = 20*log10(||ref|| / ||ref - test||); passes when
SNR exceeds ``--min-snr-db``. Exit code is non-zero on failure.

Does not modify the ``stereo2spatial`` package; only imports from it.
"""

from __future__ import annotations

import argparse
import gc
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.export.onnx_infer_reference import run_onnx_inference  # noqa: E402
from stereo2spatial.inference import run_inference  # noqa: E402
from stereo2spatial.training.config import load_config  # noqa: E402


def synth_stereo_clip(
    path: Path, *, seconds: float, sample_rate: int, seed: int = 7
) -> None:
    """Write a deterministic stereo test clip (sines + light noise)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    left = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.1 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.3 * np.sin(2 * np.pi * 277.0 * t) + 0.1 * np.sin(2 * np.pi * 330.0 * t)
    noise = 0.02 * rng.standard_normal((n, 2))
    stereo = np.stack([left, right], axis=1) + noise
    stereo = (stereo / np.max(np.abs(stereo)) * 0.5).astype(np.float32)
    sf.write(str(path), stereo, sample_rate, subtype="FLOAT")


def _read_channels_first(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T  # [C, S]


def _snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    error = reference - test
    ref_norm = float(np.linalg.norm(reference))
    err_norm = float(np.linalg.norm(error))
    if err_norm == 0.0:
        return float("inf")
    if ref_norm == 0.0:
        return float("-inf")
    return 20.0 * np.log10(ref_norm / err_norm)


def verify(
    *,
    config_path: str,
    checkpoint: str,
    onnx_path: str,
    chunk_frames: int,
    weights_source: str,
    solver: str,
    solver_steps: int,
    overlap_seconds: float,
    seed: int,
    input_path: str | None,
    seconds: float,
    min_snr_db: float,
    max_abs_tol: float,
    device: str,
    cpu_threads: int,
    out_dir: str | None,
) -> bool:
    # Keep CPU pressure low even when the heavy matmuls run on the GPU.
    torch.set_num_threads(max(1, int(cpu_threads)))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    config = load_config(config_path)
    sample_rate = int(config.data.sample_rate)
    patch_size = int(config.model.patch_size)
    # chunk_seconds chosen so run_inference's chunk_frames == the static graph T.
    chunk_seconds = float(chunk_frames * patch_size) / float(sample_rate)

    if out_dir is not None:
        work_dir = Path(out_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="onnx_ab_"))
    if input_path is None:
        input_audio = work_dir / "input.wav"
        synth_stereo_clip(input_audio, seconds=seconds, sample_rate=sample_rate)
        print(f"[ab] synthesized test clip: {input_audio} ({seconds}s @ {sample_rate})")
    else:
        input_audio = Path(input_path)

    torch_out = work_dir / "torch_out.wav"
    onnx_out = work_dir / "onnx_out.wav"

    print(f"[ab] running PyTorch run_inference ({device}, EMA)...")
    run_inference(
        config=config,
        checkpoint=checkpoint,
        input_audio_path=input_audio,
        output_audio_path=torch_out,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        solver=solver,  # type: ignore[arg-type]
        solver_steps=solver_steps,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=seed,
        device=device,
        show_progress=False,
        normalize_peak=False,
        mix_style=None,
        mix_style_preset=None,
        weights_source=weights_source,  # type: ignore[arg-type]
    )

    # Release the PyTorch model/VRAM before loading the ONNX session so both 285M
    # models are never resident at once.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[ab] running ONNX reference host loop ({device})...")
    run_onnx_inference(
        onnx_path=onnx_path,
        input_audio_path=input_audio,
        output_audio_path=onnx_out,
        solver=solver,
        solver_steps=solver_steps,
        overlap_seconds=overlap_seconds,
        seed=seed,
        mix_style=None,
        device=device,
        intra_op_threads=cpu_threads,
    )

    reference = _read_channels_first(torch_out)
    test = _read_channels_first(onnx_out)
    if reference.shape != test.shape:
        print(f"[ab][FAIL] shape mismatch: torch={reference.shape} onnx={test.shape}")
        return False

    print(f"[ab] torch render: {torch_out}")
    print(f"[ab] onnx  render: {onnx_out}")
    max_abs = float(np.max(np.abs(reference - test)))
    snr = _snr_db(reference, test)
    peak = float(np.max(np.abs(reference)))
    ok = snr >= min_snr_db and max_abs <= max_abs_tol
    status = "PASS" if ok else "FAIL"
    print(
        f"[ab][{status}] solver={solver}/{solver_steps} "
        f"SNR={snr:.1f} dB (min {min_snr_db:.0f}) "
        f"max|d|={max_abs:.3e} (tol {max_abs_tol:.1e}) "
        f"ref_peak={peak:.3e}"
    )
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B the ONNX reference host loop against run_inference."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        required=True,
        help="Static T the .onnx was exported with (sets run_inference chunk size).",
    )
    parser.add_argument(
        "--weights-source", default="ema", choices=("auto", "ema", "student")
    )
    parser.add_argument("--solver", default="heun", choices=("heun", "res6s", "res_6s"))
    parser.add_argument("--solver-steps", type=int, default=4)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--input", default=None, help="Optional input audio; synthesized if omitted."
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--min-snr-db", type=float, default=40.0)
    parser.add_argument("--max-abs-tol", type=float, default=5e-3)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="Cap torch + onnxruntime CPU threads to keep the machine responsive.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Keep the torch/onnx renders here (else a temp dir is used).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ok = verify(
        config_path=args.config,
        checkpoint=args.checkpoint,
        onnx_path=args.onnx,
        chunk_frames=args.chunk_frames,
        weights_source=args.weights_source,
        solver=args.solver,
        solver_steps=args.solver_steps,
        overlap_seconds=args.overlap_seconds,
        seed=args.seed,
        input_path=args.input,
        seconds=args.seconds,
        min_snr_db=args.min_snr_db,
        max_abs_tol=args.max_abs_tol,
        device=args.device,
        cpu_threads=args.cpu_threads,
        out_dir=args.out_dir,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
