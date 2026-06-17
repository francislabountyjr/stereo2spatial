"""Verify an exported ONNX graph matches the PyTorch SpatialDiT forward pass.

This is the de-risking step before any JS/Swift host code: it proves the exported
graph is numerically faithful to ``run_inference``'s underlying network, and it
proves the ``mix_style_mask=0`` toggle exactly reproduces the ``mix_style=None``
(no-style) path the model was trained with.

Checks performed (all on identical, seeded random inputs):

1. torch vs ONNX, mix-style ACTIVE (mask=1) -> clean_prediction + mem_out match.
2. torch vs ONNX, mix-style OFF (mask=0)    -> clean_prediction + mem_out match.
3. torch-only equivalence: model(mix_style=None) == model(mix_style=X, mask=0).
   This is the guarantee that the static graph preserves the "omit mix-style"
   feature. Tolerance here is very tight (adding an exact-zero tensor is a no-op).

Exit code is non-zero if any check exceeds tolerance.

This script does not modify the ``stereo2spatial`` package; it only imports from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.export.export_onnx import (  # noqa: E402
    INPUT_NAMES,
    OnnxExportWrapper,
    assert_exportable_config,
    build_model,
    make_example_inputs,
)
from stereo2spatial.inference.checkpoint import (  # noqa: E402
    load_model_weights,
    resolve_checkpoint_path,
)
from stereo2spatial.training.config import load_config  # noqa: E402


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def _ort_session(onnx_path: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    # Parity is checked on CPU for determinism; CUDA EP can differ at ~1e-3.
    providers = ["CPUExecutionProvider"]
    print(f"[parity] onnxruntime providers available={available}, using={providers}")
    return ort.InferenceSession(onnx_path, providers=providers)


def _run_onnx(session, inputs: tuple[torch.Tensor, ...]) -> list[np.ndarray]:
    feed = {}
    for name, tensor in zip(INPUT_NAMES, inputs):
        array = tensor.detach().cpu().numpy()
        # ONNX bool tensors must stay bool; everything else as exported dtype.
        feed[name] = array
    return session.run(None, feed)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _report(label: str, torch_out, onnx_out, tol: float) -> bool:
    clean_diff = _max_abs_diff(_to_numpy(torch_out[0]), onnx_out[0])
    mem_diff = _max_abs_diff(_to_numpy(torch_out[1]), onnx_out[1])
    ok = clean_diff <= tol and mem_diff <= tol
    status = "PASS" if ok else "FAIL"
    print(
        f"[parity][{status}] {label}: "
        f"clean max|d|={clean_diff:.3e} mem max|d|={mem_diff:.3e} (tol={tol:.1e})"
    )
    return ok


def verify(
    *,
    config_path: str,
    checkpoint: str,
    onnx_path: str,
    chunk_frames: int,
    weights_source: str,
    tol: float,
    none_tol: float,
    seed: int,
) -> bool:
    config = load_config(config_path)
    model = build_model(config)
    assert_exportable_config(model)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=checkpoint, output_dir=config.output_dir
    )
    load_model_weights(
        model=model, checkpoint_path=checkpoint_path, weights_source=weights_source
    )
    model = model.float().eval()
    wrapper = OnnxExportWrapper(model).eval()

    session = _ort_session(onnx_path)

    all_ok = True
    with torch.no_grad():
        # 1. mix-style active
        active = make_example_inputs(
            model, chunk_frames=chunk_frames, seed=seed, mix_style_active=True
        )
        torch_active = wrapper(*active)
        onnx_active = _run_onnx(session, active)
        all_ok &= _report("mix-style ON (mask=1)", torch_active, onnx_active, tol)

        # 2. mix-style off (same inputs, mask flipped to 0)
        off = (*active[:5], torch.zeros_like(active[5]), active[6])
        torch_off = wrapper(*off)
        onnx_off = _run_onnx(session, off)
        all_ok &= _report("mix-style OFF (mask=0)", torch_off, onnx_off, tol)

        # 3. torch-only: mask=0 must equal the genuine mix_style=None path.
        zt, t, z_cond, valid_mask, mix_style, _, mem = active
        clean_none, mem_none = model(
            zt=zt,
            t=t,
            z_cond=z_cond,
            valid_mask=valid_mask,
            mix_style=None,
            mem=mem,
            return_mem=True,
        )
        none_clean_diff = _max_abs_diff(
            _to_numpy(clean_none), _to_numpy(torch_off[0])
        )
        none_mem_diff = _max_abs_diff(_to_numpy(mem_none), _to_numpy(torch_off[1]))
        none_ok = none_clean_diff <= none_tol and none_mem_diff <= none_tol
        status = "PASS" if none_ok else "FAIL"
        print(
            f"[parity][{status}] mask=0 == mix_style=None: "
            f"clean max|d|={none_clean_diff:.3e} mem max|d|={none_mem_diff:.3e} "
            f"(tol={none_tol:.1e})"
        )
        all_ok &= none_ok

    print(f"[parity] {'ALL CHECKS PASSED' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare an exported ONNX graph against the PyTorch forward pass."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True, help="Path to the exported .onnx file.")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        required=True,
        help="Must match the static frame count the .onnx was exported with.",
    )
    parser.add_argument("--weights-source", default="auto", choices=("auto", "ema", "student"))
    parser.add_argument(
        "--tol",
        type=float,
        default=2e-3,
        help="Max allowed abs diff between torch and ONNX (fp32 CPU is usually ~1e-5).",
    )
    parser.add_argument(
        "--none-tol",
        type=float,
        default=1e-6,
        help="Max allowed abs diff for the mask=0 vs mix_style=None equivalence.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ok = verify(
        config_path=args.config,
        checkpoint=args.checkpoint,
        onnx_path=args.onnx,
        chunk_frames=args.chunk_frames,
        weights_source=args.weights_source,
        tol=args.tol,
        none_tol=args.none_tol,
        seed=args.seed,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
