"""Reference host inference loop driving the exported ONNX graph.

This is the executable specification for the eventual TypeScript/Swift/Kotlin host:
it reimplements everything that wraps the network -- the fixed-step flow-matching
solver, chunking, overlap-add, and recurrent memory threading -- in plain numpy,
calling the ONNX graph (via onnxruntime) for each network forward. The amplitude
lift and audio file I/O are reused from the ``stereo2spatial`` package and marked as
HOST-BOUNDARY: they are pure DSP that a production host must re-implement in its own
language, but reusing them here keeps the A/B against ``run_inference`` honest.

Numerics that MUST be ported faithfully (all reproduced below):
  * z0 Gaussian noise + per-chunk ordering (here via torch RNG so the A/B is tight;
    production hosts may use any Gaussian source -- the model is robust to the draw).
  * clean->velocity conversion ``v = (clean - z) / max(1 - t, eps)``.
  * fixed-step solver (heun, res6s) including the final clean projection.
  * memory tokens held constant across a chunk's solve, refreshed once at the end.
  * static-T contract: every chunk fed to the graph is exactly ``chunk_frames``;
    the tail is zero-padded and masked via ``valid_mask``.

Does not modify the ``stereo2spatial`` package; only imports from it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# HOST-BOUNDARY imports: pure DSP a production host re-implements natively.
from stereo2spatial.common.amplitude_lift import (  # noqa: E402
    undo_wavflow_output_lift,
    wavflow_source_transform,
)
from stereo2spatial.inference.audio import (  # noqa: E402
    read_audio_channels_first,
    write_audio_channels_first,
)
from stereo2spatial.inference.sampling import _res6s_tableau  # noqa: E402

# Must match stereo2spatial.inference.sampling.
CLEAN_PREDICTION_EPS = 1e-4
INTEGRATION_T_END = 1.0 - CLEAN_PREDICTION_EPS


def _preload_cuda_runtime() -> None:
    """Make the CUDA/cuDNN DLLs discoverable to onnxruntime before session creation.

    onnxruntime-gpu does not bundle the CUDA runtime; torch (cu13) ships it under
    torch/lib. Importing torch (done at module import) loads cuDNN/cuBLAS into the
    process, and we also add torch/lib to the DLL search path so a clean
    onnxruntime-gpu install resolves them without relying purely on load order.
    """
    import os

    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if lib_dir.is_dir():
        try:
            os.add_dll_directory(str(lib_dir))
        except (OSError, AttributeError):
            pass
    # NB: do NOT call ort.preload_dlls() here. With a CUDA-12 onnxruntime-gpu build
    # against torch cu13 it noisily fails to find cublas64_12 etc.; the session loads
    # fine anyway via the imported-torch cuDNN + the system CUDA 12 runtime on PATH.


class OnnxSpatialModel:
    """ONNX session + sidecar metadata, exposing one network forward."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        device: str = "cpu",
        intra_op_threads: int = 2,
    ):
        if str(device).lower() == "cuda":
            _preload_cuda_runtime()
        import onnxruntime as ort

        # Quiet the global logger too: the per-session level does not catch the
        # CUDA ScatterND kernel notices emitted from the strided RoPE write.
        ort.set_default_logger_severity(3)

        self.onnx_path = Path(onnx_path)
        meta_path = self.onnx_path.with_name(self.onnx_path.stem + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing sidecar metadata {meta_path}. Re-run export_onnx.py."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mem_path = self.onnx_path.with_name(self.meta["mem_init_file"])
        self.mem_init = np.load(mem_path).astype(np.float32)  # [M, H]

        if str(device).lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        if intra_op_threads and intra_op_threads > 0:
            session_options.intra_op_num_threads = int(intra_op_threads)
        # Silence the repeated "ScatterND with reduction='none'" CUDA notices from the
        # strided RoPE write; the even/odd indices are disjoint so it is correct.
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=providers,
        )
        active = self.session.get_providers()
        if str(device).lower() == "cuda" and active[:1] != ["CUDAExecutionProvider"]:
            # Refuse to run a 285M model on CPU by accident -- that is the path that
            # pins every core. With a clean single onnxruntime-gpu install,
            # get_providers() reliably drops CUDA here when its DLLs fail to load.
            raise RuntimeError(
                "Requested CUDA execution but onnxruntime did not load the CUDA EP "
                f"(active providers: {active}). Aborting to avoid a heavy CPU run. "
                "Likely a CUDA/cuDNN DLL mismatch; ensure torch (which ships the "
                "CUDA runtime) is importable and onnxruntime-gpu matches it."
            )
        print(f"[ref] onnxruntime providers active: {active}")
        model_meta = self.meta["model"]
        self.target_channels = int(model_meta["target_channels"])
        self.cond_channels = int(model_meta["cond_channels"])
        self.patch_size = int(model_meta["patch_size"])
        self.hidden_dim = int(model_meta["hidden_dim"])
        self.num_memory_tokens = int(model_meta["num_memory_tokens"])
        self.mix_style_dim = int(model_meta["mix_style_dim"])
        self.chunk_frames = int(self.meta["chunk_frames"])
        self._input_names = {i.name for i in self.session.get_inputs()}
        # fp16 graphs declare fp16 inputs; feed the float inputs at the right dtype.
        self._float_np = (
            np.float16
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else np.float32
        )

    def mem_init_batch(self) -> np.ndarray:
        """Return the learned initial memory tokens as ``[1, M, H]``."""
        return self.mem_init[None].copy()

    def forward(
        self,
        *,
        zt: np.ndarray,
        t: np.ndarray,
        z_cond: np.ndarray,
        valid_mask: np.ndarray,
        mix_style: np.ndarray,
        mix_style_mask: np.ndarray,
        mem: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fdt = self._float_np
        feed = {
            "zt": zt.astype(fdt),
            "t": t.astype(fdt),
            "z_cond": z_cond.astype(fdt),
            "mix_style": mix_style.astype(fdt),
            "mix_style_mask": mix_style_mask.astype(fdt),
            "mem": mem.astype(fdt),
        }
        # The flash/fused graph is exported --no-attn-mask (no valid_mask input).
        # Full chunks need no mask; the host overlaps the last window so songs longer
        # than one chunk never pad. Only feed valid_mask if the graph declares it.
        if "valid_mask" in self._input_names:
            feed["valid_mask"] = valid_mask.astype(np.bool_)
        clean, mem_out = self.session.run(None, feed)
        return clean, mem_out


# --- chunking / overlap-add (ports of stereo2spatial.common.windowing) ----------


def segment_starts(total_frames: int, window_frames: int, stride_frames: int) -> list[int]:
    """Left-aligned window starts with guaranteed tail coverage."""
    if total_frames <= window_frames:
        return [0]
    starts = list(range(0, total_frames - window_frames + 1, stride_frames))
    last_start = total_frames - window_frames
    if starts[-1] == last_start:
        return starts
    if len(starts) >= 2 and starts[-2] + window_frames >= last_start:
        starts[-1] = last_start
    else:
        starts.append(last_start)
    return starts


def chunk_weight(
    chunk_length: int, overlap_frames: int, is_first: bool, is_last: bool
) -> np.ndarray:
    """Triangular overlap-add weights for a chunk."""
    weight = np.ones(chunk_length, dtype=np.float32)
    if overlap_frames <= 0:
        return weight
    fade_len = min(overlap_frames, chunk_length)
    if fade_len <= 1:
        return weight
    if not is_first:
        weight[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    if not is_last:
        weight[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return weight


# --- patchify (ports of runner._patch_audio / _unpatch_audio) -------------------


def patchify(audio: np.ndarray, patch_size: int) -> tuple[np.ndarray, int]:
    """Channel-first ``[C,S]`` -> waveform patches ``[C,P,T]`` (+ sample count)."""
    channels, sample_count = audio.shape
    frame_count = max(1, math.ceil(sample_count / patch_size))
    padded = frame_count * patch_size
    if padded != sample_count:
        pad = np.zeros((channels, padded - sample_count), dtype=audio.dtype)
        audio = np.concatenate([audio, pad], axis=1)
    patches = audio.reshape(channels, frame_count, patch_size).transpose(0, 2, 1)
    return np.ascontiguousarray(patches), sample_count


def unpatch(patches: np.ndarray, sample_count: int) -> np.ndarray:
    """Waveform patches ``[C,P,T]`` -> channel-first ``[C,S]``."""
    channels = patches.shape[0]
    audio = patches.transpose(0, 2, 1).reshape(channels, -1)
    return np.ascontiguousarray(audio[:, :sample_count])


# --- solver ---------------------------------------------------------------------


def _velocity(clean: np.ndarray, z: np.ndarray, t_value: float) -> np.ndarray:
    denom = max(1.0 - float(t_value), CLEAN_PREDICTION_EPS)
    return (clean - z) / denom


def _sample_chunk_onnx(
    model: OnnxSpatialModel,
    *,
    cond_chunk: np.ndarray,
    valid_mask: np.ndarray,
    z0_chunk: np.ndarray,
    solver: str,
    solver_steps: int,
    mem: np.ndarray,
    mix_style: np.ndarray,
    mix_style_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one chunk's flow ODE via ONNX forwards; return (z1, mem_out)."""

    def predict_clean(t_value: float, z_state: np.ndarray) -> np.ndarray:
        t = np.full((1,), float(t_value), dtype=np.float32)
        clean, _ = model.forward(
            zt=z_state,
            t=t,
            z_cond=cond_chunk,
            valid_mask=valid_mask,
            mix_style=mix_style,
            mix_style_mask=mix_style_mask,
            mem=mem,
        )
        return clean

    solver = solver.lower()
    if solver == "heun":
        dt = INTEGRATION_T_END / float(solver_steps)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t0 = step_idx * dt
            t1 = (step_idx + 1) * dt
            v0 = _velocity(predict_clean(t0, z_state), z_state, t0)
            z_euler = z_state + dt * v0
            v1 = _velocity(predict_clean(t1, z_euler), z_euler, t1)
            z_state = z_state + 0.5 * dt * (v0 + v1)
        z1_chunk = predict_clean(INTEGRATION_T_END, z_state)
    elif solver in {"res6s", "res_6s"}:
        dt = INTEGRATION_T_END / float(solver_steps)
        c_nodes, a_matrix, b_weights = _res6s_tableau(dt)
        z_state = z0_chunk
        for step_idx in range(solver_steps):
            t_base = step_idx * dt
            velocities: list[np.ndarray] = []
            for stage_idx, stage_c in enumerate(c_nodes):
                if stage_idx == 0:
                    z_stage = z_state
                else:
                    acc = np.zeros_like(z_state)
                    for weight, vel in zip(a_matrix[stage_idx][:stage_idx], velocities):
                        if weight != 0.0:
                            acc = acc + float(weight) * vel
                    z_stage = z_state + dt * acc
                t_stage = min(t_base + stage_c * dt, INTEGRATION_T_END)
                velocities.append(
                    _velocity(predict_clean(t_stage, z_stage), z_stage, t_stage)
                )
            acc = np.zeros_like(z_state)
            for weight, vel in zip(b_weights, velocities):
                if weight != 0.0:
                    acc = acc + float(weight) * vel
            z_state = z_state + dt * acc
        z1_chunk = predict_clean(INTEGRATION_T_END, z_state)
    else:
        raise ValueError(f"reference host loop supports heun/res6s, got {solver!r}")

    # Refresh memory once from the final clean prediction (matches sampling.py).
    t_end = np.full((1,), INTEGRATION_T_END, dtype=np.float32)
    _, mem_out = model.forward(
        zt=z1_chunk,
        t=t_end,
        z_cond=cond_chunk,
        valid_mask=valid_mask,
        mix_style=mix_style,
        mix_style_mask=mix_style_mask,
        mem=mem,
    )
    return z1_chunk, mem_out


def generate_spatial_signal_onnx(
    model: OnnxSpatialModel,
    cond_signal: np.ndarray,
    *,
    chunk_frames: int,
    overlap_frames: int,
    solver: str,
    solver_steps: int,
    seed: int,
    mix_style: np.ndarray | None = None,
    noise_device: str = "cpu",
) -> np.ndarray:
    """Sample target waveform patches from conditioning patches with overlap-add.

    ``noise_device`` selects the torch RNG device for z0 so it matches the device
    ``run_inference`` used (CUDA and CPU RNG sequences differ for the same seed).
    """
    cond_channels, patch_size, total_frames = cond_signal.shape
    target_channels = model.target_channels

    stride_frames = chunk_frames - overlap_frames
    starts = segment_starts(total_frames, chunk_frames, stride_frames)

    assembled = np.zeros((target_channels, patch_size, total_frames), dtype=np.float32)
    weight_sum = np.zeros((total_frames,), dtype=np.float32)

    # z0 noise via torch RNG to match sampling.py exactly (same device + seed + order).
    generator = torch.Generator(device=noise_device).manual_seed(int(seed))
    z0_full = (
        torch.randn(
            (1, target_channels, patch_size, total_frames),
            generator=generator,
            device=noise_device,
        )
        .cpu()
        .numpy()
    )

    mem = model.mem_init_batch()

    if mix_style is None:
        mix_style_vec = np.zeros((1, model.mix_style_dim), dtype=np.float32)
        mix_style_mask = np.zeros((1,), dtype=np.float32)
    else:
        mix_style_vec = np.asarray(mix_style, dtype=np.float32).reshape(1, -1)
        mix_style_mask = np.ones((1,), dtype=np.float32)

    for idx, start in enumerate(starts):
        end = min(start + chunk_frames, total_frames)
        segment_length = min(chunk_frames, total_frames - start)
        cond_chunk = cond_signal[:, :, start:end]
        z0_chunk = z0_full[..., start:end]

        if segment_length < chunk_frames:
            pad_t = chunk_frames - segment_length
            cond_chunk = np.concatenate(
                [cond_chunk, np.zeros((cond_channels, patch_size, pad_t), np.float32)],
                axis=-1,
            )
            pad_z0 = (
                torch.randn(
                    (1, target_channels, patch_size, pad_t),
                    generator=generator,
                    device=noise_device,
                )
                .cpu()
                .numpy()
            )
            z0_chunk = np.concatenate([z0_chunk, pad_z0], axis=-1)

        cond_chunk = cond_chunk[None]  # [1, C, P, chunk_frames]
        valid_mask = np.zeros((1, chunk_frames), dtype=np.bool_)
        valid_mask[:, :segment_length] = True

        z1_chunk, mem = _sample_chunk_onnx(
            model,
            cond_chunk=cond_chunk,
            valid_mask=valid_mask,
            z0_chunk=z0_chunk,
            solver=solver,
            solver_steps=solver_steps,
            mem=mem,
            mix_style=mix_style_vec,
            mix_style_mask=mix_style_mask,
        )

        pred_chunk = z1_chunk[0, :, :, :segment_length]
        weight = chunk_weight(
            segment_length, overlap_frames, idx == 0, idx == len(starts) - 1
        )
        assembled[:, :, start:end] += pred_chunk * weight[None, None, :]
        weight_sum[start:end] += weight

    weight_sum = np.clip(weight_sum, 1e-8, None)
    return assembled / weight_sum[None, None, :]


# --- end-to-end (ports of runner.run_inference, ONNX network) -------------------


def _prepare_conditioning_audio(audio: np.ndarray, cond_channels: int) -> np.ndarray:
    """Map mono/stereo input to the model conditioning channel count."""
    if audio.shape[0] not in {1, 2}:
        raise ValueError(f"Input must be mono or stereo, got {audio.shape[0]}")
    if cond_channels == 1:
        return audio.mean(axis=0, keepdims=True) if audio.shape[0] == 2 else audio
    if cond_channels == 2:
        return np.broadcast_to(audio, (2, audio.shape[1])).copy() if audio.shape[0] == 1 else audio
    raise ValueError(f"cond_channels must be 1 or 2, got {cond_channels}")


def run_onnx_inference(
    *,
    onnx_path: str | Path,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    solver: str,
    solver_steps: int,
    overlap_seconds: float,
    seed: int,
    mix_style: np.ndarray | None = None,
    device: str = "cpu",
    intra_op_threads: int = 2,
) -> np.ndarray:
    """Run end-to-end ONNX inference and return the decoded ``[C,S]`` waveform."""
    model = OnnxSpatialModel(
        onnx_path, device=device, intra_op_threads=intra_op_threads
    )
    sample_rate = int(model.meta["sample_rate"])
    patch_size = model.patch_size
    lift = model.meta["amplitude_lift"]

    audio_t, actual_sr = read_audio_channels_first(  # HOST-BOUNDARY (I/O)
        Path(input_audio_path), target_sample_rate=sample_rate
    )
    audio = audio_t.float().numpy()
    conditioning = _prepare_conditioning_audio(audio, model.cond_channels)

    if bool(lift["enabled"]) and str(lift["mode"]) == "wavflow":
        cond_t, _gain = wavflow_source_transform(  # HOST-BOUNDARY (DSP)
            torch.from_numpy(conditioning),
            target_rms=float(lift["target_rms"]),
            scale=float(lift["scale"]),
            peak_limit=float(lift["peak_limit"]),
            eps=float(lift["eps"]),
        )
        conditioning = cond_t.numpy()

    cond_signal, sample_count = patchify(conditioning, patch_size)

    chunk_frames = model.chunk_frames
    patch_fps = float(actual_sr) / float(patch_size)
    overlap_frames = min(
        int(round(overlap_seconds * patch_fps)), max(0, chunk_frames - 1)
    )
    if cond_signal.shape[-1] < chunk_frames:
        # Static graph requires exactly chunk_frames; the generator pads, but warn
        # because run_inference would instead shrink the window here.
        print(
            f"[ref][warn] audio shorter than chunk_frames ({cond_signal.shape[-1]} < "
            f"{chunk_frames}); padding to the static graph size."
        )

    pred_signal = generate_spatial_signal_onnx(
        model,
        cond_signal,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        solver=solver,
        solver_steps=solver_steps,
        seed=seed,
        mix_style=mix_style,
        noise_device=device,
    )

    decoded = unpatch(pred_signal.astype(np.float32), sample_count)
    decoded_t = torch.from_numpy(decoded)
    if bool(lift["enabled"]) and str(lift["mode"]) == "wavflow":
        decoded_t = undo_wavflow_output_lift(  # HOST-BOUNDARY (DSP, incl. LUFS)
            decoded_t,
            scale=float(lift["scale"]),
            sample_rate=actual_sr,
            target_lufs=float(lift["output_lufs"]),
            eps=float(lift["eps"]),
        )

    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_audio_channels_first(  # HOST-BOUNDARY (I/O)
        output_path,
        decoded_t,
        actual_sr,
        model.meta.get("channel_order"),
    )
    return decoded_t.numpy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ONNX-driven reference inference (host-loop specification)."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--output-audio", required=True)
    parser.add_argument("--solver", default="heun", choices=("heun", "res6s", "res_6s"))
    parser.add_argument("--solver-steps", type=int, default=4)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--intra-op-threads", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decoded = run_onnx_inference(
        onnx_path=args.onnx,
        input_audio_path=args.input_audio,
        output_audio_path=args.output_audio,
        solver=args.solver,
        solver_steps=args.solver_steps,
        overlap_seconds=args.overlap_seconds,
        seed=args.seed,
        device=args.device,
        intra_op_threads=args.intra_op_threads,
    )
    print(f"[ref] wrote {args.output_audio} shape={tuple(decoded.shape)}")


if __name__ == "__main__":
    main()
