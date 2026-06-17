"""Export a SpatialDiT checkpoint to ONNX for on-device (web/mobile) inference.

This exports ONLY the per-forward network as a single ONNX graph with a fixed,
static input signature. The iterative flow-matching solver, amplitude lift, audio
chunking, and overlap-add stay in host code (TypeScript/Swift/Kotlin), matching how
diffusion/DiT models ship on-device.

Exported signature (all inputs always present, so the graph has no data-dependent
control flow)::

    inputs:  zt[1,2,P,T], t[1], z_cond[1,2,P,T], valid_mask[1,T],
             mix_style[1,K], mix_style_mask[1], mem[1,M,H]
    outputs: clean_prediction[1,2,P,T], mem_out[1,M,H]

The "no mix-style" feature is preserved exactly: set ``mix_style_mask = 0`` and the
style context is multiplied by zero before being added to the time embedding, which
is bit-identical to the ``mix_style=None`` path the model was trained with under
mix-style dropout. A neutral style *vector* would NOT be equivalent (the style MLP's
trained bias is non-zero), so the mask is the only faithful toggle.

By default this exports at a single STATIC frame count ``T = chunk_frames`` using the
TorchScript exporter. The last partial audio chunk is handled host-side by zero-pad +
``valid_mask`` (exactly as the Python sampler already does). Pass ``--dynamic-frames``
to instead use the dynamo exporter with ``T`` as a dynamic axis (more flexible, less
battle-tested op coverage).

This script does not modify the ``stereo2spatial`` package; it only imports from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stereo2spatial.inference.checkpoint import (  # noqa: E402
    load_model_weights,
    resolve_checkpoint_path,
)
from stereo2spatial.modeling import SpatialDiT  # noqa: E402
from stereo2spatial.training.config import TrainConfig, load_config  # noqa: E402

INPUT_NAMES = [
    "zt",
    "t",
    "z_cond",
    "valid_mask",
    "mix_style",
    "mix_style_mask",
    "mem",
]
OUTPUT_NAMES = ["clean_prediction", "mem_out"]


def _rope_export_friendly(
    x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
) -> torch.Tensor:
    """Drop-in for embeddings.apply_rotary_embedding that avoids ScatterND.

    The model applies RoPE with a strided write ``out[..., 0::2] = ...`` which exports
    to ONNX ``ScatterND`` (a slow atomic scatter — ~38% of onnxruntime runtime). This
    reshape/stack formulation is numerically identical (same interleaved-pair
    convention) but exports to fast Reshape/Mul/Concat ops. Used only at export time
    via monkeypatch; the model source is unchanged.
    """
    if rope is None:
        return x
    cos, sin = rope
    x_float = x.float()
    x_pairs = x_float.reshape(*x_float.shape[:-1], x_float.shape[-1] // 2, 2)
    even = x_pairs[..., 0]
    odd = x_pairs[..., 1]
    cos = cos.to(device=x.device)
    sin = sin.to(device=x.device)
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    out = torch.stack((rot_even, rot_odd), dim=-1).reshape(x_float.shape)
    return out.to(dtype=x.dtype)


def build_model(config: TrainConfig) -> SpatialDiT:
    """Build a SpatialDiT exactly as the inference runner does (no weights loaded)."""
    model_config = config.model
    return SpatialDiT(
        target_channels=model_config.target_channels,
        cond_channels=model_config.cond_channels,
        patch_size=model_config.patch_size,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        dropout=model_config.dropout,
        timestep_embed_dim=model_config.timestep_embed_dim,
        timestep_scale=model_config.timestep_scale,
        max_period=model_config.max_period,
        num_memory_tokens=getattr(model_config, "num_memory_tokens", 0),
        mix_style_dim=getattr(model_config, "mix_style_dim", 0),
        waveform_level_depth=getattr(model_config, "waveform_level_depth", 0),
        waveform_micro_patch_size=getattr(
            model_config, "waveform_micro_patch_size", 16
        ),
        waveform_hidden_dim=getattr(model_config, "waveform_hidden_dim", 16),
        waveform_num_heads=getattr(model_config, "waveform_num_heads", None),
        waveform_mlp_ratio=getattr(model_config, "waveform_mlp_ratio", 2.0),
        final_output_kernel_size=getattr(model_config, "final_output_kernel_size", 7),
        final_output_zero_init=getattr(model_config, "final_output_zero_init", False),
        rope_enabled=getattr(model_config, "rope_enabled", True),
        rope_theta=getattr(model_config, "rope_theta", 10000.0),
        activation_checkpointing=False,
    )


def assert_exportable_config(model: SpatialDiT) -> None:
    """Fail early for model configs this exporter does not yet wire up."""
    if model.num_memory_tokens <= 0:
        raise NotImplementedError(
            "This exporter assumes recurrent memory tokens (num_memory_tokens > 0). "
            "The target config uses 32. A mem-less variant would need a different "
            "input signature."
        )
    if model.mix_style_dim <= 0:
        raise NotImplementedError(
            "This exporter assumes mix_style_dim > 0 so mix-style can be toggled via "
            "mix_style_mask. The target config uses 10."
        )
    if model.amplitude_gain_conditioning:
        raise NotImplementedError(
            "amplitude_gain_conditioning=True is not wired into the export signature. "
            "The target config has it disabled; add an amplitude_gain input to support it."
        )
    if model.waveform_level_depth > 0:
        # Not a hard error: the waveform branch is exportable too, but it is inactive
        # for the target config and untested here. Surface it loudly.
        print(
            "[warn] waveform_level_depth > 0: the waveform-refinement branch will be "
            "included in the graph but has not been parity-checked by this tooling."
        )


class OnnxExportWrapper(nn.Module):
    """Static-signature wrapper around ``SpatialDiT.forward`` for ONNX export.

    Always passes ``valid_mask``, ``mix_style``, ``mix_style_mask`` and ``mem`` so the
    traced graph contains no input-presence branches. ``mix_style_mask`` is a [1]
    float in {0, 1}: 0 reproduces the ``mix_style=None`` (no-style) path exactly.
    """

    def __init__(self, model: SpatialDiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        mix_style: torch.Tensor,
        mix_style_mask: torch.Tensor,
        mem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model(
            zt=zt,
            t=t,
            z_cond=z_cond,
            valid_mask=valid_mask,
            mix_style=mix_style,
            mix_style_mask=mix_style_mask,
            mem=mem,
            return_mem=True,
        )


class OnnxExportWrapperNoMask(nn.Module):
    """Static-signature wrapper that drops ``valid_mask`` (passes ``None``).

    For full (unpadded) chunks the mask is a no-op, so this is numerically identical
    to the masked wrapper there. Dropping it removes the SDPA NaN-guard/mask ops,
    yielding a clean attention that the flash-MHA fusion (fuse_mha.py) can rewrite.
    The host must feed exactly ``chunk_frames`` per call and zero-pad the tail; the
    final padded chunk differs slightly (zero-pad keys participate in attention).
    """

    def __init__(self, model: SpatialDiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        mix_style: torch.Tensor,
        mix_style_mask: torch.Tensor,
        mem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model(
            zt=zt,
            t=t,
            z_cond=z_cond,
            valid_mask=None,
            mix_style=mix_style,
            mix_style_mask=mix_style_mask,
            mem=mem,
            return_mem=True,
        )


INPUT_NAMES_NO_MASK = [n for n in INPUT_NAMES if n != "valid_mask"]


def make_example_inputs(
    model: SpatialDiT,
    *,
    chunk_frames: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    valid_frames: int | None = None,
    mix_style_active: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Build one batch of example inputs matching the export signature.

    ``valid_frames`` (defaults to all ``chunk_frames``) controls how many leading
    frames the ``valid_mask`` marks as real; the rest emulate a zero-padded tail.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    valid_count = int(chunk_frames if valid_frames is None else valid_frames)
    if not 0 < valid_count <= chunk_frames:
        raise ValueError("valid_frames must be in (0, chunk_frames]")

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    zt = randn(1, model.target_channels, model.patch_size, chunk_frames)
    z_cond = randn(1, model.cond_channels, model.patch_size, chunk_frames)
    t = torch.full((1,), 0.5, device=device, dtype=dtype)

    valid_mask = torch.zeros((1, chunk_frames), device=device, dtype=torch.bool)
    valid_mask[:, :valid_count] = True

    mix_style = torch.rand(
        (1, model.mix_style_dim), generator=generator
    ).to(device=device, dtype=dtype)
    mix_style_mask = torch.full(
        (1,), 1.0 if mix_style_active else 0.0, device=device, dtype=dtype
    )

    mem = model.init_memory(batch_size=1, device=torch.device(device), dtype=dtype)
    assert mem is not None  # guaranteed by assert_exportable_config

    return zt, t, z_cond, valid_mask, mix_style, mix_style_mask, mem


def _data_attr(config: TrainConfig, name: str, default):
    return getattr(config.data, name, default)


def write_export_sidecars(
    *,
    config: TrainConfig,
    model: SpatialDiT,
    onnx_path: Path,
    chunk_frames: int,
    sample_rate: int,
) -> tuple[Path, Path]:
    """Write the host-side companions to the ONNX graph.

    The graph alone is not enough to run inference on-device. The host also needs
    (a) the learned initial memory tokens (the graph takes ``mem`` as an input, so
    something must seed it) and (b) the amplitude-lift / channel metadata that lives
    outside the network. These are emitted next to the .onnx as ``<stem>.mem_init.npy``
    and ``<stem>.meta.json``.
    """
    mem_init = model.mem_init
    if mem_init is None:
        raise RuntimeError("model has no mem_init to export")
    mem_path = onnx_path.with_name(onnx_path.stem + ".mem_init.npy")
    np.save(mem_path, mem_init.detach().cpu().float().numpy())

    channel_order = getattr(config.training, "downmix_channel_order", None)
    meta = {
        "format": "stereo2spatial_onnx_sidecar_v1",
        "onnx_file": onnx_path.name,
        "mem_init_file": mem_path.name,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "sample_rate": int(sample_rate),
        "chunk_frames": int(chunk_frames),
        "model": {
            "target_channels": model.target_channels,
            "cond_channels": model.cond_channels,
            "patch_size": model.patch_size,
            "hidden_dim": model.hidden_dim,
            "num_heads": model.num_heads,
            "head_dim": model.hidden_dim // model.num_heads,
            "num_memory_tokens": model.num_memory_tokens,
            "mix_style_dim": model.mix_style_dim,
        },
        "channel_order": (
            list(channel_order) if channel_order is not None else None
        ),
        "amplitude_lift": {
            "enabled": bool(_data_attr(config, "amplitude_lift_enabled", False)),
            "mode": str(_data_attr(config, "amplitude_lift_mode", "rms")),
            "reference": str(_data_attr(config, "amplitude_lift_reference", "source")),
            "target_rms": float(_data_attr(config, "amplitude_lift_target_rms", 0.33)),
            "scale": float(_data_attr(config, "amplitude_lift_scale", 3.0)),
            "peak_limit": float(_data_attr(config, "amplitude_lift_peak_limit", 1.0)),
            "output_lufs": float(
                _data_attr(config, "amplitude_lift_output_lufs", -23.0)
            ),
            "clip_value": _data_attr(config, "amplitude_lift_clip_value", None),
            "gain_power": float(_data_attr(config, "amplitude_lift_gain_power", 1.0)),
            "gain_min_value": _data_attr(
                config, "amplitude_lift_gain_min_value", None
            ),
            "eps": float(_data_attr(config, "amplitude_lift_eps", 1.0e-8)),
        },
    }
    meta_path = onnx_path.with_name(onnx_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path, mem_path


def export_onnx(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    chunk_frames: int,
    weights_source: str = "auto",
    opset: int = 17,
    dynamic_frames: bool = False,
    exporter: str = "torchscript",
    no_attn_mask: bool = False,
    dtype: torch.dtype = torch.float32,
    sample_rate: int = 48000,
) -> Path:
    """Export the model to ``output_path`` and return the written path."""
    config = load_config(config_path)
    model = build_model(config)
    assert_exportable_config(model)

    checkpoint_path = resolve_checkpoint_path(
        checkpoint=checkpoint, output_dir=config.output_dir
    )
    used_source = load_model_weights(
        model=model, checkpoint_path=checkpoint_path, weights_source=weights_source
    )
    model = model.to(dtype=dtype).eval()

    full_inputs = make_example_inputs(
        model, chunk_frames=chunk_frames, dtype=dtype, device="cpu"
    )
    if no_attn_mask:
        wrapper: nn.Module = OnnxExportWrapperNoMask(model).eval()
        # full_inputs order: zt, t, z_cond, valid_mask, mix_style, mix_style_mask, mem
        example_inputs = (*full_inputs[:3], *full_inputs[4:])
        input_names = INPUT_NAMES_NO_MASK
    else:
        wrapper = OnnxExportWrapper(model).eval()
        example_inputs = full_inputs
        input_names = INPUT_NAMES

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_frames and not no_attn_mask:
        # T is dynamic on every tensor that carries the frame axis.
        dynamic_axes = {
            "zt": {3: "frames"},
            "z_cond": {3: "frames"},
            "valid_mask": {1: "frames"},
            "clean_prediction": {3: "frames"},
        }

    # dynamo (torch.export) emits a fused scaled_dot_product_attention op that
    # onnxruntime runs efficiently; TorchScript decomposes attention into
    # MatMul+Softmax (slow / memory-heavy at large T). dynamo is also required for
    # true dynamic frames (keeps arange(T) symbolic).
    use_dynamo = exporter == "dynamo" or dynamic_frames
    print(
        f"[export] weights={used_source} dtype={dtype} opset={opset} "
        f"chunk_frames={chunk_frames} exporter={'dynamo' if use_dynamo else 'torchscript'} "
        f"dynamic_frames={dynamic_frames}"
    )
    # Swap RoPE for the ScatterND-free version during export only (model unchanged).
    import stereo2spatial.modeling.layers as _layers_mod

    _orig_rope = _layers_mod.apply_rotary_embedding
    _layers_mod.apply_rotary_embedding = _rope_export_friendly
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(out_path),
                input_names=input_names,
                output_names=OUTPUT_NAMES,
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=use_dynamo,
            )
    finally:
        _layers_mod.apply_rotary_embedding = _orig_rope

    try:
        import onnx

        onnx.checker.check_model(str(out_path))
        print("[export] onnx.checker passed")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[export][warn] onnx.checker reported: {exc}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {out_path} ({size_mb:.1f} MB)")

    meta_path, mem_path = write_export_sidecars(
        config=config,
        model=model,
        onnx_path=out_path,
        chunk_frames=chunk_frames,
        sample_rate=sample_rate,
    )
    print(f"[export] wrote {meta_path}")
    print(f"[export] wrote {mem_path}")
    return out_path


def _resolve_chunk_frames(args: argparse.Namespace) -> int:
    if args.chunk_frames is not None:
        return int(args.chunk_frames)
    patch_fps = float(args.sample_rate) / float(args.patch_size)
    return max(1, round(float(args.chunk_seconds) * patch_fps))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a SpatialDiT checkpoint to a single-forward ONNX graph for "
            "on-device inference. The solver loop and audio DSP stay in host code."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training/inference YAML config used to build the model.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint directory (step_XXXXXXX), a .safetensors/.pt file, or 'latest'."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the .onnx file.",
    )
    parser.add_argument(
        "--weights-source",
        default="auto",
        choices=("auto", "ema", "student"),
        help="auto prefers EMA when present.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help=(
            "Static frame count T to export at. If omitted, derived from "
            "--chunk-seconds * (--sample-rate / --patch-size)."
        ),
    )
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--patch-size", type=int, default=200)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--exporter",
        default="torchscript",
        choices=("torchscript", "dynamo"),
        help=(
            "ONNX exporter. 'dynamo' (torch.export) emits fused attention that runs "
            "far faster at large T; 'torchscript' decomposes attention."
        ),
    )
    parser.add_argument(
        "--dynamic-frames",
        action="store_true",
        help=(
            "Export with T as a dynamic axis via the dynamo exporter (experimental). "
            "Default is a static T (recommended; pad+mask the tail host-side)."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export weights in fp16 (~570 MB vs ~1.1 GB). RMSNorm stays fp32 internally.",
    )
    parser.add_argument(
        "--no-attn-mask",
        action="store_true",
        help=(
            "Drop valid_mask (pass None). Removes the SDPA NaN-guard so fuse_mha.py "
            "can rewrite attention to flash MultiHeadAttention. Host must feed full "
            "chunk_frames and zero-pad the tail."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunk_frames = _resolve_chunk_frames(args)
    export_onnx(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_path=args.output,
        chunk_frames=chunk_frames,
        weights_source=args.weights_source,
        opset=args.opset,
        dynamic_frames=args.dynamic_frames,
        exporter=args.exporter,
        no_attn_mask=args.no_attn_mask,
        dtype=torch.float16 if args.fp16 else torch.float32,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
