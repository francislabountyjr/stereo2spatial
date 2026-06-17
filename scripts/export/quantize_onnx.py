"""Weight-only quantize the flash ONNX graph to int4/int8 (block-wise) for mobile.

Uses onnxruntime's MatMulNBitsQuantizer (QOperator/MatMulNBits format) — the same op
ORT-Web's WebGPU backend runs for in-browser LLMs. MatMulNBits dequantizes per block
and accumulates in higher precision, so it both shrinks the model (~4-8x) AND avoids
the plain-fp16 accumulation overflow that NaN'd on WebGPU.

The exported graph stores the big Linears as a mix of MatMul (857 MB) and Gemm
(270 MB, the per-layer adaLN time_mod/cond_mod). The NBits quantizer only targets
MatMul, so we first rewrite Gemm -> MatMul (pre-transposing the weight) so everything
quantizes. The flash MultiHeadAttention op and other ops are untouched.

Model source is unchanged; this operates on the exported graph + copies the sidecars.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def gemm_to_matmul(model: onnx.ModelProto) -> int:
    """Rewrite standard Linear Gemm nodes to MatMul(+Add) so they're quantizable."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes = []
    converted = 0
    for n in g.node:
        if n.op_type != "Gemm" or n.input[1] not in inits:
            new_nodes.append(n)
            continue
        attrs = {a.name: a for a in n.attribute}
        alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
        beta = attrs["beta"].f if "beta" in attrs else 1.0
        transA = attrs["transA"].i if "transA" in attrs else 0
        transB = attrs["transB"].i if "transB" in attrs else 0
        if transA != 0 or alpha != 1.0 or beta != 1.0:
            new_nodes.append(n)  # non-standard; leave it (stays fp32)
            continue

        w = numpy_helper.to_array(inits[n.input[1]])
        if transB:
            w = np.ascontiguousarray(w.T)  # [out,in] -> [in,out] for MatMul(A, W)
        w_name = n.input[1] + "_mm"
        g.initializer.append(numpy_helper.from_array(w, name=w_name))

        mm_out = n.output[0] if len(n.input) < 3 else n.output[0] + "_mm"
        new_nodes.append(helper.make_node(
            "MatMul", [n.input[0], w_name], [mm_out], name=(n.name or w_name) + "_mm"))
        if len(n.input) >= 3:  # bias -> Add
            new_nodes.append(helper.make_node(
                "Add", [mm_out, n.input[2]], [n.output[0]], name=(n.name or w_name) + "_bias"))
        converted += 1
    del g.node[:]
    g.node.extend(new_nodes)
    return converted


def quantize(input_path: str | Path, output_path: str | Path, bits: int, block_size: int) -> Path:
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    input_path, output_path = Path(input_path), Path(output_path)
    model = onnx.load(str(input_path))
    n_conv = gemm_to_matmul(model)
    print(f"[quant] converted {n_conv} Gemm -> MatMul")

    cfg = DefaultWeightOnlyQuantConfig(
        block_size=block_size, is_symmetric=False, bits=bits,
    )
    quant = MatMulNBitsQuantizer(model, algo_config=cfg)
    quant.process()
    print(f"[quant] quantized to int{bits}, block_size={block_size}")

    # clear stale external data so re-runs don't bloat the file
    data_file = output_path.with_name(output_path.name + ".data")
    for stale in (output_path, data_file):
        if stale.exists():
            stale.unlink()
    quant.model.save_model_to_file(str(output_path), use_external_data_format=True)

    # sidecars (same runtime contract; just rename file references)
    meta_path = input_path.with_name(input_path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shutil.copy(input_path.with_name(input_path.stem + ".mem_init.npy"),
                output_path.with_name(output_path.stem + ".mem_init.npy"))
    meta["onnx_file"] = output_path.name
    meta["mem_init_file"] = output_path.stem + ".mem_init.npy"
    meta["quantization"] = {"bits": bits, "block_size": block_size, "op": "MatMulNBits"}
    output_path.with_name(output_path.stem + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    size = sum(p.stat().st_size for p in [output_path, data_file] if p.exists())
    print(f"[quant] wrote {output_path} (total {size / 1024**2:.0f} MB)")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="int4/int8 block-wise quantize for WebGPU.")
    p.add_argument("--input", required=True, help="fp32 flash ONNX (from fuse_mha).")
    p.add_argument("--output", required=True)
    p.add_argument("--bits", type=int, default=4, choices=(4, 8))
    p.add_argument("--block-size", type=int, default=128)
    return p


def main() -> None:
    a = build_parser().parse_args()
    quantize(a.input, a.output, a.bits, a.block_size)


if __name__ == "__main__":
    main()
