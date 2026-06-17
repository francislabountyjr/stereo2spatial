"""Rewrite a no-mask SpatialDiT ONNX graph to use flash MultiHeadAttention.

onnxruntime cannot auto-fuse this model's custom attention (RoPE + RMSNorm on Q/K)
into a flash kernel, and its decomposed attention materializes the score matrix
(huge VRAM at the trained 10 s / T=2400 window). This script does deterministic
graph surgery -- with the model source untouched -- replacing each decomposed
``MatMul -> Softmax -> MatMul`` block with a single ``com.microsoft.MultiHeadAttention``
node, which runs a memory-efficient (flash) kernel on CUDA.

Measured at T=2400, fp16 (RTX 2080 Ti): decomposed 3.59 GB -> fused **1.43 GB** peak
(below PyTorch's 1.87 GB), parity 51 dB SNR vs fp32, ~837 ms/forward.

Input must be a graph exported with ``export_onnx.py --exporter dynamo --no-attn-mask``
(the no-mask path produces the clean decomposed pattern this matches). The
``*.meta.json`` / ``*.mem_init.npy`` sidecars are copied next to the output.

This script imports nothing from the model package; it operates purely on the ONNX
graph + sidecar metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def fuse_mha(input_path: str | Path, output_path: str | Path) -> Path:
    """Replace decomposed attention with com.microsoft.MultiHeadAttention."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    meta_path = input_path.with_name(input_path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    num_heads = int(meta["model"]["num_heads"])
    head_dim = int(meta["model"]["head_dim"])

    model = onnx.load(str(input_path))
    # Capture tensor shapes/dtypes before surgery (clean graph, no custom-domain ops).
    inferred = onnx.shape_inference.infer_shapes(model)
    vi_type = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input):
        tt = v.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        vi_type[v.name] = (shape, tt.elem_type)

    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for node in g.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)

    shape3d = "mha_shape_bshd_to_bsd"
    shape4d = "mha_shape_bsd_to_bshd"
    g.initializer.append(
        helper.make_tensor(shape3d, TensorProto.INT64, [3], [0, 0, -1])
    )
    g.initializer.append(
        helper.make_tensor(shape4d, TensorProto.INT64, [4], [0, 0, num_heads, head_dim])
    )

    new_nodes: list = []
    redirect: dict[str, str] = {}
    dead: set[int] = set()
    count = 0
    for sm in [n for n in g.node if n.op_type == "Softmax"]:
        qk = producer[sm.input[0]]
        if qk.op_type != "MatMul":
            raise RuntimeError(f"unexpected scores producer: {qk.op_type}")
        q_mul, k_mul = producer[qk.input[0]], producer[qk.input[1]]
        if q_mul.op_type != "Mul" or k_mul.op_type != "Mul":
            raise RuntimeError("expected scale Muls on Q and K (got a different graph)")
        scale = set(q_mul.input) & set(k_mul.input)  # the shared scale tensor
        q = next(x for x in q_mul.input if x not in scale)   # [B,H,Sq,D]
        k_t = next(x for x in k_mul.input if x not in scale)  # [B,H,D,Sk] (K^T)
        av = consumers[sm.output[0]][0]
        v = next(i for i in av.input if i != sm.output[0])    # [B,H,Sk,D]
        attn_out = av.output[0]                               # [B,H,Sq,D]
        dead.update(id(n) for n in (q_mul, k_mul, qk, sm, av))

        prefix = f"mha{count}_"

        def merge(src: str, name: str, pre: tuple = ()) -> list:
            # [B,H,S,D] -> [B,S,H*D] for MultiHeadAttention 3D inputs
            tr = helper.make_node(
                "Transpose", [src], [prefix + name + "_t"], perm=[0, 2, 1, 3],
                name=prefix + name + "_tr",
            )
            rs = helper.make_node(
                "Reshape", [prefix + name + "_t", shape3d], [prefix + name],
                name=prefix + name + "_rs",
            )
            return [*pre, tr, rs]

        k_bhsd = helper.make_node(
            "Transpose", [k_t], [prefix + "k_bhsd"], perm=[0, 1, 3, 2],
            name=prefix + "k_untranspose",
        )
        block = []
        block += merge(q, "q3")
        block += merge(prefix + "k_bhsd", "k3", pre=(k_bhsd,))
        block += merge(v, "v3")
        block.append(helper.make_node(
            "MultiHeadAttention",
            [prefix + "q3", prefix + "k3", prefix + "v3"], [prefix + "mha"],
            domain="com.microsoft", num_heads=num_heads, name=prefix + "mha",
        ))
        block.append(helper.make_node(
            "Reshape", [prefix + "mha", shape4d], [prefix + "o4"], name=prefix + "o4",
        ))
        block.append(helper.make_node(
            "Transpose", [prefix + "o4"], [prefix + "out"], perm=[0, 2, 1, 3],
            name=prefix + "out_tr",
        ))
        new_nodes += block
        redirect[attn_out] = prefix + "out"
        count += 1

    if count == 0:
        raise RuntimeError("no Softmax attention blocks found to fuse")

    # Replace the per-layer memory-token Pad (CPU op + host<->device copy each layer)
    # with a Concat of a constant zeros prefix, which onnxruntime runs on CUDA.
    inits = {i.name: i for i in g.initializer}
    n_pad = 0
    for pad in [n for n in g.node if n.op_type == "Pad"]:
        mode = next((a.s for a in pad.attribute if a.name == "mode"), b"constant")
        if mode != b"constant" or pad.input[1] not in inits:
            continue
        if len(pad.input) >= 3 and pad.input[2]:  # nonzero constant_value -> skip
            if float(numpy_helper.to_array(inits[pad.input[2]])) != 0.0:
                continue
        pads_val = numpy_helper.to_array(inits[pad.input[1]]).tolist()
        rank = len(pads_val) // 2
        begin, end = pads_val[:rank], pads_val[rank:]
        nz = [i for i, b in enumerate(begin) if b > 0]
        if end != [0] * rank or len(nz) != 1:  # only single-axis front pad
            continue
        axis, amount = nz[0], begin[nz[0]]
        shape, elem = vi_type.get(pad.input[0], (None, None))
        if not shape or any(d <= 0 for d in shape):
            continue
        zshape = list(shape)
        zshape[axis] = amount
        np_dt = helper.tensor_dtype_to_np_dtype(elem)
        zname = f"pad2concat_zeros_{n_pad}"
        g.initializer.append(
            numpy_helper.from_array(np.zeros(zshape, dtype=np_dt), name=zname)
        )
        new_nodes.append(helper.make_node(
            "Concat", [zname, pad.input[0]], [pad.output[0]], axis=axis,
            name=f"pad2concat_{n_pad}",
        ))
        dead.add(id(pad))
        n_pad += 1
    print(f"[fuse_mha] replaced {n_pad} Pad ops with CUDA Concat")

    # Rewire consumers of each old attention output to the new MHA output.
    for node in g.node:
        for idx, inp in enumerate(node.input):
            if inp in redirect:
                node.input[idx] = redirect[inp]

    # onnxruntime does not prune dead decomposed nodes, so drop them explicitly.
    kept = [n for n in g.node if id(n) not in dead]
    ordered = _topological_sort(
        kept + new_nodes,
        available={i.name for i in g.input} | {i.name for i in g.initializer},
    )
    del g.node[:]
    g.node.extend(ordered)
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # onnx appends external data; delete stale outputs so re-runs don't bloat the
    # .data file (a re-run otherwise grows it by the full weight size each time).
    data_file = output_path.with_name(output_path.name + ".data")
    for stale in (output_path, data_file):
        if stale.exists():
            stale.unlink()
    onnx.save(
        model, str(output_path), save_as_external_data=True,
        location=output_path.name + ".data", all_tensors_to_one_file=True,
    )

    # Copy sidecars; the runtime contract (dims, chunk_frames, lift) is unchanged,
    # but the file-reference fields must point at the fused graph's own names.
    shutil.copy(
        input_path.with_name(input_path.stem + ".mem_init.npy"),
        output_path.with_name(output_path.stem + ".mem_init.npy"),
    )
    meta["onnx_file"] = output_path.name
    meta["mem_init_file"] = output_path.stem + ".mem_init.npy"
    output_path.with_name(output_path.stem + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[fuse_mha] rewrote {count} attention blocks, removed {len(dead)} dead nodes")
    print(f"[fuse_mha] wrote {output_path}")
    return output_path


def _topological_sort(nodes: list, available: set) -> list:
    ordered: list = []
    remaining = nodes[:]
    while remaining:
        nxt = []
        progressed = False
        for n in remaining:
            if all(i == "" or i in available for i in n.input):
                ordered.append(n)
                available.update(n.output)
                progressed = True
            else:
                nxt.append(n)
        remaining = nxt
        if not progressed:
            raise RuntimeError(f"topological sort stuck on {len(remaining)} nodes")
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse decomposed attention into flash MultiHeadAttention."
    )
    parser.add_argument(
        "--input", required=True,
        help="ONNX exported with --exporter dynamo --no-attn-mask.",
    )
    parser.add_argument("--output", required=True, help="Path for the fused .onnx.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fuse_mha(args.input, args.output)


if __name__ == "__main__":
    main()
