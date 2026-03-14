import os
import sys
from typing import Dict, Optional, List
import shutil

import onnx
from onnx import helper, numpy_helper, shape_inference
import numpy as np


def _get_rank(value_info: Optional[onnx.ValueInfoProto]) -> Optional[int]:
    if value_info is None:
        return None
    t = value_info.type
    if not t.HasField("tensor_type"):
        return None
    shape = t.tensor_type.shape
    if not shape:
        return None
    return len(shape.dim)


def _collect_ranks(model: onnx.ModelProto) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    g = model.graph

    # From value_info, inputs, outputs
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        r = _get_rank(vi)
        if r is not None:
            ranks[vi.name] = r

    # From initializers
    for init in g.initializer:
        ranks[init.name] = len(init.dims)

    return ranks


def _opset_version(model: onnx.ModelProto) -> int:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    # default fallback
    return 13


def _get_concat_axis(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "axis":
            return attr.i
    # default axis
    return 0


def fix_concat_rank_mismatch(model: onnx.ModelProto) -> int:
    """
    Insert Unsqueeze before Concat inputs that are rank-0 scalars
    when other inputs to the same Concat are rank-1.
    Returns number of fixes applied.
    """
    opset = _opset_version(model)
    g = model.graph

    # Run shape inference to populate value_info
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        # If inference fails, we still attempt using initializer ranks
        pass

    ranks = _collect_ranks(model)

    fixes = 0
    new_nodes = []
    for idx, node in enumerate(g.node):
        if node.op_type != "Concat":
            new_nodes.append(node)
            continue

        input_ranks: List[Optional[int]] = [ranks.get(inp) for inp in node.input]
        has_rank0 = any(r == 0 for r in input_ranks)
        has_rank1 = any(r == 1 for r in input_ranks)

        if not (has_rank0 and has_rank1):
            new_nodes.append(node)
            continue

        # Replace rank-0 inputs with Unsqueeze
        updated_inputs = list(node.input)
        for i, (inp, r) in enumerate(zip(node.input, input_ranks)):
            if r != 0:
                continue

            unsq_output = f"{inp}__unsq_for_concat_{idx}"
            if opset >= 13:
                axes_name = f"{inp}__axes_for_concat_{idx}"
                axes_tensor = numpy_helper.from_array(np.array([0], dtype=np.int64), name=axes_name)
                g.initializer.append(axes_tensor)
                unsq_node = helper.make_node(
                    "Unsqueeze",
                    inputs=[inp, axes_name],
                    outputs=[unsq_output],
                    name=f"Unsqueeze_for_concat_{idx}_{i}",
                )
            else:
                unsq_node = helper.make_node(
                    "Unsqueeze",
                    inputs=[inp],
                    outputs=[unsq_output],
                    name=f"Unsqueeze_for_concat_{idx}_{i}",
                    axes=[0],
                )
            new_nodes.append(unsq_node)
            updated_inputs[i] = unsq_output
            fixes += 1

        # Recreate Concat node with updated inputs
        new_concat = helper.make_node(
            "Concat",
            inputs=updated_inputs,
            outputs=list(node.output),
            name=node.name,
            axis=_get_concat_axis(node),
        )
        # Copy over other attributes if any
        for attr in node.attribute:
            if attr.name == "axis":
                continue
            new_concat.attribute.extend([attr])

        new_nodes.append(new_concat)

    # Replace graph nodes
    del g.node[:]
    g.node.extend(new_nodes)

    return fixes


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_ocr_onnx.py /path/to/plate_ocr.onnx")
        return 1

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return 1

    backup_path = model_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")
    else:
        print(f"[INFO] Backup already exists: {backup_path}")

    model = onnx.load(model_path)
    fixes = fix_concat_rank_mismatch(model)

    if fixes == 0:
        print("[WARN] No Concat rank-0 inputs found to fix.")

    # Best-effort model check
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[WARN] Model check failed after fix: {e}")

    onnx.save(model, model_path)
    print(f"[INFO] Saved patched model to: {model_path}")
    print(f"[INFO] Fixes applied: {fixes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
