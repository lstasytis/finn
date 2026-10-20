"""Validate the _small.json folding configs (tooling only).

    python check_small_foldings.py <worktree> [model-dir regex]

For each committed config with a _small.json sibling: apply both to the convert checkpoint of the
reference estimate build ($FINN_BUILD_DIR/est_*), then check the documented constraints
(docs/finn/reference/folding-constraints.rst): per-layer divisibility (via the folded shapes) and
ConvolutionInputGenerator SIMD == PE of a following VVAU / Pool; InsertDWC must succeed. Prints the
violations and the slowest nodes with their committed -> small cycles.
"""
import glob
import os
import re
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.general import ApplyConfig

WT = os.path.abspath(sys.argv[1])
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
BENCH = os.path.join(WT, "tests/benchmark")
EST = os.environ["FINN_BUILD_DIR"]


def violations(model):
    out = []
    for node in model.graph.node:
        inst = getCustomOp(node)
        try:
            inst.get_folded_input_shape()
            inst.get_folded_output_shape()
        except Exception as e:  # noqa: BLE001
            out.append(f"{node.name}: folded shape: {e}")
        if node.op_type.startswith("ConvolutionInputGenerator"):
            cons = model.find_consumer(node.output[0])
            if cons is not None and cons.op_type.startswith(("VVAU", "Pool")):
                simd, pe = inst.get_nodeattr("SIMD"), getCustomOp(cons).get_nodeattr("PE")
                if simd != pe:
                    out.append(f"{node.name} SIMD {simd} != {cons.name} PE {pe}")
    try:
        model.transform(InsertDWC())
    except Exception as e:  # noqa: BLE001
        out.append(f"InsertDWC: {e}")
    return out


for small in sorted(glob.glob(f"{BENCH}/*/folding_config/*_small.json")):
    mdir = small[len(BENCH) + 1 :].split("/")[0]
    if ONLY and not re.search(ONLY, mdir):
        continue
    full = small.replace("_small.json", ".json")
    stem = os.path.basename(full)[: -len(".json")]
    # the estimate build of this config: its applied folding equals the committed one
    cands = []
    for est in glob.glob(f"{EST}/est_{mdir}_*/"):
        if "_small_" in est or not os.path.isfile(est + "intermediate_models/phase_convert_to_hardware.onnx"):
            continue
        cands.append(est)
    found = None
    for est in cands:
        applied = ModelWrapper(est + "intermediate_models/step_apply_folding_config.onnx")
        conv = ModelWrapper(est + "intermediate_models/phase_convert_to_hardware.onnx")
        ref = conv.transform(ApplyConfig(full))
        if all(
            getCustomOp(a).get_nodeattr(k) == getCustomOp(b).get_nodeattr(k)
            for a, b in zip(applied.graph.node, ref.graph.node)
            for k in ("PE", "SIMD")
            if k in getCustomOp(a).get_nodeattr_types()
        ):
            found = (conv, ref)
            break
    if found is None:
        print(f"{stem}: no matching estimate build")
        continue
    conv, ref = found
    sm = conv.transform(ApplyConfig(small))
    bad = violations(sm)
    ref = ref.transform(AnnotateCycles())
    sm = sm.transform(AnnotateCycles())
    rows = sorted(
        (
            (getCustomOp(b).get_nodeattr("cycles_estimate"), getCustomOp(a).get_nodeattr("cycles_estimate"), a.name)
            for a, b in zip(ref.graph.node, sm.graph.node)
        ),
        reverse=True,
    )
    worst_full = max(r[1] for r in rows)
    print(f"{stem}: {'OK' if not bad else 'VIOLATIONS'}; max cycles {worst_full} -> {rows[0][0]}")
    for v in bad:
        print("   ", v)
    for c_small, c_full, name in rows[:3]:
        print(f"    {name}: {c_full} -> {c_small} ({c_small / max(c_full, 1):.1f}x)")
