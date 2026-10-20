"""Issue 014: unpadded vs padded optimizer folding vs the committed folding at the reference
target, estimate only (no FIFO sizing). Usage:
  python fold_compare.py <ref est dir> <board> <out.json> [paddings=0,4] [seed=1]
Prints per run: achieved cycles (with DWCs), as-built LUT/BRAM/URAM/DSP, padded nodes, pad DWCs,
and the _better decisions taken by the search."""
import copy
import json
import os
import sys
import time

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

RES = ["LUT", "BRAM_18K", "URAM", "DSP"]
BOUNDS = ["MW", "MH", "NumChannels", "Channels", "IFMChannels"]


def describe(model, folder, ref_model=None):
    m = copy.deepcopy(model).transform(GiveUniqueNodeNames())
    built = folder._resources_as_built(m)
    d = m.transform(InsertDWC()).transform(SpecializeLayers(folder.fpgapart))
    d = d.transform(AnnotateCycles())
    cycles = d.analysis(dataflow_performance)["max_cycles"]
    pad_dwcs = 0
    for n in d.graph.node:
        if n.op_type.startswith("StreamingDataWidthConverter"):
            inst = getCustomOp(n)
            pad_dwcs += int(inst.pads_or_crops()) if hasattr(inst, "pads_or_crops") else 0
    padded, fold = [], {}
    ref = {n.name: getCustomOp(n) for n in ref_model.graph.node} if ref_model else {}
    for n in m.graph.node:
        if not n.domain.startswith("finn"):
            continue
        inst = getCustomOp(n)
        types = inst.get_nodeattr_types()
        fold[n.name] = {a: inst.get_nodeattr(a) for a in ("PE", "SIMD") if a in types}
        fold[n.name]["cycles"] = int(inst.get_exp_cycles())
        if n.name in ref:
            for b in BOUNDS:
                if b in types and inst.get_nodeattr(b) != ref[n.name].get_nodeattr(b):
                    padded.append(n.name)
                    break
    return dict(
        cycles=int(cycles),
        res={r: float(built.get(r, 0.0)) for r in RES},
        pad_dwcs=pad_dwcs,
        padded_nodes=padded,
        folding=fold,
    )


def main():
    ref_dir, board, out = sys.argv[1:4]
    paddings = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,4").split(",")]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    perf = json.load(open(os.path.join(ref_dir, "report/estimate_network_performance.json")))
    target = int(perf["max_cycles"])
    im = os.path.join(ref_dir, "intermediate_models")
    start = ModelWrapper(os.path.join(im, "phase_convert_to_hardware.onnx"))
    committed = ModelWrapper(os.path.join(im, "step_apply_folding_config.onnx"))
    results = {"ref_dir": ref_dir, "board": board, "target": target}
    probe = SetFolding(target, platform=board)
    results["committed"] = describe(committed, probe, start)
    print("committed", results["committed"]["cycles"], results["committed"]["res"], flush=True)
    for pad in paddings:
        decisions = []
        orig = SetFolding._better

        def logged(self, a, b, t, maximize, _orig=orig):
            r = _orig(self, a, b, t, maximize)
            if a is not None and b is not None:
                decisions.append(
                    dict(
                        a=(a.achieved, a.fits, round(a.cost, 5), {k: a.metrics.get(k, 0) for k in RES}),
                        b=(b.achieved, b.fits, round(b.cost, 5), {k: b.metrics.get(k, 0) for k in RES}),
                        took_b=r is b,
                    )
                )
            return r

        SetFolding._better = logged
        try:
            folder = SetFolding(
                target, platform=board, folding_maximum_padding=pad, seed=seed, folding_effort=50
            )
            t0 = time.time()
            model, _ = folder.apply(copy.deepcopy(start))
            dt = time.time() - t0
        finally:
            SetFolding._better = orig
        r = describe(model, folder, start)
        r.update(time_s=dt, decisions=decisions, report=folder.last_search_report)
        results[f"pad{pad}"] = r
        print(
            f"pad{pad}", r["cycles"], r["res"], "pad_dwcs", r["pad_dwcs"], "padded",
            len(r["padded_nodes"]), "decisions", [(d["a"][:3], d["b"][:3], d["took_b"]) for d in decisions],
            f"{dt:.1f}s", flush=True,
        )
    json.dump(results, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
