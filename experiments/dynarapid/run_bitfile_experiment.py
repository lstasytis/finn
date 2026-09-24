"""Bitfile generation (step_synthesize_bitfile / ZynqBuild) with and without DynaRapid.

    vivado   : regular ZynqBuild -- stitched IP per kernel, shell block design,
               synthesis + implementation of everything in Vivado
    dynarapid: compute kernels placed and routed by DynaRapid (components built in
               parallel or taken from the library), inserted into the shell as locked
               pre-routed cells; Vivado only implements the shell around them

Usage:
    python run_bitfile_experiment.py --model <dir>/dataflow_ipgen.onnx --out <dir> \
        --mode vivado|dynarapid [--library <libdir>] [--workers 28] [--board KV260_SOM]
"""

import argparse
import json
import os
import re
import time
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild


def run_times(proj):
    """Elapsed seconds of the synth_1 and impl_1 runs (from their runme logs)."""
    res = {}
    for run in ("synth_1", "impl_1"):
        log = os.path.join(proj, "finn_zynq_link.runs", run, "runme.log")
        if not os.path.isfile(log):
            continue
        txt = open(log, errors="ignore").read()
        total = 0
        for cmd, h, m, s in re.findall(
            r"^(\w+): Time \(s\): cpu = [\d:]+ ; elapsed = (\d+):(\d+):(\d+)", txt, re.M
        ):
            secs = int(h) * 3600 + int(m) * 60 + int(s)
            res["%s_%s_s" % (run, cmd)] = res.get("%s_%s_s" % (run, cmd), 0) + secs
            total += secs
    return res


def wns(proj):
    rpt = os.path.join(
        proj, "finn_zynq_link.runs", "impl_1", "top_wrapper_timing_summary_routed.rpt"
    )
    if not os.path.isfile(rpt):
        return None
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+(\S+)", open(rpt).read(), re.S)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["vivado", "dynarapid"], required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--board", default="KV260_SOM")
    args = ap.parse_args()

    os.environ["NUM_DEFAULT_WORKERS"] = str(args.workers)
    os.makedirs(args.out, exist_ok=True)
    model = ModelWrapper(args.model)
    clk_ns = float(model.get_metadata_prop("dynarapid_clk_ns"))
    dr = None
    if args.mode == "dynarapid":
        dr = {
            "library_dir": args.library,
            "workers": args.workers,
            "out_dir": os.path.join(args.out, "dynarapid"),
        }
    t0 = time.time()
    model = model.transform(
        ZynqBuild(
            args.board,
            clk_ns,
            partition_model_dir=os.path.join(args.out, "partitions"),
            dynarapid=dr,
        )
    )
    res = {"mode": args.mode, "model": args.model, "total_s": time.time() - t0}
    proj = model.get_metadata_prop("vivado_pynq_proj")
    res["project"] = proj
    res["bitfile"] = model.get_metadata_prop("bitfile")
    res["wns_ns"] = wns(proj)
    res.update(run_times(proj))
    for n in model.graph.node:
        k = ModelWrapper(getCustomOp(n).get_nodeattr("model"))
        r = k.get_metadata_prop("dynarapid_result")
        if r is not None:
            r = json.loads(r)
            res.setdefault("dynarapid", {})[n.name] = {
                kk: r.get(kk)
                for kk in (
                    "components_s",
                    "components_built",
                    "unique_components",
                    "dynarapid_s",
                    "pnr_total_s",
                )
            }
    with open(os.path.join(args.out, "bitfile_experiment.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
