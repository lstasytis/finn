"""Compare FINN's Vivado OOC place-and-route with the DynaRapid flow on one model.

Input: a dataflow ONNX model produced by prepare_model.py (all nodes through IP gen).

  baseline : CreateStitchedIP(run_synth, run_pnr) -- block design of all layers,
             OOC synthesis, opt_design / place_design / route_design in Vivado
  dynarapid: per-node components (parallel synth + pblock P&R), graph from the ONNX,
             DynaRapid placement + stitching + RWRoute

Usage:
    python run_experiment.py --model <dir>/dataflow_ipgen.onnx --out <dir> \
        --library <libdir> [--mode both|baseline|dynarapid] [--workers 28]
"""

import argparse
import json
import os
import re
import time
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.dynarapid.flow import dynarapid_pnr


def run_baseline(model, part, clk_ns, workers):
    os.environ["NUM_DEFAULT_WORKERS"] = str(workers)
    t0 = time.time()
    model = model.transform(CreateStitchedIP(part, clk_ns, run_synth=True, run_pnr=True))
    res = {"total_s": time.time() - t0}
    proj = model.get_metadata_prop("vivado_stitch_proj")
    res["project"] = proj
    res.update(vivado_phase_times(os.path.join(proj, "vivado.log")))
    res.update(parse_ooc(proj))
    return res


def vivado_phase_times(log):
    """Elapsed seconds of the synthesis run and of opt/place/route in the stitching log."""
    res = {}
    if not os.path.isfile(log):
        return res
    txt = open(log, errors="ignore").read()
    for cmd in ("opt_design", "place_design", "route_design"):
        m = re.search(r"%s: Time \(s\): cpu = [\d:]+ ; elapsed = (\d+):(\d+):(\d+)" % cmd, txt)
        if m:
            h, mi, s = map(int, m.groups())
            res[cmd + "_s"] = h * 3600 + mi * 60 + s
    m = re.search(r"wait_on_run: Time \(s\): cpu = [\d:]+ ; elapsed = (\d+):(\d+):(\d+)", txt)
    if m:
        h, mi, s = map(int, m.groups())
        res["synth_run_s"] = h * 3600 + mi * 60 + s
    return res


def parse_ooc(proj):
    res = {}
    tpath = os.path.join(proj, "ooc_timing.rpt")
    if os.path.isfile(tpath):
        txt = open(tpath).read()
        m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+(\S+)", txt, re.S)
        if m:
            res["wns_ns"] = float(m.group(1))
            res["tns_ns"] = float(m.group(2))
    upath = os.path.join(proj, "ooc_utilization.rpt")
    if os.path.isfile(upath):
        txt = open(upath).read()
        for key, pat in (
            ("LUT", r"\| CLB LUTs\*?\s+\|\s+(\d+)"),
            ("FF", r"\| CLB Registers\s+\|\s+(\d+)"),
            ("BRAM", r"\| Block RAM Tile\s+\|\s+([\d.]+)"),
            ("DSP", r"\| DSPs\s+\|\s+(\d+)"),
        ):
            m = re.search(pat, txt)
            res[key] = float(m.group(1)) if m else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--mode", choices=["both", "baseline", "dynarapid"], default="both")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--name", default=None)
    ap.add_argument("--num-shapes", type=int, default=1)
    ap.add_argument("--target-util", type=float, default=0.8)
    ap.add_argument("--bitstream", action="store_true")
    ap.add_argument("--pblock-mode", choices=["shaped", "fast"], default="fast")
    args = ap.parse_args()

    model = ModelWrapper(args.model)
    part = model.get_metadata_prop("dynarapid_fpga_part")
    clk_ns = float(model.get_metadata_prop("dynarapid_clk_ns"))
    name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    os.makedirs(args.out, exist_ok=True)
    result = {"model": args.model, "part": part, "clk_ns": clk_ns, "workers": args.workers}
    result["nodes"] = [n.op_type for n in model.graph.node]

    if args.mode in ("both", "dynarapid"):
        result["dynarapid"] = dynarapid_pnr(
            model,
            os.path.join(args.out, "dynarapid"),
            args.library,
            part,
            clk_ns,
            workers=args.workers,
            graph_name=name,
            num_shapes=args.num_shapes,
            target_util=args.target_util,
            bitstream=args.bitstream,
            pblock_mode=args.pblock_mode,
        )
        print("DynaRapid:", {k: v for k, v in result["dynarapid"].items() if k != "components"})
    if args.mode in ("both", "baseline"):
        result["baseline"] = run_baseline(ModelWrapper(args.model), part, clk_ns, args.workers)
        print("Baseline:", result["baseline"])

    with open(os.path.join(args.out, "experiment.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
