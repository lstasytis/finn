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


STAGE_TIMES = []


def instrument():
    """Record the wall-clock time of every FINN transformation ZynqBuild applies."""
    import finn.transformation.fpgadataflow.make_zynq_proj as mzp

    for cls_name in (
        "PrepareIP",
        "HLSSynthIP",
        "CreateStitchedIP",
        "MakeZYNQProject",
        "DynaRapidPnR",
        "InsertIODMA",
        "CreateDataflowPartition",
    ):
        cls = getattr(mzp, cls_name)
        orig = cls.apply

        def timed(self, model, _orig=orig, _name=cls_name):
            t0 = time.time()
            ret = _orig(self, model)
            names = ",".join(n.op_type for n in model.graph.node)[:60]
            STAGE_TIMES.append({"stage": _name, "s": round(time.time() - t0, 1), "nodes": names})
            return ret

        cls.apply = timed


def run_times(proj):
    """Elapsed seconds of the synth_1 and impl_1 runs (from their runme logs), and of the
    out-of-context IP synthesis runs of the block design (run in parallel before synth_1)."""
    res = {}
    runs_dir = os.path.join(proj, "finn_zynq_link.runs")
    ooc = {}
    for run in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        log = os.path.join(runs_dir, run, "runme.log")
        if run in ("synth_1", "impl_1") or not os.path.isfile(log):
            continue
        m = re.findall(r"^synth_design: Time \(s\):.*elapsed = (\d+):(\d+):(\d+)", open(log, errors="ignore").read(), re.M)
        if m:
            h, mi, se = m[-1]
            ooc[run] = int(h) * 3600 + int(mi) * 60 + int(se)
    if ooc:
        res["ooc_ip_synth_s"] = ooc
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
    ap.add_argument(
        "--mode",
        choices=["vivado", "dynarapid", "dynarapid-kernel", "islands"],
        required=True,
        help="dynarapid: whole accelerator by DynaRapid in a pre-implemented shell; "
        "dynarapid-kernel: only the compute kernel by DynaRapid, shell implemented by Vivado",
    )
    ap.add_argument("--shell-lib", default=None)
    ap.add_argument("--library", default=None)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--board", default="ZCU104")
    ap.add_argument("--islands", default="auto", help="islands mode: number of islands")
    ap.add_argument("--clk", type=float, default=None, help="clock (ns), default from model")
    args = ap.parse_args()

    os.environ["NUM_DEFAULT_WORKERS"] = str(args.workers)
    instrument()
    os.makedirs(args.out, exist_ok=True)
    model = ModelWrapper(args.model)
    clk_ns = args.clk or float(model.get_metadata_prop("dynarapid_clk_ns"))
    dr = None
    if args.mode == "islands":
        dr = {
            "flow": "islands",
            "islands": args.islands,
            "workers": args.workers,
            "out_dir": os.path.join(args.out, "islands"),
            "shell_lib": args.shell_lib,
            # HLS IP of this build only (no reuse across builds: cold)
            "library_dir": os.path.join(args.out, "iplib", "lib"),
        }
    if args.mode.startswith("dynarapid"):
        dr = {
            "library_dir": args.library,
            "workers": args.workers,
            "out_dir": os.path.join(args.out, "dynarapid"),
            "shell": args.mode == "dynarapid",
            "shell_lib": args.shell_lib,
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
    if args.mode in ("dynarapid", "islands"):
        res["dynarapid_zynq"] = json.loads(model.get_metadata_prop("dynarapid_result"))
        res["wns_ns"] = res["dynarapid_zynq"].get("wns_ns")
        res["stages"] = STAGE_TIMES
        with open(os.path.join(args.out, "bitfile_experiment.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return
    res["wns_ns"] = wns(proj)
    res.update(run_times(proj))
    res["stages"] = STAGE_TIMES
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
