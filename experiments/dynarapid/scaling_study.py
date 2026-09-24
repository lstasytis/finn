"""Component-size scaling study: 2-layer MVAU -> MVAU graph with increasing PE/SIMD.

For each scale factor s, MVAU_hls_0 -> MVAU_hls_1 of the reduced TFC is folded with
PE and SIMD multiplied by s (clipped to legal divisors), prepared up to IP generation,
and implemented both with DynaRapid (fresh library) and with the Vivado OOC baseline.

Usage:
    python scaling_study.py --out <dir> --scales 1 2 4 8 16 [--workers 28]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def pblock_info(lib, dcp):
    """(rows, cols, valid places) of the component's first pblock, from its database."""
    data = os.path.join(lib, dcp + ".data")
    if not os.path.isfile(data):
        return None
    txt = open(data).read()
    rows = re.search(r"# of Rows : (\d+)", txt)
    cols = re.search(r"# of Columns : (\d+)", txt)
    valid = re.search(r"# of valid places : (\d+)", txt)
    return {
        "rows": int(rows.group(1)) if rows else None,
        "cols": int(cols.group(1)) if cols else None,
        "valid_places": int(valid.group(1)) if valid else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="dir with the TFC dataflow_streamlined.onnx")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    summary = []
    for s in args.scales:
        d = os.path.join(args.out, "s%d" % s)
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, "dataflow_streamlined.onnx")
        if not os.path.isfile(base):
            subprocess.run(["cp", os.path.join(args.base, "dataflow_streamlined.onnx"), base])
        model = os.path.join(d, "dataflow_ipgen.onnx")
        if not os.path.isfile(model):
            run(
                [
                    sys.executable,
                    os.path.join(HERE, "prepare_model.py"),
                    "--topology",
                    "tfc",
                    "--nodes",
                    "MVAU_hls_0,MVAU_hls_1",
                    "--scale",
                    str(s),
                    "--out",
                    d,
                ],
                os.path.join(d, "prepare.log"),
            )
        row = {"scale": s}
        lib = os.path.join(d, "lib", "lib")
        run(
            [
                sys.executable,
                os.path.join(HERE, "run_experiment.py"),
                "--model",
                model,
                "--out",
                os.path.join(d, "dr"),
                "--library",
                lib,
                "--mode",
                "dynarapid",
                "--workers",
                str(args.workers),
                "--name",
                "s%d" % s,
            ],
            os.path.join(d, "dr.log"),
        )
        rj = os.path.join(d, "dr", "dynarapid", "dynarapid_result.json")
        if os.path.isfile(rj):
            r = json.load(open(rj))
            row["dr_status"] = r["status"]
            row["dr_components_s"] = r.get("components_s")
            row["dr_stitch_s"] = r.get("dynarapid_s")
            row["dr_check"] = r.get("check")
            row["components"] = []
            for c in r["components"]:
                info = pblock_info(lib, c["dcp"]) or {}
                util = glob.glob(
                    os.path.join(d, "lib", "work", "vhdlSynthDCPs", c["dcp"] + ".util")
                )
                luts = None
                if util:
                    m = re.search(r"\| CLB LUTs\*?\s+\|\s+(\d+)", open(util[0]).read())
                    luts = int(m.group(1)) if m else None
                row["components"].append(
                    dict(
                        node=c["node"],
                        status=c["status"],
                        synth_s=c.get("synth_s"),
                        pblock_s=c.get("pblock_s"),
                        luts=luts,
                        **info
                    )
                )
        if not args.skip_baseline:
            run(
                [
                    sys.executable,
                    os.path.join(HERE, "run_experiment.py"),
                    "--model",
                    model,
                    "--out",
                    os.path.join(d, "base"),
                    "--library",
                    lib,
                    "--mode",
                    "baseline",
                    "--workers",
                    str(args.workers),
                ],
                os.path.join(d, "base.log"),
            )
            bj = os.path.join(d, "base", "experiment.json")
            if os.path.isfile(bj):
                row["baseline"] = json.load(open(bj)).get("baseline")
        summary.append(row)
        with open(os.path.join(args.out, "scaling_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
