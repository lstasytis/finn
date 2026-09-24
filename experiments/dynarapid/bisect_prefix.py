"""Find the first node at which the DynaRapid design stops matching the stitched IP.

For each prefix length k, the first k nodes of the (IP-generated) dataflow model are cut
out, placed and routed with DynaRapid (components come from the library) and verified
with verify_netlist.py.

Usage: python bisect_prefix.py --model <dataflow_ipgen.onnx> --library <lib> --out <dir> --k 4 8 12
"""

import argparse
import json
import os
import subprocess
import sys
from qonnx.core.modelwrapper import ModelWrapper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_model import extract_subgraph  # noqa: E402

from finn.util.dynarapid.flow import dynarapid_pnr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", required=True)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    for k in args.k:
        m = ModelWrapper(args.model)
        names = [n.name for n in m.graph.node][args.start : k]
        m = extract_subgraph(m, names)
        d = os.path.join(args.out, "k%d_%d" % (args.start, k))
        os.makedirs(d, exist_ok=True)
        mfile = os.path.join(d, "model.onnx")
        m.save(mfile)
        part = m.get_metadata_prop("dynarapid_fpga_part")
        clk = float(m.get_metadata_prop("dynarapid_clk_ns"))
        r = dynarapid_pnr(
            m,
            os.path.join(d, "dr"),
            args.library,
            part,
            clk,
            graph_name="p%d_%d" % (args.start, k),
            check=False,
            no_clock=True,
        )
        subprocess.run(
            [
                sys.executable,
                os.path.join(here, "verify_netlist.py"),
                "--model",
                mfile,
                "--dcp",
                r["routed_dcp"],
                "--out",
                os.path.join(d, "verify"),
                "--frames",
                "3",
            ],
            stdout=open(os.path.join(d, "verify.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        try:
            v = json.load(open(os.path.join(d, "verify", "verify.json")))
            status = v["all_match"]
        except Exception:
            status = "error"
        print("k=%d nodes=%s last=%s match=%s" % (k, len(names), names[-1], status), flush=True)


if __name__ == "__main__":
    main()
