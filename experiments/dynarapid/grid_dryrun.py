"""Dry run of the island floorplans on a finished island build (no Vivado): per-node resources
from its synthesis results, then plan_islands with the given packing / grid knobs.

    python grid_dryrun.py <run>/islands [--packing grid] [--grid cols=12,rows=60,order=rows,pack=1]
"""
import argparse
import hashlib
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper

from finn.util.dynarapid.components import component_name
from finn.util.rwislands.device import load_device
from finn.util.rwislands.flow import island_region, plan_islands, shell_anchor
from finn.util.rwislands.netlist import channel_graph
from finn.util.rwislands.profiles import synth_args


def node_resources(isl_dir):
    r = json.load(open(os.path.join(isl_dir, "rwislands_zynq.json")))
    accel = ModelWrapper(os.path.join(isl_dir, "accel.onnx"))
    part, clk = r["part"], r["clk_ns"]
    synth = r["synth"]
    out = {}
    for n in accel.graph.node:
        base = component_name(accel, n, part, clk)
        hit = [k for k in synth if k == base or k.startswith(base + "_")]
        out[n.name] = synth[hit[0]]["util"] if hit else {}
    return accel, part, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isl_dir")
    ap.add_argument("--packing", default="grid")
    ap.add_argument("--grid", default="")
    a = ap.parse_args()
    os.environ["FINN_RWI_GRID"] = a.grid
    accel, part, nres = node_resources(a.isl_dir)
    missing = [k for k, v in nres.items() if not v]
    dev = load_device(part)
    g = channel_graph(accel)
    region, first = island_region(part, dev)
    res = {}
    names = [n.name for n in accel.graph.node]
    isl, ranges = plan_islands(names, nres, dev, region, first, 64, "auto", res, a.packing,
                               anchor=shell_anchor(part, dev), boundary=set(), g=g)
    costs = sorted((v["cost"] for v in res["islands"].values()), reverse=True)
    print(json.dumps({"packing": a.packing, "grid": a.grid, "islands": len(isl), "missing_res": len(missing),
                      "floorplan": {k: v for k, v in res.get("floorplan", {}).items() if k != "lanes"},
                      "largest_costs": [round(c) for c in costs[:5]], "median_cost": round(costs[len(costs) // 2]),
                      "max_tiles_per_island": max(len(v["rects"]) for v in res["islands"].values())}))


if __name__ == "__main__":
    main()
