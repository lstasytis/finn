"""Re-run the overlay planner offline on a finished (or plan-failed) overlay build: its accel.onnx
and node synthesis results (rwislands_zynq.json), the cached overlay's tiles.

    FINN_RWI_OVERLAY=... python replan_overlay.py <run>/islands [greedy|dp] [override.json]

override.json: {node name: {"lut": ..., ...}} for nodes without synthesis results (estimates).
Prints per-tile utilization (max over slice / SLICEM / DSP / BRAM / URAM) and the link usage.
"""

import json
import os
import sys
import time

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.util.dynarapid.components import component_name
from finn.util.rwislands import overlay as ov
from finn.util.rwislands.device import capacity, load_device
from finn.util.rwislands.floorplan import need
from finn.util.rwislands.netlist import boundary_ports, channel_graph


def main(d, mode="dp", override=None):
    res = json.load(open(os.path.join(d, "rwislands_zynq.json")))
    accel = ModelWrapper(os.path.join(d, "accel.onnx"))
    part, clk = res["part"], res["clk_ns"]
    g = channel_graph(accel)
    synth = res["synth"]
    sk = next(iter(synth)).rsplit("_", 1)[1]
    dcps = {n.name: "%s_%s" % (component_name(accel, n, part, clk), sk) for n in accel.graph.node}
    ovr = json.load(open(override)) if override else {}
    names = [n.name for n in accel.graph.node]
    node_res = {}
    for n in names:
        if n in ovr:
            node_res[n] = ovr[n]
        elif dcps[n] in synth:
            node_res[n] = synth[dcps[n]]["util"]
        else:
            raise SystemExit("no synthesis result for %s" % n)
    ports = json.load(open(os.path.join(res["shell"]["shell_dir"], "ports.json")))
    cfg_nodes = {e["node"] for e in ports if not e["iodma"]} if mode == "dp" else set()
    boundary = {n for n in names if len(boundary_ports(g, [n])) > 1} - cfg_nodes
    tiles = [tuple(t) for t in res["overlay"]["tiles"]] if isinstance(res["overlay"]["tiles"], list) else \
        [tuple(t) for t in json.loads(res["overlay"]["tiles"])]
    kn = ov.overlay_knobs()
    kn_plan = dict(kn)
    if cfg_nodes:
        from finn.util.rwislands import cfgnet

        a_w = max(c["width"] for e in ports if e["node"] in cfg_nodes for c in e["in"] if c["name"].endswith("_AW"))
        rf, rb = cfgnet.reserved_bits(a_w, len(cfg_nodes))
        kn_plan.update(wf=kn["wf"] - rf, wb=kn["wb"] - rb)
    dev = load_device(part)
    t0 = time.time()
    if mode == "dp":
        where, util = ov.plan_overlay_dp(names, node_res, g, dev, tiles, boundary, hard=kn["hard"] / 100.0,
                                         tile0=bool(kn.get("tile0", 1)), kn=kn_plan)
    else:
        where, util = ov.plan_overlay(names, node_res, g, dev, tiles, boundary, hard=kn["hard"] / 100.0,
                                      tile0=bool(kn.get("tile0", 1)))
    print("plan %s: %.1f s, %s" % (mode, time.time() - t0, util))
    if where is None:
        return
    cfg = ov.cfg_plan(ports, where) if cfg_nodes else None
    links, usage = ov.link_plan(g, where, len(tiles), kn, reserved=cfg["reserved"] if cfg else None)
    if cfg:
        print("cfg chain: %d nodes, last tile %d, request %d bits" % (cfg["n"], cfg["K"], cfg["rq"]))
    if os.environ.get("REPLAN_VERILOG"):
        used = sorted(set(where.values()) | {t for hops in links.values() for e, _, _, _ in hops for t in (e, e + 1)}
                      | (set(range(cfg["K"] + 1)) if cfg else set()))
        od = os.environ["REPLAN_VERILOG"]
        os.makedirs(od, exist_ok=True)
        mods = set()
        for k in used:
            mem = [n for n in names if where[n] == k]
            v, ms = ov.tile_verilog(k, len(tiles), kn, ports, g, mem, dcps, links, cfg=cfg, where=where)
            open(os.path.join(od, "tile%d.v" % k), "w").write(v)
            mods |= set(ms)
        print("tile netlists in %s; modules: %s" % (od, sorted(m for m in mods if "cfg" in m or "w%d_" % (cfg["rq"] if cfg else 0) in m)))
    caps = [capacity(dev.sites_in(*t)) for t in tiles]
    one = {"lut": 1, "bram": 1, "dsp": 1, "uram": 1}
    worst = 0
    for k in range(len(tiles)):
        mem = [n for n in names if where[n] == k]
        if not mem:
            continue
        tot = {}
        for n in mem:
            for kk, v in node_res[n].items():
                tot[kk] = tot.get(kk, 0) + v
        req = need(tot, one)
        r = {kk: req[kk] / caps[k][kk] for kk in req if caps[k][kk]}
        worst = max(worst, max(r.values()))
        print("tile %2d: %3d nodes  %s  lut %6d dsp %4d" % (k, len(mem), " ".join("%s %.2f" % (kk, v) for kk, v in r.items()),
                                                         tot.get("lut", 0), tot.get("dsp", 0)))
    print("tiles used %d, worst node utilization %.2f, link bits max f %d b %d" % (
        len({where[n] for n in names}), worst, max(v for (e, dd), v in usage.items() if dd == "f"),
        max(v for (e, dd), v in usage.items() if dd == "b")))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:]))
