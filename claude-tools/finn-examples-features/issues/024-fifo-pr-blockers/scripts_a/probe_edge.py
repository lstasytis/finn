"""Issue 024 A: for the edges into the named consumers, print every edge_depth input and
the result per pass (causal x latency). usage: probe_edge.py <tav.onnx> <consumer> [...]"""
import sys

import finn.transformation.fpgadataflow.derive_characteristic as dc
from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper(sys.argv[1])
gp = int(m.get_metadata_prop("sizing_period"))
want = set(sys.argv[2:])
orig = dc.edge_depth
cur = {}


def wrapped(peak, rt, edge, fill_only, thr, gper, pp, idle=0, refill=0, t_above=0, t_below=None):
    cons = sys._getframe(1).f_locals["consumer"].name
    d = orig(peak, rt, edge, fill_only, thr, gper, pp, idle, refill, t_above, t_below)
    if cons in want:
        t_chain = edge.t_up if thr else edge.t_full
        allow = edge.per_frame * (1.0 - min(t_chain, gper) / float(gper))
        burst = dc.burst_above_rate(rt, edge.per_frame / float(gper))
        print(f"{cur['p']} {cons}: d={d} peak={peak} fill={fill_only} thr={thr} pf={edge.per_frame} "
              f"t_up={edge.t_up} t_full={edge.t_full} allow={allow:.0f} burst={burst} refill={refill} "
              f"idle={idle} t_above={t_above} t_below={t_below} gp={gper}")
    return d


dc.edge_depth = wrapped
for c in (0, 1):
    for la in (0, 1):
        cur["p"] = f"c{c}l{la}"
        dc.derive_chained_tav_depths(m, gp, c, la)
