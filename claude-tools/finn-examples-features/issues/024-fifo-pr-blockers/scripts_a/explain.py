"""Issue 018: offline sizing of a saved TAV model (013's tav_model.py) with the rule that
sets each edge's depth. Wraps edge_depth (no code change): per pass (causal x latency) it
records peak, slack allowance, burst floor, refill floor and the cap, and names the term
that sets the final depth (max over the passes, then the join floors).
usage: explain.py <tav.onnx> [-a] [-n N]   (-a: all edges, else the N largest in bits)"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
period = int(m.get_metadata_prop("sizing_period"))
nshow = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 25
orig = dc.edge_depth
rec = {}
cur = {}


def wrapped(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor=0, refill=0):
    f = sys._getframe(1).f_locals
    t = f["tensor"]
    t_chain = edge.t_up if throttleable else edge.t_full
    allow = edge.per_frame * (1.0 - min(t_chain, gp) / float(gp))
    burst = dc.burst_above_rate(read_times, edge.per_frame / float(gp))
    capped = throttleable and edge.t_up < pp and not (peak > 0 and allow < dc.CAP_MARGIN * peak)
    d = orig(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor, refill)
    after = int(round(peak - allow))
    if fill_only:
        why = "idle" if idle_floor > peak else "peak(fill)"
    elif d == after and after > 0:
        why = "peak-slack" if allow > 0.5 else "peak"
    elif capped and d == dc.THROTTLED_CAP:
        why = "cap256"
    elif not throttleable and d == min(peak, refill) and refill > min(peak, burst):
        why = "refill"
    elif d == min(peak, burst):
        why = "burst" if burst < peak else "burst=peak"
    else:
        why = "?"
    rec.setdefault(t, {})[cur["p"]] = dict(
        d=d, peak=int(peak), allow=allow, burst=burst, refill=int(refill), why=why,
        thr=throttleable, pf=edge.per_frame, cons=f["consumer"].name, slack=f.get("slack", 0),
    )
    return d


dc.edge_depth = wrapped
for c in (0, 1):
    for la in (0, 1):
        cur["p"] = f"c{c}l{la}"
        dc.derive_chained_tav_depths(m, period, c, la)
hw = [n for n in m.graph.node if is_hls_node(n) or is_rtl_node(n)]
floors = dc.join_window_floors(m, hw) if hasattr(dc, "join_window_floors") else {}
rows = []
for n in m.graph.node:
    for i, t in enumerate(n.output):
        c = m.find_consumer(t)
        if c is None:
            continue
        r = rec.get(t, {})
        dep = max([2] + [v["d"] for v in r.values()])
        win = max(r, key=lambda p: r[p]["d"]) if r else "-"
        if floors.get(t, 0) > dep:
            dep, win = floors[t], "joinfloor"
        try:
            w = getCustomOp(n).get_outstream_width(i)
        except Exception:
            w = 0
        rows.append((dep * w, t, n.name, c.name, dep, w, win, r))
tot = sum(x[0] for x in rows)
print(f"period {period}  edges {len(rows)}  total {tot / 8192:.1f} kB ({tot / 1024:.1f} kbit)")
show = rows if "-a" in sys.argv else sorted(rows, key=lambda x: -x[0])[:nshow]
for bits, t, p, c, dep, w, win, r in show:
    v = r.get(win, {})
    passes = " ".join(f"{k}={r[k]['d']}" for k in sorted(r))
    print(f"{p:34s} -> {c:34s} d={dep:5d} w={w:4d} {bits / 8192:6.2f}kB [{win} {v.get('why', '')}] "
          f"peak={v.get('peak')} allow={v.get('allow', 0):.0f} burst={v.get('burst')} "
          f"refill={v.get('refill')} slack={v.get('slack')} pf={v.get('pf')} thr={v.get('thr')} "
          f"| {passes}")
