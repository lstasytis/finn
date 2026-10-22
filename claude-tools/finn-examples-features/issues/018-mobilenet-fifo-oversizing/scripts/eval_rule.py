"""Issue 018: offline what-if of an edge_depth variant on a TAV model: patches edge_depth,
runs DeriveFIFOSizes, prints totals and the edges that change against the unpatched sizer.
Variants: nofloor (throttleable consumer with allowance >= CAP_MARGIN * peak: no burst floor),
slackfloor (throttleable consumer: burst floor less what the consumer can wait, i.e. the
slack of the slowest node at or below it, in tokens at the frame rate)
usage: eval_rule.py <tav.onnx> <variant>"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

orig = dc.edge_depth


def nofloor(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor=0, refill=0):
    if not fill_only and throttleable and edge.t_up < pp:
        allow = edge.per_frame * (1.0 - min(edge.t_up, gp) / float(gp))
        if allow >= dc.CAP_MARGIN * peak:
            return 0
    return orig(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor, refill)


TDOWN = {}


def downstream_periods(model):
    """slowest period of a node and everything downstream of it"""
    nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]
    per = dc.node_periods(nodes, dc.read_node_tavs(model, nodes))
    out = {}
    for n in reversed(nodes):
        below = [out.get(c.name, 0) for c in (model.find_consumer(t) for t in n.output) if c]
        out[n.name] = max([per.get(n.name, 0)] + below)
    return out


def slackfloor(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor=0, refill=0):
    if fill_only or not throttleable:
        return orig(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor, refill)
    cons = sys._getframe(1).f_locals["consumer"].name
    t_chain = edge.t_up
    allow = edge.per_frame * (1.0 - min(t_chain, gp) / float(gp))
    after_slack = int(round(peak - allow))
    # the consumer may wait as long as the slowest node at or below it has slack
    wait = edge.per_frame * (1.0 - min(TDOWN[cons], gp) / float(gp))
    burst = dc.burst_above_rate(read_times, edge.per_frame / float(gp))
    floor = min(peak, max(0, int(round(burst - wait))))
    if edge.t_up < pp and not (peak > 0 and allow < dc.CAP_MARGIN * peak):
        floor = min(floor, dc.THROTTLED_CAP)
    return max(after_slack, floor)


def cover(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor=0, refill=0):
    """throttleable consumer: the burst floor less the backlog the FIFOs above the producer
    (back to a pacer) already hold, upstream_cover in tokens of this edge"""
    if fill_only or not throttleable:
        return orig(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor, refill)
    f = sys._getframe(1).f_locals
    cov_cycles = f["upstream_cover"](f["tensor"])
    allow = edge.per_frame * (1.0 - min(edge.t_up, gp) / float(gp))
    after_slack = int(round(peak - allow))
    floor = min(peak, dc.burst_above_rate(read_times, edge.per_frame / float(gp)))
    if edge.t_up < pp and not (peak > 0 and allow < dc.CAP_MARGIN * peak):
        floor = min(floor, dc.THROTTLED_CAP)
    floor = max(0, floor - int(cov_cycles * edge.per_frame / float(gp)))
    return max(after_slack, floor)


VARIANTS = {"nofloor": nofloor, "slackfloor": slackfloor, "cover": cover}


def size(path, fn):
    dc.edge_depth = fn
    m = ModelWrapper(path)
    m = m.transform(dc.DeriveFIFOSizes(period=int(m.get_metadata_prop("sizing_period"))))
    out = {}
    for n in m.graph.node:
        if not (is_hls_node(n) or is_rtl_node(n)):
            continue
        inst = getCustomOp(n)
        for i, t in enumerate(n.output):
            c = m.find_consumer(t)
            if c is None:
                continue
            try:
                w = inst.get_outstream_width(i)
            except Exception:
                w = 0
            out[t] = (n.name, c.name, inst.get_nodeattr("outFIFODepths")[i], w)
    return out


if __name__ != "__main__":
    VARIANTS["none"] = orig
TDOWN.update(downstream_periods(ModelWrapper(sys.argv[1])))
if __name__ == "__main__":
    a = size(sys.argv[1], orig)
    b = size(sys.argv[1], VARIANTS[sys.argv[2]])
    ka = sum(v[2] * v[3] for v in a.values()) / 1024
    kb = sum(v[2] * v[3] for v in b.values()) / 1024
    for t in a:
        if a[t][2] != b[t][2]:
            print(f"  {a[t][0]:34s} -> {a[t][1]:34s} {a[t][2]:6d} -> {b[t][2]:6d} w={a[t][3]}")
    print(f"TOTAL kbit {ka:.1f} -> {kb:.1f} ({kb / ka if ka else 1:.3f})")
