"""Issue 024 A: offline depths of a TAV model with the worktree's DeriveFIFOSizes, baseline vs a
rule variant (monkey-patched edge_depth), listing every edge that changes.
VARIANT=nocap2: no throttled cap where t_above + t_below > CAP_MARGIN * period.
usage: whatif.py <tav.onnx> [variant]"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

orig = dc.edge_depth


def nocap2(peak, rt, edge, fill_only, thr, gp, pp, idle=0, refill=0, t_above=0, t_below=None):
    if fill_only:
        return orig(peak, rt, edge, fill_only, thr, gp, pp, idle, refill, t_above, t_below)
    t_chain = edge.t_up if thr else edge.t_full
    allowance = edge.per_frame * (1.0 - min(t_chain, gp) / float(gp))
    after_slack = int(round(peak - allowance))
    floor = min(peak, dc.burst_above_rate(rt, edge.per_frame / float(gp)))
    if not thr:
        floor = max(floor, min(peak, refill))
    tb = gp if t_below is None else t_below
    if (thr and edge.t_up < pp and not (peak > 0 and allowance < dc.CAP_MARGIN * peak)
            and t_above + tb <= dc.CAP_MARGIN * gp):
        floor = min(floor, dc.THROTTLED_CAP)
    if (thr and t_below is not None and dc.CAP_MARGIN * t_above >= (dc.CAP_MARGIN - 1) * gp
            and t_above + t_below <= dc.CAP_MARGIN * gp):
        floor = 0
    return max(after_slack, floor)


VARIANTS = {"nocap2": nocap2}


def size(m):
    m = m.transform(dc.DeriveFIFOSizes(period=int(m.get_metadata_prop("sizing_period"))))
    out = {}
    for n in m.graph.node:
        if not (is_hls_node(n) or is_rtl_node(n)):
            continue
        inst = getCustomOp(n)
        for i, t in enumerate(n.output):
            c = m.find_consumer(t)
            if c is not None:
                out[t] = (n.name, c.name, inst.get_nodeattr("outFIFODepths")[i], inst.get_outstream_width(i))
    return out


path = sys.argv[1]
var = sys.argv[2] if len(sys.argv) > 2 else "nocap2"
base = size(ModelWrapper(path))
dc.edge_depth = VARIANTS[var]
new = size(ModelWrapper(path))
bb = sum(v[2] * v[3] for v in base.values())
nb = sum(v[2] * v[3] for v in new.values())
for t, (p, c, d, w) in base.items():
    if new[t][2] != d:
        print(f"  {p:36s} -> {c:36s} {d:7d} -> {new[t][2]:7d} (w={w})")
print(f"{path.split('/')[-1]}: base {bb / 1024:.1f} kbit, {var} {nb / 1024:.1f} kbit, ratio {nb / max(bb, 1):.4f}")
