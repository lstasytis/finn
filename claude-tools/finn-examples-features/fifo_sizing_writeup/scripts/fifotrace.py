"""Tracing helpers for the chained-TAV FIFO sizer (derive_characteristic.py).

Nothing in FINN is modified: the module-level helpers that
``derive_chained_tav_depths`` / ``DeriveFIFOSizes`` look up at call time are
wrapped for the duration of a ``Trace`` context, and every wrapper calls the
original. The caller's frame is inspected to learn which node / tensor a call
belongs to. ``edge_depth`` is decomposed again (allowance, burst floor, cap)
and the recomputed depth is asserted to equal the one the code returned.

Run inside a worktree environment of branch feature/dev/analytical-fifo-sizing
(see ../fifo_sizing_analytical.tex, appendix "Reproducing the numbers").
"""

import sys
import numpy as np
from contextlib import contextmanager
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import save_tav_npy

PART = "xc7z020clg400-1"


# ---------------------------------------------------------------------------
# hand-made TAVs
# ---------------------------------------------------------------------------


def tav(period, cycles):
    """Cumulative TAV over two periods from the cycles (within one frame) at
    which a token is transferred; a cycle may appear more than once."""
    per = np.zeros(2 * period, dtype=np.int64)
    for f in range(2):
        for c in cycles:
            per[f * period + c] += 1
    return np.cumsum(per)


def inject(model, node_name, tin, tout):
    """Store hand-made TAVs on a node exactly as the tree-model path does."""
    inst = getCustomOp(model.get_node_from_name(node_name))
    for name, curve in (("io_chrc_in", tin), ("io_chrc_out", tout)):
        arr = np.asarray(curve, dtype=np.int32).reshape(1, -1)
        inst.set_nodeattr(name, save_tav_npy(inst, name, arr))
    inst.set_nodeattr("io_chrc_period", len(tin) // 2)


def specialize(model, part=PART):
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(part))
    return model.transform(GiveUniqueNodeNames())


# ---------------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------------


class Trace:
    def __init__(self):
        self.edges = {}  # (pass, tensor) -> dict
        self.nodes = {}  # (pass, node) -> dict
        self.passes = []
        self.pass_depths = {}
        self.join_floors = None
        self.cur = None

    def _node_rec(self, name):
        return self.nodes.setdefault((self.cur, name), {})


def _caller(depth=2):
    return sys._getframe(depth).f_locals


@contextmanager
def tracing():
    tr = Trace()
    orig = {
        k: getattr(dc, k)
        for k in (
            "derive_chained_tav_depths",
            "edge_depth",
            "peak_occupancy",
            "local_clock",
            "causal_writes",
            "block_writes",
            "phase_lead",
            "absorbed_frame_floor",
            "join_window_floors",
            "graph_input_schedule",
        )
    }

    def w_pass(model, global_period, causal=True, latency=True):
        tr.cur = (causal, latency)
        tr.passes.append(tr.cur)
        res = orig["derive_chained_tav_depths"](model, global_period, causal, latency)
        tr.pass_depths[tr.cur] = dict(res)
        return res

    def w_gis(curve, global_period):
        res = orig["graph_input_schedule"](curve, global_period)
        loc = _caller()
        tr.edges.setdefault((tr.cur, loc["t"]), {})["graph_input_writes"] = res.copy()
        return res

    def w_clock(span, in_scheds):
        res = orig["local_clock"](span, in_scheds)
        loc = _caller()
        tr._node_rec(loc["node"].name)["clock"] = res.copy()
        return res

    def w_lead(out_curve, in_curves):
        res = orig["phase_lead"](out_curve, in_curves)
        loc = _caller()
        tr._node_rec(loc["node"].name)["lead"] = res
        return res

    def w_causal(write_times, in_scheds, lead=0):
        res = orig["causal_writes"](write_times, in_scheds, lead)
        loc = _caller()
        r = tr._node_rec(loc["node"].name)
        r["writes_raw"] = np.array(write_times).copy()
        r["writes_causal"] = res.copy()
        return res

    def w_block(write_times, node, in_scheds):
        res = orig["block_writes"](write_times, node, in_scheds)
        tr._node_rec(node.name)["writes_block"] = np.array(res).copy()
        return res

    def w_peak(write_times, read_times, period, per_frame, fill_only):
        res = orig["peak_occupancy"](write_times, read_times, period, per_frame, fill_only)
        loc = _caller()
        n = min(len(write_times), len(read_times))
        w = np.sort(np.asarray(write_times[:n]))
        r = np.sort(np.asarray(read_times[:n]))
        rec = {"peak": res, "peak_per_frame": per_frame}
        if per_frame and 2 * per_frame <= n:
            rec["peak_fill"] = dc.periodic_peak(w[:per_frame], r[:per_frame], period)
            rec["peak_steady"] = dc.periodic_peak(
                w[n - per_frame : n], r[n - per_frame : n], period
            )
        rec["peak_joint"] = dc.periodic_peak(w, r, period) if n else 0
        # the tensor is "t" in the node loop of derive_chained_tav_depths
        tr.edges.setdefault((tr.cur, loc["t"]), {}).update(rec)
        return res

    def w_edge(
        peak,
        read_times,
        edge,
        fill_only,
        throttleable,
        global_period,
        pacer_period,
        idle_floor=0,
        refill=0,
    ):
        res = orig["edge_depth"](
            peak,
            read_times,
            edge,
            fill_only,
            throttleable,
            global_period,
            pacer_period,
            idle_floor,
            refill,
        )
        loc = _caller()  # size_edge
        tensor = loc["tensor"]
        t_chain = edge.t_up if throttleable else edge.t_full
        allowance = edge.per_frame * (1.0 - min(t_chain, global_period) / float(global_period))
        after_slack = int(round(peak - allowance))
        burst = dc.burst_above_rate(read_times, edge.per_frame / float(global_period))
        floor = min(peak, burst)
        if not throttleable:
            floor = max(floor, min(peak, refill))
        cap = (
            throttleable
            and edge.t_up < pacer_period
            and not (peak > 0 and allowance < dc.CAP_MARGIN * peak)
        )
        floor_capped = min(floor, dc.THROTTLED_CAP) if cap else floor
        mine = (
            max(int(peak), int(idle_floor)) if fill_only else max(after_slack, floor_capped)
        )
        assert mine == res, (tensor, mine, res)
        tr.edges.setdefault((tr.cur, tensor), {}).update(
            dict(
                consumer=loc["consumer"].name,
                writes=np.array(edge.writes).copy(),
                reads=np.array(read_times).copy(),
                per_frame=edge.per_frame,
                t_up=edge.t_up,
                t_prop=edge.t_prop,
                t_full=edge.t_full,
                t_chain=t_chain,
                fill_only=fill_only,
                throttleable=throttleable,
                global_period=global_period,
                pacer_period=pacer_period,
                allowance=allowance,
                after_slack=after_slack,
                burst=burst,
                refill=refill,
                slack=loc["slack"],
                idle_floor=idle_floor,
                cap_applies=cap,
                floor=floor,
                floor_capped=floor_capped,
                depth=res,
            )
        )
        return res

    def w_absorb(node, tensor):
        res = orig["absorbed_frame_floor"](node, tensor)
        tr.edges.setdefault((tr.cur, tensor), {}).update(
            dict(consumer=node.name, absorbed=res, depth=res)
        )
        return res

    def w_join(model, nodes, period=None):
        res = orig["join_window_floors"](model, nodes, period)
        tr.join_floors = dict(res)
        return res

    wrappers = dict(
        derive_chained_tav_depths=w_pass,
        edge_depth=w_edge,
        peak_occupancy=w_peak,
        local_clock=w_clock,
        causal_writes=w_causal,
        block_writes=w_block,
        phase_lead=w_lead,
        absorbed_frame_floor=w_absorb,
        join_window_floors=w_join,
        graph_input_schedule=w_gis,
    )
    for k, f in wrappers.items():
        setattr(dc, k, f)
    try:
        yield tr
    finally:
        for k, f in orig.items():
            setattr(dc, k, f)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def edge_names(model):
    """{tensor: "producer -> consumer"} for every internal edge."""
    out = {}
    for t in [v.name for v in model.graph.value_info] + [x.name for x in model.graph.output]:
        p, c = model.find_producer(t), model.find_consumer(t)
        if p is not None and c is not None:
            out[t] = "%s -> %s" % (p.name, c.name)
    return out


def final_depths(model):
    """The depth InsertFIFO would create on each internal edge, plus graph I/O."""
    res = {}
    for node in model.graph.node:
        inst = getCustomOp(node)
        for i, t in enumerate(node.input):
            if t in [x.name for x in model.graph.input]:
                res["<in> -> %s" % node.name] = inst.get_nodeattr("inFIFODepths")[i]
        for i, t in enumerate(node.output):
            cons = model.find_consumer(t)
            od = inst.get_nodeattr("outFIFODepths")[i]
            if cons is None:
                res["%s -> <out>" % node.name] = od
                continue
            ci = getCustomOp(cons).get_nodeattr("inFIFODepths")[list(cons.input).index(t)]
            res["%s -> %s" % (node.name, cons.name)] = max(od, ci)
    return res


def node_table(model, period=None):
    """own period, drives_pacer, in_branch per node, as the pass computes them."""
    nodes = [n for n in model.graph.node if dc.is_hls_node(n) or dc.is_rtl_node(n)]
    tavs = dc.read_node_tavs(model, nodes)
    own = dc.node_periods(nodes, tavs)
    gp = max(int(period or 0), max(own.values(), default=0), 1)
    drives = dc.pacer_drivers(model, nodes, own, gp * dc.PACER_TOLERANCE)
    branch = dc.reconvergent_branch_nodes(model, nodes)
    rows = []
    for n in nodes:
        tin, tout = tavs[n.name][0], tavs[n.name][1]
        rows.append(
            dict(
                node=n.name,
                period=own.get(n.name),
                reads_per_frame=None if tin is None else dc.frame_tokens(tin),
                writes_per_frame=None if tout is None else dc.frame_tokens(tout),
                drives_pacer=drives[n.name],
                in_branch=n.name in branch,
            )
        )
    return gp, rows


def fmt_arr(a, limit=40):
    a = list(np.asarray(a).tolist())
    if len(a) <= limit:
        return "[" + " ".join(str(x) for x in a) + "]"
    h = limit // 2
    return (
        "["
        + " ".join(str(x) for x in a[:h])
        + " ... "
        + " ".join(str(x) for x in a[-h:])
        + "] (%d)" % len(a)
    )


KEYS = (
    "peak_fill",
    "peak_steady",
    "peak_joint",
    "peak",
    "per_frame",
    "t_up",
    "t_prop",
    "t_full",
    "t_chain",
    "fill_only",
    "throttleable",
    "allowance",
    "after_slack",
    "burst",
    "slack",
    "refill",
    "idle_floor",
    "cap_applies",
    "floor",
    "floor_capped",
    "absorbed",
    "depth",
)


def report(model, tr, title, show_sched=True, passes=None, period=None):
    """Print the node table, per-pass edge records (all passes, or ``passes``),
    the per-pass depths, the join floors and the final depths."""
    print("=" * 78)
    print(title)
    print("=" * 78)
    gp, rows = node_table(model, period)
    print("global period (pass input %s, after max with TAVs) = %d" % (period, gp))
    for r in rows:
        print("  node %-38s period=%-6s rd/frame=%-5s wr/frame=%-5s drives_pacer=%s in_branch=%s"
              % (r["node"], r["period"], r["reads_per_frame"], r["writes_per_frame"],
                 r["drives_pacer"], r["in_branch"]))
    names = edge_names(model)
    for p in passes or tr.passes:
        print("-- pass causal=%s latency=%s" % p)
        for (q, nname), rec in tr.nodes.items():
            if q != p or not show_sched:
                continue
            for k in ("lead", "clock", "writes_raw", "writes_causal", "writes_block"):
                if k in rec:
                    print("   node %s %s = %s" % (nname, k, rec[k] if k == "lead" else fmt_arr(rec[k])))
        for (q, t), rec in tr.edges.items():
            if q != p:
                continue
            if "graph_input_writes" in rec and show_sched:
                print("   graph input %s writes = %s" % (t, fmt_arr(rec["graph_input_writes"])))
            if "depth" not in rec:
                continue
            print("   edge %s [%s]" % (names.get(t, t), t))
            if show_sched and "writes" in rec:
                print("      writes = %s" % fmt_arr(rec["writes"]))
                print("      reads  = %s" % fmt_arr(rec["reads"]))
            print("      " + " ".join("%s=%s" % (k, _r(rec[k])) for k in KEYS if k in rec))
    print("-- per-pass depths (tensor: causal/latency FF FT TF TT)")
    allt = sorted({t for d in tr.pass_depths.values() for t in d})
    for t in allt:
        vals = [tr.pass_depths.get(p, {}).get(t, "-") for p in tr.passes]
        print("   %-50s %s" % (names.get(t, t), vals))
    print("-- join window floors:", tr.join_floors)
    print("-- final depths:")
    for k, v in final_depths(model).items():
        print("   %-60s %s" % (k, v))


def _r(v):
    if isinstance(v, float):
        return "%.2f" % v
    return v


@contextmanager
def join_floor_internals():
    """Record, for every join input, the locals of ``join_window_floors`` at
    the line ``if backlog > 1:`` (need, times, total, taken, deadlock, paced,
    backlog), by a line tracer on that one function: the real code runs."""
    import inspect

    code = dc.join_window_floors.__code__
    lines, start = inspect.getsourcelines(dc.join_window_floors)
    target = start + next(i for i, l in enumerate(lines) if "if backlog > 1:" in l)
    rows = []

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == target:
            loc = frame.f_locals
            rows.append(
                {
                    k: (np.array(loc[k]).copy() if hasattr(loc[k], "__len__") else loc[k])
                    for k in ("t", "k", "need", "times", "total", "taken", "deadlock",
                              "paced", "backlog", "period")
                }
            )
            rows[-1]["join"] = loc["join"].name
            rows[-1]["fork"] = loc["fork"].name
        return local

    def glob(frame, event, arg):
        if event == "call" and frame.f_code is code:
            return local
        return None

    old = sys.gettrace()
    sys.settrace(glob)
    try:
        yield rows
    finally:
        sys.settrace(old)
