"""Worked examples for fifo_sizing_analytical.tex.

Every number in the document's example tables is printed by this script. Two
kinds of graphs are used:

* toy graphs: real FINN HW nodes (MVAU_hls/MVAU_rtl, DuplicateStreams_hls,
  ElementwiseAdd_hls) built with the helpers of
  tests/fpgadataflow/test_fifosizing_chained_tav.py, whose TAVs are replaced by
  hand-made ones (``fifotrace.inject``) so that schedules are a few tokens long;
* the unit-test graphs themselves, with their tree-model TAVs.

Usage (inside a worktree of feature/dev/analytical-fifo-sizing, environment
from wtenv.sh, tests/fpgadataflow on PYTHONPATH):

    python fifo_examples.py [example ...]   # default: all

Ablations ("without rule X") monkeypatch one module-level name of
derive_characteristic for the duration of one call; FINN itself is not
modified.
"""

import sys
import numpy as np
from contextlib import contextmanager
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.transformation.fpgadataflow.derive_characteristic import DeriveFIFOSizes

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fifotrace import join_floor_internals, final_depths, fmt_arr, inject, report, specialize, tav, tracing  # noqa

from test_fifosizing_chained_tav import (  # noqa: E402
    characterize,
    make_mlp,
    make_residual,
    make_residual_rtl,
)

ZU = "xczu7ev-ffvc1156-2-e"


@contextmanager
def patched(**kw):
    old = {k: getattr(dc, k) for k in kw}
    for k, v in kw.items():
        setattr(dc, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(dc, k, v)


def rename(model, names):
    """Give the nodes (in graph order) the role names used in the document."""
    for node, name in zip(model.graph.node, names):
        node.name = name
    return model


def chain(names, tokens):
    """A chain of MVAU_hls nodes, MW = MH = SIMD = PE = 2: one token a vector,
    ``tokens`` vectors a frame, no width converters. Named ``names``."""
    m = make_mlp([(2, 2, 2, 2)] * len(names), nvec=tokens)
    for node in m.graph.node:
        node.attribute.append(helper.make_attribute("preferred_impl_style", "hls"))
    return rename(specialize(m), names)


#: passes whose per-edge records are printed (the per-pass depth table always
#: covers all four): (causal, latency)
SHOW_PASSES = [(True, True)]


def size(model, period, title, show=True, **patch):
    with patched(**patch), tracing() as tr:
        model = model.transform(DeriveFIFOSizes(period=period))
    report(model, tr, title, show_sched=show, period=period, passes=SHOW_PASSES)
    return model, tr


def depths_only(model, period, **patch):
    with patched(**patch):
        m = model.transform(DeriveFIFOSizes(period=period))
    return final_depths(m)


def every(step, n, start=0):
    return [start + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# E1: schedule construction on a chain, paced graph input, peak occupancy
# ---------------------------------------------------------------------------


def e1():
    """in -> A (fast, 1 token/cycle) -> B (pacer: read every 2nd cycle, write a
    burst in the second half) -> C (pacer rate)."""
    m = chain(['A', 'B', 'C'], 8)
    T = 16
    inject(m, "A", tav(8, range(8)), tav(8, range(8)))  # A
    inject(m, "B", tav(16, every(2, 8)), tav(16, range(8, 16)))  # B
    inject(m, "C", tav(16, every(2, 8)), tav(16, every(2, 8)))  # C
    print("TAV A in ", fmt_arr(tav(8, range(8))))
    print("TAV B in ", fmt_arr(tav(16, every(2, 8))), "out", fmt_arr(tav(16, range(8, 16))))
    print("TAV C in ", fmt_arr(tav(16, every(2, 8))))
    size(m, T, "E1 chain, paced input")

    # ablation: a source that delivers everything at time 0
    def instant(curve, global_period):
        return np.zeros(int(curve[-1]), dtype=np.int64)

    size(m, T, "E1 chain, instantaneous source (graph_input_schedule ablated)",
         graph_input_schedule=instant)


# ---------------------------------------------------------------------------
# E2: slack relaxation, reset past the pacer, burst floor
# ---------------------------------------------------------------------------


def e2():
    """in -> P (pacer, T = 32, burst of 8 writes at the end of its frame)
    -> X (fast, one-to-one) -> C (throttleable, reads every k cycles)."""
    T = 32
    for k, pc in ((3, 24), (2, 16)):
        m = chain(['P', 'X', 'C'], 8)
        inject(m, "P", tav(T, every(4, 8)), tav(T, range(24, 32)))  # P
        inject(m, "X", tav(8, range(8)), tav(8, range(8)))  # X
        inject(m, "C", tav(pc, every(k, 8)), tav(pc, every(k, 8)))  # C
        m, tr = size(m, T, "E2 relaxation, C reads every %d cycles (period %d)" % (k, pc))
        print("   without burst floor:",
              depths_only(m, T, burst_above_rate=lambda r, rate: 0))
        # what the edge X -> C would get without the reset past the pacer:
        # t_up = t_full = 32 (a fresh Edge with t_up replaced)
        rec = tr.edges[((True, True), "t1")]
        e = dc.Edge(rec["writes"], rec["per_frame"], rec["t_full"], rec["t_full"], rec["t_full"])
        print("   X->C with t_up = t_full (no reset past the pacer): edge_depth =",
              dc.edge_depth(rec["peak"], rec["reads"], e, False, True, T, T * dc.PACER_TOLERANCE))


# ---------------------------------------------------------------------------
# E3: throttled cap and cap margin
# ---------------------------------------------------------------------------


def e3():
    """E2 scaled to 1024 tokens a frame: P (T = 4096) -> X -> C (every 2nd cycle).
    X's period decides whether the allowance clears CAP_MARGIN * peak."""
    T = 4096
    for px in (1024, 1600):
        m = chain(['P', 'X', 'C'], 1024)
        inject(m, "P", tav(T, every(4, 1024)), tav(T, range(T - 1024, T)))
        inject(m, "X", tav(px, range(1024)), tav(px, range(1024)))
        inject(m, "C", tav(2048, every(2, 1024)), tav(2048, every(2, 1024)))
        size(m, T, "E3 throttled cap, X period %d" % px, show=False)
        print("   without cap:", depths_only(m, T, THROTTLED_CAP=10**9))
        print("   without margin:", depths_only(m, T, CAP_MARGIN=0.0))


# ---------------------------------------------------------------------------
# E4: pacer-bound chain (t_full) and pacer tolerance
# ---------------------------------------------------------------------------


def e4():
    """in -> P1 (pacer, burst at the end) -> X (fast) -> C, C at the pacer's
    period: C cannot be throttled, so the allowance uses t_full = T."""
    T = 32
    m = chain(['P', 'X', 'C'], 8)
    inject(m, "P", tav(T, every(4, 8)), tav(T, range(24, 32)))
    inject(m, "X", tav(8, range(8)), tav(8, range(8)))
    inject(m, "C", tav(T, every(4, 8)), tav(T, every(4, 8)))
    m, tr = size(m, T, "E4 pacer-bound consumer")
    print("   if C were throttleable (pacer_drivers ablated):",
          depths_only(m, T, pacer_drivers=lambda model, nodes, own, pp: {
              n.name: False for n in nodes}))
    # pacer tolerance: C one cycle short of the pacer at T = 400
    T = 400
    m = chain(['P', 'X', 'C'], 8)
    inject(m, "P", tav(T, every(50, 8)), tav(T, range(T - 8, T)))
    inject(m, "X", tav(8, range(8)), tav(8, range(8)))
    inject(m, "C", tav(T - 1, every(49, 8)), tav(T - 1, every(49, 8)))
    size(m, T, "E4b pacer tolerance, C period %d of %d" % (T - 1, T), show=False)
    print("   exact pacer test (PACER_TOLERANCE = 1):",
          depths_only(m, T, PACER_TOLERANCE=1.0))


# ---------------------------------------------------------------------------
# E5: refill floor and upstream cover
# ---------------------------------------------------------------------------


def e5():
    """in -> [U ->] A (reads 8, then writes 8: a window node) -> C (pacer,
    one read every 4 cycles). A's writes pause between frames."""
    T = 32
    for with_u in (False, True):
        names = ["U", "A", "C"] if with_u else ["A", "C"]
        m = chain(names, 8)
        names = list(names)
        if with_u:
            inject(m, names.pop(0), tav(8, range(8)), tav(8, range(8)))  # U
        inject(m, names[0], tav(16, range(8)), tav(16, range(8, 16)))  # A
        inject(m, names[1], tav(T, every(4, 8)), tav(T, every(4, 8)))  # C
        title = "E5 refill floor, %s" % ("with U above A" if with_u else "A fed by the input")
        size(m, T, title)
        print("   without refill floor:", depths_only(m, T, refill_floor=lambda w, r, s: 0))


# ---------------------------------------------------------------------------
# E6: fork / join, reconvergent branch, idle-window floor, node latency
# ---------------------------------------------------------------------------


def residual_toy(T, p_branch_reads, p_branch_writes, pb):
    """P0 -> fork -> (shortcut | M) -> join, PE = 1, 8 tokens a frame."""
    m = make_residual_rtl(nvec=4, ch=2, n_mvau=1)
    m = rename(specialize(m, ZU), ["P0", "F", "M", "J"])
    inject(m, "P0", tav(T, every(T // 8, 8)), tav(T, every(T // 8, 8)))
    inject(m, "M", tav(pb, p_branch_reads), tav(pb, p_branch_writes))
    return m


def e6():
    T = 64
    # M: reads its 8 tokens in cycles 0..7, writes them in 8..15 of a 16-cycle frame
    m = residual_toy(T, range(8), range(8, 16), 16)
    m, tr = size(m, T, "E6 residual: P0 -> fork -> (shortcut | M) -> join")
    print("   without idle-window floor:",
          depths_only(m, T, idle_window_floor=lambda c, th: 0))
    print("   without join floor:",
          depths_only(m, T, join_window_floors=lambda model, nodes, period=None: {}))
    # relaxing join/branch edges like chain edges: fill_only forced False via
    # peak_occupancy/edge_depth is not a module-level switch; recompute edge_depth
    for (p, t), rec in tr.edges.items():
        if p == (True, True) and rec.get("fill_only"):
            e = dc.Edge(rec["writes"], rec["per_frame"], rec["t_up"], rec["t_prop"], rec["t_full"])
            print("   %s as a chain edge: edge_depth = %d (fill_only: %d)" % (
                t, dc.edge_depth(rec["peak"], rec["reads"], e, False, rec["throttleable"], T,
                                 T * dc.PACER_TOLERANCE), rec["depth"]))


# ---------------------------------------------------------------------------
# E7: absorbed-frame floor
# ---------------------------------------------------------------------------


def e7():
    T = 16
    m = chain(['A', 'L'], 8)
    inject(m, "A", tav(16, every(2, 8)), tav(16, every(2, 8)))
    inject(m, "L", np.zeros(32, dtype=np.int64), tav(16, range(8, 16)))
    size(m, T, "E7 absorbed frame: the consumer's TAV reads nothing")


# ---------------------------------------------------------------------------
# E8: causal writes and phase lead (function level and in a graph)
# ---------------------------------------------------------------------------


def e8():
    print("=" * 78)
    print("E8 causal_writes / phase_lead, function level")
    in_times = np.arange(16) * 10
    print("  4:1 compactor, input times", fmt_arr(in_times))
    print("  TAV writes (all at 0)   ->",
          fmt_arr(dc.causal_writes(np.zeros(4, dtype=np.int64), {"x": (in_times, None)})))
    one_to_one = np.array([1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6])
    reads = np.array([0, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 6])
    print("  phase_lead(out=%s, in=%s) = %d" % (one_to_one.tolist(), reads.tolist(),
                                                dc.phase_lead(one_to_one, [reads])))
    print("  phase_lead expanding (out x2) =", dc.phase_lead(one_to_one * 2, [reads]))
    w = np.arange(8) * 10
    print("  causal_writes(writes=%s, lead=2) = %s" % (
        w.tolist(), dc.causal_writes(w, {"x": (np.arange(8) * 10, None)}, lead=2).tolist()))
    # in a graph: in -> P -> K (4:1 compactor, writes 1 cycle after its first
    # read) -> C ; and the phase-lead node in a residual (unit test)
    T = 32
    m = chain(['P', 'K', 'C'], 8)
    inject(m, "P", tav(T, every(4, 8)), tav(T, every(4, 8)))  # P, 8 tokens
    inject(m, "K", tav(8, range(8)), tav(8, [1, 5]))  # K: 8 -> 2
    inject(m, "C", tav(16, [0, 8]), tav(16, [0, 8]))  # C
    size(m, T, "E8b compactor K between P and C")


def residual_branch(names):
    """P0 -> F -> (shortcut | names...) -> J, PE = 1, 8 tokens a frame."""
    m = make_residual_rtl(nvec=4, ch=2, n_mvau=len(names))
    return rename(specialize(m, ZU), ["P0", "F"] + list(names) + ["J"])


def e8c():
    """Causal hold in a branch: K compacts 8 -> 2 but its TAV writes its first
    output after reading 2 inputs; E expands 2 -> 8 again."""
    T = 32
    m = residual_branch(["K", "E"])
    inject(m, "P0", tav(T, every(4, 8)), tav(T, every(4, 8)))
    inject(m, "K", tav(8, range(8)), tav(8, [1, 5]))
    inject(m, "E", tav(8, [0, 4]), tav(8, range(8)))
    size(m, T, "E8c compactor K and expander E inside a residual branch")


def e8d():
    """Phase lead in a branch: M is one-to-one, its TAV writes 2 tokens
    (cycles 0, 1) before its first read (cycle 2)."""
    T = 32
    m = residual_branch(["M"])
    inject(m, "P0", tav(T, every(4, 8)), tav(T, every(4, 8)))
    inject(m, "M", tav(10, range(2, 10)), tav(10, range(8)))
    size(m, T, "E8d phase lead of M inside a residual branch")


def e9():
    """block_writes and node_latency: a dynamic MVAU whose in1 (the matrix)
    comes from its own producer."""
    print("=" * 78)
    print("E9 block_writes, function level (unit test numbers)")
    mvau = helper.make_node(
        "MVAU_rtl", ["a", "b"], ["c"], domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow", MW=4, MH=2, SIMD=1, PE=1, numInputVectors=[1, 3],
        mem_mode="dynamic", inputDataType="INT4", weightDataType="INT4",
        outputDataType="INT4")
    print("  node_latency =", dc.node_latency(mvau))
    from qonnx.custom_op.registry import getCustomOp
    print("  get_tav_input_blocks =", getCustomOp(mvau).get_tav_input_blocks())
    matrix = np.arange(16) * 10
    writes = np.arange(12, dtype=np.int64)
    print("  matrix words at", fmt_arr(matrix))
    print("  writes", fmt_arr(writes), "->", fmt_arr(dc.block_writes(writes, mvau, {"b": (matrix, None)})))
    shuffle = helper.make_node(
        "InnerShuffle_rtl", ["x"], ["y"], domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow", data_type="INT8", in_shape=[1, 16, 8],
        transpose_in_shape=[1, 16, 8], SIMD=2)
    print("  InnerShuffle node_latency =", dc.node_latency(shuffle))


# ---------------------------------------------------------------------------
# E10: join window floor, internals
# ---------------------------------------------------------------------------


def e10():
    """Fork (PE 1) -> (shortcut | MVAU MW = MH = 8, SIMD = PE = 1) -> join,
    2 vectors a frame: tree-model TAVs, the floor's walk traced."""
    from qonnx.custom_op.registry import getCustomOp
    m, period = characterize(make_residual(nvec=2, ch=8, simd=1, pe=1))
    for n in m.graph.node:
        inst = getCustomOp(n)
        try:
            print("  %s exp_cycles=%s window=%s" % (n.name, inst.get_exp_cycles(),
                                                     inst.get_tav_input_window(0)))
        except Exception as e:  # noqa
            print("  %s exp_cycles=%s" % (n.name, inst.get_exp_cycles()))
    with join_floor_internals() as rows:
        size(m, period, "E10 join window floor", show=False)
    for r in rows:
        print("  join %s input %s (fork %s, period %s): backlog %d" % (
            r["join"], r["t"], r["fork"], r["period"], r["backlog"]))
        for key in ("k", "need", "times", "total", "taken", "deadlock", "paced"):
            print("     %-8s %s" % (key, fmt_arr(np.round(r[key], 1), 40)))


# ---------------------------------------------------------------------------
# U: the unit-test graphs (tree-model TAVs), per pass
# ---------------------------------------------------------------------------


def unit():
    m, period = characterize(make_mlp([(64, 64, 8, 8), (64, 64, 2, 64), (64, 64, 64, 64)], 16))
    size(m, period, "U1 test_chained_tav_mlp_chain", show=False)
    m, period = characterize(make_residual(nvec=16, ch=64, simd=8, pe=64))
    size(m, period, "U2 test_chained_tav_residual", show=False)
    m, period = characterize(make_residual_rtl(nvec=16, ch=32, n_mvau=2), ZU)
    size(m, period, "U3 test_chained_tav_residual_mvau_rtl", show=False)
    m, period = characterize(make_residual(nvec=16, ch=64, simd=1, pe=1))
    size(m, period, "U4 test_join_window_floor (fork PE 1, MVAU SIMD = PE = 1)", show=False)
    m = make_residual_rtl(nvec=16, ch=32, n_mvau=4)
    from qonnx.custom_op.registry import getCustomOp
    for node in m.graph.node[2:6]:
        getCustomOp(node).set_nodeattr("SIMD", 32)
        getCustomOp(node).set_nodeattr("PE", 32)
    m, period = characterize(m, ZU)
    size(m, period, "U5 test_chained_tav_residual_latency", show=False)
    # a compacting width converter: MVAU PE = 2 -> DWC 2:8 -> MVAU SIMD = 8
    m, period = characterize(make_mlp([(64, 64, 8, 2), (64, 64, 8, 8)], 16))
    size(m, period, "U6 compacting DWC (PE 2 -> SIMD 8)", show=False)


EXAMPLES = dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8, e8c=e8c, e8d=e8d, e9=e9, e10=e10, unit=unit)

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    args = sys.argv[1:]
    if "--all-passes" in args:
        args.remove("--all-passes")
        SHOW_PASSES[:] = [(False, False), (False, True), (True, False), (True, True)]
    for name in args or list(EXAMPLES):
        EXAMPLES[name]()
