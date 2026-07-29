"""Rank op types by the FIFO depth their tree model actually costs.

Raw TAV error cannot rank tree models -- an 8.9% error on a node whose
neighbours sit at the minimum depth of 2 is worth nothing, a 0.1% error next to
a 1109-deep FIFO is worth a lot. So this measures the only thing that matters:
run the whole ``chained_tav`` sizing chain twice and diff the resulting depth
vector, edge by edge.

Three modes per model, all offline (TAVs reload from ``tav_cache/``, no rtlsim,
no IP synthesis):

    --tree none              every node's TAV from the rtlsim cache   (reference)
    --tree prefer            tree model where one exists, cache otherwise
    --tree only:<OpType>     cache everywhere except that one op type

The ``only:`` runs are what isolate an op type's cost from every other's; the
``prefer`` run is the headline number the evaluation table reports.

    uv run python ci/experiments/tree_rank.py --models cnv-w2a2 vgg10 --out r.json
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.testing_util.tav_refs import ensure_finn_env  # noqa: E402

TRANSPARENT = ("StreamingFIFO", "StreamingDataWidthConverter")


def coalesced_edges(model):
    """One row per programmable (coalesced) FIFO of a live sized model.

    Mirrors ``tr_loop.coalesced_fifos`` but also walks *down* to the first
    non-transparent consumer, because attribution needs both ends of the edge.
    """
    from qonnx.custom_op.registry import getCustomOp

    prod = {o: n for n in model.graph.node for o in n.output}
    cons = {}
    for n in model.graph.node:
        for i in n.input:
            cons.setdefault(i, n)
    fifos = [n for n in model.graph.node if n.op_type.startswith("StreamingFIFO")]
    next_fifo = {n.input[0]: n for n in fifos}
    rows = []
    for node in fifos:
        up = prod.get(node.input[0])
        if up is not None and up.op_type.startswith("StreamingFIFO"):
            continue  # inside a split chain; only chain heads are programmable
        total, cur = 0, node
        while cur is not None:
            total += int(getCustomOp(cur).get_nodeattr("depth"))
            cur = next_fifo.get(cur.output[0])
        hops, tensor, anchor, anchor_op, port = 0, node.input[0], "GLOBAL_IN", "GLOBAL_IN", 0
        while True:
            p = prod.get(tensor)
            if p is None:
                break
            if not p.op_type.startswith(TRANSPARENT):
                anchor, anchor_op, port = p.name, p.op_type, list(p.output).index(tensor)
                break
            hops += 1
            tensor = p.input[0]
        t, sink, sink_op = node.output[0], "GLOBAL_OUT", "GLOBAL_OUT"
        while True:
            c = cons.get(t)
            if c is None:
                break
            if not c.op_type.startswith(TRANSPARENT):
                sink, sink_op = c.name, c.op_type
                break
            t = c.output[0]
        rows.append(
            dict(
                key=f"{anchor}#{port}#{hops}",
                producer=anchor,
                producer_op=anchor_op,
                consumer=sink,
                consumer_op=sink_op,
                width=int(getCustomOp(node).get_instream_width()),
                depth=total,
            )
        )
    return rows


def run(cfg, tree, strategy="chained_tav"):
    """Size once. Returns (edges, kB, wall seconds, per-op TAV timing)."""
    import contextlib
    import io

    from tr_loop import size_model

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        model = size_model(cfg, strategy, tree)
    wall = time.time() - t0
    edges = coalesced_edges(model)
    kb = sum(e["width"] * e["depth"] for e in edges) / 8.0 / 1000.0
    return edges, kb, wall, getattr(size_model, "timing", {})


def diff(ref, cand):
    """Per-edge depth delta, joined on the coalesced edge key."""
    by = {e["key"]: e for e in ref}
    out = []
    for e in cand:
        r = by.get(e["key"])
        if r is None:
            continue
        d = e["depth"] - r["depth"]
        out.append(
            dict(
                key=e["key"],
                producer_op=e["producer_op"],
                consumer_op=e["consumer_op"],
                width=e["width"],
                ref=r["depth"],
                cand=e["depth"],
                delta=d,
                delta_kB=d * e["width"] / 8.0 / 1000.0,
            )
        )
    return out


def tav_capable_op_types(cfg):
    """Op types the sizer asks for a TAV for, in this model, that have a tree."""
    import contextlib
    import io

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.general import GiveUniqueNodeNames

    from finn.builder.build_dataflow_steps import _tav_capable_nodes
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

    import tr_loop

    src = os.path.join(cfg["size"], "intermediate_models", "step_generate_estimate_reports.onnx")
    with contextlib.redirect_stdout(io.StringIO()):
        model = ModelWrapper(src)
        model = model.transform(InsertDWC())
        model = model.transform(SpecializeLayers(tr_loop.PART))
        model = model.transform(GiveUniqueNodeNames())
    have, lack = {}, {}
    for node in _tav_capable_nodes(model):
        try:
            has = getCustomOp(node).get_tree_model() is not None
        except Exception:
            has = False
        (have if has else lack)[node.op_type] = (have if has else lack).get(node.op_type, 0) + 1
    return have, lack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--strategy", default="chained_tav")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-per-op", action="store_true")
    ap.add_argument("--ops", nargs="*", default=None, help="isolate only these op types")
    args = ap.parse_args()

    ensure_finn_env()
    from tr_loop import MODELS

    results = {}
    for name in args.models:
        cfg = MODELS[name]
        print("=" * 78)
        print(name)
        have, lack = tav_capable_op_types(cfg)
        print("  tree-model op types : %s" % dict(sorted(have.items())))
        print("  NO tree model       : %s" % dict(sorted(lack.items())))
        try:
            ref, kb_ref, w_ref, t_ref = run(cfg, "none", args.strategy)
        except SystemExit as e:
            print("  SKIP (rtlsim reference unavailable): %s" % e)
            continue
        cand, kb_c, w_c, t_c = run(cfg, "prefer", args.strategy)
        d = diff(ref, cand)
        l1 = sum(abs(x["delta"]) for x in d)
        l1kb = sum(abs(x["delta_kB"]) for x in d)
        print(
            "  none  : %3d FIFOs  depth %8d  %8.3f kB  (%.1fs)"
            % (len(ref), sum(e["depth"] for e in ref), kb_ref, w_ref)
        )
        print(
            "  prefer: %3d FIFOs  depth %8d  %8.3f kB  (%.1fs)   L1 %d  |dkB| %.3f"
            % (len(cand), sum(e["depth"] for e in cand), kb_c, w_c, l1, l1kb)
        )
        entry = dict(
            model=name,
            have=have,
            lack=lack,
            n_fifos=len(ref),
            depth_none=sum(e["depth"] for e in ref),
            depth_prefer=sum(e["depth"] for e in cand),
            kB_none=kb_ref,
            kB_prefer=kb_c,
            wall_none=w_ref,
            wall_prefer=w_c,
            timing_none={k: {o: sum(v) for o, v in s.items()} for k, s in t_ref.items()},
            timing_prefer={k: {o: sum(v) for o, v in s.items()} for k, s in t_c.items()},
            n_none={k: {o: len(v) for o, v in s.items()} for k, s in t_ref.items()},
            n_prefer={k: {o: len(v) for o, v in s.items()} for k, s in t_c.items()},
            prefer_L1=l1,
            prefer_L1_kB=l1kb,
            prefer_edges=[x for x in d if x["delta"]],
            per_op={},
        )
        if not args.skip_per_op:
            for op in sorted(have if args.ops is None else [o for o in have if o in args.ops]):
                o_edges, o_kb, o_w, _ = run(cfg, "only:%s" % op, args.strategy)
                od = diff(ref, o_edges)
                o_l1 = sum(abs(x["delta"]) for x in od)
                o_kb1 = sum(abs(x["delta_kB"]) for x in od)
                moved = [x for x in od if x["delta"]]
                print(
                    "    only:%-34s L1 %7d  |dkB| %7.3f  net %+8.3f kB  %d/%d edges moved"
                    % (op, o_l1, o_kb1, o_kb - kb_ref, len(moved), len(od))
                )
                entry["per_op"][op] = dict(
                    n_nodes=have[op],
                    L1=o_l1,
                    L1_kB=o_kb1,
                    net_kB=o_kb - kb_ref,
                    kB=o_kb,
                    wall=o_w,
                    n_moved=len(moved),
                    edges=moved[:40],
                )
        results[name] = entry
        if args.out:
            json.dump(results, open(args.out, "w"), indent=1)

    print("\n" + "=" * 78)
    print("RANKING: mean FIFO-storage divergence caused by each op type's tree model")
    print("=" * 78)
    agg = {}
    for name, e in results.items():
        for op, r in e["per_op"].items():
            a = agg.setdefault(op, dict(models=[], L1=[], L1_kB=[], net_kB=[], nodes=0))
            a["models"].append(name)
            a["L1"].append(r["L1"])
            a["L1_kB"].append(r["L1_kB"])
            a["net_kB"].append(r["net_kB"])
            a["nodes"] += r["n_nodes"]
    rows = []
    for op, a in agg.items():
        n = len(a["L1_kB"])
        rows.append(
            (
                sum(a["L1_kB"]) / n,
                op,
                a["nodes"],
                n,
                sum(a["L1"]) / n,
                max(a["L1_kB"]),
                sum(a["net_kB"]) / n,
                [m for m, v in zip(a["models"], a["L1_kB"]) if v > 0],
            )
        )
    rows.sort(reverse=True)
    print(
        "%-38s %6s %5s %10s %10s %10s %10s"
        % ("op type", "nodes", "mdls", "mean L1", "mean |dkB|", "worst kB", "mean net")
    )
    print("-" * 96)
    for mean_kb, op, nodes, nm, mean_l1, worst, net, where in rows:
        print(
            "%-38s %6d %5d %10.1f %10.4f %10.4f %+10.4f  %s"
            % (op, nodes, nm, mean_l1, mean_kb, worst, net, ",".join(where))
        )
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
