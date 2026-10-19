"""Impact of each sizing rule: depth, storage and rtlsim throughput with vs without.

    python impact_tables.py <fixtures_dir> <results_dir> <variant_prefix> <out.json>

Depth/storage per model come from <fixture>/baseline.json (full sizer) and
<fixture>/<variant_prefix><rule>.json (rule removed); the stream width of every
edge from the characterized model. Throughput comes from the feature-flow result
rows: flow fifo_heuristic_analytical (full sizer), fifo_largefifo_rtlsim
(reference) and fifo_abl_<rule> (rule removed), latest complete row per model.
"""
import glob
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

fixtures_dir, results_dir, prefix, out_path = sys.argv[1:5]


def widths(fixture):
    m = ModelWrapper(os.path.join(fixture, "characterized.onnx"))
    w = {}
    for n in m.graph.node:
        inst = getCustomOp(n)
        for i, t in enumerate(n.output):
            w[t] = int(inst.get_outstream_width(i))
        for i, t in enumerate(n.input):
            if t not in w and m.find_producer(t) is None:
                try:
                    if inst.get_instream_width(i) > 0:
                        w[t] = int(inst.get_instream_width(i))
                except Exception:
                    pass
    return w


def totals(path, w):
    d = json.load(open(path))
    depth = sum(v["depth"] for v in d.values())
    kbit = sum(v["depth"] * w.get(t, 0) for t, v in d.items()) / 1024.0
    return depth, kbit


def rows(flow):
    path = os.path.join(results_dir, flow + ".jsonl")
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("build_ok") and not r.get("quick") and r.get("rtlsim_throughput_fps"):
            out[r["model"]] = r
    return out


result = {}
for fixture in sorted(glob.glob(os.path.join(fixtures_dir, "*/"))):
    model = os.path.basename(fixture.rstrip("/"))
    if not os.path.isfile(os.path.join(fixture, "baseline.json")):
        continue
    w = widths(fixture)
    base_depth, base_kbit = totals(os.path.join(fixture, "baseline.json"), w)
    entry = {"full": {"depth": base_depth, "kbit": base_kbit}, "rules": {}}
    full = rows("fifo_heuristic_analytical").get(model)
    ref = rows("fifo_largefifo_rtlsim").get(model)
    entry["full"]["fps"] = full["rtlsim_throughput_fps"] if full else None
    entry["full"]["fps_sim_kbit"] = full["fifo_total_kbit"] if full else None
    entry["reference"] = (
        {"fps": ref["rtlsim_throughput_fps"], "kbit": ref["fifo_total_kbit"]} if ref else None
    )
    for path in sorted(glob.glob(os.path.join(fixture, prefix + "*.json"))):
        rule = os.path.basename(path)[len(prefix) : -5]
        depth, kbit = totals(path, w)
        abl = rows("fifo_abl_" + rule).get(model)
        entry["rules"][rule] = {
            "depth": depth,
            "kbit": kbit,
            "depth_pct": 100.0 * (depth - base_depth) / base_depth,
            "kbit_pct": 100.0 * (kbit - base_kbit) / base_kbit if base_kbit else 0.0,
            "fps": abl["rtlsim_throughput_fps"] if abl else None,
            "fps_pct": (
                100.0 * (abl["rtlsim_throughput_fps"] - full["rtlsim_throughput_fps"])
                / full["rtlsim_throughput_fps"]
                if abl and full
                else None
            ),
        }
    result[model] = entry
json.dump(result, open(out_path, "w"), indent=1)

for model, e in result.items():
    print(
        "### %s: full depth %d, %.1f kbit, fps %s (reference fps %s, %s kbit)"
        % (
            model,
            e["full"]["depth"],
            e["full"]["kbit"],
            e["full"]["fps"],
            e["reference"]["fps"] if e["reference"] else None,
            e["reference"]["kbit"] if e["reference"] else None,
        )
    )
    for rule, r in e["rules"].items():
        if r["depth"] == e["full"]["depth"] and r["fps"] is None:
            continue
        print(
            "  %-20s without: depth %+7.1f%%  kbit %+7.1f%%  fps %s"
            % (
                rule,
                -r["depth_pct"] if False else r["depth_pct"],
                r["kbit_pct"],
                "%+.2f%%" % r["fps_pct"] if r["fps_pct"] is not None else "-",
            )
        )
