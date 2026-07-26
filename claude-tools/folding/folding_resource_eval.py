#!/usr/bin/env python3
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluate how SetFolding-optimizer parameter choices affect the final resource
consumption of the finn-examples models, and emit LaTeX tables for write-ups.

For every (model x parameter-configuration) pair it runs the folding optimizer on
a specialize-layers checkpoint and records the achieved throughput (max_cycles)
and the aggregate LUT / BRAM / DSP / URAM estimate. Results are cached to JSON so
re-runs and table regeneration are cheap.

Outputs, under --outdir (default tools/folding_eval_out/):
  * results.json                    -- raw measurements (cache)
  * table_<model>.tex               -- one booktabs table per model
                                       (rows = configs, cols = cycles + resources)
  * folding_resource_eval.tex       -- standalone doc that \\input's every table

Compile the standalone doc with:  pdflatex folding_resource_eval.tex
(requires the booktabs package).

Because the finn-examples models are not part of the main FINN test set, point the
tool at your own specialize-layers checkpoints via --models-json or edit MODELS
below. Missing checkpoints are skipped with a warning.

Examples
--------
  # default matrix over whichever default checkpoints exist
  python tools/folding_resource_eval.py

  # only two models, a custom config subset, higher effort
  python tools/folding_resource_eval.py --models gtsrb vgg10 \\
      --configs baseline prefer_uram prefer_lut --effort 100

  # regenerate the .tex from the cached results without recomputing
  python tools/folding_resource_eval.py --tex-only
"""
import argparse
import json
import os
import time
import traceback

# ---------------------------------------------------------------------------
# Defaults -- edit these (or pass --models-json) to match your environment.
# Each entry: name -> {"checkpoint": <step_specialize_layers.onnx>, "platform": <board>}
# ---------------------------------------------------------------------------
FINN_BUILD = os.environ.get("FINN_BUILD_DIR", "/tmp/finn_dev_" + os.environ.get("USER", ""))


def _ckpt(sweep):
    return f"{FINN_BUILD}/{sweep}/intermediate_models/step_specialize_layers.onnx"


MODELS = {
    "cybersec": {
        "checkpoint": _ckpt("sweep_cybersecurity-mlp_analytic_rtlsim"),
        "platform": "U250",
    },
    "gtsrb": {"checkpoint": _ckpt("sweep_gtsrb_analytic_model_based_844dulqr"), "platform": "U250"},
    "vgg10": {
        "checkpoint": _ckpt("sweep_vgg10-radioml_analytic_model_based_058h_msv"),
        "platform": "U250",
    },
    "mobilenet": {
        "checkpoint": _ckpt("sweep_mobilenet-v1_analytic_model_based"),
        "platform": "U55C",
    },
}

# The parameter axis: named SetFolding configurations to compare. Kwargs are
# passed straight to SetFolding (target_cycles_per_frame=None means "maximize").
CONFIGS = {
    "baseline": dict(target_cycles_per_frame=10000),
    "tight_target": dict(target_cycles_per_frame=2000),
    "maximize": dict(target_cycles_per_frame=None),
    "prefer_bram": dict(target_cycles_per_frame=10000, prefer_memory="bram"),
    "prefer_uram": dict(target_cycles_per_frame=10000, prefer_memory="uram"),
    "prefer_lut": dict(target_cycles_per_frame=10000, prefer_compute="lut"),
    "prefer_dsp": dict(target_cycles_per_frame=10000, prefer_compute="dsp"),
    "padding4": dict(target_cycles_per_frame=10000, folding_maximum_padding=4),
}

RESOURCES = ["LUT", "BRAM_18K", "DSP", "URAM"]
COLUMNS = ["max_cycles"] + RESOURCES


def measure(checkpoint, platform, effort, seed, kwargs):
    """Run SetFolding on a checkpoint and return {max_cycles, LUT, BRAM_18K, DSP, URAM}."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
    from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
    from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
    from finn.transformation.fpgadataflow.set_folding import SetFolding, part_map

    model = ModelWrapper(checkpoint).transform(AnnotateCycles())
    model = model.transform(
        SetFolding(platform=platform, style="optimizer", folding_effort=effort, seed=seed, **kwargs)
    )
    model = model.transform(AnnotateCycles())
    perf = model.analysis(dataflow_performance)
    est = {}
    for node in model.graph.node:
        inst = getCustomOp(node)
        est[inst] = inst.node_res_estimation(part_map[platform])
    agg = aggregate_dict_keys(est)
    row = {"max_cycles": int(perf["max_cycles"])}
    row.update({r: float(agg.get(r, 0.0)) for r in RESOURCES})
    return row


def evaluate(models, configs, effort, seed, cache):
    """Run every (model, config) pair not already in the cache; update in place."""
    for mname in models:
        spec = MODELS[mname]
        if not os.path.isfile(spec["checkpoint"]):
            print(f"[skip] {mname}: checkpoint not found ({spec['checkpoint']})")
            continue
        cache.setdefault(mname, {})
        for cname in configs:
            if cname in cache[mname]:
                print(f"[cached] {mname}/{cname}")
                continue
            t0 = time.time()
            try:
                row = measure(spec["checkpoint"], spec["platform"], effort, seed, CONFIGS[cname])
                row["seconds"] = round(time.time() - t0, 1)
                cache[mname][cname] = row
                print(f"[ok] {mname}/{cname}: {row}")
            except Exception as e:
                traceback.print_exc()
                cache[mname][cname] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[fail] {mname}/{cname}: {e}")
    return cache


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.0f}"
    return str(v)


def _tex_escape(s):
    return s.replace("_", r"\_")


def emit_model_table(mname, platform, rows, configs):
    """One booktabs table: rows = configs, columns = max_cycles + resources."""
    header = ["config", "cycles", "LUT", "BRAM18", "DSP", "URAM"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Folding-parameter effect on resources -- {_tex_escape(mname)} "
        rf"({_tex_escape(platform)}).}}",
        rf"  \label{{tab:folding-{mname}}}",
        r"  \begin{tabular}{l r r r r r}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for cname in configs:
        row = rows.get(cname)
        if row is None:
            continue
        if "error" in row:
            cells = [
                _tex_escape(cname),
                r"\multicolumn{5}{c}{" + _tex_escape(row["error"][:40]) + "}",
            ]
        else:
            cells = [_tex_escape(cname)] + [_fmt(row.get(c)) for c in COLUMNS]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def emit_latex(cache, configs, outdir):
    table_files = []
    for mname, rows in cache.items():
        platform = MODELS[mname]["platform"]
        tex = emit_model_table(mname, platform, rows, configs)
        fn = f"table_{mname}.tex"
        with open(os.path.join(outdir, fn), "w") as f:
            f.write(tex)
        table_files.append(fn)
        print(f"[tex] wrote {fn}")

    doc = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\section*{SetFolding parameter vs.\ resource-consumption study}",
    ]
    for fn in sorted(table_files):
        doc.append(rf"\input{{{fn}}}")
    doc += [r"\end{document}", ""]
    main = os.path.join(outdir, "folding_resource_eval.tex")
    with open(main, "w") as f:
        f.write("\n".join(doc))
    print(f"[tex] wrote {main} (pdflatex-compilable, needs booktabs)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--effort", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "folding_eval_out"))
    ap.add_argument("--models-json", help="JSON file overriding the MODELS checkpoint/platform map")
    ap.add_argument(
        "--tex-only", action="store_true", help="regenerate .tex from cached results only"
    )
    args = ap.parse_args()

    if args.models_json:
        with open(args.models_json) as f:
            MODELS.update(json.load(f))

    os.makedirs(args.outdir, exist_ok=True)
    cache_path = os.path.join(args.outdir, "results.json")
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    if not args.tex_only:
        cache = evaluate(args.models, args.configs, args.effort, args.seed, cache)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        print(f"[cache] wrote {cache_path}")

    emit_latex({m: cache[m] for m in cache if m in args.models}, args.configs, args.outdir)


if __name__ == "__main__":
    main()
