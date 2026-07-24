#!/usr/bin/env python3
"""Folding evaluation: compare the resource-aware optimizer against the reference
folding JSON at matched throughput, using REAL OOC-synth resources (not estimates).

Wiring notes:
- Caches the pre-folding checkpoint (step_specialize_layers) per model in a stable
  dir, so re-running a folding method only re-runs folding -> codegen -> ipgen ->
  OOC synth, never the streamline/convert/specialize prefix.
- "json": apply the model's reference folding_config_file.
  "optimizer": SetFolding style=optimizer targeting the JSON's achieved cycles.
  "naive": legacy greedy SetFolding at the same target.
- Resources come from report/ooc_synth_and_timing.json (Vivado OOC synth).

Usage: python folding_eval.py <model_dir> <board> [methods=json,optimizer] [effort]
Run from the finn repo root.
"""
import importlib.util
import inspect
import json
import os
import sys
import glob

REPO = "/home/lstasytis/backup/finn"
os.chdir(REPO)
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir
from qonnx.core.modelwrapper import ModelWrapper

model_dir = sys.argv[1]
board = sys.argv[2] if len(sys.argv) > 2 else "Pynq-Z1"
methods = (sys.argv[3] if len(sys.argv) > 3 else "json,optimizer").split(",")
effort = int(sys.argv[4]) if len(sys.argv) > 4 else 100

BUILD = os.environ["FINN_BUILD_DIR"]
CACHE = f"{BUILD}/foldeval_{model_dir}_{board}"

cands = glob.glob(f"tests/benchmark/{model_dir}/test_build_*.py")
mod_path = cands[0]
sys.path.insert(0, os.path.dirname(os.path.abspath(mod_path)))
spec = importlib.util.spec_from_file_location("bench_mod", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model_file = mod.model_file


def base_cfg(out):
    sig = inspect.signature(mod.configure_build)
    if len(list(sig.parameters)) == 1:
        cfg = mod.configure_build(board)
    else:
        cfg = mod.configure_build(board, out)
    cfg.output_dir = out
    cfg.verify_steps = []
    cfg.enable_build_pdb_debug = False
    cfg.save_intermediate_models = True
    return cfg


def steps_upto(cfg, stop):
    """The model's step list truncated up to and including `stop`."""
    steps = cfg.steps or list(build_cfg.default_build_dataflow_steps)
    out = []
    for s in steps:
        out.append(s)
        nm = s if isinstance(s, str) else getattr(s, "__name__", "")
        if nm == stop:
            break
    return out


# ---- 1) shared pre-folding checkpoint (streamline/convert/specialize) ----
ckpt = f"{CACHE}/intermediate_models/step_specialize_layers.onnx"
if not os.path.isfile(ckpt):
    os.makedirs(CACHE, exist_ok=True)
    cfg = base_cfg(CACHE)
    cfg.generate_outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
    # phase-based lists bundle specialize inside phase_convert_to_hardware
    steps = cfg.steps or list(build_cfg.default_build_dataflow_steps)
    trunc = []
    for s in steps:
        nm = s if isinstance(s, str) else getattr(s, "__name__", "")
        trunc.append(s)
        if nm in ("phase_convert_to_hardware", "step_specialize_layers"):
            break
    cfg.steps = trunc
    print(f"[foldeval] building shared pre-folding checkpoint -> {CACHE}")
    assert build.build_dataflow_cfg(model_file, cfg) == 0, "checkpoint build failed"
else:
    print(f"[foldeval] reusing cached checkpoint {ckpt}")


def parse_synth_util(proj_dir):
    """Top-of-design resources from the OOC synth utilization report (no P&R)."""
    rpt = os.path.join(proj_dir, "finn_design_partition_util.rpt")
    if not os.path.isfile(rpt):
        return None
    with open(rpt) as f:
        for line in f:
            if line.startswith("| finn_design_wrapper "):
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                # cols: Instance, Module, Total LUTs, Logic LUTs, LUTRAMs, SRLs,
                #       FFs, RAMB36, RAMB18, DSP Blocks
                lut, _, lutram, srl, ff, ramb36, ramb18, dsp = (int(x) for x in c[2:10])
                return {"LUT": lut, "LUTRAM": lutram, "SRL": srl, "FF": ff,
                        "BRAM_18K": ramb36 * 2 + ramb18, "DSP": dsp}
    return None


def run_method(method):
    # stable per-method dir so a completed synth is reused on re-run
    out = f"{CACHE}/method_{method}"
    cached = parse_synth_util(out + "/stitched_ip")
    if cached is not None and os.path.isfile(out + "/report/estimate_network_performance.json"):
        perf = json.load(open(out + "/report/estimate_network_performance.json"))
        cached = dict(cached)
        cached["cycles"] = perf.get("max_cycles")
        print(f"[foldeval] reusing cached synth for method={method}")
        return cached, out
    os.makedirs(out, exist_ok=True)
    cfg = base_cfg(out)
    # regular OOC synthesis only (no place & route) is enough for resource counts
    cfg.stitched_ip_gen_dcp = True
    cfg.generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
    ]
    # from the specialize checkpoint: fold -> minimize -> codegen -> ipgen -> stitched+OOC
    cfg.steps = [
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_create_stitched_ip",
    ]
    # pick the folding driver
    if method == "json":
        cfg.target_fps = None  # rely on folding_config_file (already set by configure_build)
    else:
        cfg.folding_config_file = None
        cfg.folding_style = "optimizer" if method == "optimizer" else "naive"
        cfg.folding_effort = effort
        # target = JSON's achieved cycles (read from the json run if available)
        cyc = TARGETS.get("json")
        clock_hz = 1e9 / cfg.synth_clk_period_ns
        cfg.target_fps = int(clock_hz / cyc) if cyc else None
    assert build.build_dataflow_cfg(ckpt, cfg) == 0, f"{method} build failed"
    res = parse_synth_util(out + "/stitched_ip")
    perf = json.load(open(out + "/report/estimate_network_performance.json"))
    res = dict(res or {})
    res["cycles"] = perf.get("max_cycles")
    return res, out


TARGETS = {}
results = {}
# json first so optimizer/naive can target its cycles
for m in sorted(methods, key=lambda x: 0 if x == "json" else 1):
    print(f"[foldeval] running method={m}")
    r, out = run_method(m)
    results[m] = r
    if m == "json":
        TARGETS["json"] = r["cycles"]
    print(f"[foldeval] {m}: {r}  ({out})")

print(f"\n=== FOLDING SYNTH COMPARISON ({model_dir}, {board}) ===")
cols = ["cycles", "LUT", "FF", "BRAM_18K", "DSP", "LUTRAM", "SRL"]
print(f"{'metric':<10}" + "".join(f"{m:>12}" for m in methods))
for c in cols:
    print(f"{c:<10}" + "".join(f"{str(results[m].get(c)):>12}" for m in methods))
json.dump(results, open(f"{CACHE}/folding_eval_results.json", "w"), indent=2)
print(f"\nsaved {CACHE}/folding_eval_results.json")
