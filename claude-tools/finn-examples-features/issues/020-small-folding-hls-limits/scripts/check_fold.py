"""Estimate-only feature build (folding optimizer) and a check of every node against the
HLS limits: check_fold.py <worktree> <model dir> <small 0|1> <mode target|maximize> <args...>"""
import glob
import importlib.util
import os
import sys
import time

wt, mdir, small, mode = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
args = sys.argv[5:]
bench = os.path.join(wt, "tests/benchmark")
sys.path[:0] = [bench, os.path.join(bench, mdir)]
spec = importlib.util.spec_from_file_location("tb", os.path.join(bench, mdir, f"test_build_{mdir}.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

from finn.transformation.fpgadataflow.set_folding import Optimizer  # noqa: E402

over = dict(mod.OPTIMIZER_FOLDING)
name = f"chk020-{mode}"
if mode == "target":
    over["target_fps"] = int(mod.estimated_throughput(mod.reference_estimate(*args, small=small)))
t0 = time.time()
metrics = mod.run_feature_build(small, *args, name, estimate_only=True, **over)
print("metrics", {k: metrics.get(k) for k in ("est_fps", "step_times_s")}, f"{time.time() - t0:.0f} s")
out = sorted(glob.glob(os.path.join(os.environ["FINN_BUILD_DIR"], f"{name}{'-small' if small else ''}_*")),
             key=os.path.getmtime)[-1]
m = ModelWrapper(os.path.join(out, "intermediate_models", "step_target_fps_parallelization.onnx"))
bad = [n.name for n in m.graph.node if Optimizer.breaks_hls_limits(getCustomOp(n))]
for n in m.graph.node:
    if n.op_type in ("MVAU_hls", "VVAU_hls"):
        i = getCustomOp(n)
        print(" ", n.name, i.get_nodeattr("mem_mode"), "PE", i.get_nodeattr("PE"), "SIMD", i.get_nodeattr("SIMD"),
              "MW/MH", i.get_nodeattr("MW") if n.op_type == "MVAU_hls" else "-",
              i.get_nodeattr("MH") if n.op_type == "MVAU_hls" else "-")
print(out)
print("HLS limit violations:", bad or "none")
