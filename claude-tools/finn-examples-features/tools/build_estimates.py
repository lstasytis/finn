"""Build (or find cached) the reference estimate builds of the feature tests.

Usage (inside a worktree env, see wtenv.sh):
    python build_estimates.py [model_dir ...]      # default: all
Imports each tests/benchmark/<model>/test_build_<model>.py and calls its
reference_estimate() for every FEATURE_BUILDS entry; prints the build dirs.
"""
import importlib.util
import os
import sys
import traceback

root = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")
sys.path.insert(0, root)
models = sys.argv[1:] or sorted(
    d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, f"test_build_{d}.py"))
)
failed = []
for d in models:
    sys.path.insert(0, os.path.join(root, d))
    spec = importlib.util.spec_from_file_location(
        f"tb_{d.replace('-', '_')}", os.path.join(root, d, f"test_build_{d}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for entry in mod.FEATURE_BUILDS:
        args = tuple(entry.values) if hasattr(entry, "values") else (entry,)
        try:
            out = mod.reference_estimate(*args)
            print("EST", d, *args, out, mod.estimated_throughput(out), flush=True)
        except Exception:
            traceback.print_exc()
            print("EST_FAIL", d, *args, flush=True)
            failed.append((d, args))
sys.exit(1 if failed else 0)
