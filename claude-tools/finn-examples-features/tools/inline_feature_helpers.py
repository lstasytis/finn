"""Make each model's feature tests self-contained (tooling only; user requirement 2026-10-20).

    python inline_feature_helpers.py <worktree> <base ref of benchmark_helpers.py>

Copies the feature-test helper block of tests/benchmark/benchmark_helpers.py (from "# --- feature
tests" to "# --- benchmark aggregation", plus small_folding) into every
tests/benchmark/<model>/test_build_<model>.py that uses it, drops those names from the file's
`from benchmark_helpers import (...)`, adds the imports the block needs, and restores
benchmark_helpers.py to <base ref> (upstream's version). Run isort + black afterwards.
"""
import os
import re
import subprocess
import sys

WT, BASE = os.path.abspath(sys.argv[1]), sys.argv[2]
BENCH = os.path.join(WT, "tests/benchmark")
HP = os.path.join(BENCH, "benchmark_helpers.py")
src = open(HP).read()
start = src.index("# --- feature tests (folding, DWC padding, FIFO sizing)")
end = src.index("# --- benchmark aggregation")
block = src[start:end].rstrip() + "\n"
m = re.search(r"\ndef small_folding\(path\):\n(?:    .*\n|\n)*", src)
if m:
    block += "\n" + m.group(0).strip("\n") + "\n"
# drop the shared header comment, keep the code
block = block[block.index("# estimate-only builds") :]
names = re.findall(r"^(?:def |@contextlib\.contextmanager\ndef )?([A-Za-z_]+)(?: = |\()", block, re.M)
names = sorted(set(n for n in names if n not in ("contextlib",)))
LOAD_JSON = '''

def _load_json(path):
    """Load a JSON file, None if it is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
'''
HEADER = (
    "# Feature-test helpers, replicated in every model's test file so that each file is\n"
    "# self-contained (only upstream's benchmark helpers are shared).\n\n"
)
STDLIB = ["contextlib", "fcntl", "functools", "glob", "json", "os", "shutil", "time"]

for mdir in sorted(os.listdir(BENCH)):
    path = os.path.join(BENCH, mdir, f"test_build_{mdir}.py")
    if not os.path.isfile(path):
        continue
    t = open(path).read()
    if "def run_feature_build" not in t or "def cached_build" in t:
        continue
    imp = re.search(r"from benchmark_helpers import \((.*?)\)", t, re.S)
    used = [n.strip() for n in imp.group(1).split(",") if n.strip()]
    kept = [n for n in used if n not in names]
    if "find_cached_build" not in kept:
        kept.append("find_cached_build")
    t = t[: imp.start()] + "from benchmark_helpers import (\n" + "".join(
        f"    {n},\n" for n in kept
    ) + ")" + t[imp.end() :]
    for mod in STDLIB:
        if not re.search(rf"^import {mod}$", t, re.M):
            t = t.replace("\nimport pytest\n", f"\nimport pytest\n\nimport {mod}\n", 1)
    if not re.search(r"^import numpy as np$", t, re.M):
        t = t.replace("\nimport pytest\n", "\nimport pytest\n\nimport numpy as np\n", 1)
    if "import finn.builder.build_dataflow_config as build_cfg" not in t:
        t = t.replace(
            "import finn.builder.build_dataflow as build\n",
            "import finn.builder.build_dataflow as build\n"
            "import finn.builder.build_dataflow_config as build_cfg\n",
            1,
        )
    if not re.search(r"from finn\.util\.basic import .*make_build_dir", t):
        t = t.replace(
            "import finn.builder.build_dataflow as build\n",
            "import finn.builder.build_dataflow as build\nfrom finn.util.basic import make_build_dir\n",
            1,
        )
    # the block goes right after the feature-test header comment
    hdr = t.index("# --- feature tests: FIFO sizing")
    para_end = t.index("\n\n", hdr) + 2
    t = t[:para_end] + HEADER + block + LOAD_JSON + "\n\n" + t[para_end:]
    open(path, "w").write(t)
    print("inlined", path)

subprocess.check_call(["git", "-C", WT, "checkout", BASE, "--", "tests/benchmark/benchmark_helpers.py"])
print("restored benchmark_helpers.py from", BASE)
