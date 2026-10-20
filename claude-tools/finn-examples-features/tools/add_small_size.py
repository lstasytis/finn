"""Add the small-folding variant to the per-model feature tests (tooling only).

    python add_small_size.py <worktree>

In every tests/benchmark/<model>/test_build_<model>.py with feature tests: configure_build,
reference_estimate (own cache prefix "est_<tag>_small_...") and run_feature_build take `small`;
test_<m>_fifo_sizing / _folding / _dwc get `@pytest.mark.parametrize("size", ["small", "full"])`
(maximize skips "small": it does not use the committed folding). benchmark_helpers gets
small_folding(). Run black afterwards.
"""
import os
import re
import sys

BENCH = os.path.join(sys.argv[1], "tests/benchmark")
HELPER = '''

def small_folding(path):
    """The reduced variant (``<name>_small.json``) of a committed folding config: every
    node's parallelism cut 4-8x (claude-tools make_small_foldings.py), so that the
    feature tests' stitched-IP rtlsim runs on a much smaller design first."""
    small = path[: -len(".json")] + "_small.json"
    assert os.path.isfile(small), f"no small folding config {small}"
    return small
'''


def sub1(pattern, repl, src, flags=0):
    out, n = re.subn(pattern, repl, src, count=1, flags=flags)
    assert n == 1, pattern
    return out


def patch_tests(src):
    body_end = r"(?=\n\n\n|\Z)"
    for kind in ("fifo_sizing", "folding", "dwc"):
        m = re.search(rf"\ndef (test_\w+_{kind})\(([^)]*)\):\n", src)
        if m is None:
            continue
        # decorator block above the def: insert the size parametrization on top of its
        # parametrizations (applied last -> last in the test id)
        block_start = src.rfind("\n\n\n", 0, m.start()) + 3
        first_param = src.index("@pytest.mark.parametrize", block_start)
        src = (
            src[:first_param]
            + '@pytest.mark.parametrize("size", ["small", "full"])\n'
            + src[first_param:]
        )
        m = re.search(rf"\ndef (test_\w+_{kind})\(([^)]*)\):\n", src)
        head = f"\ndef {m.group(1)}({m.group(2)}, size):\n    small = size == \"small\"\n"
        if kind == "folding":
            head += (
                '    if small and mode == "maximize":\n'
                '        pytest.skip("maximize does not use the committed folding")\n'
            )
        end = re.compile(body_end).search(src, m.end()).start()
        body = src[m.end() : end]
        body = re.sub(r"reference_estimate\((board(?:, model)?)\)", r"reference_estimate(\1, small=small)", body)
        body = re.sub(r"run_feature_build\(\s*", "run_feature_build(small, ", body)
        src = src[: m.start()] + head + body + src[end:]
    return src


for mdir in sorted(os.listdir(BENCH)):
    path = os.path.join(BENCH, mdir, f"test_build_{mdir}.py")
    if not os.path.isfile(path):
        continue
    src = open(path).read()
    if "def run_feature_build" not in src or "small_folding" in src:
        continue
    src = sub1(
        r"def configure_build\((board(?:, model)?), output_dir, \*\*overrides\):",
        r"def configure_build(\1, output_dir, small=False, **overrides):",
        src,
    )
    src = sub1(
        r"folding_config_file=folding,",
        "folding_config_file=small_folding(folding) if small else folding,",
        src,
    )
    src = sub1(
        r"def reference_estimate\((board(?:, model)?)\):",
        r"def reference_estimate(\1, small=False):",
        src,
    )
    src = sub1(
        r"(            output_dir,\n)(            generate_outputs=ESTIMATE_OUTPUTS,)",
        r"\1            small=small,\n\2",
        src,
    )
    src = sub1(r'f"est_([A-Za-z0-9-]+)_', r"""f"est_\1_{'small_' if small else ''}""", src)
    src = sub1(
        r"def run_feature_build\((board(?:, model)?), name,",
        r"def run_feature_build(small, \1, name,",
        src,
    )
    src = sub1(
        r"ref_dir = reference_estimate\((board(?:, model)?)\)",
        r"ref_dir = reference_estimate(\1, small=small)",
        src,
    )
    src = sub1(r'make_build_dir\(f"\{name\}_', """make_build_dir(f"{name}{'-small' if small else ''}_""", src)
    src = sub1(
        r"(        output_dir,\n)(        steps=feature_steps\()",
        r"\1        small=small,\n\2",
        src,
    )
    src = patch_tests(src)
    src = sub1(r"from benchmark_helpers import \(\n", "from benchmark_helpers import (\n    small_folding,\n", src)
    open(path, "w").write(src)
    print("ok", path)

hp = os.path.join(BENCH, "benchmark_helpers.py")
h = open(hp).read()
if "def small_folding" not in h:
    open(hp, "w").write(h.rstrip("\n") + "\n" + HELPER)
    print("ok", hp)
