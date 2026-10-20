"""Add the optional FIFO-aware folding mode to every model's folding test (tooling only).

    python add_fifoaware_mode.py <worktree>
"""
import os
import re
import sys

BENCH = os.path.join(sys.argv[1], "tests/benchmark")
for mdir in sorted(os.listdir(BENCH)):
    path = os.path.join(BENCH, mdir, f"test_build_{mdir}.py")
    if not os.path.isfile(path):
        continue
    t = open(path).read()
    if 'def test_' not in t or '"fold-target"' not in t or "fifo-aware" in t:
        continue
    t = t.replace(
        '@pytest.mark.parametrize("mode", ["target", "maximize"])',
        '@pytest.mark.parametrize("mode", ["target", "maximize", "fifo-aware"])',
        1,
    )
    args = re.search(r'run_feature_build\(\s*small, (board(?:, model)?), "fold-target"', t).group(1)
    new = (
        '    elif mode == "fifo-aware":\n'
        "        # optional, slower: the search scores foldings with FIFOs sized by the analytic sizer\n"
        "        metrics = run_feature_build(\n"
        f'            small, {args}, "fold-fifoaware", target_fps=target_fps,\n'
        "            enable_folding_fifo_heuristic=True, **OPTIMIZER_FOLDING\n"
        "        )\n"
        '        assert metrics["est_fps"] >= target_fps\n'
        "    else:\n"
        "        # no target: the optimizer maximizes the throughput within the budget\n"
    )
    old = "    else:\n        # no target: the optimizer maximizes the throughput within the budget\n"
    assert t.count(old) == 1, path
    t = t.replace(old, new)
    open(path, "w").write(t)
    print("ok", path)
