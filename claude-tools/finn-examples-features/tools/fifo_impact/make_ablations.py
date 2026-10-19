"""Write one scratch copy of derive_characteristic.py per ablated heuristic.

    python make_ablations.py <out_dir>

Each variant is the worktree module with one textual edit; the edit must match
exactly or the script fails, so a refactor that moves code is noticed.
"""
import os
import sys

src = os.path.join(
    os.environ["FINN_ROOT"], "src/finn/transformation/fpgadataflow/derive_characteristic.py"
)
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
base = open(src).read()

ABLATIONS = {
    # slack relaxation off: allowance = 0
    "no_slack": [
        (
            "    allowance = edge.per_frame * (1.0 - min(edge.t_up, global_period) / float(global_period))\n",
            "    allowance = 0.0\n",
        )
    ],
    # burst floor off
    "no_burst_floor": [
        (
            "    floor = min(peak, burst_above_rate(read_times, edge.per_frame / float(global_period)))\n",
            "    floor = 0\n",
        )
    ],
    # throttled cap off
    "no_cap": [("        floor = min(floor, THROTTLED_CAP)\n    return max(after_slack, floor)", "        pass\n    return max(after_slack, floor)")],
    # cap margin guard off
    "no_cap_margin": [("        and not (peak > 0 and allowance < CAP_MARGIN * peak)\n", "        and True\n")],
    # pacer tolerance off
    "no_pacer_tol": [("PACER_TOLERANCE = 0.995\n", "PACER_TOLERANCE = 1.0\n")],
    # only the non-causal pass
    "no_causal_pass": [
        (
            "        depths = derive_chained_tav_depths(model, self.period, causal=True)\n",
            "        depths = {}\n",
        )
    ],
    # only the causal pass
    "only_causal_pass": [
        (
            "        phased = derive_chained_tav_depths(model, self.period, causal=False)\n",
            "        phased = {}\n",
        )
    ],
    # peak over both frames at once, no fill/steady split
    "no_frame_split": [
        ("    if per_frame and 2 * per_frame <= n:\n", "    if False:\n"),
    ],
    # fill_only edges take the max of both frames like chain edges
    "no_fill_only": [("        if fill_only:\n            return fill\n", "        if False:\n            return fill\n")],
    # graph input not paced: every token available at time 0
    "no_input_pacing": [
        (
            "    return (np.arange(1, int(curve[-1]) + 1) * rate).astype(np.int64)\n",
            "    return np.zeros(int(curve[-1]), dtype=np.int64)\n",
        )
    ],
    # global period from the caller's estimate only
    "no_measured_period": [
        (
            "    global_period = max(int(global_period or 0), max(own_period.values(), default=0), 1)\n",
            "    global_period = max(int(global_period or 0), 1)\n",
        )
    ],
    # idle-window floor off
    "no_idle_floor": [("    half = len(curve) // 2\n    if half < 2:\n        return 0\n", "    half = len(curve) // 2\n    if True:\n        return 0\n")],
    # joins / branch edges relaxed like chain edges
    "no_join_rule": [("    if fill_only:\n        return max(int(peak), int(idle_floor))\n", "    if False:\n        return max(int(peak), int(idle_floor))\n")],
    # QUIRK input_rows_collapsed off
    "no_quirk_collapse": [("            if rows_in is not None and input_rows_collapsed(model, node):\n", "            if False:\n")],
    # QUIRK inherited fork/join lengths off
    "no_quirk_inherit": [("    inherited = inherited_fork_join_tav_lengths(model)\n", "    inherited = {}\n")],
    # ElementwiseAdd with a TAV is characterized, not transparent
    "no_transparent_ewadd": [
        (
            '    "ElementwiseAdd_hls",\n    "ElementwiseAdd_rtl",\n',
            "",
        )
    ],
    # drives_pacer: nothing is throttleable
    "no_throttleable": [("        drives_pacer[node.name] = own_period.get(node.name, 0) >= pacer_period or below\n", "        drives_pacer[node.name] = True\n")],
    # pacer propagation: t_prop never reset past a pacer
    "no_pacer_reset": [("        propagated = 0 if own >= pacer_period else chain\n", "        propagated = chain\n")],
}

for name, edits in ABLATIONS.items():
    s = base
    if not all(old in s for old, _ in edits):
        print("skipped (pattern gone):", name)
        continue
    for old, new in edits:
        s = s.replace(old, new)
    with open(os.path.join(out_dir, name + ".py"), "w") as f:
        f.write(s)
print("wrote", len(ABLATIONS), "variants to", out_dir)
