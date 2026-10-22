"""DWC tests compare the folding optimizer without and with padding (tooling only, issue 022).

    python add_dwc_pad_compare.py <worktree>

Per model test file: run_feature_build stores extra metrics, the padding-comparison helpers are
added after assert_target_or_trade, and test_<m>_dwc
  * runs the optimizer estimate-only at the reference target with padding off and on for every
    seed in DWC_SEEDS and takes the seed with the largest resource gain of padding,
  * (full size) compares the maximize-mode throughput without and with padding,
  * builds the chosen padded folding (stitched IP + rtlsim, as before; no skip any more) with
    the comparison in its feature_metrics.json ("pad_compare"),
  * warns if the padded folding needs more resources than the unpadded one, fails above 5 %.
Run isort + black on the files afterwards.
"""
import os
import re
import sys

HELPERS = '''

# DWC padding vs none: the optimizer at the same target and seed with padding off and on
RESOURCES = ("LUT", "BRAM_18K", "URAM", "DSP")
# padding only widens the search space: a padded folding needing more resources than the
# unpadded one is annealing noise up to this fraction (warning), beyond it the test fails
PADDING_WORSE_TOLERANCE = 0.05


def estimated_resources(output_dir):
    """A build's estimated LUT/BRAM_18K/URAM/DSP counts (FIFOs excluded)."""
    layers = _load_json(os.path.join(output_dir, "report", "estimate_layer_resources.json"))
    used = dict.fromkeys(RESOURCES, 0.0)
    for name, res in (layers or {}).items():
        if name != "total" and not name.startswith("StreamingFIFO") and isinstance(res, dict):
            for r in RESOURCES:
                used[r] += float(res.get(r, 0) or 0)
    return used


def resource_ratio(new, base):
    """Mean of new/base over the resources ``base`` uses, and the resources only ``new``
    uses (the res./ref. metric of the claude-tools summary)."""
    ratios = [new[r] / base[r] for r in RESOURCES if base[r] > 0]
    extra = [r for r in RESOURCES if base[r] == 0 and new[r] > 0]
    return (sum(ratios) / len(ratios) if ratios else 1.0), extra


def padding_summary(metrics):
    """The numbers of one estimate-only folding kept in the comparison."""
    return dict(
        est_fps=metrics["est_fps"],
        pad_dwcs=metrics.get("pad_dwcs"),
        resources=estimated_resources(metrics["output_dir"]),
        output_dir=metrics["output_dir"],
    )


def padding_comparison(build, target_fps):
    """Estimate-only optimizer foldings at ``target_fps`` without and with padding (same
    seed) for every seed in DWC_SEEDS; ``build(name, estimate_only=True, **overrides)``
    runs one. Returns the comparison of the seed where padding saves the most resources."""
    best, best_key, seeds = None, None, []
    for seed in DWC_SEEDS:
        runs = [
            padding_summary(
                build(
                    name,
                    estimate_only=True,
                    **dict(
                        OPTIMIZER_FOLDING,
                        target_fps=target_fps,
                        folding_maximum_padding=pad,
                        folding_seed=seed,
                    ),
                )
            )
            for name, pad in (("dwc-est-nopad", 0), ("dwc-est", FOLDING_MAX_PADDING))
        ]
        ratio, extra = resource_ratio(runs[1]["resources"], runs[0]["resources"])
        row = dict(seed=seed, res_ratio=ratio, res_extra=extra, nopad=runs[0], pad=runs[1])
        seeds.append(dict(seed=seed, res_ratio=ratio, res_extra=extra, pad_dwcs=runs[1]["pad_dwcs"]))
        # on a tie the padded folding that pads, so that the hardware build has padding DWCs
        key = (bool(extra), ratio, not runs[1]["pad_dwcs"])
        if best is None or key < best_key:
            best, best_key = row, key
    if best["res_extra"] or best["res_ratio"] > 1:
        best["verdict"] = "worse"
    elif not best["pad"]["pad_dwcs"]:
        best["verdict"] = "no gain (no padding DWC)"
    else:
        best["verdict"] = "gain" if best["res_ratio"] < 1 else "no gain"
    return dict(best, target_fps=target_fps, seeds=seeds)


def padding_maximize(build):
    """Estimated throughput of the maximize-mode optimizer (no target) without and with
    padding."""
    nopad, pad = [
        padding_summary(
            build(
                name,
                estimate_only=True,
                **dict(OPTIMIZER_FOLDING, folding_maximum_padding=padding),
            )
        )
        for name, padding in (("dwc-max-nopad", 0), ("dwc-max", FOLDING_MAX_PADDING))
    ]
    ratio, extra = resource_ratio(pad["resources"], nopad["resources"])
    fps_ratio = pad["est_fps"] / nopad["est_fps"]
    if fps_ratio < 1:
        warnings.warn("maximize: padded folding %.3fx the unpadded throughput" % fps_ratio)
    return dict(fps_ratio=fps_ratio, res_ratio=ratio, res_extra=extra, nopad=nopad, pad=pad)


def assert_padding_not_worse(compare):
    """Beyond annealing noise the padded folding needs no more resources than the unpadded
    one at the same target and seed (warning above 1.00x, failure above
    1 + PADDING_WORSE_TOLERANCE)."""
    ratio, extra = compare["res_ratio"], compare["res_extra"]
    msg = "padded folding needs %.3fx the resources of the unpadded one%s (seed %d)" % (
        ratio,
        " and " + "/".join(extra) if extra else "",
        compare["seed"],
    )
    if compare["verdict"] == "worse":
        warnings.warn(msg)
    assert ratio <= 1 + PADDING_WORSE_TOLERANCE, msg
'''

BODY = """    build = functools.partial(run_feature_build, small, {args})
    # padding off vs on at the reference target (same seed): the seed where padding saves the
    # most resources is built, also when padding gains nothing ("verdict" in the metrics)
    compare = padding_comparison(build, target_fps)
    if not small:
        # maximize mode (no small variant): the throughput padding adds within the budget
        compare["maximize"] = padding_maximize(build)
    padded = dict(
        OPTIMIZER_FOLDING,
        target_fps=target_fps,
        folding_maximum_padding=FOLDING_MAX_PADDING,
        folding_seed=compare["seed"],
    )
    metrics = build("dwc", extra_metrics=dict(pad_compare=compare), **padded)
    assert_target_or_trade(metrics, ref_dir, target_fps, board)
    assert_padding_not_worse(compare)
"""

BENCH = os.path.join(sys.argv[1], "tests/benchmark")
for mdir in sorted(os.listdir(BENCH)):
    path = os.path.join(BENCH, mdir, f"test_build_{mdir}.py")
    if not os.path.isfile(path):
        continue
    t = open(path).read()
    if "def assert_target_or_trade" not in t or "def padding_comparison" in t:
        continue
    # run_feature_build stores extra metrics with its own
    old = "name, estimate_only=False, **overrides):"
    assert t.count(old) == 1, path
    t = t.replace(old, "name, estimate_only=False, extra_metrics=None, **overrides):")
    old = "check the stitched IP in rtlsim. Returns the metrics.\"\"\""
    assert t.count(old) == 1, path
    t = t.replace(
        old,
        "check the stitched IP in rtlsim. Returns the metrics\n"
        "    (with ``extra_metrics``, also stored in feature_metrics.json).\"\"\"",
    )
    old = "test=name, step_times_s=step_times)"
    assert t.count(old) == 1, path
    t = t.replace(old, "test=name, step_times_s=step_times, **(extra_metrics or {}))")
    # helpers after assert_target_or_trade
    anchor = "\n\ndef reference_estimate("
    assert t.count(anchor) == 1 and t.index(anchor) > t.index("def assert_target_or_trade"), path
    t = t.replace(anchor, HELPERS + anchor)
    # the DWC test body
    m = re.search(
        r"    # the folding search is cheap: find a seed.*?"
        r"metrics = run_feature_build\(small, (board(?:, model)?), \"dwc\", \*\*padded\)\n"
        r"    assert_target_or_trade\(metrics, ref_dir, target_fps, board\)\n",
        t,
        re.S,
    )
    assert m, path
    t = t[: m.start()] + BODY.format(args=m.group(1)) + t[m.end() :]
    open(path, "w").write(t)
    print("ok", path)
