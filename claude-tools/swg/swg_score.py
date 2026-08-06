# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Score a candidate SWG nest against the FSM oracle, per configuration.

Reports, over two back-to-back periods, the signed worst divergence of the
cumulative read and write vectors -- ``under`` (reads early / writes late, the
undersizing direction) and ``over`` -- plus the period error, plus whether the
error *grows* between the first and the second period, which is the thing a
wrong period does and a constant offset does not.

    python claude-tools/swg/swg_score.py --matrix all
"""

import argparse
import numpy as np
import os
import sys

os.environ.setdefault(
    "FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.insert(0, os.path.dirname(__file__))

from swg_configs import get_matrix  # noqa: E402
from swg_fold import period_delta  # noqa: E402
from swg_tav import build_swg  # noqa: E402

if os.environ.get("SWG_DEMAND_LEAD"):
    raise SystemExit(
        "SWG_DEMAND_LEAD is no longer honoured: the tree model and its helpers now live "
        "entirely inside ConvolutionInputGenerator.get_tree_model, so there is no "
        "module-level swg_demand_lead to patch. Edit the nested demand_lead in "
        "src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py and score with "
        "swg_score.py directly."
    )


def cum2(delta):
    """Two back-to-back periods of cumulative (reads, writes)."""
    one = np.cumsum(np.asarray(delta, dtype=np.int64), axis=0)
    return np.concatenate([one, one + one[-1]])


def score(ref_delta, cand_node):
    ref = cum2(ref_delta)
    cand = cand_node.cumulative(periods=2)
    n = min(len(ref), len(cand))
    d = cand[:n] - ref[:n]
    # reads early (cand in above ref) or writes late (cand out below ref) shrink a FIFO
    under = max(int(d[:, 0].max(initial=0)), int((-d[:, 1]).max(initial=0)))
    over = max(int((-d[:, 0]).max(initial=0)), int(d[:, 1].max(initial=0)))
    half = n // 2
    growth = (
        int(np.abs(d[half:]).max(initial=0)) - int(np.abs(d[:half]).max(initial=0)) if half else 0
    )
    return dict(
        under=under,
        over=over,
        growth=growth,
        p_ref=len(ref_delta),
        p_cand=len(cand) // 2,
        tok=max(int(ref[-1, 0]), int(ref[-1, 1])),
        r_ref=int(np.sum(np.asarray(ref_delta)[:, 0])),
        w_ref=int(np.sum(np.asarray(ref_delta)[:, 1])),
        r_cand=int(cand_node.sum(0)),
        w_cand=int(cand_node.sum(1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="all")
    ap.add_argument("--aligned", action="store_true", default=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--const", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for c in get_matrix(a.matrix):
        model, inst = build_swg(c)
        if "_rtl" not in type(inst).__name__:
            continue
        style = inst.select_impl_style()
        ref, p = period_delta(inst, aligned=True)
        if ref is None:
            continue
        node = inst.get_tree_model()
        if node is None:
            rows.append((c, style, None))
            continue
        s = score(ref, node)
        s["frac_under"] = s["under"] / max(1, s["tok"])
        s["frac_over"] = s["over"] / max(1, s["tok"])
        s["frac_period"] = abs(s["p_cand"] - s["p_ref"]) / max(1, s["p_ref"])
        rows.append((c, style, s))

    ok = [r for r in rows if r[2] is not None]
    print(f"{len(ok)}/{len(rows)} configs modelled")
    for key in ("frac_under", "frac_over", "frac_period"):
        v = sorted((r[2][key] for r in ok), reverse=True)
        print(f"  worst {key}: {v[0]:.4f}   p95 {v[len(v)//20]:.4f}   median {v[len(v)//2]:.4f}")
    bad_tok = [r for r in ok if r[2]["r_cand"] != r[2]["r_ref"] or r[2]["w_cand"] != r[2]["w_ref"]]
    print(f"  token counts exact on {len(ok)-len(bad_tok)}/{len(ok)}")
    grow = [r for r in ok if r[2]["growth"] > a.const]
    print(f"  error grows between periods on {len(grow)}/{len(ok)}")
    fail = [
        r
        for r in ok
        if (r[2]["under"] > a.const and r[2]["frac_under"] > 0.01)
        or (r[2]["over"] > a.const and r[2]["frac_over"] > 0.10)
    ]
    print(
        f"  GATE (--const {a.const} --fail-under 0.01 --fail-over 0.10): "
        f"{len(ok)-len(fail)}/{len(ok)} pass, {len(fail)} fail"
    )

    print("\nfailing the gate:")
    for c, style, s in sorted(fail, key=lambda r: -max(r[2]["frac_under"], r[2]["frac_over"]))[
        : a.top
    ]:
        print(
            f"  u{s['frac_under']:.4f}({s['under']}) o{s['frac_over']:.4f}({s['over']})"
            f" p{s['frac_period']:.4f} grow{s['growth']:+d} | per {s['p_ref']}->{s['p_cand']}"
            f" rd {s['r_ref']}->{s['r_cand']} wr {s['w_ref']}->{s['w_cand']}"
            f" | {style[:4]} k{c['k']} ifm{c['ifm_dim']} s{c['stride']} d{c['dilation']}"
            f" ch{c['ifm_ch']} simd{c['simd']} dw{c['dw']} pw{c['parallel_window']}"
        )
    print("\nworst by frac_under:")
    for c, style, s in sorted(ok, key=lambda r: -r[2]["frac_under"])[: a.top]:
        print(
            f"  u{s['frac_under']:.4f} o{s['frac_over']:.4f} p{s['frac_period']:.4f}"
            f" grow{s['growth']:+d} | per {s['p_ref']}->{s['p_cand']}"
            f" rd {s['r_ref']}->{s['r_cand']} wr {s['w_ref']}->{s['w_cand']}"
            f" | {style[:4]} k{c['k']} ifm{c['ifm_dim']} s{c['stride']} d{c['dilation']}"
            f" ch{c['ifm_ch']} simd{c['simd']} dw{c['dw']} pw{c['parallel_window']}"
        )
    print("\nworst by frac_over:")
    for c, style, s in sorted(ok, key=lambda r: -r[2]["frac_over"])[: a.top]:
        print(
            f"  u{s['frac_under']:.4f} o{s['frac_over']:.4f} p{s['frac_period']:.4f}"
            f" grow{s['growth']:+d} | per {s['p_ref']}->{s['p_cand']}"
            f" rd {s['r_ref']}->{s['r_cand']} wr {s['w_ref']}->{s['w_cand']}"
            f" | {style[:4]} k{c['k']} ifm{c['ifm_dim']} s{c['stride']} d{c['dilation']}"
            f" ch{c['ifm_ch']} simd{c['simd']} dw{c['dw']} pw{c['parallel_window']}"
        )


if __name__ == "__main__":
    main()
