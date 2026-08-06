# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fast SWG tree-model harness: no rtlsim, no vivado, no ipgen.

The committed tree model is the reference. ``dump`` freezes its token access
vectors for a configuration matrix into a golden file; ``check`` rebuilds them
from the working tree and reports, per configuration, how far the new model has
moved and -- more importantly -- *in which direction*.

Direction is what decides whether a change is safe. The FIFO sizer takes a
depth from the gap between a producer's cumulative writes and a consumer's
cumulative reads, so for the SWG:

* reading **earlier** than the reference (``in`` above the golden ``in``)
  shrinks the FIFO in front of it -- an undersize risk;
* writing **later** than the reference (``out`` below the golden ``out``)
  shrinks the FIFO behind it -- an undersize risk.

The opposite two directions cost depth but never correctness. So a candidate
model is judged by ``undersize`` (must stay near zero) and ``oversize`` (the
budget the user allows: single-digit percent of the tokens moved).

Usage
-----
    python claude-tools/swg/swg_tav.py dump  --matrix all -o goldens/base.npz
    python claude-tools/swg/swg_tav.py check --matrix all -g goldens/base.npz
    python claude-tools/swg/swg_tav.py check -g goldens/base.npz --fail-under 0
    python claude-tools/swg/swg_tav.py show  --cfg '{"k":[3,3],...}'

``check`` exits non-zero when any configuration exceeds the thresholds, so it
works as the inner loop of an edit-test cycle and as a pytest.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(__file__))

from onnx import TensorProto, helper  # noqa: E402
from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.general.im2col import compute_conv_output_dim  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: E402
from qonnx.util.basic import qonnx_make_model  # noqa: E402

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw  # noqa: E402
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers  # noqa: E402
from swg_configs import get_matrix  # noqa: E402

PART = "xc7z020clg400-1"


def cfg_key(c):
    return json.dumps(c, sort_keys=True)


def build_swg(c):
    """A single specialised ConvolutionInputGenerator node for a configuration."""
    k_h, k_w = c["k"]
    ifm_h, ifm_w = c["ifm_dim"]
    s_h, s_w = c["stride"]
    d_h, d_w = c["dilation"]
    ifm_ch, idt = c["ifm_ch"], DataType["INT8"]
    ofm_h = compute_conv_output_dim(ifm_h, k_h, s_h, 0, d_h)
    ofm_w = compute_conv_output_dim(ifm_w, k_w, s_w, 0, d_w)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_h, ifm_w, ifm_ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_h, ofm_w, k_h * k_w * ifm_ch]
    )
    node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        stride=[s_h, s_w],
        kernel_size=[k_h, k_w],
        input_shape=str((1, ifm_h, ifm_w, ifm_ch)),
        dilations=[d_h, d_w],
        pad_amount=[0, 0, 0, 0],
        pad_value=0,
        depthwise=c["dw"],
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes=[node], name="im2col", inputs=[inp], outputs=[outp])
        )
    )
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model = model.transform(to_hw.InferConvInpGen())
    hw = getCustomOp(model.graph.node[0])
    if c.get("impl") == "hls":
        hw.set_nodeattr("preferred_impl_style", "hls")
    model = model.transform(SpecializeLayers(PART))
    model = model.transform(GiveUniqueNodeNames())
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", c["simd"])
    if model.graph.node[0].op_type == "ConvolutionInputGenerator_rtl":
        inst.set_nodeattr("parallel_window", c["parallel_window"])
        inst.set_nodeattr("M", c.get("m", 1))
    return model, inst


def tav(c):
    """(in_cum, out_cum, period, op_type) from the tree model alone.

    Nothing here touches rtlsim: ``get_tree_model`` plus ``cumulative`` is the
    whole path the analytical sizer uses, so this is milliseconds per config.
    """
    model, inst = build_swg(c)
    tree = inst.get_tree_model()
    if tree is None:
        return None
    cum = tree.cumulative(periods=2)
    period = cum.shape[0] // 2
    return dict(
        inp=cum[:, 0].astype(np.int64),
        out=cum[:, 1].astype(np.int64),
        period=period,
        op_type=model.graph.node[0].op_type,
        tree_name=tree.name,
    )


def compare(gold, new):
    """Signed divergence of a candidate TAV against the golden one.

    Vectors of different length are compared on their common prefix and the
    length difference is reported separately: a period that moved is a bigger
    statement than a few tokens of phase, and the sizer sees both.
    """
    n = min(len(gold["inp"]), len(new["inp"]))
    m = min(len(gold["out"]), len(new["out"]))
    din = new["inp"][:n] - gold["inp"][:n]
    dout = new["out"][:m] - gold["out"][:m]
    tokens = max(int(gold["inp"][-1]), int(gold["out"][-1]), 1)
    # reading early or writing late both shrink a FIFO
    undersize = max(int(din.max(initial=0)), int((-dout).max(initial=0)))
    oversize = max(int((-din).max(initial=0)), int(dout.max(initial=0)))
    return dict(
        undersize=undersize,
        oversize=oversize,
        undersize_frac=undersize / tokens,
        oversize_frac=oversize / tokens,
        period_gold=int(gold["period"]),
        period_new=int(new["period"]),
        period_frac=abs(int(new["period"]) - int(gold["period"])) / max(int(gold["period"]), 1),
        tokens=tokens,
    )


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: v.item() if v.dtype == object else v for k, v in z.items()}


def cmd_dump(args):
    cfgs = get_matrix(args.matrix)
    store, skipped, t0 = {}, [], time.time()
    for c in cfgs:
        key = cfg_key(c)
        try:
            r = tav(c)
        except Exception as e:  # a config the op itself rejects is not our problem
            skipped.append((key, "%s: %s" % (type(e).__name__, e)))
            continue
        if r is None:
            skipped.append((key, "no tree model"))
            continue
        store[key] = np.array(r, dtype=object)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(
        "dumped %d/%d configs to %s in %.1fs (%d skipped)"
        % (len(store), len(cfgs), args.out, time.time() - t0, len(skipped))
    )
    if args.verbose:
        for key, why in skipped:
            print("  skip %s -- %s" % (key, why))


def cmd_check(args):
    gold = load(args.golden)
    cfgs = get_matrix(args.matrix) if args.matrix else [json.loads(k) for k in gold]
    rows, missing, t0 = [], [], time.time()
    for c in cfgs:
        key = cfg_key(c)
        if key not in gold:
            continue
        try:
            new = tav(c)
        except Exception as e:
            rows.append((c, dict(error="%s: %s" % (type(e).__name__, e))))
            continue
        if new is None:
            missing.append(key)
            continue
        rows.append((c, compare(gold[key], new)))
    bad = []
    for c, r in rows:
        if "error" in r:
            bad.append((c, r))
        elif (
            (r["undersize"] > args.const and r["undersize_frac"] > args.fail_under)
            or (r["oversize"] > args.const and r["oversize_frac"] > args.fail_over)
            or r["period_frac"] > args.fail_period
        ):
            bad.append((c, r))
    worst = sorted(
        (r for _, r in rows if "error" not in r),
        key=lambda r: (r["undersize_frac"], r["oversize_frac"]),
        reverse=True,
    )
    print("checked %d configs in %.2fs" % (len(rows), time.time() - t0))
    if worst:
        u = max(r["undersize_frac"] for r in worst)
        o = max(r["oversize_frac"] for r in worst)
        p = max(r["period_frac"] for r in worst)
        print("worst undersize %.4f  oversize %.4f  period %.4f" % (u, o, p))
    if missing:
        print("%d configs lost their tree model (now fall back to rtlsim):" % len(missing))
        for key in missing[: args.top]:
            print("  " + key)
    for c, r in bad[: args.top]:
        print("FAIL %s\n     %s" % (cfg_key(c), r))
    print("%d/%d configs outside budget" % (len(bad), len(rows)))
    return 1 if (bad or (missing and not args.allow_missing)) else 0


def cmd_show(args):
    c = json.loads(args.cfg)
    r = tav(c)
    if r is None:
        print("no tree model for this config")
        return 1
    print("op_type=%s tree=%s period=%d" % (r["op_type"], r["tree_name"], r["period"]))
    print("tokens in=%d out=%d" % (r["inp"][-1], r["out"][-1]))
    if args.plot:
        for i in range(0, r["period"], max(1, r["period"] // 60)):
            print("%8d  in=%-8d out=%-8d" % (i, r["inp"][i], r["out"][i]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--matrix", default="all", choices=["pytest", "models", "stress", "all"])
    d.add_argument("-o", "--out", default="claude-tools/swg/goldens/base.npz")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(fn=cmd_dump)

    c = sub.add_parser("check")
    c.add_argument("-g", "--golden", default="claude-tools/swg/goldens/base.npz")
    c.add_argument("--matrix", default=None, choices=["pytest", "models", "stress", "all"])
    c.add_argument("--fail-under", type=float, default=0.0, help="max undersize fraction")
    c.add_argument(
        "--const",
        type=int,
        default=0,
        help="absolute token budget below which a divergence is ignored, whatever "
        "the fraction -- a few cycles of constant wind-up error. The fractions "
        "govern above it.",
    )
    c.add_argument("--fail-over", type=float, default=0.10, help="max oversize fraction")
    c.add_argument("--fail-period", type=float, default=0.05)
    c.add_argument("--allow-missing", action="store_true")
    c.add_argument("--top", type=int, default=20)
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("show")
    s.add_argument("--cfg", required=True)
    s.add_argument("--plot", action="store_true")
    s.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
