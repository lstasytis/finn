# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split dominant MVAU nodes into k smaller MVAUs working in parallel, lane-wise, without
throughput loss (finn.custom_op.fpgadataflow.streaminglanes).

Output split (k divides PE):
    X -> DuplicateStreams(k) -> MVAU_i(MW, MH/k, PE/k, SIMD) -> StreamingLaneMerge(PE lanes) -> Y
  MVAU_i computes the output lanes [i*PE/k, (i+1)*PE/k) of every output fold, i.e. the original
  output channels f*PE + i*PE/k + j (weight columns and threshold rows in that order); the merge
  puts the k words of a fold side by side.

Input split (k divides SIMD; for small PE, e.g. MobileNet's PE=1 layers):
    X -> StreamingLaneSplit(SIMD lanes) -> MVAU_i(MW/k, MH, PE, SIMD/k, no activation)
      -> ElementwiseAdd tree -> [Thresholding] -> Y
  MVAU_i gets the input lanes [i*SIMD/k, (i+1)*SIMD/k) of every input fold (weight rows of the
  input channels f*SIMD + i*SIMD/k + j) and produces partial sums; their sum is the original
  accumulator, thresholded afterwards if the MVAU had an activation.

Every part has 1/k of the multipliers at the original fold counts; the split/merge move one word
per cycle on every port, so the parts run in lockstep at the original rate.

Used by the island flow (finn.util.rwislands, dynarapid option "split"): a single large node is
otherwise one island whose synthesis and place and route bound the parallel build time (VGG10
8x: four fully unrolled MVAUs, PE = MH = 32, 58k LUTs / 768 DSPs, ~370 s synthesis and ~800 s
P&R each; MobileNet: SIMD = 512, PE = 1).
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes

MVAU_TYPES = ("MVAU", "MVAU_hls", "MVAU_rtl")
FPGADATAFLOW = "finn.custom_op.fpgadataflow"


def work(node):
    """Size proxy of an MVAU: multiply-accumulates per cycle times operand bits."""
    inst = getCustomOp(node)
    wbits = DataType[inst.get_nodeattr("weightDataType")].bitwidth()
    abits = inst.get_input_datatype().bitwidth()
    return inst.get_nodeattr("PE") * inst.get_nodeattr("SIMD") * wbits * abits


def _largest_divisor(n, k_max):
    for k in range(min(k_max, n), 1, -1):
        if n % k == 0:
            return k
    return 1


def split_plan(node, k_want):
    """("out" | "in", k) for the node, or (None, 1) if it cannot be split."""
    inst = getCustomOp(node)
    k_out = _largest_divisor(inst.get_nodeattr("PE"), k_want)
    k_in = _largest_divisor(inst.get_nodeattr("SIMD"), k_want)
    if k_out >= max(2, k_in):
        return "out", k_out
    if k_in >= 2:
        return "in", k_in
    return None, 1


def splittable(model, node):
    inst = getCustomOp(node)
    if node.op_type not in MVAU_TYPES:
        return False
    if inst.get_nodeattr("runtime_writeable_weights") == 1 or inst.get_nodeattr("mem_mode") == "external":
        return False
    if inst.get_nodeattr("binaryXnorMode") == 1:
        return False
    # weights (and thresholds) must be initializers to be sliced
    return all(model.get_initializer(t) is not None for t in node.input[1:])


def acc_datatype(W, idt):
    """Smallest integer type holding x @ W for every x of type idt (bounds every partial sum
    too, since each term's extreme has the sign of the whole sum's extreme or is 0)."""
    lo, hi = idt.min(), idt.max()
    pos = np.maximum(W * hi, W * lo).sum(axis=0)
    neg = np.minimum(W * hi, W * lo).sum(axis=0)
    vmin, vmax = int(min(0, neg.min())), int(max(0, pos.max()))
    if vmin >= 0:
        return DataType.get_smallest_possible(vmax)
    return DataType.get_smallest_possible(min(vmin, -vmax - 1))


class SplitLargeMVAU(Transformation):
    """Split every MVAU whose size (work()) exceeds total / n_parts into up to
    ceil(size / (total / n_parts)) parts, at most k_max (see split_plan). Models without MVAUs
    are unchanged."""

    def __init__(self, n_parts=16, k_max=8):
        super().__init__()
        self.n_parts = n_parts
        self.k_max = k_max

    def apply(self, model):
        mvaus = [n for n in model.graph.node if n.op_type in MVAU_TYPES]
        if not mvaus:
            return model, False
        total = sum(work(n) for n in mvaus)
        target = total / float(self.n_parts)
        changed = False
        for node in mvaus:
            if not splittable(model, node):
                continue
            want = min(self.k_max, int(np.ceil(work(node) / target)))
            if want < 2:
                continue
            how, k = split_plan(node, want)
            if how == "out":
                self.split_out(model, node, k)
            elif how == "in":
                self.split_in(model, node, k)
            else:
                continue
            changed = True
        if changed:
            model = model.transform(InferShapes())
        return model, False

    # helpers
    def _tensor(self, model, n, shape, dt):
        model.graph.value_info.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, shape))
        model.set_tensor_datatype(n, dt)

    def _part(self, model, node, name, ins, out, attrs):
        sub = helper.make_node(node.op_type, ins, [out], name=name, domain=node.domain)
        sub.attribute.extend([a for a in node.attribute])
        si = getCustomOp(sub)
        for k, v in attrs.items():
            si.set_nodeattr(k, v)
        si.set_nodeattr("inFIFODepths", [2])
        si.set_nodeattr("outFIFODepths", [2])
        return sub

    def _replace(self, model, node, new):
        g = model.graph
        idx = list(g.node).index(node)
        g.node.remove(node)
        for j, n in enumerate(new):
            g.node.insert(idx + j, n)

    def split_out(self, model, node, k):
        inst = getCustomOp(node)
        mw, mh, pe, simd = (inst.get_nodeattr(a) for a in ("MW", "MH", "PE", "SIMD"))
        nf = mh // pe
        vecs = list(inst.get_nodeattr("numInputVectors"))
        idt, odt = inst.get_input_datatype(), inst.get_output_datatype()
        x, y, name = node.input[0], node.output[0], node.name
        W = model.get_initializer(node.input[1])
        T = model.get_initializer(node.input[2]) if len(node.input) > 2 else None
        new = []
        xs = ["%s_dup_out%d" % (name, i) for i in range(k)]
        for t in xs:
            self._tensor(model, t, vecs + [mw], idt)
        new.append(helper.make_node(
            "DuplicateStreams", [x], xs, name=name + "_dup", domain=FPGADATAFLOW, backend="fpgadataflow",
            NumChannels=mw, PE=simd, NumOutputStreams=k, inputDataType=idt.name, numInputVectors=vecs,
            inFIFODepths=list(inst.get_nodeattr("inFIFODepths")), outFIFODepths=[2] * k))
        q = pe // k
        ys = []
        for i in range(k):
            cols = [f * pe + i * q + j for f in range(nf) for j in range(q)]
            w = "%s_W%d" % (name, i)
            model.set_initializer(w, W[:, cols].copy())
            model.set_tensor_datatype(w, model.get_tensor_datatype(node.input[1]))
            ins = [xs[i], w]
            if T is not None:
                t = "%s_T%d" % (name, i)
                model.set_initializer(t, T.copy() if T.shape[0] == 1 else T[cols].copy())
                model.set_tensor_datatype(t, model.get_tensor_datatype(node.input[2]))
                ins.append(t)
            yi = "%s_part%d_out" % (name, i)
            self._tensor(model, yi, vecs + [mh // k], odt)
            new.append(self._part(model, node, "%s_part%d" % (name, i), ins, yi, {"MH": mh // k, "PE": q}))
            ys.append(yi)
        new.append(helper.make_node(
            "StreamingLaneMerge", ys, [y], name=name + "_merge", domain=FPGADATAFLOW, backend="fpgadataflow",
            NumChannels=mh, Lanes=pe, NumParts=k, dataType=odt.name, numInputVectors=vecs,
            inFIFODepths=[2] * k, outFIFODepths=list(inst.get_nodeattr("outFIFODepths"))))
        self._replace(model, node, new)

    def split_in(self, model, node, k):
        inst = getCustomOp(node)
        mw, mh, pe, simd = (inst.get_nodeattr(a) for a in ("MW", "MH", "PE", "SIMD"))
        sf = mw // simd
        vecs = list(inst.get_nodeattr("numInputVectors"))
        idt, odt = inst.get_input_datatype(), inst.get_output_datatype()
        act = inst.get_nodeattr("noActivation") == 0
        x, y, name = node.input[0], node.output[0], node.name
        W = model.get_initializer(node.input[1])
        T = model.get_initializer(node.input[2]) if act else None
        adt = acc_datatype(W, idt) if act else odt
        new = []
        xs = ["%s_lanes_out%d" % (name, i) for i in range(k)]
        for t in xs:
            self._tensor(model, t, vecs + [mw // k], idt)
        new.append(helper.make_node(
            "StreamingLaneSplit", [x], xs, name=name + "_lanes", domain=FPGADATAFLOW, backend="fpgadataflow",
            NumChannels=mw, Lanes=simd, NumParts=k, dataType=idt.name, numInputVectors=vecs,
            inFIFODepths=list(inst.get_nodeattr("inFIFODepths")), outFIFODepths=[2] * k))
        q = simd // k
        parts = []
        for i in range(k):
            rows = [f * simd + i * q + j for f in range(sf) for j in range(q)]
            w = "%s_W%d" % (name, i)
            model.set_initializer(w, W[rows, :].copy())
            model.set_tensor_datatype(w, model.get_tensor_datatype(node.input[1]))
            pi = "%s_part%d_out" % (name, i)
            self._tensor(model, pi, vecs + [mh], adt)
            new.append(self._part(
                model, node, "%s_part%d" % (name, i), [xs[i], w], pi,
                {"MW": mw // k, "SIMD": q, "noActivation": 1, "outputDataType": adt.name}))
            parts.append(pi)
        # balanced tree of two-input adders, all in the accumulator type
        level, n_add = parts, 0
        while len(level) > 1:
            nxt = []
            for a in range(0, len(level) - 1, 2):
                s = "%s_sum%d" % (name, n_add)
                last = len(level) == 2 and not act
                out = y if last else s
                if not last:
                    self._tensor(model, s, vecs + [mh], adt)
                new.append(helper.make_node(
                    "ElementwiseAdd", [level[a], level[a + 1]], [out], name="%s_add%d" % (name, n_add),
                    domain=FPGADATAFLOW, backend="fpgadataflow", lhs_dtype=adt.name, rhs_dtype=adt.name,
                    out_dtype=adt.name, lhs_shape=vecs + [mh], rhs_shape=vecs + [mh], out_shape=vecs + [mh],
                    lhs_style="input", rhs_style="input", PE=pe, inFIFODepths=[2, 2],
                    outFIFODepths=list(inst.get_nodeattr("outFIFODepths")) if last else [2]))
                nxt.append(out)
                n_add += 1
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        if act:
            t = "%s_T" % name
            model.set_initializer(t, T.copy())
            model.set_tensor_datatype(t, model.get_tensor_datatype(node.input[2]))
            new.append(helper.make_node(
                "Thresholding", [level[0], t], [y], name=name + "_thr", domain=FPGADATAFLOW, backend="fpgadataflow",
                NumChannels=mh, PE=pe, numSteps=int(T.shape[1]), inputDataType=adt.name,
                weightDataType=model.get_tensor_datatype(node.input[2]).name, outputDataType=odt.name,
                ActVal=inst.get_nodeattr("ActVal"), numInputVectors=vecs, inFIFODepths=[2],
                outFIFODepths=list(inst.get_nodeattr("outFIFODepths"))))
        self._replace(model, node, new)
