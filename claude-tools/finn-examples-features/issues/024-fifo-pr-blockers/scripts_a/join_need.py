"""Issue 024 A: structural (deadlock) buffer need at every join with >= 2 streamed inputs
(ElementwiseAdd, StreamingConcat, ...), in the style of join_window_floors but with
pixel tokens at the join and explicit sliding-window needs for FMPadding, SWG, Upsample,
Split and Concat. For join pixel p: W(p) = max_i fork words input i needs for pixels 0..p;
input i then holds (its words producible from W(p) - 1 fork words) - p * c_i words.
Compares the largest backlog with the depth of the FIFO in front of the join input in a
FIFO-inserted build model (same node names). One frame.
usage: join_need.py <model without FIFOs (TAV model)> [build model with FIFOs]"""
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
order = {n.name: i for i, n in enumerate(m.graph.node)}


def fw(inst, ind=0, out=False):
    s = inst.get_folded_output_shape(ind) if out else inst.get_folded_input_shape(ind)
    return int(np.prod(s[:-1]))


def pixels(inst, ind=0, out=False):
    s = inst.get_folded_output_shape(ind) if out else inst.get_folded_input_shape(ind)
    return int(np.prod(s[:-2]))


def need_words(node, inst, i, k):
    """input words of input i needed for the first k output words (k: array, >= 1)."""
    op = node.op_type
    o = 0
    n_out = fw(inst, o, True)
    wpo = n_out // pixels(inst, o, True)  # output words per pixel
    q = (k - 1) // wpo  # output pixel of word k
    if op.startswith("FMPadding"):
        H, W = inst.get_nodeattr("ImgDim")
        t, l, b, r = inst.get_nodeattr("Padding")
        Wo = W + l + r
        wpi = fw(inst, 0) // (H * W)
        rr, cc = q // Wo, q % Wo
        rows = np.clip(rr - t, 0, H)
        inrow = (rr >= t) & (rr < t + H)
        cols = np.where(inrow, np.clip(cc - l + 1, 0, W), 0)
        return np.maximum(1, (rows * W + cols) * wpi)
    if op.startswith("ConvolutionInputGenerator"):
        H, W = inst.get_nodeattr("IFMDim")
        OH, OW = inst.get_nodeattr("OFMDim")
        kh, kw = inst.get_nodeattr("ConvKernelDim")
        sh, sw = inst.get_nodeattr("Stride")
        dh, dw = inst.get_nodeattr("Dilation")
        wpi = fw(inst, 0) // (H * W)
        oy, ox = q // OW, q % OW
        last = np.minimum((oy * sh + (kh - 1) * dh) * W + ox * sw + (kw - 1) * dw, H * W - 1)
        return (last + 1) * wpi
    if op.startswith("UpsampleNearestNeighbour"):
        HI, WI = inst.get_nodeattr("HI"), inst.get_nodeattr("WI")
        HO, WO = inst.get_nodeattr("HO"), inst.get_nodeattr("WO")
        wpi = fw(inst, 0) // (HI * WI)
        r, c = q // WO, q % WO
        return ((r * HI // HO) * WI + c * WI // WO + 1) * wpi
    if op.startswith(("StreamingConcat", "StreamingSplit")):
        wpi = fw(inst, i) // pixels(inst, i)
        return (q + 1) * wpi
    table = inst.get_tav_input_need(i)
    if table is not None:
        frames, j = np.divmod(k - 1, len(table))
        return table[j] + frames * fw(inst, i)
    a, b = inst.get_tav_input_window(i)
    return -(-k // b) * a


def producer(t):
    return m.find_producer(t)


def ancestors(tensors):
    seen, stack = set(), [producer(t) for t in tensors]
    while stack:
        n = stack.pop()
        if n is None or n.name in seen:
            continue
        seen.add(n.name)
        stack.extend(producer(t) for t in n.input)
    return seen


depth_of = {}
if len(sys.argv) > 2:
    b = ModelWrapper(sys.argv[2])
    for n in b.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            c = b.find_consumer(n.output[0])
            p = b.find_producer(n.input[0])
            if c is not None:
                idx = list(c.input).index(n.output[0])
                depth_of[(c.name, idx)] = (getCustomOp(n).get_nodeattr("depth"), p.name if p else "-")

for join in m.graph.node:
    ins = [(i, t) for i, t in enumerate(join.input) if producer(t) is not None]
    if len(ins) < 2:
        continue
    common = set.intersection(*[ancestors([t]) for _, t in ins])
    if not common:
        continue
    fork = m.get_node_from_name(max(common, key=order.get))
    above = ancestors(fork.input)
    memo = {}

    def walk(t, mm):
        if t in memo and len(memo[t]) >= mm:
            return memo[t][:mm]
        p = producer(t)
        if p is None or p.name in above:
            return None
        inst = getCustomOp(p)
        k = np.arange(1, mm + 1, dtype=np.int64)
        o = list(p.output).index(t)
        if p.name == fork.name:
            n_out, n_in = fw(inst, o, True), fw(inst, 0)
            res = -(-k * n_in // n_out)
        else:
            res = None
            for i, x in enumerate(p.input):
                if producer(x) is None:
                    continue
                idx = need_words(p, inst, i, k)
                idx = np.clip(idx, 1, None)
                up = walk(x, int(idx.max()))
                if up is None:
                    continue
                v = up[idx - 1]
                res = v if res is None else np.maximum(res, v)
            if res is None:
                return None
        memo[t] = res
        return res

    jinst = getCustomOp(join)
    P = pixels(jinst, ins[0][0])
    rows = []
    walks = []
    for i, t in ins:
        c = fw(jinst, i) // P
        w = walk(t, fw(jinst, i))
        walks.append((i, t, c, w))
    if any(w[3] is None for w in walks):
        continue
    p = np.arange(P)
    W = np.max(np.stack([w[(p + 1) * c - 1] for _, _, c, w in walks]), axis=0)
    out = []
    for i, t, c, w in walks:
        held = np.searchsorted(w, W - 1, side="right") - p * c
        need = int(held.max())
        d, src = depth_of.get((join.name, i), ("?", "?"))
        flag = "  <<< FIFO below need" if isinstance(d, int) and d < need else ""
        out.append(f"   in{i} from {src:34s} need {need:7d} depth {d}{flag}")
    print(f"{join.name} (fork {fork.name})")
    print("\n".join(out))
