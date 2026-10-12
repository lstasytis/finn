"""Align a depthwise sliding window generator (SWG) feeding a Pool node with the pool's PE.

A depthwise SWG emits (C/SIMD, K, SIMD); Pool_hls (finn-hlslib Pool_batch) accumulates K
consecutive words of PE channels. With SWG SIMD != Pool PE and a width converter in between (as
FINN's InsertDWC leaves it), the pool sums the wrong elements: ResNet50 (folding_pe_div2, U55C):
global average pool SWG SIMD = 2048 (12288-bit streams, ~50k LUTs incl. 39k LUTRAM in the tail) ->
DWC -> Pool PE = 2. Here the SWG gets SIMD = PE; the converters / FIFOs between are re-sized,
equal-width converters removed. The SWG then needs 49 * 2048 / 2 cycles, far below the model's
bottleneck.

    python fix_pool_swg_simd.py <in.onnx> <out.onnx>
"""

import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

PASS = ("StreamingFIFO_rtl", "StreamingFIFO", "StreamingDataWidthConverter_rtl", "StreamingDataWidthConverter")
CODEGEN = ("code_gen_dir_ipgen", "ipgen_path", "ip_path", "ip_vlnv", "code_gen_dir_cppsim", "executable_path")


def chain_to_pool(model, swg):
    """Nodes from the SWG to a Pool consumer through FIFOs / DWCs, or None."""
    chain, n = [], swg
    while True:
        cs = model.find_consumers(n.output[0]) or []
        if len(cs) != 1:
            return None
        n = cs[0]
        if n.op_type.startswith("Pool"):
            return chain, n
        if n.op_type not in PASS:
            return None
        chain.append(n)


def clear_codegen(node):
    inst = getCustomOp(node)
    for a in CODEGEN:
        if a in inst.get_nodeattr_types():
            inst.set_nodeattr(a, "")


def upstream_dwc(model, n):
    n = model.find_producer(n.input[0])
    while n is not None and n.op_type in PASS:
        if "DataWidthConverter" in n.op_type:
            return n
        n = model.find_producer(n.input[0])
    return None


def downstream_dwc(model, n):
    cs = model.find_consumers(n.output[0]) or []
    while len(cs) == 1 and cs[0].op_type in PASS:
        if "DataWidthConverter" in cs[0].op_type:
            return cs[0]
        cs = model.find_consumers(cs[0].output[0]) or []
    return None


def resize_chain(model, chain, folded, w, changed, upstream):
    """FIFOs of the chain get the folded shape; the first converter met gets width w on the side of
    the changed node (removed if both widths are then equal)."""
    for n in chain:
        i = getCustomOp(n)
        if "DataWidthConverter" in n.op_type:
            side, other = ("outWidth", "inWidth") if upstream else ("inWidth", "outWidth")
            if i.get_nodeattr(other) == w:
                if upstream:
                    for c in model.find_consumers(n.output[0]):
                        c.input[0] = n.input[0]
                else:
                    succ = model.find_consumers(n.output[0])[0]
                    succ.input[0] = n.input[0]
                model.graph.node.remove(n)
                changed.append("  %s removed" % n.name)
            else:
                i.set_nodeattr(side, w)
                clear_codegen(n)
                changed.append("  %s %s -> %d" % (n.name, side, w))
            return
        i.set_nodeattr("folded_shape", list(folded))
        clear_codegen(n)
        changed.append("  %s folded shape -> %s" % (n.name, tuple(folded)))


def lower_simd(model, swg, chain, pe, changed):
    s = getCustomOp(swg)
    changed.append("%s SIMD %d -> %d" % (swg.name, s.get_nodeattr("SIMD"), pe))
    s.set_nodeattr("SIMD", pe)
    clear_codegen(swg)
    up, n = [], model.find_producer(swg.input[0])
    while n is not None and n.op_type in PASS:
        up.append(n)
        if "DataWidthConverter" in n.op_type:
            break
        n = model.find_producer(n.input[0])
    resize_chain(model, up, s.get_folded_input_shape(), s.get_instream_width(), changed, upstream=True)
    resize_chain(model, chain, s.get_folded_output_shape(), s.get_outstream_width(), changed, upstream=False)


def raise_pe(model, swg, chain, pool, simd, changed):
    p = getCustomOp(pool)
    changed.append("%s PE %d -> %d" % (pool.name, p.get_nodeattr("PE"), simd))
    p.set_nodeattr("PE", simd)
    clear_codegen(pool)
    # converters between the SWG and the pool now see equal widths (from the pool's side)
    resize_chain(model, list(reversed(chain)), p.get_folded_input_shape(), p.get_instream_width(), changed, upstream=True)
    down, cs = [], model.find_consumers(pool.output[0]) or []
    while len(cs) == 1 and cs[0].op_type in PASS:
        down.append(cs[0])
        if "DataWidthConverter" in cs[0].op_type:
            break
        cs = model.find_consumers(cs[0].output[0]) or []
    resize_chain(model, down, p.get_folded_output_shape(), p.get_outstream_width(), changed, upstream=False)


def fix(model):
    changed = []
    for swg in list(model.graph.node):
        if not swg.op_type.startswith("ConvolutionInputGenerator"):
            continue
        s = getCustomOp(swg)
        if not s.get_nodeattr("depthwise"):
            continue
        r = chain_to_pool(model, swg)
        if r is None:
            continue
        chain, pool = r
        p = getCustomOp(pool)
        pe = p.get_nodeattr("PE")
        simd = s.get_nodeattr("SIMD")
        if simd == pe:
            continue
        if upstream_dwc(model, swg) is not None:
            lower_simd(model, swg, chain, pe, changed)
        elif simd > pe and downstream_dwc(model, pool) is not None:
            raise_pe(model, swg, chain, pool, simd, changed)
        else:
            changed.append("%s: SIMD %d != %s PE %d, no converter to absorb it (not fixed)" % (swg.name, simd, pool.name, pe))
    return model, changed


if __name__ == "__main__":
    m = ModelWrapper(sys.argv[1])
    m, changed = fix(m)
    print("\n".join(changed) or "nothing to fix")
    m.save(sys.argv[2])
