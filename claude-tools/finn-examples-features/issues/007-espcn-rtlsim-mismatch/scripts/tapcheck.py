# Compare FIFO tap logs (cycle hexword per line, written by the diagnostic fifo_template.v)
# against the Python reference stream of the compute node feeding the FIFO (issue 007).
# usage: tapcheck.py <stitched model.onnx> <odir> [fifo names...]   (default: all *.tap in odir)
import glob
import numpy as np
import os
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.util.data_packing import npy_to_rtlsim_input

S = os.path.dirname(os.path.abspath(__file__))
E = "/home/lstasytis/finn/build/finn_build/fx/fifo/est_espcn_KV260_SOM_uu6ftob8"
m = ModelWrapper(sys.argv[1])
odir = sys.argv[2]
fifos = sys.argv[3:] or sorted(
    os.path.basename(f)[:-4] for f in glob.glob(odir + "/*.tap")
)
g = ModelWrapper(E + "/intermediate_models/step_convert_to_hw.onnx")
ctx = np.load(S + "/ctx_hw.npz")
gnodes = [n for n in g.graph.node if n.op_type != "Transpose"]
skip = ("StreamingFIFO", "StreamingDataWidthConverter")
names = [n.name for n in m.graph.node]
compute = [n.name for n in m.graph.node if not n.op_type.startswith(skip)]


def ref_out(spec_name):
    k = compute.index(spec_name)
    if spec_name.startswith("InnerShuffle"):
        return ctx[gnodes[k - 1].output[0]].transpose(0, 1, 3, 2)
    if spec_name.startswith("OuterShuffle"):
        k -= 1
    return ctx[gnodes[k].output[0]]


for fname in fifos:
    i = names.index(fname)
    j = i - 1
    while j >= 0 and m.graph.node[j].op_type.startswith(skip):
        j -= 1
    if j < 0:
        print("%s: no compute node in front (graph input), skipped" % fname)
        continue
    src = m.graph.node[j].name
    fifo = getCustomOp(m.graph.node[i])
    width = fifo.get_instream_width_padded()
    dt = m.get_tensor_datatype(m.graph.node[i].output[0])
    ref = ref_out(src)
    # stream element width = the FIFO's unpadded word; pack the reference's last axis into it
    ew = fifo.get_instream_width()
    nel = ew // dt.bitwidth()
    flat = ref.reshape(-1)
    assert flat.size % nel == 0, (fname, flat.size, nel)
    packed = npy_to_rtlsim_input(flat.reshape(1, -1, nel), dt, width)
    tap = odir + "/" + fname + ".tap"
    cyc, got, isx = [], [], []
    with open(tap) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                cyc.append(int(p[0]))
                x = "x" in p[1].lower()
                isx.append(x)
                got.append(None if x else int(p[1], 16))
    n = min(len(got), len(packed))
    bad = np.array([isx[q] or got[q] != packed[q] for q in range(n)], dtype=bool)
    nbad = int(bad.sum())
    nx = sum(isx[:n])
    msg = "%s <- %s: width %d (%s x%d) tap %d/%d words, mismatches %d (X words %d)" % (
        fname, src, width, dt.name, nel, len(got), len(packed), nbad, nx)
    if nbad:
        k = int(np.argmax(bad))
        msg += "; first bad word %d (cycle %d): got %s exp %x" % (
            k, cyc[k], "X" if isx[k] else "%x" % got[k], packed[k])
        if nx:
            kx = isx.index(True)
            msg += "; first X word %d (cycle %d)" % (kx, cyc[kx])
        zeros = sum(1 for q in np.where(bad)[0] if got[q] == 0)
        msg += "; bad words that are 0: %d" % zeros
    print(msg)
