# Issue 012: compare an espcn_sim.py output (dataflow partition = global_out) with the golden output
# and with the failing build's verification output. usage: golden_cmp.py <out.npy> [<FAIL.npy>]
import sys
import numpy as np

gold = np.load("/home/lstasytis/finn/tests/benchmark/verification_io/espcn-bsd300_output.npy")
out = np.load(sys.argv[1])
cands = {"reshape": out.reshape(gold.shape) if out.size == gold.size else None}
if out.ndim == 4 and out.shape[-1] == gold.shape[1]:
    cands["nhwc->nchw"] = out.transpose(0, 3, 1, 2)
for k, o in cands.items():
    if o is None or o.shape != gold.shape:
        continue
    d = np.abs(o - gold)
    print(f"{k}: shape {o.shape} max|diff| {d.max():.6f} (1/255 = {1/255:.6f}) "
          f"n>1e-6 {(d > 1e-6).sum()} n>1/255+eps {(d > 1/255 + 1e-6).sum()} zeros {(o == 0).sum()} "
          f"golden zeros {(gold == 0).sum()} allclose(atol=1/255) {np.allclose(o, gold, atol=1/255 + 1e-6)}")
if len(sys.argv) > 2:
    f = np.load(sys.argv[2]).reshape(gold.shape)
    d = np.abs(f - gold)
    print(f"failing build output: max|diff| {d.max():.4f} n>1/255 {(d > 1/255 + 1e-6).sum()} zeros {(f == 0).sum()}")
