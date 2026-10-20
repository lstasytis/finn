"""Predicted cycles/frame of the tiled MVU incl. the input_gen stall (FIFO branch's mvau_tiled_params)."""
import math, sys, re
def pred(mw, mh, simd, pe, th, niv):
    sf, nf = mw // simd, mh // pe
    n = 1
    for v in niv: n *= v
    if th == 1:
        return n * nf * sf, 0
    fm = sf * th
    buf = 1 << int(math.ceil(math.log2(fm + 2)))
    stall = max(0, ((th - 1) * sf + 4 - th) - (buf - fm))
    return (n // th) * (nf * fm + stall), stall
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"): continue
    a = line.split()
    mw, mh, simd, pe, th = map(int, a[:5]); niv = [int(x) for x in a[5].split(",")]
    p, st = pred(mw, mh, simd, pe, th, niv)
    print(line, "-> pred", p, "stall/group", st)
