"""Compare measured RESULT lines (logs/grid*.out) with the current get_exp_cycles()."""
import glob, re, sys
from onnx import helper
from qonnx.custom_op.registry import getCustomOp
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/logs/grid*.out")):
    for line in open(f):
        if not line.startswith("RESULT"): continue
        p = line.split()
        tag = p[1]; a = tag.split("_")
        mw, mh, simd, pe, th = map(int, a[:5]); niv = [int(x) for x in a[5].split("x")]
        mem, idt, wdt = a[6], a[7], a[8]
        if mem == "external": mem = "external_mem"; idt, wdt = a[8], a[9]
        if mem == "internal": mem = "internal_decoupled"; idt, wdt = a[8], a[9]
        rtl = float(p[p.index("rtlsim") + 1])
        node = helper.make_node("MVAU_rtl", ["i", "w"], ["o"], domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow", MW=mw, MH=mh, SIMD=simd, PE=pe, TH=th, numInputVectors=niv,
            inputDataType=idt, weightDataType=wdt, outputDataType="INT32", noActivation=1, mem_mode=mem)
        inst = getCustomOp(node)
        exp = inst.get_exp_cycles(); base = int(mh/pe*mw/simd*len(niv) and mh//pe*(mw//simd)*__import__("numpy").prod(niv))
        old = base * th
        rows.append((mem, mw, mh, simd, pe, th, niv, old, base, exp, rtl, rtl/exp))
rows.sort(key=lambda r: (r[0], r[5], r[1], r[2]))
print("| mem_mode | MW | MH | SIMD | PE | TH | numInputVectors | dev est (x TH) | compute (no stall) | new est | rtlsim/frame | rtlsim/new |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
worst = 0
for r in rows:
    print("| %s | %d | %d | %d | %d | %d | %s | %d | %d | %d | %.1f | %.3f |" % r)
    worst = max(worst, abs(r[11]-1))
print("n", len(rows), "worst |ratio-1|", round(worst, 4))
