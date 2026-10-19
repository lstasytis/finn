import pickle, sys
import numpy as np
log = pickle.load(open(sys.argv[1], "rb"))
for e in log:
    if not e["causal"] or e["peak"] < 8:
        continue
    w = np.sort(np.asarray(e["w"])); r = np.sort(np.asarray(e["r"])); pf = e["pf"]; gp = e["gp"]
    n = min(len(w), len(r))
    w, r = w[:n], r[:n]
    # consecutive write gaps over both frames
    gaps = np.diff(w)
    k = int(np.argmax(gaps))
    reads_in_gap = int(np.searchsorted(r, w[k + 1], "left") - np.searchsorted(r, w[k], "right"))
    # reads between write k and write k+1 the consumer wants
    # the consumer's rate: tokens per cycle
    span_r = r[n - 1] - r[n - pf]
    print(f"{e['cons']:34s} peak={e['peak']:4d} depth={e['depth']:4d} allow={e['allow']:6.1f} floor={e['floor']:4d} "
          f"t_up={e['t_up']} pf={pf} maxgap={gaps[k]} at tok {k} reads_in_gap={reads_in_gap} "
          f"w-span f1={w[pf-1]-w[0]} f2={w[n-1]-w[n-pf]} r-span f2={span_r} w1end->w2start={w[pf]-w[pf-1]} r1end->r2start={r[pf]-r[pf-1]}")
