"""Per op-type totals of node_est.py outputs: python est_by_op.py est_*.json"""
import collections, json, sys
for f in sys.argv[1:]:
    rows = json.load(open(f))
    tot = collections.defaultdict(collections.Counter)
    for r in rows:
        for k, v in r["est"].items():
            tot[r["op"]][k] += v
        tot[r["op"]]["n"] += 1
    print(f)
    for op, c in sorted(tot.items(), key=lambda x: -x[1]["LUT"]):
        print(f"   {op:34s} n={c['n']:3d} LUT {c['LUT']:9.0f} BRAM18 {c['BRAM_18K']:5.0f} URAM {c['URAM']:3.0f} DSP {c['DSP']:4.0f}")
