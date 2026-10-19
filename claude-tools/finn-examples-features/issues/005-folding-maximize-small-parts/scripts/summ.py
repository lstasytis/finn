import json, glob, os, re, sys
tags = sys.argv[1:]
S = os.path.dirname(os.path.abspath(__file__))
rows = {}
for t in tags:
    for f in sorted(glob.glob(f"{S}/{t}/*.log")):
        k = os.path.basename(f)[:-4]
        txt = open(f).read()
        m = re.search(r"COMMITTED view: cycles (\d+) res (\{[^}]*\}) fits (\w+)", txt)
        com = (int(m.group(1)), eval(m.group(2)), m.group(3)) if m else None
        j = f[:-4] + ".json"
        r = json.load(open(j)) if os.path.exists(j) else None
        rows.setdefault(k, {})[t] = (com, r, re.search(r"time ([\d.]+)", txt))
for k, d in rows.items():
    com = next(v[0] for v in d.values() if v[0])
    print(f"{k:32s} committed {com[0]:>8} cyc LUT {com[1]['LUT']:>8.0f} BRAM {com[1]['BRAM_18K']:>5.0f} URAM {com[1]['URAM']:>4.0f} fits {com[2]}")
    for t, (c, r, tm) in d.items():
        if r is None:
            print(f"    {t:8s} (no result)"); continue
        rep = r["report"]; res = rep["resources_as_built"]
        print(f"    {t:8s} {rep['achieved_cycles_per_frame']:>8} cyc ratio {com[0]/rep['achieved_cycles_per_frame']:6.3f} fits {rep['fits_budget']!s:5} LUT {res['LUT']:>8.0f} BRAM {res['BRAM_18K']:>5.0f} URAM {res['URAM']:>4.0f} DSP {res['DSP']:>4.0f} probes {len(r['probes'])} t {tm.group(1) if tm else '?'}s")
