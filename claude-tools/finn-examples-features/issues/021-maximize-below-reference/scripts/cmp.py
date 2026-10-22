"""Measured (OOC synthesis, util_hier.rpt) vs FINN-estimated (node_est.py) resources per node.
    python cmp.py <ooc dir> [<ooc dir> ...]     (expects <dir>/util_hier.rpt, est_<dir>.json)
Prints per node and per op type; writes <dir>/cmp.json. LUT = CLB LUTs (logic + LUTRAM + SRL),
BRAM18 = RAMB18 + 2 * RAMB36."""
import collections
import json
import os
import sys


def parse_hier(path):
    """Instance -> resources of the depth-1 cells under finn_design_i."""
    rows, header = {}, None
    for line in open(path):
        if not line.startswith("|"):
            continue
        cols = [c for c in line.rstrip().split("|")[1:-1]]
        if header is None and cols and cols[0].strip() == "Instance":
            header = [c.strip() for c in cols]
            continue
        if header is None or len(cols) != len(header):
            continue
        name = cols[0]
        depth = (len(name) - len(name.lstrip())) // 2
        rec = dict(zip(header, [c.strip() for c in cols]))
        rows.setdefault("__all__", []).append((depth, name.strip(), rec))
    out = {}
    for depth, name, rec in rows.get("__all__", []):
        if depth == 0:
            out["__total__"] = rec
        elif depth == 2:  # wrapper > finn_design_i > node
            out[name] = rec
    return out


def num(rec, key):
    try:
        return float(rec.get(key, 0) or 0)
    except ValueError:
        return 0.0


def measured(rec):
    return dict(
        LUT=num(rec, "Total LUTs"),
        LUTRAM=num(rec, "LUTRAMs"),
        SRL=num(rec, "SRLs"),
        FF=num(rec, "FFs"),
        BRAM_18K=num(rec, "RAMB18") + 2 * num(rec, "RAMB36"),
        URAM=num(rec, "URAM"),
        DSP=num(rec, "DSP Blocks") or num(rec, "DSP48 Blocks"),
    )


for d in sys.argv[1:]:
    d = d.rstrip("/")
    hier = parse_hier(os.path.join(d, "util_hier.rpt"))
    est = json.load(open(os.path.join(os.path.dirname(d) or ".", os.environ.get("EST", "est") + f"_{os.path.basename(d)}.json")))
    res = []
    for r in est:
        inst = next((k for k in hier if k.split(" ")[0] in (r["name"], r["name"] + "_0")), None)
        if inst is None:  # BD cell names: <node name>
            inst = next((k for k in hier if k.startswith(r["name"] + " ")), None)
        m = measured(hier[inst]) if inst else None
        res.append(dict(name=r["name"], op=r["op"], est=r["est"], meas=m, attrs=r["attrs"]))
    tot_m = measured(hier["__total__"])
    json.dump(dict(nodes=res, total=tot_m), open(os.path.join(d, os.environ.get("EST", "est").replace("est", "cmp") + ".json"), "w"), indent=1)
    print(f"== {d}: measured total LUT {tot_m['LUT']:.0f} (LUTRAM {tot_m['LUTRAM']:.0f}) "
          f"BRAM18 {tot_m['BRAM_18K']:.0f} DSP {tot_m['DSP']:.0f}; "
          f"est LUT {sum(r['est'].get('LUT', 0) for r in res):.0f} "
          f"BRAM18 {sum(r['est'].get('BRAM_18K', 0) for r in res):.0f}")
    byop = collections.defaultdict(lambda: collections.Counter())
    for r in res:
        if r["meas"] is None:
            print("   unmatched", r["name"])
            continue
        c = byop[r["op"]]
        c["n"] += 1
        for k in ("LUT", "BRAM_18K", "DSP"):
            c["e" + k] += r["est"].get(k, 0)
            c["m" + k] += r["meas"][k]
        c["mLUTRAM"] += r["meas"]["LUTRAM"]
        if r["op"].startswith(("MVAU", "VVAU", "Thresh", "Conv")):
            a = r["attrs"]
            print(f"   {r['name']:24s} PE {a.get('PE')} SIMD {a.get('SIMD')} MW {a.get('MW')} "
                  f"MH {a.get('MH')} W {a.get('weightDataType')} I {a.get('inputDataType')} "
                  f"wmem {a.get('wmem')} {a.get('mem_mode', '')[:12]} ram {a.get('ram_style')} xnor {a.get('binaryXnorMode')} | est LUT "
                  f"{r['est'].get('LUT', 0):7.0f} BRAM {r['est'].get('BRAM_18K', 0):4.0f} | meas LUT "
                  f"{r['meas']['LUT']:7.0f} (LUTRAM {r['meas']['LUTRAM']:5.0f}) BRAM "
                  f"{r['meas']['BRAM_18K']:4.0f}")
    for op, c in sorted(byop.items(), key=lambda x: -x[1]["eLUT"]):
        print(f"   [{op:32s}] n={c['n']:3d} LUT est {c['eLUT']:8.0f} meas {c['mLUT']:7.0f} "
              f"(LUTRAM {c['mLUTRAM']:5.0f}) BRAM est {c['eBRAM_18K']:4.0f} meas {c['mBRAM_18K']:4.0f}"
              f" DSP est {c['eDSP']:4.0f} meas {c['mDSP']:4.0f}")
