"""Issue 014: markdown table of the eval_build.py results: python eval_table.py <eval dir> <tag>"""
import glob
import json
import os
import sys

d, tag = sys.argv[1:3]
rows = {}
for f in sorted(glob.glob(os.path.join(d, f"{tag}_*_p*.json"))):
    r = json.load(open(f))
    rows.setdefault((r["model"], r["board"]), {})[r["pad"]] = r
R = ("LUT", "BRAM_18K", "URAM", "DSP")


def fmt(res):
    return " / ".join("%.0f" % res.get(r, 0) for r in R)


print("| model (board) | target cycles | unpadded: cycles, LUT / BRAM / URAM / DSP (no FIFOs) "
      "| padded: cycles, LUT / BRAM / URAM / DSP (no FIFOs) | padded nodes, pad DWCs | "
      "FIFOs incl. (unpadded -> padded) LUT / BRAM | fold s (unpadded / padded) |")
print("|---|---|---|---|---|---|---|")
for (m, b), p in sorted(rows.items()):
    a, c = p.get(0), p.get(4)
    if a is None or c is None:
        continue
    print(
        f"| {m} ({b}) | {a['ref_cycles']} | {a['cycles']}, {fmt(a['res_nofifo'])} | "
        f"{c['cycles']}, {fmt(c['res_nofifo'])} | {len(c['padded_nodes'])}, {c['pad_dwcs']} | "
        f"{a['res']['LUT']:.0f} / {a['res']['BRAM_18K']:.0f} -> {c['res']['LUT']:.0f} / "
        f"{c['res']['BRAM_18K']:.0f} | {a['fold_s'] or 0:.0f} / {c['fold_s'] or 0:.0f} |"
    )
