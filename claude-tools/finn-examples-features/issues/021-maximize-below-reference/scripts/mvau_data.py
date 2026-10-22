"""Dataset of HLS MVAU nodes, measured (OOC synthesis) vs estimated, from <dir>/cmp.json.
    python mvau_data.py <ooc root> > mvau.tsv"""
import glob
import json
import math
import os
import sys

BITS = {"BINARY": 1, "BIPOLAR": 1}


def bits(dt):
    if dt in BITS:
        return BITS[dt]
    return int("".join(c for c in dt if c.isdigit()) or 32)


root = sys.argv[1]
cols = ["build", "name", "P", "Q", "MW", "MH", "W", "A", "B", "acc", "xnor", "wmem", "tmem", "mm",
        "ram", "noact", "est_lut", "meas_lut", "meas_lutram", "meas_srl", "est_bram", "meas_bram"]
print("\t".join(cols))
for f in sorted(glob.glob(os.path.join(root, "*", "cmp.json"))):
    b = os.path.basename(os.path.dirname(f))
    for r in json.load(open(f))["nodes"]:
        if r["op"] != "MVAU_hls" or r["meas"] is None:
            continue
        a = r["attrs"]
        noact = a.get("noActivation", 0)
        B = 0 if noact else bits(a["outputDataType"])
        row = [b, r["name"], a["PE"], a["SIMD"], a["MW"], a["MH"], bits(a["weightDataType"]),
               bits(a["inputDataType"]), B, bits(a.get("accDataType", "INT32")),
               a.get("binaryXnorMode", 0), a.get("wmem"), a.get("tmem"), a["mem_mode"][9:12],
               a["ram_style"], noact, r["est"].get("LUT", 0), r["meas"]["LUT"],
               r["meas"]["LUTRAM"], r["meas"]["SRL"], r["est"].get("BRAM_18K", 0),
               r["meas"]["BRAM_18K"]]
        print("\t".join(str(x) for x in row))
