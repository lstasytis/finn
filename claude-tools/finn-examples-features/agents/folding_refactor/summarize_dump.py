#!/usr/bin/env python3
"""Table of a dump directory of dump_folding.py: result (or error) and runtime per model x mode."""
import collections
import glob
import json
import os
import sys

d = sys.argv[1]
timing = {}
if os.path.isfile(os.path.join(d, "_timing.json")):
    timing = json.load(open(os.path.join(d, "_timing.json")))
rows = collections.defaultdict(dict)
modes = []
for f in sorted(glob.glob(os.path.join(d, "*__*__*.json"))):
    key = os.path.basename(f)[:-5]
    model, board, mode = key.split("__")
    if mode not in modes:
        modes.append(mode)
    dump = json.load(open(f))
    r = dump["result"]
    if r == "ok":
        t = timing.get(key)
        cyc = dump["performance"]["max_cycles"]
        r = "%d cyc %s" % (cyc, "%.0fs" % t if isinstance(t, float) else "")
    elif r == "error":
        r = "ERR " + dump["error"][:28]
    rows["%s@%s" % (model, board)][mode] = r
print("| model@board | " + " | ".join(modes) + " |")
print("|---" * (len(modes) + 1) + "|")
for k, v in sorted(rows.items()):
    print("| %s | %s |" % (k, " | ".join(v.get(m, "-") for m in modes)))
