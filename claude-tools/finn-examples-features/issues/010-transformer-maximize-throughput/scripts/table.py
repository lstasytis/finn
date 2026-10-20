"""Markdown table from nodesim.py RESULT lines, one row per node config. table.py results.txt"""
import json
import sys

rows = []
for line in open(sys.argv[1]):
    if not line.startswith("RESULT"):
        continue
    d = json.loads(line[7:])
    rows.append(d)
print("| node | estimate | rtlsim (cycles/frame) | ratio | first-frame latency |")
print("|---|---|---|---|---|")
for d in rows:
    print("| %s | %d | %.0f | %.3f | %d |" % (d["first"], d["max_exp"], d["rtlsim"], d["ratio"], d["latency"]))
