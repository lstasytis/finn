"""Diff two depth JSONs written by size_fixture.py: python compare.py a.json b.json"""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
same = True
for k in sorted(set(a) | set(b)):
    da = a.get(k, {}).get("depth")
    db = b.get(k, {}).get("depth")
    if da != db:
        same = False
        e = a.get(k) or b.get(k)
        print("  %s -> %s: %s -> %s" % (e["prod"], e["cons"], da, db))
ta = sum(v["depth"] for v in a.values())
tb = sum(v["depth"] for v in b.values())
print("%s: %s (total depth %d -> %d)" % (sys.argv[2], "identical" if same else "DIFFERENT", ta, tb))
sys.exit(0 if same else 1)
