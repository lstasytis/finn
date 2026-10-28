"""Issue 024 A: diff two size_dump.py outputs. usage: size_diff.py <a.json> <b.json> [name]"""
import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ta = sum(v[2] * v[3] for v in a.values())
tb = sum(v[2] * v[3] for v in b.values())
for t, v in a.items():
    if b[t][2] != v[2]:
        print(f"  {v[0]:36s} -> {v[1]:36s} {v[2]:7d} -> {b[t][2]:7d} (w={v[3]})")
name = sys.argv[3] if len(sys.argv) > 3 else ""
print(f"{name}: {ta / 1024:.1f} -> {tb / 1024:.1f} kbit ({tb / max(ta, 1):.4f})")
