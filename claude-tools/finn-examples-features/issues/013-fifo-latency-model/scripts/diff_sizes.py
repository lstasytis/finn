"""Diff two size_offline.py -v outputs: edges whose depth changed. usage: diff_sizes.py a b"""
import re
import sys


def load(p):
    d = {}
    for line in open(p):
        m = re.match(r"(\S+)\s+(\S+)\s+->\s+(\S+).*depth=\s*(\d+)\s+w=(\d+)", line)
        if m:
            d[m.group(1)] = (m.group(2), m.group(3), int(m.group(4)), int(m.group(5)))
    return d


a, b = load(sys.argv[1]), load(sys.argv[2])
for t in a:
    if t in b and a[t][2] != b[t][2]:
        print(f"{a[t][0]:34s} -> {a[t][1]:34s} {a[t][2]:7d} -> {b[t][2]:7d}  w={a[t][3]}")
