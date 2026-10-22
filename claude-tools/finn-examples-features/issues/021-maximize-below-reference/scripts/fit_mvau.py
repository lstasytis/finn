"""Least-squares fit of an HLS MVAU LUT model on mvau.tsv (mvau_data.py).
    python fit_mvau.py mvau.tsv [exclude-regex]
Model (compute LUTs, excluding the decoupled weight stream's LUTRAM/SRL):
  LUT = k0 + P * (kpe + kx*Q*xnor + km*Q*W*A*(1-xnor) + ka*Q*(W+A)*(1-xnor) + kc*acc*(1+nthr))"""
import csv
import re
import sys

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
excl = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None


def feats(r):
    P, Q, W, A = (float(r[k]) for k in ("P", "Q", "W", "A"))
    x = float(r["xnor"])
    acc, B = float(r["acc"]), float(r["B"])
    nthr = 0 if r["noact"] == "1" else 2**B - 1
    return [1.0, P, P * Q * x, P * Q * W * A * (1 - x), P * Q * (W + A) * (1 - x),
            P * acc * (1 + nthr)]


names = ["k0", "kpe", "kx", "km", "ka", "kc"]
X, y, keep = [], [], []
for r in rows:
    if excl and excl.search(r["build"] + ":" + r["name"]):
        continue
    meas = float(r["meas_lut"])
    if r["mm"] == "dec":
        meas -= float(r["meas_lutram"]) + float(r["meas_srl"])
    X.append(feats(r))
    y.append(meas)
    keep.append(r)
X, y = np.array(X), np.array(y)
# relative error weighting: big nodes dominate the budget, small ones should not be ignored
w = 1.0 / np.sqrt(np.maximum(y, 100))
coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
print("  ".join(f"{n}={c:.3f}" for n, c in zip(names, coef)))
pred = X @ coef
tot = {}
for r, p, m in zip(keep, pred, y):
    e = float(r["est_lut"])
    tot.setdefault(r["build"], [0, 0, 0])
    tot[r["build"]][0] += m
    tot[r["build"]][1] += p
    tot[r["build"]][2] += e
    print(f"{r['build']:20s} {r['name']:11s} P{r['P']:>3} Q{r['Q']:>4} W{r['W']} A{r['A']} x{r['xnor']} "
          f"wmem {r['wmem']:>5} {r['mm']} meas {m:7.0f} fit {p:7.0f} ({p / max(m, 1):5.2f}) "
          f"finn {e:7.0f} ({e / max(m, 1):5.2f})")
print("per build (MVAU_hls compute LUT): meas, fit, finn")
for b, (m, p, e) in tot.items():
    print(f"  {b:20s} {m:8.0f} {p:8.0f} ({p / m:4.2f}) {e:8.0f} ({e / m:4.2f})")
