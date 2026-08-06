# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Score a replacement demand-lead law against the bar, in one command.

``swg_demand_lead(p)`` in ``convolutioninputgenerator.py`` is the one term known
to be short. This monkey-patches a candidate over it, runs the two scoring
matrices, and prints the verdict against the numbers the current law achieves.
The bar is: improve the models tier **without** regressing the 384-config gate,
the growth count, or token exactness.

    python claude-tools/swg/swg_integrate.py                      # current law
    python claude-tools/swg/swg_integrate.py --law mymod:my_lead  # a candidate
"""

import argparse
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("FINN_ROOT", os.path.abspath(os.path.join(HERE, "../..")))
sys.path.insert(0, HERE)

# the numbers the committed law reaches; a candidate has to hold all of these
BAR = {"gate_all": 309, "grow_all": 15, "tok_all": 379, "gate_models": 14, "tok_models": 19}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", default=None, help="module:function replacing swg_demand_lead")
    a = ap.parse_args()

    env = dict(os.environ)
    if a.law:
        mod, fn = a.law.split(":")
        importlib.import_module(mod)  # fail early if it does not import
        env["SWG_DEMAND_LEAD"] = a.law

    got = {}
    for matrix in ("all", "models"):
        out = subprocess.run(
            [sys.executable, os.path.join(HERE, "swg_score.py"), "--matrix", matrix],
            capture_output=True,
            text=True,
            env=env,
        ).stdout
        print(f"--- {matrix} ---")
        for line in out.splitlines():
            if any(k in line for k in ("modelled", "worst", "token", "grows", "GATE")):
                print(" ", line.strip())
                if "GATE" in line:
                    got["gate_" + matrix] = int(line.split(":")[1].split("/")[0].strip())
                if "grows" in line:
                    got["grow_" + matrix] = int(line.split("on")[1].split("/")[0].strip())
                if "token" in line:
                    got["tok_" + matrix] = int(line.split("on")[1].split("/")[0].strip())

    print("\nverdict vs the current law:")
    ok = True
    for key, want in BAR.items():
        have = got.get(key)
        if have is None:
            continue
        good = have <= want if key.startswith("grow") else have >= want
        ok &= good
        print(f"  {key:<14} {have:>5}  (bar {want})  {'ok' if good else 'REGRESSION'}")
    print("KEEP" if ok else "REVERT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
