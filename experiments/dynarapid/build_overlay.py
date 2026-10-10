"""Build (or reuse) the static tile overlay of a cached shell, outside a model build.

    FINN_RWI_OVERLAY="cols=16,rows=60" python build_overlay.py --shell <shell dir> --part xcu55c-fsvh2892-2L-e
"""

import argparse
import json
import os
import time

from finn.util.rwislands import overlay as ov
from finn.util.rwislands.device import load_device
from finn.util.rwislands.flow import island_region, shell_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    args.shell = os.path.abspath(args.shell)
    dev = load_device(args.part)
    region, first = island_region(args.part, dev)
    anchor = shell_anchor(args.part, dev)
    t0 = time.time()
    odir, info = ov.build_overlay(args.shell, args.part, dev, region + list(first), anchor,
                                  os.path.join(os.path.dirname(args.shell), "overlays"), jobs=args.jobs)
    print(json.dumps({"dir": odir, "status": info.get("status"), "s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
