"""Post-process an existing component library built before the clock was unrouted.

For every placed-and-routed component: unroute the clock net (its out-of-context
clock-tree PIPs restrict relocation to the same position within a clock region),
rewrite checkpoint, EDIF and RapidWright metadata, and regenerate the placement
database. New libraries do not need this (GenerateFastPblocks unroutes the clock).

Usage: python unroute_library_clock.py --library <dir>/lib --part <part> [--jobs 10]
"""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

from finn.util.dynarapid.tools import (
    PART_TO_DYNARAPID,
    dynarapid_env,
    dynarapid_root,
    run_java,
    run_vivado,
)


def fix(dcp):
    base = dcp[: -len(".dcp")]
    d = os.path.dirname(dcp)
    rw = os.path.join(dynarapid_root(), "RapidWright", "tcl", "rapidwright.tcl")
    tcl = base + "_unroute_clk.tcl"
    with open(tcl, "w") as f:
        f.write(
            "open_checkpoint %s\n"
            "route_design -unroute -nets [get_nets -of [get_ports clk]]\n"
            "write_checkpoint -force %s\n"
            "write_edif -force %s.edf\n"
            "source %s\n"
            "generate_metadata %s %s/ 0\n" % (dcp, dcp, base, rw, dcp, d)
        )
    rc, t = run_vivado(tcl, base + "_unroute_clk.log", d)
    return rc, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    lib = os.path.abspath(args.library)
    work = os.path.join(os.path.dirname(lib), "work")
    dcps = sorted(glob.glob(os.path.join(lib, "*", "*_placedRouted.dcp")))
    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(fix, dcps))
    print(
        "unrouted clock in %d checkpoints, failures: %d" % (len(res), sum(r[0] != 0 for r in res))
    )
    env = dynarapid_env(work, lib, args.part, 5.0)
    comps = sorted({os.path.basename(os.path.dirname(d)) for d in dcps})

    def db(c):
        return run_java(
            "ch.agsl.dynarapid.entry.GenerateDatabase",
            ["-part", PART_TO_DYNARAPID[args.part], "-m", c],
            env,
            os.path.join(work, "components", c, "database_noclk.log"),
        )

    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(db, comps))
    print("regenerated %d databases, failures: %d" % (len(res), sum(r[0] != 0 for r in res)))


if __name__ == "__main__":
    main()
