# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locating and invoking DynaRapid (Java) and Vivado for the DynaRapid P&R flow."""

import os
import shutil
import subprocess
import time

# DynaRapid short part names for the FPGA parts it has maps for
PART_TO_DYNARAPID = {
    "xck26-sfvc784-2LV-c": "xck26",
    "xczu3eg-sfvc784-1-e": "xczu3eg",
    "xcvu13p-fsga2577-1-i": "xcvu13p",
    # other parts are passed by their full name
    "xcu250-figd2104-2L-e": "xcu250-figd2104-2L-e",
}

# Map region (starti, startj, endi, endj) in which library pblocks are generated.
# RapidWright relocates the modules afterwards, but the region must stay well clear
# of the device edges: close to an edge Vivado routes with edge-specific long wires
# (e.g. WW12 near the PS boundary) that break (antennas) once the module is moved.
# Large components (e.g. MVAUs with weights in LUTRAM) need most of the device height;
# the legacy pin-exposing flow additionally needs free rows around the pblock.
PBLOCK_REGION = {
    "xck26": "20,4,219,40",
    "xczu3eg": "20,4,219,40",
    "xcvu13p": "260,20,459,127",  # inside one SLR (240 map rows each)
    "xcu250-figd2104-2L-e": "260,20,459,127",  # same die as xcvu13p
}


def dynarapid_root():
    root = os.environ.get("DYNARAPID_ROOT")
    if root is None:
        root = os.path.join(os.environ["FINN_ROOT"], "deps", "DynaRapid")
    assert os.path.isfile(os.path.join(root, "dynarapid_setup.sh")), (
        "DynaRapid not found at %s, run fetch-repos.sh" % root
    )
    return root


def java_bin():
    java_home = os.environ.get("JAVA_HOME")
    if java_home and os.path.isfile(os.path.join(java_home, "bin", "java")):
        return os.path.join(java_home, "bin", "java")
    java = shutil.which("java")
    assert java is not None, "java not found, DynaRapid needs a JDK (11)"
    return java


def classpath():
    root = dynarapid_root()
    rw = os.path.join(root, "RapidWright")
    cp = [
        os.path.join(root, "build", "classes", "java", "main"),
        os.path.join(rw, "bin"),
        os.path.join(rw, "jars", "*"),
    ]
    assert os.path.isdir(cp[0]), "DynaRapid is not built, run ./gradlew compileJava in %s" % root
    return ":".join(cp)


def dynarapid_env(work_dir, library_dir, part, clk_ns, vivado_threads=1):
    """Environment for DynaRapid processes sharing one work dir and one library."""
    env = dict(os.environ)
    short = PART_TO_DYNARAPID[part]
    env.update(
        {
            "RAPIDWRIGHT_PATH": os.path.join(dynarapid_root(), "RapidWright"),
            "DYNARAPID_WORK_DIR": work_dir,
            "DYNARAPID_LIBRARY_DIR": library_dir,
            "DYNARAPID_PBLOCK_REGION": PBLOCK_REGION[short],
            "DYNARAPID_VIVADO_THREADS": str(vivado_threads),
            "DYNARAPID_CLK_PERIOD": "%.3f" % clk_ns,
            "RW_QUIET_MESSAGE": "1",
        }
    )
    return env


def run_java(main_class, args, env, log_file, heap="8G", cwd=None):
    """Run a DynaRapid entry point, return (returncode, seconds)."""
    cmd = [java_bin(), "-Xmx" + heap, "-cp", classpath(), main_class] + list(args)
    t0 = time.time()
    with open(log_file, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        ret = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)
    return ret.returncode, time.time() - t0


def run_vivado(tcl_file, log_file, cwd, env=None):
    """Run a Vivado batch script, return (returncode, seconds)."""
    cmd = ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-notrace", "-source", tcl_file]
    t0 = time.time()
    with open(log_file, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    return ret.returncode, time.time() - t0
