# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locating and invoking DynaRapid (Java) and Vivado for the DynaRapid P&R flow."""

import fcntl
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager

# measured peak resident memory on the xczu7ev (Vivado 2023.1): component place and route
# 4.6 GB, DynaRapid JVM with the device model 2.8 GB
VIVADO_GB = 4.5
JVM_GB = 3.0


def vivado_version():
    """Vivado release of the active toolchain (e.g. "2024.2"), part of all cache keys:
    checkpoints and HLS output of one release are not reused by another."""
    return os.path.basename(os.path.normpath(os.environ.get("XILINX_VIVADO", ""))) or "unknown"


def avail_memory_gb():
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30


def vivado_slots():
    """(lock directory, number of slots) of the machine-wide limit on concurrent Vivado runs
    (DYNARAPID_VIVADO_SLOTS overrides the number; default from cores and free memory)."""
    d = os.path.join(os.environ.get("FINN_BUILD_DIR", "/tmp"), "dynarapid_vivado_slots")
    n = os.environ.get("DYNARAPID_VIVADO_SLOTS")
    if n is None:
        n = min(os.cpu_count() or 1, int(0.85 * avail_memory_gb() * 0.6 / VIVADO_GB))
    return d, max(1, int(n))


@contextmanager
def vivado_slot():
    """Hold one of the machine-wide Vivado slots (lock files shared with DynaRapid)."""
    d, n = vivado_slots()
    os.makedirs(d, exist_ok=True)
    while True:
        for k in range(n):
            f = open(os.path.join(d, "slot%d" % k), "a")
            try:
                fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            try:
                yield
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(0.5 + random.random())

# DynaRapid short part names for the FPGA parts it has maps for
PART_TO_DYNARAPID = {
    "xck26-sfvc784-2LV-c": "xck26",
    "xczu3eg-sfvc784-1-e": "xczu3eg",
    "xcvu13p-fsga2577-1-i": "xcvu13p",
    # other parts are passed by their full name
    "xcu250-figd2104-2L-e": "xcu250-figd2104-2L-e",
    "xczu7ev-ffvc1156-2-e": "xczu7ev-ffvc1156-2-e",  # ZCU104
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
    # 360 x 43 map (the map columns below the PS are dropped, see MapBuilderFPGA)
    "xczu7ev-ffvc1156-2-e": "20,4,339,39",
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
    slot_dir, slots = vivado_slots()
    env.setdefault("DYNARAPID_VIVADO_SLOTS_DIR", slot_dir)
    env.setdefault("DYNARAPID_VIVADO_SLOTS", str(slots))
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
    with vivado_slot():
        with open(log_file, "w") as f:
            ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    return ret.returncode, time.time() - t0
