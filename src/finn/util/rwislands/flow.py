# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Island bitfile flow for Zynq: parallel out-of-context place and route of FINN node groups
at their final location, stitched with RapidWright, inserted into a pre-implemented shell.

    ZynqBuild partitions (idma, kernel, odma) -> one accelerator graph
      -> in parallel: shell (cached per board / clock / IODMA interfaces)
                      out-of-context synthesis of every node      (resource numbers)
      -> islands: the node chain cut into K groups, snake floorplan (islands.floorplan)
      -> in parallel: place and route of every island in its pblock (one Vivado run each)
                      synthesis of the accelerator top (islands as black boxes)
      -> RapidWright: fill the black boxes with the routed islands, route the nets between
                      islands (IslandStitcher.java, RWRoute partial routing)
      -> Vivado: shell + accelerator, route boundary/clock nets, bitstream

No component library, relocation or placement database: every island is placed and routed
once, where it ends up.
"""

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from finn.util.dynarapid.components import (
    component_name,
    direct_sources,
    direct_synth_tcl,
    synth_tcl,
    vendor_ip_cores,
)
from finn.util.dynarapid.graph import mm_ports
from finn.util.dynarapid.shell import (
    PS_BOUNDARY_INT_X,
    SHELL_STRIP_COLS,
    assemble_tcl,
    build_shell,
)
from finn.util.dynarapid.tools import java_bin, run_vivado, usable_cpus, vivado_slots
from finn.util.dynarapid.zynq import _reports, merge_partitions
from finn.util.rwislands.device import load_device
from finn.util.rwislands.floorplan import floorplan, island_cost, partition
from finn.util.rwislands.netlist import TOP_MODULE, channel_graph, island_verilog, top_verilog

HERE = os.path.dirname(os.path.abspath(__file__))

# Vivado's link_design of RTL + checkpoints reports a spurious error in 2024.2
# (Designutils 20-50) although the design is linked
_LINK = """if {[catch {link_design -top %s -part %s -mode out_of_context} err]} {
    if {[catch {current_design}] || [llength [get_cells -quiet -hier -filter {IS_BLACKBOX}]] > 0} {error $err}
    puts "INFO: link_design error ignored, design linked: $err"
}"""


def rapidwright_root():
    for d in (
        os.environ.get("RAPIDWRIGHT_PATH"),
        os.path.join(os.environ["FINN_ROOT"], "deps", "RapidWright"),
        os.path.join(os.environ["FINN_ROOT"], "deps", "DynaRapid", "RapidWright"),
    ):
        if d and os.path.isdir(os.path.join(d, "jars")):
            return d
    raise RuntimeError("RapidWright not found (set RAPIDWRIGHT_PATH)")


def stitcher_classpath():
    """Compile IslandStitcher.java (once per source version) against RapidWright."""
    rw = rapidwright_root()
    cp = "%s:%s" % (os.path.join(rw, "bin"), os.path.join(rw, "jars", "*"))
    src = os.path.join(HERE, "IslandStitcher.java")
    out = os.path.join(os.environ["FINN_BUILD_DIR"], "rwislands", "classes")
    cls = os.path.join(out, "IslandStitcher.class")
    if not os.path.isfile(cls) or os.path.getmtime(cls) < os.path.getmtime(src):
        os.makedirs(out, exist_ok=True)
        javac = os.path.join(os.path.dirname(java_bin()), "javac")
        subprocess.run([javac, "-cp", cp, "-d", out, src], check=True)
    return "%s:%s" % (out, cp)


def parse_util(f):
    """Resources of a synthesized component from report_utilization."""
    res = dict.fromkeys(("lut", "lutram", "ff", "carry", "bram", "dsp", "uram"), 0.0)
    if not os.path.isfile(f):
        return res
    txt = open(f, errors="ignore").read()
    pats = {
        "lut": r"\|\s*CLB LUTs\*?\s*\|\s*([\d.]+)",
        "lutram": r"\|\s*LUT as Memory\s*\|\s*([\d.]+)",
        "ff": r"\|\s*CLB Registers\s*\|\s*([\d.]+)",
        "carry": r"\|\s*CARRY8\s*\|\s*([\d.]+)",
        "bram": r"\|\s*Block RAM Tile\s*\|\s*([\d.]+)",
        "dsp": r"\|\s*DSPs\s*\|\s*([\d.]+)",
        "uram": r"\|\s*URAM\s*\|\s*([\d.]+)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        if m:
            res[k] = float(m.group(1))
    return res


def synthesize(accel, dcps, work, part, clk_ns, cpus, slots):
    """Out-of-context synthesis of every distinct component, in parallel. Nodes with plain HDL
    are synthesized in shared Vivado sessions (the start and device load are paid once per
    session), the others (weight streamer hierarchies) through a one-node block design.
    Returns {dcp: {"status", "synth_s", "util"}}."""
    synth_dir = os.path.join(work, "synth")
    os.makedirs(synth_dir, exist_ok=True)
    first = {}
    for n in accel.graph.node:
        first.setdefault(dcps[n.name], n)
    direct, bd = [], []
    for dcp, n in first.items():
        (direct if direct_sources(n) is not None else bd).append((dcp, n))
    # as many sessions as CPUs left over by the block-design runs
    n_sess = max(1, min(len(direct), max(1, min(cpus, slots) - len(bd))))
    sessions = [direct[k::n_sess] for k in range(n_sess)] if direct else []
    res = {}

    def run(tcl_lines, name, items):
        d = os.path.join(work, "components", name)
        os.makedirs(d, exist_ok=True)
        tcl = os.path.join(d, "synth.tcl")
        with open(tcl, "w") as f:
            f.write("\n".join(tcl_lines) + "\n")
        t0 = time.time()
        rc, t = run_vivado(tcl, os.path.join(d, "synth.log"), d)
        for dcp, _ in items:
            ok = os.path.isfile(os.path.join(synth_dir, dcp + "_synth.dcp"))
            res[dcp] = {
                "status": "ok" if ok else "synth_failed",
                "synth_s": t,
                "done_s": time.time() - t0,
                "util": parse_util(os.path.join(synth_dir, dcp + ".util")),
            }

    def bd_job(item):
        dcp, n = item
        d = os.path.join(work, "components", dcp)
        os.makedirs(d, exist_ok=True)
        body = synth_tcl(accel, n, dcp, d, synth_dir, part, clk_ns, 1, metadata=False)
        run(body.splitlines(), dcp, [item])

    def sess_job(items):
        lines = ["set_param general.maxThreads 1"]
        for dcp, n in items:
            d = os.path.join(work, "components", dcp)
            os.makedirs(d, exist_ok=True)
            files, top = direct_sources(n)
            body = direct_synth_tcl(n, dcp, d, synth_dir, part, 1, files, top, metadata=False)
            lines += [l for l in body.splitlines() if not l.startswith("set_param")]
            lines += ["close_design", "remove_files -quiet [get_files -quiet]"]
        run(lines, "session_" + items[0][0], items)

    with ThreadPoolExecutor(max_workers=max(1, min(slots, len(bd) + len(sessions)))) as ex:
        futs = [ex.submit(bd_job, it) for it in bd] + [ex.submit(sess_job, s) for s in sessions]
        for f in futs:
            f.result()
    return res


def island_tcl(name, island_v, dcp_files, part, clk_ns, ranges, cr, threads, out_dir):
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        'proc stamp {n} {global t0; puts "STAMP $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
        "read_verilog %s" % island_v,
    ]
    t += ["read_checkpoint %s" % f for f in dcp_files]
    t += [
        _LINK % (name, part),
        "stamp link",
        "create_clock -period %.3f -name clk [get_ports clk]" % clk_ns,
        # clock source of the out-of-context run (the clock routing is discarded when the
        # island is stitched; Vivado routes the real clock tree at assembly)
        "set bufg [lindex [get_sites -quiet -filter {SITE_TYPE == BUFGCE && CLOCK_REGION == %s}] 0]"
        % cr,
        'if {$bufg == ""} {set bufg [lindex [get_sites -filter {SITE_TYPE == BUFGCE}] 0]}',
        "set_property HD.CLK_SRC $bufg [get_ports clk]",
        "create_pblock pb",
        "resize_pblock pb -add {%s}" % " ".join(ranges),
        "add_cells_to_pblock pb -top",
        "set_property CONTAIN_ROUTING 1 [get_pblocks pb]",
        "opt_design",
        "stamp opt",
        "place_design",
        "stamp place",
        'if {[catch {route_design} err]} {puts "INFO: route_design error: $err"}',
        "stamp route",
        "report_route_status -file %s/route_status.rpt" % out_dir,
        "report_utilization -file %s/utilization.rpt" % out_dir,
        "write_checkpoint -force %s/%s_routed.dcp" % (out_dir, name),
        "write_edif -force %s/%s_routed.edf" % (out_dir, name),
        "stamp write",
    ]
    return "\n".join(t) + "\n"


def top_tcl(top_v, part, out_dir):
    return "\n".join(
        [
            "set_param general.maxThreads 2",
            "read_verilog %s" % top_v,
            "synth_design -top %s -part %s -mode out_of_context" % (TOP_MODULE, part),
            "write_checkpoint -force %s/top_bb.dcp" % out_dir,
            "write_edif -force %s/top_bb.edf" % out_dir,
        ]
    ) + "\n"


def choose_islands(n_nodes, total_cost, slots, islands):
    if islands not in (None, "auto"):
        return int(islands)
    # every Vivado run costs ~1 min of fixed effort on the xczu7ev: islands of at least
    # ~6000 cost units (LUT + FF/2 + ...), no more islands than parallel Vivado runs
    return max(1, min(n_nodes, slots, int(total_cost // 6000) + 1))


def rw_islands_zynq_build(
    kernel_models,
    board,
    part,
    clk_ns,
    out_dir,
    shell_lib,
    islands="auto",
    workers=None,
    rwroute_max_iter=30,
):
    """Build the bitfile of a ZynqBuild design (partition models with generated IP).
    Returns a result dict with the bitfile, hwh and per-stage times."""
    os.makedirs(out_dir, exist_ok=True)
    cpus = workers or usable_cpus()
    _, slots = vivado_slots()
    work = os.path.join(out_dir, "work")
    res = {"out_dir": out_dir, "board": board, "part": part, "clk_ns": clk_ns, "flow": "rwislands"}
    stamps = {}
    t_total = time.time()

    def stamp(k):
        stamps[k] = time.time() - t_total

    accel = merge_partitions(kernel_models)
    accel.save(os.path.join(out_dir, "accel.onnx"))
    vendor = {n.name: vendor_ip_cores(n) for n in accel.graph.node}
    vendor = {k: v for k, v in vendor.items() if v}
    if vendor:
        res.update(status="unsupported_vendor_ip", vendor_ip=vendor)
        return _done(res, stamps, out_dir, t_total)
    ports = mm_ports(accel)
    assert ports, "the island shell flow needs IODMAs at the accelerator boundary"
    g = channel_graph(accel)
    dcps = {n.name: component_name(accel, n, part, clk_ns) for n in accel.graph.node}
    dev = load_device(part)

    # 1. shell (cached) and node synthesis, concurrently
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_shell = ex.submit(
            build_shell, board, part, clk_ns, ports, shell_lib, max(1, min(8, cpus // 4))
        )
        synth = synthesize(accel, dcps, work, part, clk_ns, cpus, slots)
        stamp("synth")
        shell_dir, shell_res = f_shell.result()
    stamp("shell")
    res["shell"] = shell_res
    res["synth"] = synth
    failed = [d for d, r in synth.items() if r["status"] != "ok"]
    if failed:
        res.update(status="synth_failed", failed=failed)
        return _done(res, stamps, out_dir, t_total)
    if shell_res["status"] not in ("built", "cached"):
        res["status"] = "shell_failed"
        return _done(res, stamps, out_dir, t_total)

    # the assembly Vivado opens the shell now and waits for the stitched accelerator
    asm_dir = os.path.join(out_dir, "assembly")
    os.makedirs(asm_dir, exist_ok=True)
    bitfile = os.path.join(out_dir, "resizer.bit")
    accel_dcp = os.path.join(work, "stitch", "accel_routed.dcp")
    trigger = os.path.join(asm_dir, "accel_ready")
    if os.path.exists(trigger):
        os.remove(trigger)
    asm_tcl = os.path.join(asm_dir, "assemble.tcl")
    with open(asm_tcl, "w") as f:
        f.write(
            assemble_tcl(
                shell_dir, accel_dcp, asm_dir, bitfile, threads=min(16, cpus), reports="min",
                trigger=trigger,
                unfix_static="shell",
            )
        )
    asm_pool = ThreadPoolExecutor(max_workers=1)
    f_asm = asm_pool.submit(run_vivado, asm_tcl, os.path.join(asm_dir, "assemble.log"), asm_dir)

    def abort(status, **kw):
        with open(trigger, "w") as f:
            f.write("abort")
        f_asm.result()
        res.update(status=status, **kw)
        return _done(res, stamps, out_dir, t_total)

    # 2. islands and floorplan
    names = [n.name for n in accel.graph.node]
    node_res = {n: synth[dcps[n]]["util"] for n in names}
    costs = [island_cost(node_res[n]) for n in names]
    k = choose_islands(len(names), sum(costs), slots, islands)
    segs = partition(costs, k)
    isl = {"island_%d" % i: names[a:b] for i, (a, b) in enumerate(segs)}
    isl_res = []
    for mem in isl.values():
        tot = {}
        for n in mem:
            for kk, v in node_res[n].items():
                tot[kk] = tot.get(kk, 0) + v
        isl_res.append(tot)
    x0 = PS_BOUNDARY_INT_X[part] + SHELL_STRIP_COLS
    region = (x0, dev.xmax, 0, dev.ymax)
    try:
        rects, ranges, util, lanes = floorplan(dev, isl_res, region)
    except RuntimeError as e:
        return abort("floorplan_failed", error=str(e))
    stamp("floorplan")
    res["islands"] = {
        name: {"nodes": mem, "res": r, "cost": sum(costs[names.index(n)] for n in mem), "rects": rc}
        for (name, mem), r, rc in zip(isl.items(), isl_res, rects)
    }
    res["floorplan"] = {"util": util, "lanes": lanes, "region": region}

    # 3. island place and route + accelerator top, in parallel
    isl_dir = os.path.join(work, "islands")
    threads = max(1, min(8, cpus // max(1, len(isl))))
    synth_dir = os.path.join(work, "synth")

    def island_job(name):
        mem = isl[name]
        d = os.path.join(isl_dir, name)
        os.makedirs(d, exist_ok=True)
        v = os.path.join(d, name + ".v")
        with open(v, "w") as f:
            f.write(island_verilog(name, g, mem, dcps))
        files = [os.path.join(synth_dir, dc + "_synth.dcp") for dc in sorted({dcps[n] for n in mem})]
        r0 = res["islands"][name]["rects"][0]
        cr = next(s.cr for s in dev.sites_in(*r0) if s.type.startswith("SLICE"))
        tcl = os.path.join(d, "island.tcl")
        with open(tcl, "w") as f:
            f.write(island_tcl(name, v, files, part, clk_ns, ranges[list(isl).index(name)], cr, threads, d))
        rc, t = run_vivado(tcl, os.path.join(d, "island.log"), d)
        info = {"pnr_s": t, "rc": rc}
        info.update(_island_reports(d))
        info["stamps"] = {
            kk: float(vv)
            for kk, vv in re.findall(r"^STAMP (\w+) ([\d.]+)", open(os.path.join(d, "island.log")).read(), re.M)
        }
        info["dcp"] = os.path.join(d, name + "_routed.dcp")
        info["ok"] = rc == 0 and os.path.isfile(info["dcp"]) and info.get("routing_errors") == 0
        return name, info

    def top_job():
        d = os.path.join(work, "top")
        os.makedirs(d, exist_ok=True)
        v = os.path.join(d, "top.v")
        with open(v, "w") as f:
            f.write(top_verilog(g, isl))
        tcl = os.path.join(d, "top.tcl")
        with open(tcl, "w") as f:
            f.write(top_tcl(v, part, d))
        rc, t = run_vivado(tcl, os.path.join(d, "top.log"), d)
        return rc, t, d

    with ThreadPoolExecutor(max_workers=max(1, min(slots, len(isl) + 1))) as ex:
        f_top = ex.submit(top_job)
        # largest islands first
        order = sorted(isl, key=lambda n: -res["islands"][n]["cost"])
        infos = dict(f.result() for f in [ex.submit(island_job, n) for n in order])
        top_rc, top_s, top_dir = f_top.result()
    stamp("islands")
    for name, info in infos.items():
        res["islands"][name].update(info)
    res["top_s"] = top_s
    bad = [n for n, i in infos.items() if not i["ok"]]
    if bad or top_rc != 0:
        return abort("island_failed", failed=bad, top_rc=top_rc)

    # 4. stitch with RapidWright
    st_dir = os.path.join(work, "stitch")
    os.makedirs(st_dir, exist_ok=True)
    cmd = [
        java_bin(),
        "-Xmx32G",
        "-cp",
        stitcher_classpath(),
        "IslandStitcher",
        os.path.join(top_dir, "top_bb.dcp"),
        os.path.join(top_dir, "top_bb.edf"),
        accel_dcp,
        str(rwroute_max_iter),
        str(max(1, min(16, cpus))),
    ]
    cmd += ["%s=%s,%s" % (n, i["dcp"], i["dcp"].replace(".dcp", ".edf")) for n, i in infos.items()]
    t0 = time.time()
    log = os.path.join(st_dir, "stitch.log")
    env = dict(os.environ, RW_QUIET_MESSAGE="1")
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    res["stitch_s"] = time.time() - t0
    txt = open(log, errors="ignore").read()
    res["stitch_stamps"] = {kk: float(vv) for kk, vv in re.findall(r"^STAMP (\w+) ([\d.]+)", txt, re.M)}
    m = re.search(r"RESULT unrouted_pins (\d+)", txt)
    res["stitch_unrouted_pins"] = int(m.group(1)) if m else None
    stamp("stitch")
    if rc != 0 or not os.path.isfile(accel_dcp):
        return abort("stitch_failed")

    # 5. assembly (the shell is already open in the waiting Vivado)
    t0 = time.time()
    with open(trigger, "w") as f:
        f.write("go")
    rc, _ = f_asm.result()
    asm_pool.shutdown()
    res["assembly_s"] = time.time() - t0
    log = os.path.join(asm_dir, "assemble.log")
    stamp("assembly")
    res["assembly_stamps"] = {
        kk: float(vv) for kk, vv in re.findall(r"^STAMP (\w+) ([\d.]+)", open(log).read(), re.M)
    }
    res.update(_reports(asm_dir))
    res["timing_rpt"] = os.path.join(asm_dir, "timing_summary.rpt")
    res["utilization_xml"] = os.path.join(asm_dir, "synth_report.xml")
    if rc != 0 or not os.path.isfile(bitfile):
        res["status"] = "assembly_failed"
        return _done(res, stamps, out_dir, t_total)
    hwh = os.path.join(out_dir, "resizer.hwh")
    shutil.copy(os.path.join(shell_dir, "top.hwh"), hwh)
    res.update(status="ok", bitfile=bitfile, hwh=hwh)
    return _done(res, stamps, out_dir, t_total)


def _island_reports(d):
    res = {}
    rs = os.path.join(d, "route_status.rpt")
    if os.path.isfile(rs):
        m = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", open(rs).read())
        res["routing_errors"] = int(m.group(1)) if m else None
    return res


def _done(res, stamps, out_dir, t_total):
    res["stamps"] = stamps
    res["total_s"] = time.time() - t_total
    with open(os.path.join(out_dir, "rwislands_zynq.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res
