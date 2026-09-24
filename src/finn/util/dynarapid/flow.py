# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end DynaRapid place-and-route of a FINN dataflow model.

    ONNX (all nodes through IP generation)
      -> one pre-implemented component per node      (parallel Vivado + DynaRapid jobs)
      -> DynaRapid dot graph from the ONNX topology
      -> DynaRapid GenerateDesign: place components, stitch, route with RWRoute
      -> (optional) Vivado check of the routed DCP and bitstream
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.dynarapid.components import build_component, component_name
from finn.util.dynarapid.graph import onnx_to_dot
from finn.util.dynarapid.tools import (
    PART_TO_DYNARAPID,
    dynarapid_env,
    run_java,
    run_vivado,
)


def build_library(
    model,
    work_dir,
    library_dir,
    part,
    clk_ns,
    workers,
    num_shapes=1,
    target_util=0.8,
    vivado_threads=None,
    pblock_mode="fast",
):
    """Build (or reuse) the components of all nodes. Returns ({node: dcp}, [results])."""
    # memory initialization files are referenced with relative paths ("./x.dat") in some
    # generated HDL; make them absolute as CreateStitchedIP does, or synthesis silently
    # leaves the memories empty
    model = model.transform(ReplaceVerilogRelPaths())
    dcps = {n.name: component_name(model, n, part, clk_ns) for n in model.graph.node}
    uniq = {}
    for n in model.graph.node:
        uniq.setdefault(dcps[n.name], n)

    # spare workers are used for speculative pblock attempts at lower utilization
    # (the densest successful one is kept), which avoids sequential retries
    ncpu = os.cpu_count() or workers
    pblock_parallel = max(1, min(3, ncpu // max(1, len(uniq))))
    # every job holds a JVM with the device model and one Vivado per attempt; on large
    # devices memory, not cores, limits the number of parallel jobs
    avail_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    per_job_gb = _job_memory_gb(part) * pblock_parallel
    mem_workers = max(1, int(0.85 * avail_gb / per_job_gb))
    if mem_workers < workers:
        print("DynaRapid: limiting parallel component jobs to %d (memory)" % mem_workers)
        workers = mem_workers
    # with few components, each Vivado run can use several threads
    if vivado_threads is None:
        vivado_threads = max(1, min(8, ncpu // max(1, len(uniq) * pblock_parallel)))

    def job(item):
        dcp, node = item
        return build_component(
            model,
            node,
            dcp,
            work_dir,
            library_dir,
            part,
            clk_ns,
            num_shapes=num_shapes,
            vivado_threads=vivado_threads,
            target_util=target_util,
            pblock_mode=pblock_mode,
            pblock_parallel=pblock_parallel,
        )

    # sort by (rough) size so the largest components start first
    items = sorted(uniq.items(), key=lambda it: -_node_size_hint(it[1]))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(job, items))
    return dcps, results


def _job_memory_gb(part):
    """Rough peak memory of one component job (JVM + Vivado) for a device."""
    large = part.startswith(("xcvu", "xcu2", "xcu5", "xcvp", "xcvc"))
    return 10.0 if large else 3.0


def _node_size_hint(node):
    size = 1
    for a in node.attribute:
        if a.name in ("PE", "SIMD"):
            size *= max(1, a.i)
    return size


def check_tcl(dcp, rpt_dir, clk_ns, bitfile=None):
    tcl = [
        "open_checkpoint %s" % dcp,
        "report_route_status -file %s/route_status.rpt" % rpt_dir,
        "report_timing_summary -file %s/timing_summary.rpt" % rpt_dir,
        "report_utilization -file %s/utilization.rpt" % rpt_dir,
        "report_drc -file %s/drc.rpt" % rpt_dir,
    ]
    if bitfile is not None:
        # the design is a bare accelerator without I/O constraints
        tcl += [
            "set_property SEVERITY {Warning} [get_drc_checks NSTD-1]",
            "set_property SEVERITY {Warning} [get_drc_checks UCIO-1]",
            "write_bitstream -force %s" % bitfile,
        ]
    return "\n".join(tcl) + "\n"


def parse_reports(rpt_dir):
    res = {}
    with open(os.path.join(rpt_dir, "route_status.rpt")) as f:
        txt = f.read()
    m = re.search(r"# of routable nets\.+ :\s+(\d+)", txt)
    res["routable_nets"] = int(m.group(1)) if m else None
    m = re.search(r"# of fully routed nets\.+ :\s+(\d+)", txt)
    res["fully_routed_nets"] = int(m.group(1)) if m else None
    m = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", txt)
    res["nets_with_routing_errors"] = int(m.group(1)) if m else None
    with open(os.path.join(rpt_dir, "timing_summary.rpt")) as f:
        txt = f.read()
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+(\S+)", txt, re.S)
    if m:
        res["wns_ns"] = None if m.group(1) == "NA" else float(m.group(1))
        res["tns_ns"] = None if m.group(2) == "NA" else float(m.group(2))
    with open(os.path.join(rpt_dir, "utilization.rpt")) as f:
        txt = f.read()
    for key, pat in (
        ("LUT", r"\| CLB LUTs\*?\s+\|\s+(\d+)"),
        ("FF", r"\| CLB Registers\s+\|\s+(\d+)"),
        ("BRAM", r"\| Block RAM Tile\s+\|\s+([\d.]+)"),
        ("DSP", r"\| DSPs\s+\|\s+(\d+)"),
    ):
        m = re.search(pat, txt)
        res[key] = float(m.group(1)) if m else None
    drc = os.path.join(rpt_dir, "drc.rpt")
    if os.path.isfile(drc):
        res["drc"] = {
            rule: int(n)
            for rule, n in re.findall(
                r"^\| (\S+)\s+\| [^|]+\| [^|]+\|\s+(\d+)\s+\|", open(drc).read(), re.M
            )
        }
    return res


def dynarapid_pnr(
    model,
    out_dir,
    library_dir,
    part,
    clk_ns,
    workers=28,
    graph_name="finn_design",
    placer="greedy",
    num_shapes=1,
    target_util=0.8,
    vivado_threads=None,
    check=True,
    bitstream=False,
    work_dir=None,
    no_clock=False,
    pblock_mode="fast",
):
    """Run the full DynaRapid flow on a FINN dataflow model; returns a result dict.

    no_clock: leave clk as a plain port (for embedding into a shell), otherwise DynaRapid
    drives it through a BUFGCE and routes the clock itself."""
    os.makedirs(out_dir, exist_ok=True)
    # synthesized components and DynaRapid scratch live next to the library by default
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(os.path.abspath(library_dir)), "work")
    res = {"graph": graph_name, "nodes": len(model.graph.node)}
    t_total = time.time()

    # 1. component library (only missing components are built)
    t0 = time.time()
    dcps, comp_results = build_library(
        model,
        work_dir,
        library_dir,
        part,
        clk_ns,
        workers,
        num_shapes=num_shapes,
        target_util=target_util,
        vivado_threads=vivado_threads,
        pblock_mode=pblock_mode,
    )
    res["components_s"] = time.time() - t0
    res["components"] = comp_results
    res["unique_components"] = len(comp_results)
    res["components_built"] = sum(r["status"] == "built" for r in comp_results)
    failed = [r for r in comp_results if r["status"] not in ("built", "cached")]
    if failed:
        res["status"] = "component_failed"
        res["total_s"] = time.time() - t_total
        _dump(res, out_dir)
        return res

    # 2. graph
    t0 = time.time()
    dot, idmap = onnx_to_dot(model, dcps, graph_name)
    dot_file = os.path.join(out_dir, graph_name + ".dot")
    with open(dot_file, "w") as f:
        f.write(dot)
    with open(os.path.join(out_dir, graph_name + "_nodes.json"), "w") as f:
        json.dump({k: {"node": v, "dcp": dcps[v]} for k, v in idmap.items()}, f, indent=2)
    res["graph_s"] = time.time() - t0

    # 3. DynaRapid placement, stitching and routing
    env = dynarapid_env(work_dir, library_dir, part, clk_ns, vivado_threads or 1)
    args = ["-f", dot_file, "-part", PART_TO_DYNARAPID[part], "-placer", placer]
    args += ["-threads", str(max(1, workers))]
    if no_clock:
        # clk stays a plain port (no BUFGCE): the design is embedded into a shell later
        args += ["-noClock"]
    rc, res["dynarapid_s"] = run_java(
        "ch.agsl.dynarapid.GenerateDesign",
        args,
        env,
        os.path.join(out_dir, "generate_design.log"),
        heap="32G",
    )
    routed = os.path.join(work_dir, "designs", graph_name, graph_name + "_routed.dcp")
    res["routed_dcp"] = routed
    if rc != 0 or not os.path.isfile(routed):
        res["status"] = "dynarapid_failed"
        res["total_s"] = time.time() - t_total
        _dump(res, out_dir)
        return res
    res["pnr_total_s"] = time.time() - t_total
    res["status"] = "routed"

    # 4. optional Vivado check / bitstream (not part of the P&R time)
    if check or bitstream:
        rpt_dir = os.path.join(out_dir, "reports")
        os.makedirs(rpt_dir, exist_ok=True)
        bitfile = os.path.join(out_dir, graph_name + ".bit") if bitstream else None
        tcl = os.path.join(rpt_dir, "check.tcl")
        with open(tcl, "w") as f:
            f.write(check_tcl(routed, rpt_dir, clk_ns, bitfile))
        rc, res["check_s"] = run_vivado(tcl, os.path.join(rpt_dir, "check.log"), rpt_dir)
        try:
            res["check"] = parse_reports(rpt_dir)
        except (OSError, AttributeError) as e:
            res["check"] = {"error": str(e)}
        if bitstream:
            res["bitstream"] = bitfile if os.path.isfile(bitfile) else None
    res["total_s"] = time.time() - t_total
    _dump(res, out_dir)
    return res


def _dump(res, out_dir):
    with open(os.path.join(out_dir, "dynarapid_result.json"), "w") as f:
        json.dump(res, f, indent=2)
