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
from finn.util.dynarapid.components import (
    batch_databases,
    batch_pblocks,
    build_component,
    component_name,
    direct_sources,
    has_pblocks,
    synth_session,
    synth_size,
)
from finn.util.dynarapid.graph import onnx_to_dot
from finn.util.dynarapid.tools import (
    JVM_GB,
    PART_TO_DYNARAPID,
    avail_memory_gb,
    dynarapid_env,
    vivado_slots,
    run_java,
    run_vivado,
    usable_cpus,
)

# cost model of one batched pblock run (GenerateBatchPblocks, xczu7ev, Vivado 2024.2, CNV):
# ~140 s for 1 component, ~230 s for 6 - the fixed cost (Vivado start, device, placer)
# dominates, the components add little each. That is on an idle machine: when the runs (and
# the large components and syntheses going on beside them) outnumber the CPUs, every run
# slows down; median run time over its idle value: 1.0 at 56 runs on 128 CPUs, 1.17 at 27 on
# 32, 1.7 at 14 on 16 (CNV, 128-thread EPYC) ~ BATCH_CPU_LOAD * concurrent runs / CPUs
BATCH_RUN_S = 120.0
BATCH_ITEM_S = 20.0
BATCH_CPU_LOAD = 1.3


def batch_plan(n_items, slots, ncpu, jvms):
    """How to spread n_items small components over batched pblock runs on this machine.

    slots: concurrent Vivado runs allowed (memory, tools.vivado_slots); ncpu: usable CPUs;
    jvms: DynaRapid JVMs that fit into memory. Chooses the components per Vivado run k that
    minimizes the estimated makespan waves * (BATCH_RUN_S + k * BATCH_ITEM_S) * CPU load
    (with many slots and CPUs: small batches, all in parallel; with few: large batches, fewer
    fixed costs), then how many of those runs share one JVM (group = k * per_jvm)."""
    n = max(1, n_items)

    def load(runs):
        return max(1.0, BATCH_CPU_LOAD * min(slots, runs) / max(1, ncpu))

    best = None
    # at most 8 per run: larger batches were not tried (congestion, packing on the device)
    for k in range(1, 9):
        runs = -(-n // k)
        cost = -(-runs // slots) * (BATCH_RUN_S + k * BATCH_ITEM_S) * load(runs)
        # ties: the larger batch (less CPU time and memory)
        if best is None or cost <= best[0]:
            best = (cost, k)
    k = best[1]
    runs = -(-n // k)
    # every group (task of the batch pool) holds one JVM while its runs go on
    pool = max(1, min(jvms, slots))
    per_jvm = max(1, min(3, -(-runs // pool)))
    # Vivado threads per batch run: the CPUs spread over the runs that go on at once (more
    # threads than CPUs slow all runs down, see BATCH_CPU_LOAD)
    threads = max(1, min(4, ncpu // max(1, min(slots, runs))))
    # time limit of a batched run, after which it is retried at lower utilization (the
    # DynaRapid default is a fixed 300 s): twice the expected time under the CPU load; a
    # limit hit by a healthy but slow run costs a retry and an individual rebuild at the end
    expected = (BATCH_RUN_S + k * BATCH_ITEM_S) * load(runs)
    timeout = max(300, int(2 * expected))
    return dict(
        k=k, group=k * per_jvm, per_jvm=per_jvm, pool=pool, threads=threads, timeout_s=timeout
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
    batch=True,
    batch_util=0.6,
    large_luts=5000,
    large_bram=2,
):
    """Build (or reuse) the components of all nodes. Returns ({node: dcp}, [results]).

    batch: generate the pblocks of many components in shared Vivado runs (the fixed cost of
    a place-and-route run dominates small components), falling back to individual runs."""
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
    ncpu = usable_cpus()
    pblock_parallel = max(1, min(3, ncpu // max(1, len(uniq))))
    # every job holds a JVM with the device model while its Vivado runs are going on; the
    # Vivado runs themselves are limited machine-wide (tools.vivado_slots)
    jvms = max(1, int(0.85 * avail_memory_gb() * 0.4 / JVM_GB))
    mem_workers = jvms
    if mem_workers < workers:
        print("DynaRapid: limiting parallel component jobs to %d (memory)" % mem_workers)
        workers = mem_workers
    # with few components, each Vivado run can use several threads
    if vivado_threads is None:
        vivado_threads = max(1, min(8, ncpu // max(1, len(uniq) * pblock_parallel)))

    def job(item, stages=("synth", "pblock", "database")):
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
            stages=stages,
        )

    # sort by (rough) size so the largest components start first
    items = sorted(uniq.items(), key=lambda it: -_node_size_hint(it[1]))
    if not batch:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            results = list(ex.map(job, items))
        return dcps, results

    # batched and pipelined: components are synthesized in parallel; whenever `group` of
    # them are synthesized, their pblocks are generated together (shared Vivado runs) and
    # then their placement databases, while the other syntheses go on. Components that fail
    # in a batch are built individually at the end.
    from concurrent.futures import as_completed

    t_start = time.time()
    by_dcp = {}
    todo = []
    for it in items:
        if os.path.isfile(os.path.join(library_dir, it[0] + ".bin.data")):
            by_dcp[it[0]] = {"dcp": it[0], "node": it[1].name, "op_type": it[1].op_type, "status": "cached"}
        else:
            todo.append(it)
    slots = vivado_slots()[1]
    # groups (each one DynaRapid run with 1-3 batched Vivado runs) sized for this machine
    plan = batch_plan(len(todo), slots, ncpu, jvms)
    group = plan["group"]
    print(
        "DynaRapid library plan: %d components, %d Vivado slots, %d CPUs: %s"
        % (len(todo), slots, ncpu, plan)
    )
    batched_ok = set()
    timeline = {}

    def large_job(item):
        r = job(item, ("synth", "pblock", "database"))
        by_dcp[r["dcp"]].update(r)
        return [r["dcp"]]

    def pblock_group(dcps):
        need = [d for d in dcps if not has_pblocks(library_dir, d)]
        ok, _ = batch_pblocks(
            need,
            work_dir,
            library_dir,
            part,
            clk_ns,
            util=batch_util,
            batches=max(1, -(-len(need) // plan["k"])),
            threads=plan["threads"],
            timeout_s=plan["timeout_s"],
        )
        batched_ok.update(ok)
        db = [d for d in dcps if has_pblocks(library_dir, d)]
        batch_databases(db, work_dir, library_dir, part, clk_ns)
        return dcps

    # each group holds a DynaRapid JVM (device model) while its batches run
    with ThreadPoolExecutor(max_workers=max(1, workers)) as synth_ex, ThreadPoolExecutor(
        max_workers=plan["pool"]
    ) as batch_ex:
        # directly synthesizable components in sessions (several per Vivado run), the others
        # (block design) individually
        direct = [it for it in todo if direct_sources(it[1]) is not None]
        other = [it for it in todo if direct_sources(it[1]) is None]
        n_sess = max(1, min(slots, len(direct)))
        size = max(1, min(8, (len(direct) + n_sess - 1) // n_sess))
        sessions = [direct[k : k + size] for k in range(0, len(direct), size)]
        synth_futs = [synth_ex.submit(job, it, ("synth",)) for it in other]
        synth_futs += [
            synth_ex.submit(synth_session, sess, work_dir, library_dir, part, clk_ns)
            for sess in sessions
        ]
        # large components (many LUTs / BRAMs) are implemented individually as soon as they
        # are synthesized: batching does not save much for them, and a large congested
        # component slows down (and can fail) the whole batch (Vivado 2024.2: the TFC 784x64
        # MVAU with 3.5 BRAM tiles kept a 7-component batch routing until its time limit;
        # the TFC input IODMA with 2 BRAM tiles congests at the batch utilization 0.6 even
        # alone, the individual path's hedged lower-utilization attempt avoids the limit)
        by_node = dict(todo)
        batch_futs, ready = [], []
        for f in as_completed(synth_futs):
            rs = f.result()
            for r in rs if isinstance(rs, list) else [rs]:
                by_dcp[r["dcp"]] = r
                if r["status"] != "synthesized":
                    continue
                luts, bram = synth_size(work_dir, r["dcp"])
                if luts > large_luts or bram >= large_bram:
                    item = (r["dcp"], by_node[r["dcp"]])
                    batch_futs.append(batch_ex.submit(large_job, item))
                else:
                    ready.append(r["dcp"])
            if len(ready) >= group:
                batch_futs.append(batch_ex.submit(pblock_group, ready))
                ready = []
        timeline["synth_done_s"] = time.time() - t_start
        if ready:
            batch_futs.append(batch_ex.submit(pblock_group, ready))
        for f in batch_futs:
            f.result()
    timeline["batches_done_s"] = time.time() - t_start

    # components without pblocks (failed in their batch): individually, then their database
    # (and components whose synthesis failed within a session: once more on their own)
    fallback = [
        it
        for it in todo
        if by_dcp[it[0]]["status"] in ("synthesized", "synth_failed")
        and not has_pblocks(library_dir, it[0])
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for r in ex.map(lambda it: job(it, ("synth", "pblock")), fallback):
            by_dcp[r["dcp"]].update(r)
    db = [
        d
        for d, r in by_dcp.items()
        if r["status"] in ("synthesized", "pblocks")
        and has_pblocks(library_dir, d)
        and not os.path.isfile(os.path.join(library_dir, d + ".bin.data"))
    ]
    batch_databases(db, work_dir, library_dir, part, clk_ns)
    timeline["done_s"] = time.time() - t_start
    for d, r in by_dcp.items():
        if r["status"] in ("synthesized", "pblocks", "built"):
            built = os.path.isfile(os.path.join(library_dir, d + ".bin.data"))
            if not has_pblocks(library_dir, d):
                r["status"] = "pblock_failed"
            else:
                r["status"] = "built" if built else "database_failed"
            r["batched"] = d in batched_ok
    stats = dict(timeline, batched=len(batched_ok), fallback=len(fallback), **plan)
    print("DynaRapid library:", stats)
    for r in by_dcp.values():
        r["library_stats"] = stats
    return dcps, list(by_dcp.values())


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
    place_region=None,
    blocked_tiles=None,
    place_order=None,
    rwroute_max_iter=None,
):
    """Run the full DynaRapid flow on a FINN dataflow model; returns a result dict.

    place_region: "top,bottom,left,right" DynaRapid map rows/columns the components are
    placed in (clipped to the map), default the whole map.
    blocked_tiles: file with tile names (one per line) the components must not cover.
    place_order: None (graph order) or "size" (largest components first).

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
    if place_region is not None:
        env["DYNARAPID_PLACE_REGION"] = place_region
    if blocked_tiles is not None:
        env["DYNARAPID_BLOCKED_TILES"] = blocked_tiles
    if place_order is not None:
        env["DYNARAPID_PLACE_ORDER"] = place_order
    if rwroute_max_iter is not None:
        env["DYNARAPID_RWROUTE_MAX_ITER"] = str(rwroute_max_iter)
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
