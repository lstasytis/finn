# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DynaRapid bitfile flow for Zynq: the whole accelerator (IODMAs and compute layers) is
placed and routed by DynaRapid and inserted into a pre-implemented shell.

    ZynqBuild partitions (idma, kernel, odma)
      -> one accelerator graph                          (merge_partitions)
      -> in parallel:
           DynaRapid P&R of the accelerator             (components in parallel, cached)
           pre-implemented shell                        (cached per board / interfaces)
      -> assembly: shell + routed accelerator, route boundary and clock nets, bitstream

Compared to the regular flow, no stitched IP is packaged, no block design with the
accelerator is synthesized and the accelerator is never placed or routed by Vivado.
"""

import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.util.dynarapid.flow import build_library, dynarapid_pnr
from finn.util.dynarapid.graph import mm_ports
from finn.util.dynarapid.shell import SHELL_STRIP_COLS, assemble_tcl, build_shell
from finn.util.dynarapid.tools import run_vivado


def default_library_dir(part):
    return os.path.join(os.environ["FINN_BUILD_DIR"], "dynarapid_library", part, "lib")


def merge_partitions(kernel_models):
    """One model with the nodes of all (consecutive) dataflow partitions. The partitions
    share their boundary tensor names with the parent graph."""
    first, last = kernel_models[0], kernel_models[-1]
    nodes, inits, vis = [], {}, {}
    for km in kernel_models:
        nodes += list(km.graph.node)
        for i in km.graph.initializer:
            inits[i.name] = i
        for v in list(km.graph.value_info) + list(km.graph.input) + list(km.graph.output):
            vis.setdefault(v.name, v)
    ins = [vis[t.name] for t in first.graph.input]
    outs = [vis[t.name] for t in last.graph.output]
    io = {t.name for t in ins + outs}
    graph = helper.make_graph(
        nodes,
        "dynarapid_accel",
        ins,
        outs,
        initializer=list(inits.values()),
        value_info=[v for n, v in vis.items() if n not in io],
    )
    model = ModelWrapper(
        helper.make_model(
            graph, producer_name="finn-dynarapid", opset_imports=list(first.model.opset_import)
        )
    )
    # annotations (datatypes, layouts) of all partitions
    seen = set()
    for km in kernel_models:
        for a in km.graph.quantization_annotation:
            if a.tensor_name not in seen:
                seen.add(a.tensor_name)
                model.graph.quantization_annotation.append(a)
    return model


def dynarapid_zynq_build(
    kernel_models,
    board,
    part,
    clk_ns,
    out_dir,
    library_dir=None,
    shell_lib=None,
    workers=None,
):
    """Build the bitfile of a ZynqBuild design (partition models with generated IP).
    Returns a result dict with the bitfile, hwh and per-stage times."""
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or os.cpu_count()
    library_dir = library_dir or default_library_dir(part)
    shell_lib = shell_lib or os.path.join(os.path.dirname(os.path.abspath(library_dir)), "shells")
    res = {"out_dir": out_dir, "board": board, "part": part, "clk_ns": clk_ns}
    t_total = time.time()

    accel = merge_partitions(kernel_models)
    accel.save(os.path.join(out_dir, "accel.onnx"))
    ports = mm_ports(accel)
    assert ports, "the DynaRapid shell flow needs IODMAs at the accelerator boundary"

    def shell_job():
        return build_shell(board, part, clk_ns, ports, shell_lib, jobs=max(1, workers // 4))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_shell = ex.submit(shell_job)
        # the component library does not depend on the shell: build it meanwhile
        f_lib = ex.submit(
            build_library,
            accel,
            os.path.join(os.path.dirname(os.path.abspath(library_dir)), "work"),
            library_dir,
            part,
            clk_ns,
            workers,
        )
        shell_dir, shell_res = f_shell.result()
        f_lib.result()
    res["parallel_s"] = time.time() - t0
    res["shell"] = shell_res
    if shell_res["status"] not in ("built", "cached"):
        res["status"] = "shell_failed"
        return _done(res, out_dir, t_total)

    blocked = os.path.join(shell_dir, "blocked_tiles.txt")

    def accel_job(place_order=None):
        return dynarapid_pnr(
            accel,
            os.path.join(out_dir, "dynarapid"),
            library_dir,
            part,
            clk_ns,
            workers=workers,
            graph_name="accel",
            check=False,
            no_clock=True,
            # keep the components out of the strip next to the PS used by the shell
            place_region="0,100000,%d,100000" % SHELL_STRIP_COLS,
            # sites the shell cannot give to the accelerator (if any)
            blocked_tiles=blocked if os.path.isfile(blocked) else None,
            place_order=place_order,
        )

    # components are in the library now: place, stitch and route the accelerator
    t0 = time.time()
    accel_res = accel_job()
    log = os.path.join(out_dir, "dynarapid", "generate_design.log")
    if accel_res["status"] == "dynarapid_failed" and "Could not find placement" in open(log).read():
        # the greedy placer in graph order can fill the few column bands that large
        # components (BRAMs) can go to with small ones: place the largest ones first
        res["place_retry"] = "size"
        accel_res = accel_job(place_order="size")
    res["stitch_s"] = time.time() - t0
    res["accel"] = {k: v for k, v in accel_res.items() if k != "components"}
    res["accel_components"] = accel_res.get("components")
    if accel_res["status"] != "routed":
        res["status"] = "accel_" + accel_res["status"]
        return _done(res, out_dir, t_total)

    # assembly
    asm_dir = os.path.join(out_dir, "assembly")
    os.makedirs(asm_dir, exist_ok=True)
    bitfile = os.path.join(out_dir, "resizer.bit")
    tcl = os.path.join(asm_dir, "assemble.tcl")
    with open(tcl, "w") as f:
        f.write(assemble_tcl(shell_dir, accel_res["routed_dcp"], asm_dir, bitfile))
    log = os.path.join(asm_dir, "assemble.log")
    rc, res["assembly_s"] = run_vivado(tcl, log, asm_dir)
    res["assembly_stamps"] = {
        k: float(v) for k, v in re.findall(r"^STAMP (\w+) ([\d.]+)", open(log).read(), re.M)
    }
    res.update(_reports(asm_dir))
    if rc != 0 or not os.path.isfile(bitfile):
        res["status"] = "assembly_failed"
        return _done(res, out_dir, t_total)
    hwh = os.path.join(out_dir, "resizer.hwh")
    shutil.copy(os.path.join(shell_dir, "top.hwh"), hwh)
    res.update(status="ok", bitfile=bitfile, hwh=hwh)
    return _done(res, out_dir, t_total)


def _reports(rpt_dir):
    res = {}
    rs = os.path.join(rpt_dir, "route_status.rpt")
    if os.path.isfile(rs):
        txt = open(rs).read()
        m = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", txt)
        res["nets_with_routing_errors"] = int(m.group(1)) if m else None
    ts = os.path.join(rpt_dir, "timing_summary.rpt")
    if os.path.isfile(ts):
        txt = open(ts).read()
        txt = txt[txt.find("Design Timing Summary") :]
        m = re.search(r"WNS\(ns\)\s+TNS\(ns\)[^\n]*\n\s*-[- ]+\n\s+(\S+)\s+(\S+)", txt)
        if m:
            res["wns_ns"] = None if m.group(1) == "NA" else float(m.group(1))
    return res


def _done(res, out_dir, t_total):
    res["total_s"] = time.time() - t_total
    with open(os.path.join(out_dir, "dynarapid_zynq.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res