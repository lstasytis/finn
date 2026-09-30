# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-implemented Zynq shell for the DynaRapid bitfile flow.

The regular ZynqBuild implements the shell (PS, AXI interconnect, SmartConnect, IODMAs)
together with the accelerator in one global Vivado run. Here the shell is implemented on
its own, once, and cached:

  * the whole accelerator (IODMAs and compute layers, placed and routed by DynaRapid) is
    one cell of the shell: during the shell implementation a placeholder (DONT_TOUCH, so
    no shell logic around it is optimized away), afterwards a black box with unrouted
    boundary nets. (DFX partitions were tried first: snapping excludes several full-height
    columns next to clocking/configuration columns on the xczu7ev, which leaves too few
    relocation sites for components with BRAMs.)
  * the IODMAs keep their block-design names (idma0, odma0, ...) as thin pass-through
    modules between the interconnects and the black box, so that the hardware handoff
    (.hwh) of the shell has the address map the FINN driver expects;
  * the shell logic lives next to the PS (below it and in a narrow strip beside its
    fabric interface; pblock with EXCLUDE_PLACEMENT and CONTAIN_ROUTING), the rest of the
    device is left to DynaRapid.

The shell only depends on the board, the clock and the widths of the IODMA memory-mapped
interfaces, so it is shared by all models with the same interfaces.

Filling it (read_checkpoint -cell with the DynaRapid-routed accelerator), routing the
boundary and clock nets and writing the bitstream is assemble_tcl().
"""

import hashlib
import json
import os
import shutil

from finn.util.dynarapid.tools import run_vivado, vivado_version

CORE_MODULE = "finn_accel_core"
CORE_CELL = "accel"

# AXI signals exposed on the bridge interfaces (AXI4 / AXI4-Lite subsets of what the HLS
# IODMA drives; the others are tied off / left open)
_MASTER_SIGS = {
    "AW": ["ADDR", "LEN", "SIZE", "BURST", "LOCK", "CACHE", "PROT", "QOS"],
    "W": ["DATA", "STRB", "LAST"],
    "B": ["RESP"],
    "AR": ["ADDR", "LEN", "SIZE", "BURST", "LOCK", "CACHE", "PROT", "QOS"],
    "R": ["DATA", "RESP", "LAST"],
}
_SLAVE_SIGS = {
    "AW": ["ADDR"],
    "W": ["DATA", "STRB"],
    "B": ["RESP"],
    "AR": ["ADDR"],
    "R": ["DATA", "RESP"],
}

# DynaRapid map columns (after MapBuilderFPGA drops the columns below the PS) that are
# left to the shell, next to the PS fabric interface: the PS AXI pins enter the fabric
# through the INT column right of the PS
SHELL_STRIP_COLS = 3
# INT column of the first map column (the PS boundary) per part
PS_BOUNDARY_INT_X = {
    "xczu7ev-ffvc1156-2-e": 27,
    # first full-height column right of the PS (sites in all rows from here on)
    "xczu9eg-ffvb1156-2-e": 24,
}


def shell_key(board, part, clk_ns, ports, shell_x0=0):
    sig = {
        "board": board,
        "part": part,
        "clk_ns": clk_ns,
        "dmas": [
            {
                "id": d["id"],
                "in": [(c["name"], c["signals"]) for c in d["in"]],
                "out": [(c["name"], c["signals"]) for c in d["out"]],
            }
            for d in ports
        ],
        "version": 4,
        "vivado": vivado_version(),
    }
    if shell_x0:
        # shell confined to INT columns [shell_x0, PS boundary + strip) (island flow)
        sig["shell_x0"] = shell_x0
    h = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
    return "shell%s" % h


def _bus(w):
    return "[%d:0] " % (w - 1) if w > 1 else ""


def core_verilog(ports):
    """Placeholder with the DynaRapid top-level ports of the accelerator, used while the
    shell is implemented: every input feeds a register driving all outputs, so that no
    shell logic around it is optimized away. Turned into a black box afterwards."""
    decl = ["    input clk", "    input rst"]
    ins, outs = ["rst"], []
    for d in ports:
        for c in d["in"]:
            decl += [
                "    input %s%s" % (_bus(c["width"]), c["data"]),
                "    input %s" % c["valid"],
                "    output %s" % c["ready"],
            ]
            ins += [c["data"], c["valid"]]
            outs.append((c["ready"], 1))
        for c in d["out"]:
            decl += [
                "    output %s%s" % (_bus(c["width"]), c["data"]),
                "    output %s" % c["valid"],
                "    input %s" % c["ready"],
            ]
            ins.append(c["ready"])
            outs += [(c["data"], c["width"]), (c["valid"], 1)]
    body = [
        '    (* DONT_TOUCH = "true" *) reg p = 0;',
        "    always @(posedge clk) p <= ^{%s};" % ", ".join(ins),
    ]
    # one register per output bit: outputs sharing a driver would share their routing,
    # which is invalid once the placeholder is removed and they become separate nets
    for k, (o, w) in enumerate(outs):
        body += [
            '    (* DONT_TOUCH = "true" *) reg %sq%d = 0;' % (_bus(w), k),
            "    always @(posedge clk) q%d <= {%d{p}};" % (k, w),
            "    assign %s = q%d;" % (o, k),
        ]
    return '(* keep_hierarchy = "yes" *)\nmodule %s (\n%s\n);\n%s\nendmodule\n' % (
        CORE_MODULE,
        ",\n".join(decl),
        "\n".join(body),
    )


def bridge_verilog(d):
    """Pass-through module named after the IODMA: AXI interfaces towards the interconnects,
    the packed elastic channels towards the accelerator core (ports core_<name>)."""
    chans = {}
    for c in d["in"] + d["out"]:
        chans[c["name"]] = c
    # IODMAs: AXI master + AXI-Lite control; compute nodes: only an AXI-Lite slave
    master = [c["name"] for c in d["in"] + d["out"] if c["name"].startswith("m_axi")]
    slave = [c["name"] for c in d["in"] + d["out"] if c["name"].startswith("s_axi")][0]
    m_if = master[0].rsplit("_", 1)[0] if master else None  # e.g. m_axi_gmem
    s_if = slave.rsplit("_", 1)[0]  # e.g. s_axi_control / s_axilite
    ports, body = [], []
    ports.append(
        '    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)\n'
        '    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF %s, ASSOCIATED_RESET ap_rst_n" *)\n'
        "    input ap_clk" % ":".join(x for x in (m_if, s_if) if x)
    )
    ports.append(
        '    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)\n'
        '    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)\n'
        "    input ap_rst_n"
    )
    ports += ["    output core_clk", "    output core_rst"]
    body += ["    assign core_clk = ap_clk;", "    assign core_rst = ap_rst_n;"]

    def axi(if_name, is_master, sigs):
        nonlocal ports
        first = True
        for ch in ("AW", "W", "B", "AR", "R"):
            c = chans["%s_%s" % (if_name, ch)]
            # direction of the channel payload at the bridge's AXI interface
            payload_out = (ch in ("AW", "W", "AR")) == is_master
            to_core = not payload_out  # payload flows from the interconnect into the core
            packed = {sig[len(c["name"]) :]: (off, w) for (sig, w), off in _offsets(c)}
            core_data = "core_" + c["data"]
            core_valid = "core_" + c["valid"]
            core_ready = "core_" + c["ready"]
            # core-side ports (named after the DynaRapid ports)
            if to_core:
                ports += [
                    "    output %s%s" % (_bus(c["width"]), core_data),
                    "    output %s" % core_valid,
                    "    input %s" % core_ready,
                ]
            else:
                ports += [
                    "    input %s%s" % (_bus(c["width"]), core_data),
                    "    input %s" % core_valid,
                    "    output %s" % core_ready,
                ]
            # AXI side
            for s in ["VALID", "READY"] + sigs[ch]:
                full = "%s_%s%s" % (if_name, ch, s)
                if s == "VALID":
                    d_out = payload_out
                elif s == "READY":
                    d_out = not payload_out
                else:
                    d_out = payload_out
                if s in ("VALID", "READY"):
                    w = 1
                elif s in packed:
                    w = packed[s][1]
                    if s == "LOCK":
                        w = 1  # AXI4: single-bit lock (HLS drives an AXI3 2-bit lock)
                else:
                    continue
                attr = '    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 %s %s%s" *)' % (
                    if_name,
                    ch,
                    s,
                )
                if first:
                    attr += (
                        '\n    (* X_INTERFACE_PARAMETER = "PROTOCOL %s, MODE %s" *)'
                        % ("AXI4" if is_master else "AXI4LITE", "Master" if is_master else "Slave")
                    )
                    first = False
                ports.append(
                    "%s\n    %s %s%s" % (attr, "output" if d_out else "input", _bus(w), full)
                )
                # wiring
                if s == "VALID":
                    if to_core:
                        body.append("    assign %s = %s;" % (core_valid, full))
                    else:
                        body.append("    assign %s = %s;" % (full, core_valid))
                elif s == "READY":
                    if to_core:
                        body.append("    assign %s = %s;" % (full, core_ready))
                    else:
                        body.append("    assign %s = %s;" % (core_ready, full))
                else:
                    off, pw = packed[s]
                    if to_core:
                        body.append(
                            "    assign %s[%d:%d] = %s;" % (core_data, off + w - 1, off, full)
                            if c["width"] > 1
                            else "    assign %s = %s;" % (core_data, full)
                        )
                    else:
                        rng = "[%d:%d]" % (off + w - 1, off) if c["width"] > 1 else ""
                        body.append("    assign %s = %s%s;" % (full, core_data, rng))
            # payload bits of the core input that the bridge does not drive
            if to_core:
                for (sig, w), off in _offsets(c):
                    s = sig[len(c["name"]) :]
                    driven = s in sigs[ch]
                    lo = off + (1 if (driven and s == "LOCK") else (w if driven else 0))
                    if lo < off + w:
                        body.append(
                            "    assign %s[%d:%d] = 0;" % (core_data, off + w - 1, lo)
                            if c["width"] > 1
                            else "    assign %s = 0;" % core_data
                        )

    if m_if:
        axi(m_if, True, _MASTER_SIGS)
    axi(s_if, False, _SLAVE_SIGS)
    return "module %s_bridge (\n%s\n);\n%s\nendmodule\n" % (
        d["id"],
        ",\n".join(ports),
        "\n".join(body),
    )


def _offsets(c):
    off = 0
    for sig, w in c["signals"]:
        yield (sig, w), off
        off += w


def shell_tcl(board, part, clk_ns, ports, shell_dir, src_file, jobs, shell_x0=0):
    """Vivado script building and implementing the shell with the accelerator as a DFX
    partition. Writes shell_routed.dcp (partition empty, static routing locked) and
    top.hwh into shell_dir."""
    from finn.transformation.fpgadataflow import templates

    fclk_mhz = int(1 / (clk_ns * 0.001))
    ps_x = PS_BOUNDARY_INT_X[part] + SHELL_STRIP_COLS
    # AXI-Lite slaves: every entry (IODMA control, compute-node parameters); AXI masters: IODMAs
    n_axilite = len(ports)
    n_aximm = len([d for d in ports if d.get("iodma", True)])
    # board/PS setup from the regular Zynq shell template (up to the IP instantiations)
    head = templates.custom_zynq_shell_template.split("#custom IP instantiations")[0]
    head = head % (fclk_mhz, n_axilite, n_aximm, board, part)
    head = head.replace(
        "create_project finn_zynq_link ./ -part $FPGA_PART",
        "create_project finn_zynq_link %s -part $FPGA_PART\nadd_files -norecurse %s\n"
        "update_compile_order -fileset sources_1" % (shell_dir, src_file),
    )
    t = [head]
    k_mm = 0
    for k, d in enumerate(ports):
        s_if = [c["name"] for c in d["in"] if c["name"].startswith("s_axi")][0].rsplit("_", 1)[0]
        t.append("create_bd_cell -type module -reference %s_bridge %s" % (d["id"], d["id"]))
        if d.get("iodma", True):
            t.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/m_axi_gmem] "
                "[get_bd_intf_pins smartconnect_0/S%02d_AXI]" % (d["id"], k_mm)
            )
            k_mm += 1
        t += [
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
            "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]" % (d["id"], s_if, k),
            "assign_axi_addr_proc %s/%s" % (d["id"], s_if),
            "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins smartconnect_0/aclk]" % d["id"],
            "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]"
            % d["id"],
        ]
    t.append("create_bd_cell -type module -reference %s %s" % (CORE_MODULE, CORE_CELL))
    t += [
        "connect_bd_net [get_bd_pins %s/core_clk] [get_bd_pins %s/clk]" % (ports[0]["id"], CORE_CELL),
        "connect_bd_net [get_bd_pins %s/core_rst] [get_bd_pins %s/rst]" % (ports[0]["id"], CORE_CELL),
    ]
    for d in ports:
        for c in d["in"] + d["out"]:
            for p in (c["data"], c["valid"], c["ready"]):
                t.append(
                    "connect_bd_net [get_bd_pins %s/core_%s] [get_bd_pins %s/%s]"
                    % (d["id"], p, CORE_CELL, p)
                )
    t += [
        "apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} }"
        "  [get_bd_pins axi_interconnect_0/M*_ACLK]",
        "save_bd_design",
        "assign_bd_address",
        "validate_bd_design",
        'set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [get_files top.bd]',
        "make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top",
        "set_property top top_wrapper [current_fileset]",
        "update_compile_order -fileset sources_1",
        "launch_runs synth_1 -jobs %d" % jobs,
        "wait_on_run [get_runs synth_1]",
        "open_run synth_1",
        # the accelerator placeholder: kept as a hierarchy, black box after implementation
        "set rp [get_cells -hier -filter {ORIG_REF_NAME == %s || REF_NAME == %s}]"
        % (CORE_MODULE, CORE_MODULE),
        "set_property DONT_TOUCH true $rp",
        # the shell left of INT column X%d (below the PS and next to its fabric interface)
        "proc sites_by_int_x {type cmp x} {",
        "  set res {}",
        "  foreach s [get_sites -filter \"SITE_TYPE =~ $type\"] {",
        "    regexp {_X(\\d+)Y} [get_tiles -of $s] -> tx",
        "    if {[expr $tx $cmp $x] && $tx >= %d} {lappend res $s}" % shell_x0,
        "  }",
        "  return $res",
        "}",
        "proc site_range {sites prefix} {",
        "  set xs {}; set ys {}",
        "  foreach s $sites {",
        "    if {[regexp \"^${prefix}_X(\\\\d+)Y(\\\\d+)$\" $s -> x y]} {lappend xs $x; lappend ys $y}",
        "  }",
        "  if {[llength $xs] == 0} {return {}}",
        "  set xs [lsort -integer $xs]; set ys [lsort -integer $ys]",
        "  return \"${prefix}_X[lindex $xs 0]Y[lindex $ys 0]:${prefix}_X[lindex $xs end]Y[lindex $ys end]\"",
        "}",
        "set sh_ranges {}",
        "foreach {type prefix} {SLICE* SLICE DSP48E2 DSP48E2 RAMBFIFO36 RAMB36 RAMB18* RAMB18} {",
        "  set r [site_range [sites_by_int_x $type < %d] $prefix]" % ps_x,
        "  if {$r != {}} {lappend sh_ranges $r}",
        "}",
        "puts \"shell region: $sh_ranges\"",
        "create_pblock pb_shell",
        "set rpname [get_property NAME $rp]",
        "set sh_cells {}",
        "foreach c [get_cells -hier -filter {IS_PRIMITIVE && REF_NAME != PS8 && REF_NAME !~ BUFG*}] {",
        "  if {![string match \"$rpname/*\" $c]} {lappend sh_cells $c}",
        "}",
        "add_cells_to_pblock pb_shell $sh_cells",
        "resize_pblock pb_shell -add $sh_ranges",
        "set_property CONTAIN_ROUTING true [get_pblocks pb_shell]",
        "set_property EXCLUDE_PLACEMENT true [get_pblocks pb_shell]",
        "opt_design",
        "place_design",
        "route_design",
        "report_route_status -file %s/shell_route_status.rpt" % shell_dir,
        "report_timing_summary -file %s/shell_timing.rpt" % shell_dir,
        "report_utilization -file %s/shell_utilization.rpt" % shell_dir,
        "update_design -cell $rp -black_box",
        # boundary nets are routed from scratch when the accelerator is inserted
        "route_design -unroute -nets [get_nets -of [get_pins $rpname/*] -filter {TYPE != GLOBAL_CLOCK}]",
        "lock_design -level routing",
        "write_checkpoint -force %s/shell_routed.dcp" % shell_dir,
        "set f [open %s/rp_cell.txt w]; puts $f [get_property NAME $rp]; close $f" % shell_dir,
        "foreach hwh [glob -nocomplain %s/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh "
        "%s/finn_zynq_link.srcs/sources_1/bd/top/hw_handoff/top.hwh] {file copy -force $hwh %s/top.hwh}"
        % (shell_dir, shell_dir, shell_dir),
    ]
    return "\n".join(t) + "\n"


def build_shell(board, part, clk_ns, ports, shell_lib, jobs=8, shell_x0=0):
    """Build (or reuse) the pre-implemented shell. Returns (shell_dir, result dict).
    shell_x0: lowest INT column of the shell region (0: all columns left of the strip)."""
    key = shell_key(board, part, clk_ns, ports, shell_x0)
    shell_dir = os.path.join(shell_lib, key)
    done = os.path.join(shell_dir, "shell_routed.dcp")
    res = {"shell": key, "shell_dir": shell_dir}
    if os.path.isfile(done) and os.path.isfile(os.path.join(shell_dir, "top.hwh")):
        res.update(status="cached", shell_s=0.0)
        return shell_dir, res
    if os.path.isdir(shell_dir):
        shutil.rmtree(shell_dir)
    os.makedirs(shell_dir)
    src = os.path.join(shell_dir, "shell_sources.v")
    with open(src, "w") as f:
        f.write("// generated by finn.util.dynarapid.shell\n")
        f.write(core_verilog(ports))
        for d in ports:
            f.write(bridge_verilog(d))
    with open(os.path.join(shell_dir, "ports.json"), "w") as f:
        json.dump(ports, f, indent=2)
    tcl = os.path.join(shell_dir, "shell.tcl")
    with open(tcl, "w") as f:
        f.write(shell_tcl(board, part, clk_ns, ports, shell_dir, src, jobs, shell_x0))
    rc, res["shell_s"] = run_vivado(tcl, os.path.join(shell_dir, "shell.log"), shell_dir)
    ok = rc == 0 and os.path.isfile(done) and os.path.isfile(os.path.join(shell_dir, "top.hwh"))
    res["status"] = "built" if ok else "shell_failed"
    # the project (IP output products, runs) is not needed any more
    if ok:
        for sub in ("finn_zynq_link.runs", "finn_zynq_link.cache", "finn_zynq_link.ip_user_files"):
            shutil.rmtree(os.path.join(shell_dir, sub), ignore_errors=True)
    return shell_dir, res


def assemble_tcl(
    shell_dir,
    accel_dcp,
    out_dir,
    bitfile,
    threads=16,
    keep_dcp=False,
    reports="full",
    trigger=None,
    unfix_static="after_read",
):
    """Fill the shell's accelerator cell with the DynaRapid-routed accelerator, route the
    remaining (boundary and clock) nets and write the bitstream. trigger: the script opens the
    shell right away and then waits for this file ("go": accelerator ready, else abort), so
    that Vivado's start and the shell load overlap with the accelerator build."""
    rp = open(os.path.join(shell_dir, "rp_cell.txt")).read().strip()
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        "proc stamp {name} {global t0; puts \"STAMP $name [expr ([clock milliseconds] - $t0) / 1000.0]\"}",
        "open_checkpoint %s/shell_routed.dcp" % shell_dir,
        "stamp open_shell",
    ]
    # the shell's routing is locked, its static (VCC/GND) nets included; the accelerator's
    # static pins join these nets, so the router must be able to change them. unfix_static:
    # "shell" = only the shell's static nets, before the accelerator is read (fast: small
    # design; the island flow's stitcher unlocks the accelerator's own static routing),
    # "after_read" = all static nets after reading the accelerator (DynaRapid: relocated
    # components' static routing comes in fixed; slow on large designs)
    unfix = "set_property IS_ROUTE_FIXED 0 [get_nets -hier -quiet -filter {TYPE == POWER || TYPE == GROUND}]"
    if unfix_static == "shell":
        t.append(unfix)
    if trigger is not None:
        t += [
            "while {![file exists %s]} {after 200}" % trigger,
            "after 200",
            "set f [open %s]; set go [string trim [read $f]]; close $f" % trigger,
            'if {$go != "go"} {puts "ABORT: accelerator not built"; exit 1}',
            "set t0 [clock milliseconds]",
            "stamp wait",
        ]
    t += [
        "read_checkpoint -cell %s %s" % (rp, accel_dcp),
        "stamp read_accel",
    ]
    if unfix_static == "after_read":
        t.append(unfix)
    t += [
        "route_design",
        "stamp route",
    ]
    if reports == "min":
        # bitstream first (the result); the reports follow in the same session
        t += [
            "write_bitstream -force -no_partial_bitfile %s" % bitfile,
            "stamp bitstream",
        ]
    t += [
        "report_route_status -file %s/route_status.rpt" % out_dir,
        "report_timing_summary -file %s/timing_summary.rpt" % out_dir,
        # the same hierarchical report as the regular Zynq flow (post-synthesis resources)
        "report_utilization -hierarchical -hierarchical_depth 4 -format xml -file %s/synth_report.xml"
        % out_dir,
    ]
    if reports == "full":
        t += [
            "report_utilization -file %s/utilization.rpt" % out_dir,
            # per-cell resources in the format of FINN's synthesis report (post_synth_res)
            "report_utilization -hierarchical -hierarchical_depth 6 -format xml -file %s/utilization.xml"
            % out_dir,
        ]
    t.append("stamp reports")
    if reports != "min":
        t += [
            "write_bitstream -force -no_partial_bitfile %s" % bitfile,
            "stamp bitstream",
        ]
    if keep_dcp:
        t.append("write_checkpoint -force %s/final_routed.dcp" % out_dir)
    return "\n".join(t) + "\n"