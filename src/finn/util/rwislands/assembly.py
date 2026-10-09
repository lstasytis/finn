# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly of the island flow: the pre-implemented shell plus the stitched accelerator ->
bitstream (one Vivado run, started early: it opens the shell while the islands are built and
waits for a trigger file).

After read_checkpoint -cell almost everything is routed: the islands inside, the stitcher
between them. Left are the shell/accelerator boundary nets, the accelerator's clock loads, the
static (VCC/GND) nets merged from both sides (the merge drops part of the shell's static
routing) and nets the stitcher gave up on. A full route_design initializes the router for the
whole design (VGG10 on the xczu7ev: RT build 35-53 s, router init incl. the ILP clock placer and
a full timing update 50-90 s) before it touches a net. Its real cost used to be elsewhere:
overlaps between the shell's routing and the islands (434 shell nets through the island columns)
and between adjacent islands (shared INT columns of URAM sites) made it rip up and re-route
thousands of nets (VGG10: 287 s); without them it routes ~400 nets (VGG10 ~150 s, TFC 20 s).

incremental (opt-in, FINN_RWI_ASM_ROUTE=incremental): the interactive router on exactly the nets
that need it (route_design -nets), then a check; if any net is still unrouted or in conflict,
or setup or hold fail after two timing-driven re-routes of the hold-failing nets, the full
route_design runs as before. It saves little (its init still places the clock) and fails on
nets it cannot finish, since it may not rip up unlisted nets (VGG10: one net, after 170 s)."""

import os

from finn.util.rwislands.profiles import profile

# nets that are not fully and legally routed, from report_route_status -list_all_nets (seconds
# on a 1M-net design, unlike get_nets -hier -filter {ROUTE_STATUS ...}, which takes minutes)
_NETS_NEEDING_ROUTE = r"""
proc nets_needing_route {rpt} {
    report_route_status -list_all_nets -file $rpt
    set f [open $rpt]
    set sec ""
    set names {}
    while {[gets $f line] >= 0} {
        if {[regexp {^(\S.*):\s*$} $line -> h]} {
            set sec $h
            continue
        }
        if {$sec in {"Unrouted Nets" "Partially Routed Nets" "Nets with Routing or Site Pin Conflicts" "Nets with Antennas or Islands"}} {
            if {[regexp {^    (\S+)$} $line -> n]} {lappend names $n}
        }
    }
    close $f
    return [get_nets -quiet [lsort -unique $names]]
}
proc route_errors {rpt} {
    report_route_status -file $rpt
    set f [open $rpt]; set txt [read $f]; close $f
    if {[regexp {# of nets with routing errors\.+ :\s+(\d+)} $txt -> n]} {return $n}
    return -1
}
proc hold_nets {} {
    get_nets -quiet -of [get_timing_paths -quiet -hold -slack_lesser_than 0 -max_paths 10000 -nworst 1] -filter {TYPE != GLOBAL_CLOCK}
}
proc worst_slack {kind} {
    set p [get_timing_paths -quiet -$kind -max_paths 1 -nworst 1]
    if {![llength $p]} {return 1e9}
    return [get_property SLACK $p]
}
"""


def assemble_tcl(shell_dir, accel_dcp, out_dir, bitfile, part, threads=16, trigger=None, incremental=False):
    """Assembly script. part selects the baseline's route / post-route phys_opt (profiles);
    incremental: interactive routing of the remaining nets first (see the module doc)."""
    steps = profile(part)
    route_cmd, post_route = steps["route"], steps["post_route_phys_opt"]
    rp = open(os.path.join(shell_dir, "rp_cell.txt")).read().strip()
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        'proc stamp {name} {global t0; puts "STAMP $name [expr ([clock milliseconds] - $t0) / 1000.0]"}',
        _NETS_NEEDING_ROUTE,
        "open_checkpoint %s/shell_routed.dcp" % shell_dir,
        "stamp open_shell",
        # the shell's routing is locked, its static nets included; the accelerator's static pins
        # join them (the black box is still empty: a cheap query). One segment name of each
        # static net is kept: after the merge they are the routable handles of VCC/GND (the
        # report's GLOBAL_LOGIC0/1 are no net names; the merge drops part of the shell's static
        # routing, e.g. ~4.6k GND pins of the PS on the xczu7ev)
        "set static [get_nets -hier -quiet -filter {TYPE == POWER || TYPE == GROUND}]",
        "set_property IS_ROUTE_FIXED 0 $static",
        "set static_names {}",
        "foreach ty {POWER GROUND} {",
        "  set n [lindex [filter $static \"TYPE == $ty\"] 0]",
        "  if {$n != {}} {lappend static_names [get_property NAME $n]}",
        "}",
    ]
    if trigger is not None:
        t += [
            "while {![file exists %s]} {after 200}" % trigger,
            "after 200",
            # "go" (accelerator ready), "go full" (ready, but use the full router: the stitcher
            # left pins unrouted, which the interactive router may not be able to reach), or abort
            "set f [open %s]; set go [string trim [read $f]]; close $f" % trigger,
            'if {[lindex $go 0] != "go"} {puts "ABORT: accelerator not built"; exit 1}',
            'set incremental [expr {%d && [lindex $go 1] != "full"}]' % int(incremental),
            "set t0 [clock milliseconds]",
            "stamp wait",
        ]
    t += [
        "read_checkpoint -cell %s %s" % (rp, accel_dcp),
        "stamp read_accel",
        # nets the stitcher ripped up and re-routed without timing (to reach a boxed-in pin): routed
        # again here, timing-driven, by route_design
        "set rrf %s" % os.path.join(os.path.dirname(accel_dcp), "rerouted_nets.txt"),
        "if {[file exists $rrf]} {",
        "  set f [open $rrf]; set rr {}",
        "  foreach n [split [string trim [read $f]] \"\\n\"] {if {$n != {}} {lappend rr %s/$n}}" % rp,
        "  close $f",
        "  set rrn [get_nets -quiet $rr]",
        '  puts "ASSEMBLY stitch_rerouted [llength $rr] nets, [llength $rrn] found"',
        "  if {[llength $rrn]} {set_property IS_ROUTE_FIXED 0 $rrn; route_design -unroute -nets $rrn}",
        "}",
        # the shell's reset reaches each island's first reset register over one long wire (U55C
        # VGG10 at 200 MHz: 6.2 ns); the reset is held for many cycles and the accelerator only
        # starts when the host programs it, so that path gets 3 cycles (the register slices are
        # not ready in reset, so islands leaving reset a cycle apart lose nothing)
        "set rq [get_cells -quiet %s/island_*/rst_q0_reg]" % rp,
        "if {[llength $rq]} {set_multicycle_path -setup 3 -end -to $rq; set_multicycle_path -hold 2 -end -to $rq}",
        "set full 1",
    ]
    if trigger is None:
        t.append("set incremental %d" % int(incremental))
    if incremental:
        t += [
            "if {$incremental} {",
        ]
        t += [
            "set nr [concat [get_nets -quiet $static_names] [nets_needing_route %s/route_list.rpt]]" % out_dir,
            'puts "ASSEMBLY nets_to_route [llength $nr]"',
            # (the interactive router gives up by itself, with an error, when it cannot resolve an
            # overlap with a net it may not rip up: VGG10 with conflicting islands, after ~5 min)
            "set ok 1",
            "if {[llength $nr] && [catch {route_design -nets $nr} msg]} {",
            '  puts "ASSEMBLY interactive route failed: $msg"',
            "  set ok 0",
            "}",
            "stamp route_nets",
            "set err [expr {$ok ? [route_errors %s/route_status_incr.rpt] : -1}]" % out_dir,
            "set wns [expr {$err == 0 ? [worst_slack setup] : -1}]",
            "set whs [expr {$err == 0 ? [worst_slack hold] : -1}]",
            'puts "ASSEMBLY incremental route_errors $err wns $wns whs $whs"',
            # hold: the interactive router does not fix hold; the islands are routed with a hold
            # margin (their clock tree is re-routed here), the rest is re-routed timing-driven
            "for {set i 0} {$i < 2 && $err == 0 && $whs < 0} {incr i} {",
            "  set hn [hold_nets]",
            '  puts "ASSEMBLY hold_reroute [llength $hn] nets"',
            # (locked shell nets included: their skew changes with the accelerator's clock loads)
            "  set_property IS_ROUTE_FIXED 0 $hn",
            "  route_design -unroute -nets $hn",
            "  if {[catch {route_design -nets $hn -auto_delay}]} {set err -1; break}",
            "  set err [route_errors %s/route_status_incr.rpt]" % out_dir,
            "  set wns [worst_slack setup]",
            "  set whs [worst_slack hold]",
            '  puts "ASSEMBLY after hold_reroute route_errors $err wns $wns whs $whs"',
            "}",
            "set full [expr {$err != 0 || $wns < 0 || $whs < 0}]",
            "stamp check",
            "}",
        ]
    t += [
        "if {$full} {",
        '  puts "ASSEMBLY full_route"',
        "  %s" % route_cmd,
        "  stamp route",
        # hold repair: the shell is routed and locked with its own clock tree; with the
        # accelerator's loads on the clock the skew changes and a few locked shell paths can
        # miss hold. Their nets are unlocked, unrouted and routed again (up to three passes,
        # each only while violations remain)
        "  for {set i 0} {$i < 3} {incr i} {",
        "    set hp [get_timing_paths -quiet -hold -slack_lesser_than 0 -max_paths 10000 -nworst 1]",
        "    if {![llength $hp]} {break}",
        "    set hn [get_nets -quiet -of $hp -filter {TYPE != GLOBAL_CLOCK}]",
        '    puts "HOLD_REPAIR [llength $hp] paths [llength $hn] nets"',
        "    set_property IS_ROUTE_FIXED 0 $hn",
        "    route_design -unroute -nets $hn",
        "    %s" % route_cmd,
        "    stamp hold_repair",
        "  }",
        "}",
    ]
    if post_route:
        # the baseline's post-route phys_opt; it only acts on failing paths, so it is skipped
        # when setup is met (it would only re-time the design, ~30 s on VGG10)
        t += [
            "if {[worst_slack setup] < 0} {%s}" % post_route,
            "stamp post_route_phys_opt",
        ]
    t += [
        "write_bitstream -force -no_partial_bitfile %s" % bitfile,
        "stamp bitstream",
        "report_route_status -file %s/route_status.rpt" % out_dir,
        "report_timing_summary -file %s/timing_summary.rpt" % out_dir,
        # the same hierarchical report as the regular Zynq flow (post-synthesis resources)
        "report_utilization -hierarchical -hierarchical_depth 4 -format xml -file %s/synth_report.xml"
        % out_dir,
        "stamp reports",
    ]
    return "\n".join(t) + "\n"


def final_tcl(full_dcp, rp, out_dir, bitfile, part, threads=16):
    """Assembly of a design that RapidWright routed completely (IslandStitcher --shell): load it,
    write the bitstream, report. No route_design: on a 1M-net design its fixed cost alone (RT
    build, ILP clock placement, two full timing updates, hold-fix passes) is ~10 min (U55C VGG10
    8x, 2026-10-09) with only ~800 boundary/clock/static nets left to route.

    The reports run after the bitstream; when they show a problem (routing errors, setup or hold
    missed: RWRoute is not timing-driven here and does not fix hold) the regular route_design
    with its hold repair runs as a fallback and the bitstream is written again, so the result
    is never a bitstream that is known to be wrong."""
    load = full_dcp.replace(".dcp", "_load.tcl")
    steps = profile(part)
    route_cmd = steps["route"]
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        'proc stamp {name} {global t0; puts "STAMP $name [expr ([clock milliseconds] - $t0) / 1000.0]"}',
        _NETS_NEEDING_ROUTE,
        # encrypted shell IP (XDMA, SmartConnect, ...): RapidWright's load script reads its .edn
        # netlists, the checkpoint and links
        ("source %s" % load) if os.path.isfile(load) else ("open_checkpoint %s" % full_dcp),
        "stamp open",
        # (see assemble_tcl: the reset path to the islands is a multicycle path)
        "set rq [get_cells -quiet %s/island_*/rst_q0_reg]" % rp,
        "if {[llength $rq]} {set_multicycle_path -setup 3 -end -to $rq; set_multicycle_path -hold 2 -end -to $rq}",
        "set ok [expr {![catch {write_bitstream -force -no_partial_bitfile %s} msg]}]" % bitfile,
        'if {!$ok} {puts "FINAL bitstream failed: $msg"}',
        "stamp bitstream",
        "set err [route_errors %s/route_status.rpt]" % out_dir,
        "set wns [worst_slack setup]",
        "set whs [worst_slack hold]",
        'puts "FINAL route_errors $err wns $wns whs $whs"',
        "stamp check",
        "if {!$ok || $err != 0 || $wns < 0 || $whs < 0} {",
        '  puts "FINAL fallback route_design"',
        "  set_property IS_ROUTE_FIXED 0 [get_nets -hier -quiet -filter {TYPE == POWER || TYPE == GROUND}]",
        "  %s" % route_cmd,
        "  stamp route",
        "  for {set i 0} {$i < 3} {incr i} {",
        "    set hp [get_timing_paths -quiet -hold -slack_lesser_than 0 -max_paths 10000 -nworst 1]",
        "    if {![llength $hp]} {break}",
        "    set hn [get_nets -quiet -of $hp -filter {TYPE != GLOBAL_CLOCK}]",
        '    puts "HOLD_REPAIR [llength $hp] paths [llength $hn] nets"',
        "    set_property IS_ROUTE_FIXED 0 $hn",
        "    route_design -unroute -nets $hn",
        "    %s" % route_cmd,
        "    stamp hold_repair",
        "  }",
        "  write_bitstream -force -no_partial_bitfile %s" % bitfile,
        "  stamp bitstream_fallback",
        "  report_route_status -file %s/route_status.rpt" % out_dir,
        "}",
        "report_timing_summary -file %s/timing_summary.rpt" % out_dir,
        "report_utilization -hierarchical -hierarchical_depth 4 -format xml -file %s/synth_report.xml"
        % out_dir,
        "stamp reports",
    ]
    return "\n".join(t) + "\n"
