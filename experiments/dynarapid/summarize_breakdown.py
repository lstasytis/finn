"""Per-step wall-time breakdown of the timed builds (all flows), as markdown tables.

Sources (all timed one build at a time, cached shells for the island flow):
  island flow, U55C Vivado-only shell: $D/rwu/vshell_timing/<model>/   (rwislands_zynq.json stamps)
  global Vivado, same U55C shell:      $D/rwu/vshell_timing/<model>_vivado/ (bitfile_experiment.json)
  Vitis platform flow (U55C):          the v++ link projects of run_u55c_timing.sh's bitfile runs
  FINN Vivado ZynqBuild (ZCU104):      $D/rwi/timing/<model>_vivado/
  island flow (ZCU104):                $D/rwi/timing/<model>_islands_zprofile/

Usage: python summarize_breakdown.py [> tables.md]
"""

import datetime
import json
import os
import re
import sys

D = os.environ.get("FINN_BUILD_DIR", "/home/lstasytis/finn/build/finn_build")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_u55c import link_info  # noqa: E402


def fmt(v):
    return "-" if v is None else "%d" % round(v)


def table(title, cols, rows, note=None):
    out = ["#### " + title, ""]
    out.append("| model | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for name, vals in rows:
        out.append("| %s | " % name + " | ".join(fmt(v) for v in vals) + " |")
    if note:
        out += ["", note]
    return "\n".join(out) + "\n"


def island_row(out_dir, wall):
    """Island flow: FINN preparation (partitions, IODMA HLS), node synthesis, floorplan, island
    P&R, stitch, assembly (read accelerator, route (+ hold repair, post-route phys_opt),
    bitstream, reports). The assembly Vivado opens the shell while the islands are built."""
    r = json.load(open(os.path.join(out_dir, "islands", "rwislands_zynq.json")))
    st, a = r["stamps"], r.get("assembly_stamps", {})
    total = r["total_s"]
    # (route_nets/check: interactive routing and its check, FINN_RWI_ASM_ROUTE=incremental)
    seq = ["read_accel", "route_nets", "check", "route", "hold_repair", "post_route_phys_opt", "bitstream", "reports"]
    prev, asm = 0.0, {}
    for k in seq:
        if k in a:
            asm[k] = a[k] - prev
            prev = a[k]
    vals = [
        wall - total,
        st["synth"],
        st["floorplan"] - st["shell"],
        st["islands"] - st["floorplan"],
        st["stitch"] - st["islands"],
        asm.get("read_accel"),
        sum(asm.get(k, 0) for k in ("route_nets", "check", "route", "hold_repair", "post_route_phys_opt")),
        asm.get("bitstream"),
        asm.get("reports"),
        wall,
    ]
    return vals, len(r.get("islands", {}))


ISLAND_COLS = [
    "FINN prep",
    "node synth (parallel)",
    "floorplan",
    "island P&R (parallel)",
    "stitch (RapidWright)",
    "asm: read accel",
    "asm: route",
    "asm: bitstream",
    "asm: reports",
    "total",
]


def vivado_row(out_dir, wall):
    """Global Vivado (FINN's MakeZYNQProject flow): FINN preparation (HLS, stitched IPs),
    block-design / out-of-context IP synthesis, top synthesis, opt, place, phys_opt, route,
    (post-route phys_opt), bitstream, other (reports, checkpoints, link_design)."""
    r = json.load(open(os.path.join(out_dir, "bitfile_experiment.json")))
    stages = r.get("stages", [])
    proj = sum(s["s"] for s in stages if s["stage"] == "MakeZYNQProject")
    prep = wall - proj
    synth = r.get("synth_1_synth_design_s", 0)
    impl = {k[len("impl_1_") : -2]: v for k, v in r.items() if k.startswith("impl_1_") and k.endswith("_s")}
    phys = impl.get("phys_opt_design", 0)
    named = ["opt_design", "place_design", "phys_opt_design", "route_design", "write_bitstream"]
    other = sum(v for k, v in impl.items() if k not in named)
    ooc = proj - synth - sum(impl.values())
    # post-route phys_opt is folded into phys_opt_design by Vivado's step timing (same command)
    vals = [prep, ooc, synth, impl.get("opt_design"), impl.get("place_design"), phys,
            impl.get("route_design"), impl.get("write_bitstream"), other, wall]
    return vals


VIVADO_COLS = [
    "FINN prep (HLS, stitched IPs)",
    "BD + OOC IP synth",
    "top synth",
    "opt",
    "place",
    "phys_opt",
    "route",
    "bitstream",
    "other (reports, ckpts)",
    "total",
]


def vitis_row(link_dir, run_start, wall):
    """Vitis flow: before the link (FINN: HLS, stitched IP, XO packaging), v++ link phases
    (vpl synth = platform IPs + kernel out-of-context synthesis, opt, place (+ phys_opt), route,
    bitstream), the rest of the link (v++ setup, xclbin), after the link."""
    li = link_info(link_dir)
    start = os.path.getmtime(os.path.join(link_dir, "run_vitis_link.sh"))
    pre = start - run_start
    txt = open(os.path.join(link_dir, "v++_a.log"), errors="ignore").read()
    m = re.search(r"Total elapsed time: (\d+)h (\d+)m (\d+)s", txt)
    link_total = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None
    phases = [li.get(k) for k in ("synth_s", "opt_s", "place_s", "route_s", "bitstream_s")]
    rest = link_total - sum(p or 0 for p in phases) if link_total else None
    post = wall - pre - (link_total or 0)
    return [pre] + phases + [rest, post, wall]


VITIS_COLS = [
    "before link (HLS, stitched IP, XO)",
    "vpl synth (platform + kernel)",
    "opt",
    "place (+phys_opt)",
    "route",
    "bitstream",
    "rest of link",
    "after link",
    "total",
]


def runs(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    parts = []
    T = os.path.join(D, "rwu", "vshell_timing")
    v3 = os.path.join(HERE, "data", "u55c_vivado_shell_timing_runs_v3.jsonl")
    isl = {r["model"]: r for r in runs(v3)}
    base = {r["model"]: r for r in runs(os.path.join(HERE, "data", "u55c_vivado_shell_baseline_runs.jsonl"))}
    names = {"tfc-w1a1": "TFC", "vgg10": "VGG10", "mnv1": "MobileNet (U250 folding)"}

    rows = []
    for m in ("tfc-w1a1", "vgg10", "mnv1"):
        if m in isl:
            vals, k = island_row(os.path.join(T, m), isl[m]["wall_s"])
            rows.append(("%s (%d islands)" % (names[m], k), vals))
    parts.append(table("U55C, island flow before (2026-10-06), Vivado-only shell (cached; built once in ~20 min)", ISLAND_COLS, rows,
                       "FINN prep = partitioning and IODMA HLS before the island flow. Assembly route includes "
                       "post-route phys_opt / hold repair where used (none ran here)."))

    rows = []
    for m in ("tfc-w1a1", "vgg10", "mnv1"):
        if m in base:
            rows.append((names[m], vivado_row(os.path.join(T, m + "_vivado"), base[m]["wall_s"])))
    parts.append(table("U55C, global Vivado flow, same Vivado-only shell (FINN MakeZYNQProject, Vivado defaults)",
                       VIVADO_COLS, rows,
                       "BD + OOC IP synth = block design, output products and out-of-context synthesis of all IPs "
                       "(shell IPs and FINN's stitched IPs, in parallel)."))

    vitis = [
        ("TFC", "vitis_link_proj_fejfakac", "2026-09-30 13:21:20", 4860),
        ("CNV", "vitis_link_proj_stfvla27", "2026-09-30 16:12:46", 5458),
        ("VGG10", "vitis_link_proj_ea5ci9ra", "2026-10-02 12:32:07", 7307),
        ("MobileNet (U250 folding)", "vitis_link_proj_t4qngpqg", "2026-10-02 17:42:42", 10756),
    ]
    rows = []
    for name, link, start, wall in vitis:
        L = os.path.join(D, link)
        if os.path.isdir(L):
            t0 = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp()
            rows.append((name, vitis_row(L, t0, wall)))
    parts.append(table("U55C, Vitis platform flow (xilinx_u55c_gen3x16_xdma_3_202210_1, v++ defaults)", VITIS_COLS, rows,
                       "The v++ link implements the whole dynamic region of the platform (HBM subsystem, "
                       "interconnect, SLR crossings) per model and writes its partial bitstream."))

    R = os.path.join(D, "rwi", "timing")
    zr = {}
    for r in runs(os.path.join(R, "runs.txt")):
        zr[(r["model"], r["mode"])] = r
    znames = {"tfc": "TFC", "cnv": "CNV", "cnv1": "CNV PE=SIMD=1", "vgg10": "VGG10", "mnv1": "MobileNet (ZCU104 folding)"}
    rows = []
    for m in ("tfc", "cnv", "cnv1", "vgg10", "mnv1"):
        d = os.path.join(R, m + "_vivado")
        if os.path.isfile(os.path.join(d, "bitfile_experiment.json")):
            wall = json.load(open(os.path.join(d, "bitfile_experiment.json")))["total_s"]
            rows.append((znames[m], vivado_row(d, wall)))
    parts.append(table("ZCU104, FINN Vivado ZynqBuild (synth Flow_PerfOptimized_high, impl Performance_ExtraTimingOpt)",
                       VIVADO_COLS, rows, "phys_opt includes the post-route phys_opt pass."))

    rows = []
    for m in ("tfc", "cnv", "cnv1", "vgg10", "mnv1"):
        d = os.path.join(R, m + "_islands_zprofile")
        if os.path.isfile(os.path.join(d, "islands", "rwislands_zynq.json")):
            wall = json.load(open(os.path.join(d, "bitfile_experiment.json")))["total_s"]
            vals, k = island_row(d, wall)
            rows.append(("%s (%d islands)" % (znames[m], k), vals))
    parts.append(table("ZCU104, island flow before (2026-10-06), Zynq shell (cached), FINN's Zynq strategies", ISLAND_COLS, rows,
                       "Assembly route includes the baseline's post-route phys_opt."))

    # 2026-10-07 (v2): conflict-free assembly (contained shell + moat, URAM column pairs), partition
    # pins, reset pipeline, island hold margin, soft-preserve stitching, concurrent IODMA HLS
    rows = []
    for m in ("tfc", "cnv", "cnv1", "vgg10", "mnv1"):
        d = os.path.join(R, m + "_islands_v2")
        if os.path.isfile(os.path.join(d, "islands", "rwislands_zynq.json")):
            wall = json.load(open(os.path.join(d, "bitfile_experiment.json")))["total_s"]
            vals, k = island_row(d, wall)
            rows.append(("%s (%d islands)" % (znames[m], k), vals))
    parts.append(table("ZCU104, island flow v2 (2026-10-07), same shell settings and strategies", ISLAND_COLS, rows,
                       "Assembly route: full route_design (the baseline's directive) + post-route phys_opt only "
                       "when setup fails."))
    v4 = {r["model"]: r for r in runs(os.path.join(HERE, "data", "u55c_vivado_shell_timing_runs_v4.jsonl"))}
    rows = []
    for m in ("tfc-w1a1", "vgg10", "mnv1"):
        if m in v4:
            vals, k = island_row(os.path.join(T, m + "_islands"), v4[m]["wall_s"])
            rows.append(("%s (%d islands)" % (names[m], k), vals))
    parts.append(table("U55C, island flow v2 (2026-10-07), Vivado-only shell (cached)", ISLAND_COLS, rows))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
