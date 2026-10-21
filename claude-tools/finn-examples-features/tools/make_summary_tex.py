"""Write summary/summary.tex: result tables (FIFO sizing, folding, DWC) from the campaign data and the two
lists of fixes (FINN compiler bugs, bugs in our transformations) with +/- lines and files from git.

    python make_summary_tex.py [--variant 2] [--sizes small,full]

Re-run at any time; the tables show the campaign's current state (not-yet-run cells are marked).
"""
import argparse
import datetime
import os
import subprocess

from run_campaign import LOGDIR, M, TIER1, TIER2, TIER_TRANSFORMER, build_queue, junit_result
from status_table import MULTI, job_metrics, latest, load, split_param

OUT = os.path.join(M, "claude-tools/finn-examples-features/summary/summary.tex")
VARIANT = "2"

# (title, issue, what, branch, commits)
FINN_BUGS = [
    ("MVU output queue overflow at SF = 1", "007, 012",
     "mvu\\_vvu\\_axi's output queue was sized for an outdated core latency (non-power-of-2 SIMD and non-INT8 DSP58 "
     "cores are deeper) and a registered output lock let one more item in: under output backpressure results were "
     "overwritten and read back as X (espcn: whole pixels zeroed). The RTL's own assertion fired unseen.",
     "feature/dev/mvu-output-queue", ["55a0edb83"]),
    ("Tiled MVAU expected cycles", "009",
     "get\\_exp\\_cycles multiplied by TH (bert estimate 4x pessimistic); upstream's c15fb8a9a drops it, plus the "
     "two per-group stalls of the tiled MVU (activation replay, output reorder) modelled from the RTL.",
     "feature/dev/mvau-exp-cycles", ["187efdac7", "cbd53df4c"]),
    ("StreamingSplit/Concat at 5 cycles per word", "010, 012",
     "the generated HLS top was not pipelined (interval 5) while the estimate said 1 word/cycle; a dataflow pragma "
     "makes them 1 word/cycle (and smaller).", "feature/dev/split-concat-dataflow", ["bcb838bbb"]),
    ("GlobalAccPool on non-square inputs", "011",
     "AccPool\\_Batch assumed a square image: the transformer token pool ([1,1,64]) summed one token only.",
     "fix/011-up (upstream PR)", ["d87bf52e3"]),
    ("Requant\\_rtl with unsigned inputs", "011",
     "the RTL reads its input as signed but specialization accepted any integer type: UINT8 values >= 128 became "
     "negative (radioml, vision).", "fix/011-up (upstream PR)", ["aff7232c9"]),
    ("Per-frame HLS overhead in MVAU\\_hls / LabelSelect\\_hls estimates", "002",
     "the HLS top function is not frame-pipelined: +7 / +4 cycles per frame missing from get\\_exp\\_cycles "
     "(tfc/MLP rtlsim at 0.53x of the estimate).", "feature/dev/hls-exp-cycles-overhead", ["2169d7b6d"]),
    ("Phantom URAM estimate of embedded MVAU weights", "005",
     "internal\\_embedded weights were costed as URAM, so the folding budget saw URAM use that does not exist.",
     "feature/dev/set-folding-optimizer", ["48a274bdb"]),
    ("DWC cycles in elements instead of words", "--",
     "StreamingDataWidthConverter estimated cycles per frame in elements, not stream words.",
     "feature/dev/set-folding-optimizer", ["474ea4474"]),
    ("Missing platform resource counts", "--",
     "KV260\\_SOM, VCK190 and V80 had no resource counts, so no budget for folding on them.",
     "feature/dev/set-folding-optimizer", ["023c304d9"]),
]
OUR_BUGS = [
    ("FIFO: dynamic MVAUs not characterized", "001",
     "attention matmuls (mem\\_mode dynamic) fell back to rtlsim characterization, which never fed the second input "
     "(timeout); tree model + in1 TAV added.", "feature/dev/analytical-fifo-sizing", ["1c7d93f2e", "19ed61ff7"]),
    ("FIFO: balanced foldings 2-6 % below the estimate", "006",
     "pacer-bound chains and a refill floor for pacer-driving consumers.",
     "feature/dev/analytical-fifo-sizing", ["ba7cfe45a"]),
    ("FIFO: tiled MVAU tree model", "009",
     "the replay stall from the shared stall function; reorder-stall foldings declined to rtlsim.",
     "feature/dev/analytical-fifo-sizing", ["e8c2cc992"]),
    ("FIFO: MVAU\\_rtl phase lead", "010",
     "TAVs opening with outputs before the first read made residual shortcuts 63 words short (0.965x).",
     "feature/dev/analytical-fifo-sizing", ["c74d17dc5"]),
    ("FIFO: no node latency", "013",
     "pipeline latency, InnerShuffle page and dynamic-MVAU matrix hold missing: shortcuts ~300 words short at fast "
     "foldings (0.73x).", "feature/dev/analytical-fifo-sizing", ["5e25783c8", "86942e9ee"]),
    ("FIFO: residual skip-edge deadlock", "015",
     "skip FIFOs below the other branch's input window (radioml 20 words for a 48-word token): deadlock; join "
     "window floor (deadlock / paced backlog + 25 %).", "feature/dev/analytical-fifo-sizing",
     ["729e7d033", "b0fbc933e", "57aaa6e98", "8aeb70760", "99de01f40", "4655649a7", "39dac9a18"]),
    ("FIFO: Split/Concat tree models and rtlsim harness", "016",
     "tree models hard-coded interval 5 (wrong with the dataflow pragma); node rtlsim drove only stream 0 "
     "(Split crashed, Concat timed out).", "feature/dev/analytical-fifo-sizing", ["af38b555c"]),
    ("Folding: SWG pairing in branching graphs, streamed weights", "003",
     "an SWG was paired with the next node instead of its consumer (resnet, yolov8); dynamic-MVAU weights resized.",
     "feature/dev/set-folding-optimizer", ["648bf6279", "4f2a7d5a7"]),
    ("Folding: DWC bottleneck missed in repair", "004",
     "unnamed DWC bottlenecks were never repaired (vgg10 1280 -> 1028 cycles).",
     "feature/dev/set-folding-optimizer", ["085be7d79"]),
    ("Folding: padding next to a reshape", "004", "padding next to a reshape broke InsertDWC.",
     "feature/dev/set-folding-optimizer", ["4410df870"]),
    ("Folding: maximize mode on small parts", "005", "maximize searched beyond the as-built budget.",
     "feature/dev/set-folding-optimizer", ["7c458eeb0"]),
    ("Folding: transformer / branching graphs folded slowly", "008",
     "every evaluation walked the whole graph (FIFO estimate, repair/polish re-scoring): bert 509 s -> 14 s per fold; "
     "partitions follow branches.", "feature/dev/set-folding-optimizer", ["4b09bdfd3"]),
    ("Folding: generic Shuffle not folded", "008", "the optimizer skipped Shuffle layers (naive folder folds them).",
     "feature/dev/set-folding-optimizer", ["81b625ae3"]),
    ("Folding: Shuffle SIMD without valid decomposition", "--",
     "probed SIMD values that raise or that a decomposition stage rejects were offered and not restored (bert).",
     "feature/dev/set-folding-optimizer", ["8e24e65ea", "081f70a4d"]),
    ("Folding: URAM for embedded constants off Versal", "--",
     "ram\\_style ultra offered for internal\\_embedded ElementwiseBinary/Lookup constants on non-Versal parts.",
     "feature/dev/set-folding-optimizer", ["57594bf0c"]),
    ("Folding: StreamingSplit/Concat SIMD never folded", "010", "they were in DO\\_NOT\\_FOLD (0.2x at maximize).",
     "feature/dev/set-folding-optimizer", ["a2d217bc6", "c38d441c8"]),
    ("Folding/DWC: padding rarely chosen", "014",
     "8x HLS penalty on (RTL) padding DWCs, polish could not reach padded SWG tuples, cheaper padded foldings "
     "rejected on peak utilization, conv PE never padded.", "feature/dev/set-folding-optimizer", ["a1caca9ed"]),
    ("Folding: Split -> Concat DWC heuristic crash", "014", "'Malformed model' for a Concat fed by a Split's second "
     "output (yolov8).", "feature/dev/set-folding-optimizer", ["2620ceeeb"]),
    ("DWC: InsertDWC datatype check and plain width conversions", "003",
     "compared the producer with the consumer's input 0 instead of the connected input; dynamic-MVAU weight "
     "conversions read as crops.", "feature/dev/generalized-datawidthconverter", ["44c8618d3", "0a3af1d37"]),
]


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def numstat(commits):
    files, plus, minus = {}, 0, 0
    for c in commits:
        out = subprocess.run(["git", "-C", M, "show", "--numstat", "--format=", c],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            a, d, f = line.split("\t")
            a, d = int(a), int(d)
            plus, minus = plus + a, minus + d
            fa, fd = files.get(f, (0, 0))
            files[f] = (fa + a, fd + d)
    return plus, minus, files


def fix_list(items):
    lines = ["\\begin{enumerate}"]
    for title, issue, what, branch, commits in items:
        plus, minus, files = numstat(commits)
        flist = "; ".join(f"\\texttt{{{tex(f)}}} (+{a}/$-${d})" for f, (a, d) in sorted(files.items()))
        title, what = title.replace("%", "\\%"), what.replace("%", "\\%")
        lines.append(
            f"\\item \\textbf{{{title}}} (issue {issue}). {what}\\\\\n"
            f"\\emph{{Fix:}} +{plus}/$-${minus} lines, branch \\texttt{{{tex(branch)}}}, "
            f"commit(s) \\texttt{{{' '.join(commits)}}}.\\\\\n{{\\footnotesize {flist}}}"
        )
    lines.append("\\end{enumerate}")
    return "\n".join(lines)


def result_of(job, state):
    jid = job["id"]
    if jid in state.get("running", {}):
        return "running"
    res = junit_result(os.path.join(LOGDIR, jid + ".xml"))
    if res is None:
        fin = state.get("finished", {}).get(jid, {}).get("result")
        return {"blocked": "blocked", "aborted": "aborted"}.get(fin, "pending" if fin is None else "killed")
    return res


MARK = {"passed": "\\checkmark", "failed": "$\\times$", "skipped": "skip", "running": "run",
        "pending": "--", "blocked": "blk", "killed": "--", "aborted": "--"}
RES = ("LUT", "BRAM_18K", "URAM", "DSP")
# colleagues' LiveFIFO result (best order): best total FIFO size [kB] and the interval it reaches [cycles],
# keyed by (model dir, model) of the matching row
LIVEFIFO = {
    ("kws", None): (0.4, 490), ("cybersecurity-mlp", None): (0.1, 70),
    ("bnn-pynq", "tfc-w1a1"): (0.2, 70), ("bnn-pynq", "tfc-w2a2"): (0.2, 70),
    ("bnn-pynq", "cnv-w1a1"): (2.7, 32774), ("bnn-pynq", "cnv-w2a2"): (10.1, 115206),
    ("vgg10-radioml", None): (2.2, 1027), ("mobilenet_v1", None): (22.1, 394272),
    ("resnet50", None): (202.7, 903174), ("transformer", "transformer_radioml"): (17.7, 5120),
    ("transformer", "transformer_vision"): (85.0, 81920), ("transformer", "transformer_language"): (84.5, 81920),
}
# the same folding: our build's estimated interval within this fraction of LiveFIFO's
SAME_FOLDING = 0.02


def livefifo_cells(job, state, live):
    """Size [rel.] and Intvl [degr.] of our analytical FIFO sizing (full) vs LiveFIFO's best, only where
    both size the same folding; else n/c with LiveFIFO's interval."""
    if live is None:
        return ["", ""]
    if job is None or result_of(job, state) not in ("passed", "failed"):
        return ["--", "--"]
    m = job_metrics(job) or {}
    est = load(os.path.join(m.get("output_dir", ""), "report", "estimate_network_performance.json")) or {}
    perf = load(os.path.join(m.get("output_dir", ""), "report", "rtlsim_performance.json")) or {}
    kb, ivl = live
    ours = est.get("max_cycles")
    if not ours or abs(ours / ivl - 1) > SAME_FOLDING:
        return ["n/c", "n/c (%d)" % ivl]
    size = "%.1fx" % (m["fifo_kbit"] / 8 / kb) if m.get("fifo_kbit") is not None else ""
    got = perf.get("steady_state_cycles") or perf.get("interval_cycles")
    degr = ("%.1f\\%%" % max(0.0, 100.0 * (got / ivl - 1))).replace(".0\\%", "\\%") if got else ""
    return [size, degr]
HEADERS = {"ratio": "rtlsim/est.", "tref": "rtlsim/ref.", "res": "res./ref.", "kbit": "FIFO [kB]",
           "pad": "pad.\\ DWCs", "tfifo": "time [s]", "tfold": "time [s]"}


def resources(output_dir):
    """Estimated LUT/BRAM_18K/URAM/DSP of a build without its FIFOs (FIFO sizing has its own table)."""
    rep = load(os.path.join(output_dir, "report", "estimate_layer_resources.json")) if output_dir else None
    if not rep:
        return None
    tot = dict.fromkeys(RES, 0.0)
    for layer, r in rep.items():
        if layer == "total" or layer.startswith("StreamingFIFO") or not isinstance(r, dict):
            continue
        for k in RES:
            tot[k] += float(r.get(k, 0) or 0)
    return tot


def reference_resources(job, jobs):
    """Resources of the committed folding built the same way (the model's FIFO test build: committed
    folding + its DWCs, FIFOs excluded); falls back to the estimate-only build (no DWCs), flagged."""
    fifo = next((j for j in jobs if j["model"] == job["model"] and j["param"] == job["param"]
                 and j["feature"] == "fifo" and j.get("size") == job.get("size")), None)
    m = job_metrics(fifo) if fifo else None
    r = resources(m.get("output_dir")) if m else None
    if r:
        return r, False
    board, model = split_param(job["param"])
    mtag = f"{job['model']}_{model}" if job["model"] in MULTI else job["model"]
    small = "small_" if job.get("size") == "small" else ""
    est = latest("folding" + VARIANT, f"est_{mtag}_{small}{board}_", "report/estimate_layer_resources.json")
    return (resources(os.path.dirname(os.path.dirname(est))) if est else None), True


def resource_ratio(new, ref, fallback):
    """Mean of new/committed over the resources the committed folding uses; resources only the new
    folding uses are named instead (no ratio exists)."""
    if not new or not ref:
        return ""
    ratios = [new[k] / ref[k] for k in RES if ref[k] > 0]
    extra = [k.split("_")[0] for k in RES if ref[k] == 0 and new[k] > 0]
    out = "%.2f" % (sum(ratios) / len(ratios)) if ratios else "--"
    if extra:
        out += " +" + "/".join(extra)
    return out + ("$^\\dagger$" if fallback else "")


def cells(job, state, fields, jobs):
    """Result mark followed by one cell per metric (empty if the job did not finish)."""
    if job is None:
        return ["n/a"] + [""] * len(fields)
    res = result_of(job, state)
    if res not in ("passed", "failed"):
        return [MARK.get(res, res)] + [""] * len(fields)
    m = job_metrics(job) or {}
    t = m.get("step_times_s") or {}
    out = [MARK[res]]
    for f in fields:
        if f == "ratio":
            out.append("%.3f" % m["rtlsim_vs_est"] if m.get("rtlsim_vs_est") is not None else "")
        elif f == "tref":
            ok = m.get("rtlsim_fps") and m.get("ref_est_fps")
            out.append("%.3f" % (m["rtlsim_fps"] / m["ref_est_fps"]) if ok else "")
        elif f == "res":
            ref, fallback = reference_resources(job, jobs)
            out.append(resource_ratio(resources(m.get("output_dir")), ref, fallback))
        elif f == "kbit":
            out.append("%.2f" % (m["fifo_kbit"] / 8) if m.get("fifo_kbit") is not None else "")
        elif f == "pad":
            out.append("%d" % m["pad_dwcs"] if m.get("pad_dwcs") is not None else "")
        elif f == "tfifo":
            out.append("%.0f" % t["step_set_fifo_depths"] if "step_set_fifo_depths" in t else "")
        elif f == "tfold":
            k = "step_target_fps_parallelization"
            out.append("%.0f" % t[k] if k in t else "")
    return out


def table(caption, label, groups, rows, legend, extra=None):
    """groups: [(group title, [metric keys])]; rows: [[model, nodes, est] + cells...]. Grid with a
    two-level header and alternating row shading."""
    head1, head2, col = ["", "", ""], ["model (board)", "\\#nodes", "est.\\ fps"], 4
    clines = []
    for title, fields in groups:
        width = 1 + len(fields)
        head1.append(f"\\multicolumn{{{width}}}{{c|}}{{\\textbf{{{title}}}}}")
        head2 += ["result"] + [HEADERS[f] for f in fields]
        clines.append(f"\\cline{{{col}-{col + width - 1}}}")
        col += width
    if extra:
        title, heads = extra
        head1.append(f"\\multicolumn{{{len(heads)}}}{{c|}}{{\\textbf{{{title}}}}}")
        head2 += heads
        clines.append(f"\\cline{{{col}-{col + len(heads) - 1}}}")
    spec = "|l|r|r|" + "|".join("c" + "r" * len(f) for _, f in groups) + "|" + (
        "r" * len(extra[1]) + "|" if extra else "")
    return "\n".join([
        "\\begin{landscape}\\begin{table}[p]\\centering\\scriptsize",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\rowcolors{3}{gray!12}{white}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{spec}}}\\hline",
        " & ".join(head1) + " \\\\" + "".join(clines),
        " & ".join(head2) + " \\\\\\hline\\hline",
        *[" & ".join(r) + " \\\\\\hline" for r in rows],
        "\\end{tabular}}",
        f"\\par\\medskip\\raggedright\\footnotesize\\textbf{{Columns.}} {legend}",
        "\\end{table}\\end{landscape}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="2")
    ap.add_argument("--sizes", default="small,full")
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    sizes = args.sizes.split(",")
    state = load(os.path.join(LOGDIR, "campaign_state.json")) or {}
    jobs = build_queue(args.variant, sizes)
    tiers = TIER1 + TIER2 + (TIER_TRANSFORMER if args.variant else [])

    def job(mdir, p, feature, size):
        return next((j for j in jobs if j["model"] == mdir and j["param"] == p
                     and j["feature"] == feature and j.get("size") == size), None)

    fifo_rows, fold_rows, dwc_rows = [], [], []
    FIFO_F, FOLD_F, FAW_F, DWC_F = (["ratio", "kbit", "tfifo"], ["tref", "res", "tfold"],
                                    ["tref", "res", "kbit", "tfold"], ["pad", "tref", "res"])
    for mdir, _, params in tiers:
        for p, *_ in params:
            board, model = split_param(p)
            name = tex(f"{model or mdir} ({board})")
            mtag = f"{mdir}_{model}" if mdir in MULTI else mdir
            est = layers = None
            for wt in ("fifo", "folding", "dwc"):
                est = est or load(latest(wt + args.variant, f"est_{mtag}_{board}_",
                                         "report/estimate_network_performance.json"))
                layers = layers or load(latest(wt + args.variant, f"est_{mtag}_{board}_",
                                               "report/estimate_layer_resources.json"))
            est_s = "%.1f" % est["estimated_throughput_fps"] if est else "--"
            nodes = str(len([k for k in layers if k != "total"])) if layers else "--"
            head = [name, nodes, est_s]
            row = list(head)
            for f in ("fifo", "fifo-rtlsim"):
                for sz in ("small", "full"):
                    row += cells(job(mdir, p, f, sz), state, FIFO_F, jobs)
            row += livefifo_cells(job(mdir, p, "fifo", "full"), state, LIVEFIFO.get((mdir, model)))
            fifo_rows.append(row)
            row = list(head)
            for sz in ("small", "full"):
                row += cells(job(mdir, p, "fold-target", sz), state, FOLD_F, jobs)
            row += cells(job(mdir, p, "fold-max", "full"), state, FOLD_F, jobs)
            row += cells(job(mdir, p, "fold-fifo", "full"), state, FAW_F, jobs)
            fold_rows.append(row)
            row = list(head)
            for sz in ("small", "full"):
                row += cells(job(mdir, p, "dwc", sz), state, DWC_F, jobs)
            dwc_rows.append(row)

    now = datetime.datetime.now()
    doc = [
        "\\documentclass[a4paper,10pt]{article}",
        "\\usepackage[margin=18mm]{geometry}\\usepackage{booktabs}\\usepackage{amssymb}\\usepackage[table]{xcolor}"
        "\\usepackage{graphicx}\\usepackage{pdflscape}\\usepackage[hidelinks]{hyperref}",
        "\\title{FINN feature tests: FIFO sizing, folding and DWC padding on the finn-examples models}",
        f"\\date{{Snapshot {now:%Y-%m-%d %H:%M} (campaign variant {args.variant}, new upstream base)}}",
        "\\begin{document}\\maketitle",
        "\\section*{Results}",
        "Each test resumes the convert-phase checkpoint of a cached estimate-only build with the committed folding, builds the stitched IP and checks it in stitched-IP rtlsim. Each table lists, per model, the result and the measured numbers of every test variant; the column definitions are given under each table.",
        table("FIFO sizing on the committed folding: the analytical sizer (ours) and the rtlsim-based heuristic",
              "tab:fifo", [("analytical, small", FIFO_F), ("analytical, full", FIFO_F),
                           ("rtlsim heuristic, small", FIFO_F), ("rtlsim heuristic, full", FIFO_F)],
              fifo_rows, "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{\\#nodes}: compute layers of the committed folding (before FIFO/DWC insertion). \\emph{est.\\ fps}: analytic throughput of the committed folding JSON (the reference). \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$ (its reference is the small committed folding); \\emph{full}: the committed folding. Step times are wall-clock under the campaign's load. " + "\\emph{rtlsim/est.}: stitched-IP rtlsim throughput divided by the "
              "committed folding's estimate (1.00 = the FIFOs reach the full throughput). \\emph{FIFO [kB]}: "
              "total storage of all inserted FIFOs (depth $\\times$ stream width, 1 kB = 1024 bytes). \\emph{time [s]}: time "
              "of \\texttt{step\\_set\\_fifo\\_depths}. \\emph{vs LiveFIFO best}: our analytical sizing (full) "
              "against the colleagues' LiveFIFO result (best order, their table): \\emph{Size [rel.]} = our total "
              "FIFO size / LiveFIFO's best size in kB, \\emph{Intvl [degr.]} = how much our stitched-IP rtlsim "
              "interval exceeds the interval LiveFIFO reaches (0\\% = none). Only comparable where both size the "
              "same folding (our estimated interval within 2\\% of theirs); n/c = their folding differs from the "
              "committed folding JSON (their interval in cycles in brackets); empty = model not in their table.",
              extra=("vs LiveFIFO best", ["Size [rel.]", "Intvl [degr.]"])),
        table("Folding optimizer (FIFOs sized analytically) vs.\\ the committed folding JSON", "tab:fold",
              [("target, small", FOLD_F), ("target, full", FOLD_F), ("maximize", FOLD_F),
               ("FIFO-aware (target)", FAW_F)], fold_rows,
              "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{\\#nodes}: compute layers of the committed folding (before FIFO/DWC insertion). \\emph{est.\\ fps}: analytic throughput of the committed folding JSON (the reference). \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$ (its reference is the small committed folding); \\emph{full}: the committed folding. Step times are wall-clock under the campaign's load. " + "\\emph{rtlsim/ref.}: stitched-IP rtlsim throughput of the build divided by the committed folding's estimated throughput (1.00 = same throughput as the provided JSON; maximize: the speed-up). \\emph{res./ref.}: estimated resources of the build divided by those of the committed folding built the same way (DWCs included, FIFOs excluded): mean of the LUT, BRAM\\_18K, URAM and DSP ratios over the resources the committed folding uses ($\\leq 1.00$ = same or fewer resources); a resource only the new folding uses is named after it (e.g.\\ +URAM); $^\\dagger$ = reference without DWCs (the committed folding's FIFO test has not finished yet). " + "\\emph{target}: optimizer at the committed folding's throughput; "
              "\\emph{maximize}: no target, maximum throughput within the device budget; \\emph{FIFO-aware}: "
              "target mode with \\texttt{enable\\_folding\\_fifo\\_heuristic} (optional, slower). "
              "\\emph{FIFO [kB]}: total FIFO storage of the build (1 kB = 1024 bytes). \\emph{time [s]}: time of "
              "\\texttt{step\\_target\\_fps\\_parallelization}."),
        table("DWC padding: optimizer with up to 4 channels of padding per layer at the committed throughput",
              "tab:dwc", [("small", DWC_F), ("full", DWC_F)], dwc_rows,
              "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{\\#nodes}: compute layers of the committed folding (before FIFO/DWC insertion). \\emph{est.\\ fps}: analytic throughput of the committed folding JSON (the reference). \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$ (its reference is the small committed folding); \\emph{full}: the committed folding. Step times are wall-clock under the campaign's load. " + "\\emph{rtlsim/ref.}: stitched-IP rtlsim throughput of the build divided by the committed folding's estimated throughput (1.00 = same throughput as the provided JSON; maximize: the speed-up). \\emph{res./ref.}: estimated resources of the build divided by those of the committed folding built the same way (DWCs included, FIFOs excluded): mean of the LUT, BRAM\\_18K, URAM and DSP ratios over the resources the committed folding uses ($\\leq 1.00$ = same or fewer resources); a resource only the new folding uses is named after it (e.g.\\ +URAM); $^\\dagger$ = reference without DWCs (the committed folding's FIFO test has not finished yet). " + "\\emph{pad.\\ DWCs}: padding/cropping data width converters in the build "
              "(skip: the padded search inserts none, no hardware build). A result below 1.00 rtlsim/ref.\\ is "
              "accepted if within 10 \\% and its resource saving is larger than the throughput loss."),
        "\\clearpage\\section*{Bugs fixed in the FINN compiler (outside our transformations)}",
        "Found while making the feature tests pass; each fix is a separate branch (dev-side PR), not part of the "
        "FIFO-sizing, folding or DWC PRs. Line counts include tests and docs.",
        fix_list(FINN_BUGS),
        "\\section*{Bugs fixed in our transformations (FIFO sizing, folding, DWC)}",
        "Fixed on the respective PR branches. Line counts include tests and docs.",
        fix_list(OUR_BUGS),
        "\\section*{Open}",
        "\\begin{itemize}",
        "\\item transformer\\_language: output wrong after the deadlock fix (issue 017, in progress).",
        "\\item OuterShuffle\\_hls reaches only II=3 (3$\\times$ its estimate; espcn tail), upstream.",
        "\\item FIFO tree models emit one row per Split/Concat (unequal stream widths may under-size), "
        "node-by-node verification aborts on 3D Requant\\_rtl tensors (upstream).",
        "\\end{itemize}",
        "\\end{document}",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
