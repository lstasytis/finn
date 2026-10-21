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
# metric -> (header, formatter of the feature_metrics.json row)
METRICS = {
    "ratio": ("rtlsim/est.", lambda m, t: "%.3f" % m["rtlsim_vs_est"] if m.get("rtlsim_vs_est") is not None else ""),
    "ref": ("est./ref.", lambda m, t: "%.2f" % m["est_vs_ref"] if m.get("est_vs_ref") is not None else ""),
    "kbit": ("FIFO [kbit]", lambda m, t: "%.1f" % m["fifo_kbit"] if m.get("fifo_kbit") is not None else ""),
    "pad": ("pad.\\ DWCs", lambda m, t: "%d" % m["pad_dwcs"] if m.get("pad_dwcs") is not None else ""),
    "tfifo": ("FIFO step [s]", lambda m, t: "%.0f" % t["step_set_fifo_depths"] if "step_set_fifo_depths" in t else ""),
    "tfold": ("folding step [s]",
              lambda m, t: "%.0f" % t["step_target_fps_parallelization"] if "step_target_fps_parallelization" in t else ""),
}


def cells(job, state, fields):
    """Result mark followed by one cell per metric (empty if the job did not finish)."""
    if job is None:
        return ["n/a"] + [""] * len(fields)
    res = result_of(job, state)
    if res not in ("passed", "failed"):
        return [MARK.get(res, res)] + [""] * len(fields)
    m = job_metrics(job) or {}
    t = m.get("step_times_s") or {}
    return [MARK[res]] + [METRICS[f][1](m, t) for f in fields]


def table(caption, label, groups, rows, legend):
    """groups: [(group title, [metric keys])]; rows: [[model, est] + cells...]. Two-level header."""
    ncols = 2 + sum(1 + len(f) for _, f in groups)
    head1, head2, rules, col = ["", ""], ["model (board)", "est.\\ fps"], [], 3
    for title, fields in groups:
        width = 1 + len(fields)
        head1.append(f"\\multicolumn{{{width}}}{{c}}{{{title}}}")
        head2 += ["result"] + [METRICS[f][0] for f in fields]
        rules.append(f"\\cmidrule(lr){{{col}-{col + width - 1}}}")
        col += width
    return "\n".join([
        "\\begin{landscape}\\begin{table}[p]\\centering\\scriptsize",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{l r {' '.join('c' + 'r' * len(f) for _, f in groups)}}}\\toprule",
        " & ".join(head1) + " \\\\", "".join(rules),
        " & ".join(head2) + " \\\\\\midrule",
        *[" & ".join(r) + " \\\\" for r in rows],
        "\\bottomrule\\end{tabular}}",
        f"\\par\\medskip\\raggedright\\footnotesize\\textbf{{Columns.}} {legend}",
        "\\end{table}\\end{landscape}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="2")
    ap.add_argument("--sizes", default="small,full")
    args = ap.parse_args()
    sizes = args.sizes.split(",")
    state = load(os.path.join(LOGDIR, "campaign_state.json")) or {}
    jobs = build_queue(args.variant, sizes)
    tiers = TIER1 + TIER2 + (TIER_TRANSFORMER if args.variant else [])

    def job(mdir, p, feature, size):
        return next((j for j in jobs if j["model"] == mdir and j["param"] == p
                     and j["feature"] == feature and j.get("size") == size), None)

    fifo_rows, fold_rows, dwc_rows = [], [], []
    FIFO_F, FOLD_F, MAX_F, FAW_F, DWC_F = (["ratio", "kbit", "tfifo"], ["ratio", "ref", "tfold"],
                                          ["ratio", "ref", "tfold"], ["ratio", "kbit", "tfold"],
                                          ["pad", "ratio", "ref"])
    for mdir, _, params in tiers:
        for p, *_ in params:
            board, model = split_param(p)
            name = tex(f"{model or mdir} ({board})")
            mtag = f"{mdir}_{model}" if mdir in MULTI else mdir
            est = None
            for wt in ("fifo", "folding", "dwc"):
                est = est or load(latest(wt + args.variant, f"est_{mtag}_{board}_",
                                         "report/estimate_network_performance.json"))
            est_s = "%.1f" % est["estimated_throughput_fps"] if est else "--"
            row = [name, est_s]
            for f in ("fifo", "fifo-rtlsim"):
                for sz in ("small", "full"):
                    row += cells(job(mdir, p, f, sz), state, FIFO_F)
            fifo_rows.append(row)
            row = [name, est_s]
            for sz in ("small", "full"):
                row += cells(job(mdir, p, "fold-target", sz), state, FOLD_F)
            row += cells(job(mdir, p, "fold-max", "full"), state, MAX_F)
            row += cells(job(mdir, p, "fold-fifo", "full"), state, FAW_F)
            fold_rows.append(row)
            row = [name, est_s]
            for sz in ("small", "full"):
                row += cells(job(mdir, p, "dwc", sz), state, DWC_F)
            dwc_rows.append(row)

    now = datetime.datetime.now()
    doc = [
        "\\documentclass[a4paper,10pt]{article}",
        "\\usepackage[margin=18mm]{geometry}\\usepackage{booktabs}\\usepackage{amssymb}"
        "\\usepackage{graphicx}\\usepackage{pdflscape}\\usepackage[hidelinks]{hyperref}",
        "\\title{FINN feature tests: FIFO sizing, folding and DWC padding on the finn-examples models}",
        f"\\date{{Snapshot {now:%Y-%m-%d %H:%M} (campaign variant {args.variant}, new upstream base)}}",
        "\\begin{document}\\maketitle",
        "\\section*{Results}",
        "Each test resumes the convert-phase checkpoint of a cached estimate-only build with the committed folding, builds the stitched IP and checks it in stitched-IP rtlsim. Each table lists, per model, the result and the measured numbers of every test variant; the column definitions are given under each table.",
        table("FIFO sizing on the committed folding: the analytical sizer (ours) and the rtlsim-based heuristic",
              "tab:fifo", [("analytical, small", FIFO_F), ("analytical, full", FIFO_F),
                           ("rtlsim heuristic, small", FIFO_F), ("rtlsim heuristic, full", FIFO_F)],
              fifo_rows, "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{est.\\ fps}: analytic throughput of the committed folding (the reference). \\emph{rtlsim/est.}: stitched-IP rtlsim throughput divided by the build's own analytic estimate. \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$; \\emph{full}: committed folding. Step times are wall-clock under the campaign's load, not isolated measurements. " + "\\emph{FIFO [kbit]}: total storage of all inserted FIFOs "
              "(depth $\\times$ stream width). \\emph{FIFO step [s]}: time of \\texttt{step\\_set\\_fifo\\_depths}."),
        table("Folding optimizer (FIFOs sized analytically)", "tab:fold",
              [("target, small", FOLD_F), ("target, full", FOLD_F), ("maximize", MAX_F),
               ("FIFO-aware (target)", FAW_F)], fold_rows,
              "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{est.\\ fps}: analytic throughput of the committed folding (the reference). \\emph{rtlsim/est.}: stitched-IP rtlsim throughput divided by the build's own analytic estimate. \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$; \\emph{full}: committed folding. Step times are wall-clock under the campaign's load, not isolated measurements. " + "\\emph{target}: optimizer at the committed folding's throughput (must be met); "
              "\\emph{maximize}: optimizer with no target, maximizing throughput within the device budget; "
              "\\emph{FIFO-aware}: target mode with \\texttt{enable\\_folding\\_fifo\\_heuristic} (the search "
              "scores foldings with analytically sized FIFOs; optional, slower). \\emph{est./ref.}: the "
              "optimizer folding's estimated throughput divided by the committed folding's (1.00 = target met "
              "exactly, maximize: speed-up). \\emph{FIFO [kbit]}: total FIFO storage of the build. "
              "\\emph{folding step [s]}: time of \\texttt{step\\_target\\_fps\\_parallelization}."),
        table("DWC padding: optimizer with up to 4 channels of padding per layer at the committed throughput",
              "tab:dwc", [("small", DWC_F), ("full", DWC_F)], dwc_rows,
              "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{est.\\ fps}: analytic throughput of the committed folding (the reference). \\emph{rtlsim/est.}: stitched-IP rtlsim throughput divided by the build's own analytic estimate. \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$; \\emph{full}: committed folding. Step times are wall-clock under the campaign's load, not isolated measurements. " + "\\emph{pad.\\ DWCs}: number of padding/cropping data width converters in the build "
              "(skip: the padded search inserts none, so no hardware build is needed). \\emph{est./ref.}: "
              "padded folding's estimated throughput divided by the committed folding's (below 1.00 is accepted "
              "if within 10 \\% and its resource saving is larger than the throughput loss)."),
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
