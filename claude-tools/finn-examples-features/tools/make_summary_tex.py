"""Write summary/summary.tex: result tables (FIFO sizing, folding, DWC) from the campaign data and the two
lists of fixes (FINN compiler bugs, bugs in our transformations) with +/- lines and files from git.

    python make_summary_tex.py [--variant 2] [--sizes small,full]

Re-run at any time; the tables show the campaign's current state (not-yet-run cells are marked).
"""
import argparse
import datetime
import glob
import os
import subprocess

from run_campaign import LOGDIR, M, TIER1, TIER2, TIER_TRANSFORMER, build_queue, junit_result
from status_table import MULTI, est_prefix, job_metrics, latest, load, ordered_params, split_param

TOOLS = os.path.dirname(os.path.abspath(__file__))
# reference resources re-estimated with the current estimate model (ref_resources.py, issue 023)
REF_DIR = os.path.join(M, "build/finn_build/fx/ref_resources")
# issue 021's estimate model reached test2/folding and test2/dwc at 2026-10-22 04:29:13 UTC (last
# estimate commit 55c9bb8a2): reference estimates written before are not comparable
EST_MODEL_SINCE = datetime.datetime(2026, 10, 22, 4, 29, 13, tzinfo=datetime.timezone.utc).timestamp()

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
    ("Per-frame HLS overhead in MVAU\\_hls / LabelSelect\\_hls estimates", "002, 025",
     "the HLS top function is not frame-pipelined: +7 / +4 cycles per frame missing from get\\_exp\\_cycles "
     "(tfc/MLP rtlsim at 0.53x of the estimate). Upstream has since made the MVAU/VVAU HLS cores free-running, "
     "so only the LabelSelect\\_hls part (+4) is still needed (tfc: 7/9/9 cycles measured, 3/6/6 estimated).",
     "feature/dev/hls-exp-cycles-overhead", ["2169d7b6d"]),
    ("Resource estimates far from synthesis", "021",
     "FINN's LUT/BRAM estimates put the committed AUP-ZU3 foldings at 1.6--12x the device; MVAU, RTL DWC and "
     "OuterShuffle estimates calibrated to OOC synthesis, embedded weight ROMs above 16 words costed as BRAM.",
     "feature/dev/resource-estimates", ["09e30dd34", "55c9bb8a2"]),
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
    ("FIFO: yolov8 PAN concat deadlock", "024",
     "the join window floor covered two-stream adds only and took FMPadding, sliding windows and upsampling as "
     "one-to-one: a concat shortcut held 9086 of the 13842 words it needs (no output in 383 M cycles); Concat joins "
     "and per-node input needs added.", "feature/dev/analytical-fifo-sizing", ["2cd2a077d"]),
    ("FIFO: throttled cap between slow stages", "024",
     "edges behind a throttled producer were capped at 256 words even between two slow stages (cnv-w2a2 small "
     "0.98x); the cap now needs the decoupled-chain condition.", "feature/dev/analytical-fifo-sizing", ["271ca1e5e"]),
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
    ("Folding: foldings Vitis HLS cannot compile", "021",
     "streams above 8191 bits (resnet50 V80 maximize) and 65536-entry embedded weight ROMs (cnv small, csynth "
     "stuck for 22 h) were offered; HLS limits checked per node.", "feature/dev/set-folding-optimizer", ["69d005339"]),
    ("Folding: maximize below the committed foldings", "021",
     "LUT budget and estimates: maximize reached 0.35--0.96x of the committed AUP-ZU3 foldings; budget at 0.75 of "
     "the platform's LUTs with estimates corrected by the weight values.", "feature/dev/set-folding-optimizer",
     ["00951c1e7", "f3e0dfade"]),
    ("Folding: LabelSelect kept at PE = 1", "023",
     "LabelSelect\\_hls was never folded, capping maximize (tfc-w1a1 0.60x of the committed folding); its PE now "
     "folds over the labels.", "feature/dev/set-folding-optimizer", ["80bd64832", "54fa8d2b5"]),
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
           "pad": "\\#pad DWCs", "tfifo": "time [s]", "tfold": "time [s]",
           "padres": "res.\\ pad/no-pad", "padfps": "fps pad/no-pad (max.)",
           "itime": "islands [s]", "vtime": "Vivado [s]", "speedup": "speedup", "wns": "WNS isl./Viv. [ns]",
           "nisl": "\\#isl."}

# island-flow benchmark (issue 023): its own driver and build dir, test2/islands worktree
ISL_DIR = os.path.join(M, "build/finn_build/fx/islands2")
ISL_FAIL = {"unsupported_vendor_ip": "$\\times$ IP", "timing_failed": "$\\times$ timing",
            "island_failed": "$\\times$ island"}


def islands_data():
    """Per (folding board, model or None, size, flow): the newest feature_metrics.json and the driver's
    newest job state. Island time without a shell built inside the job (as collect_results.py)."""
    metrics, states = {}, {}
    jobs = {j["id"]: j for j in load(os.path.join(ISL_DIR, "driver", "jobs.json")) or []}
    for jid, st in (load(os.path.join(ISL_DIR, "driver", "state.json")) or {}).items():
        # retries (<id>_r, <id>_vip2, ...) run the node of their base job <model>__<size>__<flow>
        mid, size, flow = jid.split("__")
        base = jobs.get(jid) or jobs.get(f"{mid}__{size}__{flow.split('_')[0]}")
        if base is None:
            continue
        # test node ...[<board>-[<model>-]<size>-<flow>]
        board, model = split_param(base["node"].rsplit("[", 1)[1].rstrip("]"))
        *model, size, flow = model.rsplit("-", 2)
        key = (board, model[0] if model else None, size, flow)
        if key not in states or (st.get("start") or 0) >= (states[key].get("start") or 0):
            states[key] = st
    for fn in glob.glob(os.path.join(ISL_DIR, "islands-*", "report", "feature_metrics.json")):
        m = load(fn)
        if not m:
            continue
        flow, st = m["test"].split("-", 1)[1], m.get("stamps") or {}
        if flow == "islands" and "shell" in st and "synth" in st:
            built = m.get("shell_status") == "built" or (m.get("shell_lock_wait_s") or 0) > 1
            m["time_s"] = m["total_s"] - (max(0.0, st["shell"] - st["synth"]) if built else 0.0)
        m["_mtime"] = os.path.getmtime(fn)
        size = "small" if m["tag"].endswith("_small") else "full"
        key = (m.get("model_board"), m["tag"][: -len("_small")] if size == "small" else m["tag"], size, flow)
        if key not in metrics or m["_mtime"] > metrics[key]["_mtime"]:
            metrics[key] = m
    return metrics, states


def islands_cells(mdir, board, model, size, data):
    """result, island time, Vivado time, speedup, WNS island/Vivado, #islands of one model and size."""
    metrics, states = data
    tag = mdir.replace("-", "").replace("_", "") + (f"_{model}" if model else "")
    # model_board + tag prefix: the bnn-pynq KV260 run is tagged ..._kv260
    pick = [m for (b, t, s, f), m in metrics.items() if b == board and s == size and f == "islands"
            and (t == tag or t.startswith(tag + "_"))]
    base = [m for (b, t, s, f), m in metrics.items() if b == board and s == size and f == "baseline"
            and (t == tag or t.startswith(tag + "_"))]
    isl, viv = (max(pick, key=lambda m: m["_mtime"]) if pick else None,
                max(base, key=lambda m: m["_mtime"]) if base else None)
    st = states.get((board, model, size, "islands"), {}).get("state")
    if board in ("VCK190", "V80"):
        return ["n/a", "", "", "", "", ""]
    if st == "running":
        res = "run"
    elif isl is None:
        res = {"fail": "$\\times$", "skipped": "blk"}.get(st, "--")
    elif isl.get("status") != "ok":
        res = ISL_FAIL.get(isl.get("status"), "$\\times$")
    elif (isl.get("verify") or {}).get("golden_match"):
        res = "\\checkmark"
    else:
        res = "$\\times$ verify" if isl.get("verify") else "bit"
    ok = isl is not None and isl.get("status") == "ok"
    vok = viv is not None and viv.get("status") == "ok"
    t, tv = (isl or {}).get("timing") or {}, (viv or {}).get("timing") or {}
    wns = "/".join("%.2f" % x["wns"] for x in (t, tv) if x.get("wns") is not None)
    return [res,
            "%.0f" % isl["time_s"] if ok else "",
            "%.0f" % viv["time_s"] if vok else "",
            "%.2fx" % (viv["time_s"] / isl["time_s"]) if ok and vok else "",
            wns if isl is not None and isl.get("timing") else "",
            str(isl.get("islands", "")) if ok else ""]


def resources(output_dir):
    """Estimated LUT/BRAM_18K/URAM/DSP of a build without its FIFOs (FIFO sizing has its own table)."""
    rep = load(os.path.join(output_dir, "report", "estimate_layer_resources.json")) if output_dir else None
    return layer_resources(rep)


def layer_resources(rep):
    """Sum of a per-layer estimate's LUT/BRAM_18K/URAM/DSP without the FIFOs."""
    if not rep:
        return None
    tot = dict.fromkeys(RES, 0.0)
    for layer, r in rep.items():
        if layer == "total" or layer.startswith("StreamingFIFO") or not isinstance(r, dict):
            continue
        for k in RES:
            tot[k] += float(r.get(k, 0) or 0)
    return tot


def new_model(output_dir):
    """Whether a folding/DWC build started after the estimate model change (its estimates are
    issue 021's): the first timestamp of its build_dataflow.log."""
    try:
        with open(os.path.join(output_dir, "build_dataflow.log")) as f:
            stamp = f.readline()[1:20]
        start = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except (OSError, TypeError, ValueError):
        return True
    return start.replace(tzinfo=datetime.timezone.utc).timestamp() >= EST_MODEL_SINCE


def reference_resources(job, jobs, current_model=True):
    """Resources of the committed folding built the same way (the model's FIFO test build: committed
    folding + its DWCs, FIFOs excluded). For a build with the current estimate model: the FIFO build
    re-estimated with that model by ref_resources.py, else the estimate-only build (no DWCs,
    flagged) if it was written with that model, else None (empty cell). For an older build: the
    FIFO build's own report (the FIFO builds' estimates are the old model too)."""
    fifo = next((j for j in jobs if j["model"] == job["model"] and j["param"] == job["param"]
                 and j["feature"] == "fifo" and j.get("size") == job.get("size")), None)
    m = job_metrics(fifo) if fifo else None
    if not current_model:
        return (resources(m.get("output_dir")) if m else None), False
    ref = load(os.path.join(REF_DIR, fifo["id"] + ".json")) if m else None
    if ref and ref.get("source") == m.get("output_dir"):
        return layer_resources(ref["layers"]), False
    prefix = est_prefix(job["model"], job["param"], job.get("size") == "small")
    est = latest("folding" + VARIANT, prefix, "report/estimate_layer_resources.json")
    if not est or os.path.getmtime(est) < EST_MODEL_SINCE:
        return None, True
    return resources(os.path.dirname(os.path.dirname(est))), True


def update_reference_resources():
    """Run ref_resources.py in the folding worktree (FINN environment) for new FIFO builds."""
    wt = os.path.join(M, "build/wt", "folding" + VARIANT)
    cmd = f"source {TOOLS}/wtenv.sh {wt} >/dev/null 2>&1 && python {TOOLS}/ref_resources.py"
    if subprocess.run(["bash", "-c", cmd], cwd=TOOLS).returncode:
        print("warning: ref_resources.py failed, res./ref. may miss references")


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
            ref, fallback = reference_resources(job, jobs, new_model(m.get("output_dir")))
            out.append(resource_ratio(resources(m.get("output_dir")), ref, fallback))
        elif f == "kbit":
            out.append("%.2f" % (m["fifo_kbit"] / 8) if m.get("fifo_kbit") is not None else "")
        elif f == "pad":
            out.append("%d" % m["pad_dwcs"] if m.get("pad_dwcs") is not None else "")
        elif f == "padres":  # DWC test: padded / unpadded optimizer folding, same seed, at the target
            c = m.get("pad_compare")
            out.append("%.2f" % c["res_ratio"] + "".join(" +" + r.split("_")[0] for r in c["res_extra"])
                       if c else "")
        elif f == "padfps":  # DWC test: padded / unpadded maximize-mode throughput
            mx = (m.get("pad_compare") or {}).get("maximize")
            out.append("%.3f" % mx["fps_ratio"] if mx else "")
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
        "\\end{table}\\end{landscape}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="2")
    ap.add_argument("--sizes", default="small,full")
    ap.add_argument("--no-ref-update", action="store_true", help="skip ref_resources.py")
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    if not args.no_ref_update:
        update_reference_resources()
    sizes = args.sizes.split(",")
    state = load(os.path.join(LOGDIR, "campaign_state.json")) or {}
    jobs = build_queue(args.variant, sizes)
    tiers = TIER1 + TIER2 + (TIER_TRANSFORMER if args.variant else [])

    def job(mdir, p, feature, size):
        return next((j for j in jobs if j["model"] == mdir and j["param"] == p
                     and j["feature"] == feature and j.get("size") == size), None)

    fifo_rows, fold_rows, dwc_rows, isl_rows = [], [], [], []
    isl_data = islands_data()
    ISL_F = ["itime", "vtime", "speedup", "wns", "nisl"]
    FIFO_F, FOLD_F, FAW_F, DWC_F = (["ratio", "kbit", "tfifo"], ["tref", "res", "tfold"],
                                    ["tref", "res", "kbit", "tfold"], ["pad", "padres", "tref", "res"])
    DWC_FULL = ["pad", "padres", "padfps", "tref", "res"]
    for mdir, p in ordered_params(tiers):
        if True:
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
            row += cells(job(mdir, p, "dwc", "small"), state, DWC_F, jobs)
            row += cells(job(mdir, p, "dwc", "full"), state, DWC_FULL, jobs)
            dwc_rows.append(row)
            row = [name, nodes, est_s]
            for sz in ("small", "full"):
                row += islands_cells(mdir, board, model, sz, isl_data)
            isl_rows.append(row)

    now = datetime.datetime.now()
    doc = [
        "\\documentclass[a4paper,10pt]{article}",
        "\\usepackage[margin=18mm]{geometry}\\usepackage{booktabs}\\usepackage{amssymb}\\usepackage[table]{xcolor}"
        "\\usepackage{graphicx}\\usepackage{pdflscape}\\usepackage[hidelinks]{hyperref}",
        "\\title{FINN feature tests: FIFO sizing, folding and DWC padding on the finn-examples models}",
        f"\\date{{Snapshot {now:%Y-%m-%d %H:%M} (campaign variant {args.variant}, new upstream base)}}",
        "\\begin{document}\\maketitle",
        "\\section*{Results}",
        "Key: \\checkmark{} passed, $\\times$ failed, skip = skipped, run = running, blk = blocked, -- = not run; ref.\\ = the committed folding JSON; small = its parallelism cut 4--8$\\times$; res./ref.\\ = mean of the LUT, BRAM, URAM and DSP ratios; 1 kB = 1024 bytes.",
        "Island flow (Table~\\ref{tab:islands}): \\checkmark{} = bitstream, timing met and output equal to the "
        "golden output (verify\\_accel); $\\times$ IP = HLS floating-point vendor IP (fadd/fmul) not supported by "
        "the flow, $\\times$ timing / island = timing missed / island P\\&R failed, bit = bitstream not yet verified; "
        "AUP-ZU3\\_8GB and KV260\\_SOM foldings built for the ZCU104; island and Vivado times from the same "
        "checkpoint with the same thread cap, under the campaign's load (indicative); speedup = Vivado / islands.",
        table("FIFO sizing results on the committed folding.",
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
        table("Folding optimizer results.", "tab:fold",
              [("target, small", FOLD_F), ("target, full", FOLD_F), ("maximize", FOLD_F),
               ("FIFO-aware (target)", FAW_F)], fold_rows,
              "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{\\#nodes}: compute layers of the committed folding (before FIFO/DWC insertion). \\emph{est.\\ fps}: analytic throughput of the committed folding JSON (the reference). \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$ (its reference is the small committed folding); \\emph{full}: the committed folding. Step times are wall-clock under the campaign's load. " + "\\emph{rtlsim/ref.}: stitched-IP rtlsim throughput of the build divided by the committed folding's estimated throughput (1.00 = same throughput as the provided JSON; maximize: the speed-up). \\emph{res./ref.}: estimated resources of the build divided by those of the committed folding built the same way (DWCs included, FIFOs excluded): mean of the LUT, BRAM\\_18K, URAM and DSP ratios over the resources the committed folding uses ($\\leq 1.00$ = same or fewer resources); a resource only the new folding uses is named after it (e.g.\\ +URAM); $^\\dagger$ = reference without DWCs (the committed folding's FIFO test has not finished yet). " + "\\emph{target}: optimizer at the committed folding's throughput; "
              "\\emph{maximize}: no target, maximum throughput within the device budget; \\emph{FIFO-aware}: "
              "target mode with \\texttt{enable\\_folding\\_fifo\\_heuristic} (optional, slower). "
              "\\emph{FIFO [kB]}: total FIFO storage of the build (1 kB = 1024 bytes). \\emph{time [s]}: time of "
              "\\texttt{step\\_target\\_fps\\_parallelization}."),
        table("DWC padding results: the padded optimizer folding built in hardware and compared with the "
              "unpadded optimizer at the same seed.",
              "tab:dwc", [("small", DWC_F), ("full", DWC_FULL)], dwc_rows,
              "\\emph{result}: \\checkmark{} passed (output equal to the golden output and rtlsim throughput $\\geq 0.98\\times$ the build's own estimate), $\\times$ failed, skip = skipped, run = running, blk = blocked (its small run failed), -- = not run yet; metric cells stay empty unless the test finished. \\emph{\\#nodes}: compute layers of the committed folding (before FIFO/DWC insertion). \\emph{est.\\ fps}: analytic throughput of the committed folding JSON (the reference). \\emph{small}: committed folding with every node's parallelism cut 4--8$\\times$ (its reference is the small committed folding); \\emph{full}: the committed folding. Step times are wall-clock under the campaign's load. " + "\\emph{rtlsim/ref.}: stitched-IP rtlsim throughput of the build divided by the committed folding's estimated throughput (1.00 = same throughput as the provided JSON; maximize: the speed-up). \\emph{res./ref.}: estimated resources of the build divided by those of the committed folding built the same way (DWCs included, FIFOs excluded): mean of the LUT, BRAM\\_18K, URAM and DSP ratios over the resources the committed folding uses ($\\leq 1.00$ = same or fewer resources); a resource only the new folding uses is named after it (e.g.\\ +URAM); $^\\dagger$ = reference without DWCs (the committed folding's FIFO test has not finished yet). " + "\\emph{\\#pad DWCs}: padding/cropping data width converters in the build. "
              "\\emph{res.\\ pad/no-pad}: estimated resources of the padded / the unpadded optimizer folding "
              "(same seed, reference target; the seed with the largest saving of DWC\\_SEEDS), mean of the LUT, "
              "BRAM\\_18K, URAM and DSP ratios ($<1.00$ = padding saves). \\emph{fps pad/no-pad (max.)}: "
              "estimated throughput of the padded / unpadded optimizer in maximize mode (full size only). A "
              "result below 1.00 rtlsim/ref.\\ is accepted if within 10 \\% and its resource saving is larger "
              "than the throughput loss."),
        table("RapidWright island flow (bitstream) vs FINN's Vivado bitfile flow on the committed folding.",
              "tab:islands", [("small", ISL_F), ("full", ISL_F)], isl_rows,
              "\\emph{result}: \\checkmark{} bitstream with timing met (setup and hold) and the accelerator netlist's "
              "output equal to the golden output (verify\\_accel, 1--4 frames), $\\times$ IP = the flow does not "
              "support the vendor IP cores in the model (HLS floating-point operators, e.g.\\ fadd/fmul), $\\times$ timing = an "
              "island or the assembly misses timing, $\\times$ island = an island's P\\&R failed, bit = bitstream "
              "built but not yet verified, blk = blocked (its small run failed), run = running, -- = not run yet, "
              "n/a = Versal (no island shell). Boards: the AUP-ZU3\\_8GB and KV260\\_SOM foldings are built for "
              "the ZCU104 (island shell), U55C for the U55C (Vivado-only XDMA shell); same clock as the model's "
              "build config. \\emph{islands [s]}: wall-clock of the island flow from the hardware checkpoint to "
              "the bitstream (node synthesis, parallel island P\\&R, RapidWright stitching, assembly route), "
              "cached shell (a shell built in the job is subtracted); \\emph{Vivado [s]}: FINN's normal bitfile "
              "flow from the same checkpoint with the same thread cap (8--32 threads, pinned); "
              "\\emph{speedup} = Vivado / islands. Both measured under the campaign's load (1-min load "
              "80--130 on 128 threads), so the times are indicative. \\emph{WNS}: worst setup slack of the "
              "island build / of the Vivado build. \\emph{\\#isl.}: islands placed and routed in parallel."),
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
        "\\item Island flow (issue 023, running on spare CPU): HLS floating-point vendor IP unsupported (resnet18, "
        "yolov8, transformer\\_radioml/vision), vgg10 small misses 250 MHz (WNS $-$0.19 ns), mobilenet small: island "
        "P\\&R failed; full-size tfc/lfc foldings are slower than Vivado (0.63--0.92$\\times$).",
        "\\item LabelSelect\\_hls per-frame restart (issue 025): with upstream's free-running MVAUs it is the "
        "tfc bottleneck, 7/9/9 cycles a frame against an estimate of 3/6/6 (tfc-w1a1/w1a2/w2a2 FIFO tests fail); "
        "+4 in its estimate (estimates PR). cybersecurity-mlp: 1-cycle frames read 0.5$\\times$ over a 2-frame "
        "rtlsim (start-up slip), correct from 3 frames on.",
        "\\item Folding PR alone (issue 026): kws maximize fails in dev's \\texttt{characterize} FIFO sizing "
        "(period = estimate + 10 cycles, Thresholding\\_rtl needs +20); passes with the FIFO PR merged.",
        "\\item yolov8\\_4w4a small, optimizer folding (fold-target): builds, but the output differs from the "
        "golden output (not a FIFO problem, not investigated).",
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
