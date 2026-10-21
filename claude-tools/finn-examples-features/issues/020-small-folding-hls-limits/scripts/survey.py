"""Embedded HLS MVAU weight initializers (PE x WMEM entries, SIMD x weight bits per entry) of every
code_gen_ipgen_MVAU_hls_* under the cwd (run in build/finn_build/fx) vs the csynth outcome
(ok / err / unfinished), Compile/Link instruction count and time. Output: survey_output.txt."""
import glob, re, os, collections
rows = collections.defaultdict(list)
for d in glob.glob("*/code_gen_ipgen_MVAU_hls_*"):
    tops = glob.glob(d + "/top_*.cpp")
    if not tops:
        continue
    if not os.path.isfile(d + "/params.h"):
        continue
    head = open(d + "/params.h").read(300)
    m = re.search(r"FixedPointWeights<(\d+),\s*ap_u?int<(\d+)>,\s*(\d+),\s*(\d+)>", head)
    if not m:
        continue
    simd, w, pe, wmem = map(int, m.groups())
    log = d + "/vitis_hls.log"
    txt = open(log, errors="ignore").read() if os.path.isfile(log) else ""
    ins = re.search(r"There were ([\d,]+) instructions in the design after the 'Compile/Link'", txt)
    done = re.search(r"Finished Command csynth_design.*?Elapsed time: ([\d:]+)", txt)
    ok = "ok" if done and "ERROR" not in txt else ("err" if "ERROR" in txt else "unfinished")
    rows[(pe * wmem, pe, wmem, simd * w)].append((ok, done.group(1) if done else "-", ins.group(1) if ins else "-"))
for k in sorted(rows):
    v = rows[k]
    c = collections.Counter(x[0] for x in v)
    times = sorted({x[1] for x in v if x[0] == "ok"})
    print(f"entries={k[0]:7d} PE={k[1]:3d} WMEM={k[2]:6d} bits={k[3]:4d}: {dict(c)} times={times[-3:]} ins={sorted({x[2] for x in v})[-2:]}")
