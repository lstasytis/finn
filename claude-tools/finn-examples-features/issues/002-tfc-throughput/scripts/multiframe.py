import sys, json
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.util.rtlsim import annotate_rtlsim_performance
m = ModelWrapper(sys.argv[1])
m.set_metadata_prop("rtlsim_so", "")
for n in [int(x) for x in sys.argv[2].split(",")]:
    d = xsi_fifosim(m, n, max_iters=10000, behav=False)
    d = {k: v for k, v in d.items() if "max_count" not in k}
    d = annotate_rtlsim_performance(d, n, 10.0)
    ss = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
    print("RESULT", n, "latency", d["latency_cycles"], "last_interval", d["interval_cycles"],
          "avg_interval", ss, "cycles", d["cycles"], flush=True)
