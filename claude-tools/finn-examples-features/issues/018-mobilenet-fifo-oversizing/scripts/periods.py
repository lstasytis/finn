"""Issue 018: per node of a TAV model: period, frame tokens in/out, drives-pacer flag.
usage: periods.py <tav.onnx>"""
import sys

from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
gp = int(m.get_metadata_prop("sizing_period"))
nodes = [n for n in m.graph.node if is_hls_node(n) or is_rtl_node(n)]
tavs = dc.read_node_tavs(m, nodes)
per = dc.node_periods(nodes, tavs)
gp = max(gp, max(per.values()))
dp = dc.pacer_drivers(m, nodes, per, gp * dc.PACER_TOLERANCE)
print("global period", gp)
for n in nodes:
    tin, tout = tavs[n.name][0], tavs[n.name][1]
    fi = dc.frame_tokens(tin) if tin is not None else -1
    fo = dc.frame_tokens(tout) if tout is not None else -1
    print(f"{n.name:36s} period {per.get(n.name, 0):8d} ({per.get(n.name, 0) / gp:5.3f}) "
          f"in {fi:7d} out {fo:7d} drives_pacer {dp[n.name]}")
