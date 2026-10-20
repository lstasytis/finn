"""Per-edge first-frame timing of the causal pass on a saved TAV model (needs the temporary
_DEBUG_EDGES hook): first token, last token of frame 1, per producer.
usage: edge_times.py <tav.onnx>"""
import sys

from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.derive_characteristic as dc

m = ModelWrapper(sys.argv[1])
period = int(m.get_metadata_prop("sizing_period"))
dc._DEBUG_EDGES = {}
dc.derive_chained_tav_depths(m, period, causal=True)
for n in m.graph.node:
    for t in n.output:
        e = dc._DEBUG_EDGES.get(t)
        if e is None:
            continue
        w = e.writes
        pf = e.per_frame
        print(f"{n.name:36s} {t:24s} pf={pf:6d} first={int(w[0]):7d} f1_end={int(w[pf-1]):7d} "
              f"f2_first={int(w[pf]) if len(w) > pf else -1:7d}")
