"""Offline FIFO sizing of a saved TAV model (tav_model.py): the depths of DeriveFIFOSizes (max of
the causal and non-causal pass) with the worktree's derive_characteristic, optionally with the
causal_writes rule extended to one-to-one (VARIANT=eq) or all (VARIANT=all) nodes.
usage: size_offline.py <tav.onnx> [tensor ...]"""
import os
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.derive_characteristic as dc

variant = os.environ.get("VARIANT", "base")
orig = dc.causal_writes


def causal_ext(write_times, in_scheds):
    if not in_scheds or len(write_times) == 0:
        return write_times
    out = np.array(write_times, dtype=np.int64, copy=True)
    n_out = len(out)
    fill = max(1, n_out // 2)
    if variant == "lead":
        # writes before the first input token can exist belong to the previous frame:
        # renumber the output tokens so that the first frame starts after them
        first_in = min(int(s[0]) for s, _ in in_scheds.values() if len(s))
        lead = int(np.sum(out[:fill] < first_in))
        if 0 < lead < fill:
            span = int(out[n_out - 1] - out[n_out - 1 - fill])
            out = np.concatenate([out[lead:], out[n_out - fill : n_out - fill + lead] + span])
        return orig(out, in_scheds)
    idx = np.arange(1, fill + 1, dtype=np.int64)
    for sched, _ in in_scheds.values():
        n_in = len(sched)
        if n_in == 0 or (variant == "eq" and n_in < n_out):
            continue
        need = np.minimum(-(-idx * (n_in // 2) // fill), n_in)
        out[:fill] = np.maximum(out[:fill], sched[need - 1])
    return np.maximum.accumulate(out)


if variant != "base":
    dc.causal_writes = causal_ext
m = ModelWrapper(sys.argv[1])
period = int(m.get_metadata_prop("sizing_period"))
a = dc.derive_chained_tav_depths(m, period, causal=False)
b = dc.derive_chained_tav_depths(m, period, causal=True)
d = {t: max(a.get(t, 0), b.get(t, 0)) for t in set(a) | set(b)}
show = sys.argv[2:] or sorted(d)
for t in show:
    print(f"{variant} {t:32s} noncausal={a.get(t, 0):6d} causal={b.get(t, 0):6d} depth={d.get(t, 0):6d}")
print(f"{variant} TOTAL {sum(max(2, v) for v in d.values())}")
