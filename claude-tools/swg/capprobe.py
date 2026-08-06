"""Collect the chained-TAV trace and dump the rows for one consumer."""
import json

import finn.transformation.fpgadataflow.derive_characteristic as DC

TARGET = "ConvolutionInputGenerator_rtl_13"
_orig = DC.derive_chained_tav_depths
_seen = []


def _wrapped(model, *a, **kw):
    tr = []
    kw["trace"] = tr
    out = _orig(model, *a, **kw)
    _seen.append(tr)
    return out


DC.derive_chained_tav_depths = _wrapped
import atexit  # noqa: E402

print("CAPPROBE armed", flush=True)


def dump():
    for i, tr in enumerate(_seen):
        rows = [r for r in tr if str(r.get("consumer")) == TARGET or "pacer_period" in r]
        print(f"CAPPROBE pass{i}: {len(tr)} rows, {len(rows)} of interest", flush=True)
        for r in rows:
            print("CAPPROBE", json.dumps(r, default=str), flush=True)


atexit.register(dump)
