"""pytest plugin used by the TAV (token access vector) evaluation harness.

This plugin is *external* instrumentation: it is loaded into a normal pytest
run with ``-p _tav_eval_plugin`` and does not require any edit to the FINN
source tree. That property matters for the AlphaEvolve-style use case, where an
LLM rewrites ``get_tree_model`` in ``src/`` between runs -- we never want our
measurement code to live in a file the optimizer might overwrite.

It does two things at runtime (so it is import-order independent):

1. Forces the rtlsim characterization to use the on-disk cache. The analytical
   ``tree_model`` token access vector is recomputed every run (cheap, pure
   Python), while the slow rtlsim reference is computed once and then reused
   from ``$FINN_BUILD_DIR``.
2. Captures, for every characterization test, the analytical token access
   vector and the rtlsim reference vector for both ports, and writes a JSON
   record per test into ``$TAV_EVAL_OUT`` containing the decorator parameters,
   the pass/fail verdict and the element-wise delta of the two vectors.

The harness (``tav_eval.py``) reads those JSON records back on the host and
renders the human-readable log.
"""

import json
import os
import re
import sys
import types

import numpy as np

# FINN's util.test hard-imports torchvision (only for unrelated image-resize
# helpers, not the TAV path). This sandbox image deliberately ships no
# torch/torchvision/CUDA stack, so stub torchvision out before importing
# finn.util.test, otherwise collection of the characterization tests fails.
if "torchvision" not in sys.modules:
    try:  # pragma: no cover - prefer the real package when present
        import torchvision  # noqa: F401
    except Exception:
        _tv = types.ModuleType("torchvision")
        _tr = types.ModuleType("torchvision.transforms")
        _fn = types.ModuleType("torchvision.transforms.functional")
        _tr.functional = _fn
        _tv.transforms = _tr
        sys.modules["torchvision"] = _tv
        sys.modules["torchvision.transforms"] = _tr
        sys.modules["torchvision.transforms.functional"] = _fn

import finn.util.test as _ft

# per-test capture buffer, populated by the patched compare function.
# each entry: (analytical_vector, rtlsim_vector, verdict)
_captured = []

# per-test node metadata, populated by the patched characteristic fnc, so the
# host-side local oracle (agent_stub/tav_runtime.py) can replay the candidate's
# get_tree_model with the exact attrs / class name the real node had.
_node_meta = {}


def _jsonable(v):
    """Coerce a node attribute value into something json.dump can handle
    (numpy scalars/arrays, bytes, tuples) -- best effort, drop on failure."""
    try:
        if isinstance(v, (bool, int, float, str)) or v is None:
            return v
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        json.dumps(v)
        return v
    except Exception:
        return None


# Zero-argument query methods a get_tree_model may call on the node beyond
# get_nodeattr (e.g. DWC's get_number_input_values). Their results depend only on
# the node's attributes -- not on the candidate tree -- so capturing them once per
# case from the real node lets the host MockSelf replay them faithfully. Methods
# that don't exist or raise on a given node are simply skipped.
_CAPTURE_METHODS = (
    "get_number_input_values",
    "get_number_output_values",
    "get_exp_cycles",
    "get_folded_input_shape",
    "get_folded_output_shape",
    "get_normal_input_shape",
    "get_normal_output_shape",
    "get_instream_width",
    "get_outstream_width",
    "get_instream_width_padded",
    "get_outstream_width_padded",
)


def _capture_node_meta(inst):
    """Record the specialized node's class name, onnx name/op_type, full
    attribute set, and the results of a curated set of zero-arg query methods.
    Capturing every declared attr (not just the ones the current baseline reads)
    plus the common helper-method results keeps the local oracle robust to
    candidates that reach for a different attr/method than the baseline did."""
    try:
        attrs = {}
        try:
            names = list(inst.get_nodeattr_types().keys())
        except Exception:
            names = []
        for name in names:
            try:
                attrs[name] = _jsonable(inst.get_nodeattr(name))
            except Exception:
                continue
        methods = {}
        for mname in _CAPTURE_METHODS:
            try:
                fn = getattr(inst, mname, None)
                if callable(fn):
                    methods[mname] = _jsonable(fn())
            except Exception:
                continue
        _node_meta.update(
            {
                "class_name": type(inst).__name__,
                "onnx_node_name": getattr(inst.onnx_node, "name", ""),
                "op_type": getattr(inst.onnx_node, "op_type", ""),
                "node_attrs": attrs,
                "node_methods": methods,
            }
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# runtime monkeypatches (resolved through module globals at call time, so they
# take effect regardless of when the test module imported these symbols)
# ---------------------------------------------------------------------------
_orig_get_characteristic_fnc = _ft.get_characteristic_fnc
_orig_compare_two_chr_funcs = _ft.compare_two_chr_funcs


def _patched_get_characteristic_fnc(
    model, node0, part, target_clk_ns, strategy, caching=False
):
    # always cache (and reuse) the rtlsim reference so we skip rtlsim once it
    # has been computed at least once for a given parameter set
    if strategy == "rtlsim":
        caching = True
    inst = _orig_get_characteristic_fnc(
        model, node0, part, target_clk_ns, strategy, caching
    )
    # capture metadata off the specialized node so the host local oracle can
    # replay get_tree_model faithfully (the analytical/tree_model pass carries
    # the same attrs as the rtlsim pass, so either call populates this).
    _capture_node_meta(inst)
    return inst


def _patched_compare_two_chr_funcs(a, b, max_allowed_volume_delta, max_allowed_length_delta):
    res = _orig_compare_two_chr_funcs(a, b, max_allowed_volume_delta, max_allowed_length_delta)
    try:
        _captured.append((np.asarray(a).ravel(), np.asarray(b).ravel(), bool(res)))
    except Exception:
        _captured.append((np.asarray([]), np.asarray([]), bool(res)))
    return res


_ft.get_characteristic_fnc = _patched_get_characteristic_fnc
_ft.compare_two_chr_funcs = _patched_compare_two_chr_funcs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _out_dir():
    d = os.environ.get("TAV_EVAL_OUT")
    if d:
        os.makedirs(d, exist_ok=True)
    return d


def _sanitize(nodeid):
    return re.sub(r"[^A-Za-z0-9._-]", "_", nodeid)


def _port_record(port, analytical, rtlsim, verdict):
    a = np.asarray(analytical).astype(np.int64).ravel()
    b = np.asarray(rtlsim).astype(np.int64).ravel()
    n = int(min(a.size, b.size))
    delta = (a[:n] - b[:n]) if n > 0 else np.asarray([], dtype=np.int64)
    return {
        "port": port,
        "verdict": bool(verdict),
        "len_analytical": int(a.size),
        "len_rtlsim": int(b.size),
        "len_delta": int(a.size - b.size),
        "peak_volume_delta": int(np.max(np.abs(delta))) if delta.size else 0,
        "delta_vector": delta.tolist(),
        # full raw vectors -- the rtlsim one is the optimization target the host
        # local oracle fits against; the analytical one lets the host confirm its
        # reimplementation matches FINN exactly (see tav_runtime.selftest...).
        "rtlsim_vector": b.tolist(),
        "analytical_vector": a.tolist(),
    }


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------
def pytest_runtest_setup(item):
    _captured.clear()
    _node_meta.clear()


def pytest_runtest_makereport(item, call):
    # only act on the main "call" phase
    if call.when != "call":
        return
    out_dir = _out_dir()
    if out_dir is None:
        return

    if call.excinfo is None:
        outcome = "passed"
    elif call.excinfo.errisinstance(getattr(__import__("_pytest").outcomes, "Skipped", ())):
        outcome = "skipped"
    else:
        outcome = "failed"

    params = {}
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        params = {k: repr(v) for k, v in callspec.params.items()}

    ports = []
    # captured order inside tree_model_test is: input port, then output port
    port_names = ["input", "output"]
    for idx, (a, b, verdict) in enumerate(_captured[:2]):
        ports.append(_port_record(port_names[idx] if idx < 2 else f"port{idx}", a, b, verdict))

    record = {
        "nodeid": item.nodeid,
        "test": getattr(item, "originalname", None) or item.name,
        "params": params,
        "outcome": outcome,
        "longrepr": None,
        "ports": ports,
        # node metadata for the host local oracle (empty if capture failed)
        "node_meta": dict(_node_meta),
    }
    if outcome in ("failed", "error") and call.excinfo is not None:
        # keep it short -- just the exception type and message
        record["longrepr"] = f"{call.excinfo.typename}: {call.excinfo.exconly()}"[:2000]

    fname = os.path.join(out_dir, _sanitize(item.nodeid) + ".json")
    with open(fname, "w") as f:
        json.dump(record, f)
