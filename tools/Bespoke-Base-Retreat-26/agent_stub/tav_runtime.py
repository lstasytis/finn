"""Faithful, dependency-free reimplementation of FINN's analytical TAV
derivation, so a candidate ``get_tree_model`` can be evaluated on the host
(outside the FINN docker container) and produce a token access vector that is
**byte-identical** to what the dockerized characterization pytest computes.

Why this exists
---------------
The dockerized validator does NOT just traverse the candidate tree. It runs
``HWCustomOp.derive_token_access_vectors_using_tree_model`` (see the mirror at
``inputs/src/finn/custom_op/fpgadataflow/hwcustomop.py``), which:

  * derives the period from ``get_total_cycles(0)``,
  * traverses the tree for **two** periods (input port), with a node-type and
    impl-style specific ``apply_micro_buffer_correction`` after each period,
  * traverses two periods for the output port.

A naive "just call ``traverse_phase_tree`` once" oracle silently disagrees with
docker. So the two pieces below are lifted **verbatim** from FINN (only adapted
to return plain Python ``int`` lists instead of numpy arrays, which the storage
layer needs but the comparison does not):

  * ``Characteristic_Node``  -- copied from ``finn.util.basic``; pure Python, no
    third-party deps. The candidate builds instances of exactly this class.
  * ``derive_tavs``          -- the body of
    ``derive_token_access_vectors_using_tree_model`` plus its nested
    ``apply_micro_buffer_correction``, with numpy stripped out.

The only thing the host cannot know without a real run is the node's
compile-time attributes (``self.get_nodeattr(...)``), its class name (carries
the ``_rtl``/``_hls`` suffix the candidate branches on) and its
``onnx_node.name`` (used by the micro-buffer correction). Those are captured
from one real docker run by ``_tav_eval_plugin`` and replayed here through
``MockSelf`` -- which closes the only fidelity gap.

If a captured analytical vector is shipped alongside the metadata,
``selftest_against_capture`` can confirm this reimplementation matches FINN on
that node/config exactly; the loop does this automatically on the baseline so a
silent divergence is caught before any LLM iteration trusts the local oracle.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable

try:  # candidates may reference ``np``; provide it if the host has it.
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover - host without numpy
    np = None

try:  # candidates may reference qonnx's ``DataType`` (e.g. Thresholding/MVAU).
    from qonnx.core.datatype import DataType  # noqa: F401
except Exception:  # pragma: no cover - host without qonnx
    DataType = None


# ---------------------------------------------------------------------------
# Characteristic_Node -- copied verbatim from finn.util.basic (pure Python).
# Keep this in lockstep with inputs/src/finn/util/basic.py.
# ---------------------------------------------------------------------------
class Characteristic_Node:
    def __init__(self, name, sub_phases, leaf):
        self.name = name
        self.sub_phases = sub_phases
        self.cycles_eval = None
        self.cycles_inputs = None
        self.cycles_outputs = None
        self.leaf = leaf
        self.debug = False

    def sum(self, op):
        if self.leaf:
            if op == 2:
                return sum([x[0] for x in self.sub_phases])
            else:
                return sum([x[0] * x[1][op] for x in self.sub_phases])
        else:
            return sum([x[0] * x[1].sum(op) for x in self.sub_phases])

    def traverse_phase_tree(self, op, counter, cycles, ch_fnc):
        if self.leaf:
            for phase in self.sub_phases:
                for _ in range(phase[0]):
                    if op == 2:
                        counter += 1
                    else:
                        counter += phase[1][op]
                    cycles += 1
                    ch_fnc.append(counter)
            return counter, cycles, ch_fnc
        else:
            for phase in self.sub_phases:
                for _ in range(phase[0]):
                    counter, cycles, ch_fnc = phase[1].traverse_phase_tree(
                        op, counter, cycles, ch_fnc
                    )
            return counter, cycles, ch_fnc

    def get_total_cycles(self, op):
        counter = 0
        cycles = 0
        ch_fnc = []
        counter, cycles, ch_fnc = self.traverse_phase_tree(op, counter, cycles, ch_fnc)
        last_update = 0
        last_val = ch_fnc[op]
        for i in range(1, len(ch_fnc[1:]) + 1):
            if ch_fnc[i] > last_val:
                last_update = i
                last_val = ch_fnc[i]
        return cycles, last_update, ch_fnc


# ---------------------------------------------------------------------------
# MockSelf -- the minimal `self` the candidate's get_tree_model needs.
# ---------------------------------------------------------------------------
class _MockSelfBase:
    def __init__(self, attrs: dict, onnx_node_name: str, op_type: str):
        self._attrs = dict(attrs)
        self.onnx_node = SimpleNamespace(name=onnx_node_name, op_type=op_type)

    def get_nodeattr(self, name, *args, **kwargs):
        if name in self._attrs:
            return self._attrs[name]
        raise KeyError(
            f"node attribute {name!r} was not captured from the docker run; "
            "the local oracle cannot supply it (re-run the baseline so the "
            "plugin captures the full attribute set, or use --oracle docker)."
        )


def make_mock_self(class_name: str, attrs: dict, onnx_node_name: str, op_type: str):
    """Build a `self` whose ``__class__.__name__`` is exactly ``class_name``
    (so candidate ``"_rtl" in self.__class__.__name__`` branching matches the
    real specialized node), backed by the captured attribute dict."""
    cls = type(class_name, (_MockSelfBase,), {})
    return cls(attrs, onnx_node_name, op_type)


# ---------------------------------------------------------------------------
# compile + bind a candidate get_tree_model
# ---------------------------------------------------------------------------
def load_get_tree_model(source: str) -> Callable:
    """Exec candidate source (a file whose only required top-level def is
    ``get_tree_model(self)``) in a namespace that mirrors the symbols available
    inside the spliced FINN module, and return the bound function object."""
    ns: dict = {
        "Characteristic_Node": Characteristic_Node,
        "math": math,
        "np": np,
        "DataType": DataType,
    }
    code = compile(source, "<candidate_get_tree_model>", "exec")
    exec(code, ns)  # noqa: S102 - sandboxed/optimizer-controlled source
    fn = ns.get("get_tree_model")
    if fn is None or not callable(fn):
        raise ValueError("candidate does not define a top-level get_tree_model(self)")
    return fn


# ---------------------------------------------------------------------------
# derive_tavs -- lifted from HWCustomOp.derive_token_access_vectors_using_tree_model
# (numpy stripped; returns python int lists). Keep in lockstep with the mirror.
# ---------------------------------------------------------------------------
def _apply_micro_buffer_correction(start, txn_in, period, class_name, onnx_node_name):
    buffer = 0
    if "FMPadding" in onnx_node_name:
        buffer = 1 if "_rtl" in class_name else 2
    if "StreamingDataWidthConverter" in onnx_node_name:
        buffer = 1 if "_rtl" in class_name else 2
    if "Pool" in onnx_node_name:
        buffer = 1 if "_rtl" in class_name else 2
    if "MVAU" in onnx_node_name:
        buffer = 1 if "_rtl" in class_name else 2

    if buffer > 0:
        if period < 14:
            return txn_in

        if buffer == 2:
            if txn_in[start + 1] - txn_in[start] >= 1:
                buffer = 1
            else:
                txn_in[start + 1] += 1

        idx = start + buffer
        while idx < len(txn_in):
            if txn_in[idx] - txn_in[idx - 1] < buffer:
                txn_in[idx] += buffer
            idx += 1

        idx = len(txn_in) - 1
        last = txn_in[idx]
        while last == txn_in[idx]:
            txn_in[idx] -= buffer
            idx -= 1
        if buffer == 2:
            txn_in[idx] -= 1

    return txn_in


def derive_tavs(get_tree_model: Callable, mock_self) -> tuple[list[int], list[int]]:
    """Reproduce FINN's analytical (tree_model) derivation for both ports.

    Returns (txn_in, txn_out) as plain python int lists -- exactly the vectors
    the dockerized plugin ravels out of io_chrc_in / io_chrc_out and compares
    against the rtlsim reference.
    """
    class_name = type(mock_self).__name__
    onnx_node_name = mock_self.onnx_node.name

    # period derived from the input-port traversal (op=0)
    chr_node = get_tree_model(mock_self)
    period, _in_clocks, _ = chr_node.get_total_cycles(0)

    # ---- input port: two periods, with micro-buffer correction after each ----
    top = get_tree_model(mock_self)
    counter, cycles = 0, 0
    txn_in: list[int] = []
    counter, cycles, txn_in = top.traverse_phase_tree(0, counter, cycles, txn_in)
    txn_in = _apply_micro_buffer_correction(0, txn_in, period, class_name, onnx_node_name)
    cycles = len(txn_in)
    counter, cycles, txn_in = top.traverse_phase_tree(0, counter, cycles, txn_in)
    txn_in = _apply_micro_buffer_correction(period, txn_in, period, class_name, onnx_node_name)

    # ---- output port: two periods (no correction) ----
    counter, cycles = 0, 0
    txn_out: list[int] = []
    counter, cycles, txn_out = top.traverse_phase_tree(1, counter, cycles, txn_out)
    cycles = period
    counter, cycles, txn_out = top.traverse_phase_tree(1, counter, cycles, txn_out)

    return txn_in, txn_out


def derive_for_case(source: str, case: dict) -> tuple[list[int], list[int]]:
    """Top-level helper: compile ``source`` and derive (in, out) vectors for one
    captured ``case`` dict (keys: class_name, node_attrs, onnx_node_name,
    op_type)."""
    fn = load_get_tree_model(source)
    mock = make_mock_self(
        case["class_name"],
        case["node_attrs"],
        case.get("onnx_node_name", case["class_name"]),
        case.get("op_type", ""),
    )
    return derive_tavs(fn, mock)


def selftest_against_capture(source: str, case: dict) -> tuple[bool, str]:
    """Confirm this host reimplementation matches the analytical vectors FINN
    actually captured for this case (``case['analytical_in']`` /
    ``['analytical_out']``). Used on the baseline to prove the local oracle is
    faithful before any LLM iteration relies on it."""
    ref_in = case.get("analytical_in")
    ref_out = case.get("analytical_out")
    if ref_in is None or ref_out is None:
        return True, "no captured analytical vectors to self-test against"
    got_in, got_out = derive_for_case(source, case)
    if got_in == list(ref_in) and got_out == list(ref_out):
        return True, "local oracle matches FINN's analytical derivation exactly"
    di = next((i for i, (a, b) in enumerate(zip(got_in, ref_in)) if a != b), None)
    do = next((i for i, (a, b) in enumerate(zip(got_out, ref_out)) if a != b), None)
    return False, (
        f"local oracle DIVERGES from FINN: in len {len(got_in)} vs {len(ref_in)} "
        f"(first diff @ {di}), out len {len(got_out)} vs {len(ref_out)} (first diff @ {do})"
    )


# ---------------------------------------------------------------------------
# smoke test (host-runnable, no numpy/finn needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _FMPADDING = '''
def get_tree_model(self):
    IMGDIM = self.get_nodeattr("ImgDim")
    PADDING = self.get_nodeattr("Padding")
    NUMCHANNELS = self.get_nodeattr("NumChannels")
    SIMD = self.get_nodeattr("SIMD")
    batch_size = self.get_nodeattr("numInputVectors")
    IMPL_STYLE = "rtl" if "_rtl" in (self.__class__.__name__) else "hls"
    NF = int(NUMCHANNELS / SIMD)
    y_padding_top, x_padding_left, y_padding_bottom, x_padding_right = PADDING
    y_dim = IMGDIM[0]
    x_dim = IMGDIM[1]
    loop_overhead = 1 if (IMPL_STYLE == "hls" and NF == 1) else 0
    ch_pad = Characteristic_Node("Channel_Pad", [(NF, [0, 1]), (loop_overhead, [0, 0])], True)
    ch_pass = Characteristic_Node("Channel_Pass", [(NF, [1, 1]), (loop_overhead, [0, 0])], True)
    x_inner_line = Characteristic_Node("inner", [(x_padding_left, ch_pad), (x_dim, ch_pass), (x_padding_right, ch_pad)], False)
    x_outer_line = Characteristic_Node("outer", [(x_padding_left + x_dim + x_padding_right, ch_pad)], False)
    fmpadding = Characteristic_Node("fm", [(y_padding_top, x_outer_line), (y_dim, x_inner_line), (y_padding_bottom, x_outer_line)], False)
    return Characteristic_Node("top", [(batch_size, fmpadding)], False)
'''
    case = {
        "class_name": "FMPadding_rtl",
        "onnx_node_name": "FMPadding_rtl_0",
        "op_type": "FMPadding_rtl",
        "node_attrs": {
            "ImgDim": [4, 4],
            "Padding": [1, 1, 1, 1],
            "NumChannels": 4,
            "SIMD": 2,
            "numInputVectors": 1,
        },
    }
    tin, tout = derive_for_case(_FMPADDING, case)
    print(f"input  vector ({len(tin)}):", tin)
    print(f"output vector ({len(tout)}):", tout)
    assert tin and tout, "derivation produced empty vectors"
    assert all(b >= a for a, b in zip(tin, tin[1:])), "input TAV must be monotonic non-decreasing"
    print("OK: derivation runs, vectors are monotonic cumulative counts")
