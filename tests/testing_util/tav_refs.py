# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared plumbing for the stored-rtlsim-TAV tree-model references.

Both the generator (``ci/experiments/gen_cig_refs.py``, which needs Vivado) and
the fast regression tests (``tests/fpgadataflow/test_tav_tree_models.py``, which
do not) build the same single-node graph from the same configuration dict, so
that lives here rather than in either of them.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_DIR = os.path.join(ROOT, "tests", "tav_refs")

PART = "xczu7ev-ffvc1156-2-e"
CLK = 10


def ensure_finn_env():
    """Set the paths that only the ``finn`` CLI entry points would otherwise set.

    Single-node IP generation reads ``FINN_RTLLIB`` / ``FINN_CUSTOM_HLS`` straight
    out of the environment. Under a plain ``uv run python`` or ``uv run pytest``
    the missing variable surfaces from ``_codegen_single_node`` as
    ``Exception: Custom op_type ... is currently not supported``, which is a
    thoroughly misleading message for an unset environment variable.
    """
    os.environ.setdefault("FINN_RTLLIB", os.path.join(ROOT, "finn-rtllib"))
    os.environ.setdefault("FINN_CUSTOM_HLS", os.path.join(ROOT, "custom_hls"))
    os.environ.setdefault("FINN_XSI", os.path.join(ROOT, "finn_xsi", "finn_xsi"))
    os.environ.setdefault("FINN_TESTS", os.path.join(ROOT, "tests"))
    if "FINN_DEPS" not in os.environ:
        for cand in (
            os.path.join(os.path.expanduser("~"), ".finn", "deps"),
            os.path.join(ROOT, "finn_deps"),
        ):
            if os.path.isdir(cand):
                os.environ["FINN_DEPS"] = cand
                break
    os.environ.setdefault("NUM_DEFAULT_WORKERS", "4")


def cig_cfg(k, ifm, ch, simd, s=(1, 1), d=(1, 1), dw=0, pw=0):
    return {
        "k": list(k),
        "ifm": list(ifm),
        "ch": ch,
        "simd": simd,
        "s": list(s),
        "d": list(d),
        "dw": dw,
        "pw": pw,
    }


def cig_key(c):
    return (
        "k{k[0]}x{k[1]}_ifm{ifm[0]}x{ifm[1]}_ch{ch}_simd{simd}"
        "_s{s[0]}x{s[1]}_d{d[0]}x{d[1]}_dw{dw}_pw{pw}".format(**c)
    )


def build_cig_model(c, part=PART):
    """A one-node graph holding exactly the ConvolutionInputGenerator described."""
    from onnx import TensorProto, helper
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.general.im2col import compute_conv_output_dim
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.util.basic import qonnx_make_model
    import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

    idt = DataType["INT2"]
    k_h, k_w = c["k"]
    ifm_h, ifm_w = c["ifm"]
    s_h, s_w = c["s"]
    d_h, d_w = c["d"]
    ofm_h = compute_conv_output_dim(ifm_h, k_h, s_h, 0, d_h)
    ofm_w = compute_conv_output_dim(ifm_w, k_w, s_w, 0, d_w)
    ch = c["ch"]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_h, ifm_w, ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_h, ofm_w, k_h * k_w * ch]
    )
    node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        stride=[s_h, s_w],
        kernel_size=[k_h, k_w],
        input_shape=str((1, ifm_h, ifm_w, ch)),
        dilations=[d_h, d_w],
        pad_amount=[0, 0, 0, 0],
        pad_value=0,
        depthwise=c["dw"],
    )
    graph = helper.make_graph(nodes=[node], name="im2col", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="im2col"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(SpecializeLayers(part))
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", c["simd"])
    if model.graph.node[0].op_type == "ConvolutionInputGenerator_rtl":
        inst.set_nodeattr("parallel_window", c["pw"])
        inst.set_nodeattr("M", 1)
    return model


def read_tav(value):
    """Decode a stored TAV, whichever way the tree it came from stores them.

    finn-plus keeps token access vectors as a gzipped base64 string in the node
    attribute; this tree spills them to a .npy next to the build and keeps the
    path. References are portable between the two, so the reader has to be.
    """
    import finn.util.basic as fub

    if hasattr(fub, "load_tav_npy") and isinstance(value, str) and value.endswith(".npy"):
        return fub.load_tav_npy(value)
    return fub.decompress_string_to_numpy(value)


def tree_tavs(inst):
    """(io_chrc_in, io_chrc_out) recomputed from this node's tree model."""

    n_inps = int(np.prod(inst.get_folded_input_shape()[:-1]))
    io_dict = {"inputs": {"in0": list(range(n_inps))}, "outputs": {"out0": []}}
    inst.derive_token_access_vectors_using_tree_model(0, io_dict=io_dict)
    return (
        read_tav(inst.get_nodeattr("io_chrc_in")),
        read_tav(inst.get_nodeattr("io_chrc_out")),
    )


def as_rows(a):
    """(k, n) view of a TAV: k streams by n cycles."""
    a = np.asarray(a)
    return a.reshape(1, -1) if a.ndim == 1 else a.reshape(a.shape[0], -1)


# --- generic single-node rebuild, for op types whose tree model depends only on
#     node attributes ------------------------------------------------------
#
# The transformer op types (ElementwiseAdd, ReplicateStream, StreamingSplit,
# StreamingConcat, Reshape, Squeeze, Unsqueeze, Lookup,
# ScaledDotProductAttention) all derive their folded shapes from node attributes
# alone, never from the surrounding tensors. So a reference can be replayed by
# recreating a bare node with the recorded attributes -- no shape inference, no
# SpecializeLayers, no ONNX graph plumbing. That is what makes these references
# cheap to store and to check.

# Attributes that describe where a build put its scratch files, not what the
# hardware does. Recording them would make references machine-specific.
VOLATILE_ATTRS = {
    "code_gen_dir_cppsim",
    "code_gen_dir_ipgen",
    "cycles_estimate",
    "cycles_rtlsim",
    "executable_path",
    "ip_path",
    "ip_vlnv",
    "ipgen_path",
    "node_name",
    "res_estimate",
    "res_hls",
    "res_synth",
    "rtlsim_so",
    "rtlsim_trace",
    "io_chrc_in",
    "io_chrc_in_original",
    "io_chrc_out",
    "io_chrc_out_original",
    "io_chrc_period",
    "io_chrc_pads",
    # paths to the .npy the stretch stage spilled to the build's scratch dir.
    # Recorded, they made the key machine-specific and every configuration
    # looked distinct once per build it appeared in -- the attention reference
    # held 33 entries for 5 configurations.
    "io_chrc_in_stretch",
    "io_chrc_out_stretch",
}


def node_spec(inst):
    """Everything needed to rebuild this node's tree model somewhere else."""
    node = inst.onnx_node
    attrs = {}
    for name in inst.get_nodeattr_types():
        if name in VOLATILE_ATTRS:
            continue
        try:
            value = inst.get_nodeattr(name)
        except Exception:
            continue
        if isinstance(value, (list, tuple)):
            value = [int(v) if isinstance(v, (int, np.integer)) else v for v in value]
        elif isinstance(value, np.integer):
            value = int(value)
        attrs[name] = value
    return {
        "op_type": node.op_type,
        "domain": node.domain,
        "name": node.name,
        "n_inputs": len(node.input),
        "n_outputs": len(node.output),
        "attrs": attrs,
    }


def build_node(spec):
    """Rebuild the custom op described by ``spec`` as a standalone node.

    The attributes go on to the protobuf node *before* it is wrapped, not via
    ``set_nodeattr`` afterwards. Several ops read their own attributes from
    ``__init__`` -- ``ReplicateStream`` sizes ``outFIFODepths`` from ``num``,
    for instance -- and would raise "Required attribute num unspecified" on a
    node that is still bare at that point.
    """
    from onnx import helper
    from qonnx.custom_op.registry import getCustomOp

    # The two FINN trees this reference base is shared between do not always
    # spell an attribute the same way: finn-plus's generalized DWC carries
    # in_shape/out_shape where this tree still has a single shape. Translate
    # the ones that are known to mean the same thing, and drop attributes the
    # target operator does not declare rather than failing to build the node.
    ALIASES = {"in_shape": "shape"}
    attrs = {}
    for name, value in spec["attrs"].items():
        # onnx cannot type an empty list, and an attribute whose recorded value
        # is the empty list carries no information anyway
        if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
            continue
        attrs[name] = value
    node = helper.make_node(
        spec["op_type"],
        ["in%d" % i for i in range(spec["n_inputs"])],
        ["out%d" % i for i in range(spec["n_outputs"])],
        domain=spec["domain"],
        name=spec["name"],
        **attrs,
    )
    inst = getCustomOp(node)
    known = set(inst.get_nodeattr_types())
    extra = {ALIASES[k]: v for k, v in attrs.items() if k in ALIASES and ALIASES[k] in known}
    if extra:
        node = helper.make_node(
            spec["op_type"],
            ["in%d" % i for i in range(spec["n_inputs"])],
            ["out%d" % i for i in range(spec["n_outputs"])],
            domain=spec["domain"],
            name=spec["name"],
            **{**{k: v for k, v in attrs.items() if k in known}, **extra},
        )
        inst = getCustomOp(node)
    return inst


def node_key(spec):
    """A stable, readable key for one node configuration."""
    import hashlib

    payload = json.dumps(spec["attrs"], sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:6]
    return "%s_%s" % (spec["op_type"], digest)
