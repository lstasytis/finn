# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FINN dataflow ONNX graph -> DynaRapid dot graph.

DynaRapid reads Dynamatic-style dot files: one line per node with its
component ("dcp") and the bit widths of its input and output channels, and
one line per channel connection ("from = outK, to = inL", 1-based). Here each
FINN node becomes one node and each streaming tensor between two nodes one
channel; graph inputs/outputs stay unconnected and become top-level ports.

Nodes are called n0, n1, ... in the dot file (DynaRapid strips underscores
from names and special-cases names containing e.g. "cst", "sink" or "MC");
the mapping to FINN node names is returned alongside.
"""

from finn.util.dynarapid.components import stream_interfaces


def streaming_inputs(model, node):
    """ONNX input tensor names of the node that are AXI streams (not initializers)."""
    return [t for t in node.input if model.get_initializer(t) is None]


def onnx_to_dot(model, dcp_names, graph_name="finn_design"):
    """Build the dot text.

    dcp_names: dict FINN node name -> component name.
    Returns (dot_text, {dot id: FINN node name}).
    """
    ids = {n.name: "n%d" % i for i, n in enumerate(model.graph.node)}
    lines = ["Digraph %s {" % graph_name]
    widths = {}
    for node in model.graph.node:
        ins, outs = stream_interfaces(node)
        s_in = streaming_inputs(model, node)
        assert len(s_in) == len(ins), "%s: %d streaming inputs but %d s_axis" % (
            node.name,
            len(s_in),
            len(ins),
        )
        assert len(node.output) == len(outs), "%s: output count mismatch" % node.name
        widths[node.name] = ([w for _, w in ins], [w for _, w in outs])
        in_str = " ".join("in%d:%d" % (i + 1, w) for i, (_, w) in enumerate(ins))
        out_str = " ".join("out%d:%d" % (j + 1, w) for j, (_, w) in enumerate(outs))
        lines.append(
            '\t"%s" [type = "Operator", op = "finn", dcp = "%s", in = "%s", out = "%s"];'
            % (ids[node.name], dcp_names[node.name], in_str, out_str)
        )
    for node in model.graph.node:
        for j, t in enumerate(node.output):
            for cons in model.find_consumers(t) or []:
                i = streaming_inputs(model, cons).index(t)
                w_out = widths[node.name][1][j]
                w_in = widths[cons.name][0][i]
                assert w_out == w_in, "width mismatch %s.out%d (%d) -> %s.in%d (%d)" % (
                    node.name,
                    j,
                    w_out,
                    cons.name,
                    i,
                    w_in,
                )
                lines.append(
                    '\t"%s" -> "%s" [from = "out%d", to = "in%d"];'
                    % (ids[node.name], ids[cons.name], j + 1, i + 1)
                )
    lines.append("}")
    return "\n".join(lines) + "\n", {v: k for k, v in ids.items()}


def external_ports(model):
    """Graph inputs/outputs as DynaRapid top-level port names, in graph I/O order.

    DynaRapid names the ports of unconnected channels <node>_din_<i>, <node>_valid_in_<i>,
    <node>_ready_out_<i> (inputs) and <node>_dout_<j>, <node>_valid_out_<j>,
    <node>_ready_in_<j> (outputs), with node = n<k> and 0-based channel indices.
    Returns (inputs, outputs), each a list of dicts with the port names and width.
    """
    ids = {n.name: "n%d" % i for i, n in enumerate(model.graph.node)}
    ins, outs = [], []
    for t in model.graph.input:
        for cons in model.find_consumers(t.name):
            i = streaming_inputs(model, cons).index(t.name)
            width = stream_interfaces(cons)[0][i][1]
            nid = ids[cons.name]
            ins.append(
                {
                    "data": "%s_din_%d" % (nid, i),
                    "valid": "%s_valid_in_%d" % (nid, i),
                    "ready": "%s_ready_out_%d" % (nid, i),
                    "width": width,
                }
            )
    for t in model.graph.output:
        prod = model.find_producer(t.name)
        j = list(prod.output).index(t.name)
        width = stream_interfaces(prod)[1][j][1]
        nid = ids[prod.name]
        outs.append(
            {
                "data": "%s_dout_%d" % (nid, j),
                "valid": "%s_valid_out_%d" % (nid, j),
                "ready": "%s_ready_in_%d" % (nid, j),
                "width": width,
            }
        )
    return ins, outs


def kernel_wrapper_verilog(wrapper_name, core_name, ins, outs, black_box=True):
    """Verilog with the stitched-IP interface (ap_clk, ap_rst_n, s_axis_i, m_axis_j) that
    instantiates the DynaRapid-routed core as a black box, to be filled in with
    read_checkpoint -cell during implementation."""

    def bus(w):
        return "[%d:0] " % (w - 1) if w > 1 else ""

    def axis_port(bus_if, sig, direction, width=""):
        return '    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 %s %s" *) %s %s%s_%s' % (
            bus_if,
            sig,
            direction,
            width,
            bus_if,
            sig.lower(),
        )

    core_ports = ["    input clk", "    input rst"]
    for p in ins:
        core_ports += [
            "    input %s%s" % (bus(p["width"]), p["data"]),
            "    input %s" % p["valid"],
            "    output %s" % p["ready"],
        ]
    for p in outs:
        core_ports += [
            "    output %s%s" % (bus(p["width"]), p["data"]),
            "    output %s" % p["valid"],
            "    input %s" % p["ready"],
        ]
    busifs = ":".join(
        ["s_axis_%d" % i for i in range(len(ins))] + ["m_axis_%d" % j for j in range(len(outs))]
    )
    wr_ports = [
        '    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)\n'
        '    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF %s, ASSOCIATED_RESET ap_rst_n" *)\n'
        "    input ap_clk" % busifs,
        '    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)\n'
        '    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)\n'
        "    input ap_rst_n",
    ]
    conns = ["        .clk(ap_clk)", "        .rst(ap_rst_n)"]
    for k, p in enumerate(ins):
        b = "s_axis_%d" % k
        wr_ports += [
            axis_port(b, "TDATA", "input", bus(p["width"])),
            axis_port(b, "TVALID", "input"),
            axis_port(b, "TREADY", "output"),
        ]
        conns += [
            "        .%s(%s_tdata)" % (p["data"], b),
            "        .%s(%s_tvalid)" % (p["valid"], b),
            "        .%s(%s_tready)" % (p["ready"], b),
        ]
    for k, p in enumerate(outs):
        b = "m_axis_%d" % k
        wr_ports += [
            axis_port(b, "TDATA", "output", bus(p["width"])),
            axis_port(b, "TVALID", "output"),
            axis_port(b, "TREADY", "input"),
        ]
        conns += [
            "        .%s(%s_tdata)" % (p["data"], b),
            "        .%s(%s_tvalid)" % (p["valid"], b),
            "        .%s(%s_tready)" % (p["ready"], b),
        ]
    txt = (
        "// generated by finn.util.dynarapid: stitched-IP interface around the\n"
        "// DynaRapid-implemented core\n"
    )
    if black_box:
        # filled from the routed checkpoint (read_checkpoint -cell) during implementation
        txt += "(* black_box *)\nmodule %s (\n%s\n);\nendmodule\n\n" % (
            core_name,
            ",\n".join(core_ports),
        )
    txt += "module %s (\n%s\n);\n    %s core (\n%s\n    );\nendmodule\n" % (
        wrapper_name,
        ",\n".join(wr_ports),
        core_name,
        ",\n".join(conns),
    )
    return txt
