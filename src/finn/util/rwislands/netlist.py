# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verilog of the islands and of the accelerator top.

Every FINN node is a synthesized component (module = component name) with the elastic-channel
ports of finn.util.dynarapid.components: clk, rst, dataInArray_i / pValidArray_i / readyArray_i
per input channel and dataOutArray_j / validArray_j / nReadyArray_j per output channel (AXI
streams first, then the memory-mapped channels of IODMAs).

An island instantiates its nodes (black boxes, filled from their synthesis checkpoints at link
time) and exposes every channel that leaves it. The accelerator top instantiates the islands
(black boxes, filled from their routed checkpoints by the stitcher). Channel ports are named
<node id>_din_<i>, <node id>_valid_in_<i>, <node id>_ready_out_<i> (inputs) and
<node id>_dout_<j>, <node id>_valid_out_<j>, <node id>_ready_in_<j> (outputs), the names the
pre-implemented shell (finn.util.dynarapid.shell) connects to."""

from finn.util.dynarapid.components import channels
from finn.util.dynarapid.graph import node_ids, streaming_inputs, streaming_outputs

TOP_MODULE = "finn_accel_core"


def _bus(w):
    return "[%d:0] " % (w - 1) if w > 1 else ""


def channel_graph(model):
    """Per node: id, input/output channels and stream connections.
    Returns {node name: {"id", "ins": [(name, width)], "outs": [...],
    "src": {i: (producer, j)}, "dst": {j: (consumer, i)}}}."""
    ids = node_ids(model)
    g = {}
    for n in model.graph.node:
        ins, outs = channels(n)
        g[n.name] = {"id": ids[n.name], "ins": ins, "outs": outs, "src": {}, "dst": {}}
    for n in model.graph.node:
        for j, t in enumerate(streaming_outputs(n)):
            cons = model.find_consumers(t) or []
            assert len(cons) <= 1, "%s: stream %s has several consumers" % (n.name, t)
            for c in cons:
                i = streaming_inputs(model, c).index(t)
                g[n.name]["dst"][j] = (c.name, i)
                g[c.name]["src"][i] = (n.name, j)
    return g


def _in_names(nid, i):
    return "%s_din_%d" % (nid, i), "%s_valid_in_%d" % (nid, i), "%s_ready_out_%d" % (nid, i)


def _out_names(nid, j):
    return "%s_dout_%d" % (nid, j), "%s_valid_out_%d" % (nid, j), "%s_ready_in_%d" % (nid, j)


def component_stub(dcp, node_g):
    ports = ["    input clk", "    input rst"]
    for i, (_, w) in enumerate(node_g["ins"]):
        ports += [
            "    input %sdataInArray_%d" % (_bus(w), i),
            "    input pValidArray_%d" % i,
            "    output readyArray_%d" % i,
        ]
    for j, (_, w) in enumerate(node_g["outs"]):
        ports += [
            "    output %sdataOutArray_%d" % (_bus(w), j),
            "    output validArray_%d" % j,
            "    input nReadyArray_%d" % j,
        ]
    return "(* black_box *)\nmodule %s (\n%s\n);\nendmodule\n" % (dcp, ",\n".join(ports))


def island_ports(g, members):
    """Boundary channels of an island: ([(node, i, width)] inputs, [(node, j, width)] outputs)."""
    mem = set(members)
    ins, outs = [], []
    for n in members:
        for i, (_, w) in enumerate(g[n]["ins"]):
            src = g[n]["src"].get(i)
            if src is None or src[0] not in mem:
                ins.append((n, i, w))
        for j, (_, w) in enumerate(g[n]["outs"]):
            dst = g[n]["dst"].get(j)
            if dst is None or dst[0] not in mem:
                outs.append((n, j, w))
    return ins, outs


def boundary_ports(g, members):
    """Ports of an island that connect to the accelerator boundary (the shell): the channels
    without a producer / consumer in the graph (IODMA memory-mapped and control interfaces) and
    the reset. Island-to-island channels are routed by the stitcher instead."""
    ins, outs = island_ports(g, members)
    names = ["rst"]
    for n, i, _ in ins:
        if g[n]["src"].get(i) is None:
            names += list(_in_names(g[n]["id"], i))
    for n, j, _ in outs:
        if g[n]["dst"].get(j) is None:
            names += list(_out_names(g[n]["id"], j))
    return names


def _port_decl(g, ins, outs):
    decl = ["    input clk", "    input rst"]
    for n, i, w in ins:
        d, v, r = _in_names(g[n]["id"], i)
        decl += ["    input %s%s" % (_bus(w), d), "    input %s" % v, "    output %s" % r]
    for n, j, w in outs:
        d, v, r = _out_names(g[n]["id"], j)
        decl += ["    output %s%s" % (_bus(w), d), "    output %s" % v, "    input %s" % r]
    return decl


def island_verilog(name, g, members, dcps):
    """Island module (plus black-box stubs of its components)."""
    mem = set(members)
    ins, outs = island_ports(g, members)
    body = []
    # wires of the channels inside the island, named after the producer
    for n in members:
        nid = g[n]["id"]
        for j, (_, w) in enumerate(g[n]["outs"]):
            dst = g[n]["dst"].get(j)
            if dst is not None and dst[0] in mem:
                body += [
                    "    wire %sw_%s_%d_data;" % (_bus(w), nid, j),
                    "    wire w_%s_%d_valid;" % (nid, j),
                    "    wire w_%s_%d_ready;" % (nid, j),
                ]
    # reset pipeline (rst is FINN's active-low ap_rst_n, held for many cycles by the shell):
    # the shell's reset reaches one flip-flop per island instead of fanning out unregistered into
    # every island (U55C MobileNet: the kernel reset across three SLRs was the assembly route's
    # critical path, WNS -0.99 ns, and dominated its timing-driven iterations); the island's own
    # fanout is timed and replicated inside the island's place and route. (The island is linked as
    # a structural netlist, without synthesis: primitives only.)
    body += [
        "    wire rst_q0, rst_q1;",
        "    FDRE #(.INIT(1'b0)) rst_q0_reg (.C(clk), .CE(1'b1), .R(1'b0), .D(rst), .Q(rst_q0));",
        "    FDRE #(.INIT(1'b0)) rst_q1_reg (.C(clk), .CE(1'b1), .R(1'b0), .D(rst_q0), .Q(rst_q1));",
    ]
    for n in members:
        nid = g[n]["id"]
        conns = ["        .clk(clk)", "        .rst(rst_q1)"]
        for i in range(len(g[n]["ins"])):
            src = g[n]["src"].get(i)
            if src is not None and src[0] in mem:
                pid, pj = g[src[0]]["id"], src[1]
                d, v, r = "w_%s_%d_data" % (pid, pj), "w_%s_%d_valid" % (pid, pj), "w_%s_%d_ready" % (pid, pj)
            else:
                d, v, r = _in_names(nid, i)
            conns += [
                "        .dataInArray_%d(%s)" % (i, d),
                "        .pValidArray_%d(%s)" % (i, v),
                "        .readyArray_%d(%s)" % (i, r),
            ]
        for j in range(len(g[n]["outs"])):
            dst = g[n]["dst"].get(j)
            if dst is not None and dst[0] in mem:
                d, v, r = "w_%s_%d_data" % (nid, j), "w_%s_%d_valid" % (nid, j), "w_%s_%d_ready" % (nid, j)
            else:
                d, v, r = _out_names(nid, j)
            conns += [
                "        .dataOutArray_%d(%s)" % (j, d),
                "        .validArray_%d(%s)" % (j, v),
                "        .nReadyArray_%d(%s)" % (j, r),
            ]
        body.append("    %s %s (\n%s\n    );" % (dcps[n], n, ",\n".join(conns)))
    txt = "// generated by finn.util.rwislands\n"
    for dcp in sorted({dcps[n] for n in members}):
        n0 = next(n for n in members if dcps[n] == dcp)
        txt += component_stub(dcp, g[n0])
    txt += "module %s (\n%s\n);\n%s\nendmodule\n" % (
        name,
        ",\n".join(_port_decl(g, ins, outs)),
        "\n".join(body),
    )
    return txt


def top_verilog(g, islands):
    """Accelerator top: islands (name -> member list) as black boxes, the channels between
    islands as wires, the other channels (IODMA memory-mapped interfaces) as top-level ports."""
    where = {n: k for k, mem in islands.items() for n in mem}
    top_ins, top_outs, body, stubs, wires = [], [], [], [], []
    for k, mem in islands.items():
        ins, outs = island_ports(g, mem)
        stubs.append(
            "(* black_box *)\nmodule %s (\n%s\n);\nendmodule\n" % (k, ",\n".join(_port_decl(g, ins, outs)))
        )
        conns = ["        .clk(clk)", "        .rst(rst)"]
        for n, i, w in ins:
            names = _in_names(g[n]["id"], i)
            src = g[n]["src"].get(i)
            if src is None:
                top_ins.append((n, i, w))
                sig = names
            else:
                pid, pj = g[src[0]]["id"], src[1]
                sig = ("x_%s_%d_data" % (pid, pj), "x_%s_%d_valid" % (pid, pj), "x_%s_%d_ready" % (pid, pj))
            conns += ["        .%s(%s)" % (a, b) for a, b in zip(names, sig)]
        for n, j, w in outs:
            names = _out_names(g[n]["id"], j)
            dst = g[n]["dst"].get(j)
            if dst is None:
                top_outs.append((n, j, w))
                sig = names
            else:
                nid = g[n]["id"]
                sig = ("x_%s_%d_data" % (nid, j), "x_%s_%d_valid" % (nid, j), "x_%s_%d_ready" % (nid, j))
                assert where[dst[0]] != k
                wires += [
                    "    wire %s%s;" % (_bus(w), sig[0]),
                    "    wire %s;" % sig[1],
                    "    wire %s;" % sig[2],
                ]
            conns += ["        .%s(%s)" % (a, b) for a, b in zip(names, sig)]
        body.append("    %s %s (\n%s\n    );" % (k, k, ",\n".join(conns)))
    txt = "// generated by finn.util.rwislands\n" + "".join(stubs)
    txt += "module %s (\n%s\n);\n%s\nendmodule\n" % (
        TOP_MODULE,
        ",\n".join(_port_decl(g, top_ins, top_outs)),
        "\n".join(wires + body),
    )
    return txt
