# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turning FINN dataflow nodes into pre-implemented DynaRapid components.

Every node becomes one component:
  1. a single-node Vivado block design, built with the node's own IPI commands
     (the same ones CreateStitchedIP uses), so HLS IP, RTL modules and weight
     streamers are wired exactly as in the regular FINN flow;
  2. a thin Verilog adapter renaming the AXI-Stream ports to DynaRapid's
     elastic-channel convention (dataInArray_i/pValidArray_i/readyArray_i,
     dataOutArray_j/validArray_j/nReadyArray_j, clk, rst);
  3. out-of-context synthesis into DynaRapid's work directory;
  4. DynaRapid pblock generation (placement and routing inside a pblock with
     the ports routed to its boundary) and the placement database.

Components are named after a hash of everything that determines their netlist,
so identical nodes (within a model or across builds) share one library entry.
"""

import hashlib
import json
import os
import re
import uuid
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.dynarapid.tools import (
    PART_TO_DYNARAPID,
    dynarapid_env,
    dynarapid_root,
    run_java,
    run_vivado,
    vivado_version,
)

# node attributes that do not influence the generated hardware
_VOLATILE_ATTRS = {
    "code_gen_dir_cppsim",
    "code_gen_dir_ipgen",
    "ipgen_path",
    "ip_path",
    "ip_vlnv",
    "executable_path",
    "rtlsim_so",
    "rtlsim_trace",
    "exec_mode",
    "res_estimate",
    "res_hls",
    "res_synth",
    "cycles_estimate",
    "cycles_rtlsim",
    "partition_id",
    "slr",
    "mem_port",
    "device_id",
}


# bump when the way components are implemented changes (invalidates cached components)
#   2: no BRAM cascades (synth_design -max_bram_cascade_height 1)
#   3: RTL / single-cell HLS nodes synthesized from their HDL directly (no block design)
#   4: batch split drops LUT route-through cells (Vivado lost their site), split sanity check
#   5: batch split drops only 5LUT route-through cells (6LUT ones are kept)
#   6: pblock estimate: >= 85 % of the LUT cells, LUTRAM at half SLICEM density (2024.2)
#   7: pblocks built by one resize_pblock (2024.2 dropped sites); estimate of 6 reverted
# the Vivado release is part of the key as well (tools.vivado_version)
FLOW_VERSION = 7


def component_name(model, node, part, clk_ns):
    """Content-addressed component name: <optype><hash>, no underscores.

    DynaRapid parses pblock names (<dcp>_I<i>_J<j>_R<r>_C<c>) by searching for
    the first "_I", "_J", ... so the component name itself must not contain '_'.
    """
    h = hashlib.sha256()
    h.update(
        ("%s|%s|%s|%d|%s" % (node.op_type, part, clk_ns, FLOW_VERSION, vivado_version())).encode()
    )
    for a in sorted(node.attribute, key=lambda a: a.name):
        if a.name in _VOLATILE_ATTRS:
            continue
        h.update(a.SerializeToString())
    for t in node.input:
        init = model.get_initializer(t)
        if init is not None:
            h.update(init.tobytes())
            h.update(str(model.get_tensor_datatype(t)).encode())
        else:
            h.update(str(model.get_tensor_datatype(t)).encode())
    for t in node.output:
        h.update(str(model.get_tensor_datatype(t)).encode())
    short = node.op_type.replace("_", "").lower()
    # "_I"/"_J"/"_R"/"_C" are matched case-sensitively, lowercase is safe
    return "%sx%s" % (short, h.hexdigest()[:12])


def is_iodma(node):
    return node.op_type.startswith("IODMA")


def has_axi(node):
    """IODMA (AXI master + AXI-Lite control) or a compute node with an AXI-Lite slave."""
    if is_iodma(node):
        return True
    intf = getCustomOp(node).get_verilog_top_module_intf_names()
    return bool(intf.get("axilite"))


_IP_ATTRS = ("code_gen_dir_ipgen", "ipgen_path", "ip_path", "ip_vlnv")


def save_ip_attrs(node, dcp, library_dir):
    """Record where the generated IP of an HLS node is, next to its library entry."""
    inst = getCustomOp(node)
    attrs = {a: inst.get_nodeattr(a) for a in _IP_ATTRS}
    if attrs["ip_path"] and os.path.isdir(attrs["ip_path"]):
        os.makedirs(library_dir, exist_ok=True)
        with open(os.path.join(library_dir, dcp + ".ip.json"), "w") as f:
            json.dump(attrs, f)


def reuse_ip_attrs(node, dcp, library_dir):
    """Point an HLS node to the generated IP of an identical node (same component) from an
    earlier build, so that HLS synthesis is skipped. Returns True if reused."""
    f = os.path.join(library_dir, dcp + ".ip.json")
    if not os.path.isfile(f):
        return False
    attrs = json.load(open(f))
    if not all(os.path.isdir(attrs[a]) for a in ("code_gen_dir_ipgen", "ip_path")):
        return False
    inst = getCustomOp(node)
    for a, v in attrs.items():
        inst.set_nodeattr(a, v)
    return True


def stream_interfaces(node):
    """(inputs, outputs) as lists of (verilog interface name, padded width).

    Only the AXI streams; the memory-mapped interfaces of IODMA nodes are extra
    channels after these (see channels())."""
    intf = getCustomOp(node).get_verilog_top_module_intf_names()
    # compute nodes may have an AXI-Lite slave (runtime-writeable weights/thresholds): it is
    # passed through to the shell like the IODMA control interfaces
    allowed = ("axilite", "aximm") if is_iodma(node) else ("axilite",)
    unsupported = [
        k for k in ("axilite", "aximm", "ap_none", "clk2x") if intf.get(k) and k not in allowed
    ]
    assert not unsupported, "%s: interfaces %s not supported by the DynaRapid flow" % (
        node.name,
        unsupported,
    )
    return [(n, int(w)) for n, w in intf["s_axis"]], [(n, int(w)) for n, w in intf["m_axis"]]


# AXI channels: signals carried as channel data (besides VALID/READY), and whether the
# channel is driven by the master
_AXI_CHANNELS = {
    "AW": (("ADDR", "ID", "LEN", "SIZE", "BURST", "LOCK", "CACHE", "PROT", "QOS", "REGION", "USER"), True),
    "W": (("DATA", "STRB", "LAST", "ID", "USER"), True),
    "AR": (("ADDR", "ID", "LEN", "SIZE", "BURST", "LOCK", "CACHE", "PROT", "QOS", "REGION", "USER"), True),
    "R": (("DATA", "RESP", "LAST", "ID", "USER"), False),
    "B": (("RESP", "ID", "USER"), False),
}
# channel order of the memory-mapped interfaces (after the stream channels)
_AXI_IN_ORDER = [("slave", "AW"), ("slave", "W"), ("slave", "AR"), ("master", "R"), ("master", "B")]
_AXI_OUT_ORDER = [("slave", "R"), ("slave", "B"), ("master", "AW"), ("master", "W"), ("master", "AR")]


def hls_verilog_dir(node):
    """impl/verilog of an HLS node (next to the packaged IP in impl/ip)."""
    ip_path = getCustomOp(node).get_nodeattr("ip_path")
    return os.path.join(os.path.dirname(os.path.normpath(ip_path)), "verilog")


def hls_top_name(node):
    """Top module of an HLS node (named after the node at IP generation time)."""
    return getCustomOp(node).get_nodeattr("ip_vlnv").split(":")[2]


def hls_top_ports(node):
    """{port: (direction, width)} of the HLS top module, parameters at their defaults."""
    top = os.path.join(hls_verilog_dir(node), hls_top_name(node) + ".v")
    txt = open(top).read()
    params = {}
    for name, expr in re.findall(r"^\s*parameter\s+(\w+)\s*=\s*([^;]+);", txt, re.M):
        params[name] = _eval_width(expr, params)
    ports = {}
    for d, rng, name in re.findall(
        r"^\s*(input|output)\s+(?:wire\s+|reg\s+)?(\[[^\]]+\])?\s*(\w+)\s*;", txt, re.M
    ):
        width = 1
        if rng:
            msb, lsb = rng[1:-1].split(":")
            width = _eval_width(msb, params) - _eval_width(lsb, params) + 1
        ports[name] = (d, width)
    return ports


def _clog2(v):
    return max(0, (int(v) - 1).bit_length())


def _eval_width(expr, params):
    expr = expr.replace("$clog2", "_clog2").replace("**", "^^")
    expr = re.sub(
        r"\b(?!_clog2\b)[A-Za-z_]\w*\b", lambda m: str(params[m.group(0)]), expr
    ).replace("^^", "**")
    assert re.fullmatch(r"[\d\s+\-*/()_clog2]+", expr), "cannot evaluate %s" % expr
    return int(eval(expr.replace("/", "//"), {"_clog2": _clog2}))


def _verilog_ports(path, module=None):
    """{port: (direction, width)} of an ANSI Verilog module, parameters at their defaults."""
    txt = open(path).read()
    params = {}
    for name, expr in re.findall(r"\bparameter\s+(?:integer\s+)?(\w+)\s*=\s*([^,;\n]+?)\s*[,;\n]", txt):
        expr = expr.strip().rstrip(",")
        if expr.startswith('"'):
            continue
        try:
            params[name] = _eval_width(expr, params)
        except Exception:
            pass
    ports = {}
    for d, rng, name in re.findall(
        r"\b(input|output)\s+(?:wire\s+|reg\s+)?(\[[^\]]+\])?\s*(\w+)", txt
    ):
        width = 1
        if rng:
            msb, lsb = rng[1:-1].split(":")
            try:
                width = _eval_width(msb, params) - _eval_width(lsb, params) + 1
            except Exception:
                continue  # not needed (only the AXI ports are looked up)
        ports[name] = (d, width)
    return ports


def axi_port_widths(node):
    """{port: width} of the node's AXI(-Lite) ports: the HLS top module for HLS nodes, else
    the generated RTL module carrying the AXI-Lite interface (e.g. the memstream wrapper of
    an MVAU with runtime-writeable weights)."""
    try:
        return {p: w for p, (_, w) in hls_top_ports(node).items()}
    except Exception:
        pass
    ifname = getCustomOp(node).get_verilog_top_module_intf_names()["axilite"][0]
    d = getCustomOp(node).get_nodeattr("code_gen_dir_ipgen")
    for f in sorted(os.listdir(d)):
        if f.endswith((".v", ".sv")):
            path = os.path.join(d, f)
            if ifname + "_AWADDR" in open(path, errors="ignore").read():
                return {p: w for p, (_, w) in _verilog_ports(path).items()}
    raise RuntimeError("%s: AXI-Lite port widths not found" % node.name)


def axi_channels(node):
    """Memory-mapped interfaces of an IODMA node as elastic channels.

    Returns (inputs, outputs): lists of dicts {name, width, valid, ready, signals}, where
    signals is [(verilog port, width)] packed LSB first into the channel data, and name
    is <interface>_<channel> (e.g. m_axi_gmem_AR). The channels follow the fixed orders
    _AXI_IN_ORDER / _AXI_OUT_ORDER."""
    intf = getCustomOp(node).get_verilog_top_module_intf_names()
    ifs = {"slave": intf["axilite"][0]}
    if intf.get("aximm"):
        ifs["master"] = intf["aximm"][0][0]
    widths = axi_port_widths(node)

    def chan(role, ch):
        pre = "%s_%s" % (ifs[role], ch)
        sigs = [(pre + s, widths[pre + s]) for s in _AXI_CHANNELS[ch][0] if pre + s in widths]
        return {
            "name": pre,
            "width": sum(w for _, w in sigs),
            "valid": pre + "VALID",
            "ready": pre + "READY",
            "signals": sigs,
        }

    return (
        [chan(*c) for c in _AXI_IN_ORDER if c[0] in ifs],
        [chan(*c) for c in _AXI_OUT_ORDER if c[0] in ifs],
    )


def channels(node):
    """All elastic channels of a component: (inputs, outputs), lists of (name, width).
    The AXI streams come first (in interface order), so that graph connections use the
    stream index; IODMA nodes add their memory-mapped channels after them."""
    ins, outs = stream_interfaces(node)
    if has_axi(node):
        a_in, a_out = axi_channels(node)
        ins = ins + [(c["name"], c["width"]) for c in a_in]
        outs = outs + [(c["name"], c["width"]) for c in a_out]
    return ins, outs


def adapter_verilog(dcp, bd_name, n_in, n_out, in_widths, out_widths, axi=None):
    """axi: (inputs, outputs, bd interface) of an AXI-Lite slave made external in the block
    design (channels after the streams, BD ports <bd interface>_<channel><signal> lowercase)."""
    ports = ["    input clk", "    input rst"]
    conns = ["        .ap_clk(clk)", "        .ap_rst_n(rst)"]
    for i in range(n_in):
        ports += [
            "    input [%d:0] dataInArray_%d" % (in_widths[i] - 1, i),
            "    input pValidArray_%d" % i,
            "    output readyArray_%d" % i,
        ]
        conns += [
            "        .s_axis_%d_tdata(dataInArray_%d)" % (i, i),
            "        .s_axis_%d_tvalid(pValidArray_%d)" % (i, i),
            "        .s_axis_%d_tready(readyArray_%d)" % (i, i),
        ]
    for j in range(n_out):
        ports += [
            "    output [%d:0] dataOutArray_%d" % (out_widths[j] - 1, j),
            "    output validArray_%d" % j,
            "    input nReadyArray_%d" % j,
        ]
        conns += [
            "        .m_axis_%d_tdata(dataOutArray_%d)" % (j, j),
            "        .m_axis_%d_tvalid(validArray_%d)" % (j, j),
            "        .m_axis_%d_tready(nReadyArray_%d)" % (j, j),
        ]
    if axi is not None:
        a_in, a_out, bd_if = axi

        def bd_port(c, sig):
            # e.g. s_axilite_AWADDR -> s_axilite_0_awaddr
            return "%s_%s" % (bd_if, sig[len(c["name"]) - 2 :].lower())

        for k, c in enumerate(a_in, start=n_in):
            ports += [
                "    input [%d:0] dataInArray_%d" % (c["width"] - 1, k),
                "    input pValidArray_%d" % k,
                "    output readyArray_%d" % k,
            ]
            off = 0
            for sig, w in c["signals"]:
                conns.append("        .%s(dataInArray_%d[%d:%d])" % (bd_port(c, sig), k, off + w - 1, off))
                off += w
            conns += [
                "        .%s(pValidArray_%d)" % (bd_port(c, c["valid"]), k),
                "        .%s(readyArray_%d)" % (bd_port(c, c["ready"]), k),
            ]
        for k, c in enumerate(a_out, start=n_out):
            ports += [
                "    output [%d:0] dataOutArray_%d" % (c["width"] - 1, k),
                "    output validArray_%d" % k,
                "    input nReadyArray_%d" % k,
            ]
            off = 0
            for sig, w in c["signals"]:
                conns.append("        .%s(dataOutArray_%d[%d:%d])" % (bd_port(c, sig), k, off + w - 1, off))
                off += w
            conns += [
                "        .%s(validArray_%d)" % (bd_port(c, c["valid"]), k),
                "        .%s(nReadyArray_%d)" % (bd_port(c, c["ready"]), k),
            ]
    return (
        "// DynaRapid adapter, generated by finn.util.dynarapid\n"
        "// NOTE: rst is FINN's active-low ap_rst_n\n"
        "module %s (\n%s\n);\n    %s_wrapper inst (\n%s\n    );\nendmodule\n"
        % (dcp, ",\n".join(ports), bd_name, ",\n".join(conns))
    )


_HDL_EXT = (".v", ".sv", ".vh", ".svh")


def vendor_ip_cores(node):
    """Xilinx IP cores an HLS node instantiates (e.g. floating_point for float arithmetic).
    Their netlists are encrypted: RapidWright places and routes around them but cannot write
    their contents, so they would end up as black boxes in the assembled design."""
    try:
        ip_path = getCustomOp(node).get_nodeattr("ip_path")
    except AttributeError:
        return []
    d = os.path.join(ip_path, "hdl", "ip") if ip_path else ""
    return sorted(os.listdir(d)) if d and os.path.isdir(d) else []


def direct_sources(node):
    """(HDL files, top module) of nodes that can be synthesized from their HDL directly (no
    project, no block design): a single HLS IP cell, or RTL files with one module reference.
    None for other nodes (e.g. MVAU with a weight streamer hierarchy), memory init files."""
    cmds = getCustomOp(node).code_generation_ipi()
    if len(cmds) == 1:
        m = re.match(r"create_bd_cell -type ip -vlnv xilinx\.com:hls:(\w+):[\d.]+ \S+$", cmds[0])
        if m:
            vdir = hls_verilog_dir(node)
            if not os.path.isdir(vdir):
                return None
            if any(not f.endswith(_HDL_EXT) for f in os.listdir(vdir)):
                return None  # e.g. memory initialization files of HLS ROMs
            return sorted(os.path.join(vdir, f) for f in os.listdir(vdir)), m.group(1)
    files, top = [], None
    for c in cmds:
        if c.startswith("file mkdir"):
            continue
        m = re.match(r"add_files (?:-copy_to \S+ )?-norecurse (\S+)$", c)
        if m and m.group(1).endswith(_HDL_EXT):
            files.append(m.group(1))
            continue
        m = re.match(r"create_bd_cell -type module -reference (\S+) \S+$", c)
        if m and top is None:
            top = m.group(1)
            continue
        return None
    return (files, top) if files and top else None


def module_port_dirs(files, top):
    """{port: 'input'/'output'} of a Verilog/SystemVerilog module."""
    for f in files:
        txt = open(f, errors="ignore").read()
        m = re.search(r"\bmodule\s+%s\b(.*?)\bendmodule\b" % re.escape(top), txt, re.S)
        if m:
            body = re.sub(r"//[^\n]*|/\*.*?\*/|\(\*.*?\*\)", " ", m.group(1), flags=re.S)
            return {
                name: d
                for d, name in re.findall(
                    r"\b(input|output)\b(?:\s+(?:wire|reg|logic))?(?:\s*\[[^\]]*\])?\s*(\w+)",
                    body,
                )
            }
    raise AssertionError("module %s not found in %s" % (top, files))


def direct_adapter_verilog(dcp, node, top, port_dirs):
    """Adapter instantiating the node's top module directly: streams (and the AXI channels
    of IODMAs) mapped to elastic ports, channel data packed LSB first, unused inputs tied to
    0 (as in the block design), unused outputs left open."""
    s_in, s_out = stream_interfaces(node)
    a_in, a_out = axi_channels(node) if has_axi(node) else ([], [])
    try:
        widths = {p: w for p, (_, w) in hls_top_ports(node).items()}
    except Exception:
        widths = {}
    decl = ["    input clk", "    input rst"]
    conns = ["        .ap_clk(clk)", "        .ap_rst_n(rst)"]
    used = {"ap_clk", "ap_rst_n"}
    ties = []
    k = 0
    for name, w in s_in:
        decl += [
            "    input [%d:0] dataInArray_%d" % (w - 1, k),
            "    input pValidArray_%d" % k,
            "    output readyArray_%d" % k,
        ]
        tw = min(w, widths.get(name + "_TDATA", w))
        conns += [
            "        .%s_TDATA(dataInArray_%d[%d:0])" % (name, k, tw - 1),
            "        .%s_TVALID(pValidArray_%d)" % (name, k),
            "        .%s_TREADY(readyArray_%d)" % (name, k),
        ]
        used |= {name + "_TDATA", name + "_TVALID", name + "_TREADY"}
        k += 1
    for c in a_in:
        decl += [
            "    input [%d:0] dataInArray_%d" % (c["width"] - 1, k),
            "    input pValidArray_%d" % k,
            "    output readyArray_%d" % k,
        ]
        off = 0
        for sig, w in c["signals"]:
            conns.append("        .%s(dataInArray_%d[%d:%d])" % (sig, k, off + w - 1, off))
            used.add(sig)
            off += w
        conns += [
            "        .%s(pValidArray_%d)" % (c["valid"], k),
            "        .%s(readyArray_%d)" % (c["ready"], k),
        ]
        used |= {c["valid"], c["ready"]}
        k += 1
    k = 0
    for name, w in s_out:
        decl += [
            "    output [%d:0] dataOutArray_%d" % (w - 1, k),
            "    output validArray_%d" % k,
            "    input nReadyArray_%d" % k,
        ]
        tw = min(w, widths.get(name + "_TDATA", w))
        if tw < w:
            conns.append("        .%s_TDATA(dataOutArray_%d[%d:0])" % (name, k, tw - 1))
            ties.append("    assign dataOutArray_%d[%d:%d] = 0;" % (k, w - 1, tw))
        else:
            conns.append("        .%s_TDATA(dataOutArray_%d)" % (name, k))
        conns += [
            "        .%s_TVALID(validArray_%d)" % (name, k),
            "        .%s_TREADY(nReadyArray_%d)" % (name, k),
        ]
        used |= {name + "_TDATA", name + "_TVALID", name + "_TREADY"}
        k += 1
    for c in a_out:
        decl += [
            "    output [%d:0] dataOutArray_%d" % (c["width"] - 1, k),
            "    output validArray_%d" % k,
            "    input nReadyArray_%d" % k,
        ]
        off = 0
        for sig, w in c["signals"]:
            conns.append("        .%s(dataOutArray_%d[%d:%d])" % (sig, k, off + w - 1, off))
            used.add(sig)
            off += w
        conns += [
            "        .%s(validArray_%d)" % (c["valid"], k),
            "        .%s(nReadyArray_%d)" % (c["ready"], k),
        ]
        used |= {c["valid"], c["ready"]}
        k += 1
    for p, d in sorted(port_dirs.items()):
        if p not in used and d == "input":
            conns.append("        .%s(0)" % p)
    return (
        "// DynaRapid adapter (direct instantiation), generated by finn.util.dynarapid\n"
        "// NOTE: rst is FINN's active-low ap_rst_n\n"
        "module %s (\n%s\n);\n%s    %s inst (\n%s\n    );\nendmodule\n"
        % (dcp, ",\n".join(decl), "".join(t + "\n" for t in ties), top, ",\n".join(conns))
    )


def direct_synth_tcl(
    node, dcp, comp_dir, synth_dir, part, threads, files, top, metadata=True, directive=None
):
    """Vivado script for a directly synthesized component: HDL + adapter, OOC synthesis
    (non-project: no project, IP catalog or block design). metadata: also write DynaRapid's
    metadata file (needs DynaRapid's RapidWright Tcl)."""
    adapter = os.path.join(comp_dir, dcp + ".v")
    with open(adapter, "w") as f:
        f.write(direct_adapter_verilog(dcp, node, top, module_port_dirs(files, top)))
    dcp_file = os.path.join(synth_dir, dcp + "_synth.dcp")
    sv = [f for f in files if f.endswith((".sv", ".svh"))]
    v = [f for f in files if f.endswith((".v", ".vh"))]
    tcl = ["set_param general.maxThreads %d" % threads]
    if sv:
        tcl.append("read_verilog -sv [list %s]" % " ".join(sv))
    if v:
        tcl.append("read_verilog [list %s]" % " ".join(v))
    tcl += [
        "read_verilog %s" % adapter,
        # no BRAM cascades: a cascade must stay within a clock region (DRC CASC-31), which
        # restricts the relocation of the component to clock-region aligned positions
        "synth_design -top %s -part %s -mode out_of_context -max_bram_cascade_height 1%s"
        % (dcp, part, " -directive %s" % directive if directive else ""),
        "write_checkpoint -force %s" % dcp_file,
        "write_edif -force %s" % dcp_file.replace(".dcp", ".edf"),
        "report_utilization -packthru -file %s" % os.path.join(synth_dir, dcp + ".util"),
    ]
    if metadata:
        tcl += _metadata_tcl(dcp_file, synth_dir)
    return "\n".join(tcl) + "\n"


def _metadata_tcl(dcp_file, synth_dir):
    rw_tcl = os.path.join(dynarapid_root(), "RapidWright", "tcl", "rapidwright.tcl")
    return ["source %s" % rw_tcl, "generate_metadata %s %s 0" % (dcp_file, synth_dir)]


def synth_tcl(
    model, node, dcp, comp_dir, synth_dir, part, clk_ns, threads, metadata=True, directive=None
):
    """Vivado script: single-node block design + adapter, OOC synthesis, util + metadata."""
    inst = getCustomOp(node)
    ins, outs = stream_interfaces(node)
    helper = CreateStitchedIP(part, clk_ns)
    helper.create_cmds += inst.code_generation_ipi()
    helper.connect_clk_rst(node)
    helper.connect_s_axis_external(node)
    helper.connect_m_axis_external(node)
    axi = None
    if has_axi(node):
        # AXI-Lite slave of a compute node (runtime-writeable parameters): external port
        helper.connect_axi(node, model)
        a_in, a_out = axi_channels(node)
        bd_if = inst.get_verilog_top_module_intf_names()["axilite"][0] + "_0"
        axi = (a_in, a_out, bd_if)
    bd_name = dcp + "bd"
    adapter = os.path.join(comp_dir, dcp + ".v")
    with open(adapter, "w") as f:
        f.write(
            adapter_verilog(
                dcp, bd_name, len(ins), len(outs), [w for _, w in ins], [w for _, w in outs], axi
            )
        )
    ip_dirs = "$::env(FINN_ROOT)/finn-rtllib/memstream %s" % inst.get_nodeattr("ip_path")
    fclk_hz = round(1e9 / clk_ns)
    bd_file = "%s/prj.srcs/sources_1/bd/%s/%s.bd" % (comp_dir, bd_name, bd_name)
    dcp_file = os.path.join(synth_dir, dcp + "_synth.dcp")
    tcl = [
        "set_param general.maxThreads %d" % threads,
        "create_project -force prj %s -part %s" % (comp_dir, part),
        "set_msg_config -id {[BD 41-1753]} -suppress",
        "set_property ip_repo_paths [list %s] [current_project]" % ip_dirs,
        "update_ip_catalog",
        'create_bd_design "%s"' % bd_name,
    ]
    tcl += helper.create_cmds + helper.connect_cmds
    tcl += [
        "set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk]" % fclk_hz,
        "validate_bd_design",
        "save_bd_design",
        "add_files -norecurse [make_wrapper -files [get_files %s] -top]" % bd_file,
        "add_files -norecurse %s" % adapter,
        "set_property top %s [current_fileset]" % dcp,
        # synthesize the whole block design in one global run
        "set_property synth_checkpoint_mode None [get_files %s]" % bd_file,
        "generate_target all [get_files %s]" % bd_file,
        # no BRAM cascades: a cascade must stay within a clock region (DRC CASC-31), which
        # restricts the relocation of the component to clock-region aligned positions
        "synth_design -top %s -part %s -mode out_of_context -max_bram_cascade_height 1%s"
        % (dcp, part, " -directive %s" % directive if directive else ""),
        "write_checkpoint -force %s" % dcp_file,
        "write_edif -force %s" % dcp_file.replace(".dcp", ".edf"),
        "report_utilization -packthru -file %s" % os.path.join(synth_dir, dcp + ".util"),
    ]
    if metadata:
        tcl += _metadata_tcl(dcp_file, synth_dir)
    return "\n".join(tcl) + "\n"


def build_component(
    model,
    node,
    dcp,
    work_dir,
    library_dir,
    part,
    clk_ns,
    num_shapes=1,
    vivado_threads=1,
    target_util=0.8,
    pblock_mode="fast",
    pblock_parallel=1,
    stages=("synth", "pblock", "database"),
):
    """Synthesize one component and generate its pre-implemented library entry.

    Returns a dict with status and per-stage seconds. Skips all work if the
    component already exists in the library. `stages` selects the steps to run (the
    batched flow synthesizes all components first and generates the pblocks together).
    """
    res = {"dcp": dcp, "node": node.name, "op_type": node.op_type}
    if os.path.isfile(os.path.join(library_dir, dcp + ".bin.data")):
        res.update(status="cached", synth_s=0.0, pblock_s=0.0, database_s=0.0)
        return res
    env = dynarapid_env(work_dir, library_dir, part, clk_ns, vivado_threads)
    synth_dir = os.path.join(work_dir, "vhdlSynthDCPs")
    comp_dir = os.path.join(work_dir, "components", dcp)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    short = PART_TO_DYNARAPID[part]

    # 1. out-of-context synthesis of the single-node block design + adapter
    meta = os.path.join(synth_dir, dcp + "_synth_0_metadata.txt")
    rc, res["synth_s"] = 0, 0.0
    if not os.path.isfile(meta) and "synth" in stages:
        tcl_file = os.path.join(comp_dir, "synth.tcl")
        with open(tcl_file, "w") as f:
            direct = direct_sources(node)
            if direct is not None:
                f.write(
                    direct_synth_tcl(
                        node, dcp, comp_dir, synth_dir, part, vivado_threads, direct[0], direct[1]
                    )
                )
            else:
                f.write(
                    synth_tcl(model, node, dcp, comp_dir, synth_dir, part, clk_ns, vivado_threads)
                )
        rc, res["synth_s"] = run_vivado(
            tcl_file, os.path.join(comp_dir, "synth.log"), comp_dir, env
        )
    if rc != 0 or not os.path.isfile(meta):
        res["status"] = "synth_failed"
        return res
    synth_log = os.path.join(comp_dir, "synth.log")
    if (
        os.path.isfile(synth_log)
        and "could not open $readmem" in open(synth_log, errors="ignore").read()
    ):
        # a memory initialization file was not found: the netlist would have empty memories
        os.remove(meta)
        res["status"] = "synth_missing_meminit"
        return res

    if "pblock" not in stages:
        res["status"] = "synthesized"
        return res

    # 2. pblock placement and routing with exposed pins (DynaRapid + Vivado)
    # "shaped": DynaRapid's pin-exposing flow (three Vivado runs) with a compact pblock
    # "fast"  : a single Vivado run, port nets left for RWRoute at stitching time
    if pblock_mode == "fast":
        entry = "ch.agsl.dynarapid.pblockgenerator.GenerateFastPblocks"
        pargs = ["-part", short, "-m", dcp, "-util", str(target_util)]
        pargs += ["-parallel", str(pblock_parallel)]
    else:
        entry = "ch.agsl.dynarapid.pblockgenerator.GenerateShapedPblocks"
        pargs = ["-part", short, "-m", dcp, "-num", str(num_shapes), "-util", str(target_util)]
    comp_lib = os.path.join(library_dir, dcp)
    if os.path.isdir(comp_lib) and any(
        f.endswith("_placedRouted.dcp") for f in os.listdir(comp_lib)
    ):
        # pblock from an earlier (interrupted) build, only the database is missing
        rc, res["pblock_s"] = 0, 0.0
    else:
        rc, res["pblock_s"] = run_java(
            entry,
            pargs,
            env,
            os.path.join(comp_dir, "pblocks.log"),
        )
    if (
        rc != 0
        or not os.path.isdir(comp_lib)
        or not any(f.endswith("_placedRouted.dcp") for f in os.listdir(comp_lib))
    ):
        res["status"] = "pblock_failed"
        return res

    if "database" not in stages:
        res["status"] = "pblocks"
        return res

    # 3. placement database (valid relocation sites) + binary database
    rc, res["database_s"] = run_java(
        "ch.agsl.dynarapid.entry.GenerateDatabase",
        ["-part", short, "-m", dcp],
        env,
        os.path.join(comp_dir, "database.log"),
    )
    ok = rc == 0 and os.path.isfile(os.path.join(library_dir, dcp + ".bin.data"))
    res["status"] = "built" if ok else "database_failed"
    with open(os.path.join(comp_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def has_pblocks(library_dir, dcp):
    d = os.path.join(library_dir, dcp)
    # a pblock is complete once its RapidWright metadata exists (a failed batch split can
    # leave a checkpoint without it)
    return os.path.isdir(d) and any(f.endswith("_placedRouted_0_metadata.txt") for f in os.listdir(d))


def batch_pblocks(
    dcps, work_dir, library_dir, part, clk_ns, util=0.6, batches=4, threads=4, timeout_s=None
):
    """Generate the pblocks of several synthesized components in shared Vivado runs
    (DynaRapid GenerateBatchPblocks). Returns (set of components with pblocks, seconds).

    timeout_s: time limit of one batched Vivado run, after which its components are retried
    at lower utilization (default in DynaRapid: 300 s)."""
    if not dcps:
        return set(), 0.0
    env = dynarapid_env(work_dir, library_dir, part, clk_ns, threads)
    if timeout_s is not None:
        env["DYNARAPID_BATCH_TIMEOUT_S"] = str(int(timeout_s))
    os.makedirs(os.path.join(work_dir, "batches"), exist_ok=True)
    lst = os.path.join(work_dir, "batches", "components_%s.txt" % uuid.uuid4().hex[:8])
    with open(lst, "w") as f:
        f.write("\n".join(dcps) + "\n")
    log = lst.replace(".txt", ".log")
    args = ["-part", PART_TO_DYNARAPID[part], "-f", lst, "-util", str(util)]
    args += ["-batches", str(batches)]
    rc, t = run_java(
        "ch.agsl.dynarapid.pblockgenerator.GenerateBatchPblocks", args, env, log, heap="12G"
    )
    ok = set(re.findall(r"^BATCH_OK (\S+)", open(log).read(), re.M))
    return {d for d in ok if has_pblocks(library_dir, d)}, t


def batch_databases(dcps, work_dir, library_dir, part, clk_ns):
    """Placement databases of several components in one DynaRapid run. Returns seconds."""
    if not dcps:
        return 0.0
    env = dynarapid_env(work_dir, library_dir, part, clk_ns)
    os.makedirs(os.path.join(work_dir, "batches"), exist_ok=True)
    lst = os.path.join(work_dir, "batches", "databases_%s.txt" % uuid.uuid4().hex[:8])
    with open(lst, "w") as f:
        f.write("\n".join(dcps) + "\n")
    rc, t = run_java(
        "ch.agsl.dynarapid.entry.GenerateDatabase",
        ["-part", PART_TO_DYNARAPID[part], "-f", lst],
        env,
        lst.replace(".txt", ".log"),
    )
    return t


def synth_session(model_items, work_dir, library_dir, part, clk_ns, threads=1):
    """Synthesize several directly synthesizable components (direct_sources) one after the
    other in one Vivado run: the Vivado start and device load (~30 s of a ~45 s run for a
    small component) are paid once. model_items: [(dcp, node)]. Returns result dicts like
    build_component(stages=("synth",))."""
    env = dynarapid_env(work_dir, library_dir, part, clk_ns, threads)
    synth_dir = os.path.join(work_dir, "vhdlSynthDCPs")
    os.makedirs(synth_dir, exist_ok=True)
    tcl = ["set_param general.maxThreads %d" % threads, "set t0 [clock milliseconds]"]
    todo = []
    res = {}
    for dcp, node in model_items:
        r = {"dcp": dcp, "node": node.name, "op_type": node.op_type, "synth_s": 0.0}
        res[dcp] = r
        meta = os.path.join(synth_dir, dcp + "_synth_0_metadata.txt")
        if os.path.isfile(meta):
            r["status"] = "synthesized"
            continue
        comp_dir = os.path.join(work_dir, "components", dcp)
        os.makedirs(comp_dir, exist_ok=True)
        files, top = direct_sources(node)
        body = direct_synth_tcl(node, dcp, comp_dir, synth_dir, part, threads, files, top)
        tcl += [l for l in body.splitlines() if not l.startswith("set_param")]
        tcl += [
            'puts "SESSION_DONE %s [expr ([clock milliseconds] - $t0) / 1000.0]"' % dcp,
            "close_design",
            "remove_files -quiet [get_files -quiet]",
        ]
        todo.append(dcp)
    if todo:
        sess_dir = os.path.join(work_dir, "components", todo[0])
        tcl_file = os.path.join(sess_dir, "synth_session.tcl")
        with open(tcl_file, "w") as f:
            f.write("\n".join(tcl) + "\n")
        log = os.path.join(sess_dir, "synth_session.log")
        rc, t = run_vivado(tcl_file, log, sess_dir, env)
        done = dict(re.findall(r"^SESSION_DONE (\S+) ([\d.]+)", open(log, errors="ignore").read(), re.M))
        prev = 0.0
        for dcp in todo:
            meta = os.path.join(synth_dir, dcp + "_synth_0_metadata.txt")
            ok = dcp in done and os.path.isfile(meta)
            res[dcp]["status"] = "synthesized" if ok else "synth_failed"
            if dcp in done:
                res[dcp]["synth_s"] = float(done[dcp]) - prev
                prev = float(done[dcp])
    return list(res.values())


def synth_size(work_dir, dcp):
    """(LUTs, BRAM tiles) of a synthesized component from its utilization report."""
    f = os.path.join(work_dir, "vhdlSynthDCPs", dcp + ".util")
    luts, bram = 0, 0.0
    if os.path.isfile(f):
        txt = open(f, errors="ignore").read()
        m = re.search(r"\|\s*CLB LUTs\*?\s*\|\s*(\d+)", txt)
        luts = int(m.group(1)) if m else 0
        m = re.search(r"\|\s*Block RAM Tile\s*\|\s*([\d.]+)", txt)
        bram = float(m.group(1)) if m else 0.0
    return luts, bram
