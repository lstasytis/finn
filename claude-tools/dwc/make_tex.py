#!/usr/bin/env python3
"""Generate a LaTeX (pgfplots groupplot) *figure fragment* comparing DWC-variant
resources. Reads plotdata.json (LUTs) + plotdata_ff.json (FFs), writes
dwc_resource_analysis.tex: three stacked LUT|FF pairs (one per width-scaling
case), each with its own caption. Ready to paste into an existing document.

Required preamble in the host document:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.16}
    \\usepgfplotslibrary{groupplots}
    \\usepackage{subcaption}
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(BASE, "plotdata.json")))
data_ff = json.load(open(os.path.join(BASE, "plotdata_ff.json")))

# line key -> (legend, color, style, mark)
LINES = [
    ("rtl",        "old RTL (multiple)",                     "black", "solid",  "*"),
    ("hls_mult",   "old HLS (multiple)",                     "blue",  "solid",  "square*"),
    ("hls_nomult", "old HLS (non-multiple)",                 "blue",  "dashed", "square"),
    ("gen_mult",   "new HLS with padding (multiple)",        "red",   "solid",  "triangle*"),
    ("gen_nomult", "new HLS with padding (non-multiple)",    "red",   "dashed", "triangle"),
    ("vpc_mult",   "vpc RTL (multiple)",                     "green!55!black", "solid",  "diamond*"),
    ("vpc_nomult", "vpc RTL (non-multiple)",                 "green!55!black", "dashed", "diamond"),
    ("newhls",     "new HLS hoisted (non-multiple)",         "violet",         "dashed", "otimes*"),
    ("newrtl",     "new RTL pipelined (non-multiple)",       "orange!90!black","dashed", "pentagon*"),
]
# (plot key, subfigure caption, xlabel, x-value mode, x cap)
# x cap trims points whose x exceeds it, keeping all three rows on a ~1k-bit scale.
PLOTS = [
    ("plot1", "Increasing input width (output fixed at 10). Left: LUTs, right: FFs. "
     "vpc (downscale) leaves the frame: its per-lane placement crossbar grows "
     "linearly with input width (up to 31k/62k LUT at 1000/1005 in; see analysis).",
     "input width (bits)", "in", None),
    ("plot2", "Increasing output width (input fixed at 10). Left: LUTs, right: FFs.",
     "output width (bits)", "out", None),
    ("plot3", "Increasing both widths near-equally. Left: LUTs, right: FFs.",
     "input+output (bits)", "sum", 1024),
]
# (source data, column ylabel)
COLS = [(data, "LUTs"), (data_ff, "FFs")]

XVAL = {"in": lambda i, o: i, "out": lambda i, o: o, "sum": lambda i, o: i + o}


def coords(points, xmode, xcap):
    xf = XVAL[xmode]
    out = []
    for inw, outw, val in sorted(points, key=lambda r: (r[0] + r[1])):
        if val is None or isinstance(val, str):
            continue
        x = xf(inw, outw)
        if xcap is not None and x > xcap:
            continue
        out.append((x, val))
    return out


def plots_for(plot_key, src, xmode, xcap):
    s = []
    for key, legend, color, style, mark in LINES:
        pts = coords(src.get(plot_key, {}).get(key, []), xmode, xcap)
        if not pts:
            continue
        coordstr = " ".join(f"({x:g},{y})" for x, y in pts)
        s.append(f"\\addplot[color={color},{style},mark={mark}] coordinates {{{coordstr}}};")
    return s


def legend_block():
    s = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[hide axis, scale only axis, width=1pt, height=1pt,",
        r"  xmin=0, xmax=1, ymin=0, ymax=1,",
        r"  legend columns=2,",
        r"  legend style={at={(0.5,0.5)}, anchor=center, draw=none, font=\tiny,"
        r" /tikz/every even column/.append style={column sep=8pt}},",
        r"]",
    ]
    for key, legend, color, style, mark in LINES:
        s.append(f"\\addlegendimage{{color={color},{style},mark={mark}, line width=0.7pt, mark size=1.1pt}}")
        s.append(f"\\addlegendentry{{{legend}}}")
    s += [r"\end{axis}", r"\end{tikzpicture}"]
    return s


def pair(plot_key, caption, xlabel, xmode, xcap):
    s = [
        r"\begin{subfigure}{\columnwidth}",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=0.9cm},",
        r"  width=2.9cm, height=2.2cm, scale only axis,",
        f"  xlabel={{{xlabel}}},",
        r"  label style={font=\tiny}, ylabel style={yshift=-1pt},",
        r"  grid=both, ymin=0, ymax=6000, tick label style={font=\tiny},",
        r"  scaled y ticks=false, ytick={0,2000,4000,6000}, yticklabels={0,2k,4k,6k},",
        r"  legend style={font=\tiny, inner sep=1pt, row sep=-2.5pt, column sep=3pt, draw=none, fill=white, fill opacity=0.7, text opacity=1},",
        r"  every axis plot/.append style={line width=0.6pt, mark size=1.0pt},",
        r"]",
    ]
    for src, ylabel in COLS:
        s.append(f"\\nextgroupplot[ylabel={{{ylabel}}}]")
        s += plots_for(plot_key, src, xmode, xcap)
    s += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        f"\\caption{{{caption}}}",
        r"\end{subfigure}",
    ]
    return s


doc = [
    r"% DWC resource analysis -- three LUT|FF pairs (one per width-scaling case).",
    r"% Real Vivado OOC synth (xc7z020). Paste into a document with:",
    r"%   \usepackage{pgfplots}  \pgfplotsset{compat=1.16}",
    r"%   \usepgfplotslibrary{groupplots}  \usepackage{subcaption}",
    r"% Sized for ONE column of a 2-column doc (\columnwidth); acmart styles the caption.",
    r"% For a full-width version, change 'figure' -> 'figure*' and \columnwidth -> \textwidth.",
    r"\begin{figure}[t]",
    r"\centering",
]
doc += legend_block()
doc.append(r"\par\medskip")
for idx, (pk, caption, xlabel, xmode, xcap) in enumerate(PLOTS):
    doc += pair(pk, caption, xlabel, xmode, xcap)
    if idx != len(PLOTS) - 1:
        doc.append(r"\par\medskip")
doc += [
    r"\label{fig:dwc-resources}",
    r"\end{figure}",
]

out = "\n".join(doc) + "\n"
open(os.path.join(BASE, "dwc_resource_analysis.tex"), "w").write(out)
print("wrote dwc_resource_analysis.tex (three captioned LUT|FF pairs)")
