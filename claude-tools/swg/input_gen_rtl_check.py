# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Run ``input_gen.sv`` under Vivado xsim and diff it against ``input_gen_ref.py``.

``input_gen_ref.py`` is a transliteration, and a transliteration is a guess until
something runs the real thing. This does: it generates a two-signal testbench
around ``input_gen`` -- ``ivld`` and ``ordy`` tied high, which is the stimulus
FIFO characterisation uses -- compiles it with ``xvlog``/``xelab``, and compares
the per-cycle ``irdy``/``ovld`` trace with the reference's.

Needs Vivado on the path (``xvlog``, ``xelab``); verilator is not used and is not
needed. Compiling costs ~20 s and simulating costs nothing, so a batch of nests
shares one testbench. It is the slow, occasional check behind
``input_gen_tav.py``'s fast one -- and the only one in the chain that is not
another reading of the same source.

    python claude-tools/swg/input_gen_rtl_check.py
    python claude-tools/swg/input_gen_rtl_check.py --case "1x1 s1 8x8 c8 simd4"
    python claude-tools/swg/input_gen_rtl_check.py --random 60 --quiet
"""

import argparse
import numpy as np
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input_gen_model import tree_model  # noqa: E402
from input_gen_ref import buf_size, loop_nest_conv, simulate  # noqa: E402

SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../finn-rtllib/mvu_tiled/input_gen.sv"
)

# No `timescale: input_gen.sv has none, and xsim refuses to mix. One testbench
# carries every nest of a batch, because xvlog + xelab cost ~18 s and the
# simulation itself costs nothing -- so the whole batch is one compile.
_UNIT = """
  localparam int unsigned DIMS%(j)d[%(d)d] = '{%(dims)s};
  localparam int unsigned COEFS%(j)d[%(d)d] = '{%(coefs)s};
  logic irdy%(j)d, ovld%(j)d;
  logic [7:0] odat%(j)d, idat%(j)d = 0;
  logic [%(d)d-1:0] olst%(j)d, odone%(j)d;
  bit [1:0] trace%(j)d[%(cycles)d];
  logic [7:0] data%(j)d[%(cycles)d];
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(%(fm)d), .D(%(d)d),
              .DIMS(DIMS%(j)d), .COEFS(COEFS%(j)d)) dut%(j)d (
    .clk(clk), .rst(rst), .idat(idat%(j)d), .ivld(ivld), .irdy(irdy%(j)d),
    .odat(odat%(j)d), .ovld(ovld%(j)d), .olst(olst%(j)d), .odone(odone%(j)d), .ordy(ordy));
  // idat is the index of the word being offered, so a correct output beat
  // carries the loop nest's own address modulo 256. Each nest accepts on its own
  // schedule, so each needs its own counter.
  always_ff @(posedge clk)  if (!rst && irdy%(j)d && ivld)  idat%(j)d <= idat%(j)d + 1;
"""
_SAMPLE = (
    "      trace%(j)d[i] = {irdy%(j)d && ivld, ovld%(j)d && ordy};\n"
    "      data%(j)d[i] = odat%(j)d;\n"
)
_REPORT = """
    $write("TRACE%(j)d ");
    for (i = 0; i < %(cycles)d; i = i + 1)  $write("%%0d%%0d", trace%(j)d[i][1], trace%(j)d[i][0]);
    $write("\\n");
    $write("DATA%(j)d ");
    for (i = 0; i < %(cycles)d; i = i + 1)
      if (trace%(j)d[i][0])  $write("%%02x", data%(j)d[i]);
    $write("\\n");
"""
_TB = """
module ig_tb;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  logic ivld = 1, ordy = 1;
  integer i;
%(units)s
  initial begin
    @(negedge clk); @(negedge clk); rst = 0;
    for (i = 0; i < %(cycles)d; i = i + 1) begin
      @(posedge clk);
%(samples)s    end
%(reports)s    $finish;
  end
endmodule
"""


def rtl_traces(batch, cycles, workdir):
    """``[(reads, writes)]`` per nest, out of one xsim run, sampled at each posedge.

    ``irdy`` and ``ovld`` are both driven straight from registers, so reading
    them from the testbench's own ``@(posedge clk)`` gives the value the cycle
    settled at, before that edge's own update.
    """
    units = samples = reports = ""
    for j, (_, (dims, coefs, fm_size)) in enumerate(batch):
        sub = dict(
            j=j,
            d=len(dims),
            dims=",".join(map(str, dims)),
            coefs=",".join(map(str, coefs)),
            fm=fm_size,
            cycles=cycles,
        )
        units += _UNIT % sub
        samples += _SAMPLE % sub
        reports += _REPORT % sub
    with open(os.path.join(workdir, "ig_tb.sv"), "w") as f:
        f.write(_TB % dict(units=units, samples=samples, reports=reports, cycles=cycles))
    subprocess.run(
        ["xvlog", "-sv", os.path.abspath(SOURCE), "ig_tb.sv"],
        cwd=workdir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    done = subprocess.run(["xelab", "-R", "ig_tb"], cwd=workdir, capture_output=True, text=True)
    out, data = {}, {}
    for line in done.stdout.splitlines():
        if line.startswith("TRACE"):
            tag, _, bits = line.strip().partition(" ")
            v = np.array(list(map(int, bits)), dtype=np.int8).reshape(-1, 2)
            out[int(tag[5:])] = (v[:, 0], v[:, 1])
        elif line.startswith("DATA"):
            tag, _, hexes = line.strip().partition(" ")
            data[int(tag[4:])] = np.array(
                [int(hexes[k : k + 2], 16) for k in range(0, len(hexes), 2)], dtype=np.int64
            )
    if len(out) != len(batch):
        raise RuntimeError(
            "xsim returned %d of %d traces:\n%s" % (len(out), len(batch), done.stdout[-2000:])
        )
    return [out[j] + (data.get(j, np.zeros(0, dtype=np.int64)),) for j in range(len(batch))]


# Sliding windows, by the mapping `loop_nest_conv` proposes, spanning what the
# matrix stresses: dilation, stride, depthwise, 1-D, and the two 1x1 buffer sizes
# that decide whether the module throttles.
WINDOWS = [
    ("3x3 s1 8x8 c4 simd2", dict(ifm_dim=(8, 8), k=(3, 3), stride=(1, 1), ifm_ch=4, simd=2)),
    ("2x2 s2 8x8 c4 simd2", dict(ifm_dim=(8, 8), k=(2, 2), stride=(2, 2), ifm_ch=4, simd=2)),
    ("1x1 s1 8x8 c8 simd4", dict(ifm_dim=(8, 8), k=(1, 1), stride=(1, 1), ifm_ch=8, simd=4)),
    ("1x1 s1 8x8 c8 simd2", dict(ifm_dim=(8, 8), k=(1, 1), stride=(1, 1), ifm_ch=8, simd=2)),
    ("3x3 s2 16x16 c8 simd8", dict(ifm_dim=(16, 16), k=(3, 3), stride=(2, 2), ifm_ch=8, simd=8)),
    (
        "3x3 d2 8x8 c4 simd2",
        dict(ifm_dim=(8, 8), k=(3, 3), stride=(1, 1), dilation=(2, 2), ifm_ch=4, simd=2),
    ),
    ("1x5 s1 1x21 c4 simd2", dict(ifm_dim=(1, 21), k=(1, 5), stride=(1, 1), ifm_ch=4, simd=2)),
    (
        "2x2 s2 16x16 c32 simd4 dw",
        dict(ifm_dim=(16, 16), k=(2, 2), stride=(2, 2), ifm_ch=32, simd=4, depthwise=1),
    ),
    (
        "3x3 s1 16x16 c8 simd4 dw",
        dict(ifm_dim=(16, 16), k=(3, 3), stride=(1, 1), ifm_ch=8, simd=4, depthwise=1),
    ),
    # The windows the narrow ptr_t deadlocked on. Every one leaves input rows
    # outside the last window -- `stride * OFMDim < IFMDim` -- so the free pointer
    # has to release them in a lump when the frame completes, and overtakes the
    # write pointer doing it.
    (
        "DL 2x2 s2 d2 8x8 c4 sd2",
        dict(ifm_dim=(8, 8), k=(2, 2), stride=(2, 2), dilation=(2, 2), ifm_ch=4, simd=2),
    ),
    (
        "DL 2x2 s2 d2 8x8 c4 sd1",
        dict(ifm_dim=(8, 8), k=(2, 2), stride=(2, 2), dilation=(2, 2), ifm_ch=4, simd=1),
    ),
    (
        "DL 2x2 s2 d2 8x8 c4 sd4",
        dict(ifm_dim=(8, 8), k=(2, 2), stride=(2, 2), dilation=(2, 2), ifm_ch=4, simd=4),
    ),
    (
        "DL 2x2 s2 d21 8x8 c4 sd2",
        dict(ifm_dim=(8, 8), k=(2, 2), stride=(2, 2), dilation=(2, 1), ifm_ch=4, simd=2),
    ),
    (
        "DL 2x2 s2 d2 8x8 c4 dw",
        dict(
            ifm_dim=(8, 8), k=(2, 2), stride=(2, 2), dilation=(2, 2), ifm_ch=4, simd=2, depthwise=1
        ),
    ),
    ("DL 1x1 s2 16x16 c8 sd4", dict(ifm_dim=(16, 16), k=(1, 1), stride=(2, 2), ifm_ch=8, simd=4)),
    (
        "DL 3x3 s2 9x9 c8 sd1 dw",
        dict(ifm_dim=(9, 9), k=(3, 3), stride=(2, 2), ifm_ch=8, simd=1, depthwise=1),
    ),
    (
        "DL 3x3 s2 12x12 c16 dw",
        dict(ifm_dim=(12, 12), k=(3, 3), stride=(2, 2), ifm_ch=16, simd=2, depthwise=1),
    ),
    # mobilenet_v1's own two. At the default cycle budget these only cover the
    # first frame, which is all the broken version ever delivered; run them with
    # --case and --max-cycles 600000 to watch the second frame arrive.
    (
        "DL mobilenet 58x58 s2 dw",
        dict(ifm_dim=(58, 58), k=(3, 3), stride=(2, 2), ifm_ch=128, simd=4, depthwise=1),
    ),
    (
        "DL mobilenet 30x30 s2 dw",
        dict(ifm_dim=(30, 30), k=(3, 3), stride=(2, 2), ifm_ch=256, simd=2, depthwise=1),
    ),
]

# The two nests `mvu_tiled_axi.sv` really instantiates, which owe nothing to that
# proposed mapping: activation replay {NF, SF, TH} and output reorder {TH, NF}.
NESTS = [
    ("mvu replay NF3 SF8 TH2", ([3, 8, 2], [0, 1, 8], 16)),
    ("mvu reorder TH6 NF2", ([6, 2], [1, 6], 12)),
    ("mvu replay NF2 SF4 TH3", ([2, 4, 3], [0, 1, 4], 12)),
]


def random_nests(count, seed, max_beats=400):
    """Nests no convolution produces: zero coefficients, transposes, overlaps.

    Only well-formed ones -- the free pointer returning exactly a frame of slots
    per frame -- since an ill-formed one drives ``Cap`` past the range its
    counter has and the RTL and the reference (unbounded Python ints) would part
    company for that reason rather than a modelling one.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        d = int(rng.integers(1, 5))
        dims = [int(rng.integers(1, 5)) for _ in range(d)]
        coefs = [int(rng.integers(0, 7)) for _ in range(d)]
        fm_size = sum((n - 1) * c for n, c in zip(dims, coefs)) + 1 + int(rng.integers(0, 4))
        if int(np.prod(dims)) > max_beats or fm_size > max_beats:
            continue
        if tree_model(dims, coefs, fm_size) is None:
            continue
        out.append(("random %d" % len(out), (dims, coefs, fm_size)))
    return out


def data_ok(dims, coefs, fm_size, odat):
    """Does every output beat carry the word the loop nest asked for?

    ``idat`` is the index of the word being offered, so output beat ``k`` of
    frame ``f`` must carry ``(A[k] + f*FM_SIZE) mod 256``. This is the check that
    a handshake trace cannot make: a module can hand back the right number of
    beats at the right cycles and still be reading the wrong addresses.
    """
    if odat.size == 0:
        return False
    beat = np.arange(int(np.prod(dims)), dtype=np.int64)
    addr = np.zeros(beat.size, dtype=np.int64)
    step = 1
    for j in range(len(dims) - 1, -1, -1):
        addr += coefs[j] * ((beat // step) % dims[j])
        step *= dims[j]
    frames = odat.size // beat.size + 1
    want = (np.concatenate([addr + f * fm_size for f in range(frames)]) % 256)[: odat.size]
    return bool(np.array_equal(want, odat))


def cases():
    out = []
    for name, kw in WINDOWS:
        kw = dict(kw)
        depthwise = kw.pop("depthwise", 0)
        kw.setdefault("dilation", (1, 1))
        out.append((name, loop_nest_conv(depthwise=depthwise, **kw)))
    return out + list(NESTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=None, help="run one case by name")
    ap.add_argument("--max-cycles", type=int, default=4000)
    ap.add_argument("--random", type=int, default=0, help="add N random loop nests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=12, help="nests per xsim compile")
    ap.add_argument("--quiet", action="store_true", help="only report divergences")
    args = ap.parse_args()

    todo = [c for c in cases() if not args.case or args.case == c[0]]
    if args.random:
        todo += random_nests(args.random, args.seed)
    bad = 0
    with tempfile.TemporaryDirectory() as workdir:
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            cycles = min(
                args.max_cycles,
                max(3 * max(int(np.prod(d)), f) + 40 for _, (d, _, f) in batch),
            )
            traces = rtl_traces(batch, cycles, workdir)
            for (name, (dims, coefs, fm_size)), (rtl_rd, rtl_wr, odat) in zip(batch, traces):
                beats = int(np.prod(dims))
                frames = cycles // beats + 2  # enough that the reference covers the whole trace
                ref_rd, ref_wr = simulate(dims, coefs, fm_size, n_frames=frames, max_cycles=cycles)
                n = min(len(rtl_rd), len(ref_rd))
                ok = np.array_equal(rtl_rd[:n], ref_rd[:n]) and np.array_equal(
                    rtl_wr[:n], ref_wr[:n]
                )
                served = data_ok(dims, coefs, fm_size, odat)
                bad += not (ok and served)
                if not (ok and served) or not args.quiet:
                    print(
                        "%-26s dims=%-22s buf=%-6d %5d cycles  handshake %-5s  data %s (%d beats)"
                        % (name, dims, buf_size(dims, coefs, fm_size), n, ok, served, odat.size)
                    )
                if not ok:
                    i = int(
                        np.flatnonzero((rtl_rd[:n] != ref_rd[:n]) | (rtl_wr[:n] != ref_wr[:n]))[0]
                    )
                    print(
                        "   first divergence at cycle %d: rtl(rd=%d,wr=%d) ref(rd=%d,wr=%d)"
                        % (i, rtl_rd[i], rtl_wr[i], ref_rd[i], ref_wr[i])
                    )
    print("%d of %d cases diverged" % (bad, len(todo)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
