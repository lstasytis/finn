# Handoff B result — a tree model for `input_gen.sv`, and a fix for the module

## The fix first

`finn-rtllib/mvu_tiled/input_gen.sv` **deadlocked** on 15 of the 227 in-scope
matrix configurations, including two of mobilenet_v1's own eleven sliding
windows. It has been fixed, in the working tree, and validated under Vivado
xsim. One parameter and one typedef:

```systemverilog
-	// Pointer type: one extra bit for signed wrap-around detection.
-	typedef logic signed [ADDR_BITS:0]  ptr_t;
+	localparam int unsigned  PTR_BITS = 1 + $clog2(BUF_SIZE + MAX_ABS_INC + 1);
+	typedef logic signed [PTR_BITS-1:0]  ptr_t;
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

**The bug.** `ptr_t` was sized for the buffer alone. That is right for `Wp` and
`Rp`, which are circular and only ever compared as differences within a buffer.
It is wrong for `Cap`, which is not a pointer but a *counter* — `irdy` is its
sign bit. A nest that does not read all of its feature map, which is what a
stride-2 window on an odd input is, must release the unread rows in one lump
when the frame completes; `INIT_MAX_OCCUPANCY` does not bound that lump against
`BUF_SIZE`, so the free pointer legitimately overtakes the write pointer and
`Cap` swings to `-(BUF_SIZE-1) - TERMINAL_FP_INC`. That wrapped positive, `irdy`
read back low, and the module never accepted another input word: it delivered
exactly one frame and hung. `Rp - WpZ` swings the same way and drives
`has_data`, so a wrap there reads unwritten data instead.

**Why this fix rather than a bigger buffer.** Growing `BUF_SIZE` to swallow the
lump would also work and was the obvious first move, but it costs real storage —
2x the BRAM on both mobilenet windows. Widening the pointers costs nothing: they
still address the buffer with their low `ADDR_BITS`, so `Buf` is unchanged and
only four registers gain a bit or two. With the wider counter the module does
what it was always meant to: the writer sees `irdy` high, runs on until it is
`BUF_SIZE-1` ahead of the free pointer again, and the reader picks up the next
frame in step.

**Validated.** `input_gen_rtl_check.py` now checks `odat` against the loop
nest's own address on every output beat, not just the handshake — a module can
hand back the right beats at the right cycles from the wrong addresses.

```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```
DL 2x2 s2 d2 8x8 c4 sd2    dims=[3, 3, 2, 2, 2]     buf=64    handshake True  data True
DL 2x2 s2 d2 8x8 c4 sd1    dims=[3, 3, 2, 2, 4]     buf=128   handshake True  data True
DL 2x2 s2 d2 8x8 c4 sd4    dims=[3, 3, 2, 2, 1]     buf=32    handshake True  data True
DL 2x2 s2 d21 8x8 c4 sd2   dims=[3, 4, 2, 2, 2]     buf=64    handshake True  data True
DL 2x2 s2 d2 8x8 c4 dw     dims=[3, 3, 2, 2, 2]     buf=64    handshake True  data True
DL 1x1 s2 16x16 c8 sd4     dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
DL 3x3 s2 9x9 c8 sd1 dw    dims=[4, 4, 8, 3, 3]     buf=256   handshake True  data True
DL 3x3 s2 12x12 c16 dw     dims=[5, 5, 8, 3, 3]     buf=256   handshake True  data True
DL mobilenet 58x58 s2 dw   dims=[28, 28, 32, 3, 3]  buf=4096  600000 cycles  491125 beats  True
DL mobilenet 30x30 s2 dw   dims=[14, 14, 128, 3, 3] buf=8192  620000 cycles  481753 beats  True
0 of 172 cases diverged      (22 named + 150 random, handshake and data)
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

The two mobilenet windows now deliver 2.1 frames in 600 000 cycles where the
broken version delivered exactly one and stopped. `input_gen.sv` still parses
clean and still elaborates at `mvu_tiled_axi.sv`'s own instantiation parameters;
the 192 `mvu_tiled` nests in the harness are unaffected, as are all 227 matrix
configurations that worked before.

Everything is in the working tree. Nothing is committed.

## Headline

**344 lines of functions, 430 lines of file**, against the
ConvolutionInputGenerator's **1012**. Add ~35 lines for the conv →
`(DIMS, COEFS, FM_SIZE)` mapping a sliding-window operator would carry on top.

That is up from an earlier 160/229. The increase bought a model that **derives
the loop nest top-down** instead of solving a whole period and folding the
result: no array the size of a frame ever exists, a 330 000-cycle frame costs a
dozen block solves and a few thousand elements of working memory, and the whole
matrix scores in 0.84 s instead of 4.3 s. It also, indirectly, bought the two
defects in the reference that the extra scrutiny turned up — see "What had to be
fixed".

| function | lines | what it is |
|---|---|---|
| `Walk` (class) | 147 | one loop iteration at a time, carrying only its state |
| `frame_blocks` | 39 | walk blocks until a frame repeats; fast-forward the interior |
| `nest_params` | 31 | `W` / `R_FLAG` / `TERMINAL_*_INC` / `BUF_SIZE`, transliterated |
| `tree_model` | 29 | entry point, the static declines |
| `_fold` | 28 | the levels inside a block (the one documented exception) |
| `block_pattern` | 22 | one iteration of a level, straight off `dims`/`coefs` |
| `_split`, `_runs`, `outer_level`, `block_delta` | 37 | supporting |

### Scores

```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```
$ python claude-tools/swg/input_gen_tav.py check              # sliding windows
checked 227 configs in 0.94s
worst undersize 0.0000  oversize 0.0000  period 0.0000
0/227 configs outside budget

$ python claude-tools/swg/input_gen_tav.py check --live --matrix mvu
checked 192 configs in 0.58s
worst undersize 0.0000  oversize 0.0000  period 0.0000

$ python claude-tools/swg/input_gen_tav.py drift              # 4 periods, per-frame error
200 configs, 0 with a per-frame drift    (mvu 192, models 19, stress 9: also 0)

$ python claude-tools/swg/input_gen_tav.py fuzz -n 6000
6000 nests matched exactly, 0 declined, 0 wrong

$ python claude-tools/swg/input_gen_rtl_check.py --random 150
0 of 172 cases diverged                  (handshake and data)
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

The bar was undersize ~0, oversize < 0.10, period < 0.05, and no error that
grows per frame. What it measures at is **0.0000 on all three fractions** and
zero per-frame drift over four periods — the model happens to be bit-identical
to the reference, which was a bonus rather than the contract. The reference in
turn is bit-identical to `input_gen.sv` under Vivado xsim.

### Structure

The tree is a real composite nest of `Characteristic_Node`, not a run-length
leaf, and the outer loop is *derived* rather than discovered:

| shape | leaves | runs | depth | period | peak array | blocks solved | time |
|---|---|---|---|---|---|---|---|
| 8×8 ch4 | 3 | 15 | 2 | 648 | 267 | 12 | 2 ms |
| 58×58 ch128 | 3 | 904 | 3 | 226 652 | 7 839 | 10 | 5 ms |
| 58×58 s2 dw | 2 | 3 017 | 1 | 275 088 | 27 007 | 6 | 8 ms |
| 113×113 ch32 | 4 | 450 | 3 | 222 174 | 3 865 | 10 | 3 ms |
| 224×224 ch3 | 2 | 320 | 1 | 333 976 | 10 420 | 8 | 4 ms |

Before the rewrite the shapes were the same (3/15/2, 3/904/3, 4/450/3) but they
were found by cutting up a materialised period: peak array 1.35 M elements on
58×58 rather than 7 839, and 2.84 s on the worst configuration rather than 4 ms.
For contrast the CIG's tree is depth 0, one leaf, ~50 000 runs.

**The one documented exception.** The levels *inside* a block are still folded
from that block's own cycles (`_fold`). It is bounded — a block is one iteration
of the outer loop, so the array is `period / DIMS[level]`, which is what the
"peak array" column is measuring — and deriving those too would need the same
carried state one level down for blocks small enough that the bookkeeping would
cost more than the array does. Everything outside that, which is the part that
scales with the period, is derived from `dims` and `coefs`.

## The verdict for the replace-the-CIG decision

Read this as input to the decision, not as an instruction to anyone. **The
tree-model cost is not a reason to keep the CIG** — 344 lines against 1012, and
strictly better accuracy. The deadlock that was the strongest argument against
replacement is fixed. What is left is one hard limit and one performance
question.

### 1. `input_gen` cannot express `parallel_window` at all

That style emits a whole `k*k` window per beat, so its output word is
`k_h*k_w*SIMD` wide while its input word is `SIMD` wide. `input_gen` has a
single `DATA_WIDTH` for both ports. **157 of the 384 matrix entries (41%) are
`parallel_window`**; every one would need a second module beside `input_gen`, or
a wider output stage inside it. The CIG's `swg_parallel_schedule` (113 lines +
59 of parameters) covers them today. This is the one decline in this report that
is provably a module limitation rather than a modelling one, and it is the
remaining blocker to a wholesale replacement.

### 2. On strided depthwise windows it is slower than the CIG

Measured frame periods, same configuration, CIG tree model vs `input_gen`
reference (both are per-cycle-accurate, so this is not a modelling artefact):

| window | CIG | `input_gen` | |
|---|---|---|---|
| 113×113 k3 s2 dw SIMD 8 | 231 240 | 265 448 | +15% |
| 58×58 k3 s2 dw SIMD 4 | 238 968 | 275 088 | +15% |
| 30×30 k3 s2 dw SIMD 2 | 252 988 | 288 968 | +14% |
| 28×28 k2 s2 dw SIMD 1 | 55 307 | 73 486 | +33% |
| 16×16 k3 s2 dw SIMD 1 | 281 502 | 236 540 | −16% |
| 7×7 k7 dw SIMD 1 | 99 282 | 82 900 | −17% |

Where it loses, it loses because the free pointer releases slots in bursts at a
level whose `R_FLAG` survives, and the writer waits between bursts. Worth
knowing before the swap; not a correctness problem.

### 3. The deadlock — fixed, see the top of this file

15 of 227 configurations, two of them mobilenet_v1's own, hung after exactly one
frame. `ptr_t` was sized for the buffer alone while `Cap` is a counter that
swings a terminal free increment past it. Fixed in `input_gen.sv` and validated
under xsim on handshakes *and* data, including both mobilenet windows over
600 000 cycles.

### There is no standalone sliding-window operator on `input_gen` yet

`input_gen.sv` is instantiated only inside `mvu_tiled_axi.sv`, twice, with nests
that are not sliding windows:

```systemverilog
input_gen #(.FM_SIZE(SF*TH), .D(3), .DIMS('{NF, SF, TH}), .COEFS('{0, 1, SF}))
          activation_replay (...)
input_gen #(.FM_SIZE(NF*TH), .D(2), .DIMS('{TH, NF}), .COEFS('{1, TH}))
          inst_reorder_out (...)
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

So `get_tree_model()` was not wired to any node and the model-level FIFO guard
was not run: there is no graph edge whose depth it would change. The model is
delivered standalone, as the handoff's step 6 provides for. The conv →
`(DIMS, COEFS, FM_SIZE)` mapping is therefore a *proposal*; it is a well-founded
one, but nothing in the repository commits to it.

## An independent corroboration

`mvau_tiled_params` in `matrixvectoractivation.py` carries a hand-derived buffer
and stall term for the activation replay:

```python
buf = 1 << int(math.ceil(math.log2(fm + 2)))
stall = max(0, ((TH - 1) * SF + 4 - TH) - (buf - fm))
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

Somebody derived that from the same RTL, by hand, and validated it against
rtlsim. Running the model on `DIMS = {NF, SF, TH}` over the whole range that
formula is documented valid for — SF 2…64, NF 1…6, TH 2/3/6/9, 192 combinations
— the model's period is `NF*SF*TH + stall` with **that exact stall, every time**,
and its `BUF_SIZE` is that exact `buf`: `0/192 disagree, worst stall error 0`.
Two independent derivations from the same source agreeing to the cycle.

## Coverage: no declines anywhere hardware exists

Modelled and scored exact — **227** sliding-window configurations, 192
`mvu_tiled` nests, **6000 of 6000** random nests:

* the convinputgenerator pytest parametrisation with `parallel_window=0`:
  kernels 2×2, 3×3, 1×5, strides 1/2 and 2×1, dilations 1/2 and 2×1, SIMD 1/2/4,
  depthwise and not, 8×8 and 1×21 feature maps;
* the models tier: **all eleven** mobilenet_v1 windows as the ZCU104 build folds
  them (including the padded 113 and the fused 7×7 tail) and all eight bnn-pynq
  cnv-w2a2 windows;
* the stress tier that is in scope: dilation 2×2 and 2×1, 1×N feature maps,
  SIMD < IFMCh depthwise, 1×1 windows, the SIMD=1 mobilenet tail;
* `--matrix mvu`: the two nests `mvu_tiled_axi.sv` actually instantiates, over
  TH 2/3/6/9 × SF 2…64 × NF 1…6. Not sliding windows, and they owe nothing to
  the proposed conv mapping, which is what makes them worth having;
* `fuzz`: random nests up to five levels deep with random coefficients including
  zeros and overlaps, and a quarter of them given a feature map too small for
  their nest.

Every decline path was taken back to the question "does hardware exist for
this?", and where the answer was yes it is now modelled rather than declined.
Two families came back:

**Nests addressing past `FM_SIZE`** — reading into the next feature map's words.
1112 of 6000 random nests. The input stream is continuous, so the module reads
them; it settles and conserves tokens. The guard was protecting a bookkeeping
assumption inside `Walk`, not a real limit — and the assumption was wrong: the
"forget accept times no later block needs" rule assumed the lowest future read
is monotone in the block index, which fails exactly when the outer loop strides
further than a feature map. It now takes the minimum over the next
`DIMS[level]` blocks, which is exact because a whole frame later every read is
`FM_SIZE` higher.

**Nests whose period spans several frames** — 571 more. A two-beat nest on a
four-entry buffer settles at five cycles covering *two* frames, the frames
alternating three and two for ever. That is a perfectly good steady state and
both the model and `input_gen_ref.tav` were declining it because they only
looked for one-frame periods. The model now detects the period by **the carried
state repeating** rather than by two frames matching, which finds multi-frame
periods for free and is simpler than what it replaced; `settled_window` searches
spans from one frame upward. (`_MAX_FRAMES` also had to go from 8 to 96: the
writer can take tens of frames to spend the credit `Cap` starts with. The
fast-forward keeps that cheap.)

**What is left.** One decline path, and it is unreachable:

| decline | evidence |
|---|---|
| the free pointer not handing back one frame of slots per frame | `INIT_FP_INC` telescopes to exactly that, so this is an invariant the elaboration guarantees rather than checks. 40 000 random nests never reached it. Kept so a future change that broke the invariant declines rather than emitting nonsense. |
| a block reads a word the free pointer never releases; recurrences or frames not settling in `_MAX_PASSES` / `_MAX_FRAMES` | never reached by the matrix, the `mvu_tiled` nests, or 6000 random nests |

Out of the *module's* scope, and declined before the model is asked: the 157
`parallel_window` configurations, for the port-width reason above. That is the
one decline with a proof behind it. `stride > kernel` is only legal with
`parallel_window` (or a 1×1 kernel), so the two stress entries for it are inside
that 157; 1×1 windows with stride > kernel are covered.

## On erring toward oversizing

There is no rounding choice to take a side on: the model is bit-identical to a
reference that is bit-identical to the RTL, on every configuration measured, so
it is neither over- nor under-sizing anything. The place a choice *would* arise
is the fallback — and there the conservative direction is already taken, because
a decline sends the node to rtlsim characterisation, which is ground truth. A
wrong tree can undersize a FIFO; rtlsim cannot. That is why the remaining
decline paths were left in place rather than replaced with a guess, even though
nothing reaches them.

## What had to be fixed in `input_gen_ref.py`

Two real defects, both in the direction that undersizes a FIFO. Neither was one
of the two the handoff flagged — **both of those came out clean**:

* the conv → `(DIMS, COEFS, FM_SIZE)` mapping is right, including the depthwise
  ordering. `convolutioninputgenerator_rtl.py` lines 49-51 state the two layouts
  the RTL SWG produces — non-depthwise `(OFMDim_H, OFMDim_W, K_H, K_W,
  IFMCh/SIMD, SIMD)`, depthwise `(OFMDim_H, OFMDim_W, IFMCh/SIMD, K_H, K_W,
  SIMD)` — and those are exactly the two nests `loop_nest_conv` emits;
* the 1×1 192-cycle period is real. `MAX_OCCUPANCY` works out at 1 there, so
  `BUF_SIZE` is 4 and only three words can be in flight against a three-cycle
  release round trip; the module settles at two beats per three cycles. The same
  convolution at SIMD=2 (`BUF_SIZE` 8) runs full rate, in both the RTL and the
  reference. Confirmed under xsim.

**Defect 1: unbounded pointer arithmetic.** `Wp`, `WpZ`, `Rp` and `Cap` were
Python integers; in the RTL all four are `ptr_t`. For the pointers the wrap is
the intended circular arithmetic, for `Cap` it is not — and with unbounded
integers the reference simply did not have the deadlock the module had. Found by
`input_gen_rtl_check.py --random 60`, which diverged on exactly one nest, and
chasing that one nest is what produced the RTL fix at the top of this file.
Fixed with a `ptr()` helper and three call sites, sized by `ptr_bits()` kept in
step with the RTL's own `PTR_BITS`.

**Defect 2: a fixed four-frame settling window.** `Cap` starts at
`-BUF_SIZE+1`, which hands the writer `BUF_SIZE-1` words of credit it never gets
back, and it spends that at one word per frame wherever the frame is one cycle
longer than the words it consumes. A buffer bigger than a feature map — every
`mvu_tiled` nest, and any window with a small feature map — takes tens of frames
to drain it, and a window taken inside that transient reports **one read per
frame too many**: a systematic read-early bias, which is exactly the direction
that shrinks the FIFO in front of the node.

What makes this one nasty is that repetition does not detect it. While the
credit is draining, consecutive frames are *bit-identical* — every cycle a read
— because what is changing is the credit, not anything visible in the schedule.
The invariant that does settle it is conservation: a settled frame reads exactly
one feature map and writes exactly one pass of the nest. `settled_window` now
requires that as well as repetition, and `tav` grows the window until it holds.
The model had the mirror-image bug in its own frame-repeat test and now compares
the carried state (accept queue included) too.

**Defect 3: one-frame periods only.** `tav` looked for a window of exactly one
frame. Some nests settle at a period spanning several — see Coverage. It now
searches spans from one upward, and takes the *earliest* qualifying window
rather than the latest, because with a multi-frame period which frame the window
opens on is a rotation of the whole schedule, and "the first one that settles" is
a choice a model walking forward from reset can also make.

Defects 2 and 3 were found by the fuzz, on nests the sliding-window matrix does
not reach — which is the argument for having a fuzz. Defect 1 was found by the
RTL check, which is the argument for having one of those too.

## Validation: yes, against real RTL simulation

The chain is `input_gen.sv` → `input_gen_ref.py` → `input_gen_model.py`, and
**both links are measured**.

**Link 1, RTL → reference.** Verilator is absent, but Vivado 2023.1 is on the
path and `input_gen` instantiates no DSPs, so it needs none of what makes
`mvu_tiled_axi_tb.sv` awkward. `input_gen_rtl_check.py` generates a testbench
around the module — `ivld` and `ordy` tied high, the FIFO characterisation
stimulus — runs it under `xvlog`/`xelab`/`xsim`, and checks two things: the
per-cycle `irdy`/`ovld` trace against `input_gen_ref.simulate()`, and **`odat`
against the loop nest's own address on every output beat**. The second matters:
a module can hand back the right beats at the right cycles from the wrong
addresses, and only the data check would see it. Each instance drives its own
`idat` counter, so a batch of nests shares one compile — compiling costs ~20 s
and simulating costs nothing.

```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]   buf=64     4000 cycles  True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]   buf=32     2252 cycles  True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]   buf=4      3074 cycles  True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]   buf=8      4000 cycles  True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]   buf=64     4000 cycles  True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]   buf=128    4000 cycles  True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]  buf=16     2723 cycles  True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]   buf=256    4000 cycles  True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3] buf=128    4000 cycles  True
mvu replay NF3 SF8 TH2     dims=[3, 8, 2]         buf=32      778 cycles  True
mvu reorder TH6 NF2        dims=[6, 2]            buf=16      200 cycles  True
mvu replay NF2 SF4 TH3     dims=[2, 4, 3]         buf=16      468 cycles  True
+ 150 random nests
0 of 162 cases diverged
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

**Link 2, reference → model.** The harness above. The two files also derive the
module's elaboration independently, and `input_gen_tav.py elab` checks that they
agree (0/576 disagree) — the dynamics could not hide a slip there, since one
steps every cycle and the other solves a recurrence, but the elaboration could.

Not run: `finn-rtllib/mvu_tiled/tb/mvu_tiled_axi_tb.sv` itself. It exercises
`cu_mvau_tiled`, which is DSP58, and the whole-node timing is a different
question from this module's.

## Files

| file | what it is |
|---|---|
| `input_gen_model.py` | the tree model — 430 lines, 344 of them functions |
| `input_gen_tav.py` | the harness: `dump`, `check`, `show`, `drift`, `elab`, `fuzz`; exit code is the verdict |
| `input_gen_rtl_check.py` | `input_gen.sv` under Vivado xsim, diffed against the reference |
| `goldens/input_gen_ref.npz` | 227 frozen reference TAVs |
| `input_gen_ref.py` | **corrected** — pointer widths, and a settling window that conserves tokens and may span frames |
| `finn-rtllib/mvu_tiled/input_gen.sv` | **fixed** — `PTR_BITS`; the deadlock. Working tree, not committed |

```bash
python claude-tools/swg/input_gen_tav.py dump                     # ~55 s, per-cycle reference
python claude-tools/swg/input_gen_tav.py check                    # ~1 s, the model against it
python claude-tools/swg/input_gen_tav.py check --matrix mvu --live  # the real instantiations
python claude-tools/swg/input_gen_tav.py drift                    # error constant across frames
python claude-tools/swg/input_gen_tav.py elab                     # the two elaborations agree
python claude-tools/swg/input_gen_tav.py fuzz -n 6000             # ~4 s, random nests
python claude-tools/swg/input_gen_rtl_check.py --random 150       # ~6 min, needs Vivado
python claude-tools/swg/input_gen_rtl_check.py \
       --case "DL mobilenet 58x58 s2 dw" --max-cycles 600000      # the deadlock, gone
```
3x3 s1 8x8 c4 simd2        dims=[6, 6, 3, 3, 2]     buf=64    handshake True  data True
2x2 s2 8x8 c4 simd2        dims=[4, 4, 2, 2, 2]     buf=32    handshake True  data True
1x1 s1 8x8 c8 simd4        dims=[8, 8, 1, 1, 2]     buf=4     handshake True  data True
1x1 s1 8x8 c8 simd2        dims=[8, 8, 1, 1, 4]     buf=8     handshake True  data True
3x3 s2 16x16 c8 simd8      dims=[7, 7, 3, 3, 1]     buf=64    handshake True  data True
3x3 d2 8x8 c4 simd2        dims=[4, 4, 3, 3, 2]     buf=128   handshake True  data True
1x5 s1 1x21 c4 simd2       dims=[1, 17, 1, 5, 2]    buf=16    handshake True  data True
2x2 s2 16x16 c32 simd4 dw  dims=[8, 8, 8, 2, 2]     buf=256   handshake True  data True
3x3 s1 16x16 c8 simd4 dw   dims=[14, 14, 2, 3, 3]   buf=128   handshake True  data True
+ the eight windows that used to deadlock, the two mobilenet ones,
  the three mvu_tiled nests, and 150 random nests
0 of 172 cases diverged
```

`black --line-length=100`, `isort` and `flake8 --max-line-length=100
--extend-ignore=E203` are clean on all four Python files. Two E501s elsewhere in
this folder (`swg_tav.py:42`, `swg_model_sizes.py:32`) pre-date this work.

## How the model works, in one paragraph

With `ivld` and `ordy` tied high the module is two coupled max-plus recurrences.
`advance(k) = max(advance(k-1) + 1, accept(A[k]) + 2)`: the read pointer steps to
output beat `k` either one cycle after the previous beat or two cycles after the
word that beat needs was accepted (`Wp`, then `WpZ`), and the beat leaves one
cycle later through the registered output stage. `accept(m) = max(accept(m-1) +
1, advance(gate[m]) + 1)`: input word `m` is taken either one cycle after the
previous word or one cycle after the advance that released its slot (`Cap` is
registered). `A` is the loop nest and `gate` is the inverse of the free-pointer
staircase at `BUF_SIZE - 2` words of slack. Both are monotone, so iterating from
the never-stalls schedule climbs to the least fixed point, which is the
earliest-possible and therefore the real one. The two are solved **one iteration
of the outer loop at a time**: `block_pattern` hands back that iteration's
addresses and slot releases straight off `dims` and `coefs`, `Walk.step` solves
it, and the state carried on is only the accept times a later block can still
reach back to — a window the width of the module's own buffer. As soon as a
block hands on the state it was handed, shifted, `Walk.repeat` takes the rest of
the level in a single step, and the frame is complete when it repeats the one
before it with the same carried state. Each distinct block becomes a phase of the
tree; the levels inside it are folded from its own cycles.
