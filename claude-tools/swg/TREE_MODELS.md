# Tree models: how the other operators were built, and what has already been learnt

Context for anyone changing a `get_tree_model()`. Everything here is either
read off the committed operators or measured; the measurements are dated.

## 1. What a tree model is

`get_tree_model()` returns a `Characteristic_Node` (`src/finn/util/basic.py`),
a run-length-encoded schedule of one **period** of the operator, from which
`cumulative(periods=2)` produces the **token access vector** (TAV): two
back-to-back frames of cumulative `(input_tokens, output_tokens)` counts, one
entry per clock cycle. `DeriveTokenAccessVectors` stores it on the node and
`DeriveFIFOSizes` sizes every edge from the gap between a producer's cumulative
writes and its consumer's cumulative reads.

A node is either a **leaf** — `[(run_length, [rd, wr]), ...]`, each entry a run
of identical cycles — or a **composite** — `[(repeat_count, child), ...]`.
Both forms materialise `O(period)` in `deltas()` (the composite branch does
`np.tile`), so **nesting buys no speed**; it buys expressiveness. A composite
tree states the loop structure of the RTL; a flat leaf bakes in a cycle array.

Units are **transactions**, not bits. Any comparison of two TAVs across a node
that changes the token rate (a DWC, a pooling stride) is meaningless unless
rescaled — that mistake once inflated FIFOs 36x.

## 2. The shapes that recur

Read these four before writing a new model; between them they cover most of
what the operator set does.

**a. The `SF`/`NF` nest — `matrixvectoractivation.py`, `thresholding.py`.**
One read per cycle for `SF` cycles, a write on the cycle the accumulation
completes, repeated `NF` times, times `numVectors`. Written as a composite of
three levels. This is the default shape; reach for something else only when the
RTL genuinely departs from it.

**b. The rate-converter — `streamingdatawidthconverter.py`, `pool.py`,
`split.py`, `concat.py`.** A short repeating block whose read and write counts
differ, repeated once per input vector. The only subtlety is *where in the
block* the transaction lands; getting it one cycle wrong is a one-token phase
error, which is usually below the noise, but getting it wrong at the *period
boundary* is not — see §4.

**c. The loop-nest input generator — `outer_shuffle.py`.** The closest existing
relative of the new `input_gen`, and the model to copy for it. A circular
buffer, a linear input stream, and an output driven by a perfect loop nest.
Its model is three phases: demand-limited (input solid, output bursty),
free-limited (output solid, input admitted as the free pointer releases slots),
and a stretch where both run at one word per cycle. The helpers
`loop_nest()`, `buffer_depth()`, `free_lead()`, `demand_phase()` and
`free_pointer_steps()` are the reusable part. 108 lines, and it *declines*
(returns `None`) when the buffer does not hold a frame, rather than guessing.

**d. The counter nest — `convolutioninputgenerator.py`.** The SWG's controller
is a five-deep counter nest and its buffer is driven entirely by that nest: one
output beat per innermost iteration, and a *draw* of input slots released
whenever a level completes. The tree states that directly — a frame of rows, a
row of windows, a window of free-pointer steps — plus one leaf each for the
buffer fill, the row-boundary stall and the final drain. 209 lines, depth 2-3,
a dozen runs on a feature map that takes 300k cycles.

Executing the FSM instead is always available and is exact by construction, but
what it produces is a *trace*: one leaf, tens of thousands of runs, no
structure. That version is kept in `claude-tools/swg/swg_fsm.py` as the oracle
the nest is scored against, which is the right place for it.

## 3. Returning `None` is a supported answer

`get_tree_model()` returning `None` makes the node fall back to rtlsim
characterisation. That is slow but correct. **A wrong tree is worse than no
tree.** Every model above declines the configurations it does not describe:
`swg_default_tree` on an HLS impl style or a shape code generation refuses,
`OuterShuffle` when the buffer is smaller than a frame. Prefer a narrow model
plus a decline over a broad model plus a guess.

## 4. Learnings that cost time to acquire

*(from the FIFO-sizing work, 2026-07-27 … 2026-07-30)*

- **The period is the thing that goes wrong, not the values.** Scored against
  1119 harvested single-node references: `Thresholding_rtl` was exact on values
  for 309/309 but had the *wrong period* on 209 of them; `MVAU_rtl` 0/23,
  `VVAU_hls` 0/9, all period errors. A wrong period is a wrong frame duration,
  and the sizer accumulates that error every frame. Check the period first.

- **Do not apply a post-hoc wind-up shift.** `apply_micro_buffer_correction`
  (adding a read at the head of a period and deducting one from the tail) was
  removed because once a tree models its own wind-up the two double-count: the
  node then delivers one token per period fewer than it consumes, and the
  occupancy sum drifts every frame. 96 of 165 harvested DWC references had the
  wrong token count from that alone. **A node whose wind-up is wrong is fixed in
  its own tree model.**

- **Do not tune a tree model against the sized number.** On mobilenetv1 the
  whole tree-vs-rtlsim gap lands on three chaotically-conditioned edges, one of
  which sizes to 993 / 6017 / 8128 / 4097 / 2 depending only on which *other*
  op types are on tree models — non-additively. That is the sizer's
  conditioning, not the tree's error. Make the tree exact; use the model-level
  number only as a guard, which is exactly what `swg_model_sizes.py compare`
  is for.

- **A smaller FIFO total is not an improvement.** rtlsim is ground truth and the
  historical smaller numbers were under-provisioning. When mobilenetv1 went
  21.3 kB -> 25.0 kB against an rtlsim reference of 26.9 kB, that was 65% of the
  gap closed, in the right direction.

- **The CIG is already exact at the sizing level.** Measured 2026-07-29 across
  49 in-graph CIG nodes in 6 models: **zero** depth divergence between
  `--tree prefer` and `--tree none`. So CIG work cannot improve FIFO sizes — the
  only thing left to win there is code size, which is what this workstream is.

- **An earlier "evolved" CIG tree was evaluated and rejected** (2026-07-27).
  Its single idea was one prepended idle cycle; mean normalised error moved
  1.07% -> 1.04% and five of eight nodes got *worse*. Do not restart that line.

- **Flat leaves are avoidable more often than the docstrings claim.** VVAU_hls,
  the DWC down-conversion, `Pool_hls` for SF>=2 and Split/Concat for >=2 streams
  all have exact nested equivalents (verified bit-for-bit). MVAU_hls/_rtl
  genuinely need a flat leaf: they place transactions modulo the period and a
  nested tree cannot rotate a transaction across the frame boundary.

## 5. Where the ConvolutionInputGenerator stands (rewritten 2026-08-05)

`src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py` was 1305 lines,
of which **1012 were the tree model**. It is now 543 lines and **243**, all of
them the FSM execution:

| function | before | after | what it is |
|---|---|---|---|
| `get_tree_model` | 503 | 8 | delegate; the ~450 lines of closed forms under it were unreachable and are gone |
| `swg_default_schedule` | 148 | 69 | the "default" style FSM, transcribed from the RTL |
| `swg_parallel_schedule` | 113 | 49 | the "parallel" style FSM |
| `swg_default_params` | 104 | — | replaced: read `prepare_codegen_default()` instead |
| `swg_default_tree` | 85 | 54 | runs the FSM, phases it, run-length-encodes it |
| `swg_parallel_params` | 59 | — | replaced: read `prepare_codegen_parallel()` instead |
| `SwgController` | — | 43 | the 5-deep counter nest both schedules used to duplicate |
| `swg_params` | — | 20 | the code-generator dict, as ints |

Measure it on any revision with `python claude-tools/swg/swg_lines.py [file]`.
Compare: MVAU 144 lines, DWC 112, OuterShuffle 108, thirteen operators under 50.

**The measured finding that made this tractable:** across a 384-configuration
matrix — the convinputgenerator pytest parametrisation, the mobilenet_v1 and
resnet50 sliding windows, and a deliberate stress set (dilation, stride>kernel,
1xN maps, SIMD<IFMCh depthwise, 1x1 windows) — **every single configuration is
served by `swg_default_exact`**, the FSM execution. The closed forms below it
were never reached, confirmed by putting a `raise` at the head of the fallback
and running the matrix *and* both model builds with it in place.

They could not be reached from the normal flow either: **there is no HLS
ConvolutionInputGenerator in this tree** (only `ConvolutionInputGenerator_rtl`
is registered), so `preferred_impl_style="hls"` is warned about and forced back
to RTL, and `swg_default_tree` covers both RTL impl styles. The one probe that
did reach them, `dynamic_mode=1`, turned out not to need them: the dynamic
template only makes the loop bounds AXI-lite writable and powers up on the same
compile-time values, and it matches rtlsim at zero tolerance, so it is now
served by the FSM too.

Two things worth knowing about the rewrite:

- **Sourcing the parameters from `prepare_codegen_*` also fixed a silent
  wrong answer.** Over 2500 random off-matrix configurations the new model is
  bit-identical to the old wherever both answer (1168 configs, 0 divergences),
  and declines 56 the old one answered — every one of which is a shape whose
  RTL *cannot be generated*: `generate_hdl` fails on all 56, 41 with "H
  increment > buffer size, try setting parallel_window=1" and 15 with a
  `math domain error` in `prepare_codegen_parallel`. The old private copy of
  the algebra had no such assertion and modelled hardware that cannot exist.
  (That `select_impl_style()` accepts configurations `prepare_codegen_parallel`
  then rejects is a pre-existing FINN bug, not a tree-model one.)
- **It costs ~40% on the fast harness** (10.3 s -> 14.4 s for 384 configs,
  75 s -> 84 s for a mobilenet sizing) because `prepare_codegen_parallel` also
  builds the reg/BRAM-FIFO Verilog strings that get thrown away. Worth it: the
  alternative is the second copy of the algebra that just proved itself wrong.

Reproduce the census:

    python claude-tools/swg/swg_tav.py dump --matrix all -o /tmp/g.npz
    python -c "import numpy as np,collections; z=np.load('/tmp/g.npz',allow_pickle=True); \
      print(collections.Counter(z[k].item()['tree_name'] for k in z.files))"

## 6. The harness in this folder

- **`swg_configs.py`** — the configuration matrix (`pytest`, `models`,
  `stress`, `all`), with the pytest skip rules in one function.
- **`swg_tav.py`** — the fast loop. `dump` freezes the committed model's TAVs;
  `check` rebuilds them from the working tree and reports the divergence
  **signed by direction**: `undersize` (reads earlier or writes later than the
  reference — shrinks a FIFO, the dangerous direction) and `oversize` (costs
  depth only). 384 configurations in **~6 seconds**, no vivado, no rtlsim, no
  ipgen. Exit code is the verdict, so it doubles as a pytest.
- **`swg_model_sizes.py`** — the model-level guard. `build` runs mobilenet_v1
  or cnv-w2a2 up to (not including) FIFO sizing and leaves a checkpoint;
  `size` re-runs sizing alone on that checkpoint and reports total FIFO kB
  plus per-FIFO depths; `compare` diffs two such reports and **fails on a
  shrink** as well as on excessive growth.
- **`goldens/base.npz`** — the frozen reference, produced from the committed
  model at `feature/analytical-fifo-sizing` (43597d82).

Regenerate the golden **only** from a known-good tree (`git stash` your edit
first). A golden regenerated from a broken model makes every later check pass.
