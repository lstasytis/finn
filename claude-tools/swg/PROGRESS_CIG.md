# Workstream A (CIG tree model shrink) — progress log

## 2026-08-05 10:52 — started
- Verified tree: branch `feature/analytical-fifo-sizing` @ 43597d82, `import
  finn.custom_op...convolutioninputgenerator` -> `/home/lstasytis/backup/finn/src/...`
  (this checkout, not an installed copy).
- Baseline `swg_tav.py check -g goldens/base.npz`: **0/384 outside budget**,
  undersize 0.0000 / oversize 0.0000 / period 0.0000, 10.3 s.
- `$FINN_BUILD_DIR` already has `swg_fifo_mobilenet_v1` and `swg_fifo_cnv-w2a2`
  checkpoints — no rebuild needed.
- `HANDOFF_INPUT_GEN_RESULT.md` not present yet -> task still live.
- Next: read the 1305-line CIG file, confirm the dead-code finding by
  instrumenting the fallback branches.

## 2026-08-05 10:54 — INSTRUCTION OVERRIDE from coordinator
- The "check whether this task is still live" section of HANDOFF_CIG.md is
  **overridden**: this task is NOT cancelled if Workstream B lands its
  `input_gen` tree model. Stop polling for `HANDOFF_INPUT_GEN_RESULT.md`;
  ignore its verdict. Carry on shrinking the CIG for the full 10 hours.
  Everything else in the handoff stands.


## 2026-08-05 10:56 — dead-code claim confirmed independently
Instrumented the fallback body of `get_tree_model` with a `raise
RuntimeError("SWG_FALLBACK_REACHED"...)` immediately after the tier-1 return,
then ran all three consumers:
- `swg_tav.py check --matrix all` (384 configs): **0/384 outside budget**, no raise.
- `swg_model_sizes.py size --model cnv-w2a2`: 12.4559 kB — *bit-identical* to
  `goldens/cnv-w2a2_base.json`, no raise.
- `swg_model_sizes.py size --model mobilenet_v1`: 141 FIFOs, 15.5 kB (75 s), no raise.
=> the ~450 lines of closed forms are unreachable. Deleting them.

Also verified (scratch script `check_params.py`, 384/384 exact): the node's own
`prepare_codegen_default()` / `prepare_codegen_parallel()` reproduce every key
that `swg_default_params` / `swg_parallel_params` compute — **ok=384 bad=0**.
So those 163 lines can be replaced by a ~12-line dict adapter with no drift risk.

Next: rewrite. Plan = (a) one `_swg_params` adapter over prepare_codegen,
(b) one shared `_SwgController` for the 5-deep counter nest both schedules
duplicate, (c) delete the fallback, (d) `get_tree_model` becomes a delegate.

## 2026-08-05 11:04 — first landing: 1012 -> 240 lines, all guards green
Rewrote the tree model:
- deleted the whole unreachable fallback body (`k1_pass`, `dw_pw0_k2s2`,
  `pw1_k2_s3`, `pw1_generic`, the k2s2 leaf, the baseline branch) — 491 lines;
  `get_tree_model` is now an 8-line delegate.
- `swg_default_params` + `swg_parallel_params` (163 lines) -> `swg_params`
  (20 lines) reading `prepare_codegen_{default,parallel}()`. No second copy of
  the RTL parameter algebra to keep in sync.
- the 5-deep counter nest both schedules duplicated -> one `SwgController`
  class (43 lines); schedules 261 -> 118.
- RLE via `itertools.groupby`.

Measurements (`claude-tools/swg/swg_lines.py`, new AST counter, committed):
| function | before | after |
|---|---|---|
| get_tree_model | 503 | 8 |
| swg_default_schedule | 148 | 69 |
| swg_parallel_schedule | 113 | 49 |
| swg_default_params | 104 | (gone) |
| swg_default_tree | 85 | 51 |
| swg_parallel_params | 59 | (gone) |
| SwgController | — | 43 |
| swg_params | — | 20 |
| **TOTAL** | **1012** | **240** |
File: 1305 -> 537 lines.

Guards, all green:
- `swg_tav.py check` 384 configs: undersize 0.0000 oversize 0.0000 period
  0.0000, **0/384 outside budget** — the TAVs are bit-identical, not merely
  in budget.
- mobilenet_v1 compare: 15.5 kB -> 15.5 kB (+0.00%), exit 0.
- cnv-w2a2 compare: 12.5 kB -> 12.5 kB (+0.00%), exit 0.
- lint (pinned black 23.3.0 / isort 5.12.0 / flake8 6.0.0): all exit 0.
- side evidence: `prepare_codegen` params matched the deleted hand-written
  params on 384/384 configs before the swap.

Cost: `check` 10.3 s -> 14.7 s (prepare_codegen_parallel also builds the
reg/BRAM-fifo Verilog strings we throw away); mobilenet sizing 75 s -> 85 s.

Next: (1) rtlsim pytest `-m node_tree_modeling`; (2) look at whether the
parallel-style FSM has an exact closed form, which would take the total toward
the 150 stretch goal — accepted only if it is bit-for-bit on all 384.

## 2026-08-05 11:07 — rtlsim pytest passes, and the pytest is proven live
- `pytest -m node_tree_modeling tests/fpgadataflow/test_fpgadataflow_convinputgenerator.py`
  -> **13 passed**, 5184 deselected, 38 s. These are the tests that compare
  against rtlsim rather than against the frozen tree, and coverage did **not**
  narrow: `swg_default_tree` returns non-None for all 13, so all 13 still run
  under the exact assertion (`max_allowed_volume_frac = 0`).
- Mutation check (memory: "tests that report success without testing
  anything"): injected one extra `(1, [0, 0])` cycle at the head of the period
  -> **13 failed**. So the suite really does run rtlsim and really does catch a
  single cycle. Reverted.

## 2026-08-05 11:08 — the 150-line stretch goal: investigated, declined
Census over the matrix: 225 configs take the `default` impl style, 159 the
`parallel` one. The parallel style's period has a very regular shape (probe
script in scratch): 1 carry write, `FIRST_WRITE_ELEM + 1` read-only cycles,
then one `[1,1]` per controller advance followed by `addr_incr - 1` read-only
cycles — i.e. exactly the loop nest with its head increments, which a composite
`Characteristic_Node` mirrors directly. Replacing `swg_parallel_schedule`
(49 lines) with that composite would take the total to roughly 200, and the
same trick on the default style might reach 150.

**Not doing it.** The whole reason this model can be trusted is that it is
exact *by construction* for every configuration, not just for the ones in the
matrix; a closed form is only exact for the cases it was checked against, and
TREE_MODELS.md §4 records that the period is precisely what closed forms get
wrong (Thresholding_rtl: values right on 309/309, period wrong on 209).
Trading that for ~90 lines is the wrong trade, and it is the trade the 450
deleted lines already lost once. Recorded in the result doc.

Remaining budget goes to: verifying the `dynamic_mode` decline, checking a few
configurations *outside* the golden matrix, and the writeup.

## 2026-08-05 11:10 — dynamic_mode: the guard was unnecessary, and is gone
Handoff step 4 asked whether `dynamic_mode` really has to decline. Tested it
against rtlsim at **zero tolerance** (`tree_model_test(..., 0.0, 0.0, 0, 0)`,
the same comparison the real pytest uses), three configurations x
dynamic_mode in {0, 1}:

    k=[3,3] s=[1,1] dw=0  dyn=0 exact_match=True   dyn=1 exact_match=True
    k=[2,2] s=[2,2] dw=0  dyn=0 exact_match=True   dyn=1 exact_match=True
    k=[3,3] s=[1,1] dw=1  dyn=0 exact_match=True   dyn=1 exact_match=True

peak volume delta 0/0 on every one. The dynamic template only makes the loop
bounds writable over AXI-lite; it powers up holding the same compile-time
values, so its schedule is the one rtlsim measures. Guard removed — such nodes
now get an exact tree instead of an rtlsim run, for the same answer.
Total 240 -> 242 lines (the docstring explaining it). check still 0/384.

## 2026-08-05 11:12 — differential sweep vs the old model, off-matrix
Imported 43597d82's module alongside the new one and asked both for a tree on
600 random configurations (k up to 7x7, ifm up to 24x24, stride up to 4,
dilation up to 3, dw and parallel_window both ways) — i.e. deliberately outside
the golden matrix:

    same 301 | both declined 286 | differ 0 | new-only 0 | old-only 13

**Zero divergence**, and the 13 the new model declines are all configurations
where the RTL cannot be generated at all: `generate_hdl` on each one fails with
`ERROR: H increment > buffer size, try setting parallel_window=1` (11 of them)
or `ValueError: math domain error` in `prepare_codegen_parallel` (2). The old
model was emitting schedules for hardware that cannot be built. Declining them
is correct, and it is a side effect of sourcing the parameters from the code
generator instead of a private copy of the algebra.
2500-config sweep running.

## 2026-08-05 11:18 — 2500-config sweep clean; final verification round green
Big differential sweep vs 43597d82, 2500 random off-matrix configurations:

    same 1168 | both declined 1276 | differ 0 | new-only 0 | old-only 56

Checked every one of the 56: **all unbuildable** — `generate_hdl` fails on each
(41x "H increment > buffer size, try setting parallel_window=1", 15x
`ValueError: math domain error` in `prepare_codegen_parallel`). Not lost
coverage; the old private copy of the algebra was modelling hardware that
cannot be generated. (Side finding: `select_impl_style()` returns "parallel"
for shapes `prepare_codegen_parallel` then dies on — pre-existing FINN bug,
left alone.)

Final state, all re-run after the last edit:
- line count **243** (file 540)
- `swg_tav.py check --fail-under 0 --fail-over 0.10`: 0/384, undersize 0.0000
- mobilenet_v1 +0.00%, cnv-w2a2 +0.00% — and **every individual FIFO depth
  identical** (141/141 and 34/34), total_bits equal to the bit
- `pytest -m node_tree_modeling`: 13 passed
- black/isort/flake8 (pinned): 0/0/0
- `goldens/base.npz` untouched (mtime 10:15, predates this session)
- only `convolutioninputgenerator.py` modified in the tracked tree; inside the
  class only `get_tree_model` differs (AST-compared against 43597d82)

Wrote `HANDOFF_CIG_RESULT.md`; updated `TREE_MODELS.md` §2d and §5; added
`swg_lines.py` so the line count is reproducible.

DONE. Nothing committed or pushed.

## 2026-08-05 11:22 — closeout
Final numbers after a last comment edit: tree model **243** lines, file 543.
`check` 0/384 (undersize 0.0000), lint 0/0/0, pytest 13 passed, both models
+0.00% with every individual FIFO depth identical. Docs updated
(`TREE_MODELS.md` §2d/§3/§5, `HANDOFF_CIG_RESULT.md` written). Working tree has
exactly one modified tracked file. Nothing committed or pushed.

Note for anyone re-measuring: `envs/rtlsim/src/finn/` is a second, complete FINN
checkout inside this working tree (git-excluded) that still has the old
1012-line model. Check what `import finn` resolves to before trusting a number.

## 2026-08-05 11:25 — REOPENED by coordinator: build the composite
Directive overrides my decline of the stretch goal. The requirement is
structural, not line count: the model must be a composite nest of
`Characteristic_Node` mirroring the loop structure — not a per-cycle FSM trace
RLE'd into a single leaf. Measured on real mobilenet_v1 layers, what I left is

    [224,224] k3 ch3    leaves=1  runs=17545  depth=0  cycles=333998
    [113,113] k3 ch32   leaves=1  runs=49296  depth=0  cycles=222228
    [58,58]   k3 ch128  leaves=1  runs=49956  depth=0  cycles=226730

vs Workstream B's depth 3 / 3-4 leaves / few hundred runs.

Bar held: undersize 0, all 384 golden TAVs bit-identical, both model compares
+0.00%, pytest green, lint clean. Where a shape's composite cannot be shown
exact -> `return None` and list it. No approximation to keep coverage.
Plan: use the existing FSM as an oracle to discover the nest structure, write
an analytical constructor, verify bit-for-bit, then the FSM becomes test-only
scaffolding.

## 2026-08-05 11:35 — accuracy bar corrected by coordinator
Exactness is NOT required after all; the handoff's own bar applies:
`swg_tav.py check --const 8 --fail-under 0.01 --fail-over 0.10`, model FIFO
totals within a few percent. Bit-identical goldens are no longer the target.
Still binding: no *systematic* undersize bias, no per-frame-growing error (must
be checked over >1 period), composite nest, no flat trace, no rtlsim, pytest
green, lint clean. The pytest's `max_allowed_volume_frac = 0` will have to move
to the real target; I will say what I set it to.
This makes a real closed-form derivation the right move rather than a fold of
the trace. Building the structural data first.

## 2026-08-05 12:40 — composite landed; one criterion FAILS, reported not hidden
Replaced the flat FSM trace with an analytical loop nest. The FSM moved to
`claude-tools/swg/swg_fsm.py` where it is the oracle, not the model.

Structure, the numbers asked for:

| shape | before | after |
|---|---|---|
| [224,224] k3 ch3 | leaves 1, runs 17545, depth 0 | **leaves 11, runs 12, depth 2** |
| [113,113] k3 ch32 | leaves 1, runs 49296, depth 0 | **leaves 17, runs 21, depth 3** |
| [58,58] k3 ch128 | leaves 1, runs 49956, depth 0 | **leaves 11, runs 13, depth 2** |

Tree model 243 -> **209** lines. `swg_tav.py check` 14.4 s -> **0.66 s** (no
cycle is walked any more). pytest 13 passed. Lint clean.

Accuracy, 326 of 384 configs modelled (58 decline):
worst undersize 0.0375, oversize 0.2188, period 0.0776, error grows between
periods on 11/326. On cnv-w2a2's own five covered CIG nodes: period exact,
read/write totals exact, **undersize 0**, growth 0.

**FAILING:** model-level FIFO totals SHRANK. mobilenet_v1 15.5 -> 15.1 kB
(-2.6%), cnv-w2a2 12.46 -> 11.16 kB (-10.4%). A shrink is undersizing and is
the one rule. Per-node the model is not undersizing (undersize 0 on every
covered cnv node, reads late not early), so this is the sizer's conditioning
plus the three declining cnv nodes -- but I could not close it in the time and
I am not going to claim otherwise. Details and the diagnosis in the result doc.

## 2026-08-05 12:15 — REOPENED again: no declines, err oversize, measure throughput
Three directives: (1) the shrink is a throughput problem, measure rtlsim
throughput on cnv-w2a2 at baseline vs new depths; (2) the residual error must
lean systematically toward MORE buffer, not be centred; (3) **no declines where
the hardware exists** — derive the in-window depthwise stall or bound it
conservatively; 384/384 must be modelled. Keep the composite; do not go back to
the flat trace.
Order of work: understand the sizer's depth formula (my per-node "conservative"
reasoning predicted growth and got a shrink, so one of my assumptions about
direction is wrong) -> kill the declines -> re-measure -> throughput.

## 2026-08-05 12:55 — 384/384 modelled; sizer direction understood; sizes closer
**(3) declines: gone.** 384/384 configurations are modelled. Two changes:
- the depthwise in-window stall is now *bounded* rather than refused: a row
  that saturates its beats is one free-pointer step behind (`beats - epw`); a
  row that does not saturate takes its row-end draw as a burst, and the part
  that will not fit inside the window's beats stalls the fetch
  (`max(0, draw_h - beats)`). Both terms are in the nest as one leaf.
- degenerate nests (`h == 1`, `w == 1` — the 1xN feature maps and the
  single-window 3x3/7x7 depthwise cases) are assembled instead of refused.
Reads are now clamped to the frame's own budget so the token totals stay right
(379/384 exact).

**Why my "conservative" reasoning got a shrink.** The sizer runs `CHAINED_TAV`
(`swg_model_sizes.py` sets it explicitly), which is *not* the legacy
stretched-pair path. On the chained path a node's writes are pushed through a
max-plus clock driven by its own inputs: making the CIG take its words *later*
delays its whole downstream timeline and **shrinks** its output FIFO. My
round-2 placement (draw at the window tail, fill at the frame tail) was
conservative by the TAV convention and anti-conservative by this sizer. Putting
the draw back at the head of the step and the fill back at the head of the
frame -- which is also what the RTL does -- moved cnv-w2a2 **11.16 -> 12.1 kB**
against a 12.46 baseline (-10.4% -> **-2.9%**).

**(2) erring oversize.** The one strong lever left is `lead`, the words still in
hand when the beats start. Swept it: derived value `BUF-1-windup` -> 12.1 kB;
`+epw` or `+beats` -> 15.9 kB (+28%); `BUF-1` -> 22.9 kB (+84%);
`min(BUF-1, cap-per_row)` -> 22.8 kB. Nothing lands in the 12.46-13.7 window --
the lever is coarse. I kept the derived value and did **not** fit it to the kB
number: that is the trap TREE_MODELS.md 4 and the original handoff both name,
and a constant fitted to cnv-w2a2 would be wrong per node.

**(1) throughput.** `swg_throughput.py` written (forces FIFO depths from a
report, then ipgen + stitched IP + rtlsim performance). Baseline run started.

## 2026-08-05 17:20 — round 3 state
- **384/384 modelled.** No declines anywhere. Tree model 214 lines, depth 2-3.
- The in-window depthwise stall is bounded at **one free-pointer step**
  (`beats - epw`) whenever the row saturates its beats *or* its row-end draw is
  bigger than a window. The first version of this bound used the full
  `draw_h - beats`, which over-stated the period by ~18% on mobilenet's
  depthwise layers and blew the FIFO total to **65.5 kB**; the bounded form
  keeps the period within 8% everywhere.
- Score, all 384: worst under 0.1055, over 0.4893, period 0.0824, token counts
  exact 379/384, **error grows on only 14/384**, gate 300/384.
- cnv-w2a2 **12.1 kB** vs 12.46 baseline (−2.9%, was −10.4%).
- mobilenet_v1 with the *previous* (unbounded) stall was 17.6 kB (+13.5%);
  re-measuring with the bounded stall now.
- pytest: **12/13**. The one failure is `k[7,7] ifm[7,7] ch1024 simd1 dw1` — a
  frame that is a single window (h=w=1), 48.9% output-volume delta. It is the
  worst config in the score too. I tried routing its whole read budget through
  the single row; that did not move it. Not fixed.
- **Throughput:** first run wasted — `step_measure_rtlsim_performance` skipped
  itself because `RTLSIM_PERFORMANCE` was not in `generate_outputs`. Fixed in
  `swg_throughput.py` and the baseline relaunched; it did get through ipgen and
  stitched IP, so the flow works. No throughput number yet.

## 2026-08-05 17:15 — final round-3 numbers
- mobilenet_v1 **17.6 kB** vs 15.5 baseline = **+13.5%** (above baseline, past
  the +10% budget). cnv-w2a2 **12.1 kB** vs 12.46 = **−2.9%** (still a shrink).
  The two models miss the window from opposite sides, so no single constant
  fixes both — the remaining error is shape, not bias.
- 384/384 modelled, 214 lines, depth 2-3, check 0.66 s, lint clean.
- pytest 12/13 (the single-window k7x7 ch1024 depthwise case).
- Throughput: harness fixed and relaunched, no number yet. That is the first
  thing to finish, and it is what decides whether the −2.9% matters.

## 2026-08-05 17:13 — throughput runs started (both), working on the shape term
Both rtlsim throughput runs are now in flight and will finish unattended:
- `swg_throughput.py --model cnv-w2a2 --tag base2 --depths goldens/cnv-w2a2_base.json`
- `swg_throughput.py --model cnv-w2a2 --tag cand  --depths goldens/cnv-w2a2_nest.json`
(candidate depths frozen to `goldens/cnv-w2a2_nest.json`, 12.07 kB / 34 FIFOs,
so the number is reproducible against exactly this tree.)
Meanwhile: directive (3), the single-window placement law, since it is the
degenerate end of the same shape-dependent residual as (2).

## 2026-08-05 17:35 — the shape term, derived (directives 2 and 3 together)
The degenerate `k7x7 ifm7x7 ch1024 dw1` case gave up the general law. Its true
frame is 48 blocks of `[(1,[1,1]),(1023,[1,0])]`, then 1022 dense writes, then
49106 write-only cycles: reads solid while the beats wait, then beats solid --
the two-phase shape of `outer_shuffle` (TREE_MODELS 2c), not a special case.

What paces it: **beat j reads input word addr[j], and addr is the loop nest
evaluated at j**. With one word per cycle the frame cannot end before
`max_j (addr[j] - j)` cycles after its beats alone would, and that maximum is a
*sum*, not a search -- take every level as far as it goes:

    lead = SUM over levels of (trips - 1) * max(0, HEAD_INCR_level - inner_beats)

Checked by hand before coding: degenerate case 6*(1024-1) + 6*(7168-7) = 49104,
against a measured 49104 (period 99282 = 50176 beats + 49104 + 2). mobilenet
113x113 dw: 2*(2-1) + 2*(222-3) = 440, measured 440.

**There are two different waits and the frame has both** -- this is why the
first attempt at using the lead alone regressed (period 0.3713, growth 74/384):
a beat waits for the word it reads (head increments -> this lead), and a read
waits for a slot to be released (tail increments -> the row stall I already
had). Keeping both, and taking `windup = max(demand_lead, buffer_fill)`, is the
version that ships.

All 384, best numbers of the session:
worst under **0.1055**, over **0.3674** (was 0.4893), period 0.1336
(p95 0.0220, median 0.0000), token counts exact 379/384,
**growth 14/384**, gate **302/384** (was 300). 233 lines.
cnv-w2a2 12.1 kB. pytest still 12/13 -- the degenerate case improved a lot
(0.4893 -> 0.3674) but is still over the 0.25 tolerance, which I left alone.

## 2026-08-05 17:40 — throughput: harness runs, rtlsim execution fails here
Both runs get through codegen, ipgen, stitched IP and XSI compile
(`Built XSI simulation shared library ... xsimk.so`), then the baseline died in
`rtlsim_exec_cppxsi`:

    FileNotFoundError: /tmp/finn_dev_lstasytis/rtlsim_finn_design_wrapper_*/results.txt

and dropped into pdb. That is the rtlsim *execution* step, downstream of
everything the tree model touches -- an environment/tooling problem, not a
modelling one. Added `enable_build_pdb_debug=False` to `swg_throughput.py` so
it fails cleanly instead of hanging. The candidate run is at the same stage.
**Still no throughput number.** Next person: the harness and the two depth sets
are ready (`goldens/cnv-w2a2_base.json`, `goldens/cnv-w2a2_nest.json`); what
needs fixing is the cppxsi results.txt path, not the flow.

## 2026-08-05 17:23 — throughput: my diagnosis was WRONG, runs relaunched
Correction from the coordinator, confirmed here: cppxsi is **not** broken. The
live `rtlsim_xsi` (PID 3501935) is advancing normally --
`@390000 ticks / 249s: s_axis_0=5%` -- and `results.txt` is only written when
the executable finishes, so my `FileNotFoundError` meant "did not finish", not
"backend broken". I called it an environment problem on too little evidence;
the real cost is wall-clock, ~1.5-2 h per depth set.
- The live run is the **candidate** (`--tag cand`, nest depths, 12.07 kB). Left
  alone to finish.
- The **baseline** is the one that died (zero-byte log in `..._dzs9z6uw`);
  relaunched as `--tag base3` with the pdb fix.
Progress signal to watch is the tick counter in `rtlsim_xsi_log.txt`, not the
absence of output.
Working the write-pacing term while they run.

## 2026-08-05 17:45 — write pacing: two attempts, both reverted
The demand phase's writes should be paced one per address step, not dense-then-
idle. Tried it twice and reverted both:
1. **Paced lead block + trim a row's last window** to keep the write count.
   Token counts collapsed 379 -> 200/384 and growth 14 -> 64. The trim's flat
   replacement leaf also loses the per-step read placement when `steps > 1`.
2. **Lead-in as the first window of the first row** (balanced by construction,
   no trim needed). Worse still: 161/384 exact, growth 169, period frac 0.99 --
   substituting the first window drops the reads that window carried, and the
   `w - 2` interior count is wrong at `w == 1`.
The idea is right and the arithmetic for it is already in `swg_demand_lead`;
what neither attempt got right is keeping the *read* budget intact while moving
beats into the lead. Left at the version that scores best. Anyone picking this
up: make the paced block carry its share of reads too, and treat `w == 1` and
`steps > 1` explicitly -- both attempts died on those, not on the pacing law.

Shipping state (all re-verified after the reverts): 384/384 modelled, 233
lines, worst under 0.1055 / over 0.3674 / period 0.1336 (p95 0.0220, median
0.0000), tokens exact 379/384, **growth 14/384**, gate 302/384, pytest 12/13,
lint clean.

## 2026-08-05 18:45 — THROUGHPUT: candidate landed
cnv-w2a2 at **nest depths** (`goldens/cnv-w2a2_nest.json`, 12.07 kB), 64 frames,
200 MHz, rtlsim_xsi, 5111 s of wall clock:

    throughput[images/s]        1672.38
    stable_throughput[images/s] 1736.01   (interval_is_steady_state: true)
    interval_cycles             115207
    latency_cycles              395738
    TIMEOUT 0  UNFINISHED_INS 0  UNFINISHED_OUTS 0  completed_output_frames 64

Clean run -- no timeout, nothing unfinished, and the steady-state interval is
valid, so the number is trustworthy rather than an artefact of a short run.
Baseline (`base3`, 12.46 kB) is at 89% in / 84% out after 4529 s; the two are
directly comparable (same 64 frames, same clock, same flow, only the FIFO
depths differ).

## 2026-08-05 18:55 — write pacing LANDED on the third attempt
What the first two got wrong was where the borrowed beats come from. Attempt 3
takes them from the **last window of the frame** -- that window emits `paced`
fewer writes over the same cycles and the same reads -- and the lead-in emits
exactly that many, one per address step. It balances by construction and it
touches one window out of `h*w`, so nothing else moves.

    paced = clip(epw - 1, 0, min(windup, beats - 1))
    lead_in = paced x [(windup//paced - 1, [1,0]), (1, [1,1])] + remainder

On the degenerate k7x7 ch1024 frame the lead is 48 x [(1022,[1,0]),(1,[1,1])],
against a measured 48 x [(1,[1,1]),(1023,[1,0])] -- the same pacing, one cycle
out of phase. Its cumulative writes at the 10% mark go **10030 -> 44** against a
reference 10, which is the misplaced mass the coordinator identified.

All 384: gate **302 -> 309/384**, median under **0.0044 -> 0.0005**, worst under
0.1055, over 0.3678, period 0.1336, tokens exact 379/384, growth 15/384.
pytest still 12/13 and cnv-w2a2 still 12.1 kB -- the pacing fixed placement, not
the two remaining size misses.

## 2026-08-05 18:58 — THROUGHPUT: baseline == candidate, bit for bit
    candidate (12.07 kB)  interval 115207  latency 395738  cycles 7653780  1672.38 img/s
    baseline  (12.46 kB)  interval 115207  latency 395738  cycles 7653780  1672.38 img/s
Identical on every field except RUNTIME_S (5111 vs 5202 s of wall clock). Both
clean: TIMEOUT 0, UNFINISHED_INS/OUTS 0, 64/64 frames, steady-state interval
valid. The two inputs really do differ -- 8 FIFOs, in both directions.

**Not concluding anything from this yet.** An insensitive measurement produces
exactly this result too, and this repo has a history of harnesses that report
success without testing anything. Negative control launched: baseline depths
with `StreamingFIFO_rtl_15` cut **2161 -> 32** (12.46 -> 4.14 kB total), which
is the largest FIFO on the path. If `interval_cycles` rises, the harness is
sensitive and the identical result is real; if it stays at 115207, both numbers
are worthless. `goldens/cnv-w2a2_control.json`.

### mobilenet_v1 throughput: scoped, feasible, launched
- 140 non-FIFO nodes vs cnv-w2a2's ~34; max `exp_cycles` per frame **1634432**
  vs cnv's interval of 115207 -- **14x the frame**, and ~4x the nodes to
  simulate per cycle.
- cnv-w2a2 ran 7.65 M cycles in 5111 s (~1500 cycles/s). Scaling by node count,
  mobilenet should manage ~400 cycles/s, so **4 frames (~7 M cycles) is roughly
  5 h of rtlsim**, on top of a mobilenet ipgen that is hours by itself.
- Launched at `--batch 4` so `interval_cycles` still has 3 steady-state frames.
  Its ipgen overlaps the control's rtlsim, which is the best use of the clock.
- This is the run that answers the user's actual concern (15.5 kB "already
  starting to degrade"); cnv-w2a2 cannot answer it.

## 2026-08-05 19:10 — model-tier accuracy after pacing, and where the +13.5% lives
`swg_score.py --matrix models` (the mobilenet_v1 and resnet50 sliding windows):

    19/19 modelled   token counts exact 19/19
    worst under 0.0360  over 0.3678  period 0.1336
    median under 0.0011  over 0.0049  period 0.0006
    GATE 14/19 pass

All five failures are **depthwise with a large channel factor** -- exactly
mobilenet_v1's dw layers (ifm 113/58/30/16/9) -- and all five have a period
that is **short** by 1.5-3%:

    ifm[16,16] ch512 simd1 dw  per 281502 -> 273337  (-2.9%, grow +5300)
    ifm[30,30] ch256 simd2 dw  per 252988 -> 249210  (-1.5%, grow +2397)
    ifm[9,9]   ch1024 simd2 dw per 239014 -> 233976  (-2.1%, grow +5038)

A short period makes the node look faster than it is, and the sizer buffers its
neighbours against that -- which is a plausible mechanism for mobilenet's
**+13.5%**, and it says the fix is the depthwise row wait, not a global lever.
Measured for the ifm16 case: my rows carry `beats - epw` = 4599 cycles of stall
each and the real wait is ~5765, so the residual is ~1166/row of *in-window*
wait the bound does not cover. Same family as the pytest failure. Not fixed;
this is the next thing to derive, and it is one term, not a family of cases.

## 2026-08-05 19:30 — row-wait attempt 4: oracle decomposed it exactly, fix reverted
Instrumented the FSM on `k3 ifm16 ch512 simd1 dw` (period 281502, nest writes
225792, extra **55710**). Every no-write cycle in the frame, by run length:

    18 runs of 511    = HEAD_INCR_SIMD - 1    ->  9198
     2 runs of 7167   = HEAD_INCR_KW   - 1    -> 14334
     6 runs of 4596 + 1 of 4600 + 1 of 2      -> 32178   (the row stall)
                                       total  = 55710

**The row stall term is already correct** -- my `beats - epw` gives 7 x 4599 =
32193 against a measured 32178. The entire residual is on the *demand* side:
measured 23532, `swg_demand_lead` gives 15352.

The counts say why. My formula predicts `(trips-1)` gaps per level: KW is exact
(2 measured, 2 predicted) but SIMD is 18 measured against 2 predicted, a factor
of `epw` = 9. Scaling the innermost level by `epw` reproduces this config
almost exactly (2*511*9 + 2*7165 = 23528 vs 23532) -- so I tried it, gated on
`steps > 1 and w > 1` to keep the single-window case out of it:

    models tier: growth 9 -> 6, median period 0.0006 -> 0.0001, tokens 19 -> 18/19
    ALL 384:     GATE 309 -> 304, growth 15 -> **37**, tokens 379 -> 372,
                 worst under 0.1055 -> 0.2483, worst period 0.1336 -> 0.2176

**Reverted** -- improves the models tier, regresses the matrix badly. Same trap
as attempts 1 and 2. What broke: the depthwise multi-window configs where the
demand phase is *short*, which the `x epw` factor over-states; 22 more configs
picked up a period error.

The real reason a constant factor cannot work: the two waits **compound** and
do not add. Reads are not 1 per cycle -- they are throttled by the free
pointer -- so word `addr[j]` is not available at cycle `addr[j]`, and
`max_j (addr[j] - j)` under-counts by however much the read stream is behind.
The next attempt needs the coupled recurrence (the shape of Workstream B's
Kleene iteration in `input_gen_model.py`), not another factor on the sum.
Snapshot restored and re-verified: GATE 309/384, growth 15/384, tokens 379/384.

## 2026-08-05 19:45 — integration seam prepared for Workstream B
Not starting a fifth attempt on the demand term; preparing the drop-in instead.
- `swg_demand_lead(p)` in `convolutioninputgenerator.py` now documents its
  contract explicitly: parameter dict in, cycle count out, no state, one caller
  (line 213). It also records what it is short by and why, so the replacement
  does not have to rediscover it: 15352 against a frame that spends 23532 on
  `k3 ifm16 ch512 simd1 dw`, and the shortfall is structural, not a factor.
- **`claude-tools/swg/swg_integrate.py`** — scores a candidate law against the
  bar in one command, no source edit needed:

      python claude-tools/swg/swg_integrate.py --law mymod:my_lead

  It monkey-patches via `SWG_DEMAND_LEAD=module:function` (honoured by
  `swg_score.py`), runs both matrices, and prints KEEP or REVERT against
  gate_all >= 309, grow_all <= 15, tok_all >= 379, gate_models >= 14,
  tok_models >= 19. Verified on the current law: **KEEP**, all five hold.
- File ownership respected: everything above is in `claude-tools/swg/` except
  the docstring and one comment in the operator, which is my file.

## 2026-08-05 19:55 — negative control: preliminary read says the harness IS sensitive
Comparing input consumed at the same simulated tick count, control (FIFO_15 cut
2161 -> 32) against candidate (2016):

    ticks     control in%/out%    candidate in%/out%
     200000      2 / 0               2 / 0
     400000      4 / 1               5 / 1
     600000      6 / 3               8 / 3

Same total transactions (196608) in both, so at equal ticks the control has
taken **25% less input** -- the crippled FIFO is back-pressuring the design.
That is the sensitivity signal, and it points to the identical baseline/
candidate result being a real finding rather than a blind measurement.
Preliminary: the definitive number is `interval_cycles`, still ~70 min out.

## 2026-08-05 20:15 — B's solver: claim confirmed, integration not landed
Ran B's `swg_wait` against my two closed-form terms on six representative
layers:

    cfg              stride   B head   B row | my windup  my stall*h
    dw s1 113        [1,1]       448       2 |       442           0   ok
    dw s1 ifm9       [1,1]     10232    2990 |      8184           0   row missing
    dw s2 ifm16      [2,2]     23532   32178 |     15352       32193   head short
    dw s2 ifm30      [2,2]     11230   15966 |      7416       16002   head short
    nondw s1 ifm30   [1,1]       216       2 |       218           0   ok
    nondw s2 224     [2,2]      1326       5 |      1328           0   ok

**B's finding (1) confirmed and extended.** My row term is correctly silent on
the stride-1 layers where B measures row = 2 -- so the unconditional-charge
diagnosis does not apply to my version, which already guards it. But `dw s1
ifm9` has row = **2990** with stride 1, so "row exists only across a stride gap"
is not the rule either; a closed-form guard is not available. That is B's
finding (2) from the other side: head/row is a labelling, and neither half is
separately closed-form.

**B's headline claim: confirmed.** Wiring `windup = w["head"] + w["row"]` and
`stall = 0` takes the models tier's period error to

    worst frac_period 0.1336 -> **0.0001**   median 0.0000

i.e. exact by construction, which is the mechanism behind mobilenet's +13.5%.

**Not landed, and honestly:** it breaks the read accounting. My lead-in block
carries one read per cycle, so a `windup` of 23532 on a frame with 131072 words
is fine but one of 50176+ is not, and token exactness fell 19/19 -> 13/19. My
attempt to cap it (`lead_reads = min(windup, n_read)`) interacts with the paced
`debt`/`gap` split and made it worse (1/19). Reverted; `swg_integrate.py` KEEP
re-verified on all five bar items.
**What the next attempt needs:** the lead-in must distribute `min(windup,
n_read)` reads across a block of `windup` cycles *and* keep the `debt` paced
writes inside it. Those two are currently computed independently and both
assume the block is exactly `windup` long with a read every cycle. That is one
function, not a redesign -- and with B's solver the period is then exact.
**Cost to weigh:** 0.4 s median / 1.5 s worst per config against 0.5 ms now, so
the 384-config check goes 0.66 s -> minutes. Per node in a real build it is
noise. My recommendation is period-only via the solver, closed forms elsewhere.

## 2026-08-05 20:45 — B's helpers integrated: the model is far better, the kB is far worse
Integrated `allocate` + `lead_in_block` (inlined, closed-form, no numpy) with
`windup = w["head"] + w["row"]` from the solver and `stall = 0`. One extra fix
of my own was needed: the lead-in must fire `d_first + d_last`, the debt the
borrowing windows could **actually** spare, not the nominal `paced` -- `row()`
clips `debt` at `beats - end`, and that clip was the token gap.

    metric            before      after     bar
    gate_all             309        311     >=309   ok
    grow_all              15          1     <=15    ok  (15x better)
    tok_all              379        382     >=379   ok
    gate_models           14         14     >=14    ok
    tok_models            19         18     >=19    MISS by one
    worst frac_period 0.1336     0.0156
    worst frac_over   0.3678     0.0952
    models frac_period 0.1336    0.0000     exact, as B predicted
    models growth        9/19       1/19
    **pytest            12/13      13/13**  -- the single-window k7x7 case passes

Model-level totals, and this is why it is not shipping:

    cnv-w2a2   12.1 -> **13.3 kB**  (+6.7% over the 12.46 baseline, inside +10%)
    mobilenet  17.6 -> **26.6 kB**  (+72% over 15.5, far outside)

So the accurate model makes cnv-w2a2 land exactly where we wanted -- above
baseline and inside budget, the first time this session -- and blows mobilenet
up. Per-node accuracy and model-level total are not the same objective on this
sizer, which is the third time that has bitten in this session.

**Left the tree on the known-good version** (`swg_integrate.py` KEEP on all
five, cnv 12.1 / mobilenet 17.6, pytest 12/13). The solver variant is preserved
at `scratchpad/solver_variant.py`; it also still imports `swg_coupled` from
`claude-tools/`, so it is not shippable as-is regardless -- the solver would
have to move into `src/finn/` (413 lines) and costs 0.4 s median per node.
**This is a judgement call above my pay grade and I am flagging it rather than
taking it**: a strictly more accurate model that fails the mobilenet size
criterion, or a less accurate one that meets it.

## 2026-08-05 20:15 — per-edge diagnostic: the +72% is ONE edge, and the tree is right there
Sized both variants and diffed all 141 / 34 FIFOs against the **exact** golden.

**mobilenet: 9 of 141 FIFOs move. One edge is 93.8% of the growth.**

    fifo                width  exact  known-good  solver   share
    StreamingFIFO_rtl_127   8    256      256     10951    93.8%   DWC -> CIG_13
    StreamingFIFO_rtl_17   16    896     1675      1756    15.1%   DWC -> FMPadding
    (six more edges move DOWN, -3.1% .. -0.1%)
    top-3 = 105.8% of growth; edges up 3, down 6, unchanged 132

**cnv-w2a2: 8 of 34 move, top-3 = 101.7%** -- same concentrated shape, and the
top edge is `_15` (CIG -> MVAU) 2161 -> 2520, which is the one that moves it
*above* baseline.

So it is **not** broad over-placement. It is two or three edges, which is
`TREE_MODELS.md` 4's conditioning exactly.

**And the node behind the exploding edge is one the solver variant gets EXACT.**
`StreamingFIFO_rtl_127` feeds `ConvolutionInputGenerator_rtl_13`
= `ifm[9,9] k3 ch1024 simd2 dw`, one of the five failing depthwise layers:

    FSM (the exact reference)   period 239014
    known-good tree             period 233976   (-2.1%, short)
    solver variant              period 239014   <- identical to the reference

The solver variant reproduces the reference's period cycle for cycle on the
very node whose input FIFO goes 256 -> 10951. A model that matches the exact
reference cannot be "over-placing" there.

**What 256 is:** `DeriveFIFOSizes.CHAINED_TAV_THROTTLED_CAP = 256`. Both the
exact model and my known-good one land this edge *on the cap*; the solver
variant does not, and its uncapped depth is 10951. Correcting the period by
2.1% moved this node across the cap's precondition (`t_up < pacer_period`) and
the cap stopped firing. That is a threshold flip in the sizer, driven by a
*more* accurate tree -- the finding is about the sizer, not the model.

**Recommendation, now that the evidence is in:** the solver variant is the
better tree on every per-node measure *and* on the one node the +72% hangs
from. The +72% is one edge falling off a 256-deep cap. I still have not
switched the tree over, because the variant imports `swg_coupled` from
`claude-tools/` and needs the solver moved into `src/finn/` (413 lines, 0.4 s
median per node) before it is shippable -- but the diagnostic has dissolved the
binary: this is not accuracy versus size, it is one sizer cap.

## 2026-08-05 20:35 — the cap flip MEASURED; my period explanation was wrong
The coordinator caught a contradiction in my own numbers and was right. I
instrumented the sizer's per-edge trace (`claude-tools/swg/capprobe.py`,
`SWG_CAPPROBE=<dir> swg_model_sizes.py size`) and measured the throttled cap's
terms on the exploding edge under both trees:

    term              known-good      solver
    t_up                  273337      281502
    pacer_period       392302.63   392302.63
    t_up < pacer         **TRUE**    **TRUE**    <- the guard I blamed NEVER flips
    drives_pacer           false       false
    peak                    7427     **11246**
    allowance            12720.8     11862.0
    allowance / peak        1.71    **1.055**
    capped                  true     **false**
    floor / depth            256     **10951**

**It is not the pacer guard and it is not the period.** What flips is
`CHAINED_TAV_CAP_MARGIN`:

    if cap_margin > 1.0 and peak > 0 and allowance < cap_margin * peak:
        guard = False

The cap refuses to trade depth away when the estimate is marginal. Under the
known-good tree allowance/peak is 1.71 and the cap fires; under the solver
variant the **peak grows 7427 -> 11246** and the ratio falls to 1.055, below
the margin, so the cap correctly declines and the edge sizes at its real
requirement. The cap is working as designed.

**Which leaves a real defect, and it is in the solver variant.** The exact FSM
reference also lands this edge on the cap, so its peak must also be around
7427. The solver variant's period on this node is exact (239014 = FSM) but its
TAV *shape* produces a 51% larger peak occupancy. So the divergence from the
exact reference is in the read pattern of `ConvolutionInputGenerator_rtl_13`
itself, not upstream, and the +72% is that divergence surfacing through a
threshold rather than pure sizer conditioning. **The solver variant is not
ready**; it needs its occupancy shape checked against `base.npz` per cycle on
this node.

Still worth recording plainly: **a 51% peak-estimate change causes a 43x depth
change (256 -> 10951) on one edge**, because a threshold sits between them.
That is TREE_MODELS.md 4 with a number attached, and it is why the model-level
kB total is a poor objective for tree work.

Tree left on the known-good version (`swg_integrate.py` KEEP, all five).

## 2026-08-05 21:05 — B's peak fix: right diagnosis, integration not landed
B's correction is clearly the right one -- the lead-in must leave the read
stream `peak` ahead, not take `windup` reads, and `windup` lumps the row share
at the head. Its own numbers show that was wrong in *both* directions (ifm58 s2
+249%, ifm5 s1 **-100%**), so it is a correction, not a margin.

Integrated `swg_frame_reads(p, windup, peak, ...)` + `lead_in_node`. Three
passes, none green on all five:

    variant                                    gate_all grow tok_all gate_m tok_m
    lead reads = fr["lead_in"] - paced              310    1     332     13     9
    lead reads = fr["lead_in"]                      310    1     349     13    16
    lead reads = peak + actual debt                 298    1   **382**   13    18
    re-share after the debt is known                300    1     375     13    18

`grow_all` is **1** in every one of them (against 15 now), and the token totals
come right once the lead-in is balanced against the *actual* debt the rows
could spare. What I could not close is the circularity: `_frame_reads` needs
`paced` to size the lead-in, `row()` clips `paced` to `beats - end`, and
re-sharing with the clipped value moves the read placement enough to cost
gate_all. It needs `row()` to report what it can spare *before* the share is
computed -- one refactor, not a redesign.

Variant preserved at `scratchpad/solver_variant_v2.py`. Tree left on the
known-good version, `swg_integrate.py` KEEP on all five.

**Honest limitation for the write-up, per B:** even when this lands, totals,
period and peak are exact but the **mid-frame curve is not** -- lumping the wait
at the head is exactly that approximation. Placing the `row` share at the row
boundaries is what would remove it.

## 2026-08-05 21:20 — refactor attempted twice more, stopping per the boundary
5. share fixed as `_frame_reads` allocated it, lead-in's *writes* follow the
   clip: gate_all 300, grow 1, **tok_all 382**, gate_m 13, tok_m 18
6. pacing removed entirely (`epw=1`, no borrowed beats, no circularity):
   gate_all **298**, grow 1, tok_all 382, gate_m 13, tok_m 18

Removing the circularity did **not** recover `gate_all`, which rules out my own
diagnosis: the 9-config shortfall is not the paced/debt clip. Every one of the
six variants holds `grow_all` at 1 and lands `tok_all` at 382 once the lead-in
balances, so the structure is right and the residual is read *placement* --
which is precisely the mid-frame curve approximation B named. Lumping the row
share at the head is what those 9 configs are measuring, and no amount of
rebalancing the head will fix it; the `row` share has to move to the row
boundaries.

Stopping here per the coordinator's boundary rather than opening a seventh.
Variants preserved (`solver_variant.py`, `_v2`, `_v3`). Tree on the known-good
version, `swg_integrate.py` KEEP on all five.

## 2026-08-05 21:30 — NEGATIVE CONTROL LANDED; write-up complete
                            interval_cycles  throughput  stable
    baseline  (12.46 kB)         115207       1672.4    1736.0
    candidate (12.07 kB)         115207       1672.4    1736.0
    control   ( 4.14 kB)      -> 142043    -> 1369.8    1408.0

Cutting FIFO_15 2161 -> 32 moves interval **+23.3%** and throughput **-18.1%**,
clean run (TIMEOUT 0, nothing unfinished, 64/64 frames, steady interval valid).
**The harness sees FIFO depth**, so the identical baseline/candidate result is a
real finding. It licenses "cnv-w2a2 is insensitive to depth in the 12.07-12.46
kB range" -- not "the shrink is safe" -- and does not transfer to mobilenet_v1,
whose run is still going.

`HANDOFF_CIG_RESULT.md` rewritten to stand alone: headline, the throughput
measurement with its control, what the model is and the two derived laws, the
eleven reverted attempts in one table, the cap-margin finding with capprobe,
B's solver and the packaging question, and an explicit "what is not done".

FINAL STATE: 384/384 modelled, 270 lines (from 1012), file 574 (from 1305),
check 0.66 s (from 10.3 s), gate 309/384, growth 15/384, tokens 379/384,
pytest 12/13, lint clean, `swg_integrate.py` KEEP on all five.
cnv-w2a2 12.1 kB (-2.9%), mobilenet_v1 17.6 kB (+13.5%). Nothing committed.
Outstanding: mobilenet throughput (running, hours), B's solver integration
(placement residual, §6), the three size/pytest gaps in §7.
