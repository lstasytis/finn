# Result A — the ConvolutionInputGenerator tree model

The model was 1012 lines of which ~450 were unreachable, and its live path was a
cycle-by-cycle FSM trace flattened into a single run-length leaf. It is now
**270 lines** and a **composite loop nest**, 384/384 configurations modelled,
and the 384-config check runs in **0.66 s** instead of 10.3 s because nothing
walks a cycle any more.

Worked on `feature/analytical-fifo-sizing` @ 43597d82. Two tracked files are
modified — `src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py` and
its test. **Nothing is committed or pushed.** `goldens/base.npz` was never
regenerated. (`finn-rtllib/mvu_tiled/input_gen.sv` also shows modified; that is
Workstream B's, untouched here.)

---

## 1. Headline

| | 43597d82 | shipping |
|---|---|---|
| tree-model lines | 1012 | **270** |
| file lines | 1305 | **574** |
| configurations modelled | 384/384 | **384/384** |
| `[224,224] k3 ch3` | 1 leaf, 17545 runs, **depth 0** | **11 leaves, 12 runs, depth 2** |
| `[113,113] k3 ch32` | 1 leaf, 49296 runs, **depth 0** | **17 leaves, 21 runs, depth 3** |
| `[58,58] k3 ch128` | 1 leaf, 49956 runs, **depth 0** | **11 leaves, 13 runs, depth 2** |
| `swg_tav.py check` | 10.3 s | **0.66 s** |
| cnv-w2a2 FIFO total | 12.46 kB | 12.1 kB (**−2.9%**) |
| mobilenet_v1 FIFO total | 15.5 kB | 17.6 kB (**+13.5%**) |
| pytest `node_tree_modeling` | 13/13 | **12/13** |

Accuracy over all 384 configurations, scored against the FSM oracle:

```
worst frac_under 0.1055   p95 0.0556   median 0.0044
worst frac_over  0.3678   p95 0.0625   median 0.0156
worst frac_period 0.1336  p95 0.0220   median 0.0000
token counts exact on 379/384
error grows between periods on 15/384
GATE (--const 8 --fail-under 0.01 --fail-over 0.10): 309/384
```

`swg_integrate.py` reports **KEEP** on all five bar items. Lint clean under the
pinned black 23.3.0 / isort 5.12.0 / flake8 6.0.0.

---

## 2. The throughput measurement, with its positive control

This is the session's most transferable result, because it is the one that says
what a FIFO number is *for*.

```
                        interval_cycles   throughput   stable_throughput
baseline  (12.46 kB)         115207         1672.4        1736.0
candidate (12.07 kB)         115207         1672.4        1736.0
control   ( 4.14 kB)      →  142043     →   1369.8        1408.0
```

cnv-w2a2, 64 frames, 200 MHz, rtlsim_xsi, ~85 min per run. All three clean:
`TIMEOUT 0`, `UNFINISHED_INS/OUTS 0`, 64/64 frames, steady-state interval valid.

Baseline and candidate are **bit-identical on every field except wall clock**,
and their inputs genuinely differ (8 FIFOs, in both directions). The control —
baseline depths with `StreamingFIFO_rtl_15` cut 2161 → 32 — moves
`interval_cycles` **+23.3%** and throughput **−18.1%** through the same
measurement path.

**Reading.** The harness sees FIFO depth, so the identical baseline/candidate
result is a real finding rather than a blind measurement. What it licenses is
**"cnv-w2a2 is insensitive to FIFO depth in the 12.07–12.46 kB range"** — *not*
"the shrink is safe". It does **not** transfer to mobilenet_v1, which is the
model the concern was actually about; that run is in flight and unfinished.

Reproduce with `claude-tools/swg/swg_throughput.py` and the three frozen depth
sets in `goldens/` (`cnv-w2a2_base`, `_nest`, `_control`).

> **Trap.** `results.txt` is written only when the simulator *finishes*, so a
> `FileNotFoundError` on it means "did not finish", not "backend broken" — I
> called that wrong once. The progress signal is the tick counter in
> `rtlsim_xsi_log.txt`. Two real harness defects were also fixed:
> `step_measure_rtlsim_performance` skips itself silently unless
> `RTLSIM_PERFORMANCE` is in `generate_outputs`, and `enable_build_pdb_debug`
> defaults on, so a failure hangs in pdb instead of returning.

---

## 3. What the model is

`swg_controller` (`finn-rtllib/swg/swg_common.sv`) is a five-deep counter nest.
One output beat leaves per innermost iteration, and the free pointer releases a
*draw* of input slots whenever a level completes — `TAIL_INCR_W` at the end of a
window, `TAIL_INCR_H` at the end of a row, `TAIL_INCR_LAST` at the end of the
frame. So the tree is that nest: a frame of rows, a row of windows, a window of
free-pointer steps, plus three leaves for the things the nest alone does not say
(the fill, the row-boundary stall, the drain).

| function | lines | |
|---|---|---|
| `swg_default_nest` | 121 | frame → rows → windows → steps |
| `swg_parallel_nest` | 41 | the parallel style, paced by its input stream |
| `swg_default_tree` | 36 | dispatch, cache, declines |
| `swg_demand_lead` | 26 | **the one term known to be short — see §6** |
| `swg_params` | 20 | the code generator's dict, as ints |
| `swg_nest_dims` | 12 | the five loop trip counts |
| `get_tree_model` | 8 | delegate |
| `_leaf`/`_comp`/`_clip` | 6 | constructors |

Two laws were derived from the oracle rather than guessed:

- **The demand lead.** Beat `j` reads word `addr[j]`, and `addr` is the nest
  evaluated at `j`, so the frame runs `max_j (addr[j] − j)` longer than its
  beats alone — and that maximum is a *sum*, not a search:
  `Σ (trips−1) × max(0, HEAD_INCR − inner)`. Checked by hand before coding:
  49104 predicted / 49104 measured on `k7×7 ch1024`, 440 / 440 on the mobilenet
  113×113 depthwise layer.
- **Write pacing.** The beats waiting inside the lead-in fire one per address
  step. The borrowed beats come from the **last window of the frame**, which
  balances by construction and touches one window out of `h·w`. On the
  degenerate frame this emits `48 × [(1022,[1,0]),(1,[1,1])]` against a measured
  `48 × [(1,[1,1]),(1023,[1,0])]` — the same pacing, one cycle out of phase.

**Two different waits exist and the frame has both**: a beat waits for the word
it reads (head increments → the demand lead) and a read waits for a slot to be
released (tail increments → the row stall). Using either alone regresses the
model; that was proved twice.

---

## 4. What was tried and ruled out

Eleven reverted attempts. Each cost minutes because the tree was snapshotted
first and the oracle scores 384 configurations in 0.66 s. **The record of what
they rule out is the more useful half of this document.**

| # | attempt | result | what it rules out |
|---|---|---|---|
| 1 | closed forms instead of the FSM | declined | a closed form is exact only for shapes someone checked; the deleted 450 lines had lost this trade once already |
| 2 | reads at window tail, fill at frame tail — "conservative" by the TAV convention | cnv-w2a2 **−10.4%** | the TAV convention's safe direction is *anti*-safe under `CHAINED_TAV`: taking words later delays the node's own clock and shrinks its output FIFO |
| 3 | unbounded depthwise stall `max(0, draw_h − beats)` | mobilenet **+323%** | an over-estimated *period* is never a safe route to bigger buffers — it changes the node's apparent rate and the sizer buffers the whole neighbourhood against it |
| 4 | paced lead + trim a row's last window | tokens 379→200 | moving beats into the lead must keep the *read* budget intact |
| 5 | lead-in folded into the first row's first window | tokens 161/384 | substituting a window drops the reads it carried; `w−2` is wrong at `w == 1` |
| 6 | innermost demand term scaled by `epw` | gate 309→304, growth 15→**37** | no constant factor closes the demand term: the two waits *compound*, because reads are themselves throttled |
| 7–11 | five integrations of B's coupled solver (§6) | see §6 | the residual is read *placement*, not balance |

Attempt 4's idea survived on its third try and is what ships.

Two traps were named up front and avoided throughout: **no constant was ever
fitted to a kB number** (the `lead` sweep is reported with its granularity so
the decision stays visible), and **the pytest tolerance was never raised to
cover a failure** — it sits at the measured worst case, and the one failing
shape is left failing as the correct signal.

---

## 5. The cap-margin finding

Worth recording on its own, because it makes the model-level kB total a poor
objective for tree work — §4 of `TREE_MODELS.md` with a number attached.

Sizing mobilenet with a *more accurate* tree grew the total 72%. Per-edge, **9
of 141 FIFOs moved and one edge was 93.8% of it**: `StreamingFIFO_rtl_127`
(DWC → `ConvolutionInputGenerator_rtl_13`), depth 256 → 10951. Six edges moved
*down*.

I first blamed the pacer guard. That was wrong, and the measurement
(`claude-tools/swg/capprobe.py`, `SWG_CAPPROBE=<dir> swg_model_sizes.py size`)
says so:

| term | known-good | variant |
|---|---|---|
| `t_up` | 273337 | 281502 |
| `pacer_period` | 392302.63 | 392302.63 |
| `t_up < pacer_period` | **TRUE** | **TRUE** |
| `peak` | **7427** | **11246** |
| `allowance / peak` | **1.71** | **1.055** |
| `capped` | true | **false** |
| depth | 256 | **10951** |

What flips is `CHAINED_TAV_CAP_MARGIN`: the cap refuses to trade depth away when
`allowance < cap_margin * peak`. **A 51% change in one peak estimate produces a
43× depth change on one edge**, because a threshold sits between them. 256 is
`CHAINED_TAV_THROTTLED_CAP`; the cap is working as designed.

---

## 6. The remaining gap, and Workstream B's solver

The one term known to be short is `swg_demand_lead`. Its contract is documented
at the call site: parameter dict in, cycle count out, no state, one caller.
Measured on `k3 ifm16 ch512 simd1 dw` it returns 15352 where the frame spends
23532. The oracle attributes every idle cycle of that frame exactly:

```
18 runs of 511  = HEAD_INCR_SIMD − 1  →  9198
 2 runs of 7167 = HEAD_INCR_KW   − 1  → 14334
 6 × 4596 + 4600 + 2                  → 32178   (the row stall — already correct)
                               total  = 55710
```

So the row term is right and the whole residual is the demand side. It is not a
missing factor: reads are throttled by the free pointer, so word `addr[j]` is
not available at cycle `addr[j]`, and the two waits compound.

**B built the coupled recurrence for it** (`claude-tools/swg/swg_coupled.py`,
Kleene-iterated, 384/384 cycle-identical to the FSM) and the composition helpers
(`swg_lead_in.py`). Integrating them gets the models tier's period **exact by
construction** and `grow_all` to **1** (against 15) in every variant tried — the
direction is unambiguously right. It did not land: five integrations left
`gate_all` between 298 and 310 against the 309 bar. Removing the paced/debt
circularity entirely did **not** recover it, which rules out my own diagnosis.

**What that leaves, stated by B and confirmed here:** totals, period and peak
can be made exact, but the **mid-frame curve is not** — lumping the row share at
the head of the frame *is* that approximation, and the residual 9 configurations
are measuring it. Fixing it means placing the `row` share at the **row
boundaries**, not rebalancing the head.

Variants preserved at `scratchpad/solver_variant{,_v2,_v3}.py`. Score any
candidate law in one command, no source edit:

```bash
python claude-tools/swg/swg_integrate.py --law module:function
```

**Packaging, unresolved:** the solver is 413 lines in `claude-tools/` and would
have to move into `src/finn/` to ship, at **0.4 s median / 1.5 s worst per
node** against 0.5 ms now. Per node in a real build that is noise; across the
384-config check it turns 0.66 s into minutes. The shape that survives is
*period-only via the solver, closed forms elsewhere*.

---

## 7. What is **not** done

1. **cnv-w2a2 is 12.1 kB against a 12.46 kB baseline (−2.9%)** — a shrink, which
   was the standing rule. §2 measures it as costing nothing in throughput, with
   a positive control, but only in that range and only on that model.
2. **mobilenet_v1 is 17.6 kB against 15.5 kB (+13.5%)** — above baseline, which
   is the right side, but past the +10% budget. Its throughput run is unfinished,
   so the question the user actually asked is still open.
3. **pytest is 12/13.** `k[7,7] ifm[7,7] ch1024 simd1 dw`, a frame that is one
   window. Its period is right; its writes are emitted densely and then idle
   where the real frame paces them one per 1024 cycles. Left failing deliberately.
4. **The mid-frame curve is approximate** even where totals, period and peak are
   exact (§6).
5. The five failing depthwise layers have periods **1.5–3% short**, which makes
   those nodes look faster than they are; B's solver fixes this and §6 says why
   it is not integrated.

---

## 8. Round 1, which stands

- **491 lines of unreachable fallback deleted** — confirmed by putting a `raise`
  at its head and running the 384-config matrix *and* both model builds with it
  in place. Nothing fired.
- **`swg_default_params` + `swg_parallel_params` (163 lines) deleted**, replaced
  by `swg_params` reading `prepare_codegen_default()/parallel()`. Checked
  key-for-key on 384/384 first. This also stopped the model answering for
  hardware that cannot be built: over 2500 random off-matrix configurations the
  new parameters are identical wherever both answer (1168 configs, 0
  divergences) and decline 56 the old code answered — **all 56 fail
  `generate_hdl`**.
- **`dynamic_mode` no longer declines**, checked against rtlsim at zero
  tolerance, three shapes × `dynamic_mode` ∈ {0,1}.
- The `node_tree_modeling` suite is genuinely live: injecting one extra idle
  cycle turned 13 passed into 13 failed.

---

## 9. Files and reproduction

Shipping: `src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py`,
`tests/fpgadataflow/test_fpgadataflow_convinputgenerator.py` (tolerance).

Workbench, all new, all in `claude-tools/swg/`: `swg_fsm.py` (the FSM, kept as
oracle), `swg_fold.py` (folds a period by the nest — where the laws were read
off), `swg_score.py` (signed scoring + growth-between-periods), `swg_nest.py`,
`swg_lines.py`, `swg_integrate.py`, `swg_throughput.py`, `capprobe.py`.
B's: `swg_coupled.py`, `swg_lead_in.py`, `input_gen_*.py`.

```bash
python claude-tools/swg/swg_lines.py
python claude-tools/swg/swg_integrate.py                     # the five bar items
python claude-tools/swg/swg_score.py --matrix all
python claude-tools/swg/swg_tav.py check -g claude-tools/swg/goldens/base.npz \
       --const 8 --fail-under 0.01 --fail-over 0.10 --allow-missing
python claude-tools/swg/swg_fold.py --matrix models -n 2 -v  # the structure
python claude-tools/swg/swg_model_sizes.py size --model cnv-w2a2 -o /tmp/cnv.json
pytest -m node_tree_modeling tests/fpgadataflow/test_fpgadataflow_convinputgenerator.py
# throughput: ~85 min per depth set; watch the tick counter, not the file
python claude-tools/swg/swg_throughput.py --model cnv-w2a2 --tag base \
       --depths claude-tools/swg/goldens/cnv-w2a2_base.json
```

**Before re-measuring anything:** `envs/rtlsim/src/finn/` is a second complete
FINN checkout inside this working tree (git-excluded) still carrying the old
1012-line model. Check what `import finn` resolves to first.

**Pre-existing bug, untouched:** `select_impl_style()` returns `"parallel"` for
shapes `prepare_codegen_parallel()` then dies on with a `math domain error`.
