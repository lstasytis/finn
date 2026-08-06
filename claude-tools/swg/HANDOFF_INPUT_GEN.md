# Handoff B — a tree model for `input_gen.sv`, the CIG's replacement

**You have 10 hours. Work autonomously.** Do not stop to ask for confirmation;
the acceptance criteria below are the decision procedure. Read
[`TREE_MODELS.md`](TREE_MODELS.md) first.

## The goal and the stakes

`finn-rtllib/mvu_tiled/input_gen.sv` is a generic loop-nest input generator and
the intended replacement for the ConvolutionInputGenerator. Build a tree model
for it — and **report its line count as a headline number**, because it decides
someone else's work:

> If the input_gen tree model comes in **under ~200 lines** and covers the
> sliding-window configurations to the same degree the current CIG model does,
> the CIG will be replaced wholesale and Handoff A (shrinking the 1012-line CIG
> tree model) is cancelled. Write `HANDOFF_INPUT_GEN_RESULT.md` in this folder
> with that verdict as soon as you know it, so the other agent can stop.

"To the same degree" means: the configuration matrix in `swg_configs.py` —
the convinputgenerator pytest parametrisation, the mobilenet_v1 and resnet50
sliding windows, and the stress set (dilation, stride > kernel, 1xN feature
maps, SIMD < IFMCh depthwise, 1x1 windows). Configurations the model declines
must be declined *explicitly* (`return None`), not modelled wrongly.

Accuracy target is the same as for the CIG and it is loose: not cycle-accurate,
a few cycles of constant error is fine, single-digit constant error on large
feature maps is fine, provided the **fraction** stays in single-digit percent.

## What the module does

With `ivld` tied high and `ordy` tied high — the stimulus FIFO characterisation
uses — the module reduces to two rules:

- **Output:** one beat per cycle whenever `has_data` (`Rp - WpZ < 0`), i.e.
  whenever the read pointer has fallen behind the registered write pointer.
  The read pointer moves by `TERMINAL_RP_INC[i]` where `i` is the outermost
  loop level that terminated this beat, so it jumps *backwards* at the end of a
  kernel row and forwards at the end of an output pixel. That is the whole
  sliding-window behaviour.
- **Input:** one beat per cycle whenever `irdy` (`Cap < 0`), i.e. whenever the
  circular buffer has a free slot. Slots are released by `TERMINAL_FP_INC[i]`,
  and only at levels whose `R_FLAG` is set — the levels that will never read
  that address again.

`BUF_SIZE` is a power of two computed at elaboration from `MAX_OCCUPANCY`, so
it is usually *larger* than the working set, and that slack is what decouples
the two sides.

Everything above is already transliterated for you.

## What you have been given

- **`input_gen_ref.py`** — a cycle-accurate Python reference for the module:
  the elaboration functions (`INIT_W`, `INIT_R_FLAG`, `INIT_RP_INC`,
  `INIT_FP_INC`, `INIT_MAX_OCCUPANCY`, `BUF_SIZE`), the pointer/counter update,
  and `tav()`, which returns one settled period as cumulative read/write
  vectors. **This is your ground truth for the fast loop** — no verilator, no
  vivado, milliseconds per configuration. It runs:

      python claude-tools/swg/input_gen_ref.py
      3x3 s1 8x8 c4 simd2    dims=[6,6,3,3,2] coefs=[16,2,16,2,1] fm=128 buf=64
                             -> period 648, 128 reads 648 writes

  **Validate the reference before you trust it.** It is a transliteration
  written from the source, not a verified model. Two things in it are known-soft:

  1. `loop_nest_conv()` — the conv → (DIMS, COEFS, FM_SIZE) mapping, especially
     the depthwise ordering. Check it against the real instantiations in
     `finn-rtllib/mvu_tiled/mvu_tiled_axi.sv` (lines ~116 and ~244) and against
     the MVAU's own account of the nest in `matrixvectoractivation.py` around
     line 1346 (`DIMS = {NF, SF, TH}`).
  2. The 1x1 case above reports a 192-cycle period for 128 beats — a
     throughput loss from a 4-entry buffer. That may be real (the pipeline
     latency dominates a tiny buffer) or it may be an off-by-one in the
     reference's output-stage timing. Settle it against the RTL.

  The authoritative check is the module's own testbench,
  `finn-rtllib/mvu_tiled/tb/mvu_tiled_axi_tb.sv`. If you can get verilator to
  run it, do — but note rtlsim is frequently unusable locally (verilator 5.x vs
  the pyverilator gap), so treat a working RTL sim as a bonus, not a
  prerequisite. Reading the SV carefully is the fallback and it is sufficient.

- **`swg_configs.py` / `swg_tav.py`** — the configuration matrix and the fast
  harness for the *current* CIG. Reuse the matrix; `swg_tav.py`'s
  `compare()` gives you the metric that matters (see below).

- **`swg_model_sizes.py`** — the model-level FIFO guard, if and when an
  input_gen-based node is wired into a graph. Baselines captured:
  mobilenet_v1 (ZCU104) 15.5 kB (74 s per re-run), cnv-w2a2 12.5 kB (3 s).
  **resnet50 is unavailable** — its build dies in
  `step_create_dataflow_partition` ("cycle-free graph violated: partition
  depends on itself"), a known pre-existing blocker unrelated to tree models.

## The metric

The FIFO sizer takes a depth from the gap between cumulative producer writes
and cumulative consumer reads. So for this operator:

- reading **earlier** than the reference shrinks the FIFO in front of it;
- writing **later** than the reference shrinks the FIFO behind it.

Both are undersizing — the dangerous direction. The opposite two cost depth
only. `swg_tav.compare()` reports them separately as `undersize` and
`oversize`; hold `undersize` at zero and `oversize` under 10%.

## Suggested plan

1. **Validate `input_gen_ref.py`** against the SV and, if possible, the
   testbench. Fix what is wrong; it is the foundation of everything after.
2. **Sweep the reference** over the whole configuration matrix and look at the
   schedules. You are looking for the three-phase structure the OuterShuffle
   model already names: a demand-limited stretch (input solid, output bursty),
   a free-limited stretch (output solid, input admitted as slots are released),
   and a middle where both run at one word per cycle.
3. **Copy OuterShuffle's model, do not invent one.**
   `src/finn/custom_op/fpgadataflow/outer_shuffle.py` models the *HLS* version
   of this same loop-nest input generator in 108 lines, with helpers
   `loop_nest()`, `buffer_depth()`, `free_lead()`, `demand_phase()`,
   `free_pointer_steps()` and `beats()`. The RTL module has the same W /
   R_FLAG / TERMINAL_RP_INC / TERMINAL_FP_INC structure, so those helpers should
   port nearly unchanged — that is the single strongest reason to expect this to
   land under 200 lines. Note it *declines* when the buffer does not hold a
   frame; keep that discipline.
4. **Score against the reference** on every configuration, report
   undersize/oversize/period fractions, and iterate. Structure the model as a
   composite `Characteristic_Node` mirroring the loop nest rather than a flat
   leaf — the nest is exactly what the tree form is for, and it makes the model
   read like the module.
5. **Decide and report the line count early.** As soon as you have a working
   model over the mobilenet/resnet50 configurations, count the lines and write
   `HANDOFF_INPUT_GEN_RESULT.md` — even if you intend to keep polishing.
   Handoff A is waiting on that number.
6. **Wire it up if it lands.** If a FINN custom op that instantiates
   `input_gen` exists or can be reached (the tiled MVAU already instantiates
   it — see `rtl/matrixvectoractivation_rtl.py`, which lists `input_gen.sv`
   among its sources), give it a `get_tree_model()` and run the model-level
   guard. If no such node exists yet as a standalone sliding-window operator,
   say so plainly in the result file and deliver the model plus its reference
   harness as a standalone module in this folder.

## Acceptance criteria

- A tree model for `input_gen` scoring, against `input_gen_ref.py` over the
  full configuration matrix: `undersize` fraction **0**, `oversize` fraction
  under 0.10, period fraction under 0.05 — or an explicit `None` for the
  configurations it declines, with the declines listed.
- A reusable test that runs the whole matrix in seconds, in the style of
  `swg_tav.py check` (same exit-code-as-verdict contract).
- The headline line count, and the verdict on whether the CIG work should stop.
- FINN lint clean: `black --line-length=100`, `isort`, `flake8
  --max-line-length=100 --extend-ignore=E203`.

## Traps

- **`BUF_SIZE` is rounded up to a power of two**, so the buffer is usually
  bigger than `MAX_OCCUPANCY`. A model that assumes the working set is the
  buffer will predict stalls that do not happen.
- **The output stage is registered** (`OVld`/`OBuf`), so a beat leaves one
  cycle after `advance`, and `advance` itself depends on `ordy`. Wind-up is one
  or two cycles and the tree must express it *itself* — do not add a post-hoc
  shift afterwards. (`TREE_MODELS.md` §4: a shift on top of a tree that already
  models its wind-up double-counts and the sizer accumulates the deficit every
  frame.)
- **The period is what usually goes wrong**, not the per-cycle values. Check
  the frame duration first on every configuration.
- **`R_FLAG` clears from the outside in.** Once a level fails the
  `COEFS[i-1]*DIMS[i-1] <= W[i-1]` test, every inner level's flag is false too,
  and no slots are released at those levels. A model that releases slots at a
  level with `R_FLAG == 0` will over-admit input.
- **A smaller FIFO total is a failure, not a win** — the baseline is what
  hardware was validated at.
- **Confirm your source tree.** Two agents have previously worked against a
  stale checkout without noticing. Check you are on
  `feature/analytical-fifo-sizing` at or after 43597d82 and that `import finn`
  resolves to this checkout.

## Report back

`HANDOFF_INPUT_GEN_RESULT.md` in this folder: the line count and the
stop-or-continue verdict for Handoff A first, then coverage (which
configurations are modelled, which declined and why), the scores, what you had
to fix in `input_gen_ref.py`, and whether the module was validated against real
RTL simulation or only by reading.
