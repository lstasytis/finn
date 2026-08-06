# Handoff A — shrink the ConvolutionInputGenerator tree model

**You have 10 hours. Work autonomously.** Do not stop to ask for confirmation;
the acceptance criteria below are the decision procedure. Read
[`TREE_MODELS.md`](TREE_MODELS.md) first — it is the background for this task
and it records mistakes that have already been paid for once.

**Before you start: check whether this task is still live.** Handoff B is
building a tree model for `finn-rtllib/mvu_tiled/input_gen.sv`, the intended
replacement for the CIG. If that model lands in under ~200 lines, the CIG will
be replaced wholesale and characterising it in a shorter form is wasted work.
Look for `HANDOFF_INPUT_GEN_RESULT.md` in this folder, or ask. If it is not
there yet, proceed — the two tasks are independent.

## The goal

`src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py` is 1305 lines,
1012 of them the tree model, against 144 for the MVAU and 108 for OuterShuffle.
Bring the SWG tree model down to something comparable — the target is **under
250 lines total**, and under 150 would be a very good result — without the FIFO
depths of mobilenet_v1 or resnet50 moving outside budget.

Accuracy requirement, in the user's words: it does not have to be
cycle-accurate. A constant error of a few cycles is fine, single digits of
constant error on large feature maps is fine; what must hold is that the
**fraction** stays in single-digit percent, and that the total FIFO size on the
finn-examples models does not grow beyond 5–10%.

**A FIFO total that comes out *lower* than the baseline is a failure, not a
win.** The baseline is what the hardware was validated at; less is undersizing.
This is the single most important asymmetry in the task, and it is why
`swg_tav.py check` reports `undersize` and `oversize` separately and defaults
`--fail-under` to 0.

## What has already been measured — start from here, do not re-derive it

Across a 384-configuration matrix (the convinputgenerator pytest
parametrisation, the mobilenet_v1 and resnet50 sliding windows, and a stress set
covering dilation, stride > kernel, 1xN feature maps, SIMD < IFMCh depthwise and
1x1 windows), **every configuration is served by tier 1**, `swg_default_exact`
— the function that executes the SWG FSM and run-length-encodes the result.

The ~450 lines of hand-derived closed forms in the body of `get_tree_model`
(`k1_pass`, `dw_pw0_k2s2`, `pw1_k2_s3`, `pw1_generic`, the `k=2x2 pw=0 s=2x2`
leaf, and the long "baseline branch") were **never reached**. They also cannot
be reached from a normal build: there is no HLS ConvolutionInputGenerator
registered in this tree, so `preferred_impl_style="hls"` is forced back to RTL,
and `swg_default_tree` covers both RTL impl styles. The only probe that fell
through to them was `dynamic_mode=1`.

That makes the shape of the work clear, and it is mostly deletion.

## Suggested plan

1. **Freeze the reference.** `goldens/base.npz` is already committed here,
   produced from 43597d82. Confirm it still checks clean before you touch
   anything:

       python claude-tools/swg/swg_tav.py check -g claude-tools/swg/goldens/base.npz

   Expect `0/384 configs outside budget` in about 6 seconds. If it does not,
   your working tree already differs from the reference — find out why before
   proceeding.

2. **Get the model-level baseline.** The mobilenet checkpoint may already be
   built (`$FINN_BUILD_DIR/swg_fifo_mobilenet_v1`); if not, `build` takes a
   while and only has to happen once.

       python claude-tools/swg/swg_model_sizes.py build --model mobilenet_v1
       python claude-tools/swg/swg_model_sizes.py size  --model mobilenet_v1 \
              -o claude-tools/swg/goldens/mobilenet_base.json

   Both baselines are already captured: mobilenet_v1 (ZCU104) = 141 FIFOs /
   **15.5 kB** (74 s per sizing re-run), cnv-w2a2 = 34 FIFOs / **12.5 kB**
   (3 s). Use cnv-w2a2 as the quick second opinion — it has 8 CIG nodes and a
   board reference (12.967 / 13.031 kB) that its number reproduces.

   **resnet50 is not available as a guard.** Its build dies in
   `step_create_dataflow_partition` with "cycle-free graph violated: partition
   depends on itself" — the known pre-existing partition blocker, unrelated to
   tree models and out of scope here. Do not spend time on it; `cnv-w2a2` is
   the substitute.

   If `size` fails because a node without a tree model needs just-in-time
   synthesis, re-run `build` with `--with-ipgen`.

3. **Confirm the dead-code finding yourself** before deleting anything — it is
   the load-bearing claim of this handoff. Instrument the fallback branches
   (a `raise` or a counter at the head of each) and run the full matrix plus
   both model builds. If nothing fires, the branches are dead.

4. **Delete the unreachable closed forms.** Replace the whole fallback body
   with *one* conservative generic model, or with `return None` (which falls
   back to rtlsim — see §3 of `TREE_MODELS.md`; a decline is a supported and
   honest answer). Decide by what `dynamic_mode` needs:

   - `dynamic_mode` only means the loop bounds arrive over AXI-lite at runtime;
     at characterisation time the node attributes still hold the compile-time
     shape. Running the FSM on them is arguably correct. **Test this**: drop the
     `dynamic_mode` guard in `swg_default_tree`, and check a dynamic-mode config
     against rtlsim with the existing pytest. If it matches, one more branch and
     one more guard disappear.

5. **Then shrink tier 1 itself** — this is where the remaining ~500 lines are.
   Ideas, roughly in order of expected payoff:

   - `swg_default_params` / `swg_parallel_params` (163 lines) duplicate
     computation that `prepare_codegen_default()` / `prepare_codegen_parallel()`
     in `rtl/convolutioninputgenerator_rtl.py` already do to generate the RTL.
     Call those instead of maintaining a second copy. This is probably the
     single biggest win available and it removes a drift risk as well as lines.
   - The two `*_schedule` functions (261 lines) are cycle-by-cycle Python
     transcriptions of the FSM. They are also the only slow part of the model:
     a 512x512 feature map materialises an 18.7M-cycle array. Either
     (a) find the closed form — the FSM is a counter nest, so the schedule is
     periodic in a way the loop structure exposes, and a *composite*
     `Characteristic_Node` mirrors that nest directly; or
     (b) keep the execution but emit runs instead of cycles.
   - `swg_default_tree`'s windowing (phase alignment to `restarts[2]`) is
     subtle and correct — see the comment about rtlsim keeping two periods out
     of the middle. Do not "simplify" it away; the phase is what makes the
     vector line up with the reference.

6. **Iterate against the fast check**, which is ~6 s per round:

       python claude-tools/swg/swg_tav.py check -g claude-tools/swg/goldens/base.npz \
              --fail-under 0 --fail-over 0.10

   `check` also reports configurations that *lost* their tree model (they now
   fall back to rtlsim). That is not automatically wrong — but each one is a
   deliberate decision, so it fails the run unless you pass `--allow-missing`.

7. **Guard at the model level** before declaring done:

       python claude-tools/swg/swg_model_sizes.py size --model mobilenet_v1 -o cand.json
       python claude-tools/swg/swg_model_sizes.py compare \
              -a claude-tools/swg/goldens/mobilenet_base.json -b cand.json --tol 0.0 --grow 0.10

8. **Run the real pytests** at the end — they are the ones that compare against
   rtlsim rather than against the frozen tree:

       pytest -m "node_tree_modeling" tests/fpgadataflow/test_fpgadataflow_convinputgenerator.py

   Note the existing test asserts an **exact** match wherever
   `swg_default_tree` applies (`max_allowed_volume_frac = 0`). If your change
   narrows where tier 1 applies, that assertion silently relaxes for those
   configurations — check you have not lost coverage by that route.

## Acceptance criteria

- `swg_tav.py check` on the full matrix: `undersize` fraction **0** everywhere;
  `oversize` fraction under 0.10; period fraction under 0.05.
- `swg_model_sizes.py compare` on mobilenet_v1 **and** cnv-w2a2: total kB
  between **-0% and +10%** of baseline. A drop of any size is a failure — chase
  it down rather than accepting it. (resnet50 is blocked; see step 2.)
- Line count of the SWG tree model (the six functions in §5 of
  `TREE_MODELS.md`) under 250, measured with the AST snippet in that file.
- `pytest -m node_tree_modeling` on the convinputgenerator tests passes.
- FINN lint clean: `black --line-length=100`, `isort`, `flake8
  --max-line-length=100 --extend-ignore=E203` (see `claude-tools/LINTING.md`).

## Traps

- **Never regenerate `goldens/base.npz` from a modified tree.** A golden taken
  from a broken model makes every later check pass. `git stash` first.
- **Do not tune against the model-level number.** mobilenetv1's sizer is
  chaotically conditioned on three edges; a tree tuned to compensate is wrong
  per-node and accidentally right on one graph. Make the tree correct and use
  the kB number only as a guard. (`TREE_MODELS.md` §4.)
- **Do not add a post-hoc wind-up shift.** It double-counts against a tree that
  models its own wind-up and the sizer accumulates the deficit every frame.
- **Do not restart the "evolved CIG tree" line of work.** It was evaluated and
  rejected in 2026-07; its one idea moved mean error 1.07% -> 1.04% and made
  five of eight nodes worse.
- **A stale source tree has burned two agents before.** Confirm you are on
  `feature/analytical-fifo-sizing` at or after 43597d82 and that
  `import finn` resolves to *this* checkout, not an installed copy.

## Report back

Write `HANDOFF_CIG_RESULT.md` in this folder: final line count, the check and
compare outputs, what you deleted and on what evidence, what you kept and why,
and anything you found that contradicts this handoff.
