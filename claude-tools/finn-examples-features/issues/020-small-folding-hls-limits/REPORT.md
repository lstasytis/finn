# Issue 020 report: foldings beyond what Vitis HLS can compile

Scripts are in `scripts/`:
- `survey.py` and its output `survey_output.txt` list every embedded HLS MVAU initializer under `fx/*/code_gen_ipgen_MVAU_hls_*` with its csynth outcome.
- `check_fold.py` runs an estimate-only optimizer build of a feature test, then checks every node against the HLS limits.

## What Vitis HLS 2024.2 compiles (measured)

The `FixedPointWeights<SIMD,W,PE,WMEM>` initializer has PE x WMEM = MW x MH / SIMD entries. csynth time follows the Compile/Link instruction count.

| Entries | Shape | Outcome |
|---|---|---|
| <= 32768 | all | ok, <= 0.6M instructions, <= ~30 min |
| 65536 | PE=1, WMEM=65536 at 4 or 8 bit; PE=8/32/64 | ok, 1.0–1.25M instructions, 20–85 min |
| 65536 | 18-bit entries (PE=1/WMEM=65536 and PE=2/WMEM=32768) | hung: 3.67M instructions, still in clang after 22 h |
| 87723 | PE=3 | hung (4.65M) |
| 131072 | PE=16, 16 bit (lfc-w1a2 small) | ok but 1–4 h |
| 131072 | PE=4, 2 bit | hung (7.3M) |
| 131072 | PE=1, WMEM=131072 | error HLS 200-1715 |
| 262144 | PE=8, 2 bit (cnv MVAU_hls_7) | hung (14.7M) |

So the safe cap is 32768 entries. Two other limits apply independently:
- Any ap_int must be at most 8191 bits. `HLSBackend.get_ap_int_max_w` asserts this, but `MVAU_hls` overrides that method without the assert.
- `MVAU_hls` ipgen asserts SIMD >= MW/1024.

## A + B: small foldings

`make_small_foldings.py` keeps a PE/SIMD combination only if all of these hold:
- Embedded entries are at most max(32768, committed).
- WMEM is at most max(32768, committed WMEM). This is the old rule, kept so unaffected models don't change.
- MVAU_hls SIMD >= MW/1024.

Node attributes are restored after combinations are tried; without that, the SWG side was scaled from a mutated state.

`check_small_foldings.py` reports the same three conditions plus any width over 8191 bits.

Results after regenerating every model:
- **cnv-w1a2/w2a2:** MVAU_hls_5 goes from PE2/SIMD9 to PE1/SIMD18, MVAU_hls_6 from PE4/SIMD1 to PE1/SIMD4, MVAU_hls_7 from PE8/SIMD1 to PE1/SIMD8 (all equal to the committed folding). Max cycles go from 33020 to 129607 (3.9x).
- **resnet50:** MVAU_hls_47/50 go from PE32/SIMD1 to PE16/SIMD2.
- **lfc-w1a1/w1a2:** would change, but are not committed because their small jobs passed; the checker flags them.
- **All other models:** unchanged and OK.

Commits:
- integ `02949f04f`, cherry-picked to test2/fifo `a5cf3a07f`, test2/folding `c9957fcf2`, test2/dwc `9d4117158`.
- claude-tools `d6aff485d` for the tools and issue files.

The old cnv small estimate builds (all three worktrees) and the resnet50 one (fifo2 only) were moved to `fx/*/superseded_020/`. The resnet50 small estimates in folding2 and dwc2 (`fx/folding2/est_resnet50_small_V80_lclp_otw`, `fx/dwc2/est_resnet50_small_V80_3szgb2zd`) were not moved because running jobs still use them; move them before those jobs are rerun.

## C: SetFolding (`69d005339`)

`Optimizer.breaks_hls_limits` flags an HLS node if:
- any stream width or its `get_ap_int_max_w()` exceeds 8191 bits,
- an MVAU_hls has SIMD < MW/1024, or
- an embedded MVAU/VVAU_hls has more than 32768 entries.

How it is enforced:
- Values that break a limit even with the other parameter at 1 are no longer offered (generic metas, SWG→MVAU, SWG→VVAU). If no value fits, all are kept.
- The coupled PE·SIMD·wbits case costs `HLS_LIMIT_PENALTY` (1e9) in the cost model and is skipped by `cheapest_guess`.
- `_refine` repair, polish and tighten never step a node beyond the limits.

Tests:
- New: `test_set_folding_hls_mvau_weight_stream_width[1|None]` gives PE·SIMD=512 (2048 without the fix); `test_set_folding_hls_mvau_simd_limits` (2 cases) gives SIMD>=2 (1 without the fix). All 4 fail without the fix.
- Changed: the U200 bound of `test_set_folding` is now the fastest legal folding (PE·SIMD=1024; 23 cycles with the HLS overhead).

Suite results:
- Folding branch: 50 passed, 5 skipped (tree lacks FIFO/DWC).
- test2/folding tree: 53 passed.
- dwc2 and integ: 55 passed each.

Landing: DWC branch merge `c9fd722dc`, then integ merge `b2f794748` (conflict with integ's own `16+7` bound, resolved to the computed bound and recorded by rerere), then cherry-picks to test2/folding `b8ef49ee5` and test2/dwc `ca240da3a`.

Model checks:
- **cnv-w1a2 small fold-target:** 842.8 fps against a 771.6 target, MVAU_hls_5/6/7 at SIMD 18/4/8, no violations.
- **resnet50 V80 maximize:** MVAU_hls_0 at PE16/SIMD49, 1660.7 fps (was 3321 with a 12544-bit stream that cannot be built), no violations. The folding step took 4142 s against 2233 s before.

## Campaign

- Killed the 16 stuck jobs as explicit PID trees, with the runner stopped so their full siblings would not start.
- Ran a temporary runner with an `--only` exclusion while the fixes landed.
- Requeued resnet50 fifo and fifo-rtlsim small plus their fulls, and `v2__fold-max__resnet50__V80`. The cnv small jobs had no XML, so they were pending again by themselves.
- The runner is back on `runner.args`.

Scratch worktree: `build/wt/fix020` (branch fix/020).
