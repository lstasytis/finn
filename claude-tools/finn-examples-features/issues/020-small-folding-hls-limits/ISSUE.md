# Issue 020: foldings beyond what Vitis HLS can compile (small foldings, SetFolding)

Read first: `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md`, the docstrings of
`tools/make_small_foldings.py`, `tools/check_small_foldings.py`, `tools/run_campaign.py`, `tools/requeue.sh`,
`tools/rebuild_branches.sh`.

Three bugs block the v2 feature-test campaign:

* **A: Vitis HLS hangs on the cnv-w1a2/w2a2 `_small.json` foldings.** 16 jobs `v2__*__bnn-pynq__*cnv-w1a2__small` /
  `*cnv-w2a2__small` sat 22 h in Vitis HLS clang (e.g. `build/finn_build/fx/fifo2/code_gen_ipgen_MVAU_hls_7_4ze69dut`:
  MW=MH=512, embedded weights, 14.6M instructions). The committed folding is PE=1, SIMD=8, the small one PE=8,
  SIMD=1: WMEM stays 32768, but the `FixedPointWeights` initializer grows 8x (PE x WMEM entries). Fix
  `make_small_foldings.py`: for HLS MVAUs with embedded weights PE x WMEM must not exceed the committed folding's (or
  a safe cap); prefer reducing PE, keep SIMD.
* **B: resnet50 small folding violates `SIMD >= MW / 1024`** for HLS MVAUs (MVAU_hls_47: SIMD=1, MW=2048, in
  `fx/fifo2/fifo-heuristic_analytical-small_resnet50_V80_1fhkjy50/build_dataflow.log`). Add the constraint to
  `make_small_foldings.py` and `check_small_foldings.py`, regenerate and check every `_small.json` (only cnv-w1a2/w2a2
  and resnet50 should change; report others), commit where the small JSONs were committed, propagate to `test2/*`.
* **C: SetFolding maximize picks HLS MVAU foldings wider than ap_int's 8191 bits** (`fx/folding2/fold-max_resnet50_V80_oacddp67`,
  MVAU_hls_0 `ap_uint<32*49*8>`). Fix `set_folding.py` on `feature/dev/set-folding-optimizer` (own worktree): never
  offer HLS MVAU/VVAU (or similar packed-width HLS ops) foldings beyond the limit; unit test next to the set_folding
  tests; land on the integration and test2 folding/dwc branches as earlier set_folding fixes were.

Campaign: kill the 16 stuck jobs by explicit PID trees; after A+B rerun the cnv small jobs (full siblings follow) and
the resnet50 `fifo`/`fifo-rtlsim` small jobs and their blocked fulls, after C `v2__fold-max__resnet50__V80`; restart
the runner with `tools/requeue.sh` without disturbing running jobs. Rules: <= 16 threads of own work, nothing new
below 200 GB MemAvailable, never `largefifo_rtlsim`, never push, no FINN RTL/HLS source changes.
