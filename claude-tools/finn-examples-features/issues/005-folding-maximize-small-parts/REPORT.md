# Report 005: maximize-mode folding on AUP-ZU3_8GB — fixed
(Written by the orchestrator from the fix agent's hand-back. Scripts in `scripts/` (hard-coded
scratchpad path `S=`), data in `data/`.)

Branch `fix/005` (worktree `build/wt/fix005`, from `test/folding` @ ca829ecc1); both commits for
`feature/dev/set-folding-optimizer` (landed there; DWC branch rebased; cherry-picked onto
`test/{folding,dwc}` and the integration branch):

| commit | file | what |
|---|---|---|
| 5b999c019 | matrixvectoractivation.py | MVAU: no URAM estimate for `internal_embedded` weights (also an upstream/dev bug) |
| f0102ca20 | set_folding.py | maximize mode stays within the budget as built |

## Root causes
A. **Phantom URAM** (main): `MVAU.uram_estimation` reports URAM for `internal_embedded` weights
whenever wmem > 128 while `bram_estimation` counts them too (embedded weights are an HLS ROM:
BRAM/LUTRAM, never URAM; `VVAU` already returns 0). On xczu3eg (0 URAM) the URAM budget is 0.001,
so one phantom URAM costs 1000 and fails `_fits`: only foldings with wmem <= 128 everywhere are
acceptable -> all weights in LUTRAM (cnv-w1a1 baseline: 115 k LUT, 0 BRAM, no probe fits).
Committed foldings showed phantom URAM too (cnv-w1a1 15, lfc-w1a1 80, cnv-w2a2 45).
B. Fallback returned the slowest probe instead of the cheapest (cnv-w1a1: 227808 cycles/115042 LUT
while a 163072-cycle probe used 114860 LUT) -> the 439 fps.
C. Search scored bipolar weights as 1 bit, the build makes them INT2 (`minimize_weight_bit_width`,
upstream): 2x weight memory, ~+35 % MVAU LUT; only the final verdict saw it (KV260 "2.63x" was
104 k LUT vs 82 k budget, never valid).
D. Refinement couldn't start when slow foldings miss the budget (lfc-w1a2: 1287 instead of 1031 cycles).
The verdict is not missing: it is in `build_dataflow_stepwise.log`.

## Fix
1. `MVAU.uram_estimation` 0 for `internal_embedded`. 2. `SetFolding._weight_types_as_built`: score
with the weight datatypes `step_minimize_bit_width` will give (originals restored after).
3. `_verify_as_built`: in throughput search a probe fits only if `_resources_as_built` fits
(~0.1 s/probe). 4. `_sweep_and_refine`: fast bound = slowest infeasible target faster than the best
fitting folding; fallback = cheapest probe.

## Results (seed 1, ratio = committed cycles / maximize cycles, resources as built; ZU3 budget 49700 LUT, 329.6 BRAM18, 0 URAM)
| model/board | committed fits? | baseline | fixed |
|---|---|---|---|
| cnv-w1a1 ZU3 | no (161 k LUT) | 0.036 over | 0.567 fits (49 k / 73) |
| cnv-w1a2 ZU3 | no (101 %) | 0.072 over | 1.194 fits |
| cnv-w2a2 ZU3 | yes | 0.072 over | **1.194** fits |
| lfc-w1a1 ZU3 | yes | 0.662 over | **1.000** fits |
| lfc-w1a2 ZU3 | yes | 15.2 over (277 k LUT) | **1.993** fits |
| tfc-w1a2 ZU3 | no | 0.652 over as built | 0.429 fits |
| kws, tfc-w1a1/w2a2, cybersecurity, gtsrb ZU3 | | | unchanged |
| cnv-w1a2 KV260 | yes | 2.63 **over** | **1.79** fits |
| cnv-w2a2 U55C | yes | 20.97 fits | same cycles, URAM 23 -> 0 |
Where the committed folding doesn't fit, FINN's LUT model (XNOR/binary HLS MVAU several times
pessimistic) is the reason, not the search. Search ~150-160 s on cnv.

## Gate (22 checkpoints x 5 modes, seed 0, frozen src copies)
108 runs: 76 identical, 32 changed (all bnn-pynq or gtsrb; naive identical everywhere; kws,
cybersecurity, tfc-w1a1/w2a2, vgg10, mobilenet, resnet18/50, yolov8, espcn, bert identical).
Changed target/fifo foldings: ZU3 cnv-w1a2/w2a2 165-237 k LUT over -> 39.9 k LUT/197 BRAM fits at
the same 33020 cycles; lfc-w1a1 392 cycles at 200 k LUT -> 519 (target) at 41 k; lfc-w1a2 784 at
278 k -> 2055 (target) at 23 k; cnv-w1a1 166 k -> 84 k LUT (still over); KV260/U55C weights move to
BRAM. tfc-w1a2, gtsrb: same folding, reported numbers changed. Padded: 6 runs fail before and after
(InsertDWC, issue 004).

## Verification
test_set_folding 32 passed/2 skipped/1 xfail (unchanged), lint clean.
`test_bnnpynq_folding[AUP-ZU3_8GB-cnv-w1a1-maximize]` PASSED (6941 est / 6917 rtlsim fps, 0.997,
"fits"). `[...cnv-w2a2-maximize]` FAILED only the bar: 1.17x committed, verified, fits, but rtlsim/est
0.9798 (28226 vs 27655 cycles, very balanced folding) -> estimate / FIFO sizing area, issue 006.

## Open points
* Upstream bug: `minimize_weight_bit_width` widens BIPOLAR weights to INT2 in every build (2x memory,
  +35 % LUT on cnv-w1a2/lfc-w1a2/tfc-w1a2); own issue (changes hardware of every bipolar model).
* FINN's LUT model for XNOR/binary HLS MVAUs is several times pessimistic.
* HANDOFF item 3: the cnv-w1a1 ZCU104 LUTRAM example was driven by phantom URAM; without it BRAM
  wins by the same objective. Peak utilization vs weighted sum still open.
* Not re-run: other boards, ZCU104.
