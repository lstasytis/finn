# 026: kws `test_kws_folding[maximize]` fails on the standalone folding PR (2026-10-28)

`feature/dev/set-folding-optimizer` at 22171408a (also before the benchmark tidy, 54fa8d2b5, and after a trial
merge of upstream/dev 98306537d with `build/deps_devsync`): `target` passes (rtlsim 1.00x estimate, verified),
`maximize` fails in `step_set_fifo_depths` with dev's `characterize` strategy:

    AssertionError: Total cycle count from rtl simulation is higher than specified period ... 510
    [Thresholding_rtl_0 period 500 exp 490]

* The maximized folding keeps `Thresholding_rtl_0` at PE=1 (490 cycles, the bottleneck; MVAU_hls_0 448).
* `characterize` uses `period = max_cycles + 10` = 500 and asserts every node's isolated rtlsim finishes within
  it; Thresholding_rtl's 20-cycle pipeline latency gives 510.
* Earlier passing maximize runs (fx/folding2, 204k fps, same 490-cycle bottleneck) ran on `test2/folding`, which
  carries the FIFO PR, so they never exercised dev's `characterize` code.
* Not caused by the folding optimizer or by the test refactor.

Options: folding tests size FIFOs with `heuristic_analytical` once the FIFO PR is in dev; or xfail `maximize`
until then; or a dev fix of the `characterize` period margin (separate PR).
