# Report 024 Part A: yolov8_4w4a small hang, cnv-w2a2 small at 0.98
## 1. yolov8_4w4a small (U55C, heuristic_analytical): deadlock at a PAN concat, fixed
* Shortcut DuplicateStreams_hls_11 -> StreamingConcat_hls_12 (SPPF output, 6x10, 15360 words a frame) was sized 9086 by the timeline pass.
  The join floor covered only ElementwiseAdd and took FMPadding/SWG/Upsample as one-to-one, so it added nothing.
* Unbounded-FIFO rtlsim: frame done at 34.9 M cycles; that shortcut peaks at 15358. It is the only join FIFO sized below its unbounded peak
  (the other one, head-2 MVAU_rtl_52 -> StreamingConcat_hls_13 at 96 vs 510, has deep branch buffers behind it).
* Real-FIFO rtlsim with occupancy monitors: the shortcut is full from 22.9 M cycles on. StreamingConcat_hls_12's other input
  (two upsamplings, FPN/PAN C2f, two stride-2 windows) never receives a word, and no FIFO peak changes from 26.8 M to 59 M cycles.
  Campaign run: no output word in 383 M cycles.
* Fix 2cd2a077d: the join floor also covers StreamingConcat (join token = gcd of the input frame sizes);
  FMPadding, ConvolutionInputGenerator and UpsampleNearestNeighbour get get_tav_input_need()
  (since the refactor 30960c3d1: `tav_input_need()` in derive_characteristic.py).
  Shortcut 9086 -> 13842 (deadlock backlog 8704, paced 11073, +25 %).
* Feature test (test2/fifo + fix/024a): PASSED, golden match; first output word at 23.7 M cycles;
  interval 11796480 = estimate (1.000x); 5450 kbit (681.3 kB; campaign build 659.0 kB); sizing 547 s.
  Log: fx/fix024a/logs/yolo4_small_ana.log (worktree build/wt/fix024a-int2).
* Depth changes (offline re-sizing): resnet18 +0.2 % / +1.6 % (full / small), resnet50 +5.6 % / +5.9 %,
  yolov8 +1.5 % / +1.7 % / +3.3 % (4w4a full, 8w8a full, 4w4a small); all other models unchanged.
  Increases are on join inputs only; resnet rtlsim not rerun (stopped at your request).
* Side finding: fold-target yolov8 small (optimizer folding) ran to completion but failed the golden check; not a FIFO problem.
## 2. cnv-w2a2 small at 0.9797x: FIFO depth, not the estimate; fixed (271ca1e5e)
* Unbounded FIFOs, or every FIFO x4: 129606 cycles = estimate. As sized: 132298.
* Cause: Pool_hls_0 -> ConvolutionInputGenerator_rtl_3 was throttled-capped at 256 (peak 307) between stages at 0.88 and 0.91 of the period,
  while MVAU_hls_2 -> ConvolutionInputGenerator_rtl_4, two nodes further down, spent the chain's whole slack (3968 -> 1715).
  The capped edge at 280 or 307 reaches the estimate; the relaxed edge alone needs at least 2400.
* Rule: throttled cap only where t_above + t_below <= CAP_MARGIN * period (the same pair test as the decoupled chain).
* Depth changes: that edge in cnv-w2a2 and cnv-w1a2 small (256 -> 307), plus two yolov8 small concat -> MVAU edges; nothing else.
* Feature tests: cnv-w2a2 small AUP and U55C (analytical), AUP (rtlsim), cnv-w1a2 small AUP (analytical): all PASSED at 1.000 (23.5-23.8 kbit).
## Commits (fix/024a)
271ca1e5e cap rule + test + docs; 2cd2a077d concat join floor + need hooks + test;
558bd1e37 docs for both rules; 50eb2946a fifosim_save_waveform docstring (the last two rebased onto 447b96e25;
cherry-picked onto feature/dev/analytical-fifo-sizing as ff7999073 / ea5e812d7, hook name fixed in 4ce1bd188).
## PR rule sentences
* The throttled cap applies only where the slowest stages above and below the edge together take at most CAP_MARGIN periods.
* The join window floor also covers StreamingConcat, and FMPadding, sliding-window generators and upsampling report the input each output word needs,
  so shortcuts hold the window lag of the long branch.
## Files
Scripts: issues/024-fifo-pr-blockers/scripts_a (perf.py with SCALE/MIN, live maxcount/count driver, join_need.py, what-if and compare tools).
Data: fx/fix024a (tav, tav_small, cmp_join, res, live_mon, live_cnt, int, int2).
Worktrees: fix024a (scratch, uncommitted count-port diagnostic patch), fix024a-fifo (fix/024a), fix024a-int, fix024a-int2.
