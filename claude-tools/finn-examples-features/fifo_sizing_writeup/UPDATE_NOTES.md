# Write-up update: 39dac9a18 -> 4ce1bd188 (feature/dev/analytical-fifo-sizing)

Rebuilt `fifo_sizing_analytical.tex` against worktree `build/wt/pr-fifo` at 4ce1bd188
(`scripts/run_examples.sh ... --all-passes`, `scripts/build_tex.py build/wt/pr-fifo`).

## Harness
- `fifotrace.py`: `edge_depth` wrapper takes `*args, **kwargs` and binds them with
  `inspect.signature`, so it follows signature changes (new `t_above`, `t_below`).
  Its re-derivation now includes the pair test of the throttled cap and the
  decoupled-chain rule (still asserted equal to the returned depth). Two ablation
  fields: `depth_without_margin` (cap margin removed, pair test kept) and
  `depth_without_pair` (the 39dac9a18 rule). The join-floor line tracer also records `c`.
- `fifo_examples.py`: `tav_input_blocks` / `tav_input_window` module functions;
  E2/E6 direct `edge_depth` calls pass the edge's `t_above`/`t_below`; new E3 variants
  with a slow S above X, new E3b (decoupled chain), new E11 (StreamingConcat floor,
  `make_concat_window` of the unit tests).
- E3's "CAP_MARGIN = 0" ablation no longer isolates the margin (CAP_MARGIN also bounds
  the pair test, so 0 removes the cap altogether, 512); the table column now comes
  from `depth_without_margin` (256, same values as before).

## Numbers
Every depth printed for E1-E10 and U1-U6 is unchanged against the 39dac9a18 output
(only random tensor names differ). New: E3 with S (2 / 2 / 512, old rule 256 / 512 / 256),
E3b (4 / 2 / 2 / 4), E11 (floors short 15, l2 12).

## Text
- Listings: all 30 `%%CODE` lines remapped (offsets +7 .. +173); the five hook
  listings in hwcustomop.py / inner_shuffle.py / matrixvectoractivation.py replaced by
  the module functions `tav_extra_latency`+`tav_input_blocks` (405-429),
  `tav_input_window` (432-463), `tav_input_need` (466-518). Node periods listing extended
  with `downstream_periods`; size_edge listing now includes `chain_above`.
- Rules: formula + R7 (pair test, cnv-w2a2 example from the .rst), new R7b decoupled
  chain (commits e3723ec90/6f734bc6e, also after 39dac9a18 though not in the brief),
  R5 exception note, R13 hook names, join floor (Concat, gcd join token, per-word
  input needs, yolov8 example), summary table, one new observation (cap vs. R7b
  overlap), appendix paths/commit. All in-text code line numbers updated.
- Nothing in the write-up referred to the MVAU tiled stall estimate, Split/Concat II
  or the free-running MVAU/VVAU HLS cores, so nothing to drop there.
