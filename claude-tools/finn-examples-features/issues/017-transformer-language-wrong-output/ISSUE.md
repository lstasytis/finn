# Issue 017: transformer_language (VCK190) stitched-IP rtlsim output wrong

Read first: `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md`, then
`issues/011-transformer-radioml-wrong-output/REPORT.md` (same model family; two upstream bugs found there:
GlobalAccPool non-square, Requant_rtl signed-only — both fixed on the integration/test2 branches; its node-by-node
tooling `scripts/nbn2.py` etc.) and `issues/015-fifo-skip-edge-deadlock/REPORT.md` (the deadlock fix that made
language finish).

Symptom: with the issue-015 fix (`fix/015-int3` = integration + fix/015, worktree build/wt/fix015-int3, logs under
`/home/lstasytis/finn/build/finn_build/fx/fix015-int3/logs/`), `test_transformer_fifo_sizing[VCK190-transformer_language-heuristic_analytical-small]`
no longer hangs but the output is wrong: 12.6 % of values over atol 0.5, max |diff| 4.56. FIFO depths cannot change
output values, so a node computes wrong data. language is the GPT-style masked LM (embedding Gather/Lookup, int64
token input, 2 layers, 2048-way output with Requant).

Tasks: (1) Python execution of the HW graph vs the golden pair (is the golden/flow correct?); (2) node-by-node
(cppsim/rtlsim per node, as in issue 011) to find the first diverging node; (3) root cause + minimal reproducer;
(4) fix on the owning branch (an upstream dev bug -> a separate, documented fix branch off `upstream/dev` with a PR
description file `PR_<branch>.md` in this folder; one of ours -> that feature branch) and verify the language small
FIFO test passes (golden output, rtlsim >= 0.98 x estimate).

Rules: at most 16 threads; start nothing while MemAvailable < 200 GB; kill only recorded PIDs (never `pkill -f`/`pgrep
-f` or a `ps | grep` pattern); RTL/HLS library changes only as a separate documented dev fix branch; never
`largefifo_rtlsim`; no pushes; don't touch other worktrees, build dirs or the campaign; list deletions. Commit messages
end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Write REPORT.md next to this file (if refused, put
its full text into your final answer after a <= 10-line summary).
