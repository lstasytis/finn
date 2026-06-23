#!/usr/bin/env python3
"""AlphaEvolve-style loop that optimizes a FINN node's get_tree_model.

Each iteration:
  1. asks the LLM (via llm_adapter.propose_candidate) for a new get_tree_model,
     given the current best source and the latest TAV-delta feedback;
  2. evaluates it with the tav_eval validator (splice -> docker pytest ->
     per-case delta vs the rtlsim cache);
  3. scores the result (lower = closer to the rtlsim reference) and keeps the
     best candidate so far;
  4. stops early once a candidate matches the reference on every case.

The loop never needs the LLM to be reachable from this repo's CI -- you point it
at your local model with --llm-cmd / $TAV_LLM_CMD (see llm_adapter.py).

Outputs (under $FINN_HOST_BUILD_DIR/tav_evolve/<node>-<ts>/):
  best_get_tree_model.py   the best candidate found
  history.json             per-iteration scores + log paths
  candidates/iter_NNN.py   every candidate tried

Usage:
  python tools/tav_eval/evolve.py ConvolutionInputGenerator \\
      --test tests/fpgadataflow/test_fpgadataflow_downsampler.py::test_fpgadataflow_analytical_characterization_downsampler \\
      --iterations 20 --llm-cmd 'python tools/Bespoke-Base-Retreat-26/generate.py'
"""

import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tav_eval  # noqa: E402
import llm_adapter  # noqa: E402


def _baseline_for(node, src_override=None, test_override=None):
    """Default seed candidate = the per-node baseline example we ship."""
    entry = tav_eval.resolve_node(node, src_override, test_override)
    base = os.path.basename(entry["src"]).replace(".py", "_tree_model.py")
    return os.path.join(HERE, "examples", base)


def _feedback_from_records(records, max_cases=12, vec_preview=24):
    """Render concise per-case feedback for the next prompt."""
    lines = []
    for r in records[:max_cases]:
        tag = tav_eval._verdict_tag(r)
        params = tav_eval._fmt_params(r.get("params", {}))
        if r.get("ports"):
            port_bits = []
            for p in r["ports"]:
                vec = p["delta_vector"][:vec_preview]
                more = len(p["delta_vector"]) - len(vec)
                vec_s = "[" + ", ".join(str(x) for x in vec) + (f", +{more}…]" if more > 0 else "]")
                port_bits.append(
                    f"{p['port']} peak={p['peak_volume_delta']} "
                    f"len_delta={p['len_delta']} delta={vec_s}"
                )
            lines.append(f"  [{tag}] {params} | " + " | ".join(port_bits))
        else:
            tail = (r.get("longrepr") or "").splitlines()
            lines.append(f"  [{tag}] {params} | {tail[-1] if tail else ''}")
    return "\n".join(lines)


def evolve(
    node,
    test_override=None,
    src_override=None,
    baseline=None,
    iterations=20,
    llm_cmd=None,
    cache_dir=None,
    out_dir=None,
    apply_best=False,
    include_extra_tests=False,
):
    baseline = baseline or _baseline_for(node, src_override, test_override)
    if not os.path.isfile(baseline):
        raise SystemExit(f"baseline candidate not found: {baseline}")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_node = re.sub(r"[^A-Za-z0-9_.-]", "_", node)
    out_dir = out_dir or os.path.join(
        tav_eval._host_build_dir(), "tav_evolve", f"{safe_node}-{run_id}"
    )
    cand_dir = os.path.join(out_dir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    def _evaluate(path):
        _, records = tav_eval.evaluate_tree_model(
            node,
            path,
            src_override=src_override,
            test_override=test_override,
            include_extra_tests=include_extra_tests,
            cache_dir=cache_dir,
            quiet=True,
            return_records=True,
        )
        return records, tav_eval.score_records(records)

    history = []

    # iteration 0: the baseline seed
    print(f"[evolve] evaluating baseline: {baseline}", file=sys.stderr)
    records, sc = _evaluate(baseline)
    best_source = open(baseline).read()
    best = {"iteration": 0, "source": best_source, **sc}
    history.append({"iteration": 0, "candidate": baseline, **sc})
    feedback = _feedback_from_records(records)
    print(f"[evolve] baseline score={sc['score']} "
          f"(pass={sc['n_pass']} fail={sc['n_fail']} error={sc['n_error']})", file=sys.stderr)

    if best["solved"]:
        print("[evolve] baseline already solves all cases; nothing to optimize.", file=sys.stderr)

    for it in range(1, iterations + 1):
        if best["solved"]:
            break
        ctx = {
            "node": node,
            "iteration": it,
            "max_iterations": iterations,
            "current_source": best["source"],
            "best_score": best["score"],
            "baseline_source": open(baseline).read(),
            "feedback": feedback,
            "history": history,
            "llm_cmd": llm_cmd,
            "repo_root": tav_eval.FINN_ROOT,
        }
        try:
            cand_source = llm_adapter.propose_candidate(ctx)
        except Exception as e:  # noqa: BLE001 - surface and continue
            print(f"[evolve] iter {it}: LLM/proposal error: {e}", file=sys.stderr)
            history.append({"iteration": it, "error": f"proposal: {e}"})
            continue

        cand_path = os.path.join(cand_dir, f"iter_{it:03d}.py")
        with open(cand_path, "w") as f:
            f.write(cand_source)

        try:
            records, sc = _evaluate(cand_path)
        except Exception as e:  # noqa: BLE001
            print(f"[evolve] iter {it}: evaluation error: {e}", file=sys.stderr)
            history.append({"iteration": it, "candidate": cand_path, "error": f"eval: {e}"})
            continue

        feedback = _feedback_from_records(records)
        improved = sc["score"] < best["score"]
        print(
            f"[evolve] iter {it}: score={sc['score']} "
            f"(pass={sc['n_pass']} fail={sc['n_fail']} error={sc['n_error']})"
            f"{'  <-- new best' if improved else ''}",
            file=sys.stderr,
        )
        history.append({"iteration": it, "candidate": cand_path, **sc})
        if improved:
            best = {"iteration": it, "source": cand_source, **sc}

    # persist results
    best_path = os.path.join(out_dir, "best_get_tree_model.py")
    with open(best_path, "w") as f:
        f.write(best["source"])
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"node": node, "best": {k: v for k, v in best.items() if k != "source"},
                   "history": history}, f, indent=1)

    # leave the node source pristine unless asked to apply the best candidate
    entry = tav_eval.resolve_node(node, src_override, test_override)
    src_path = entry["src"]
    if not os.path.isabs(src_path):
        src_path = os.path.join(tav_eval.FINN_ROOT, src_path)
    tav_eval.restore_original(src_path)
    if apply_best:
        tav_eval.replace_function(src_path, best_path)
        print(f"[evolve] applied best candidate to {src_path}", file=sys.stderr)

    print(f"\n[evolve] done. best score={best['score']} at iteration {best['iteration']}")
    print(f"[evolve] best candidate: {best_path}")
    print(f"[evolve] history:        {os.path.join(out_dir, 'history.json')}")
    return best_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("node", help="node name, e.g. ConvolutionInputGenerator (tav_eval.py --list)")
    p.add_argument("--test", help="pytest nodeid to optimize against (overrides the registry default)")
    p.add_argument("--src", help="override the node source file path")
    p.add_argument("--baseline", help="seed candidate .py (default: examples/<src>_tree_model.py)")
    p.add_argument("-n", "--iterations", type=int, default=20, help="max LLM iterations")
    p.add_argument("--llm-cmd", help="shell command for the LLM (else $TAV_LLM_CMD); "
                   "prompt on stdin, candidate get_tree_model on stdout")
    p.add_argument("--cache-dir", help="rtlsim reference cache dir (default <repo>/cached_models)")
    p.add_argument("--out", help="output directory (default under $FINN_HOST_BUILD_DIR/tav_evolve)")
    p.add_argument("--apply-best", action="store_true",
                   help="splice the best candidate into the node source at the end")
    p.add_argument("--extra-tests", action="store_true", help="also run registry extra_tests")
    args = p.parse_args(argv)
    evolve(
        args.node,
        test_override=args.test,
        src_override=args.src,
        baseline=args.baseline,
        iterations=args.iterations,
        llm_cmd=args.llm_cmd,
        cache_dir=args.cache_dir,
        out_dir=args.out,
        apply_best=args.apply_best,
        include_extra_tests=args.extra_tests,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
