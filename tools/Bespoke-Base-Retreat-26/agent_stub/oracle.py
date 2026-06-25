"""TAV evaluation oracle: the single object both the loop and the agent's
``eval_tree_model`` tool call to score a candidate ``get_tree_model``.

Three backends, chosen by ``mode``:

  * ``"local"``  -- score in-process with ``tav_runtime`` (faithful copy of
    FINN's analytical derivation) against the cached rtlsim reference vectors.
    Microseconds, fully parallel, no docker. A candidate that matches locally is
    later **confirmed** by the loop with a real docker run before it counts.
  * ``"docker"`` -- every score is a real ``tav_eval.evaluate_tree_model`` run.
    Faithful by definition, but seconds per call and serialized.
  * ``"both"``   -- score locally (fast inner loop) but also expose an explicit
    docker check the agent may call.

The cached reference vectors + node metadata come from a single baseline docker
run (the plugin now captures ``rtlsim_vector`` and ``node_meta``); the loop hands
the resulting records to :meth:`Oracle.load_cases`.

Scoring mirrors ``tav_eval.score_records`` so local and docker scores are
comparable: per port ``(Σ|delta| + |len_delta|) / max(len_rtlsim, 1) * 100``,
ERROR cases penalized heavily, ``solved`` iff every port matches exactly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agent_stub import tav_runtime

ERROR_PENALTY = 1_000_000.0
PORT_NAMES = ("input", "output")


# ---------------------------------------------------------------------------
# run-length encoding for compact, lossless vector display in feedback
# ---------------------------------------------------------------------------
def rle(vec):
    pairs = []
    for v in vec:
        if pairs and pairs[-1][1] == v:
            pairs[-1] = (pairs[-1][0] + 1, v)
        else:
            pairs.append((1, v))
    return pairs


def fmt_rle(vec, cap=None):
    """Render a vector as run-length pairs. If ``cap`` is set and the encoding
    has more than ``cap`` pairs, show only the first ``cap`` and note how many
    were elided -- used to keep the (large, redundant) TARGET/YOURS dumps bounded
    while the DELTA, which is the actionable signal, is always shown in full."""
    if not len(vec):
        return "(empty)"
    pairs = rle(vec)
    shown = pairs if (cap is None or len(pairs) <= cap) else pairs[:cap]
    s = ", ".join(f"({n},{v:+d})" if v else f"({n},0)" for n, v in shown)
    if cap is not None and len(pairs) > cap:
        s += f", ... (+{len(pairs) - cap} more pairs)"
    return s


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """One parametrized characterization case, with its rtlsim target vectors
    and the node metadata needed to replay get_tree_model locally."""

    tag: str
    nodeid: str
    params: dict
    class_name: str
    onnx_node_name: str
    op_type: str
    node_attrs: dict
    ref_in: list
    ref_out: list
    analytical_in: list = field(default_factory=list)
    analytical_out: list = field(default_factory=list)
    volume: int = 0  # rough size, for curriculum ordering (smallest first)

    @property
    def has_meta(self) -> bool:
        return bool(self.class_name and self.node_attrs)

    def to_runtime_case(self) -> dict:
        return {
            "class_name": self.class_name,
            "onnx_node_name": self.onnx_node_name,
            "op_type": self.op_type,
            "node_attrs": self.node_attrs,
            "analytical_in": self.analytical_in,
            "analytical_out": self.analytical_out,
        }


def _ports_by_name(record):
    out = {}
    for p in record.get("ports", []):
        out[p.get("port")] = p
    return out


def cases_from_records(records) -> list[Case]:
    """Build local-oracle cases from the plugin's JSON records (as returned by
    ``tav_eval.evaluate_tree_model(..., return_records=True)``)."""
    cases = []
    for rec in records:
        meta = rec.get("node_meta") or {}
        ports = _ports_by_name(rec)
        pin = ports.get("input", {})
        pout = ports.get("output", {})
        ref_in = pin.get("rtlsim_vector", [])
        ref_out = pout.get("rtlsim_vector", [])
        # skip records with no reference at all (ERROR/SKIP with empty capture)
        if not ref_in and not ref_out:
            continue
        params = rec.get("params", {})
        tag = " ".join(f"{k}={v}" for k, v in params.items()) or rec.get("nodeid", "")
        cases.append(
            Case(
                tag=tag,
                nodeid=rec.get("nodeid", ""),
                params=params,
                class_name=meta.get("class_name", ""),
                onnx_node_name=meta.get("onnx_node_name", ""),
                op_type=meta.get("op_type", ""),
                node_attrs=meta.get("node_attrs", {}),
                ref_in=list(ref_in),
                ref_out=list(ref_out),
                analytical_in=list(pin.get("analytical_vector", [])),
                analytical_out=list(pout.get("analytical_vector", [])),
                volume=max(len(ref_in), len(ref_out)),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# per-evaluation results
# ---------------------------------------------------------------------------
def _port_delta(got, ref):
    n = min(len(got), len(ref))
    delta = [got[i] - ref[i] for i in range(n)]
    len_delta = len(got) - len(ref)
    abs_delta = sum(abs(x) for x in delta)
    matched = (abs_delta == 0) and (len_delta == 0)
    return delta, len_delta, abs_delta, matched


@dataclass
class PortResult:
    port: str
    ref: list
    got: list
    delta: list
    len_delta: int
    abs_delta: int
    matched: bool

    def score(self) -> float:
        denom = max(len(self.ref), 1)
        return (self.abs_delta + abs(self.len_delta)) / denom * 100.0


@dataclass
class CaseResult:
    tag: str
    params: dict
    ports: list = field(default_factory=list)
    error: str | None = None

    @property
    def matched(self) -> bool:
        return self.error is None and all(p.matched for p in self.ports)

    def score(self) -> float:
        if self.error is not None:
            return ERROR_PENALTY
        return sum(p.score() for p in self.ports)


@dataclass
class EvalResult:
    cases: list = field(default_factory=list)
    backend: str = "local"

    @property
    def score(self) -> float:
        return sum(c.score() for c in self.cases)

    @property
    def solved(self) -> bool:
        return bool(self.cases) and all(c.matched for c in self.cases)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.cases if c.matched)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.cases if not c.matched and c.error is None)

    @property
    def n_error(self) -> int:
        return sum(1 for c in self.cases if c.error is not None)


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------
class Oracle:
    def __init__(
        self,
        node,
        mode="local",
        *,
        src_override=None,
        test_override=None,
        cache_dir=None,
        include_extra_tests=False,
        show_full_target=True,
        target_display_cap=120,
        feedback_char_budget=40000,
        max_detail_cases=10,
    ):
        self.node = node
        self.mode = mode
        self.src_override = src_override
        self.test_override = test_override
        self.cache_dir = cache_dir
        self.include_extra_tests = include_extra_tests
        self.show_full_target = show_full_target
        # cap (in RLE pairs) for EVERY vector shown in detail (TARGET, YOURS and
        # DELTA). Because the RLE front-loads the matching prefix then the onset
        # of divergence, the first ~120 pairs capture where and how the mismatch
        # begins -- the actionable part for a cumulative TAV.
        self.target_display_cap = target_display_cap
        # hard char budget for the detailed section of one feedback message, so a
        # node with many/long cases (e.g. 39 SWG cases, some 12k cycles) can't
        # produce a single message that blows the model's input limit.
        self.feedback_char_budget = feedback_char_budget
        self.max_detail_cases = max_detail_cases
        self.cases: list[Case] = []
        self._active_tags: set | None = None  # curriculum: restrict to a subset
        # docker splices the shared FINN source file, so concurrent docker evals
        # (parallel population in --oracle docker) must be serialized.
        self._docker_lock = threading.Lock()

    # -- case management ---------------------------------------------------
    def load_cases(self, records):
        self.cases = cases_from_records(records)
        return self.cases

    def set_active_cases(self, tags):
        """Curriculum: restrict local scoring to these case tags (None = all)."""
        self._active_tags = set(tags) if tags is not None else None

    def _active(self) -> list[Case]:
        if self._active_tags is None:
            return self.cases
        return [c for c in self.cases if c.tag in self._active_tags]

    def cases_by_volume(self) -> list[Case]:
        return sorted(self.cases, key=lambda c: c.volume)

    def selftest(self, baseline_source) -> tuple[bool, str]:
        """Confirm the host reimplementation matches FINN's captured analytical
        vectors on every case. Run on the baseline before trusting --oracle local."""
        if self.mode == "docker":
            return True, "docker mode: no local reimplementation to self-test"
        bad = []
        for c in self.cases:
            if not c.analytical_in and not c.analytical_out:
                continue
            ok, msg = tav_runtime.selftest_against_capture(baseline_source, c.to_runtime_case())
            if not ok:
                bad.append(f"[{c.tag}] {msg}")
        if bad:
            return False, "local oracle self-test FAILED on:\n  " + "\n  ".join(bad)
        return True, "local oracle self-test passed on all cases with captured vectors"

    # -- local scoring -----------------------------------------------------
    def evaluate_local(self, source) -> EvalResult:
        results = []
        for c in self._active():
            if not c.has_meta:
                results.append(CaseResult(c.tag, c.params, error="no captured node metadata for this case"))
                continue
            try:
                got_in, got_out = tav_runtime.derive_for_case(source, c.to_runtime_case())
            except Exception as e:  # candidate is broken / referenced a missing attr
                results.append(CaseResult(c.tag, c.params, error=f"{type(e).__name__}: {e}"))
                continue
            ports = []
            for name, got, ref in (("input", got_in, c.ref_in), ("output", got_out, c.ref_out)):
                if not ref:
                    continue
                delta, len_delta, abs_delta, matched = _port_delta(got, ref)
                ports.append(PortResult(name, ref, got, delta, len_delta, abs_delta, matched))
            results.append(CaseResult(c.tag, c.params, ports=ports))
        return EvalResult(results, backend="local")

    # -- docker scoring ----------------------------------------------------
    def evaluate_docker(self, source_path) -> dict:
        """Run the real validator. Returns dict with score/records/solved using
        tav_eval.score_records so it lines up with local scores."""
        import tav_eval  # added to sys.path by the loop

        with self._docker_lock:
            _, records = tav_eval.evaluate_tree_model(
                self.node,
                source_path,
                src_override=self.src_override,
                test_override=self.test_override,
                include_extra_tests=self.include_extra_tests,
                cache_dir=self.cache_dir,
                quiet=True,
                return_records=True,
            )
        sc = tav_eval.score_records(records)
        return {"score": sc["score"], "solved": sc["solved"], "sc": sc, "records": records}

    @staticmethod
    def _needs_docker_escalation(result: EvalResult) -> bool:
        """True if a case errored because the candidate called a node method the
        host MockSelf doesn't provide (e.g. a real sibling helper like
        get_tree_model_uniform_distribution_based, which exists after splicing
        but not on the mock). Those are valid in docker, so the fast local oracle
        must not score them as failures -- escalate to docker instead."""
        for c in result.cases:
            if c.error and "has no attribute" in c.error:
                return True
        return False

    # -- unified entry point used by the agent tool ------------------------
    def evaluate(self, source, source_path=None) -> EvalResult:
        if self.mode == "docker":
            d = self.evaluate_docker(source_path)
            return self._eval_from_records(d["records"])
        local = self.evaluate_local(source)
        # In 'both' mode, a candidate that used a real node method (not on the
        # mock) is faithfully scorable only in docker -- escalate it rather than
        # report a false ERROR that steers the agent away from a valid approach.
        if self.mode == "both" and source_path and self._needs_docker_escalation(local):
            try:
                d = self.evaluate_docker(source_path)
                return self._eval_from_records(d["records"])
            except Exception:
                return local  # docker unavailable/failed -- keep local feedback
        return local

    def _eval_from_records(self, records) -> EvalResult:
        """Render docker records into the same EvalResult shape as local, so the
        agent sees identical feedback regardless of backend."""
        results = []
        for rec in records:
            # skipped cases (unsupported param combos) are not failures and not
            # errors -- score_records ignores them, so must we, or every escalated
            # candidate eats a bogus ERROR_PENALTY for the test's skipped rows.
            if rec.get("outcome") == "skipped":
                continue
            ports = []
            err = None
            if rec.get("outcome") not in ("passed", "failed"):
                err = rec.get("longrepr") or rec.get("outcome")
            for p in rec.get("ports", []):
                ref = p.get("rtlsim_vector", [])
                got = p.get("analytical_vector", [])
                delta, len_delta, abs_delta, matched = _port_delta(got, ref)
                ports.append(PortResult(p.get("port", "?"), ref, got, delta, len_delta, abs_delta, matched))
            tag = " ".join(f"{k}={v}" for k, v in rec.get("params", {}).items())
            results.append(CaseResult(tag, rec.get("params", {}), ports=ports, error=err))
        return EvalResult(results, backend="docker")

    # -- feedback rendering ------------------------------------------------
    def _compact_line(self, c) -> str:
        """One-line summary of a case -- cheap, shown for ALL cases so the agent
        always sees the full landscape regardless of how many cases there are."""
        if c.error is not None:
            return f"  [{c.tag}] ERROR: {c.error[:140]}"
        if c.matched:
            return f"  [{c.tag}] exact"
        parts = []
        for p in c.ports:
            if p.matched:
                parts.append(f"{p.port}=exact")
            else:
                parts.append(f"{p.port} Σ|d|={p.abs_delta} len_d={p.len_delta:+d}")
        return f"  [{c.tag}] FAIL | " + " | ".join(parts)

    @staticmethod
    def _port_status(cases, port):
        """(#cases whose `port` is exact, #cases that have a `port`)."""
        have = [c for c in cases if c.error is None and any(p.port == port for p in c.ports)]
        exact = sum(1 for c in have if all(p.matched for p in c.ports if p.port == port))
        return exact, len(have)

    def _render_case_detail(self, c, focus_port=None) -> str:
        """Full per-port TARGET/YOURS/DELTA for one case, every vector RLE-capped.

        When ``focus_port`` is set, only that port is shown in full; the other
        (non-focus) port is collapsed to a one-liner, so detail tokens go to the
        port the agent is meant to be working on right now."""
        cap = self.target_display_cap
        out = [f"  [{c.tag}] (case score {round(c.score(), 1)}):"]
        for p in c.ports:
            if p.matched:
                out.append(f"     {p.port}: exact ({len(p.ref)} cycles)")
                continue
            if focus_port is not None and p.port != focus_port:
                out.append(
                    f"     {p.port}: differs (len_delta={p.len_delta:+d}, Σ|delta|={p.abs_delta}) "
                    f"-- deprioritized, get {focus_port} exact first"
                )
                continue
            peak = max((abs(x) for x in p.delta), default=0)
            out.append(
                f"     {p.port}: yours len={len(p.got)} target len={len(p.ref)} "
                f"(len_delta={p.len_delta:+d}), Σ|delta|={p.abs_delta}, peak={peak}"
            )
            if self.show_full_target:
                out.append(f"        TARGET (rtlsim): [{fmt_rle(p.ref, cap)}]")
                out.append(f"        YOURS          : [{fmt_rle(p.got, cap)}]")
            out.append(f"        DELTA (yours-target): [{fmt_rle(p.delta, cap)}]")
        return "\n".join(out)

    def format_feedback(self, result: EvalResult, *, prev_score=None) -> str:
        """Bounded feedback: a compact one-liner for every case (full landscape),
        plus full TARGET/YOURS/DELTA detail for the smallest failing cases within
        a char budget. Bounded regardless of case count / vector length, so a
        single message can never exceed the model's input limit."""
        n = len(result.cases)
        lines = [
            f"backend={result.backend} score={round(result.score, 2)} "
            f"(exact={result.n_pass} differ={result.n_fail} error={result.n_error} of {n} cases). "
            f"GOAL: drive score to 0 -- every case exact on both ports."
        ]
        if prev_score is not None:
            arrow = ("IMPROVED" if result.score < prev_score
                     else "WORSE" if result.score > prev_score else "unchanged")
            lines.append(f"  vs your previous attempt: {round(prev_score, 2)} -> "
                         f"{round(result.score, 2)} ({arrow})")
        if result.solved:
            lines.append("  ALL CASES EXACT -- this is the goal.")
            return "\n".join(lines)

        # per-port progress + the input-first focus phase
        in_exact, in_have = self._port_status(result.cases, "input")
        out_exact, out_have = self._port_status(result.cases, "output")
        lines.append(f"  INPUT ports: {in_exact}/{in_have} exact.  "
                     f"OUTPUT ports: {out_exact}/{out_have} exact.")

        def _port_fails(c, port):
            return any(p.port == port and not p.matched for p in c.ports)

        input_failing = [c for c in result.cases if c.error is None and _port_fails(c, "input")]
        output_failing = [c for c in result.cases if c.error is None and _port_fails(c, "output")]

        if input_failing:
            focus = "input"
            focus_failing = input_failing
            lines.append(
                f">> FOCUS: get EVERY input port exact first ({len(input_failing)} still differ). "
                "Input is the simpler port and the foundation the output model builds on. Overall "
                "score still matters, but prioritize input now -- output ports are summarized "
                "below, not shown in full, until all inputs match."
            )
        elif output_failing:
            focus = "output"
            focus_failing = output_failing
            lines.append(
                ">> All input ports are EXACT. FOCUS NOW: make every OUTPUT port exact while "
                "keeping the inputs exact."
            )
        else:
            focus, focus_failing = None, [c for c in result.cases if not c.matched]

        # full landscape, cheap: one line per case, worst first
        lines.append(f"\nAll {n} cases (worst first):")
        lines.extend(self._compact_line(c) for c in sorted(result.cases, key=lambda c: -c.score()))

        # detail, budgeted: smallest FOCUS-port-failing vectors first -- the per-cycle
        # pattern is clearest there and usually transfers to the larger cases.
        if focus_failing:
            label = f"{focus} port" if focus else "remaining"
            lines.append(f"\nDetail for the smallest cases whose {label} still differs "
                         "(per-cycle pattern is clearest here and usually transfers to bigger "
                         f"cases). Vectors are RLE'd (run_length,value), truncated after "
                         f"{self.target_display_cap} runs:")
            order = sorted(focus_failing, key=lambda c: max((len(p.ref) for p in c.ports), default=0))
            used, shown = 0, 0
            for c in order:
                if shown >= self.max_detail_cases:
                    break
                block = self._render_case_detail(c, focus_port=focus)
                if shown >= 1 and used + len(block) > self.feedback_char_budget:
                    break
                lines.append(block)
                used += len(block) + 1
                shown += 1
            if shown < len(focus_failing):
                lines.append(f"  (+{len(focus_failing) - shown} more case(s) with a failing "
                             f"{label} summarized above -- fix these first, then re-evaluate.)")
        return "\n".join(lines)
