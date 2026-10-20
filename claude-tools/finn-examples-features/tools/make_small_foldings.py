"""Write <name>_small.json next to each committed folding config: PE/SIMD of every node divided by
K (tooling only).

    python make_small_foldings.py <worktree> [model-dir regex]

For each (model, board) of the feature tests, the cached reference estimate build of the worktree
($FINN_BUILD_DIR/est_<tag>_<board>_*, built if missing via the test module's reference_estimate)
gives the model with the committed folding applied (step_apply_folding_config.onnx). Per node, the PE/SIMD
meta parameters of SetFolding's optimizer (their value lists respect every op's constraints and the
SWG pairing rules) take the largest total parallelism <= current / K, scaled as evenly as possible. Nodes the
optimizer does not fold (LabelSelect PE, Concat/Split SIMD) keep their committed values. Prints
committed vs small estimated cycles per model.
"""
import copy
import importlib.util
import itertools
import json
import os
import re
import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.set_folding import Optimizer
from finn.util.basic import part_map

WT = os.path.abspath(sys.argv[1])
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
BENCH = os.path.join(WT, "tests/benchmark")
# K per model dir: small models are already small (tfc-w1a1 is near fully unrolled)
K = {"bnn-pynq": 4, "kws": 4, "cybersecurity-mlp": 4, "gtsrb": 4}
K_DEFAULT = 8
FOLD_ATTRS = ("PE", "SIMD", "parallel_window")


def load_test_module(mdir):
    sys.path.insert(0, BENCH)
    sys.path.insert(0, os.path.join(BENCH, mdir))
    path = os.path.join(BENCH, mdir, f"test_build_{mdir}.py")
    spec = importlib.util.spec_from_file_location(f"tb_{mdir.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def actual_value(meta):
    """The meta value the node attributes encode, also when it is outside the optimizer's value
    set (e.g. a committed MVAU SIMD above mvau_wwidth_max). Parameter order per pairing rule:
    SWG -> MVAU [SWG SIMD, parallel_window, MVAU SIMD] (value (MVAU SIMD, SWG SIMD, pw)),
    SWG -> VVAU [SWG SIMD, parallel_window, VVAU SIMD, VVAU PE] (value SIMD * PE),
    SWG -> Pool [SWG SIMD, Pool PE] and single nodes: the last parameter's attribute."""
    reals = meta.real_values[0]
    get = [p.node.get_nodeattr(p.attr) for p in reals]
    if isinstance(meta.possible_values[0], tuple):
        return (get[2], get[0], get[1])
    if len(reals) == 4:
        return get[2] * get[3]
    return get[-1]


def primary(v):
    """A meta value's parallelism: the MVAU SIMD of an SWG -> MVAU pair, else the value."""
    return v[0] if isinstance(v, tuple) else v


def scaled_group(metas, curs, k):
    """New values for one node's PE/SIMD metas: total parallelism (product) <= current / K, as
    large as possible, the factors as evenly scaled as possible; the SWG side of a pair is
    scaled by K on its own."""
    target = np.prod([primary(c) for c in curs]) / k
    firsts = [sorted({primary(v) for v in m.possible_values}) for m in metas]
    combos = list(itertools.product(*firsts))
    fit = [c for c in combos if np.prod(c) <= target] or [min(combos, key=np.prod)]
    even = np.log(k) / len(metas)

    def skew(c):
        return sum(abs(np.log(primary(cur) / v) - even) for v, cur in zip(c, curs))

    best = max(fit, key=lambda c: (np.prod(c), -skew(c)))
    out = []
    for m, cur, v in zip(metas, curs, best):
        if not isinstance(cur, tuple):
            out.append(v)
            continue
        same = [x for x in m.possible_values if x[0] == v]
        swg = [x for x in same if x[1] <= max(1, cur[1] / k)] or same
        out.append(max(swg, key=lambda x: (-x[2], x[1])))
    return out


def max_cycles(model):
    return model.transform(AnnotateCycles()).analysis(dataflow_performance)["max_cycles"]


def small_config(model, committed, board, k):
    opt = Optimizer(copy.deepcopy(model), "small", {"max_cycles": 1}, fpgapart=part_map[board])
    opt.generate_parameter_set()
    kept, groups = [], {}
    for meta in opt.params.parameters:
        if meta.name in ("SIMD", "PE"):
            groups.setdefault(meta.node_index, []).append(meta)
    for metas in groups.values():
        curs = [actual_value(m) for m in metas]
        if any(c not in m.possible_values for m, c in zip(metas, curs)):
            kept.append(metas[0].unique_nodes[0].onnx_node.name)
        for m, v in zip(metas, scaled_group(metas, curs, k)):
            m.update_value(v)
            m.apply_value(final=False, attrs=FOLD_ATTRS)
    cfg = copy.deepcopy(committed)
    before = {n.name: n for n in model.graph.node}
    for node in opt.model.graph.node:
        old = before[node.name]
        old_attrs = {a.name: a for a in old.attribute}
        for a in node.attribute:
            if a.name in FOLD_ATTRS and (a.name not in old_attrs or a != old_attrs[a.name]):
                cfg.setdefault(node.name, {})[a.name] = a.i
    return cfg, opt.model, kept


for mdir in sorted(os.listdir(BENCH)):
    if not os.path.isfile(os.path.join(BENCH, mdir, f"test_build_{mdir}.py")):
        continue
    if ONLY and not re.search(ONLY, mdir):
        continue
    mod = load_test_module(mdir)
    if not hasattr(mod, "FEATURE_BUILDS"):
        continue
    for build in mod.FEATURE_BUILDS:
        args = tuple(build.values) if hasattr(build, "values") else (build,)
        board = args[0]
        k = K.get(mdir, K_DEFAULT)
        cfg = mod.configure_build(*args, "/tmp/unused")
        committed_path = cfg.folding_config_file
        est = mod.reference_estimate(*args)
        model = ModelWrapper(os.path.join(est, "intermediate_models/step_apply_folding_config.onnx"))
        with open(committed_path) as f:
            committed = json.load(f)
        small, folded, kept = small_config(model, committed, board, k)
        out = committed_path[: -len(".json")] + "_small.json"
        with open(out, "w") as f:
            json.dump(small, f, indent=2)
            f.write("\n")
        c0, c1 = max_cycles(model), max_cycles(folded)
        print(
            f"{mdir} {args}: K={k} committed {c0} -> small {c1} cycles ({c1 / c0:.1f}x)"
            + (f", committed value outside the optimizer's set: {kept}" if kept else ""),
            flush=True,
        )
