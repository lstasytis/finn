"""Harvest rtlsim reference TAVs for any op type out of the cached TAV caches.

``gen_transformer_refs.py`` reads ``io_chrc_*_original`` out of finished builds'
``step_set_fifo_depths.onnx``. Not every model keeps one -- mobilenetv1,
resnet50 and vgg10 are sized from ``tav_cache/<model>_zcu104.json`` instead, and
that cache is keyed by node *name* with no attributes, so it cannot be replayed
on its own.

This joins the two: rebuild the graph the cache was measured on (the same
InsertDWC + SpecializeLayers the sizer does), pair each cached node with its
attributes, and store the pair in the same node-replay format under
``tests/tav_refs/``. No rtlsim, no Vivado, seconds per model -- and it is the
only way to get a reference for ``MVAU_rtl`` (15 nodes, mobilenetv1) or
``VVAU_hls`` (13 nodes, the only model that has any).

    uv run python ci/experiments/harvest_cache_refs.py --op MVAU_rtl --op VVAU_hls
"""

import argparse
import contextlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.testing_util.tav_refs import REF_DIR, ensure_finn_env, node_key, node_spec  # noqa: E402


def ref_path(op_type):
    return os.path.join(REF_DIR, op_type.lower() + ".json")


def _same(a_str, b_str):
    import numpy as np

    from finn.util.basic import decompress_string_to_numpy

    a = np.atleast_2d(decompress_string_to_numpy(a_str))
    b = np.atleast_2d(decompress_string_to_numpy(b_str))
    return a.shape == b.shape and np.array_equal(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", action="append", default=[])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_finn_env()
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.general import GiveUniqueNodeNames

    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

    import tr_loop

    wanted = set(args.op)
    models = args.models or sorted(tr_loop.MODELS)
    refs, added, conflicts = {}, {}, []
    for op in wanted:
        p = ref_path(op)
        refs[op] = json.load(open(p)) if os.path.isfile(p) else {}

    for name in models:
        cfg = tr_loop.MODELS[name]
        cache_path = cfg["cache"]
        if not os.path.isfile(cache_path):
            continue
        blob = json.load(open(cache_path))
        strategy = str(blob.get("tav_generation_strategy", ""))
        if "RTLSIM" not in strategy.upper():
            print("%-14s skip: cache was derived with %s" % (name, strategy))
            continue
        cache = blob["nodes"]
        src = os.path.join(
            cfg["size"], "intermediate_models", "step_generate_estimate_reports.onnx"
        )
        if not os.path.isfile(src):
            print("%-14s no build" % name)
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = ModelWrapper(src)
                model = model.transform(InsertDWC())
                model = model.transform(SpecializeLayers(tr_loop.PART))
                model = model.transform(GiveUniqueNodeNames())
        except Exception as e:
            print("%-14s skip: %s" % (name, e))
            continue
        n_here = 0
        for node in model.graph.node:
            if node.op_type not in wanted or node.name not in cache:
                continue
            entry = cache[node.name]
            tav_in = entry.get("io_chrc_in_original") or entry.get("io_chrc_in")
            tav_out = entry.get("io_chrc_out_original") or entry.get("io_chrc_out")
            if not tav_in or not tav_out:
                continue
            inst = getCustomOp(node)
            spec = node_spec(inst)
            key = node_key(spec)
            if key in refs[node.op_type]:
                old = refs[node.op_type][key]
                if not (_same(old["io_chrc_in"], tav_in) and _same(old["io_chrc_out"], tav_out)):
                    conflicts.append("%s: %s vs %s/%s" % (key, old["source"], name, node.name))
                continue
            refs[node.op_type][key] = {
                "spec": spec,
                "io_chrc_in": tav_in,
                "io_chrc_out": tav_out,
                "io_chrc_period": int(entry.get("io_chrc_period", 0)),
                "source": "cache:%s/%s" % (name, node.name),
                # every tav_cache in this project was derived with
                # tav_generation_strategy=rtlsim; the harvester asserts it below
                "provenance": "rtlsim",
            }
            added[node.op_type] = added.get(node.op_type, 0) + 1
            n_here += 1
        print("%-14s +%d" % (name, n_here))

    for c in conflicts:
        print("CONFLICT %s" % c, file=sys.stderr)
    if args.dry_run:
        return
    os.makedirs(REF_DIR, exist_ok=True)
    for op in sorted(wanted):
        if not refs[op]:
            print("%-32s nothing found" % op)
            continue
        with open(ref_path(op), "w") as f:
            json.dump(refs[op], f, indent=1, sort_keys=True)
        print("%-32s %d configurations (+%d) -> %s"
              % (op, len(refs[op]), added.get(op, 0), ref_path(op)))


if __name__ == "__main__":
    main()
