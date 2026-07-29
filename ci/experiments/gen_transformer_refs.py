"""Harvest rtlsim reference TAVs for the transformer op types out of finished builds.

Why this and not ``gen_cig_refs.py``: the transformer sizing runs already paid
for per-node HLS synthesis and XSI rtlsim, and every node they touched still
carries its measured token access vectors in ``io_chrc_in_original`` /
``io_chrc_out_original`` inside ``step_set_fifo_depths.onnx``. Re-deriving those
would cost 6-13 minutes per model and a Vivado install; reading them costs
seconds and needs neither.

Note ``io_chrc_in`` (without ``_original``) is *not* the thing to read: the
stretch stage of the sizing chain overwrites it with row 0 only, so a
four-input StreamingConcat looks single-input there. ``_original`` keeps every
stream.

Writes one JSON per op type under ``tests/tav_refs/``, in the same shape as the
ConvolutionInputGenerator references, so ``tests/fpgadataflow/test_tav_tree_models.py``
can consume them with no Vivado and no board -- which is the point: the
references are committed, the checking is free, and it works in a tree that has
no working XSI at all.

    uv run python ci/experiments/gen_transformer_refs.py
    uv run python ci/experiments/gen_transformer_refs.py --op ElementwiseAdd_hls
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tests.testing_util.tav_refs import (  # noqa: E402
    REF_DIR,
    ensure_finn_env,
    node_key,
    node_spec,
)

# The op types this harvest is for: everything in the transformer models that
# had no tree model. Anything already covered is skipped -- its references are
# generated the expensive way, single node at a time, and are already committed.
OPS = [
    "ElementwiseAdd_hls",
    "ReplicateStream_hls",
    "StreamingSplit_hls",
    "StreamingConcat_hls",
    "Reshape_rtl",
    "Squeeze_hls",
    "Unsqueeze_hls",
    "Lookup_hls",
    "ScaledDotProductAttention_hls",
]


def ref_path(op_type):
    return os.path.join(REF_DIR, op_type.lower() + ".json")


def build_tav_strategy(build):
    """How this build derived its token access vectors, or None if unrecorded.

    A build run with ``tav_generation_strategy: tree_model`` writes the *tree
    model's own* schedule into ``io_chrc_*_original`` for every op type that has
    one. Harvesting those as "rtlsim references" makes the model its own
    reference and every score comes out exact for free. That is what the first
    version of this harvest did, and it is why Thresholding_rtl appeared to
    measure a period offset of 0 on 100 of 309 configurations while every
    genuinely-rtlsim source said 1, 2 or 4.
    """
    meta = os.path.join(build, "report", "metadata_bench.json")
    if not os.path.isfile(meta):
        return None
    try:
        return json.load(open(meta)).get("tav_generation_strategy")
    except Exception:
        return None


def _same_schedule(entry, tav_in, tav_out):
    import numpy as np
    from finn.util.basic import decompress_string_to_numpy

    for stored, fresh in ((entry["io_chrc_in"], tav_in), (entry["io_chrc_out"], tav_out)):
        a = np.atleast_2d(decompress_string_to_numpy(stored))
        b = np.atleast_2d(decompress_string_to_numpy(fresh))
        if a.shape != b.shape or not np.array_equal(a, b):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", action="append", default=[], help="restrict to these op types")
    ap.add_argument(
        "--builds",
        default=os.path.join(ROOT, "ci", "par", "*", "work", "buildflow", "build_output"),
        help="glob of build_output directories to harvest",
    )
    args = ap.parse_args()

    ensure_finn_env()
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    wanted = set(args.op) or set(OPS)
    refs = {}
    for op in wanted:
        path = ref_path(op)
        refs[op] = json.load(open(path)) if os.path.isfile(path) else {}

    seen_from = {}
    for build in sorted(glob.glob(args.builds)):
        onnx = os.path.join(build, "intermediate_models", "step_set_fifo_depths.onnx")
        if not os.path.isfile(onnx):
            continue
        job = build.split(os.sep)[-4]
        strategy = build_tav_strategy(build)
        try:
            model = ModelWrapper(onnx)
        except Exception as e:
            print("skip %s: %s" % (job, e), file=sys.stderr)
            continue
        for node in model.graph.node:
            if node.op_type not in wanted:
                continue
            inst = getCustomOp(node)
            if strategy != "rtlsim":
                # Only op types that had no tree model can have fallen back to
                # rtlsim in a tree-model build. Anything else would be the tree
                # model quoting itself.
                try:
                    if inst.get_tree_model() is not None:
                        continue
                except Exception:
                    continue
            try:
                tav_in = inst.get_nodeattr("io_chrc_in_original")
                tav_out = inst.get_nodeattr("io_chrc_out_original")
            except Exception:
                continue
            if not tav_in or not tav_out:
                continue
            spec = node_spec(inst)
            key = node_key(spec)
            if key in refs[node.op_type]:
                seen_from.setdefault(key, []).append(job)
                # The same configuration turning up in several builds is the
                # normal case. Two builds disagreeing about its measured
                # schedule is not: it would mean the schedule depends on
                # something the node attributes do not carry, and every tree
                # model here assumes it does not. Say so rather than silently
                # keeping whichever build was read first.
                old = refs[node.op_type][key]
                # Compare the decoded arrays, not the encoded strings: the
                # encoding gzips its payload and gzip stamps an mtime into the
                # header, so two identical schedules never encode to the same
                # bytes.
                if not _same_schedule(old, tav_in, tav_out):
                    print(
                        "CONFLICT %s: %s and %s/%s measured different schedules"
                        % (key, old["source"], job, node.name),
                        file=sys.stderr,
                    )
                continue
            refs[node.op_type][key] = {
                "spec": spec,
                "io_chrc_in": tav_in,
                "io_chrc_out": tav_out,
                "io_chrc_period": int(inst.get_nodeattr("io_chrc_period")),
                "source": "%s/%s" % (job, node.name),
                "provenance": "rtlsim" if strategy == "rtlsim" else "rtlsim-fallback",
            }
            seen_from.setdefault(key, []).append(job)

    os.makedirs(REF_DIR, exist_ok=True)
    for op in sorted(wanted):
        if not refs[op]:
            print("%-32s no nodes found" % op)
            continue
        with open(ref_path(op), "w") as f:
            json.dump(refs[op], f, indent=1, sort_keys=True)
        print("%-32s %d configurations -> %s" % (op, len(refs[op]), ref_path(op)))


if __name__ == "__main__":
    main()
