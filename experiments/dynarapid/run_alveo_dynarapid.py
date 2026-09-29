"""DynaRapid on an Alveo (Vitis) target, starting from a model prepared for linking.

Takes the model saved after FINN's Vitis bitfile step (or any model after PrepareForLinking:
StreamingDataflowPartition nodes whose kernel models carry `vitis_xo`), reuses the IODMA
kernels, and builds the xclbin with the compute kernel placed and routed by DynaRapid in a
cached Vitis shell (finn.util.dynarapid.alveo).

Usage:
    python run_alveo_dynarapid.py --model <step_synthesize_bitfile.onnx> --out <dir> \\
        --library <x>/lib [--shell-lib <dir>] [--board U55C] [--workers N]
"""

import argparse
import json
import os
import time

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.util.basic import vitis_default_platform, vitis_part_map
from finn.util.dynarapid.alveo import dynarapid_alveo_build


def link_kernels(model):
    """(compute kernel model, kernel list in link order) of a model prepared for linking."""
    kernels, compute = [], None
    for node in model.graph.node:
        assert node.op_type == "StreamingDataflowPartition", "not a link graph: %s" % node.op_type
        sdp = getCustomOp(node)
        km = ModelWrapper(sdp.get_nodeattr("model"))
        inst = sdp.get_nodeattr("instance_name") or node.name
        is_dma = inst.startswith("idma") or inst.startswith("odma")
        if is_dma:
            kernels.append({"name": node.name, "inst": inst, "xo": km.get_metadata_prop("vitis_xo"),
                            "mm": True})
        else:
            assert compute is None, "only one compute kernel is supported"
            compute = km
            kernels.append({"name": node.name, "inst": inst, "xo": None, "mm": False})
    return compute, kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--shell-lib", default=None)
    ap.add_argument("--board", default="U55C")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--clk-ns", type=float, default=None)
    args = ap.parse_args()
    t0 = time.time()
    model = ModelWrapper(args.model)
    compute, kernels = link_kernels(model)
    clk_ns = args.clk_ns or float(compute.get_metadata_prop("clk_ns") or 0) or None
    assert clk_ns, "clock period unknown, pass --clk-ns"
    res = dynarapid_alveo_build(
        compute,
        kernels,
        vitis_default_platform[args.board],
        vitis_part_map[args.board],
        clk_ns,
        args.out,
        args.library,
        shell_lib=args.shell_lib,
        workers=args.workers,
    )
    res["script_s"] = time.time() - t0
    print(json.dumps({k: v for k, v in res.items() if k not in ("accel_components",)}, indent=2))


if __name__ == "__main__":
    main()
