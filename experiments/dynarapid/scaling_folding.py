"""Folding for the parallelism-scaling experiment (run_parallelism_scaling.sh).

A model is scaled by FINN's own folding search (SetFolding, builder target_fps): every layer gets
the smallest PE/SIMD that reaches the target throughput, so resources grow with the target. The
board's hand-written folding config is still applied afterwards for everything that is not
parallelism (memory modes, RAM styles, resource types, FIFO depths): its PE / SIMD /
parallel_window entries are dropped, otherwise they would override the search.
"""

import json
import os

PARALLELISM_ATTRS = ("PE", "SIMD", "parallel_window")


def stripped_folding_config(base, out_dir):
    """Write `base` without its parallelism attributes to out_dir; returns the path."""
    cfg = json.load(open(base))
    for v in cfg.values():
        if isinstance(v, dict):
            for a in PARALLELISM_ATTRS:
                v.pop(a, None)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "folding_noparallelism_" + os.path.basename(base))
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def scaling_kwargs(base_folding, target_fps, out_dir, relax=True):
    """DataflowBuildConfig arguments for a scaled build (None: the hand-written folding).
    relax=False: no second folding pass at the bottleneck's throughput. FINN's dataflow produces
    at most one output pixel per cycle per sliding window, which caps the throughput of VGG10 and
    MobileNet near their hand-written foldings; with relaxation every layer is then folded for
    that cap and resources stop growing. Without it, the other layers keep growing towards the
    unreachable target: more resources at the same throughput (a resource knob for build time)."""
    if target_fps is None:
        return {"folding_config_file": base_folding}
    return {
        "folding_config_file": stripped_folding_config(base_folding, out_dir),
        "target_fps": int(target_fps),
        "folding_two_pass_relaxation": relax,
        # the default (36 bits) caps the weight stream of decoupled MVAUs far below the
        # hand-written configs (MobileNet U250: 32 x 3 x 4 bit)
        "mvau_wwidth_max": 1 << 16,
    }
