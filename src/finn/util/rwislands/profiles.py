# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vivado settings of the platform's regular FINN flow (the baseline). Node synthesis, island
place and route and the assembly use the same ones, so that both flows get the same
optimizations:

  zynq:  FINN's Zynq template, synth_1 Flow_PerfOptimized_high, impl_1 Performance_ExtraTimingOpt
         (steps as in its generated impl script)
  vitis: v++ defaults (opt, place, phys_opt, route; no post-route phys_opt), default synthesis
"""

import os

PROFILES = {
    "zynq": {
        "synth": "-directive PerformanceOptimized -flatten_hierarchy rebuilt -fsm_extraction one_hot "
        "-keep_equivalent_registers -resource_sharing off -no_lc -shreg_min_size 5",
        "opt": "opt_design -directive Explore",
        "place": "place_design -directive ExtraTimingOpt",
        "phys_opt": "phys_opt_design -directive AggressiveExplore",
        "route": "route_design -directive NoTimingRelaxation",
        "post_route_phys_opt": "phys_opt_design -directive AggressiveExplore",
    },
    "vitis": {
        "synth": "",
        "opt": "opt_design",
        "place": "place_design",
        "phys_opt": "phys_opt_design",
        "route": "route_design",
        "post_route_phys_opt": None,
    },
}


# runtime-oriented settings for the island flow (FINN_RWI_PROFILE=fast): QoR may drop, the
# result is still checked (routing complete, setup and hold met); for relaxed clocks
PROFILES["fast"] = {
    "synth": "-directive RuntimeOptimized",
    "opt": "opt_design -directive RuntimeOptimized",
    "place": "place_design -directive RuntimeOptimized",
    "phys_opt": "# (no phys_opt_design in the fast profile)",
    "route": "route_design -directive RuntimeOptimized",
    "post_route_phys_opt": None,
}


def profile(part):
    """Baseline flow of a part: Alveo parts (xcu*) are built with Vitis, the others with FINN's
    Zynq flow; FINN_RWI_PROFILE=fast: runtime-oriented directives everywhere (PROFILES["fast"])."""
    if os.environ.get("FINN_RWI_PROFILE") == "fast":
        return PROFILES["fast"]
    return PROFILES["vitis" if part.startswith("xcu") else "zynq"]


def synth_args(part):
    """Node synthesis options: the baseline's, unless FINN_RWI_SYNTH_DIRECTIVE overrides them
    (e.g. RuntimeOptimized: a MobileNet Thresholding_rtl with 1024 channels in LUT ROM
    synthesizes in 18 s instead of > 40 min, at ~2x its LUTs)."""
    env = os.environ.get("FINN_RWI_SYNTH_DIRECTIVE")
    return env if env is not None else profile(part)["synth"]
