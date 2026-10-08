"""Build (or reuse) the island flow's Vivado-only Alveo shell for the IODMA / AXI-Lite interfaces
of an accelerator model (accel.onnx of an earlier island run) at a given clock, untimed, so that
timed island runs find it cached.

    python build_vshell.py <accel.onnx> <clk_ns> <shell_lib> [--board U55C]
"""

import argparse
import json

from qonnx.core.modelwrapper import ModelWrapper

from finn.util.basic import vitis_part_map
from finn.util.dynarapid.graph import mm_ports
from finn.util.dynarapid.shell import build_shell

ap = argparse.ArgumentParser()
ap.add_argument("accel")
ap.add_argument("clk_ns", type=float)
ap.add_argument("shell_lib")
ap.add_argument("--board", default="U55C")
ap.add_argument("--jobs", type=int, default=8)
a = ap.parse_args()
d, r = build_shell(a.board, vitis_part_map[a.board], a.clk_ns, mm_ports(ModelWrapper(a.accel)), a.shell_lib, a.jobs)
print(json.dumps(r))
