# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_num_default_workers

from finn.util.basic import make_build_dir
from finn.util.dynarapid.flow import dynarapid_pnr


class DynaRapidPnR(Transformation):
    """Place and route the dataflow graph with DynaRapid instead of Vivado.

    Every node becomes a pre-implemented component (synthesized and placed and
    routed inside a pblock, in parallel, and cached in a library shared between
    builds); DynaRapid then places the components according to the graph,
    stitches them and routes the inter-component nets with RWRoute.

    All nodes must have gone through IP generation (HLSSynthIP / PrepareIP),
    FIFOs and DWCs must be inserted. The routed checkpoint is written to
    out_dir and recorded in the metadata property "dynarapid_routed_dcp".

    The top-level ports of the result are
      clk, rst (active-low, FINN's ap_rst_n),
      n<k>_din_<i>/n<k>_valid_in_<i>/n<k>_ready_out_<i> for graph inputs and
      n<k>_dout_<j>/n<k>_valid_out_<j>/n<k>_ready_in_<j> for graph outputs,
    where n<k> is the index of the node in the graph.
    """

    def __init__(
        self,
        fpgapart,
        clk_ns,
        out_dir=None,
        library_dir=None,
        workers=None,
        placer="greedy",
        num_shapes=1,
        target_util=0.8,
        check=True,
        bitstream=False,
        no_clock=False,
    ):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.out_dir = out_dir
        self.library_dir = library_dir
        self.workers = workers
        self.placer = placer
        self.num_shapes = num_shapes
        self.target_util = target_util
        self.check = check
        self.bitstream = bitstream
        self.no_clock = no_clock

    def apply(self, model):
        out_dir = self.out_dir or make_build_dir("dynarapid_pnr_")
        library_dir = self.library_dir or os.path.join(
            os.environ["FINN_BUILD_DIR"], "dynarapid_library", self.fpgapart, "lib"
        )
        workers = self.workers or get_num_default_workers() or os.cpu_count()
        res = dynarapid_pnr(
            model,
            out_dir,
            library_dir,
            self.fpgapart,
            self.clk_ns,
            workers=workers,
            placer=self.placer,
            num_shapes=self.num_shapes,
            target_util=self.target_util,
            check=self.check,
            bitstream=self.bitstream,
            no_clock=self.no_clock,
        )
        assert res["status"] == "routed", "DynaRapid P&R failed (%s), see %s" % (
            res["status"],
            out_dir,
        )
        model.set_metadata_prop("dynarapid_routed_dcp", res["routed_dcp"])
        model.set_metadata_prop("dynarapid_result", json.dumps(res))
        if res.get("bitstream"):
            model.set_metadata_prop("dynarapid_bitfile", res["bitstream"])
        return (model, False)
