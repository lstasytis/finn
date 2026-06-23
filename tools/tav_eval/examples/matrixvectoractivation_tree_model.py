"""Baseline candidate get_tree_model for the MVAU node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/matrixvectoractivation.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    MW = self.get_nodeattr("MW")
    MH = self.get_nodeattr("MH")

    SIMD = self.get_nodeattr("SIMD")
    PE = self.get_nodeattr("PE")
    numVectors = np.prod(self.get_nodeattr("numInputVectors"))
    SF = int(MW / SIMD)
    NF = int(MH / PE)

    IMPL_STYLE = "rtl" if "_rtl" in (self.__class__.__name__) else "hls"
    assert IMPL_STYLE in ["rtl", "hls"], "Implementation style must be 'rtl' or 'hls'"

    # additional precision which is typically unnecessary for FIFO size modelling
    # if IMPL_STYLE == "hls":
    #     output_delay = 0  # cycles before output starts
    # writing when input is read. Typically 2
    #     wind_up = 0  # about 3 cycles of wind-up for HLS MVAU
    # else:
    #     # RTL implementation
    #     output_delay = 0
    wind_up = 0

    idle = Characteristic_Node("idle cycles", [(1, [0, 0])], True)
    read = Characteristic_Node("Read a burst of input", [(1, [1, 0])], True)
    write = Characteristic_Node("update output", [(1, [0, 1])], True)
    read_and_write = Characteristic_Node("update output", [(1, [1, 1])], True)

    write_PE = Characteristic_Node(
        "iterate MW/SIMD and update an output",
        [
            (SF - 1, idle),
            (1, write),
        ],
        False,
    )

    feature_map = Characteristic_Node(
        "Compute single feature map",
        [(wind_up, idle), (SF - 1, read), (0, idle), (1, read_and_write), (NF - 1, write_PE)],
        False,
    )

    all_feature_maps = Characteristic_Node(
        "compute set of feature maps", [(1, idle), (numVectors, feature_map)], False
    )

    return all_feature_maps
