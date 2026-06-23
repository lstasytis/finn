"""Baseline candidate get_tree_model for the Thresholding node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/thresholding.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    """Return tree model for analytical FIFO sizing."""
    reps = list(self.get_nodeattr("numInputVectors"))[0]

    NumChannels = self.get_nodeattr("NumChannels")
    PE = self.get_nodeattr("PE")
    ImgDim = np.prod(list(self.get_nodeattr("numInputVectors"))) // reps

    act = DataType[self.get_nodeattr("outputDataType")]
    IMPL_STYLE = "rtl" if "_rtl" in (self.__class__.__name__) else "hls"
    assert IMPL_STYLE in ["rtl", "hls"], "Implementation style must be 'rtl' or 'hls'"

    NF = NumChannels // PE
    total_iterations = ImgDim * NF

    if IMPL_STYLE == "hls":
        output_delay = 0  # 4 if 2023.1 vivado
    else:
        if act == DataType["BIPOLAR"]:
            output_delay = 0  # 4 if 2023.1 vivado
        else:
            output_delay = 0

    if total_iterations > output_delay:
        read = Characteristic_Node("read", [(output_delay, [1, 0])], True)

        read_write = Characteristic_Node(
            "Compute", [(total_iterations - output_delay, [1, 1])], True
        )

        write = Characteristic_Node("write", [(output_delay, [0, 1])], True)

        threshold_top = Characteristic_Node(
            "Thresholding Top", [(1, read), (1, read_write), (1, write)], False
        )

    else:
        read = Characteristic_Node("Rush-in", [(total_iterations, [1, 0])], True)
        idle = Characteristic_Node("Idle", [(output_delay - total_iterations, [0, 0])], True)

        write = Characteristic_Node("Compute", [(total_iterations, [0, 1])], True)

        threshold_top = Characteristic_Node(
            "Thresholding Top", [(1, read), (1, idle), (1, write)], False
        )

    return threshold_top  # top level phase of this node
