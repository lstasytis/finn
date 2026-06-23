"""Baseline candidate get_tree_model for the VVAU node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/vectorvectoractivation.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    # key parameters
    IMPL_STYLE = "rtl" if "_rtl" in (self.__class__.__name__) else "hls"
    assert IMPL_STYLE in ["rtl", "hls"], "Implementation style must be 'rtl' or 'hls'"

    SIMD = self.get_nodeattr("SIMD")
    PE = self.get_nodeattr("PE")
    Channels = self.get_nodeattr("Channels")
    Kernel_2 = np.prod(self.get_nodeattr("Kernel"))
    NF = int(Channels / PE)
    numReps = np.prod(self.get_nodeattr("Dim"))
    dim_h, dim_w = self.get_nodeattr("Dim")

    if IMPL_STYLE == "rtl":
        SF = Kernel_2 // SIMD
    # wind_up = 5
    else:
        SF = Kernel_2 // SIMD
        # wind_up = 7

    # INNER = TOTAL_FOLD // SF

    # wind_up_stage = Characteristic_Node(
    #     "write only",
    #     [(wind_up, [1,0])],
    #     True)

    # the windup stage should also exist and delay the outputs
    # this requires the same pattern of limiting SF and is probably best done as a correction
    # after the feature map?
    # alternative is to construct a split of first, middle and last sf,
    # with the first having a longer read phase (sf+windup-1) and the last (sf-windup-1)

    write_out = Characteristic_Node("write out simd (1 for hls)", [(1, [1, 1])], True)

    compute_one_sf = Characteristic_Node("read one SF input", [(1, [1, 0])], True)

    compute_sf = Characteristic_Node(
        "process SF-1 inputs", [(SF - 1, compute_one_sf), (1, write_out)], False
    )

    compute_transaction = Characteristic_Node(
        "Compute VVAU one transaction",
        [
            (NF, compute_sf),
        ],
        False,
    )

    vvau_top = Characteristic_Node(
        "Compute VVAU input set", [(numReps, compute_transaction)], False
    )

    return vvau_top  # top level phase of this node
