"""Example candidate get_tree_model for the FMPadding node.

This file is the kind of artifact an AlphaEvolve-style optimizer would emit: a
single ``get_tree_model`` definition. The harness extracts this function and
splices it into ``src/finn/custom_op/fpgadataflow/fmpadding.py``, re-indenting
it to live as a method on the FMPadding class.

It may be written at module level (as here) or inside a class -- the harness
finds the first ``get_tree_model`` definition either way. The body may use any
symbol already imported by the target module (e.g. ``Characteristic_Node``).

This particular example reproduces the current in-tree behaviour, so it should
match the rtlsim reference exactly (all-zero delta vectors).
"""


def get_tree_model(self):
    # extract node attrs
    IMGDIM = self.get_nodeattr("ImgDim")
    PADDING = self.get_nodeattr("Padding")
    NUMCHANNELS = self.get_nodeattr("NumChannels")
    SIMD = self.get_nodeattr("SIMD")
    batch_size = self.get_nodeattr("numInputVectors")
    IMPL_STYLE = "rtl" if "_rtl" in (self.__class__.__name__) else "hls"
    assert IMPL_STYLE in ["rtl", "hls"], "Implementation style must be 'rtl' or 'hls'"

    # compute new parameters
    NF = int(NUMCHANNELS / SIMD)
    y_padding_top, x_padding_left, y_padding_bottom, x_padding_right = PADDING
    y_dim = IMGDIM[0]
    x_dim = IMGDIM[1]

    if IMPL_STYLE == "hls" and NF == 1:
        loop_overhead = 1
    else:
        loop_overhead = 0

    ch_pad = Characteristic_Node("Channel_Pad", [(NF, [0, 1]), (loop_overhead, [0, 0])], True)
    ch_pass = Characteristic_Node("Channel_Pass", [(NF, [1, 1]), (loop_overhead, [0, 0])], True)

    x_inner_line = Characteristic_Node(
        "Fill X full inner line",
        [(x_padding_left, ch_pad), (x_dim, ch_pass), (x_padding_right, ch_pad)],
        False,
    )
    x_outer_line = Characteristic_Node(
        "Pad X outer line", [(x_padding_left + x_dim + x_padding_right, ch_pad)], False
    )
    fmpadding = Characteristic_Node(
        "FMPadding FM",
        [
            (y_padding_top, x_outer_line),
            (y_dim, x_inner_line),
            (y_padding_bottom, x_outer_line),
        ],
        False,
    )
    fmpadding_top = Characteristic_Node(
        "FMPadding FM",
        [
            (batch_size, fmpadding),
        ],
        False,
    )

    return fmpadding_top
