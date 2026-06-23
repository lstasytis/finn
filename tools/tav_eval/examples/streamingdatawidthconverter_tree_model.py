"""Baseline candidate get_tree_model for the StreamingDataWidthConverter node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/streamingdatawidthconverter.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    inWidth = self.get_nodeattr("inWidth")
    outWidth = self.get_nodeattr("outWidth")

    wind_up = 0

    idle = Characteristic_Node("idle", [(1, [0, 0])], True)

    if inWidth > outWidth:
        numReps = self.get_number_input_values()
        # down-conversion
        if inWidth % outWidth != 0:
            return None  # no support for gcd partial conversion yet

        writes_per_read = inWidth // outWidth
        # read 1, write many, repeats for in-word count

        read_input = Characteristic_Node("read 1 word", [(1, [1, 1])], True)

        write_output = Characteristic_Node("write words", [(writes_per_read - 1, [0, 1])], True)

        down_convert_word = Characteristic_Node(
            "down convert all words in a single transaction",
            [(1, read_input), (1, write_output)],
            False,
        )

        dwc_top = Characteristic_Node(
            "compute a set of DWCs with down conversion",
            [(wind_up, idle), (numReps, down_convert_word)],
            False,
        )

    elif inWidth < outWidth:
        numReps = self.get_number_output_values()
        # up-conversion

        if outWidth % inWidth != 0:
            return None  # no support for gcd partial conversion yet

        reads_per_write = outWidth // inWidth
        # read 1, write many, repeats for in-word count

        read_input = Characteristic_Node(
            "read first N-1 words", [(reads_per_write - 1, [1, 0])], True
        )

        write_output = Characteristic_Node(
            "read Nth word and write output word", [(1, [1, 1])], True
        )

        up_convert_word = Characteristic_Node(
            "down convert all words in a single transaction",
            [(1, read_input), (1, write_output)],
            False,
        )

        dwc_top = Characteristic_Node(
            "compute a set of DWCs with up conversion",
            [(wind_up, idle), (numReps, up_convert_word)],
            False,
        )

    else:
        # pass-through
        numReps = self.get_number_input_values()

        pass_through = Characteristic_Node("pass-through", [(1, [1, 1])], True)

        dwc_top = Characteristic_Node(
            "DWC pass-through, no conversion", [(wind_up, idle), (numReps, pass_through)], False
        )

    return dwc_top
