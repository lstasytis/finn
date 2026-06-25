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

    # ── non-divisible width ratio (gcd partial conversion) ──────────────────
    # When neither width divides the other there is no clean read-1/write-many
    # (or read-many/write-1) nesting, so model the conversion at gcd granularity:
    # the data moves in units of g = gcd(inWidth, outWidth), one granule per
    # cycle, reading a new input word every (inWidth/g) granules and writing a
    # new output word every (outWidth/g) granules. This is an APPROXIMATION of
    # the real timing, but it reproduces the exact total number of reads
    # (= number of input words) and writes (= number of output words) over the
    # total length needed for the conversion.
    if inWidth != outWidth and inWidth % outWidth != 0 and outWidth % inWidth != 0:
        numInWords = self.get_number_input_values()
        # gcd without relying on a math/np import being present in the node module
        a, b = inWidth, outWidth
        while b:
            a, b = b, a % b
        g = a
        in_u = inWidth // g
        out_u = outWidth // g
        period = in_u * out_u  # == lcm(in_u, out_u) since gcd(in_u, out_u) == 1
        total_granules = numInWords * in_u
        n_periods = total_granules // period
        remainder = total_granules - n_periods * period

        def _granule_leaf(n_cycles, name):
            # one cycle per granule; RLE-compress consecutive identical [r, w]
            sub = []
            for k in range(n_cycles):
                rw = [1 if k % in_u == 0 else 0, 1 if k % out_u == 0 else 0]
                if sub and sub[-1][1] == rw:
                    sub[-1] = (sub[-1][0] + 1, rw)
                else:
                    sub.append((1, rw))
            return Characteristic_Node(name, sub, True)

        phases = [(wind_up, idle)]
        if n_periods > 0:
            phases.append((n_periods, _granule_leaf(period, "gcd period")))
        if remainder > 0:
            phases.append((1, _granule_leaf(remainder, "gcd remainder")))
        return Characteristic_Node("DWC gcd partial conversion (approx)", phases, False)

    if inWidth > outWidth:
        numReps = self.get_number_input_values()
        # down-conversion (outWidth divides inWidth)
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
