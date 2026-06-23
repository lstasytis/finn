"""Baseline candidate get_tree_model for the LabelSelect node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/labelselect.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    # key parameters
    # this depends on the kernel type, hls or rtl etc

    # extract node attr
    num_in_words = self.get_nodeattr("Labels")
    PE = self.get_nodeattr("PE")
    # PE = 1
    K = self.get_nodeattr("K")

    NF = num_in_words // PE

    output_delay = int(np.log2(num_in_words)) + 1
    # output_delay = NF

    print("num_in_words,PE,K,NF,output_delay")
    print(num_in_words, PE, K, NF, output_delay)
    print(f"exp cycles: {self.get_exp_cycles()}")

    read_k = Characteristic_Node("read only", [(NF, [1, 0])], True)

    compute_k = Characteristic_Node("compute k", [(output_delay, [0, 0])], True)

    write_k = Characteristic_Node("write k", [(K, [0, 1])], True)

    labelselect_top = Characteristic_Node(
        "Fill feature map", [(1, read_k), (1, compute_k), (1, write_k)], False
    )

    return labelselect_top  # top level phase of this node
