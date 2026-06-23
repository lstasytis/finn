"""Baseline candidate get_tree_model for the DuplicateStreams node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/duplicatestreams.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    # key parameters

    dim = np.prod(self.get_folded_output_shape()[1:-1])

    read_write = Characteristic_Node("passing duplicate layer", [(dim, [1, 1])], True)
    duplicatestreams_top = Characteristic_Node("compute duplicate", [(1, read_write)], False)

    return duplicatestreams_top  # top level phase of this node
