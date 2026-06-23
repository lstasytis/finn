"""Baseline candidate get_tree_model for the Pool node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/pool.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


def get_tree_model(self):
    # extract node attr

    PE = self.get_nodeattr("PE")
    Channels = self.get_nodeattr("Channels")
    KernelSize = self.get_nodeattr("KernelSize")
    OutImgDims = self.get_nodeattr("OutImgDims")
    BatchSize = self.get_nodeattr("BatchSize")

    # Derived parameters
    NF = Channels // PE  # neuron folding
    func = self.get_nodeattr("Function")
    if func == "MaxPool":
        SF = KernelSize[1] ** 2  # spatial folding per pooling window
        if KernelSize[0] == 1 or KernelSize[1] == 1:
            if KernelSize[0] == 1:
                SF = KernelSize[1] ** 2
            else:
                SF = KernelSize[0] ** 2
            SF = np.prod(KernelSize)
        reps = BatchSize * np.prod(OutImgDims)  # number of pooling windows to process
    else:
        SF = np.prod(KernelSize)  # spatial folding per pooling window
        reps = BatchSize * np.prod(OutImgDims)  # number of pooling windows to process

    # One input read per SF iteration
    read_pooling_input = Characteristic_Node("Read Pool Input", [(1, [1, 0])], True)

    readwrite_pooling_input = Characteristic_Node("Read Write Pool Input", [(1, [1, 1])], True)

    # SF - 1 reads + 1 read that overlaps with write
    compute_pool_window = Characteristic_Node(
        "Compute Pool Window",
        [(SF - 1, read_pooling_input), (1, readwrite_pooling_input)],  # overlap with output
        False,
    )

    # For each NF tile per pooling window
    compute_all_tiles = Characteristic_Node(
        "Compute All Tiles", [(NF, compute_pool_window)], False
    )

    # For each image region (spatial + batch)
    pool_top = Characteristic_Node("Top Pool Loop", [(reps, compute_all_tiles)], False)

    return pool_top
