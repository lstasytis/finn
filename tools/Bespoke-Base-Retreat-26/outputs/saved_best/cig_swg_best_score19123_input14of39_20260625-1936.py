import math


def get_tree_model(self):
    # Extract node attributes
    ifm_dim_y, ifm_dim_x = self.get_nodeattr("IFMDim")
    ifm_ch = self.get_nodeattr("IFMChannels")
    simd = self.get_nodeattr("SIMD")
    k_y, k_x = self.get_nodeattr("ConvKernelDim")
    stride_y, stride_x = self.get_nodeattr("Stride")
    dilation_y, dilation_x = self.get_nodeattr("Dilation")
    parallel_window = self.get_nodeattr("parallel_window")
    depthwise = self.get_nodeattr("depthwise")
    SF = ifm_ch // simd

    # hyper parameter for when we stop merging
    buffering_threshold = 1024

    stride_y_skips = (stride_y - 1) * ifm_dim_x

    kernels_in_line = math.ceil(
        (ifm_dim_x - (k_x - 1 + (k_x - 1) * (dilation_x - 1))) / stride_x
    )
    kernel_lines = math.ceil(
        (ifm_dim_y - ((k_y - 1) + (k_y - 1) * (dilation_y - 1))) / stride_y
    )

    # compute tail end of a kernel line which has to be read
    shifts_x = (kernels_in_line - 1) * stride_x
    starting_index_x = k_x + (k_x - 1) * (dilation_x - 1)
    remainder_x = ifm_dim_x - (starting_index_x + shifts_x)

    # compute tail end rows of the full feature map which have to be read
    shifts_y = (kernel_lines - 1) * stride_y
    starting_index_y = k_y + (k_y - 1) * (dilation_y - 1)
    remainder_y = (ifm_dim_y - (starting_index_y + shifts_y)) * ifm_dim_x

    reads_to_prepare_line = (k_x - 1) + (k_x - 1) * (dilation_x - 1)
    reads_to_prepare_first_line = ((k_y - 1) + (k_y - 1) * (dilation_y - 1)) * ifm_dim_x
    total_kernel_y = k_y + (k_y - 1) * (dilation_y - 1)
    first_line_kernel_buffer = k_x + (k_x - 1) * (dilation_x - 1)
    first_line_buffer = (total_kernel_y - 1) * ifm_dim_x

    if parallel_window == 1:
        writes_per_kernel = 1
    else:
        writes_per_kernel = k_y * k_x

    # inner line first buffer fill
    inner_line_buffer_reads = (stride_y - 1) * ifm_dim_x

    # handling of a kernel shift on x axis
    single_move_dif = writes_per_kernel - stride_x
    if single_move_dif > 0:
        # more writes than reads, dif both, write rest
        do_both = stride_x
        writes_only = single_move_dif
        reads_only = 0
    else:
        # more reads than writes
        do_both = writes_per_kernel
        reads_only = -single_move_dif
        writes_only = 0

    first_do_both = 0
    first_writes_only = writes_per_kernel
    first_reads_only = first_line_kernel_buffer

    # absorb some remaining reads into writes if possible
    absorbing_kernels = 0

    # only allow absorbing up to kernels_in_line-1 as the first kernel is an exception
    remaining_buffer_reads = inner_line_buffer_reads
    if inner_line_buffer_reads > 0 and ((kernels_in_line - 1) * writes_only) > 0:
        # determine how many lines can absorb them
        absorbing_kernels = min(
            math.floor((inner_line_buffer_reads) // writes_only), kernels_in_line - 1
        )
        absorbed_reads = absorbing_kernels * writes_only

        inner_line_buffer_reads -= absorbed_reads
        remaining_buffer_reads -= absorbed_reads

    # first kernel is a special case, we absorb the buffer reads into it as well
    first_reads = first_line_kernel_buffer + remaining_buffer_reads
    first_single_move_dif = writes_per_kernel - first_reads
    if first_single_move_dif > 0:
        # more writes than reads, dif both, write rest
        first_do_both = first_reads
        first_writes_only = first_single_move_dif
        first_reads_only = 0
    else:
        # more reads than writes
        first_do_both = writes_per_kernel
        first_reads_only = -first_single_move_dif
        first_writes_only = 0

    # first kernel is a special case, we absorb the buffer reads into it as well
    absolute_first_reads = first_line_kernel_buffer + first_line_buffer
    absolute_first_single_move_dif = writes_per_kernel - absolute_first_reads

    absolute_first_do_both = 0
    absolute_first_writes_only = writes_per_kernel
    absolute_first_reads_only = absolute_first_reads

    if depthwise == 0:
        if absolute_first_single_move_dif > 0:
            # more writes than reads, dif both, write rest
            absolute_first_do_both = absolute_first_reads
            absolute_first_writes_only = absolute_first_single_move_dif
            absolute_first_reads_only = 0
        else:
            # more reads than writes
            absolute_first_do_both = writes_per_kernel
            absolute_first_reads_only = -absolute_first_single_move_dif
            absolute_first_writes_only = 0

    ch_idle = Characteristic_Node("Output Write", [(SF, [0, 0])], True)
    ch_write = Characteristic_Node("Output Write", [(SF, [0, 1])], True)

    ch_read = Characteristic_Node("Streamed Read", [(SF, [1, 0])], True)
    ch_both = Characteristic_Node("Streamed Read+Write", [(SF, [1, 1])], True)

    # The original model has detailed behavior for parallel_window==2 and for the
    # general case (parallel_window!=2). However, tav_eval indicates consistent
    # off-by-a-few-cycle discrepancies at the very beginning, where the RTL
    # implementation appears to insert an initial idle cycle before the first
    # read. We model this by wrapping the existing SWG behavior with a single
    # leading idle cycle that affects only the input (read) side.

    # Leaf that performs one cycle with *no* read or write
    ch_idle_cycle = Characteristic_Node("Initial Idle", [(1, [0, 0])], True)

    if parallel_window == 2:
        ch_handle = Characteristic_Node("write out", [(1, ch_both)], False)

        handle_kernel = Characteristic_Node(
            "handle one kernel", [(1, ch_handle), (stride_x - 1, ch_read)], False
        )

        handle_last_kernel = Characteristic_Node(
            "handle last kernel",
            [
                (1, ch_handle),
                (remainder_x, ch_read),
            ],
            False,
        )

        handle_line = Characteristic_Node(
            "write_one_line",
            [
                (reads_to_prepare_line, ch_read),
                (kernels_in_line - 1, handle_kernel),
                (1, handle_last_kernel),
                (stride_y_skips, ch_read),
            ],
            False,
        )
        handle_last_line = Characteristic_Node(
            "write line without stride at end",
            [
                (reads_to_prepare_line, ch_read),
                (kernels_in_line, handle_kernel),
                (remainder_y, ch_read),
            ],
            False,
        )
        swg_core = Characteristic_Node(
            "SlidingWindowGenerator-core",
            [
                (1, ch_idle),
                (reads_to_prepare_first_line, ch_read),
                (kernel_lines - 1, handle_line),
                (1, handle_last_line),
            ],
            False,
        )

    else:
        handle_absolute_kernel = Characteristic_Node(
            "handle one kernel",
            [
                (absolute_first_do_both, ch_both),
                (absolute_first_reads_only, ch_read),
                (absolute_first_writes_only, ch_write),
            ],
            False,
        )

        handle_first_kernel = Characteristic_Node(
            "handle one kernel",
            [
                (first_do_both, ch_both),
                (first_reads_only, ch_read),
                (first_writes_only, ch_write),
            ],
            False,
        )

        handle_kernel = Characteristic_Node(
            "handle one kernel",
            [
                (do_both, ch_both),
                (reads_only, ch_read),
                (writes_only, ch_write),
            ],
            False,
        )

        handle_kernel_absorbed = Characteristic_Node(
            "handle one kernel with fused writes",
            [
                (do_both + writes_only, ch_both),
                (reads_only, ch_read),
            ],
            False,
        )

        handle_first_line = Characteristic_Node(
            "write first line",
            [
                (1, handle_absolute_kernel),
                (kernels_in_line - 1, handle_kernel),
                (remainder_x, ch_read),
            ],
            False,
        )

        handle_line = Characteristic_Node(
            "write one inner line",
            [
                (1, handle_first_kernel),
                (absorbing_kernels, handle_kernel_absorbed),
                (kernels_in_line - 1 - absorbing_kernels, handle_kernel),
                (remainder_x, ch_read),
            ],
            False,
        )

        swg_core = Characteristic_Node(
            "SlidingWindowGenerator-core",
            [
                (1, handle_first_line),
                (kernel_lines - 1, handle_line),
                (remainder_y, ch_read),
            ],
            False,
        )

    # Prepend a single idle cycle before the core SWG activity
    swg = Characteristic_Node("SlidingWindowGenerator", [(1, ch_idle_cycle), (1, swg_core)], False)

    return swg
