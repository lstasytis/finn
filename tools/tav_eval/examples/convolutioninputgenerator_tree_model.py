"""Baseline candidate get_tree_model for the ConvolutionInputGenerator node.

This is the current in-tree get_tree_model for src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py, exported as a standalone
candidate for the tav_eval harness. Running it as-is performs an identity
replacement, so against a matching rtlsim cache the TAV delta vectors are all
zero -- it is the starting point an optimizer mutates.

The body may reference any symbol already imported by the target module (e.g.
Characteristic_Node); the harness only extracts this function via AST and
splices it back, it never imports this file.
"""


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
    #
    # print("simd: ", simd)
    # print("ifm y, x: ", ifm_dim_y, ifm_dim_x)
    # print("K: ", k_y, k_x)
    # print("stride: ", stride_y, stride_x)
    # print("dilation: ", dilation_y, dilation_x)
    # print("parallel_window: ", parallel_window)
    # print("dw: ", depthwise)
    # print("buffer depth: ", self.get_buffer_depth())
    # print("buffering threshold: ", buffering_threshold)

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

        # print("absorbing krn: ", absorbing_kernels)
        # print("absorved reads: ", absorbed_reads)
        # print("remaining hanging reads: ", (inner_line_buffer_reads) - absorbed_reads)
        # print("remaining old kernels: ", (kernels_in_line - 2) - absorbing_kernels)
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

    if parallel_window == 2:
        # parallel window path works reliably, but should
        # eventually be using paralle window 0's structure
        # however currently is still inaccurate for some
        # configs with parallel window=0
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
        swg = Characteristic_Node(
            "SlidingWindowGenerator",
            [
                (1, ch_idle),
                (reads_to_prepare_first_line, ch_read),
                (kernel_lines - 1, handle_line),
                (1, handle_last_line),
            ],
            False,
        )

    else:
        # --- handle_first_kernel ---
        # print("\n\nhandle first kernel")
        # print(f"do_both: {first_do_both}\n")
        # print(f"reads_only: {first_reads_only}\n")
        # print(f"writes_only: {first_writes_only}\n")

        handle_absolute_kernel = Characteristic_Node(
            "handle one kernel",
            [
                (absolute_first_do_both, ch_both),
                (absolute_first_reads_only, ch_read),
                (absolute_first_writes_only, ch_write),
            ],
            False,
        )

        # --- handle_first_kernel ---
        # print("\n\nhandle first kernel")
        # print(f"do_both: {first_do_both}\n")
        # print(f"reads_only: {first_reads_only}\n")
        # print(f"writes_only: {first_writes_only}\n")

        handle_first_kernel = Characteristic_Node(
            "handle one kernel",
            [
                (first_do_both, ch_both),
                (first_reads_only, ch_read),
                (first_writes_only, ch_write),
            ],
            False,
        )

        # --- handle_kernel ---
        # print("\n\nhandle kernel")
        # print(f"do_both: {do_both}\n")
        # print(f"reads_only: {reads_only}\n")
        # print(f"writes_only: {writes_only}\n")

        handle_kernel = Characteristic_Node(
            "handle one kernel",
            [
                (do_both, ch_both),
                (reads_only, ch_read),
                (writes_only, ch_write),
            ],
            False,
        )

        # --- handle_kernel_absorbed ---
        # print("\n\nhandle absorbed kernel")
        # print(f"do_both: {do_both+writes_only}\n")
        # print(f"reads_only: {reads_only}\n")
        #
        handle_kernel_absorbed = Characteristic_Node(
            "handle one kernel with fused writes",
            [
                (do_both + writes_only, ch_both),
                (reads_only, ch_read),
            ],
            False,
        )

        # --- handle_first_line ---
        # print("\n\nhandle first line")
        # print(f"first_line_buffer: {first_line_buffer}\n")
        # print(f"first line kernelbuffer: {first_line_kernel_buffer}\n")
        # print(f"kernels_in_line: {kernels_in_line}\n")
        # print(f"remainder_x: {remainder_x}\n")

        handle_first_line = Characteristic_Node(
            "write first line",
            [
                # (first_line_buffer, ch_read),
                (1, handle_absolute_kernel),
                (kernels_in_line - 1, handle_kernel),
                (remainder_x, ch_read),
            ],
            False,
        )

        # --- handle_line ---
        # print("\n\nhandle regular line")
        # print(f"inner_line_buffer_reads: {inner_line_buffer_reads}\n")
        # print(f"absorbing_kernels: {absorbing_kernels}\n")
        # print("kernels_in_line - absorbing_kernels: ")
        # print(f"{kernels_in_line - absorbing_kernels}\n")
        # print(f"remainder_x: {remainder_x}\n")

        handle_line = Characteristic_Node(
            "write one inner line",
            [
                # (remaining_buffer_reads, ch_read),
                (1, handle_first_kernel),
                (absorbing_kernels, handle_kernel_absorbed),
                (kernels_in_line - 1 - absorbing_kernels, handle_kernel),
                (remainder_x, ch_read),
            ],
            False,
        )

        # --- swg ---
        # print("\n\nswg")
        # print(f"kernel_lines - 1: {kernel_lines - 1}\n")
        # print(f"remainder_y: {remainder_y}\n")

        swg = Characteristic_Node(
            "SlidingWindowGenerator",
            [
                (1, handle_first_line),
                (kernel_lines - 1, handle_line),
                (remainder_y, ch_read),
            ],
            False,
        )

    return swg
