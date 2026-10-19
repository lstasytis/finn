# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import functools

# Inspect information on Python objects like modules
import inspect
import itertools
import numpy as np
import scipy
import shutil
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor

# Import the elementwise binary operation module to extract names of all
# specializations (which require PE parallelism to be configured)
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import make_build_dir, part_map
from finn.util.fpgadataflow import is_fpgadataflow_node, is_hls_node, is_rtl_node
from finn.util.platforms import DEFAULT_RES_LIMITS, platforms


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


def allowed_divisors(cap, max_padding_count=0):
    """Folding factors for a dimension of size ``cap`` and the (padded) size each one folds,
    as two ascending tuples. With ``max_padding_count`` > 0 the dimension may be padded by up
    to that many elements, which adds the divisors of the padded sizes; each folding factor
    (size // value) is offered once, by the smallest size that has it."""
    values, bounds, factors = [], [], []
    for bound in range(cap, cap + max_padding_count + 1):
        for x in divisors(bound):
            if x not in values and x <= cap and bound // x not in factors:
                values.append(x)
                bounds.append(bound)
                factors.append(bound // x)
    return zip(*sorted(zip(values, bounds)))


class GridEncoder:
    """Encode ordered discrete folding values into a continuous index space so that
    scipy.dual_annealing can search over them, and decode a point back. A parameter with
    a single possible value is fixed and carries no encoded dimension."""

    _EPS = 1e-9

    def __init__(self, choices):
        self.choices = [tuple(c) for c in choices]

    @property
    def bounds(self):
        # half-step bounds keep boundary indices reachable and rounding-safe
        return [(-0.5 + self._EPS, len(c) - 0.5 - self._EPS) for c in self.choices if len(c) > 1]

    def encode(self, decoded):
        return [float(c.index(v)) for c, v in zip(self.choices, decoded) if len(c) > 1]

    def decode(self, encoded):
        it = iter(encoded)
        return [
            c[min(max(int(round(next(it))), 0), len(c) - 1)] if len(c) > 1 else c[0]
            for c in self.choices
        ]


# initializer input index and row-count attribute of the thresholds per op family
THRESHOLD_INPUTS = {"Thresholding": (1, "NumChannels"), "VVAU": (2, "Channels"), "MVAU": (2, "MH")}


@dataclass(eq=False)
class Parameter:
    """One node attribute behind a meta parameter value: ``attr`` of the custom op ``node``
    (at ``node_index`` in ``model``) takes ``value``, and the channel count ``bound_name``
    (MW, MH, NumChannels, IFMChannels, ...) the possibly padded ``bound_value``."""

    attr: str
    value: object
    node: object
    node_index: int
    op_type: str
    model: object
    bound_name: str = None
    bound_value: object = None
    update_weights: bool = False
    update_thresholds: bool = False
    reshapes_io: bool = True

    def apply(self, final=True):
        """Set the attribute; with ``final`` also make the tensors follow the (padded)
        channel counts, which the search itself does not need."""
        self.node.set_nodeattr(self.attr, self.value)
        if self.bound_name is not None:
            self.node.set_nodeattr(self.bound_name, self.bound_value)
        # an SWG emits the whole window only when its SIMD covers all channels
        if self.bound_name == "IFMChannels" and self.value < self.bound_value:
            self.node.set_nodeattr("parallel_window", 0)
        if not final:
            return
        onnx_node = self.model.graph.node[self.node_index]
        if self.reshapes_io:
            self.model.set_tensor_shape(onnx_node.input[0], self.node.get_normal_input_shape())
            self.model.set_tensor_shape(onnx_node.output[0], self.node.get_normal_output_shape())
        if self.update_thresholds:
            self._resize_thresholds(onnx_node)
        if self.update_weights:
            self._resize_weights(onnx_node)

    def _set_initializer(self, name, tensor):
        self.model.set_initializer(name, tensor)
        self.model.set_tensor_shape(name, tensor.shape)

    def _zeros(self, name, shape):
        tensor = gen_finn_dt_tensor(self.model.get_tensor_datatype(name), shape)
        tensor[...] = 0
        return tensor

    def _resize_thresholds(self, onnx_node):
        """Zero-pad (or crop) the threshold rows to the node's channel count."""
        in_idx, rows_attr = THRESHOLD_INPUTS[self.op_type.rsplit("_", 1)[0]]
        if len(onnx_node.input) <= in_idx:
            return  # an MVAU/VVAU without thresholds
        rows = self.node.get_nodeattr(rows_attr)
        name = onnx_node.input[in_idx]
        T = self.model.get_initializer(name)
        # a single row is broadcast over all channels and stays valid when padding
        if T.shape[0] in (1, rows):
            return
        T_new = self._zeros(name, (rows, T.shape[1]))
        T_new[: min(rows, T.shape[0]), :] = T[: min(rows, T.shape[0]), :]
        self._set_initializer(name, T_new)

    def _resize_weights(self, onnx_node):
        """Zero-pad (or crop) the weights to the node's (padded) MW x MH, or for a VVAU to
        its Channels x Kernel."""
        name = onnx_node.input[1]
        W = self.model.get_initializer(name)
        if self.op_type.startswith("MVAU"):
            mw, mh = self.node.get_nodeattr("MW"), self.node.get_nodeattr("MH")
            if (mw, mh) == W.shape:
                return
            # behind a sliding window generator the rows are (kernel position, channel)
            # with the channel fastest, so channel padding goes into every kernel position
            producer = self.model.find_producer(onnx_node.input[0])
            k = 1
            if producer is not None and producer.op_type.startswith("ConvolutionInputGenerator"):
                k = int(np.prod(getCustomOp(producer).get_nodeattr("ConvKernelDim")))
            if W.shape[0] % k != 0 or mw % k != 0:
                k = 1
            c_old, c_new = W.shape[0] // k, mw // k
            c, h = min(c_old, c_new), min(mh, W.shape[1])
            W_new = self._zeros(name, (k, c_new, mh))
            W_new[:, :c, :h] = W.reshape(k, c_old, W.shape[1])[:, :c, :h]
            W_new = W_new.reshape(mw, mh)
        else:
            channels, kernel = self.node.get_nodeattr("Channels"), self.node.get_nodeattr("Kernel")
            if W.shape[0] == channels and W.shape[-2:] == tuple(kernel):
                return
            W_new = self._zeros(name, (channels, W.shape[1], kernel[0], kernel[1]))
            kept = tuple(slice(min(n, m)) for n, m in zip(W_new.shape, W.shape))
            W_new[kept] = W[kept]
        self._set_initializer(name, W_new)


class MetaParameter:
    """One optimizable value that sets a group of tightly linked node attributes, e.g. the
    SIMD of an SWG together with the SIMD of the MVAU it feeds. ``possible_values`` (sorted)
    are what the search moves over; ``real_values[i]`` lists the ``Parameter``s value i sets.
    An MVAU's PE and SIMD are separate meta parameters: SIMD-PE 1-2 and 2-1 cost differently."""

    def __init__(self, name, values, real_values, node_index):
        self.name = name
        self.node_index = node_index
        self.meta_value = None
        pairs = sorted(zip(values, real_values), key=lambda pair: pair[0])
        self.possible_values = [value for value, _ in pairs]
        self.real_values = [real for _, real in pairs]
        # the nodes this parameter changes, for the cycle and resource estimates
        self.unique_nodes = list(dict.fromkeys(p.node for p in real_values[0]))

    def update_value(self, value):
        self.meta_value = value

    def apply_value(self, final=False, attrs=()):
        """Apply the current value's attributes named in ``attrs``."""
        for p in self.real_values[self.possible_values.index(self.meta_value)]:
            if p.attr in attrs:
                p.apply(final)


class ParameterSet:
    """The meta parameters of a model; ``index_list`` selects the ones being optimized
    (a partition) and ``nodes`` are the nodes they change."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.index_list = []
        self.nodes = []

    def get_max_cycles(self):
        return max([n.get_exp_cycles() for n in self.nodes])

    def get_meta_values(self):
        return [p.meta_value for p in self.parameters]

    def get_min_vals(self):
        return [p.possible_values[0] for p in self.parameters]

    def get_max_vals(self):
        return [p.possible_values[-1] for p in self.parameters]

    def add_all_params_to_index_list(self):
        self.index_list = list(range(len(self.parameters)))

    def set_values(self, values):
        for i, value in zip(self.index_list, values):
            self.parameters[i].update_value(value)

    def apply_updates(self, final=False, attrs=()):
        for i in self.index_list:
            self.parameters[i].apply_value(final, attrs)

    def assign_involved_nodes(self):
        self.nodes = list(set(n for i in self.index_list for n in self.parameters[i].unique_nodes))


@dataclass(eq=False)
class Optimizer:
    """The inner loop of the folding optimizer: extracts the model's folding parameters,
    splits them into partitions of a few neighbouring nodes and minimizes the cost model
    over each partition with scipy's dual annealing.

    Every hardware node is folded generically: it sweeps whichever of PE / SIMD the node
    declares, up to the output channels (PE) or input channels (SIMD). What is genuinely
    op-specific is in the small tables below and in the SWG pairing rules (``_pair_*``): a
    ConvolutionInputGenerator is folded together with the layer it feeds.
    """

    model: object
    name: str  # informational
    #: ``max_cycles`` (the hard constraint) and each resource's budget
    targets: dict
    fpgapart: str = "xc7z020clg400-1"
    #: channels a node may be padded by, and whether the IO nodes may be padded
    padding: int = 0
    pad_io_nodes: bool = False
    #: the annealer's evaluations per parameter, and its seed
    effort: int = 100
    seed: int = None
    mvau_wwidth_max: int = 1024
    enable_folding_dwc_heuristic: bool = True
    resource_weights: dict = field(default_factory=dict)
    allow_uram_weights: bool = False
    #: only apply the initial guess, no search
    init_run: bool = False
    #: the RESOURCE_TYPE_ATTRS that may be tuned
    resource_type_params: list = field(default_factory=list)
    params: ParameterSet = None

    # a VVAU folds SIMD over its kernel window, not its input channels
    SIMD_MAX_OVERRIDE = {
        "VVAU_hls": lambda inst: int(np.prod(inst.get_nodeattr("Kernel"))),
        "VVAU_rtl": lambda inst: int(np.prod(inst.get_nodeattr("Kernel"))),
    }
    # parameters kept unfolded (fixed at 1)
    DO_NOT_FOLD = {
        # folding LabelSelect ruins fmax (OOC place & route, xczu7ev: 10 labels 610 MHz at
        # PE=1, 390 at PE=2, 230 at PE=5; 1000 labels 530 MHz at PE=1, 13 at PE=100)
        ("LabelSelect_hls", "PE"),
        # a VVAU_hls only supports SIMD (kernel-window) folding in its RTL sibling
        ("VVAU_hls", "SIMD"),
        # concat/split fold SIMD over per-stream channel counts, which only the naive
        # folder models
        ("StreamingConcat_hls", "SIMD"),
        ("StreamingSplit_hls", "SIMD"),
    }
    # (op_type, param) -> the attribute holding the channel count that may be padded to
    # unlock finer folding factors; other nodes are folded but never padded (VVAU/SWG SIMD
    # track a kernel size, which must not be padded)
    PADDING_BOUND_ATTR = {
        (op_type, param): attr
        for op_types, param, attr in (
            ("MVAU_hls MVAU_rtl", "SIMD", "MW"),
            ("MVAU_hls MVAU_rtl", "PE", "MH"),
            ("VVAU_hls VVAU_rtl", "PE", "Channels"),
            ("Thresholding_hls Thresholding_rtl AddStreams_hls", "PE", "NumChannels"),
            ("ChannelwiseOp_hls DuplicateStreams_hls", "PE", "NumChannels"),
            ("StreamingMaxPool_hls StreamingMaxPool_rtl", "PE", "NumChannels"),
            ("FMPadding_hls FMPadding_rtl FMPadding_Pixel_hls", "SIMD", "NumChannels"),
            ("DownSampler_hls", "SIMD", "NumChannels"),
        )
        for op_type in op_types.split()
    }
    # ops whose weight / threshold initializers follow their (padded) channel counts
    OPS_WITH_WEIGHTS = {"MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl"}
    OPS_WITH_THRESHOLDS = OPS_WITH_WEIGHTS | {"Thresholding_hls", "Thresholding_rtl"}
    SWG_OPS = {"ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl"}
    # resource-type attributes tuned to trade BRAM<->URAM and LUT<->DSP
    RESOURCE_TYPE_ATTRS = ["ram_style", "resType", "ram_style_thresholds"]
    # values the estimator does not cost: "distributed" (LUTRAM) is estimated as free
    RESOURCE_TYPE_EXCLUDED_VALUES = {
        "ram_style": {"distributed"},
        "ram_style_thresholds": {"distributed"},
    }
    # rough LUT cost of an HLS DWC relative to an RTL one (DWC-avoidance heuristic)
    HLS_DWC_COST_PENALTY = 8
    # depth of the FIFOs on the graph's input and output (DeriveFIFOSizes)
    IO_FIFO_DEPTH = 5
    # attributes a cost-model evaluation applies
    PARAMETERS_TO_APPLY = ["SIMD", "PE", "parallel_window", *RESOURCE_TYPE_ATTRS]
    # dual_annealing settings: iteration cap (maxfun = effort * parameters usually binds
    # first), acceptance (-1e4, -5] and visiting (1, 3] distribution parameters
    MAXITER, ACCEPT, VISIT = 200, -0.5, 2.0
    # large FINITE cost of a missed cycle target, so that the annealer can climb out of
    # an infeasible region instead of aborting on inf
    INFEASIBLE_PENALTY = 1e6

    @staticmethod
    def max_folding_factor(node_inst, op_type, param):
        """Largest useful PE (output channels) or SIMD (input channels) of a node."""
        if param == "PE":
            return int(node_inst.get_normal_output_shape()[-1])
        if op_type in Optimizer.SIMD_MAX_OVERRIDE:
            return Optimizer.SIMD_MAX_OVERRIDE[op_type](node_inst)
        return int(node_inst.get_normal_input_shape()[-1])

    # ------------------------------------------------------------------ cost model

    def cost_model(self, values):
        """Cost of the current partition at meta ``values`` (lower is better).

        The sum of the partition's resources, each normalized by its budget and scaled by
        its weight (``prefer_memory`` / ``prefer_compute`` raise the weight of the resource
        to avoid), plus the DWC-avoidance term and the estimated FIFOs. A missed cycle
        target (the slowest node or DWC in front of one) adds INFEASIBLE_PENALTY times the
        violation ratio, so that any feasible point beats any infeasible one."""
        self.params.set_values(values)
        self.params.apply_updates(final=False, attrs=self.PARAMETERS_TO_APPLY)
        nodes = self.params.nodes
        cycles = max(
            self.params.get_max_cycles(), max(self.dwc_cycles(self.model, n) for n in nodes)
        )
        cost = self.resource_cost(self.get_resources(nodes))
        if self.enable_folding_dwc_heuristic:
            cost += self.compute_hls_dwc_cost(self.model, nodes, self.targets["LUT"])
        cost += self.resource_cost(self.fifo_resources(self.model, nodes))
        violation = cycles / self.targets["max_cycles"]
        if violation > 1:
            cost += self.INFEASIBLE_PENALTY * violation
        return cost

    def resource_cost(self, resources):
        """Resources normalized by their budgets and weighted, as in cost_model."""
        return sum(
            self.resource_weights.get(r, 1.0) * resources.get(r, 0.0) / self.targets[r]
            for r in self.targets
            if r != "max_cycles"
        )

    def get_resources(self, nodes):
        return aggregate_dict_keys({n: n.node_res_estimation(self.fpgapart) for n in nodes})

    def graph_nodes(self):
        """Every hardware node of the model as a custom op instance."""
        return [getCustomOp(n) for n in self.model.graph.node if is_fpgadataflow_node(n)]

    def compute_hls_dwc_cost(self, model, nodes, lut_capacity):
        """DWC-avoidance heuristic: for each node whose input stream shape differs from its
        producer's, the stream widths around the DWC InsertDWC will place there, relative to
        the LUT budget, and HLS_DWC_COST_PENALTY times that for a DWC the RTL one cannot do
        (non-integer width ratio or padding/cropping)."""
        cost = 0
        for node in nodes:
            prod = model.find_producer(node.onnx_node.input[0])
            if prod is None:
                continue
            prod_inst = getCustomOp(prod)
            # the producer's input width stands in for the DWC's output width here
            in_width, out_width = prod_inst.get_outstream_width(), prod_inst.get_instream_width()
            n0_out = prod_inst.get_folded_output_shape()
            op_type = node.onnx_node.op_type
            if (
                op_type.startswith("MVAU") and node.get_nodeattr("mem_mode") == "external"
            ) or op_type.startswith("StreamingConcat"):
                in_idx = [
                    i for i, name in enumerate(node.onnx_node.input) if name == prod.output[0]
                ]
                assert in_idx, "Malformed model"
                n1_in = node.get_folded_input_shape(in_idx[-1])
            else:
                n1_in = node.get_folded_input_shape()
            if np.prod(n0_out) == np.prod(n1_in) and n0_out[-1] == n1_in[-1]:
                continue
            hls = (max(in_width, out_width) % min(in_width, out_width) != 0) or (
                np.prod(n0_out) != np.prod(n1_in)
            )
            penalty = self.HLS_DWC_COST_PENALTY if hls else 1
            cost += ((in_width + out_width) * penalty) / lut_capacity
        return cost

    @staticmethod
    def dwc_cycles(model, node):
        """Cycles of the DWCs InsertDWC will put in front of ``node`` (0 if the stream
        shapes match), as StreamingDataWidthConverter estimates them: the busier side for
        an integer width ratio, both sides in turn for a general conversion or for
        padding/cropping. The latter can exceed either neighbour's cycles."""
        worst = 0
        for in_idx, name in enumerate(node.onnx_node.input):
            prod = model.find_producer(name)
            if (
                prod is None
                or not is_fpgadataflow_node(prod)
                or prod.op_type.startswith("StreamingDataWidthConverter")
                or prod.op_type == "IODMA_hls"
            ):
                continue
            out_idx = list(prod.output).index(name)
            prod_inst = getCustomOp(prod)
            n0_out = prod_inst.get_folded_output_shape(out_idx)
            n1_in = node.get_folded_input_shape(in_idx)
            if n0_out[-1] == n1_in[-1] and np.prod(n0_out) == np.prod(n1_in):
                continue
            inw, outw = prod_inst.get_outstream_width(out_idx), node.get_instream_width(in_idx)
            in_words, out_words = int(np.prod(n0_out[:-1])), int(np.prod(n1_in[:-1]))
            pads_or_crops = np.prod(prod_inst.get_normal_output_shape(out_idx)) != np.prod(
                node.get_normal_input_shape(in_idx)
            )
            if max(inw, outw) % min(inw, outw) == 0 and not pads_or_crops:
                worst = max(worst, max(in_words, out_words))
            else:
                worst = max(worst, in_words + out_words)
        return worst

    @staticmethod
    def fifo_model():
        """fifo.sv's cost model (what StreamingFIFO estimates with), or None."""
        try:
            from finn.util.resource_models import _fifo_cost, _resolve  # noqa: PLC0415
        except ImportError:
            return None
        return _fifo_cost, _resolve

    def fifo_resources(self, model, nodes):
        """Estimated resources of the FIFOs the build will put in front of ``nodes`` (one
        per edge, two around a DWC), without sizing them.

        Mirrors the analytic sizer to first order: a FIFO holds what its producer writes
        before the consumer reads it. With the upstream chain finishing a frame in ``t_up``
        cycles (its slowest node: a node paced by a slower one runs no faster) and the
        consumer taking ``t_c``, the producer runs ahead by ``words * (1 - t_up / t_c)``,
        and the sizer never charges more than the consumer's demand above the frame rate,
        ``words * (1 - t_c / T)``, plus one burst of its reads. Every FIFO is at least 2
        deep; the graph's input and output FIFOs are IO_FIFO_DEPTH deep."""
        fifo_model = self.fifo_model()
        totals = {"LUT": 0.0, "BRAM_18K": 0.0, "URAM": 0.0}
        target = max(float(self.targets.get("max_cycles", 1)), 1.0)

        chain = {}

        def upstream_cycles(onnx_node):
            if onnx_node.name not in chain:
                cycles = getCustomOp(onnx_node).get_exp_cycles()
                for name in onnx_node.input:
                    prod = model.find_producer(name)
                    if prod is not None and is_fpgadataflow_node(prod):
                        cycles = max(cycles, upstream_cycles(prod))
                chain[onnx_node.name] = cycles
            return chain[onnx_node.name]

        def add(depth, width):
            depth = max(2, int(depth))
            if fifo_model is None:
                # a shift register: its width in LUTs per 32 words
                totals["LUT"] += width * ((depth + 30) // 32) + 8
                return
            fifo_cost, resolve = fifo_model
            cost = fifo_cost(depth, width, resolve(depth, width, "auto"))
            totals["LUT"] += cost.lut
            totals["BRAM_18K"] += cost.bram
            totals["URAM"] += cost.uram

        graph_inputs = {x.name for x in model.graph.input}
        graph_outputs = {x.name for x in model.graph.output}
        for node in nodes:
            t_c = max(node.get_exp_cycles(), 1)
            for out_idx, name in enumerate(node.onnx_node.output):
                if name in graph_outputs:
                    add(self.IO_FIFO_DEPTH, node.get_outstream_width(out_idx))
            for in_idx, name in enumerate(node.onnx_node.input):
                prod = model.find_producer(name)
                if prod is None or not is_fpgadataflow_node(prod):
                    if name in graph_inputs:
                        add(self.IO_FIFO_DEPTH, node.get_instream_width(in_idx))
                    continue
                t_up = upstream_cycles(prod)
                n1_in = node.get_folded_input_shape(in_idx)
                words = int(np.prod(n1_in[:-1]))
                ahead = words * max(0.0, 1.0 - t_up / t_c)
                drain = words * max(0.0, 1.0 - t_c / target) + self.read_burst(node, in_idx)
                add(min(ahead, drain), node.get_instream_width(in_idx))
                prod_inst = getCustomOp(prod)
                out_idx = list(prod.output).index(name)
                n0_out = prod_inst.get_folded_output_shape(out_idx)
                if n0_out[-1] != n1_in[-1] or np.prod(n0_out) != np.prod(n1_in):
                    # a DWC in between: the FIFO on its input side as well
                    add(2, prod_inst.get_outstream_width(out_idx))
        return totals

    @staticmethod
    def read_burst(node, in_idx):
        """Words a node reads at once before pausing: an MVAU/VVAU takes its input
        window's words and then computes MH/PE outputs from them."""
        if node.onnx_node.op_type.startswith(("MVAU", "VVAU")):
            return int(node.get_folded_input_shape(in_idx)[-2])
        return 1

    # ------------------------------------------------------------------ search

    def execute_minimizer(self, discrete_args, init_guess):
        """Minimize the cost model over one partition: ``discrete_args`` are the ordered
        possible values per parameter, encoded to a continuous index space for
        scipy.dual_annealing, which starts at ``init_guess`` (kept unless it finds better).
        Evaluations are cached per decoded point."""
        encoder = GridEncoder(discrete_args)
        if not encoder.bounds:
            return list(init_guess)
        cache = {}

        def objective(encoded):
            if any(np.isnan(v) for v in encoded):
                return float("nan")
            decoded = tuple(encoder.decode(encoded))
            if decoded not in cache:
                cache[decoded] = self.cost_model(decoded)
            return cache[decoded]

        result = scipy.optimize.dual_annealing(
            func=objective,
            x0=encoder.encode(init_guess),
            maxiter=self.MAXITER,
            accept=self.ACCEPT,
            visit=self.VISIT,
            maxfun=self.effort * len(init_guess),
            seed=self.seed,
            bounds=encoder.bounds,
        )
        return encoder.decode(result.x)

    def _partitions(self, target_parameters, max_nodes):
        """Indices of the ``target_parameters``, split into runs of parameters belonging to
        at most ``max_nodes`` consecutive nodes. (Parameters are only collected once the
        first partition has started, which with ``max_nodes=1`` needs one on node 0.)"""
        partitions, current, nodes, last_node = [], [], 1, 0
        for i, p in enumerate(self.params.parameters):
            if p.name not in target_parameters:
                continue
            if p.node_index != last_node:
                nodes += 1
            if nodes > max_nodes and current:
                partitions.append(current)
                current, nodes = [], 1
            if nodes <= max_nodes:
                current.append(i)
            last_node = p.node_index
        if current:
            partitions.append(current)
        return partitions

    def optimize(
        self, initial_guess="max", max_nodes_in_partition=2, target_parameters=("SIMD", "PE")
    ):
        """One optimization pass over the model for the ``target_parameters``.

        ``initial_guess`` is "min" (least folding), "max" (most folding), "cheapest" (per
        node the cheapest folding meeting the cycle target, see cheapest_guess) or one meta
        value per parameter (e.g. an earlier solution). Only this pass's parameters take it;
        the others keep what an earlier pass chose. The start should be feasible, otherwise
        the annealer may give up believing there is no solution. Each partition of
        ``max_nodes_in_partition`` nodes is then minimized and its values applied for good
        (tensors included)."""
        params = self.params.parameters
        self.params.add_all_params_to_index_list()
        if initial_guess == "min":
            guess = self.params.get_min_vals()
        elif initial_guess == "max":
            guess = self.params.get_max_vals()
        elif initial_guess == "cheapest":
            guess = self.cheapest_guess()
        else:
            guess = [
                v if v in p.possible_values else self._nearest_value(p, v)
                for p, v in zip(params, initial_guess)
            ]
        for p, value in zip(params, guess):
            if p.name in target_parameters or p.meta_value is None:
                p.update_value(value)
        self.params.apply_updates(attrs=target_parameters)
        self.params.assign_involved_nodes()

        for partition in self._partitions(target_parameters, max_nodes_in_partition):
            self.params.index_list = partition
            self.params.assign_involved_nodes()
            start = [guess[i] for i in partition]
            if self.init_run:
                values = start
            else:
                choices = [tuple(params[i].possible_values) for i in partition]
                values = self.execute_minimizer(choices, start)
            self.params.set_values(values)
            self.params.apply_updates(final=True, attrs=target_parameters)

    @staticmethod
    def _nearest_value(param, value):
        if isinstance(value, str):
            return param.possible_values[0]
        if isinstance(value, tuple):
            return min(
                param.possible_values, key=lambda x: sum(abs(a - b) for a, b in zip(x, value))
            )
        return min(param.possible_values, key=lambda x: abs(x - value))

    def cheapest_guess(self):
        """Per node, the cheapest combination of its PE/SIMD values that meets the cycle
        target (among the fastest if none does), by enumerating them -- a few hundred at
        most. The annealer keeps its start unless it finds better, so a search never ends
        worse than this node-by-node optimum. The other parameters start at "max"."""
        params = self.params.parameters
        guess = self.params.get_max_vals()
        target = self.targets["max_cycles"]
        groups = {}
        for i, p in enumerate(params):
            if p.name in ("SIMD", "PE"):
                groups.setdefault(p.node_index, []).append(i)
        for indices in groups.values():
            nodes = list(dict.fromkeys(n for i in indices for n in params[i].unique_nodes))

            def set_combo(combo):
                for i, value in zip(indices, combo):
                    params[i].update_value(value)
                    params[i].apply_value(attrs=self.PARAMETERS_TO_APPLY)

            combos = []
            for combo in itertools.product(*(params[i].possible_values for i in indices)):
                set_combo(combo)
                combos.append((max(n.get_exp_cycles() for n in nodes), combo))
            fastest = min(cycles for cycles, _ in combos)
            feasible = [c for c in combos if c[0] <= target] or [
                c for c in combos if c[0] == fastest
            ]
            estimates = {}

            def estimate(n):  # a node's estimate depends on its own attributes only
                key = (id(n), str(n.onnx_node.attribute))
                if key not in estimates:
                    estimates[key] = self.resource_cost(n.node_res_estimation(self.fpgapart))
                return estimates[key]

            best = None
            for _, combo in feasible:
                set_combo(combo)
                cost = sum(estimate(n) for n in nodes)
                if best is None or cost < best[0]:
                    best = (cost, combo)
            for i, value in zip(indices, best[1]):
                guess[i] = value
        return guess

    # ------------------------------------------------------------------ parameters

    def generate_parameter_set(self):
        """Extract every optimizable folding parameter of the model: per hardware node one
        meta parameter per declared PE / SIMD, plus the resource-type attributes in
        ``resource_type_params``; an SWG and the node it feeds come from a pairing rule."""
        parameters = []
        skip_consumer = False
        for node_index, node in enumerate(self.model.graph.node):
            if skip_consumer:
                skip_consumer = False
                continue
            if node.op_type == "StreamingDataWidthConverter":
                continue
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            max_padding = self._max_padding_for_node(node_index)
            if node.op_type in self.SWG_OPS:
                parameters += self._extract_swg_pair(node, node_index, max_padding)
                skip_consumer = True
            else:
                parameters += self._extract_node(node, node_index, max_padding)
        self.params = ParameterSet(parameters)

    @staticmethod
    def _xnor_mode(inst):
        """In XNOR mode a zero bit means -1, so a zero-padded input or weight adds to the
        popcount: such an MVAU's input width (MW) must not be padded."""
        attrs = inst.get_nodeattr_types()
        return "binaryXnorMode" in attrs and inst.get_nodeattr("binaryXnorMode") == 1

    def _max_padding_for_node(self, node_index):
        """Padding budget of a node: none for the model's IO nodes unless allowed."""
        if not self.pad_io_nodes and node_index in (0, len(self.model.graph.node) - 1):
            return 0
        return self.padding

    def _param(self, inst, idx, attr, value, bound=(None, None), tensors=False, **kwargs):
        """A Parameter of ``inst`` (node ``idx``); ``bound`` is the (attribute, value) of its
        channel count, ``tensors`` makes its weights and thresholds follow that count."""
        op = inst.onnx_node.op_type
        return Parameter(attr, value, inst, idx, op, self.model, *bound, tensors, tensors, **kwargs)

    def _extract_node(self, node, node_index, max_padding):
        inst = getCustomOp(node)
        declared = inst.get_nodeattr_types()
        metas = []
        for param in ("SIMD", "PE"):
            if param not in declared or (node.op_type, param) in self.DO_NOT_FOLD:
                continue
            try:
                metas.append(self._make_folding_meta(inst, node_index, param, max_padding))
            except Exception as e:
                # leave op types the generic extraction cannot size at their folding
                warnings.warn(
                    f"SetFolding: could not extract {param} for {node.op_type} "
                    f"(node {node_index}), leaving it unfolded: {e}"
                )
        for param in self.resource_type_params:
            if param in declared:
                meta = self._make_resource_type_meta(inst, node_index, param)
                if meta is not None:
                    metas.append(meta)
        return metas

    def _make_folding_meta(self, inst, node_index, param, max_padding):
        """The meta parameter sweeping ``param`` (PE or SIMD) of a node over its folding
        factors, each offered once, padding the channel count where that is allowed."""
        op_type = inst.onnx_node.op_type
        max_value = self.max_folding_factor(inst, op_type, param)
        bound_attr = self.PADDING_BOUND_ATTR.get((op_type, param))
        padding = max_padding if bound_attr is not None else 0
        if bound_attr == "MW" and self._xnor_mode(inst):
            padding = 0
        values, real_values, seen_factors = [], [], set()
        for value, bound in zip(*allowed_divisors(max_value, padding)):
            if param == "SIMD" and op_type in self.SIMD_MAX_OVERRIDE:
                bound = inst.get_nodeattr("Kernel")  # the (unpaddable) kernel window
            factor = int(np.prod(bound)) // value
            if factor in seen_factors:
                continue
            seen_factors.add(factor)
            values.append(value)
            p = self._param(inst, node_index, param, value, (bound_attr, bound))
            p.update_weights = op_type in self.OPS_WITH_WEIGHTS
            p.update_thresholds = op_type in self.OPS_WITH_THRESHOLDS
            real_values.append([p])
        return MetaParameter(param, values, real_values, node_index)

    def _make_resource_type_meta(self, inst, node_index, param):
        """The meta parameter choosing a node's ram_style / resType, or None when fewer
        than two values remain or the estimator does not tell them apart."""
        excluded = self.RESOURCE_TYPE_EXCLUDED_VALUES.get(param, set())
        # URAM weight memories need runtime_writeable_weights=1 on Ultrascale (asserted in
        # MVAU/VVAU generate_infra_hdl): only with that set or allowed explicitly
        if param == "ram_style" and "runtime_writeable_weights" in inst.get_nodeattr_types():
            if inst.get_nodeattr("runtime_writeable_weights") != 1 and not self.allow_uram_weights:
                excluded = excluded | {"ultra"}
        allowed = sorted(set(inst.get_nodeattr_types()[param][3]) - excluded)
        if len(allowed) < 2 or not self._estimator_distinguishes(inst, param, allowed):
            return None
        real_values = [
            [self._param(inst, node_index, param, choice, reshapes_io=False)] for choice in allowed
        ]
        return MetaParameter(param, allowed, real_values, node_index)

    def _estimator_distinguishes(self, inst, param, allowed):
        """Whether two allowed values of ``param`` give different resource estimates."""
        original = inst.get_nodeattr(param)
        estimates = set()
        try:
            for choice in allowed:
                inst.set_nodeattr(param, choice)
                estimates.add(tuple(sorted(inst.node_res_estimation(self.fpgapart).items())))
        except Exception:
            return False
        finally:
            inst.set_nodeattr(param, original)
        return len(estimates) > 1

    # ------------------------------------------------------------------ SWG pairing rules
    # A ConvolutionInputGenerator (SWG) is folded together with the layer it feeds, which
    # always follows it in the graph. Each rule returns the pair's meta parameters, the
    # SWG's parameters first.

    def _extract_swg_pair(self, swg_node, node_index, max_padding):
        consumer = self.model.graph.node[node_index + 1]
        swg, consumer_inst = getCustomOp(swg_node), getCustomOp(consumer)
        if self._xnor_mode(consumer_inst):
            max_padding = 0  # the SWG's channel count is the XNOR MVAU's input width
        rules = {
            f"{op}_{backend}": rule
            for op, rule in (
                ("Pool", self._pair_swg_with_pool),
                ("MVAU", self._pair_swg_with_mvau),
                ("VVAU", self._pair_swg_with_vvau),
            )
            for backend in ("hls", "rtl")
        }
        rule = rules.get(consumer.op_type)
        if rule is None:
            raise AssertionError(
                "ConvolutionInputGenerator feeds unsupported op "
                f"{consumer.op_type}; expected Pool/MVAU/VVAU"
            )
        return rule(swg, consumer_inst, node_index, max_padding)

    def _swg_params(self, swg, node_index, simd, ifm_channels, parallel_window=None):
        """The SWG side of a pair: its SIMD over the (padded) channel count and, for the
        MVAU/VVAU rules, whether it emits the whole kernel window at once."""
        params = [self._param(swg, node_index, "SIMD", simd, ("IFMChannels", ifm_channels))]
        if parallel_window is not None:
            params.append(self._param(swg, node_index, "parallel_window", parallel_window))
        return params

    def _pair_swg_with_pool(self, swg, pool, node_index, max_padding):
        """SWG -> Pool (max pooling): SWG SIMD = Pool PE over the same channel count."""
        assert swg.get_nodeattr("depthwise") == 1
        simd_values, ifm_values = allowed_divisors(swg.get_nodeattr("IFMChannels"), max_padding)
        real_values = [
            self._swg_params(swg, node_index, simd, ifm)
            + [self._param(pool, node_index + 1, "PE", simd, ("Channels", ifm))]
            for simd, ifm in zip(simd_values, ifm_values)
        ]
        return [MetaParameter("SIMD", list(simd_values), real_values, node_index)]

    def _pair_swg_with_mvau(self, swg, mvau, node_index, max_padding):
        """SWG -> MVAU (dense convolution): one meta parameter (MVAU SIMD, SWG SIMD,
        parallel_window) over MW = kernel * IFMChannels, and the MVAU PE over MH.

        The SWG is not tied to the MVAU's SIMD: with a (cheap) DWC in between it runs at its
        own, slower pace, which keeps the FIFO in front of the MVAU shallow (an SWG matched
        to the MVAU's SIMD is MH/PE times faster and runs that far ahead). It emits the
        whole window (parallel_window) only when the MVAU SIMD exceeds the channel count.
        Channel counts go in ascending order and values are de-duplicated by their folding
        factors, so the padded space is a superset of the unpadded one."""
        ifm_real = swg.get_nodeattr("IFMChannels")
        _, ifm_values = allowed_divisors(ifm_real, max_padding)
        kernel_size = int(np.prod(swg.get_nodeattr("ConvKernelDim")))
        weight_bits = mvau.get_input_datatype(1).bitwidth()
        values, real_values, seen_factors = [], [], []
        for ifm in sorted(set(ifm_values)):
            mw = kernel_size * ifm
            for mvau_simd in divisors(mw):
                if not (weight_bits * mvau_simd < self.mvau_wwidth_max and mvau_simd > mw / 1024):
                    continue
                # a padding DWC pads only within a word: a padded channel count must be the
                # smallest multiple of the SWG's SIMD above the real one
                swg_choices = [
                    (d, 0) for d in divisors(ifm) if d <= mvau_simd and -(-ifm_real // d) * d == ifm
                ]
                if mvau_simd > ifm:
                    swg_choices.append((ifm, 1))
                for swg_simd, parallel_window in swg_choices:
                    value = (mvau_simd, swg_simd, parallel_window)
                    factors = (mw // mvau_simd, ifm // swg_simd, parallel_window)
                    if value in values or factors in seen_factors:
                        continue
                    values.append(value)
                    seen_factors.append(factors)
                    simd = self._param(mvau, node_index + 1, "SIMD", mvau_simd, ("MW", mw), True)
                    swg_params = self._swg_params(swg, node_index, swg_simd, ifm, parallel_window)
                    real_values.append(swg_params + [simd])
        # a conv MVAU's MH is never padded
        mh = mvau.get_nodeattr("MH")
        pe_values = list(divisors(mh))
        pe_real_values = [
            [self._param(mvau, node_index + 1, "PE", pe, ("MH", mh), True)] for pe in pe_values
        ]
        return [
            MetaParameter("SIMD", values, real_values, node_index),
            MetaParameter("PE", pe_values, pe_real_values, node_index),
        ]

    def _pair_swg_with_vvau(self, swg, vvau, node_index, max_padding):
        """SWG -> VVAU (depthwise convolution): one meta parameter, PE * SIMD, sets the
        VVAU PE (= SWG SIMD) over the channels and, once PE covers all channels, the VVAU
        SIMD over the kernel window with the SWG emitting whole windows."""
        assert swg.get_nodeattr("depthwise") == 1
        _, ifm_values = allowed_divisors(swg.get_nodeattr("IFMChannels"), max_padding)
        kernel_dim = swg.get_nodeattr("ConvKernelDim")
        kernel = ("Kernel", [kernel_dim[0], kernel_dim[1]])
        values, real_values, seen_pe_factors = [], [], []
        for ifm in sorted(set(ifm_values)):
            for pe in divisors(ifm):
                if ifm // pe in seen_pe_factors:
                    continue
                seen_pe_factors.append(ifm // pe)
                parallel_window = int(pe == ifm)
                for simd in divisors(int(np.prod(kernel_dim)) if parallel_window else 1):
                    if simd * pe in values:
                        continue
                    values.append(simd * pe)
                    real_values.append(
                        self._swg_params(swg, node_index, pe, ifm, parallel_window)
                        + [
                            self._param(vvau, node_index + 1, "SIMD", simd, kernel, True),
                            self._param(vvau, node_index + 1, "PE", pe, ("Channels", ifm), True),
                        ]
                    )
        return [MetaParameter("SIMD", values, real_values, node_index)]


def analytic_fifo_sizer():
    """The analytic FIFO sizer (token access vectors from the operators' tree models, no
    IP generation), or None on a tree without it."""
    try:
        from finn.transformation.fpgadataflow.derive_characteristic import (  # noqa: PLC0415
            _UNCHARACTERIZED_OPS,
            DeriveFIFOSizes,
            DeriveTokenAccessVectors,
            is_stream_join,
        )
        from finn.util.fpgadataflow import cleanup_characterization  # noqa: PLC0415
    except ImportError:
        return None

    def characterized(node):
        return node.op_type not in _UNCHARACTERIZED_OPS and not is_stream_join(node)

    return DeriveTokenAccessVectors, DeriveFIFOSizes, cleanup_characterization, characterized


def insert_and_size_fifos(model, fpga_part, work_dir):
    """Insert the FIFOs of a folded graph (DWCs already in), sized by the analytic sizer,
    so that their resources count in the folding's score. Synthesis-free: every schedule
    comes from a tree model. Returns None, after a warning, on a tree without the sizer or
    for a graph with a node without a tree model (e.g. a DWC with a non-integer width
    ratio); the caller then estimates the FIFOs. Sidecar files go to ``work_dir``."""
    sizer = analytic_fifo_sizer()
    if sizer is None:
        warnings.warn(
            "SetFolding: the analytic FIFO sizer is not available on this tree; "
            "the FIFOs are estimated instead"
        )
        return None
    DeriveTokenAccessVectors, DeriveFIFOSizes, cleanup_characterization, characterized = sizer

    model = model.transform(GiveUniqueNodeNames())
    hw_nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]
    without_tree = [
        n.name for n in hw_nodes if characterized(n) and getCustomOp(n).get_tree_model() is None
    ]
    if without_tree:
        warnings.warn(
            "SetFolding: no tree model for %s; the FIFOs of this folding are estimated"
            % ", ".join(without_tree)
        )
        return None
    # the RTL thresholding tree model expects the integer thresholds of the build
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(AnnotateCycles())
    # the sizer keeps each schedule next to the node's generated code: none exists here
    for node in model.graph.node:
        if is_hls_node(node) or is_rtl_node(node):
            getCustomOp(node).set_nodeattr("code_gen_dir_ipgen", work_dir)
    period = int(model.analysis(dataflow_performance)["max_cycles"])
    model = model.transform(
        DeriveTokenAccessVectors(model, period, "tree_model", fpga_part, 0.0, num_workers=1)
    )
    perf = functools.partial(dataflow_performance, use_characterized=True)
    model = model.transform(DeriveFIFOSizes(period=int(model.analysis(perf)["max_cycles"])))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    cleanup_characterization(model)
    model = model.transform(SpecializeLayers(fpga_part))
    return model.transform(GiveUniqueNodeNames())


def common_divisors(numbers):
    separate_divisors = []
    for num in numbers:
        individual_divisors = list(divisors(num))
        separate_divisors.append(individual_divisors)

    return functools.reduce(np.intersect1d, separate_divisors)


# a folding the search produced: the folded model, whether it fits the budget, its
# resources and cycles per frame (with the DWCs the build inserts), the meta values of
# every parameter (to start another search from) and its cost (the search's objective)
FoldResult = namedtuple("FoldResult", "model fits metrics achieved values cost")

# Find the op-type names for all HLS specializations of elementwise binary
# operations
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
]


@dataclass(eq=False)
class SetFolding(Transformation):
    """Set the parallelism (folding) attributes -- PE, SIMD and related -- of every hardware
    node of a dataflow graph, respecting each op's divisibility constraints.

    Styles (``style``):

    * ``"optimizer"`` (default): a resource-aware search. With ``target_cycles_per_frame``
      it meets that throughput at minimum (budget-weighted) resource cost; with ``None`` it
      maximizes throughput within the resource budget of ``platform``. Besides PE/SIMD it
      tunes ram_style/resType to trade BRAM<->URAM and LUT<->DSP. Every folding is scored
      on the graph with the DWCs the build will insert, and with an estimate of its FIFOs
      (or, with ``enable_folding_fifo_heuristic``, with FIFOs sized by the analytic sizer).
      ``folding_maximum_padding`` > 0 lets it pad channel counts for finer folding factors;
      the padded search starts from the unpadded result and is kept only if no worse, and
      the build inserts padding/cropping DWCs. The verdict is in ``last_search_report``.
    * ``"naive"``: the original greedy per-node folder (``apply_naive_folding``).

    In the returned model each node's cycles_estimate attribute holds its estimated cycles.
    """

    #: throughput to meet at minimum resource cost; None maximizes it (optimizer only)
    target_cycles_per_frame: int = None
    #: board (and device count) whose resource budget bounds the search
    platform: str = "Pynq-Z1"
    devices: int = 1
    #: "optimizer" or "naive"
    style: str = "optimizer"
    #: a bias, not a permission: "bram"/"uram", "lut"/"dsp" penalize the resource not named;
    #: resource_weights sets per-resource cost multipliers explicitly
    prefer_memory: str = None
    prefer_compute: str = None
    resource_weights: dict = None
    #: annealing evaluations per parameter (50-100 is the useful range)
    folding_effort: int = 250
    #: channels a layer may be padded by for finer folding factors, and whether the IO
    #: layers may be padded too (the host must then pad the input and crop the output)
    folding_maximum_padding: int = 0
    folding_pad_io_nodes: bool = False
    #: throughput probes in target mode (1: fold once for the target)
    folding_max_attempts: int = 1
    #: raise instead of warn when the returned folding exceeds the budget
    strict_budget: bool = False
    #: let weight memories go to URAM, which makes them runtime-writeable (the driver must
    #: load those weights); without it a BRAM-bound part may have no feasible folding
    allow_uram_weights: bool = False
    #: wall-clock budget of the throughput search, which then keeps the best feasible
    #: folding found (reported as search_truncated); None: no limit
    folding_search_timeout_s: float = None
    #: maximum weight stream width per PE of an MVAU
    mvau_wwidth_max: int = 1024
    #: penalize foldings that need costly DWCs
    enable_folding_dwc_heuristic: bool = True
    #: score foldings with FIFOs sized by the analytic sizer instead of the FIFO estimate
    #: (2.5-3.5x slower); auto_fifo_strategy, the build's strategy, only draws a warning
    enable_folding_fifo_heuristic: bool = False
    auto_fifo_strategy: str = None
    #: naive style: fold again at the achieved cycles if the target was missed
    two_pass_relaxation: bool = True
    #: seed of the annealer, for reproducible foldings
    seed: int = None

    # the folders apply() dispatches to; also the build config's `folding_style` values
    STYLES = ("optimizer", "naive")
    target_resources = ["LUT", "BRAM_18K", "DSP", "URAM"]
    # the resource prefer_memory / prefer_compute penalize, by preference
    PREFERENCES = {
        "prefer_memory": {"bram": "URAM", "uram": "BRAM_18K"},
        "prefer_compute": {"lut": "DSP", "dsp": "LUT"},
    }
    # cost-model weight of the resource a preference avoids (higher = avoided more strongly)
    PREFERENCE_PENALTY = 8.0
    # throughput probes when maximizing throughput (target_cycles_per_frame=None)
    MAXIMIZE_SEARCH_STEPS = 15
    # nodes per partition in the folding pass and in the resource-type pass
    NODES_PER_PARTITION = 3
    RESOURCE_PARTITION_SIZE = 8
    # whole-graph evaluations the polish may spend per folding parameter
    POLISH_EVALS_PER_PARAMETER = 6

    def __post_init__(self):
        super().__init__()
        self.fpgapart = part_map[self.platform]
        if self.enable_folding_fifo_heuristic and self.auto_fifo_strategy not in (
            None,
            "heuristic_analytical",
        ):
            warnings.warn(
                "SetFolding: the FIFO-aware folding sizes FIFOs with the analytic sizer, "
                f"not {self.auto_fifo_strategy}; the build's own FIFO sizing is unaffected"
            )
        self.resource_weights = self._build_resource_weights(
            self.resource_weights,
            prefer_memory=self.prefer_memory,
            prefer_compute=self.prefer_compute,
        )
        # the budget verdict of the last search (see _record_search_result)
        self.last_search_report = None
        self._search_truncated = False
        # sidecar files of the in-loop FIFO sizer, removed after the search
        self._fifo_work_dir = None

    @classmethod
    def _build_resource_weights(cls, overrides=None, **preferences):
        """Per-resource cost multipliers: the resource a preference does NOT name is
        penalized; explicit ``overrides`` win."""
        weights = {"LUT": 1.0, "BRAM_18K": 1.0, "DSP": 1.0, "URAM": 1.0}
        for kind, choice in preferences.items():
            if choice is None:
                continue
            table = cls.PREFERENCES[kind]
            if choice not in table:
                raise ValueError(f"{kind} must be " + ", ".join(map(repr, table)) + " or None")
            weights[table[choice]] = cls.PREFERENCE_PENALTY
        weights.update(overrides or {})
        return weights

    def apply(self, model):
        """Fold ``model`` with the configured style; returns ``(model, False)``."""
        if self.style not in self.STYLES:
            raise ValueError(
                f"unknown SetFolding style {self.style!r}; expected one of "
                + ", ".join(repr(s) for s in self.STYLES)
            )
        if self.style == "naive":
            if self.target_cycles_per_frame is None:
                raise ValueError(
                    "style='naive' requires a target_cycles_per_frame; "
                    "leaving it None (maximize throughput) is only supported by "
                    "style='optimizer'"
                )
            return self.apply_naive_folding(model)
        return self.apply_optimized_folding(model)

    # ------------------------------------------------------------------ optimizer style

    def apply_optimized_folding(self, model):
        """The optimizer style. Overview:

        1. bounds: the cycles per frame at maximal folding and the platform's budget;
        2. ``_search_folding`` picks the throughput targets to fold for (one, or a sweep and
           bisection when maximizing), with and without channel padding;
        3. ``_fold_at`` folds a copy of the model for one target: annealing of PE/SIMD per
           partition of a few nodes against ``Optimizer.cost_model``, then of
           ram_style/resType, then repair and polish on the whole graph (``_refine``),
           scored with DWCs and FIFOs (``_score``);
        4. ``_record_search_result`` publishes the budget verdict of the chosen folding."""
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        weight_types = self._weight_types_as_built(model)
        maximize = self.target_cycles_per_frame is None
        fastest_cycles = self._probe_fastest_throughput(model)
        self._budgets = self._resource_budgets()
        attempts = self.folding_max_attempts
        if maximize:
            attempts = max(attempts, self.MAXIMIZE_SEARCH_STEPS)
        # max_cycles is set per probe by _fold_at
        targets = {"max_cycles": fastest_cycles, **self._budgets}
        opt_template = self._make_optimizer(model, "folding", targets)
        if self.enable_folding_fifo_heuristic:
            self._fifo_work_dir = make_build_dir(prefix="folding_fifo_")
        try:
            model = self._search_folding(opt_template, targets, fastest_cycles, attempts, maximize)
        finally:
            if self._fifo_work_dir is not None:
                shutil.rmtree(self._fifo_work_dir, ignore_errors=True)
                self._fifo_work_dir = None
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        for node in model.graph.node:
            if node.name in weight_types:
                getCustomOp(node).set_nodeattr("weightDataType", weight_types[node.name])
        if self.allow_uram_weights:
            # MVAU/VVAU assert that URAM weight memories are runtime-writeable
            for node in model.graph.node:
                inst = getCustomOp(node)
                types = inst.get_nodeattr_types()
                if "runtime_writeable_weights" in types and "ram_style" in types:
                    if inst.get_nodeattr("ram_style") == "ultra":
                        inst.set_nodeattr("runtime_writeable_weights", 1)
        if self.folding_pad_io_nodes:
            model = self._retype_padded_io(model)
        return (model, False)

    @staticmethod
    def _weight_types_as_built(model):
        """Give the MVAUs/VVAUs the weight datatypes step_minimize_bit_width will build
        (value-based: bipolar weights become INT2, twice the weight memory), so that the
        search scores what is built. Returns the original ones by node name, which the
        folded model gets back (the build minimizes them after folding)."""
        original = {}
        for node in model.graph.node:
            inst = getCustomOp(node) if is_fpgadataflow_node(node) else None
            for op in (MVAU, VVAU):
                # (dynamic MVAUs have no weight initializer)
                if isinstance(inst, op) and model.get_initializer(node.input[1]) is not None:
                    original[node.name] = inst.get_nodeattr("weightDataType")
                    # the op's own rule, without the HLS threshold retyping of subclasses
                    op.minimize_weight_bit_width(inst, model)
        return original

    def _make_optimizer(self, model, name, targets, init_run=False):
        return Optimizer(
            model,
            name,
            targets,
            fpgapart=self.fpgapart,
            padding=0 if init_run else self.folding_maximum_padding,
            effort=self.folding_effort,
            seed=self.seed,
            mvau_wwidth_max=self.mvau_wwidth_max,
            pad_io_nodes=self.folding_pad_io_nodes,
            enable_folding_dwc_heuristic=self.enable_folding_dwc_heuristic,
            resource_weights=self.resource_weights,
            allow_uram_weights=self.allow_uram_weights,
            init_run=init_run,
            resource_type_params=Optimizer.RESOURCE_TYPE_ATTRS,
        )

    def _probe_fastest_throughput(self, model):
        """Cycles per frame with every node folded maximally: the search's fast bound."""
        probe = self._make_optimizer(copy.deepcopy(model), "probe", {"max_cycles": 1}, True)
        probe.generate_parameter_set()
        probe.optimize(
            max_nodes_in_partition=1, target_parameters=["SIMD", "PE", "parallel_window"]
        )
        return probe.model.transform(AnnotateCycles()).analysis(dataflow_performance)["max_cycles"]

    def _slowest_cycles(self, opt_template):
        """Cycles per frame at minimal folding: the search's slow bound."""
        opt = copy.deepcopy(opt_template)
        opt.generate_parameter_set()
        opt.params.add_all_params_to_index_list()
        opt.params.set_values(opt.params.get_min_vals())
        opt.params.apply_updates(final=True, attrs=["SIMD", "PE", "parallel_window"])
        return opt.model.transform(AnnotateCycles()).analysis(dataflow_performance)["max_cycles"]

    def _resource_budgets(self):
        """Per-resource budget of the platform: its resources times the default limits."""
        limit_index = {"LUT": 0, "BRAM_18K": 2, "URAM": 3, "DSP": 4}
        counts = platforms[self.platform](self.devices).resource_count_dict.values()
        return {
            r: max(DEFAULT_RES_LIMITS[i] * sum(c[r] for c in counts), 0.001)
            for r, i in limit_index.items()
        }

    def _aggregate_resources(self, model):
        insts = [getCustomOp(node) for node in model.graph.node]
        return aggregate_dict_keys(
            {inst: inst.node_res_estimation(self.fpgapart) for inst in insts}
        )

    def _cost(self, metrics, targets):
        """The search's objective: resources normalized by budget and weighted."""
        return sum(
            self.resource_weights.get(r, 1.0) * metrics.get(r, 0.0) / targets[r]
            for r in self.target_resources
        )

    def _utilization(self, metrics, targets):
        """Peak utilization over the resources the device has."""
        return max(
            [metrics.get(r, 0.0) / targets[r] for r in self.target_resources if targets[r] > 1]
            or [0.0]
        )

    def _fits(self, metrics, targets):
        return all(metrics.get(r, 0.0) <= targets[r] for r in self.target_resources)

    # ------------------------------------------------------------------ search

    def _search_folding(self, opt_template, targets, fastest_cycles, attempts, maximize):
        """Fold the model, choosing the throughput target.

        With a target and a single attempt: fold once for it (``_fold_at_best``).
        Otherwise find the fastest throughput whose folding fits the budget, between the
        fastest folding and the target (or the slowest folding when maximizing).
        Resource use is not monotonic in the throughput (at minimal folding weight memories
        are deep and narrow and pack badly into BRAM: MobileNet on ZCU104 needs ~1000
        BRAM_18K at minimal folding, ~530 at the hand-tuned one), so a bisection could
        miss every feasible point. Instead the range is sampled first and then refined
        around the fastest feasible point (``_sweep_and_refine``); only a folding verified
        to fit is returned as a solution, else the cheapest one seen, flagged as not fitting.

        With padding the unpadded search runs first and the padded one replays its probes,
        each starting from the unpadded solution there (``_fold_at_best``), and finally
        tries to improve on the best unpadded folding: padding never narrows the result.
        """
        if not maximize and attempts <= 1:
            result = self._fold_at_best(opt_template, targets, self.target_cycles_per_frame, False)
            self._record_search_result(result, targets, result.fits, result.fits, searched=False)
            return result.model

        slowest = self.target_cycles_per_frame
        if maximize:
            slowest = self._slowest_cycles(opt_template)
        lo, hi = max(int(fastest_cycles), 1), max(int(slowest), int(fastest_cycles), 1)
        n_sweep = max(4, (attempts + 1) // 2)
        n_refine = max(0, attempts - n_sweep)
        timeout = self.folding_search_timeout_s
        deadline = None if timeout is None else time.time() + timeout
        self._search_truncated = False

        def out_of_time():
            if deadline is not None and time.time() > deadline:
                self._search_truncated = True
                return True
            return False

        # the unpadded probes by target: the padded probe at a target starts from them
        probes = {}

        def fold_unpadded(t):
            if t not in probes:
                probes[t] = self._verify_as_built(self._fold_at(opt_template, targets, t, 0))
            return probes[t]

        def fold_padded(t):
            result = self._fold_at_best(opt_template, targets, t, maximize, fold_unpadded(t))
            return self._verify_as_built(result)

        search = (lo, hi, n_sweep, n_refine, out_of_time)
        best, fallback = self._sweep_and_refine(fold_unpadded, *search)
        if self.folding_maximum_padding > 0:
            best_padded, fallback_padded = self._sweep_and_refine(fold_padded, *search)
            if best is not None and not out_of_time():
                direct = self._fold_at_best(opt_template, targets, best.achieved, maximize, best)
                direct = self._verify_as_built(direct)
                best_padded = self._better(best_padded, direct, None, maximize)
            best = self._better(best, best_padded, None, maximize)
            fallback = self._better(fallback, fallback_padded, None, maximize)

        if best is not None:
            self._record_search_result(best, targets, True, True)
            return best.model
        # nothing in the range fits: return the least-resource folding, flagged
        self._record_search_result(fallback, targets, False, False)
        return fallback.model

    def _verify_as_built(self, result):
        """``result``, counted as fitting only if it still fits once step_minimize_bit_width
        has run (``_resources_as_built``), so that the search does not settle on a folding
        the verdict then rejects (MinimizeAccumulatorWidth can widen accumulators)."""
        if not result.fits:
            return result
        try:
            as_built = self._resources_as_built(result.model)
        except Exception:  # noqa: BLE001 - the verdict warns about it
            return result
        return result._replace(fits=self._fits(as_built, self._budgets))

    @staticmethod
    def _sweep_targets(lo, hi, n):
        """``n`` cycles-per-frame probes spread geometrically over [lo, hi] (the ends can
        differ by orders of magnitude: MobileNet ~1e5 to ~5e7)."""
        lo, hi = max(int(lo), 1), max(int(hi), 1)
        if hi <= lo or n <= 1:
            return [hi]
        ratio = (hi / lo) ** (1.0 / (n - 1))
        out = []
        for i in range(n):
            t = min(max(int(round(lo * (ratio**i))), lo), hi)
            if t not in out:
                out.append(t)
        if hi not in out:
            out.append(hi)
        return sorted(out)

    def _sweep_and_refine(self, fold, lo, hi, n_sweep, n_refine, out_of_time):
        """Sample ``fold(target)`` over [lo, hi], slowest (cheapest to evaluate) first, then
        bisect between the fastest folding that fits and the slowest faster target that did
        not. Returns the fastest folding verified to fit (or None) and the cheapest one seen."""
        best = fallback = None
        infeasible = []
        for t in reversed(self._sweep_targets(lo, hi, n_sweep)):
            if out_of_time():
                break
            result = fold(t)
            if result.fits:
                if best is None or result.achieved < best.achieved:
                    best = result
            else:
                infeasible.append(t)
            # the least-resource folding (by the search's cost), the faster one on a tie
            if fallback is None or (result.cost, result.achieved) < (
                fallback.cost,
                fallback.achieved,
            ):
                fallback = result

        if best is not None and n_refine:
            # slow foldings can miss the budget too (deep weight memories pack badly)
            faster = [t for t in infeasible if t < best.achieved]
            fast_bound = max(faster) + 1 if faster else 1
            slow_bound = best.achieved
            for _ in range(n_refine):
                if slow_bound - fast_bound <= 1 or out_of_time():
                    break
                mid = (fast_bound + slow_bound) // 2
                result = fold(mid)
                if result.fits:
                    if result.achieved < best.achieved:
                        best = result
                    slow_bound = min(mid, result.achieved)
                else:
                    fast_bound = mid + 1
        return best, fallback

    def _better(self, a, b, target, maximize):
        """The better of two foldings for ``target`` (``None``: no cycle limit).

        ``b`` is the wider (padded) search's result and replaces ``a`` only when it is no
        worse on every count: a solution beats a non-solution; among solutions the faster
        one wins in maximize mode, otherwise ``b`` must be no costlier in both peak
        utilization and weighted cost (and cheaper in one). Among non-solutions the faster
        one within budget wins, else the cheaper one."""
        if a is None or b is None:
            return a if b is None else b
        a_ok = a.fits and (target is None or a.achieved <= target)
        b_ok = b.fits and (target is None or b.achieved <= target)
        if a_ok != b_ok:
            return b if b_ok else a
        if not a_ok:
            if a.fits and b.fits:
                a_key, b_key = (a.achieved, a.cost), (b.achieved, b.cost)
            else:
                a_key, b_key = (not a.fits, a.cost, a.achieved), (not b.fits, b.cost, b.achieved)
            return b if b_key < a_key else a
        if maximize and a.achieved != b.achieved:
            return b if b.achieved < a.achieved else a
        a_util = self._utilization(a.metrics, self._budgets)
        b_util = self._utilization(b.metrics, self._budgets)
        no_worse = b_util <= a_util and b.cost <= a.cost
        return b if no_worse and (b_util < a_util or b.cost < a.cost) else a

    def _fold_at_best(self, opt_template, targets, target_cycles, maximize, unpadded=None):
        """Fold for ``target_cycles``; with padding the padded search starts from the
        unpadded folding (folded here unless given) and the better one is returned."""
        if self.folding_maximum_padding <= 0:
            return self._fold_at(opt_template, targets, target_cycles, 0)
        if unpadded is None:
            unpadded = self._fold_at(opt_template, targets, target_cycles, 0)
        padded = self._fold_at(
            opt_template, targets, target_cycles, self.folding_maximum_padding, unpadded.values
        )
        return self._better(unpadded, padded, target_cycles, maximize)

    def _fold_at(self, opt_template, targets, target_cycles, padding, seed=None):
        """Fold a fresh copy of the model for ``target_cycles`` with up to ``padding``
        channels of padding per layer, starting from the meta values ``seed`` if given.

        Three passes: PE/SIMD by the annealer (from ``seed``, else from the cheapest
        node-by-node folding), ram_style/resType by the annealer, then repair and polish on
        the whole graph (``_refine``). ``achieved`` is what the folded graph reaches with
        its DWCs, which can differ from the target, and is what the search compares."""
        targets["max_cycles"] = target_cycles
        opt = copy.deepcopy(opt_template)
        opt.targets = targets
        opt.padding = padding
        opt.generate_parameter_set()
        opt.optimize(
            initial_guess="cheapest" if seed is None else seed,
            max_nodes_in_partition=self.NODES_PER_PARTITION,
            target_parameters=["SIMD", "PE"],
        )
        opt.optimize(
            initial_guess="max" if seed is None else seed,
            max_nodes_in_partition=min(len(opt.model.graph.node), self.RESOURCE_PARTITION_SIZE),
            target_parameters=list(Optimizer.RESOURCE_TYPE_ATTRS),
        )
        _, achieved, metrics, cost = self._refine(opt, targets, target_cycles)
        # in place: opt.model is a private copy, and copying it per probe is expensive
        opt.model = opt.model.transform(AnnotateCycles(), make_deepcopy=False, cleanup=False)
        values = opt.params.get_meta_values()
        return FoldResult(opt.model, self._fits(metrics, targets), metrics, achieved, values, cost)

    def _score(self, model, opt):
        """Cycles and resources of the graph the build will carry: with the DWCs InsertDWC
        adds (a non-integer-ratio or padding DWC can be slower than both neighbours and
        has resources of its own) and, in the FIFO-aware mode, with sized FIFOs. Returns
        the scored graph, its cycles per frame and its resources."""
        scored = model.transform(InsertDWC())
        scored = scored.transform(SpecializeLayers(self.fpgapart))
        fifo_estimate = None
        if self.enable_folding_fifo_heuristic:
            sized = insert_and_size_fifos(scored, self.fpgapart, self._fifo_work_dir)
            if sized is None:
                # not sizeable: estimate the FIFOs, to be scored like the sized ones
                fifo_estimate = opt.fifo_resources(model, opt.graph_nodes())
            else:
                scored = sized
        scored = scored.transform(AnnotateCycles(), make_deepcopy=False, cleanup=False)
        achieved = scored.analysis(dataflow_performance)["max_cycles"]
        metrics = self._aggregate_resources(scored)
        for r, v in (fifo_estimate or {}).items():
            metrics[r] = metrics.get(r, 0.0) + v
        return scored, achieved, metrics

    @staticmethod
    def _bottleneck_nodes(scored):
        """Names of the folded nodes that bound a scored graph's cycles: the slowest
        nodes, or both neighbours of a slowest DWC."""
        # by position: the DWCs InsertDWC adds are unnamed
        cycles = [getCustomOp(n).get_nodeattr("cycles_estimate") for n in scored.graph.node]
        worst = max(cycles)
        names = set()
        for node, node_cycles in zip(scored.graph.node, cycles):
            if node_cycles != worst:
                continue
            if node.op_type.startswith("StreamingDataWidthConverter"):
                names.add(scored.find_producer(node.input[0]).name)
                names.add(scored.find_consumer(node.output[0]).name)
            else:
                names.add(node.name)
        return names

    def _refine(self, opt, targets, target_cycles):
        """Repair and polish the annealer's folding on the whole graph, DWCs included.

        The annealer scores partitions of a few nodes without the DWCs between them, so it
        can miss the target through a slow DWC and leave single-step savings. *Repair*
        raises the folding of what bounds the cycles (a node, or both neighbours of a DWC)
        one step at a time -- the cheapest step meeting the target, else the fastest --
        until the target is met; it terminates since factors only grow and at full folding
        no DWC is slower than its neighbours. *Polish* then lowers folding factors one step
        at a time where that keeps the target (and the budget, if met) and does not raise
        the cost, within POLISH_EVALS_PER_PARAMETER evaluations per parameter. Every step
        is scored like the verdict (``_score``). Leaves the model at the chosen values and
        returns ``(scored graph, cycles, resources, cost)``."""
        params = opt.params.parameters
        opt.params.add_all_params_to_index_list()
        folding = [i for i, p in enumerate(params) if p.name in ("SIMD", "PE")]

        def evaluate(values):
            opt.params.set_values(values)
            opt.params.apply_updates(final=False, attrs=opt.PARAMETERS_TO_APPLY)
            scored, achieved, metrics = self._score(opt.model, opt)
            cost = self._cost(metrics, targets)
            if not self.enable_folding_fifo_heuristic:
                # the FIFOs are not in the scored graph: estimate them
                cost += opt.resource_cost(opt.fifo_resources(opt.model, opt.graph_nodes()))
            return scored, achieved, metrics, cost

        def step(values, i, direction):
            idx = params[i].possible_values.index(values[i]) + direction
            if not 0 <= idx < len(params[i].possible_values):
                return None
            moved = list(values)
            moved[i] = params[i].possible_values[idx]
            return moved

        values = opt.params.get_meta_values()
        scored, achieved, metrics, cost = evaluate(values)

        while achieved > target_cycles:  # repair
            names = self._bottleneck_nodes(scored)
            moves = [
                moved
                for i in folding
                if any(n.onnx_node.name in names for n in params[i].unique_nodes)
                for moved in [step(values, i, +1)]
                if moved is not None
            ]
            if not moves:
                break
            results = [(moved, *evaluate(moved)) for moved in moves]
            met = [r for r in results if r[2] <= target_cycles]
            if met:
                values, scored, achieved, metrics, cost = min(met, key=lambda r: r[4])
            else:
                values, scored, achieved, metrics, cost = min(results, key=lambda r: (r[2], r[4]))

        fits = self._fits(metrics, targets)  # polish
        limit = max(target_cycles, achieved)
        budget = self.POLISH_EVALS_PER_PARAMETER * len(folding)
        improved = True
        while improved and budget > 0:
            improved = False
            for i in folding:
                moved = step(values, i, -1)
                if moved is None:
                    continue
                budget -= 1
                r = evaluate(moved)
                if r[1] <= limit and (self._fits(r[2], targets) or not fits) and r[3] <= cost:
                    values, scored, achieved, metrics, cost = (moved, *r)
                    fits = self._fits(metrics, targets)
                    improved = True
                if budget <= 0:
                    break

        opt.params.set_values(values)
        opt.params.apply_updates(final=True, attrs=opt.PARAMETERS_TO_APPLY)
        return scored, achieved, metrics, cost

    # ------------------------------------------------------------------ verdict

    def _record_search_result(self, result, targets, fits, feasible_found, searched=True):
        """Publish the budget verdict of the returned folding in ``last_search_report``
        and warn (or raise, with ``strict_budget``) if it does not fit. The verdict is
        judged on the as-built graph (``_resources_as_built``) unless the search's own
        scoring is already over budget (then it cannot change, and MinimizeAccumulatorWidth
        is slowest exactly on such minimal foldings)."""
        metrics = as_built = result.metrics
        scored_over = [r for r in self.target_resources if metrics.get(r, 0.0) > targets[r]]
        if not scored_over:
            try:
                as_built = self._resources_as_built(result.model)
            except Exception as e:  # noqa: BLE001 - never lose the result over a verdict
                warnings.warn(
                    f"SetFolding: could not re-score the folding on the as-built graph "
                    f"({type(e).__name__}: {e}); the budget verdict is the search's own "
                    f"scoring, which does not include step_minimize_bit_width.",
                    stacklevel=2,
                )
        over = {
            r: as_built.get(r, 0.0)
            for r in self.target_resources
            if as_built.get(r, 0.0) > targets[r]
        }
        self.last_search_report = {
            "fits_budget": bool(fits and not over),
            "resources_as_built": {r: as_built.get(r, 0.0) for r in self.target_resources},
            "feasible_solution_found": bool(feasible_found),
            "achieved_cycles_per_frame": result.achieved,
            "resources": {r: metrics.get(r, 0.0) for r in self.target_resources},
            "budget": {r: targets[r] for r in self.target_resources},
            "over_budget": {r: {"used": u, "budget": targets[r]} for r, u in over.items()},
            # the search's scoring fitted, the as-built graph does not
            "missed_by_search_scoring": bool(over and not scored_over),
            "search_truncated": self._search_truncated,
        }
        if not over:
            return
        detail = ", ".join(
            f"{r}: {u:.0f} > {targets[r]:.0f} ({100 * u / targets[r]:.0f}% of budget)"
            for r, u in sorted(over.items())
        )
        if feasible_found:
            why = "This solution was expected to fit and does not."
        elif searched:
            why = (
                "No folding in the searched throughput range fitted the budget, so the "
                "least-resource folding was returned; it is not a solution."
            )
        else:
            why = (
                "Meeting the requested target_cycles_per_frame costs more than "
                "the device has; relax the target or the folding will not fit."
            )
        msg = (
            f"SetFolding: the returned folding does NOT fit the {self.platform} "
            f"resource budget ({detail}). {why}"
        )
        if not scored_over:
            msg += (
                " NOTE: the folding search scored this as fitting; it only exceeds the "
                "budget once step_minimize_bit_width (MinimizeAccumulatorWidth / "
                "RoundAndClipThresholds) has run, which the search does not model."
            )
        if self.strict_budget:
            raise RuntimeError(msg + " (strict_budget=True)")
        warnings.warn(msg, stacklevel=2)

    def _resources_as_built(self, model):
        """Resources of the graph the build carries: with DWCs and after
        step_minimize_bit_width, which can also widen accumulators and thresholds (cnv-w1a2
        on Pynq-Z1: 33428 LUT / 142 BRAM scored, 40025 / 278 as built)."""
        m = copy.deepcopy(model).transform(GiveUniqueNodeNames())
        m = m.transform(InsertDWC())
        m = m.transform(SpecializeLayers(self.fpgapart))
        m = m.transform(MinimizeWeightBitWidth())
        m = m.transform(MinimizeAccumulatorWidth())
        m = m.transform(InferDataTypes())
        return self._aggregate_resources(m)

    def _retype_padded_io(self, model):
        """Give the graph input/output the padded shapes of the IO layers (the host must
        then pad its input and crop its output)."""
        input_shape = getCustomOp(model.graph.node[0]).get_normal_input_shape()
        output_shape = getCustomOp(model.graph.node[-1]).get_normal_output_shape()
        output_name = model.graph.output[0].name
        if len(model.graph.input) != 0:
            model.graph.input.remove(model.graph.input[0])
        model.graph.input.append(
            helper.make_tensor_value_info(
                model.graph.node[0].input[0], TensorProto.FLOAT, [*input_shape]
            )
        )
        if len(model.graph.output) != 0:
            model.graph.output.remove(model.graph.output[0])
        model.graph.output.append(
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [*output_shape])
        )
        return model

    # ------------------------------------------------------------------ naive style

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                # finish if target met
                break

    def apply_naive_folding(self, model):
        """
        A naive folding optimizer implementation

        If two_pass_relaxation is enabled,
        SetFolding will internally run a second time if the target cycles from the
        first pass could not be achieved, instead using the achievable target (which
        may be constrained by a single node) to obtain a balanced pipeline.

        Notable exceptions and special behavior:

        When folding dense convolution/FC compute engines ("MVAU"/MatrixVectorActivation),
        which have two attributes (PE and SIMD):

        * first increases SIMD while weight stream width per PE is <= mvau_wwidth_max
        (configurable in the SetFolding initializer, defaults to 36)
        * then increases PE until the target is met or max PE reached

        When folding depthwise convolutions ("VVAU"/VectorVectorActivation)
        or spatial reduction ops (Pool_Batch):

        * the producer of the node is expected to be a ConvolutionInputGenerator
        with depthwise=1, whose SIMD value will be set equal to the PE value of
        its consumer node
        * the VVAU also supports SIMD ("input window") parallelism next to
        PE ("channels"), but current ConvInpGen limitations require PE to be fully
        unfolded before SIMD is increased
        """

        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "PWPolyF_rtl",
            "Thresholding_hls",
            "Thresholding_rtl",
            *ELEMENTWISE_BINARY_OPS,
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "FMPadding_rtl",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_rtl",
            "StreamingSplit_hls",
            "StreamingConcat_hls",
            "LayerNorm_rtl",
            "Shuffle",
        ]
        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame and simd_val > (max_simd / 1024):
                        # finish if target met and simd value is not too low
                        break
                    if (
                        node_inst.get_input_datatype(1).bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in pe_ops:
                # Note: Keep original behavior for all custom-ops defining the
                # NumChannels attribute as it is
                try:
                    max_pe = node_inst.get_nodeattr("NumChannels")
                # Note: Some of the recent additions do not define the
                # NumChannels attribute
                except AttributeError:
                    # We can extract the channels from the normal, i.e., not
                    # folded, shape of the input in these cases
                    max_pe = node_inst.get_normal_input_shape()[-1]
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    # The depthwise SWU replays its buffered window once per channel
                    # fold, so it can be slower than the VVAU/Pool it feeds (e.g. a
                    # global pooling window over the whole feature map). Keep raising
                    # the shared parallelism until the SWU meets the target as well.
                    if swu_node_inst.get_exp_cycles() >= self.target_cycles_per_frame:
                        for val in divisors(max_pe):
                            if val <= pe:
                                continue
                            pe = val
                            node_inst.set_nodeattr("PE", pe)
                            swu_node_inst.set_nodeattr("SIMD", pe)
                            if swu_node_inst.get_exp_cycles() < self.target_cycles_per_frame:
                                break
                        else:
                            warnings.warn(
                                "Node %s did not meet the target cycles. SIMD was finalized "
                                "to %d (bound to PE of %s). Estimated: %d (cyc/frame), "
                                "Target: %d (cyc/frame)."
                                % (
                                    swu_node.name,
                                    pe,
                                    node.name,
                                    swu_node_inst.get_exp_cycles(),
                                    self.target_cycles_per_frame,
                                )
                            )
                    # enable parallel_window mode of RTL SWG if needed
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(node_inst.get_nodeattr("Kernel"))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise Exception("Undefined edge case for %s" % op_type)
                    if ksize != 1:  # pointwise vvau/pool lack a SWU
                        raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        # init/reset parallel_window mode of RTL SWG
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                        # enable parallel_window mode of RTL SWG if needed
                        simd = node_inst.get_nodeattr("SIMD")
                        cyc = node_inst.get_exp_cycles()
                        if (
                            op_type == "ConvolutionInputGenerator_rtl"
                            and simd == max_simd
                            and cyc > self.target_cycles_per_frame
                        ):
                            node_inst.set_nodeattr("parallel_window", 1)
                    else:
                        # depthwise SWGs are handled separately
                        continue
                elif op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
                    node_inst.set_nodeattr("SIMD", 1)
                    channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
                    for simd_val in common_divisors(channels_per_stream):
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            break
                elif op_type == "LayerNorm_rtl":
                    node_inst.set_nodeattr("SIMD", 1)
                    dim = int(node_inst.get_normal_input_shape()[-1])
                    for simd_val in divisors(dim):
                        if dim // simd_val > 12:
                            node_inst.set_nodeattr("SIMD", simd_val)
                            cyc = node_inst.get_exp_cycles()
                            if cyc < self.target_cycles_per_frame:
                                break
                        else:
                            break
                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            else:
                warnings.warn("SetFolding doesn't know how to handle op_type " + op_type)

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                # run again, but with lower target (that we managed) -- this
                # may be coming from a single node's constraints, but we want
                # to balance the entire dataflow pipeline instead
                # no two_pass_relaxation this time -- no guarantee we'll
                # converge otherwise
                warnings.warn(
                    "Node %s is bottleneck with %d cycles, running second pass"
                    % (perf_dict["max_cycles_node_name"], perf_dict["max_cycles"])
                )
                model = model.transform(
                    SetFolding(
                        target_cycles_per_frame=perf_dict["max_cycles"],
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                        style="naive",
                        folding_maximum_padding=0,
                    )
                )

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        return (model, False)
