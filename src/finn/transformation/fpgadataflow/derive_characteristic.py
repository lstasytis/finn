# Copyright (C) 2022, Xilinx, Inc.
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


import qonnx.custom_op.registry as registry
import warnings
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation

from finn.util.fpgadataflow import is_hls_node, is_rtl_node
import sys
import numpy


def characterized_nodes():
    return [
    "MVAU_hls",
    "MVAU_rtl",
    "StreamingDataWidthConverter_hls",
    "StreamingDataWidthConverter_rtl",
    "ConvolutionInputGenerator_rtl",
    "ConvolutionInputGenerator_hls",
    "StreamingMaxPool_hls",
    "StreamingMaxPool_rtl",
    "LabelSelect_hls",
    "LabelSelect_rtl",
    "Thresholding_hls",
    "Thresholding_rtl",
    "VVAU_hls",
    "VVAU_rtl",
    "FMPadding_hls",
    "FMPadding_rtl",
    "ChannelwiseOp_hls",
    "ChannelwiseOp_rtl",
    ]

def set_ignore_list_for_ip_gen(model: ModelWrapper):
    ch_nodes = characterized_nodes()
    
    for node in model.graph.node:
        inst = registry.getCustomOp(node)
        op_type = node.op_type
        if op_type in ch_nodes:
            inst.set_nodeattr("ipgen_ignore",1)
            print(f"IGNORING ip gen for node type: {op_type}")
        else:
            print(f"NOT ignoring ip gen for node type: {op_type}")
    return model

def unset_ignore_list_for_ip_gen(model: ModelWrapper):
    ch_nodes = characterized_nodes()
    
    for node in model.graph.node:
        inst = registry.getCustomOp(node)
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            inst.set_nodeattr("ipgen_ignore",0)

    return model

class DeriveCharacteristic(NodeLocalTransformation):
    """For each node in the graph, run rtlsim to obtain the i/o
    characteristic function for FIFO sizing and set the attribute.
    It is assumed that the PrepareRTLSim transformation was already
    called on the graph.

    This transformation performs rtlsim for each node, so it will run for
    some time (minutes to hours depending on configuration).

    * period (int) desired period over which the characteristic function
      will be derived.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(self, period, num_workers=None, manual_bypass=False):
        super().__init__(num_workers=num_workers)
        self.period = period
        self.manual_bypass = manual_bypass

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                inst = registry.getCustomOp(node)
                inst.derive_characteristic_fxns(period=self.period)
            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)

    def apply(self, model: ModelWrapper):

        print("deriving characteristic")
        (model, run_again) = super().apply(model)
        if not self.manual_bypass:
            return (model, run_again)
        # apply manual fix for DuplicateStreams and AddStreams for
        # simple residual reconvergent paths with bypass
        addstrm_nodes = model.get_nodes_by_op_type("AddStreams_hls")
        for addstrm_node in addstrm_nodes:
            # we currently only support the case where one branch is
            # a bypass
            b0 = model.find_producer(addstrm_node.input[0])
            b1 = model.find_producer(addstrm_node.input[1])
            if (b0 is None) or (b1 is None):
                warnings.warn("Found unsupported AddStreams, skipping")
                return (model, run_again)
            b0_is_bypass = b0.op_type == "DuplicateStreams_hls"
            b1_is_bypass = b1.op_type == "DuplicateStreams_hls"
            if (not b0_is_bypass) and (not b1_is_bypass):
                warnings.warn("Found unsupported AddStreams, skipping")
                return (model, run_again)
            ds_node = b0 if b0_is_bypass else b1
            comp_branch_last = b1 if b0_is_bypass else b0

            ds_comp_bout = ds_node.output[0] if b0_is_bypass else ds_node.output[1]
            comp_branch_first = model.find_consumer(ds_comp_bout)
            if comp_branch_first is None or comp_branch_last is None:
                warnings.warn("Found unsupported DuplicateStreams, skipping")
                return (model, run_again)
            comp_branch_last = registry.getCustomOp(comp_branch_last)
            comp_branch_first = registry.getCustomOp(comp_branch_first)
            # for DuplicateStreams, use comp_branch_first's input characterization
            # for AddStreams, use comp_branch_last's output characterization
            period = comp_branch_first.get_nodeattr("io_chrc_period")
            comp_branch_first_f = comp_branch_first.get_nodeattr("io_characteristic")[: 2 * period]
            comp_branch_last_f = comp_branch_last.get_nodeattr("io_characteristic")[2 * period :]
            ds_node_inst = registry.getCustomOp(ds_node)
            addstrm_node_inst = registry.getCustomOp(addstrm_node)
            ds_node_inst.set_nodeattr("io_chrc_period", period)
            ds_node_inst.set_nodeattr("io_characteristic", comp_branch_first_f * 2)
            addstrm_node_inst.set_nodeattr("io_chrc_period", period)
            addstrm_node_inst.set_nodeattr("io_characteristic", comp_branch_last_f * 2)
            warnings.warn(f"Set {ds_node.name} chrc. from {comp_branch_first.onnx_node.name}")
            warnings.warn(f"Set {addstrm_node.name} chrc. from {comp_branch_last.onnx_node.name}")
        return (model, run_again)


import numpy as np 

class DeriveFIFOSizes(NodeLocalTransformation):
    """Prerequisite: DeriveCharacteristic already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    to perform FIFO sizing, setting the in/outFIFODepths attributes of HLSCustomOp
    nodes.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(self, num_workers=None, io_fifo_depth=32):
        super().__init__(num_workers=1)
        self.io_fifo_depth = io_fifo_depth

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:

                numpy.set_printoptions(threshold=sys.maxsize)
                # lookup op_type in registry of CustomOps
                prod = registry.getCustomOp(node)
                assert not (op_type.startswith("StreamingFIFO")), "Found existing FIFOs"
                period = prod.get_nodeattr("io_chrc_period")
                cons_chrc = prod.get_nodeattr("io_chrc_in")[0]
                prod_chrc = prod.get_nodeattr("io_chrc_out")[0]
                
                assert len(prod_chrc) == 2 * period, "Found unexpected characterization attribute"
                print("derive sizes producer")
                print("PRODUCER")
                print(node.op_type)
                k = prod.get_nodeattr_types().keys()
                for el in k:
                    try:
                        if el in ["code_gen_dir_ipgen","code_gen_dir_cppsim","MW","MH","PE","SIMD","Dim","Channels","Labels","Kernel","ConvKernelDim","IFMChannels","NumChannels","OFMDim","depthwise","parallel_window","is1D","IFMDim","Stride","Dilation","ImgDim", "PoolDim","numInputVectors"]:
                            print(f'{el}: {prod.get_nodeattr(el)}')
                    except:
                        pass
                #print("period:")
                #print(period)

                #print("PRODUCER IN chr:")
                #print(prod_chrc)
                unique, counts = np.unique(cons_chrc, return_counts=True)
                io_chrc_in_concat = []

                for pair in np.asarray((unique, counts)).T:
                    if pair[1] > 1:
                      #  print(pair) 
                        io_chrc_in_concat.append(pair)

                prod.set_nodeattr("io_chrc_in_concat",np.array(io_chrc_in_concat))
                #print("PRODUCER OUT chr:")
                #print(prod_chrc)
                unique, counts = np.unique(prod_chrc, return_counts=True)
                io_chrc_out_concat = []
                for pair in np.asarray((unique, counts)).T:
                    if pair[1] > 1:
                       # print(pair) 
                        io_chrc_out_concat.append(pair)

                prod.set_nodeattr("io_chrc_out_concat",np.array(io_chrc_out_concat))
                #print("con_chrc:")
                #print(cons_chrc)


                #if any([x > 2 for x in prod.get_nodeattr("outFIFODepths")]):
                #    # FIFO depth already set, can skip this node
                #    return (node, False)
                


              #  assert True == False

                # find consumers
                model = self.ref_input_model
                out_fifo_depths = []
                for output_name in node.output:
                    cons_node = model.find_consumer(output_name)
                    if cons_node is None:
                        # could be final node, will be overridden if so
                        # need an entry in the list anyway
                        out_fifo_depths.append(self.io_fifo_depth)
                        continue
                    cons = registry.getCustomOp(cons_node)
                    k = cons.get_nodeattr_types().keys()
                   # print(cons_node.op_type)
                    for el in k:
                        try:
                            if el in ["MW","MH","PE","SIMD","Dim","Channels","Labels","Kernel","IFMChannels","NumChannels","ConvKernelDim","OFMDim","IFMDim","Stride","Dilation","ImgDim", "PoolDim","numInputVectors"]:
                                print(f'{el}: {cons.get_nodeattr(el)}')
                        except:
                            pass
                    #print("CONSUMER IN chr:")
                    cons_chrc = cons.get_nodeattr("io_chrc_in")[0]

                    unique, counts = np.unique(cons_chrc, return_counts=True)
                    
                    #for pair in np.asarray((unique, counts)).T:
                    #    if pair[1] > 1:
                    #        print(pair) 
                            

                   # print("CONSUMER OUT chr:")
                    prod_chrc = cons.get_nodeattr("io_chrc_out")[0]

                    unique, counts = np.unique(prod_chrc, return_counts=True)
                    
                    #for pair in np.asarray((unique, counts)).T:
                    #    if pair[1] > 1:
                    #        print(pair) 
                            

                    # find minimum phase shift satisfying the constraint
                    pshift_min = period - 1
                    for pshift_cand in range(period):
                        prod_chrc_part = prod_chrc[pshift_cand:period]
                        cons_chrc_part = cons_chrc[: period - pshift_cand]
                        if (prod_chrc_part >= cons_chrc_part).all():
                            pshift_min = pshift_cand
                            break
                    prod_chrc_part = prod_chrc[pshift_min : (pshift_min + period)]
                    cons_chrc_part = cons_chrc[:period]
                    fifo_depth = int((prod_chrc_part - cons_chrc_part).max())
                    out_fifo_depths.append(fifo_depth)
                   # print(f"fifo depth: {fifo_depth}")
                # set output FIFO depth for this (producing) node
                # InsertFIFO looks at the max of (outFIFODepths, inFIFODepths)
                # for each tensor
                prod.set_nodeattr("outFIFODepths", out_fifo_depths)

                # finally, check node inputs to ensure FIFOs are added to
                # any top-level inputs (at least self.io_fifo_depth deep)
                in_fifo_depths = prod.get_nodeattr("inFIFODepths")
                for i, input_name in enumerate(node.input):
                    if input_name in [x.name for x in model.graph.input]:
                        in_fifo_depths[i] = max(self.io_fifo_depth, in_fifo_depths[i])
                prod.set_nodeattr("inFIFODepths", in_fifo_depths)

            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)
