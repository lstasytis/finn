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
import numpy as np
import scipy
import warnings
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
from wrapdisc import Objective
from wrapdisc.var import GridVar

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import part_map
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.platforms import DEFAULT_RES_LIMITS, platforms

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
)

from finn.transformation.fpgadataflow import set_fifo_depths

def parameter_whitelist(padding_input):
    d = {}
    d["SIMD"] = {}
    d["PE"] = {}

    #d [ <parameter name> ] [ < op_type > ] [ padding amount, allow folding or not ]
    d["SIMD"]["DownSampler_hls"]=[padding_input,True,"NumChannels"]
    d["SIMD"]["FMPadding_hls"]=[padding_input,True,"NumChannels"]
    d["SIMD"]["FMPadding_rtl"]=[padding_input,True,"NumChannels"]
    d["SIMD"]["FMPadding_Pixel_hls"]=[padding_input,True,"NumChannels"]

    # SWG foldings are always optimized in tandem with a consumer mvau/vvau
    d["SIMD"]["ConvolutionInputGenerator_hls"]=[0,False,"IFMChannels"]  

    # SWG foldings are always optimized in tandem with a consumer mvau/vvau
    d["SIMD"]["ConvolutionInputGenerator_rtl"]=[0,False,"IFMChannels"]    
  #  d["ram_style"]["ConvolutionInputGenerator_hls"]=[0,True]    

    d["PE"]["AddStreams_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["ChannelwiseOp_hls"]=[padding_input,True,"NumChannels"]
  #  d["ram_style"]["ChannelwiseOp_hls"]=[0,True,None]
    d["PE"]["DuplicateStreams_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["GlobalAccPool_hls"]=[0,True,"NumChannels"]
    d["PE"]["Thresholding_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["Thresholding_rtl"]=[padding_input,True,"NumChannels"]
    d["PE"]["StreamingMaxPool_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["StreamingMaxPool_rtl"]=[padding_input,True,"NumChannels"]

    # Pool nodes are always optimized in tandem with a consumer mvau/vvau
    d["PE"]["Pool_hls"]=[0,False,"Channels"]                           

    # only supported for rtl variant, need to add exceptions
    # so that only if every condition to create a dsp variant is met, 
    # to then allow folding this parameter
    d["SIMD"]["VVAU_hls"]=[0,False,"Kernel"] 

    d["PE"]["VVAU_hls"]=[padding_input,True,"Channels"]

    d["SIMD"]["VVAU_rtl"]=[padding_input,True,"Kernel"]
    d["PE"]["VVAU_rtl"]=[padding_input,True,"Channels"]


   # d["resType"]["VVAU_hls"]=[0,True,None]
   # d["resType"]["VVAU_rtl"]=[0,True,None]

   # d["ram_style"]["VVAU_hls"]=[0,True,None]
  #  d["ram_style"]["VVAU_rtl"]=[0,True,None]

    d["SIMD"]["MVAU_hls"]=[padding_input,True,"MW"]
    d["PE"]["MVAU_hls"]=[padding_input,True,"MH"]

    d["SIMD"]["MVAU_rtl"]=[padding_input,True,"MW"]
    d["PE"]["MVAU_rtl"]=[padding_input,True,"MH"]
  #  d["ram_style"]["MVAU_rtl"]=[0,True,None,[3,2,1,0]]
   # d["ram_style"]["MVAU_hls"]=[0,True,None,[3,2,1,0]]
   # d["ram_style_thresholds"]["MVAU_rtl"]=[0,True,None,[2,1,0]]
   # d["ram_style_thresholds"]["MVAU_hls"]=[0,True,None,[2,1,0]]    
  #  d["resType"]["MVAU_rtl"]=[0,True,None,[1,0]]
   # d["resType"]["MVAU_hls"]=[0,True,None,[1,0]]

    # we do not fold LabelSelect due to it
    # potentially ruining fmax (TODO: heuristic for when
    # its safe to? Like certain topk to label ratio which
    # routes without issues? Or bring back once LabelSelect
    # has been improved / RTL variant added
    d["PE"]["LabelSelect_hls"]=[0,False,"Labels"]


    return d

def allowed_divisors(cap,bounding_value_exponent=1, max_padding_count=0):
    # compute all possible folding factors for a given
    # upper bound variable

    # max_padding_count allows generating values with the assumption
    # that the bounding variable could be padded by up to that many
    # elements, which dramatically increases the possible folding
    # parameters with even a small amount of extra values

    all_divs = []
    all_bounding_values = []
    for i in range(cap, cap + max_padding_count + 1):
        for x in range(1, i + 1):
            if (i**bounding_value_exponent % x) == 0:
                if x not in all_divs and x <= cap:
                    all_divs.append(x)
                    all_bounding_values.append(i)

    return zip(*sorted(zip(all_divs, all_bounding_values)))




class Parameter:
    def __init__(
        self,
        node=None,
        node_original=None,
        node_indx=None,
        op_type=None,
        name=None,
        index=None,
        value=None,
        possible_values=[],
        padding=0,
        target_cycles_per_frame = 1,
        model=None,
        default_bounding_parameter=None,
        used_bounding_parameter=None,
        bounding_value_for_each_possible_value=[],
        dependant_nodes = [],
        generator_bounding_value_for_each_possible_value = [],

    ):
        self.node = node
        self.node_original = node_original
        self.node_index = node_indx - 1
        self.op_type = op_type
        self.name = name
        self.index = index
        self.target_cycles_per_frame = target_cycles_per_frame
        self.value = value
        self.shapes_updated = 0
        self.skip_optimization = False
        self.updated = True
        self.possible_values = possible_values
        self.bounding_value_for_each_possible_value = bounding_value_for_each_possible_value
        self.dependant_nodes = dependant_nodes
        self.model = model
        self.padding=padding
        
        self.default_bounding_parameter = default_bounding_parameter
        self.used_bounding_parameter = used_bounding_parameter
        self.generator_bounding_value_for_each_possible_value = generator_bounding_value_for_each_possible_value
        self.padding_used=False
        self.ram_style_dict = {3: "ultra", 2: "block", 1: "distributed", 0: "auto"}
        self.ram_style_thresholds_dict = {2: "block", 1: "distributed", 0: "auto"}
        self.res_type_dict = {
            2: "dsp",
            1: "lut",
            0: "auto",
        }

    def update_value(self, value):
        if self.value == value:
            self.updated = False
        else:
            self.value = value
            self.updated = True


    def flag_padding(self):
        print(f"flag padding: {self.used_bounding_parameter} and {self.default_bounding_parameter[1]}")
        if self.used_bounding_parameter != self.default_bounding_parameter[1]:
            return True
        else:
            False

    def apply_value(self, final=False, filter=["SIMD,PE"]):
        # Apply a parameter to the model
        # All exceptions, transformations, node insertions and deletions happen here

        if self.name in filter:
            if self.updated:
                # tensor shapes are only updated at the very end,
                # since they do not influence the target function

                # extract out the swu/pool bounding variable if it exists due to padding swu/pool to better fold mvau/vvau

                # depthwise exception
                if self.name in ["SIMD","PE"] and self.op_type in ["VVAU_hls", "VVAU_rtl","Pool_hls"]:
                    pe = self.node.get_nodeattr("PE")
                    max_pe = self.node.get_nodeattr("Channels")
                    
                    if self.value in self.possible_values:
                        if self.name == "SIMD":
                            if pe == max_pe and self.op_type == "VVAU_rtl":
                                self.node.set_nodeattr(self.name, self.value)
                        else:
                            self.value = self.value
                            self.node.set_nodeattr(self.name, self.value) 
                            pe = self.value       



                        if self.name == "SIMD":
                            # Detect any Pool node that might exist prior to an SWG and fold               
                            producer =  self.model.find_producer(self.node.onnx_node.input[0])
                            if producer is not None:
                                if producer.op_type.startswith("Pool"):
                                    producer_inst = getCustomOp(producer)

                                    producer_inst.set_nodeattr("PE",self.value)

                                    swu_node = self.model.find_producer(producer.onnx_node.input[0])
                                else:
                                    swu_node = producer
                                if swu_node is not None:
                                    if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                                        swu_node_inst = getCustomOp(swu_node)
                                        max_swu_simd = swu_node_inst.get_nodeattr("IFMChannels") 
                                        #swu_node_inst.set_nodeattr("IFMChannels", max_pe)
                                        #max_swu_simd = swu_node_inst.get_nodeattr("IFMChannels")
                                        depthwise = swu_node_inst.get_nodeattr("depthwise")
                                        if depthwise == 1 or (depthwise == 0 and pe == max_pe):
                                            swu_node_inst.set_nodeattr("SIMD", pe)
                                        #if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                                        #    if self.value == max_pe:
                                        #        swu_node_inst.set_nodeattr("parallel_window", 1)
                                        #    else:
                                    #         swu_node_inst.set_nodeattr("parallel_window", 0)
                                        # enable parallel_window mode of RTL SWG if needed
                                        if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                                            if max_swu_simd == pe:
                                                cycles = swu_node_inst.get_exp_cycles()
                                                if cycles > self.target_cycles_per_frame:
                                                    swu_node_inst.set_nodeattr("parallel_window", 1)
                                            else:
                                                swu_node_inst.set_nodeattr("parallel_window", 0)

                            else:
                                if self.op_type in ["VVAU_hls", "VVAU_rtl"]:
                                    ksize = np.prod(self.node.get_nodeattr("Kernel"))
                                elif self.op_type == "Pool_hls":
                                    ksize = self.node.get_nodeattr("KernelSize")
                                    
                                else:
                                    raise Exception("Undefined edge case for %s" % self.op_type)
                                if ksize != 1:  # pointwise vvau/pool lack a SWU
                                    raise Exception(
                                        "Expected SWU on DW op input, found " + swu_node.op_type
                                    )


                           # self.dependant_node = swu_node_inst

                #if self.name == "SIMD" and self.op_type == "ConvolutionInputGenerator_rtl":
                #    self.node.set_nodeattr(self.name, self.value)
                #    max_simd = self.node.get_nodeattr("IFMChannels")

                 #   if self.value == max_simd:
                 #       self.node.set_nodeattr("parallel_window", 1)
                 #   else:
                 #       self.node.set_nodeattr("parallel_window", 0)

                if self.name == "ram_style":
                    
                    if (self.ram_style_dict[self.value] == "ultra"
                        and (self.node.op_type.startswith("VVAU") or self.node.op_type.startswith("MVAU"))
                        and self.node.get_nodeattr("runtime_writeable_weights") == 1):
                        pass
                    else:
                        self.node.set_nodeattr(self.name, self.ram_style_dict[self.value])



                if self.name == "resType":
                    self.node.set_nodeattr(self.name, self.res_type_dict[self.value])


                if self.name == "ram_style_thresholds":
                    self.node.set_nodeattr(self.name, self.ram_style_thresholds_dict[self.value])

                consumer = self.model.find_consumer(self.node.onnx_node.output[0])
                if consumer is not None:
                    consumer_inst = getCustomOp(consumer)

                if self.name in ["SIMD", "PE"] and self.op_type not in ["VVAU_hls", "VVAU_rtl"]:


                    if self.op_type in ["MVAU_rtl","MVAU_hls"] and self.name == "SIMD":
                        print("updating MVAU producers:")
                        print(len(self.dependent.nodes))
                        print(generator_bounding_val)
                        print(self.value)
                        if len(self.dependant_nodes) == 1:
                            generator_bounding_val = self.generator_bounding_value_for_each_possible_value[
                                self.possible_values.index(self.value)
                            ]
                            # swu node
                            self.dependant_nodes[0].set_nodeattr("IFMChannels",generator_bounding_val)

                        if len(self.dependant_nodes) == 2:
                            generator_bounding_val = self.generator_bounding_value_for_each_possible_value[
                                self.possible_values.index(self.value)
                            ]
                            # pooling node
                            self.dependant_nodes[0].set_nodeattr("NumChannels",generator_bounding_val)
                            # swu node
                            self.dependant_nodes[1].set_nodeattr("IFMChannels",generator_bounding_val)


                        producer =  self.model.find_producer(self.node.onnx_node.input[0])
                        if producer is not None:
                            
                            if producer.op_type.startswith("ConvolutionInputGenerator"):
                                swu_node_inst = getCustomOp(producer)
                                
                               # swu_node_inst.set_nodeattr("IFMChannels",generator_bounding_val)
                                #max_swu_simd = swu_node_inst.get_nodeattr("IFMChannels") 
                               # if self.value <= max_swu_simd:
                                #    swu_node_inst.set_nodeattr("SIMD",self.value)

                                if self.value == generator_bounding_val:
                                    cycles = swu_node_inst.get_exp_cycles()
                                    if cycles > self.target_cycles_per_frame:
                                        swu_node_inst.set_nodeattr("parallel_window", 1)
                                else:
                                    swu_node_inst.set_nodeattr("parallel_window", 0)

                    
                    if (
                    not ( self.op_type in ["ConvolutionInputGenerator_rtl"] and 
                    consumer_inst not in ["VVAU_hls", "VVAU_rtl"]) and 
                    self.default_bounding_parameter[0] not in ["Dim"]):
                        bounding_val = self.bounding_value_for_each_possible_value[
                            self.possible_values.index(self.value)
                    ]

                        if self.default_bounding_parameter[0] not in ["Kernel"]:
                            self.node.set_nodeattr(self.default_bounding_parameter[0], bounding_val)

                    if self.op_type.startswith("ConvolutionInputGenerator") and consumer.op_type not in ["VVAU_hls", "VVAU_rtl"]:
                        depthwise = self.node.get_nodeattr("depthwise")
                        max_simd = self.node.get_nodeattr("IFMChannels")
                        if depthwise == 1 or (depthwise == 0 and self.value == max_simd):
                            self.node.set_nodeattr(self.name, self.value)
                        if (
                            self.op_type == "ConvolutionInputGenerator_rtl"
                            and self.value == max_simd
                        ):
                            self.node.set_nodeattr("parallel_window", 0)
                        
                    else:
                        self.node.set_nodeattr(self.name, self.value)
               # print(f"applied {self.value} to {self.name} of node {self.node}")
            # after the optimization routines, when the final values are being applied,
            # additionally update any bounding parameters such as MW and MH to introduce
            # padding if necessary to support more folding factors. Crucially, update the
            # the tensor shapes as well.
            if final and self.name in ["SIMD", "PE"] and self.default_bounding_parameter[0] not in ["Dim"]:

                bounding_val = self.bounding_value_for_each_possible_value[
                    self.possible_values.index(self.value)
                ]

                if self.default_bounding_parameter[0] not in ["Kernel"]:
                    self.node.set_nodeattr(self.default_bounding_parameter[0], bounding_val)
       
                new_shape = getCustomOp(
                    self.model.graph.node[self.node_index]
                ).get_normal_output_shape()

                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].output[0], new_shape
                )
                

                old_shape = self.model.get_tensor_shape(
                    self.model.graph.node[self.node_index].output[0]
                )

                if old_shape != new_shape:
                    self.shapes_updated += 1

                if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
                    # also update weight matrix and threshold vector
                    # if its an mvau node

                    mw = self.node.get_nodeattr("MW")
                    mh = self.node.get_nodeattr("MH")

                    # proto W tensor
                    W = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])

                    if (mw, mh) != W.shape:
                        self.shapes_updated += 1
                        wdt = self.model.get_tensor_datatype(
                            self.model.graph.node[self.node_index].input[1]
                        )

                        W_new = gen_finn_dt_tensor(wdt, (mw, mh))
                        W_new[...] = 0

                        W_new[: min(mw, W.shape[0]), : min(mh, W.shape[1])] = W[
                            : min(mw, W.shape[0]), : min(mh, W.shape[1])
                        ]
                        self.model.set_initializer(
                            self.model.graph.node[self.node_index].input[1], W_new
                        )

                    # proto T tensor
                    if len(self.model.graph.node[self.node_index].input) == 3:
                        T = self.model.get_initializer(
                            self.model.graph.node[self.node_index].input[2]
                        )

                        adt = self.model.get_tensor_datatype(
                            self.model.graph.node[self.node_index].input[2]
                        )
                        T_new = gen_finn_dt_tensor(adt, (mh, T.shape[1]))
                        T_new[...] = 0

                        T_new[: min(mh, T.shape[0]), :] = T[: min(mh, T.shape[0]), :]

                        if T.shape != T_new.shape:
                            self.shapes_updated += 1
                            self.model.set_initializer(
                                self.model.graph.node[self.node_index].input[2], T_new
                            )




                if self.op_type in ["Thresholding_hls", "Thresholding_rtl"]:
                    mh = self.node.get_nodeattr("NumChannels")
                    # thresholding nodes have a weight matrix which needs to be
                    # adjusted if padding or cropping were introduced
                    T = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])
                    adt = self.model.get_tensor_datatype(
                        self.model.graph.node[self.node_index].input[1]
                    )
                    T_new = gen_finn_dt_tensor(adt, (mh, T.shape[1]))
                    T_new[...] = 0

                    T_new[: min(mh, T.shape[0]), :] = T[: min(mh, T.shape[0]), :]


                    if T.shape != T_new.shape:
                        self.shapes_updated += 1
                        self.model.set_initializer(
                            self.model.graph.node[self.node_index].input[1], T_new
                        )

                if self.op_type in ["VVAU_hls","VVAU_rtl"]:
                    W = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])
                    ch = self.node.get_nodeattr("Channels")
                    k = self.node.get_nodeattr("Kernel")
                    #assert True == False
                    if W.shape[0] != ch or W.shape[-2:] == tuple(k):
                        # padding 

                        self.shapes_updated += 1
                        wdt = self.model.get_tensor_datatype(self.model.graph.node[self.node_index].input[1])
						
                        # W_new = np.zeros(wdt.min(), wdt.max() + 1, size=(mw, mh))

                        W_new = gen_finn_dt_tensor(wdt, (ch ,W.shape[1],k[0],k[1]))
                        W_new[...] = 0

                        W_new[:min(ch, W.shape[0]) , : , : min(k[0], W.shape[2]) , : min(k[1], W.shape[3])] = W[:min(ch, W.shape[0]) , : , : min(k[0], W.shape[2]) , : min(k[1], W.shape[3])]
                        
                        self.model.set_initializer(
                            self.model.graph.node[self.node_index].input[1], W_new
                        )

                    # proto T tensor
                    if len(self.model.graph.node[self.node_index].input) == 3:
                        T = self.model.get_initializer(
                            self.model.graph.node[self.node_index].input[2]
                        )

                        ch = self.node.get_nodeattr("Channels")
                        adt = self.model.get_tensor_datatype(
                            self.model.graph.node[self.node_index].input[2]
                        )

                        T_new = gen_finn_dt_tensor(adt, (ch, T.shape[1]))
                        T_new[...] = 0

                        T_new[: min(ch, T.shape[0]), :] = T[: min(ch, T.shape[0]), :]

                        if T.shape != T_new.shape:
                            self.shapes_updated += 1
                            self.model.set_initializer(
                                self.model.graph.node[self.node_index].input[2], T_new
                            )



        if self.updated:
            return True
        else:
            return False

    def get_cycles(self):
        own_cycles = self.node.get_exp_cycles()

        # in case we have to tie in a DW input node, consider it as well
        if len(self.dependant_nodes) >0:
            other_cycles = max([x.get_exp_cycles() for x in self.dependant_nodes])
            own_cycles = max(own_cycles, other_cycles)
        #assert True==False
        return



class Value:
    def __init__(
            self,
            name = None, # SWU_SIMD, MVAU_SIMD, MVAU_PE etc
            target_value_name = None,
            target_value = None,
            target_value_last = None,
            bound_name = None,
            bound_value = None,
            bound_value_last = None,
            update_threshold_input = False,
            update_weights_input = False,
            update_input_tensor_shape = False,
            update_output_tensor_shape = False,
            node = None,                    # node instance!
            model = None,
    ):
        self.target_value_name = target_value_name,
        self.target_value = target_value,
        self.target_value_last = target_value_last,
        self.bound_name = bound_name,
        self.bound_value = bound_value,
        self.bound_value_last = bound_value_last
        self.update_threshold_input = update_threshold_input,
        self.update_weights_input = update_weights_input,
        self.update_input_tensor_shape = update_input_tensor_shape,
        self.update_output_tensor_shape = update_output_tensor_shape,
        self.node = node,
        self.op_type = node.op_type
        self.model = model,


    def update_threshold_tensor(self):
        if self.op_type in ["Thresholding_hls","Thresholding_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("NumChannels")

        elif self.op_type in ["VVAU_hls","VVAU_rtl"]:
            input_index = 2
            dim0 = self.node.get_nodeattr("Channels")

        elif self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            input_index = 2
            dim0 = self.node.get_nodeattr("MH")

        # thresholding nodes have a weight matrix which needs to be
        # adjusted if padding or cropping were introduced
            
        T = self.model.get_initializer(
            self.model.graph.node[self.node_index].input[input_index]
            )
        
        adt = self.model.get_tensor_datatype(
            self.model.graph.node[self.node_index].input[input_index]
        )
        T_new = gen_finn_dt_tensor(adt, (dim0, T.shape[1]))
        T_new[...] = 0

        T_new[: min(dim0, T.shape[0]), :] = T[: min(dim0, T.shape[0]), :]

        self.model.set_initializer(
            self.model.graph.node[self.node_index].input[input_index], T_new
        )

    def update_weight_tensor(self):

        if self.op_type in ["VVAU_hls","VVAU_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("Channels")
            dim1 = self.node.get_nodeattr("Kernel")

        elif self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("MW")
            dim1 = self.node.get_nodeattr("MH")


        W = self.model.get_initializer(self.model.graph.node[self.node_index].input[input_index])
        
        if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            if (dim0, dim1) == W.shape:
                return False
        
        if self.op_type in ["VVAU_hls","VVAU_rtl"]:
            if W.shape[0] == dim0 and W.shape[-2:] == tuple(dim1):
                return False

        wdt = self.model.get_tensor_datatype(
            self.model.graph.node[self.node_index].input[input_index]
        )

        W_new = gen_finn_dt_tensor(wdt, (dim0, dim1))
        W_new[...] = 0

        W_new[: min(dim0, W.shape[0]), : min(dim0, W.shape[1])] = W[
            : min(dim0, W.shape[0]), : min(dim0, W.shape[1])
        ]
        self.model.set_initializer(
            self.model.graph.node[self.node_index].input[1], W_new
        )

        wdt = self.model.get_tensor_datatype(self.model.graph.node[self.node_index].input[input_index])

        W_new = gen_finn_dt_tensor(wdt, (dim0 ,W.shape[1],dim1[0],dim1[1]))
        W_new[...] = 0

        W_new[ : min(dim0, W.shape[0]) , : , \
               : min(dim1[0], W.shape[2]) , \
               : min(dim1[1], W.shape[3])] = W[:min(dim0, W.shape[0]) , : \
              ,: min(dim1[0], W.shape[2]) , : min(dim1[1], W.shape[3])]
        
        self.model.set_initializer(
            self.model.graph.node[self.node_index].input[input_index], W_new
        )

        return True



    def apply_value(self,final=True):

        # update the target value being optimized
        if self.target_value != self.last_value_last:
            self.node.set_nodeattr(self.target_value_name,self.target_value)
        
        # if the bounding value has changed (ie,. MW of an MVAU) 
        # update it as well
        if self.bound_value != self.bound_value_last:
            self.node.set_nodeattr(self.bound_name,self.bound_value)
        
        # if this is the end of the minimizer routine, we update the tensor
        # shapes as well
        if final:
            op_type = self.node.op_type

            # first the io tensors only
            if self.update_input_tensor_shape:
                new_shape = self.node.get_normal_input_shape()
                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].input[0], new_shape
                )                            

            if self.update_output_tensor_shape:
                new_shape = self.node.get_normal_output_shape()
                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].output[0], new_shape
                )

            if self.update_threshold_input:
                self.update_threshold_tensor()


            if self.update_weight_input:
                self.update_weight_tensor()



class MetaParameter:
    """
    A parameter defines a single optimizable integer value (meta_value)
    which translates into a set of finn-onnx graph node attributes
    which are tighly linked together (called values)

    Examples: 
    -PE and SIMD values of an MVAU (FC layer)
    -SIMD value of an SWU and the PE and SIMD values of an MVAU (convolution)
    -SWU and Pool layer SIMD values (max pooling using SWU)

    All possible (legal) combinations of real values are stored in a list and an
    address translation is performed to map each meta_value to a set
    of real values when applying them
    """

    def __init__(
            self,
            meta_value = None,
            possible_meta_values = [],
            real_values = [], #list of lists
            model = None
    ):
        self.name = ""
        self.meta_value = meta_value
        self.possible_meta_values = possible_meta_values
        self.real_values = real_values
        self.model = model
        self.index = 0

        # we build up a list of unique nodes relates to this meta parameter
        # for future cycle calculations
        self.unique_nodes = []
        for val in real_values[0]:
            if val.node not in self.unique_nodes:
                self.unique_nodes.append(val.node)

        for value in real_values:
            self.name += f"{value.op_type}_{value.name}+"

    def update_meta_value(self,value,final=False):
        # make sure to run this once before minimizing
        if value != self.meta_value:
            self.meta_value = value
            self.index = self.possible_meta_values.index(self.meta_value)
            for val in self.real_values[self.index]:
                val.apply_value(final)

    def get_cycles(self):
        return max([n.get_exp_cycles() for n in self.unique_nodes])


class Parameter:
    def __init__(
        self,
        meta_value = None,
        value_to_index_map = [],
        values = [],
        bounding_parameter_names = [],
        bounding_parameter_values = [],


        node=None,
        node_original=None,
        node_indx=None,
        op_type=None,
        name=None,
        index=None,
        value=None,
        possible_values=[],
        padding=0,
        target_cycles_per_frame = 1,
        model=None,
        default_bounding_parameter=None,
        used_bounding_parameter=None,
        bounding_value_for_each_possible_value=[],
        dependant_nodes = [],
        generator_bounding_value_for_each_possible_value = [],

    ):
        self.node = node
        self.node_original = node_original
        self.node_index = node_indx - 1
        self.op_type = op_type
        self.name = name
        self.index = index
        self.target_cycles_per_frame = target_cycles_per_frame
        self.value = value
        self.shapes_updated = 0
        self.skip_optimization = False
        self.updated = True
        self.possible_values = possible_values
        self.bounding_value_for_each_possible_value = bounding_value_for_each_possible_value
        self.dependant_nodes = dependant_nodes
        self.model = model
        self.padding=padding
        
        self.default_bounding_parameter = default_bounding_parameter
        self.used_bounding_parameter = used_bounding_parameter
        self.generator_bounding_value_for_each_possible_value = generator_bounding_value_for_each_possible_value
        self.padding_used=False
        self.ram_style_dict = {3: "ultra", 2: "block", 1: "distributed", 0: "auto"}
        self.ram_style_thresholds_dict = {2: "block", 1: "distributed", 0: "auto"}
        self.res_type_dict = {
            2: "dsp",
            1: "lut",
            0: "auto",
        }

    def update_value(self, value):
        if self.value == value:
            self.updated = False
        else:
            self.value = value
            self.updated = True


    def flag_padding(self):
        print(f"flag padding: {self.used_bounding_parameter} and {self.default_bounding_parameter[1]}")
        if self.used_bounding_parameter != self.default_bounding_parameter[1]:
            return True
        else:
            False

    def apply_value(self, final=False, filter=["SIMD,PE"]):
        # Apply a parameter to the model
        # All exceptions, transformations, node insertions and deletions happen here

        if self.name in filter:
            if self.updated:
                # tensor shapes are only updated at the very end,
                # since they do not influence the target function

                # extract out the swu/pool bounding variable if it exists due to padding swu/pool to better fold mvau/vvau

                # depthwise exception
                if self.name in ["SIMD","PE"] and self.op_type in ["VVAU_hls", "VVAU_rtl","Pool_hls"]:
                    pe = self.node.get_nodeattr("PE")
                    max_pe = self.node.get_nodeattr("Channels")
                    
                    if self.value in self.possible_values:
                        if self.name == "SIMD":
                            if pe == max_pe and self.op_type == "VVAU_rtl":
                                self.node.set_nodeattr(self.name, self.value)
                        else:
                            self.value = self.value
                            self.node.set_nodeattr(self.name, self.value) 
                            pe = self.value       



                        if self.name == "SIMD":
                            # Detect any Pool node that might exist prior to an SWG and fold               
                            producer =  self.model.find_producer(self.node.onnx_node.input[0])
                            if producer is not None:
                                if producer.op_type.startswith("Pool"):
                                    producer_inst = getCustomOp(producer)

                                    producer_inst.set_nodeattr("PE",self.value)

                                    swu_node = self.model.find_producer(producer.onnx_node.input[0])
                                else:
                                    swu_node = producer
                                if swu_node is not None:
                                    if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                                        swu_node_inst = getCustomOp(swu_node)
                                        max_swu_simd = swu_node_inst.get_nodeattr("IFMChannels") 
                                        #swu_node_inst.set_nodeattr("IFMChannels", max_pe)
                                        #max_swu_simd = swu_node_inst.get_nodeattr("IFMChannels")
                                        depthwise = swu_node_inst.get_nodeattr("depthwise")
                                        if depthwise == 1 or (depthwise == 0 and pe == max_pe):
                                            swu_node_inst.set_nodeattr("SIMD", pe)
                                        #if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                                        #    if self.value == max_pe:
                                        #        swu_node_inst.set_nodeattr("parallel_window", 1)
                                        #    else:
                                    #         swu_node_inst.set_nodeattr("parallel_window", 0)
                                        # enable parallel_window mode of RTL SWG if needed
                                        if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                                            if max_swu_simd == pe:
                                                cycles = swu_node_inst.get_exp_cycles()
                                                if cycles > self.target_cycles_per_frame:
                                                    swu_node_inst.set_nodeattr("parallel_window", 1)
                                            else:
                                                swu_node_inst.set_nodeattr("parallel_window", 0)

                            else:
                                if self.op_type in ["VVAU_hls", "VVAU_rtl"]:
                                    ksize = np.prod(self.node.get_nodeattr("Kernel"))
                                elif self.op_type == "Pool_hls":
                                    ksize = self.node.get_nodeattr("KernelSize")
                                    
                                else:
                                    raise Exception("Undefined edge case for %s" % self.op_type)
                                if ksize != 1:  # pointwise vvau/pool lack a SWU
                                    raise Exception(
                                        "Expected SWU on DW op input, found " + swu_node.op_type
                                    )


                           # self.dependant_node = swu_node_inst

                #if self.name == "SIMD" and self.op_type == "ConvolutionInputGenerator_rtl":
                #    self.node.set_nodeattr(self.name, self.value)
                #    max_simd = self.node.get_nodeattr("IFMChannels")

                 #   if self.value == max_simd:
                 #       self.node.set_nodeattr("parallel_window", 1)
                 #   else:
                 #       self.node.set_nodeattr("parallel_window", 0)

                if self.name == "ram_style":
                    
                    if (self.ram_style_dict[self.value] == "ultra"
                        and (self.node.op_type.startswith("VVAU") or self.node.op_type.startswith("MVAU"))
                        and self.node.get_nodeattr("runtime_writeable_weights") == 1):
                        pass
                    else:
                        self.node.set_nodeattr(self.name, self.ram_style_dict[self.value])



                if self.name == "resType":
                    self.node.set_nodeattr(self.name, self.res_type_dict[self.value])


                if self.name == "ram_style_thresholds":
                    self.node.set_nodeattr(self.name, self.ram_style_thresholds_dict[self.value])

                consumer = self.model.find_consumer(self.node.onnx_node.output[0])
                if consumer is not None:
                    consumer_inst = getCustomOp(consumer)

                if self.name in ["SIMD", "PE"] and self.op_type not in ["VVAU_hls", "VVAU_rtl"]:


                    if self.op_type in ["MVAU_rtl","MVAU_hls"] and self.name == "SIMD":
                        print("updating MVAU producers:")
                        print(len(self.dependent.nodes))
                        print(generator_bounding_val)
                        print(self.value)
                        if len(self.dependant_nodes) == 1:
                            generator_bounding_val = self.generator_bounding_value_for_each_possible_value[
                                self.possible_values.index(self.value)
                            ]
                            # swu node
                            self.dependant_nodes[0].set_nodeattr("IFMChannels",generator_bounding_val)

                        if len(self.dependant_nodes) == 2:
                            generator_bounding_val = self.generator_bounding_value_for_each_possible_value[
                                self.possible_values.index(self.value)
                            ]
                            # pooling node
                            self.dependant_nodes[0].set_nodeattr("NumChannels",generator_bounding_val)
                            # swu node
                            self.dependant_nodes[1].set_nodeattr("IFMChannels",generator_bounding_val)


                        producer =  self.model.find_producer(self.node.onnx_node.input[0])
                        if producer is not None:
                            
                            if producer.op_type.startswith("ConvolutionInputGenerator"):
                                swu_node_inst = getCustomOp(producer)
                                
                               # swu_node_inst.set_nodeattr("IFMChannels",generator_bounding_val)
                                #max_swu_simd = swu_node_inst.get_nodeattr("IFMChannels") 
                               # if self.value <= max_swu_simd:
                                #    swu_node_inst.set_nodeattr("SIMD",self.value)

                                if self.value == generator_bounding_val:
                                    cycles = swu_node_inst.get_exp_cycles()
                                    if cycles > self.target_cycles_per_frame:
                                        swu_node_inst.set_nodeattr("parallel_window", 1)
                                else:
                                    swu_node_inst.set_nodeattr("parallel_window", 0)

                    
                    if (
                    not ( self.op_type in ["ConvolutionInputGenerator_rtl"] and 
                    consumer_inst not in ["VVAU_hls", "VVAU_rtl"]) and 
                    self.default_bounding_parameter[0] not in ["Dim"]):
                        bounding_val = self.bounding_value_for_each_possible_value[
                            self.possible_values.index(self.value)
                    ]

                        if self.default_bounding_parameter[0] not in ["Kernel"]:
                            self.node.set_nodeattr(self.default_bounding_parameter[0], bounding_val)

                    if self.op_type.startswith("ConvolutionInputGenerator") and consumer.op_type not in ["VVAU_hls", "VVAU_rtl"]:
                        depthwise = self.node.get_nodeattr("depthwise")
                        max_simd = self.node.get_nodeattr("IFMChannels")
                        if depthwise == 1 or (depthwise == 0 and self.value == max_simd):
                            self.node.set_nodeattr(self.name, self.value)
                        if (
                            self.op_type == "ConvolutionInputGenerator_rtl"
                            and self.value == max_simd
                        ):
                            self.node.set_nodeattr("parallel_window", 0)
                        
                    else:
                        self.node.set_nodeattr(self.name, self.value)
               # print(f"applied {self.value} to {self.name} of node {self.node}")
            # after the optimization routines, when the final values are being applied,
            # additionally update any bounding parameters such as MW and MH to introduce
            # padding if necessary to support more folding factors. Crucially, update the
            # the tensor shapes as well.
            if final and self.name in ["SIMD", "PE"] and self.default_bounding_parameter[0] not in ["Dim"]:

                bounding_val = self.bounding_value_for_each_possible_value[
                    self.possible_values.index(self.value)
                ]

                if self.default_bounding_parameter[0] not in ["Kernel"]:
                    self.node.set_nodeattr(self.default_bounding_parameter[0], bounding_val)
       
                new_shape = getCustomOp(
                    self.model.graph.node[self.node_index]
                ).get_normal_output_shape()

                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].output[0], new_shape
                )
                

                old_shape = self.model.get_tensor_shape(
                    self.model.graph.node[self.node_index].output[0]
                )

                if old_shape != new_shape:
                    self.shapes_updated += 1

                if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
                    # also update weight matrix and threshold vector
                    # if its an mvau node

                    mw = self.node.get_nodeattr("MW")
                    mh = self.node.get_nodeattr("MH")

                    # proto W tensor
                    W = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])

                    if (mw, mh) != W.shape:
                        self.shapes_updated += 1
                        wdt = self.model.get_tensor_datatype(
                            self.model.graph.node[self.node_index].input[1]
                        )

                        W_new = gen_finn_dt_tensor(wdt, (mw, mh))
                        W_new[...] = 0

                        W_new[: min(mw, W.shape[0]), : min(mh, W.shape[1])] = W[
                            : min(mw, W.shape[0]), : min(mh, W.shape[1])
                        ]
                        self.model.set_initializer(
                            self.model.graph.node[self.node_index].input[1], W_new
                        )

                    # proto T tensor
                    if len(self.model.graph.node[self.node_index].input) == 3:
                        T = self.model.get_initializer(
                            self.model.graph.node[self.node_index].input[2]
                        )

                        adt = self.model.get_tensor_datatype(
                            self.model.graph.node[self.node_index].input[2]
                        )
                        T_new = gen_finn_dt_tensor(adt, (mh, T.shape[1]))
                        T_new[...] = 0

                        T_new[: min(mh, T.shape[0]), :] = T[: min(mh, T.shape[0]), :]

                        if T.shape != T_new.shape:
                            self.shapes_updated += 1
                            self.model.set_initializer(
                                self.model.graph.node[self.node_index].input[2], T_new
                            )




                if self.op_type in ["Thresholding_hls", "Thresholding_rtl"]:
                    mh = self.node.get_nodeattr("NumChannels")
                    # thresholding nodes have a weight matrix which needs to be
                    # adjusted if padding or cropping were introduced
                    T = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])
                    adt = self.model.get_tensor_datatype(
                        self.model.graph.node[self.node_index].input[1]
                    )
                    T_new = gen_finn_dt_tensor(adt, (mh, T.shape[1]))
                    T_new[...] = 0

                    T_new[: min(mh, T.shape[0]), :] = T[: min(mh, T.shape[0]), :]


                    if T.shape != T_new.shape:
                        self.shapes_updated += 1
                        self.model.set_initializer(
                            self.model.graph.node[self.node_index].input[1], T_new
                        )

                if self.op_type in ["VVAU_hls","VVAU_rtl"]:
                    W = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])
                    ch = self.node.get_nodeattr("Channels")
                    k = self.node.get_nodeattr("Kernel")
                    #assert True == False
                    if W.shape[0] != ch or W.shape[-2:] == tuple(k):
                        # padding 

                        self.shapes_updated += 1
                        wdt = self.model.get_tensor_datatype(self.model.graph.node[self.node_index].input[1])
						
                        # W_new = np.zeros(wdt.min(), wdt.max() + 1, size=(mw, mh))

                        W_new = gen_finn_dt_tensor(wdt, (ch ,W.shape[1],k[0],k[1]))
                        W_new[...] = 0

                        W_new[:min(ch, W.shape[0]) , : , : min(k[0], W.shape[2]) , : min(k[1], W.shape[3])] = W[:min(ch, W.shape[0]) , : , : min(k[0], W.shape[2]) , : min(k[1], W.shape[3])]
                        
                        self.model.set_initializer(
                            self.model.graph.node[self.node_index].input[1], W_new
                        )

                    # proto T tensor
                    if len(self.model.graph.node[self.node_index].input) == 3:
                        T = self.model.get_initializer(
                            self.model.graph.node[self.node_index].input[2]
                        )

                        ch = self.node.get_nodeattr("Channels")
                        adt = self.model.get_tensor_datatype(
                            self.model.graph.node[self.node_index].input[2]
                        )

                        T_new = gen_finn_dt_tensor(adt, (ch, T.shape[1]))
                        T_new[...] = 0

                        T_new[: min(ch, T.shape[0]), :] = T[: min(ch, T.shape[0]), :]

                        if T.shape != T_new.shape:
                            self.shapes_updated += 1
                            self.model.set_initializer(
                                self.model.graph.node[self.node_index].input[2], T_new
                            )



        if self.updated:
            return True
        else:
            return False

    def get_cycles(self):
        own_cycles = self.node.get_exp_cycles()

        # in case we have to tie in a DW input node, consider it as well
        if len(self.dependant_nodes) >0:
            other_cycles = max([x.get_exp_cycles() for x in self.dependant_nodes])
            own_cycles = max(own_cycles, other_cycles)
        #assert True==False
        return

class ParameterSet:
    def __init__(self):
        self.parameters = []
        self.index_list = []
        self.nodes = []

    def filter(self, params_to_filter):
        # filter parameters we want to use in the set
        # useful for multi-pass optimization
        self.parameters = [x for x in self.parameters if x.name in params_to_filter]

    def get_max_cycles(self):
        return max([n.get_exp_cycles() for n in self.nodes])

    def get_vals(self):
        return [p.value for p in self.parameters]

    def get_min_vals(self):
        # get minimum possible folding values in the set
        return [p.possible_values[0] for p in self.parameters]

    def get_max_vals(self):
        # get maximum possible folding values in the set
        return [p.possible_values[-1] for p in self.parameters]

    def add_all_params_to_index_list(self):
        self.index_list = [x for x in range(len(self.parameters))]

    def set_values(self, values):
        for i in range(len(self.index_list)):
            self.parameters[self.index_list[i]].update_value(values[i])

    def apply_updates(self, final=False, filter=[]):
        # a
        for i in self.index_list:
            self.parameters[i].apply_value(final, filter)

    def assign_involved_nodes(self):
        self.nodes = []
        # for p in self.parameters:
        for i in range(len(self.index_list)):
            p = self.parameters[self.index_list[i]]
            self.nodes.append(p.node)

            # potentially have dependent nodes as well
            if len(p.dependant_nodes) > 0:
                for n in p.dependant_nodes:
                    self.nodes.append(n)

        self.nodes = list(set(self.nodes))  # make this unique


class Optimizer:
    def __init__(
        self,
        model,
        name,
        targets,
        hard_constraint_target=             "max_cycles",
        target_cycles_per_frame=            1,
        padding=                            0,
        maxfun_per_parameter=               100,
        fpgapart=                           "xc7z020clg400-1",
        parameters_to_apply=                ["SIMD", "PE", "ram_style", "resType","ram_style_thresholds"],
        penalize_hls_dwc_variant_use=       True,
        verbose=                            False,
        mvau_wwidth_max=                    1024,
        value_to_minimize_relaxation=       0.95,
        max_parameters_per_partition=       4,
        init_run=                           False,
        maxiter=                            150,
        accept = -                          0.5,
        pad_io_nodes =                      False,
        optimization_parameters =            ["SIMD", "PE", "ram_style", "resType","ram_style_thresholds"],
    ):
        self.params = None
        self.targets = targets
        self.updated_nodes = []
        self.param_indexes = []  # this might require insertion!!!
        self.param_ranges = []
        self.all_nodes = []
        self.target_cycles_per_frame = target_cycles_per_frame
        self.padding = padding
        self.mvau_wwidth_max = mvau_wwidth_max
        self.model = model
        self.pad_io_nodes = pad_io_nodes
        self.name = name
        self.fpgapart = fpgapart
        self.metrics = None
        self.init_run = init_run
        self.maxiter = maxiter
        self.accept = accept

        # 0-100, relax whether we MUST hit the required bounding value,
        # for example max_cycles
        self.value_to_minimize_relaxation = value_to_minimize_relaxation
        self.max_parameters_per_partition = max_parameters_per_partition
        self.maxfun_per_parameter = maxfun_per_parameter
        
        self.hard_constraint_target = hard_constraint_target
        self.parameters_to_apply = parameters_to_apply
        self.penalize_hls_dwc_variant_use = penalize_hls_dwc_variant_use
        self.verbose=verbose
        self.optimization_parameters = optimization_parameters

        # total number of nodes which got padded
        self.total_paddings = 0


    def cleanup_pass(self):
        # some corrections that may be necessary

        for node in self.model.graph.node:
            # SWG->MVAU should have the same stream width
            if node.op_type.startswith("ConvolutionInputGenerator"):
                node_inst = getCustomOp(node)
                mvau_or_vvau = self.model.find_consumer(node.output[0])
                if mvau_or_vvau.op_type.startswith("MVAU"):
                    mvau_inst = getCustomOp(mvau_or_vvau)
                    node_inst.set_nodeattr("SIMD",mvau_inst.get_nodeattr("SIMD"))


    def compute_hls_dwc_cost(self, model, nodes, lut_capacity, hls_dwc_cost_penalty=8):

        # Given a set of nodes and a model,
        # consider the stream widths between all adjacent nodes
        # and apply a cost penalty if the shapes mismatch relative
        # to the cost of introducing a DataWidthConverter

        # this heuristic is critical for preventing overuse of 
        # DWCs with enormous resource costs

        # hls_dwc_cost_penalty is a rough heuristic for how much
        # an HLS variant consumes in LUTs

        cost = 0
        for node in nodes:
            prod = model.find_producer(node.onnx_node.input[0])

            # check if this is not the first node of a model
            if prod is not None:
                output_name = prod.output[0]
                prod_inst = getCustomOp(prod)
                inWidth = prod_inst.get_outstream_width()
                outWidth = prod_inst.get_instream_width()

                n0_out_shape = prod_inst.get_folded_output_shape()

                # mvau has a special case with external memory
                # where we have to consider a different input
                if (
                    node.onnx_node.op_type.startswith("MVAU")
                    and node.get_nodeattr("mem_mode") == "external"
                ) or (node.onnx_node.op_type.startswith("StreamingConcat")):
                    # get input idx
                    in_idx = None
                    for idx, n_input in enumerate(node.onnx_node.input):
                        if output_name == n_input:
                            in_idx = idx
                    assert in_idx is not None, "Malformed model"
                    n1_in_shape = node.get_folded_input_shape(in_idx)
                else:
                    # use default folded input shape
                    n1_in_shape = node.get_folded_input_shape()


                # dwcs cannot be inserted between mvau/vvau and pool/swg 
                # so we only run it for other combinations
                if (not ((prod.name.startswith("ConvolutionInputGenerator") or prod.name.startswith("Pool")) and 
                    (node.onnx_node.name.startswith("Pool") or node.onnx_node.name.startswith("MVAU") or node.onnx_node.name.startswith("VVAU")))): 
                    n1_in_shape = node.get_folded_input_shape()

                    # check if we need a DWC
                    if  np.prod(n0_out_shape) != np.prod(n1_in_shape) or n0_out_shape[-1] != n1_in_shape[-1]:
                        # HLS DWC needed, expensive
                        if (max(inWidth, outWidth) % min(inWidth, outWidth) != 0) or (np.prod(n0_out_shape) != np.prod(n1_in_shape)):
                            cost += ((inWidth + outWidth) * hls_dwc_cost_penalty) / lut_capacity

                        # RTL DWC can be used cheaply
                        else:
                            cost += (inWidth + outWidth) / lut_capacity

            # extra cost penalizing large widths
          #  cost += ((opt.params.nodes[0].get_instream_width() * 4) / opt.targets["LUT"])
           # cost += ((opt.params.nodes[-1].get_outstream_width() * 4) / opt.targets["LUT"])
        return cost


    def cost_model(self, param_guess, opt):
        # the function used for determining how
        # 'good' a given folding configuration is
        # in respect to optimization targets
        # any heuristics to consider as effects
        # of folding on the effectiveness of the final
        # model should go here

        cost = 0

        # 1. apply the folding parameters
        opt.params.set_values(param_guess)
        opt.params.apply_updates(final=False, filter=self.parameters_to_apply)

        # 2. compute results
        cycles = opt.params.get_max_cycles()
        resources = self.get_resources(opt.params.nodes)
        metrics = {**{"max_cycles": cycles}, **resources}

        # 3. update cost based on all minimizable targets
        # the hard constraint (usually max_cycles) enforces
        # which target MUST be met.
        for value_to_minimize in opt.targets:
            if value_to_minimize != opt.hard_constraint_target:
                cost += metrics[value_to_minimize] / opt.targets[value_to_minimize]
            else:
                if metrics[value_to_minimize]*self.value_to_minimize_relaxation > (opt.targets[value_to_minimize]):
                    cost = np.inf     

        # 4. Add additional heuristic costs

        # 4.1 DWC heuristic to decrease the use of HLS DWCs
        # which can have massive LUT resource consumption
        # increases. All pairs are considered because
        # we optimize partitions left to right and consider
        # the DWC between a node and its left neighbor
        if self.penalize_hls_dwc_variant_use:
            cost += self.compute_hls_dwc_cost(opt.model, opt.params.nodes, opt.targets["LUT"])

        return cost

    def execute_minimizer(self, discrete_args, init_guess):
        wrapped_objective = Objective(
            self.cost_model,
            variables=discrete_args,
        )
        bounds = wrapped_objective.bounds

        if len(bounds) == 0:
            return np.array(init_guess)
        
        encoded_init_guess = wrapped_objective.encode((init_guess))
        fixed_args = tuple([self])

        optimal_args = scipy.optimize.dual_annealing(
            func=wrapped_objective,
            x0=encoded_init_guess,
            maxiter=self.maxiter,
            accept=self.accept,
            visit = 2.0,
            maxfun=self.maxfun_per_parameter * len(init_guess),
            # niter=self.optimizer_ites,
            # stepsize=self.stepsize,
            # T=self.temp,
            args=(fixed_args),
            bounds=bounds,
        )

        optimized_params = optimal_args.x
        optimized_params = np.array(wrapped_objective.decode(optimized_params))
        print("optimal params:", optimized_params)
        

        return optimized_params

    def optimize(
        self,
        partitions=2,
        initial_guess="max",
        max_nodes_in_partition=2,
        target_parameters=["SIMD", "PE"],
    ):
        
        # A single optimization pass across an entire model

        # initial guess can be "min" or "max"
        # min = least folding (makes sense when the hard constraint is resource use)
        # max = maximum folding (makes sense when the hard constraint is max_cycles)


        print("STARTED OPTIMIZER WITH PARAMS:")
        print("penalize_hls_dwc_variant_use: ",self.penalize_hls_dwc_variant_use)
        print("padding: ",self.padding)
        print("effort: ",self.maxfun_per_parameter)


        # 1. Split parameters into partitions to optimize locally

        # calculate number of partitions if not set to 1
        param_count = len(self.params.parameters)
        if param_count > self.max_parameters_per_partition and partitions != 1:
            partitions = param_count // self.max_parameters_per_partition

        if partitions == 1:
            self.params.add_all_params_to_index_list()

        indexes = self.params.index_list = [x for x in range(len(self.params.parameters))]

        if initial_guess == "min":
            init_guess = self.params.get_min_vals()
        elif initial_guess == "max":
            init_guess = self.params.get_max_vals()
        self.params.set_values(init_guess)

        self.params.apply_updates(filter=target_parameters)
        self.params.assign_involved_nodes()
        params = self.params.parameters

        # node-based partitioning

        partitions = 0
        old_node_index = 0
        index_partitions = []
        init_guess_partitions = []
        params_partitions = []

        tmp_index_partitions = []
        tmp_init_guess_partitions = []
        tmp_params_partitions = []

        i = 0
        nodes_in_partition = 1
        for param in params:
            if param.name in target_parameters:
                new_node_index = param.node_index

                if new_node_index != old_node_index:
                    nodes_in_partition += 1

                if nodes_in_partition > max_nodes_in_partition:
                    # store set and start a new one
                    if len(tmp_index_partitions) > 0:
                        index_partitions.append(tmp_index_partitions)
                        init_guess_partitions.append(tmp_init_guess_partitions)
                        params_partitions.append(tmp_params_partitions)
                        tmp_index_partitions = []
                        tmp_init_guess_partitions = []
                        tmp_params_partitions = []
                        partitions += 1
                        nodes_in_partition = 1
                if nodes_in_partition <= max_nodes_in_partition:
                    tmp_index_partitions.append(indexes[i])
                    tmp_init_guess_partitions.append(init_guess[i])
                    tmp_params_partitions.append(params[i])

                old_node_index = new_node_index
            i += 1

        # add remaining lefover tail partition
        if len(tmp_index_partitions) > 0:
            if len(tmp_index_partitions) > 0:
                index_partitions.append(tmp_index_partitions)
                init_guess_partitions.append(tmp_init_guess_partitions)
                params_partitions.append(tmp_params_partitions)
                partitions += 1


        # 2. Perform local optimization of partitions
        for p in range(partitions):

            
            # generate discrete argument list based on possible values
            # this is the input for the scipy minimizer
            discrete_args = []
            for arg in params_partitions[p]:
                discrete_args.append(GridVar(tuple(arg.possible_values)))

            # filter out parameters to the ones of the requested partition
            self.params.index_list = index_partitions[p]
            self.params.assign_involved_nodes()

            # fetch the respective initial list of parameters
            # it is very important that the initial guess is feasible
            # for the minimizer so that the cost_model call returns a non-infinity cost
            # otherwise the optimizer might give up believing there is no solution
            init_guess = init_guess_partitions[p]
            
            # an initial run to get resource consumption bounds
            if self.init_run:
                optimized_params = init_guess
            else:
                print("PARAMS IN PARTITION:")
                for param in params_partitions[p]:
                    print(param.name)
                optimized_params = self.execute_minimizer(discrete_args, init_guess)

            # apply final values, adjusting the model accordingly
            self.params.set_values(optimized_params)
            self.params.apply_updates(final=True, filter=target_parameters)

        # final surgery of the model
        self.cleanup_pass()

        print(f"final parameters, init_run={self.init_run}:")

        s = ""
        total_params = 0
        total_padding = 0
        for p in self.params.parameters:
            s += f"{p.node.onnx_node.name}, {p.name}, {p.value}\n"
            if p.name in ["SIMD","PE"]:
                total_params +=1
                p.used_bounding_parameter = p.bounding_value_for_each_possible_value[
                    p.possible_values.index(p.value)]

                if p.flag_padding():
                    total_padding+=1
            
        print(s)
        
        self.padding_result = f"{total_padding} / {total_params}"
        print("optimizer padding ratio: ",self.padding_result)
        #print("optimized param values: ",self.params.get_vals())
        for p in self.params.parameters:
            self.total_paddings += total_padding

    def get_resources(self, nodes):
        resources = {}
        for n in nodes:
            resources[n] = n.node_res_estimation(self.fpgapart)
        resources = aggregate_dict_keys(resources)
        
        return resources
    

    def get_all_resources(self):
        resources = {}
        for n in self.model.graph.node:
            node_inst = getCustomOp(n)
            resources[node_inst] = node_inst.node_res_estimation(self.fpgapart)
        resources = aggregate_dict_keys(resources)
        
        return resources

    def update_io_sizes(self):
        mw = getCustomOp(self.model.graph.node[0]).get_nodeattr("MW")
        mh = getCustomOp(self.model.graph.node[-1]).get_nodeattr("MH")
        x = np.zeros((1, mw), dtype=np.float32)
        y = np.zeros((1, mh), dtype=np.float32)
        self.model.set_initializer(self.model.graph.node[0].input[0], x)
        self.model.set_initializer(self.model.graph.node[-1].output[0], y)

    def update_values(self, values):
        self.updated_nodes = []
        for i in self.param_indexes:
            update = self.params[i].update_value(values[i])
            if update:
                if self.params[i].node not in self.updated_nodes:
                    self.updated_nodes.append(self.params[i].node)
                    self.params[i].perform_node_insertions_deletions()




    def construct_parameter_object(self, node, parameter_name, maximum_padding,whitelist, node_indx,arg_indx):
        node_inst = getCustomOp(node)
       
        possible_values = []
        bounding_values = []
        conv_exception = False
        generator_bound = 0
        stream_width_exception = False
        dependent_nodes = []
        weight_width = 0
        generator_bounding_values = []
       
        if node.op_type in ["MVAU_rtl","MVAU_hls","VVAU_hls","VVAU_rtl"] and parameter_name == "SIMD":
            # get prod
            weight_width = node_inst.get_weight_datatype().bitwidth()
            if node.op_type in ["VVAU_hls","VVAU_rtl"]:
                weight_width = 0 # we dont have the restriction for vvau
            stream_width_exception = True
            generator_node = None
            pooling_node = None
            prod_node = self.model.find_producer(node.input[0])
            if prod_node is not None:
                if prod_node.op_type in ["Pool_hls"]:
                    pooling_node = prod_node
                    prod_node_inst = getCustomOp(prod_node)
                    prod_node_inst.set_nodeattr("PE",1)
                    prod_prod_node = self.model.find_producer(prod_node.input[0])
                    if prod_prod_node is not None:
                        if prod_prod_node.op_type in ["ConvolutionInputGenerator_hls","ConvolutionInputGenerator_rtl"]:
                            generator_node = prod_prod_node
                            prod_node_inst = getCustomOp(prod_prod_node)
                            prod_node_inst.set_nodeattr("SIMD",1)

                elif prod_node.op_type in ["ConvolutionInputGenerator_hls","ConvolutionInputGenerator_rtl"]:
                    generator_node = prod_node
                    prod_node_inst = getCustomOp(prod_node)
                    prod_node_inst.set_nodeattr("SIMD",1)

            # get paddings for prod
            if generator_node is not None:
                # SWU-x-MVAU/VVAU chain, perform the tied padding generation
                # TODO: add the sophisticated stuff here later for bounding parameter generation
                folding_factors = None
                bounding_values = None
            else:
                # no tied nodes, simple pset generation with no producers
                folding_factors = None
                bounding_values = None

            if generator_node is not None:
                dependent_nodes.append(getCustomOp(generator_node))
                generator_node_inst = getCustomOp(generator_node)
                generator_bound = generator_node_inst.get_nodeattr("IFMChannels")
                kernel_size = np.prod(generator_node_inst.get_nodeattr("ConvKernelDim"))
                
            if pooling_node is not None:
                dependent_nodes.append(getCustomOp(pooling_node))

            if pooling_node is not None or generator_node is not None:
                conv_exception = True
    
        # simple cases
        if parameter_name in ["SIMD","PE"]:
            node_inst.set_nodeattr(parameter_name, 1)


        padding_internal = np.min([whitelist[parameter_name][node.op_type][0],maximum_padding])
        folding_internal = whitelist[parameter_name][node.op_type][1]

        bounding_parameter = whitelist[parameter_name][node.op_type][2]

        if len(possible_values) == 0:
            if parameter_name == "ram_style":
                possible_values = whitelist[parameter_name][node.op_type][3]
            elif parameter_name == "resType":
                possible_values = whitelist[parameter_name][node.op_type][3]
            elif parameter_name == "ram_style_thresholds":
                possible_values = whitelist[parameter_name][node.op_type][3]
            else:
                bound = node_inst.get_nodeattr(bounding_parameter)
                possible_values, bounding_values = allowed_divisors(bound, 1, padding_internal)            

                
                new_bounding_values = []
                new_possible_values = []

                # SWU -> MVAU / VVAU SIMD exception (conv layer)
                if conv_exception:
                    # fetch padding allowance of the generating SWU
                    swu_padding_internal = np.min([whitelist[parameter_name][generator_node.op_type][0],maximum_padding])

                    depthwise = generator_node_inst.get_nodeattr("depthwise")
                    M = generator_node_inst.get_nodeattr("M")
                    pw = generator_node_inst.get_nodeattr("parallel_window")

                    if pw:
                        mmv_out = M * kernel_size
                    else:
                        mmv_out = 1

                    if kernel_size == 1 or generator_node.op_type != "ConvolutionInputGenerator_rtl" or \
                            ((mmv_out > 1 or kernel_size == 1) and depthwise == 1):
                        allowed_non_max_simd = True
                    else:
                        allowed_non_max_simd = False
                    print(f"conv exception with {generator_node.op_type} as generator")
                    # for each possible padding of the SWU, generate new possible foldings of the downstream MVAU/VVAU SIMD
                    # recomputing the possible bounding parameter values if we do based on the kerneldim * numChannels params
                    for generator_padding in range(0,swu_padding_internal+1):
                        if node.op_type in ["MVAU_hls","MVAU_rtl"]:
                            new_mw_bound = (generator_padding + generator_bound) * kernel_size
                        else:
                            new_mw_bound = kernel_size

                        possible_values, bounding_values = allowed_divisors(new_mw_bound, 1, 0)
                        pairs = zip(possible_values, bounding_values)

                        for pair in pairs:
                            gen_yes = generator_node.op_type != "ConvolutionInputGenerator_rtl"
                            print(f"adding pair:,{pair[0]},{pair[1]},{generator_bound+generator_padding}")
                            print(f"{allowed_non_max_simd}, {depthwise},{kernel_size}, {gen_yes}, {pair[0] * weight_width <= self.mvau_wwidth_max}")
                            if (pair[0] * weight_width <= self.mvau_wwidth_max):
                               print("if 1")
                               if (pair[0] > (pair[1] / 1024)):
                                    print("if 2")
                                    if(allowed_non_max_simd and (generator_bound+generator_padding) % pair[0] == 0) or (not allowed_non_max_simd and (generator_bound+generator_padding) == pair[0]):
                                        print("if 3")
                                        if pair[0] not in new_possible_values:
                                            new_possible_values.append(pair[0])
                                            new_bounding_values.append(pair[1])
                                            generator_bounding_values.append(generator_bound+generator_padding)

                else:

                    pairs = zip(possible_values, bounding_values)

                    for pair in pairs:
                        if (((stream_width_exception and (pair[0] * weight_width) <= self.mvau_wwidth_max) or \
                            not stream_width_exception) and \
                            (pair[0] > (pair[1] / 1024))):

                            if pair[0] not in new_possible_values:
                                new_possible_values.append(pair[0])
                                new_bounding_values.append(pair[1])
                                
                possible_values = new_possible_values
                bounding_values = new_bounding_values

        if folding_internal:

            param = Parameter(
                node=node_inst,
                node_indx=node_indx,
                op_type=node.op_type,
                name=parameter_name,
                index=arg_indx,
                target_cycles_per_frame=self.target_cycles_per_frame,
                value=possible_values[-1],
                possible_values=possible_values,
                padding=padding_internal,
                model=self.model,
                default_bounding_parameter=(bounding_parameter, bound),
                bounding_value_for_each_possible_value=bounding_values,
                dependant_nodes=dependent_nodes,
                generator_bounding_value_for_each_possible_value=generator_bounding_values
            )
            return param
        else:
            return None



    def generate_parameter_set(self):

        # given a model, extract all optimizable parameters from it
        # as well as the possible values on these parameters
        # and the respective bounding parameter which might need to
        # be adjusted in case of padding

        model = self.model
        padding = self.padding

        whitelist = parameter_whitelist(self.padding)


        graph = model.graph
        pset = ParameterSet()

        # these ops use PE parallelism, up to a max value of NumChannels
        # print(arg_vals,virtual)

        pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            "StreamingMaxPool_hls",
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator* has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "DownSampler_hls",
            "FMPadding_hls",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_hls",
            "ConvolutionInputGenerator_rtl",
            "FMPadding_rtl",
        ]
        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]

        node_indx = 0
        arg_indx = 0
        

        for node in graph.node:
            
            
            if node.op_type == "StreamingDataWidthConverter":
                continue

            if not (is_hls_node(node) or is_rtl_node(node)):
                continue

            if self.pad_io_nodes is not True:
                if node_indx == 0 or node_indx == len(graph.node)-1:
                    # do not allow padding IO nodes
                    maximum_padding = 0
                else:
                    # allow any arbitrary padding amount for IO nodes
                    maximum_padding = 10000000

            node_indx += 1

            op_type = node.op_type
            node_inst = getCustomOp(node)

            self.all_nodes.append(node_inst)

            # test if SIMD,PE,resType,mem_type and mem_type_thresholds exists in node

            # for loop version since you cant list comprehend with our dict layout
            filtered_params = []
            for p in self.optimization_parameters:
                if p in whitelist:
                    if op_type in whitelist[p]:
                        filtered_params.append(p)

            # ignore SIMD until implemented in construction function
            #if op_type in ["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl"]:
            #    filtered_params.remove("SIMD")


            print("possible:")
            print(filtered_params)
            for parameter_name in filtered_params:
                print("trying to optimize:")
                print(node.name, parameter_name,node_indx)
                param = self.construct_parameter_object(
                    node, 
                    parameter_name,
                    maximum_padding,
                    whitelist,
                    node_indx,
                    arg_indx,
                    )
                
                if param is not None:
                    pset.parameters.append(param)
                    arg_indx+=1


            """
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                #node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)

                # special case with stream width for weights being a restriction
                max_simd = node_inst.get_nodeattr("MW")
                tmp_node = copy.deepcopy(node_inst)
                tmp_node.set_nodeattr("SIMD", max_simd)
                stream_width_bound = int(
                    self.mvau_wwidth_max / tmp_node.get_weight_datatype().bitwidth()
                )

                param_name = "SIMD"
                padding_internal = np.min([whitelist[param_name][op_type][0],maximum_padding])
                folding_internal = whitelist[param_name][op_type][1]

                possible_values, bounding_values = allowed_divisors(max_simd, 1, padding_internal)
                # SIMD has a restriction of SIMD >= MW / 1024
                # as well as a max stream width bound

                # we are overly conservative on the minimum here by using padding for the /1024 calc
                # in the rare case this introduces too much simd and restricts us, worth adjusting
                # to use the padding for the precise possible value

                # for SIMD, hls has restrictions on minimum and maximum stream
                # widths which we have to take into account

                producer_max_simd = max_simd
                if node_indx > 0:
                    swu_node = model.find_producer(node.input[0])
                    if swu_node is not None:
                        if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                            producer_max_simd = getCustomOp(swu_node).get_nodeattr("IFMChannels")
                            has_producer = True
                        elif swu_node.op_type.startswith("Pool"):
                            producer_max_simd = getCustomOp(swu_node).get_nodeattr("Channels")
                            has_producer = True
                        else:
                            has_producer = False
                    else:
                        has_producer = False


                new_bounding_values = []
                new_possible_values = []
                pairs = zip(possible_values, bounding_values)
                for pair in pairs:
                    if (pair[0] <= stream_width_bound) and (pair[0] > ((max_simd + padding_internal) / 1024) and ( (has_producer and producer_max_simd % pair[0] == 0) or has_producer is False)):
                        new_possible_values.append(pair[0])
                        new_bounding_values.append(pair[1])


                #assert True == False
                if folding_internal:
                    param = Parameter(
                        node=node_inst,
                        node_indx=node_indx,
                        op_type=op_type,
                        name=param_name,
                        index=arg_indx,
                        target_cycles_per_frame=self.target_cycles_per_frame,
                        value=new_possible_values[0],
                        possible_values=new_possible_values,
                        padding=padding_internal,
                        model=model,
                        default_bounding_parameter=("MW", max_simd),
                        bounding_value_for_each_possible_value=new_bounding_values,
                    )
                    pset.parameters.append(param)
                    arg_indx += 1

                    had_pooling_producer = False
                    had_swg_producer = False
                    # Detect any Pool node that might exist prior to an SWG and fold               
                    producer =  self.model.find_producer(node.input[0])
                    if producer is not None:
                        if producer.op_type.startswith("Pool"):
                            producer_inst = getCustomOp(producer)
                            producer_inst.set_nodeattr("PE",1)
                            swu_node = self.model.find_producer(producer.input[0])
                            had_pooling_producer = True
                        else:
                            swu_node = producer
                        if swu_node is not None:
                            if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                                had_swg_producer = True
                                swu_node_inst = getCustomOp(swu_node)

                        param.dependant_nodes = []
                        if had_pooling_producer:
                            param.dependant_nodes.append(producer_inst)
                        if had_swg_producer:
                            param.dependant_nodes.append(swu_node_inst)


            if op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
          
                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
                # cyc = node_inst.get_exp_cycles()
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    # self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                    
                    param_name = "SIMD"
                    padding_internal = np.min([whitelist[param_name][op_type][0],maximum_padding])
                    folding_internal = whitelist[param_name][op_type][1]

                    possible_values, bounding_values = allowed_divisors(max_simd, 2, padding_internal)
                    # in case the node is the first or last node, we override the padding

                    if folding_internal:
                        param = Parameter(
                            node=node_inst,
                            node_indx=node_indx,
                            op_type=op_type,
                            name=param_name,
                            index=arg_indx,
                            target_cycles_per_frame=self.target_cycles_per_frame,
                            value=possible_values[0],
                            possible_values=possible_values,
                            padding=padding_internal,
                            model=model,
                            default_bounding_parameter=("Kernel", node_inst.get_nodeattr("Kernel")),
                            bounding_value_for_each_possible_value=bounding_values,
                        )
                        pset.parameters.append(param)
                        arg_indx += 1


                        had_pooling_producer = False
                        had_swg_producer = False
                        # Detect any Pool node that might exist prior to an SWG and fold               
                        producer =  self.model.find_producer(node.input[0])
                        if producer is not None:
                            if producer.op_type.startswith("Pool"):
                                producer_inst = getCustomOp(producer)
                                producer_inst.set_nodeattr("PE",1)
                                swu_node = self.model.find_producer(producer.input[0])
                                had_pooling_producer = True 
                            else:
                                swu_node = producer
                            if swu_node is not None:
                                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                                    had_swg_producer = True
                                    swu_node_inst = getCustomOp(swu_node)

                            param.dependant_nodes = []
                            if had_pooling_producer:
                                param.dependant_nodes.append(producer_inst)
                            if had_swg_producer:
                                param.dependant_nodes.append(swu_node_inst)
            """
        self.params = pset



def insert_and_size_fifos(model,fpgapart,consider_dwc_costs, auto_fifo_strategy):
    if not consider_dwc_costs:
        model = model.transform(InsertDWC())

    cfg = DataflowBuildConfig(
        output_dir = "",
        auto_fifo_depths = True,
        split_large_fifos = True,
        auto_fifo_strategy = auto_fifo_strategy,
        folding_config_file = None,
        synth_clk_period_ns=5.0,
        generate_outputs = ["estimate_reports"],
        board=fpgapart,
    )
    model = set_fifo_depths(model,cfg)
    return model

class SetFolding(Transformation):

    """
    Attempt to set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame target_cycles_per_frame. For each
    HLSCustomOp node type, the attribute may vary but is typically one of {PE, SIMD},
    and has a certain allowed-maximum value and divisibility constraints,
    which SetFolding will take into account.

    If padding is specified >0, an optimization algorithm based on a target function
    and an optimization objective is employed with folding factors restrictions
    drastically relaxed by adding padding to all relevant nodes if this helps
    achieve the optimal folding. Special padding & cropping DWCs are also inserted where
    necessary.

    In the returned model, each node's
    cycles_estimate attribute will be set to its estimated number of cycles.

    """

    def __init__(
        self,
        target_cycles_per_frame=1000,
        mvau_wwidth_max=1024,
        two_pass_relaxation=True,
        style="optimizer",
        folding_maximum_padding=0,
        folding_max_attempts=1,
        platform="Pynq-Z1",
        folding_effort=250,
        enable_folding_dwc_heuristic=1,
        devices=1,
        verbose=False,
    ):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation
        self.max_attempts = folding_max_attempts
        self.padding = folding_maximum_padding
        self.devices = devices
        self.platform = platform
        self.fpgapart = part_map[self.platform]
        self.verbose=verbose
        self.pad_io_nodes = False
        # either "naive" or "optimizer"
        self.style = style

        # maximum function calls / parameter
        # recommended in the range of 50-200 depending on the network size
        # and how long the user is willing to wait for this step
        # ~20 parameters with <30 possible values per parameter @ 200 effort = <30s
        self.effort = folding_effort

        # self.optimization_parameters = ["SIMD","PE"]
        self.optimization_parameters = ["SIMD", "PE", "ram_style", "resType","ram_style_thresholds"]
        self.hard_constraint_target = "max_cycles"
        self.optimize_folding = True
        self.optimize_resource_types = False
        self.insert_dwcs = False
        self.consider_dwc_costs = True

        # WARNING: if set to true, this flag
        # can result in an enormous increase in 
        # the time it takes to run this transformation
        # relative to the time it takes to run
        # set_fifo_depths times (folding_max_attempts-1)
        # Recommended to only run if analytic FIFO sizing
        # is also enabled (experimental feature)
        self.consider_fifo_costs = False
        self.auto_fifo_strategy = "characterize_analytic"

        if enable_folding_dwc_heuristic == 1:
            self.penalize_hls_dwc_variant_use = True
        else:
            self.penalize_hls_dwc_variant_use = False
        
        self.target_resources = ["LUT","BRAM_18K","DSP","URAM"]



    def apply_optimized_folding(self, model):
        """
        Optimization algorithm-based folding transformation 
        using an iterative optimization algorithm and a target function
        to find optimal folding values for each node in the FINN graph,
        by default minimizing resource consumption while making sure to meet
        the target max_cycles (throughput) rate
        """

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        targets = {}
        targets["max_cycles"] = self.target_cycles_per_frame
        current_throughput_target = self.target_cycles_per_frame

        # fetch all parameters and bounds from the model by
        # running the optimizer once without minimizing cost

        init_model = copy.deepcopy(model)
        opt1 = Optimizer(
            init_model,
            "defaultOPT_for_parameter_extraction",
            targets,
            self.hard_constraint_target,
            target_cycles_per_frame=self.target_cycles_per_frame,
            padding=0,
            fpgapart=self.fpgapart,
            maxfun_per_parameter=self.effort,
            parameters_to_apply=["SIMD", "PE"],
            penalize_hls_dwc_variant_use=self.penalize_hls_dwc_variant_use,
            verbose=self.verbose,
            mvau_wwidth_max=self.mvau_wwidth_max,
            init_run=True,
            pad_io_nodes=self.pad_io_nodes,
            optimization_parameters=self.optimization_parameters
        )

        opt1.targets = targets
        opt1.generate_parameter_set()  # generate full param list
        param_set_default = opt1.params

        param_values_min = param_set_default.get_min_vals()
        param_values_max = param_set_default.get_max_vals()

        param_set_default.add_all_params_to_index_list()

        # create copies of the minimum and maximum parameters
        # for folding to use as upper and lower bounds for
        # optimization

        param_set_min = copy.deepcopy(param_set_default)
        param_set_min.set_values(param_values_min)

        param_set_max = copy.deepcopy(param_set_default)
        param_set_max.set_values(param_values_max)

        # run once to initialize all the lists and objects
        param_set_min.apply_updates(self.optimization_parameters)
        param_set_max.apply_updates(self.optimization_parameters)


        param_set_min.assign_involved_nodes()
        param_set_max.assign_involved_nodes()

       
        # assign maximum throughput achievable
        opt1.optimize(max_nodes_in_partition=1, target_parameters=["SIMD", "PE"])
        init_model = init_model.transform(AnnotateCycles())
        maximum_achievable_throughput = init_model.analysis(dataflow_performance)["max_cycles"]

       # assert True==False
        limits = DEFAULT_RES_LIMITS
        self.max_luts = limits[0] * sum(
            [r["LUT"] for r in platforms[self.platform](self.devices).resource_count_dict.values()]
        )
        self.max_bram = limits[2] * sum(
            [
                r["BRAM_18K"]
                for r in platforms[self.platform](self.devices).resource_count_dict.values()
            ]
        )
        self.max_uram = limits[3] * sum(
            [r["URAM"] for r in platforms[self.platform](self.devices).resource_count_dict.values()]
        )
        self.max_dsp = limits[4] * sum(
            [r["DSP"] for r in platforms[self.platform](self.devices).resource_count_dict.values()]
        )

        targets["LUT"] = max(self.max_luts, 0.001)
        targets["BRAM_18K"] = max(self.max_bram, 0.001)
        targets["DSP"] = max(self.max_dsp, 0.001)
        targets["URAM"] = max(self.max_uram, 0.001)

        opt2 = Optimizer(
            model,
            "padded OPT",
            targets,
            self.hard_constraint_target,
            target_cycles_per_frame=current_throughput_target,
            padding=self.padding,
            fpgapart=self.fpgapart,
            maxfun_per_parameter=self.effort,
            parameters_to_apply=self.optimization_parameters,
            penalize_hls_dwc_variant_use=self.penalize_hls_dwc_variant_use,
            verbose=self.verbose,
            mvau_wwidth_max=self.mvau_wwidth_max,
            init_run=False,
            pad_io_nodes=self.pad_io_nodes,
            optimization_parameters=self.optimization_parameters,
        )

        opt2.targets = targets
        opt2.generate_parameter_set()  # generate full param list

        # First pass which deals with folding factors only

        optimization_attempts = 0
        last_limited = False
        fifos_in_the_loop = True
        last_successful_throughput_target = self.target_cycles_per_frame
        
        current_step = 1
        min_step = 0.05

        opt2_tmp = copy.deepcopy(opt2)
        last_good_model = copy.deepcopy(opt2.model)

        # first pass
        print("entering global optimization passes")
        while current_step > min_step and optimization_attempts < self.max_attempts:
            targets["max_cycles"] = current_throughput_target
            opt2 = copy.deepcopy(opt2_tmp)
            opt2.targets = targets
            opt2.generate_parameter_set()  # generate full param list
            print(f'current_step: {current_step}, max_cycles: {targets["max_cycles"]}, attempts:{optimization_attempts}')
            # dont optimize if throughput request is impossible to meet


            opt2.target_cycles_per_frame=current_throughput_target
            if self.optimize_folding is True:
                opt2.optimize(max_nodes_in_partition=3, target_parameters=["SIMD", "PE"])

            # Second pass which adjusts ram style for memory and resource types for compute
            if self.optimize_resource_types is True:
                opt2.optimize(
                    max_nodes_in_partition=min(len(model.graph.node), 8),
                    target_parameters=["ram_style", "resType","ram_style_thresholds"],
                )

            # generate extra model with fifos and dwcs for the final estimate
                
            if self.consider_dwc_costs:
                model = model.transform(InsertDWC())
                model = model.transform(SpecializeLayers(self.fpgapart))
            
            if self.consider_fifo_costs:
                model = insert_and_size_fifos(model,self.fpgapart,
                                              self.consider_dwc_costs, 
                                              self.auto_fifo_strategy)
                model = model.transform(SpecializeLayers(self.fpgapart))

            resources = {}
            for n in opt2.model.graph.node:
                node_inst = getCustomOp(n)
                resources[node_inst] = node_inst.node_res_estimation(self.fpgapart)
            metrics = aggregate_dict_keys(resources)

            # extract costs
            overshot = False
            for resource in self.target_resources:
                if metrics[resource] > targets[resource]:
                    print(f"{resource}: {metrics[resource]} > {targets[resource]}")
                    overshot = True

            if overshot:
                # if we overshot, we try again, but with half the step size
                print(f"overshot, new target: {current_throughput_target}")
                print(f"step decreasing from {current_step} to {current_step/2}")
                print(f"target changing from {current_throughput_target} to {int(last_successful_throughput_target - last_successful_throughput_target*(current_step/2))} by decreasing on {last_successful_throughput_target} by a lower step")                
                current_step /= 2
                current_throughput_target = int(last_successful_throughput_target - last_successful_throughput_target*current_step)
                
            else:
                print(f"did not overshoot, still halving step size and repeating")
                if last_limited:
                    current_step /= 2

                else:
                    for resource in self.target_resources:
                        budget_left = 1 - (metrics[resource] / targets[resource])
                        print(f"budget: {budget_left} from {metrics[resource]} / {targets[resource]} ratio for {resource}")
                        current_step = min(current_step,budget_left)
                        print(f"new step: {current_step}")
      
                new_throughput = int(last_successful_throughput_target - last_successful_throughput_target*current_step)
                print(f"did NOT overshoot, new target: {current_throughput_target} from {last_successful_throughput_target}")
                last_good_model = copy.deepcopy(opt2.model)
                last_successful_throughput_target = copy.copy(current_throughput_target)
                current_throughput_target = new_throughput

            if current_throughput_target < maximum_achievable_throughput:
                print("requested beyond maximal folding, limiting")
                last_limited = True
                
                current_throughput_target = maximum_achievable_throughput
            else:
                last_limited = False
            optimization_attempts += 1

        model = last_good_model

        print("optimizer FINAL padding ratio: ",opt2.padding_result)

        if self.insert_dwcs:
            # In case future steps do not insert DWCs
            model = model.transform(InsertDWC())
            model = model.transform(SpecializeLayers(self.fpgapart))

        # necessary final transformation
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())


        # perform input and output tensor shape adjustment
        # this is only going to have effects if padding was performed


        if self.pad_io_nodes:
            input_mw_padded = getCustomOp(model.graph.node[0]).get_normal_input_shape()
            output_mh_padded = getCustomOp(model.graph.node[-1]).get_normal_output_shape()
            x = np.zeros(input_mw_padded, dtype=np.float32)
            y = np.zeros(output_mh_padded, dtype=np.float32)

            input_name = model.graph.input[0].name
            output_name = model.graph.output[0].name

            if len(model.graph.input) != 0:
                model.graph.input.remove(model.graph.input[0])
            input_x = helper.make_tensor_value_info(
                model.graph.node[0].input[0], TensorProto.FLOAT, [*input_mw_padded]
            )
            model.graph.input.append(input_x)

            if len(model.graph.output) != 0:
                model.graph.output.remove(model.graph.output[0])
            output_y = helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [*output_mh_padded]
            )
            model.graph.output.append(output_y)

        return (model, False)



    def divisors(self,num):
        for x in range(1, num + 1):
            if (num % x) == 0:
                yield x

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in self.divisors(max_val):
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
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            "StreamingMaxPool_hls",
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator* has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "DownSampler_hls",
            "FMPadding_hls",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_hls",
            "ConvolutionInputGenerator_rtl",
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
                for simd_val in self.divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame and simd_val > (max_simd / 1024):
                        # finish if target met and simd value is not too low
                        break
                    if (
                        node_inst.get_weight_datatype().bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in pe_ops:
                max_pe = node_inst.get_nodeattr("NumChannels")
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
                        padding=0,
                    )
                )

        # necessary final transforms
        if self.insert_dwcs:
            model.transform(InsertDWC())

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        return (model, False)

    def apply(self, model):
        if self.style == "naive":
            return self.apply_naive_folding(model)
        else:
            return self.apply_optimized_folding(model)
