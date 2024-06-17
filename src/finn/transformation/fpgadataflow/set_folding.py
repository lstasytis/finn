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

import numpy as np
import warnings
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
import copy
from finn.analysis.fpgadataflow.op_and_param_counts import (
    aggregate_dict_keys,
    op_and_param_counts,
)

from onnx import TensorProto, helper

from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)

import scipy
import random
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from qonnx.custom_op.registry import getCustomOp
from wrapdisc import Objective
from wrapdisc.var import ChoiceVar, GridVar, QrandintVar, QuniformVar, RandintVar, UniformVar


class ParameterSet():
    def __init__(self,name="Parameter Set"):
        self.name = name
        self.parameters = []
        self.index_list = []
        self.nodes = []
        self.nodes_cached = []



    def get_vals(self):
        return [p.value for p in self.parameters]

    def get_min_vals(self):
        return  [p.possible_values[0] for p in self.parameters]


    def get_max_vals(self):
        return  [p.possible_values[-1] for p in self.parameters]



    def add_all_params_to_index_list(self):
        self.index_list = [x for x in range(len(self.parameters))]
        #print("new index list:", self.index_list)

    def set_values(self,values):
       # print("setting values")
        #print(values)
        #print(self.index_list)
        #assert len(values) == len(self.index_list)

        for i in range(len(self.index_list)):
            self.parameters[self.index_list[i]].update_value(values[i])


    def apply_updates(self,final=False):
        for i in self.index_list:
            self.parameters[i].apply_value(final)

            
    def node_cache_store(self):
        self.nodes_cached = copy.deepcopy(self.nodes)

    def node_cache_load(self):
        self.nodes = self.nodes_cached

    def assign_involved_nodes(self):
        self.nodes = []
        #for p in self.parameters:
        for i in range(len(self.index_list)):
            p = self.parameters[self.index_list[i]]
            self.nodes.append(p.node)

            # potentially have dependent nodes as well
            if p.dependant_node is not None:
                self.nodes.append(p.dependant_node)

        self.nodes = list(set(self.nodes)) # make this unique




class Parameter():
    def __init__(self,
                 node = None,
                 node_indx = None,
                 op_type = None,
                 name = None,
                 index = None,
                 value = None,
                 possible_values = [],
                 padding = 0,
                 model = None,
                 default_bounding_parameter = None,
                 bounding_value_for_each_possible_value = []
                ):
        
        self.node = node
        self.node_index = node_indx -1
        self.op_type = op_type
        self.name = name
        self.index = index
        self.value = value
        self.updated = True
        self.possible_values = possible_values
        self.bounding_value_for_each_possible_value = bounding_value_for_each_possible_value
        self.dependant_node = None
        self.model = model
        self.default_bounding_parameter=default_bounding_parameter


    def update_value(self,value):
        if self.value == value:
            self.updated = False
        else:
            self.value = value
            self.updated = True

    def apply_value(self,final=False):

        if self.updated:

            # All exceptions, transformations, node insertions and deletions happen here
            if self.name == "SIMD" and self.op_type in ["VVAU_hls", "VVAU_rtl"]: # depthwise exception

                pe = self.node.get_nodeattr("PE")
                max_pe = self.node.get_nodeattr("PE")
                if pe == max_pe:
                    max_simd = np.prod(self.node.get_nodeattr("Kernel"))
                    self.possible_values,bounding_values = allowed_divisors(max_simd,self.padding)
                    if self.value in self.possible_values:
                        self.value = self.value
                        self.node.set_nodeattr(self.name, self.value)

                        swu_node = self.model.find_producer(self.node.input[0])

                        if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                            swu_node_inst = getCustomOp(swu_node)

                            swu_node_inst.set_nodeattr("SIMD", pe)
                            # enable parallel_window mode of RTL SWG if needed
                            if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                                if self.op_type.startswith("VVAU") and self.value > 1:
                                    swu_node_inst.set_nodeattr("parallel_window", 1)
                               # else:
                                #    swu_node_inst.set_nodeattr("parallel_window", 0)
                        else:
                            if self.op_type in ["VVAU_hls", "VVAU_rtl"]:
                                ksize = np.prod(self.node.get_nodeattr("Kernel"))
                            elif self.op_type == "Pool_hls":
                                ksize = self.node.get_nodeattr("KernelSize")
                            else:
                                raise Exception("Undefined edge case for %s" % self.op_type)
                            if ksize != 1:  # pointwise vvau/pool lack a SWU
                                raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
                        
                        self.dependant_node = swu_node


            if self.name == "SIMD" and self.op_type == "ConvolutionInputGenerator_rtl":

                self.node.set_nodeattr(self.name, self.value)

                if self.value == self.possible_values[-1]:
                    self.node.set_nodeattr("parallel_window",1)
                else:
                    self.node.set_nodeattr("parallel_window",0)


            else:
                self.node.set_nodeattr(self.name, self.value)


            bounding_val = self.bounding_value_for_each_possible_value[self.possible_values.index(self.value)]
            print(f"new bounding val (non-final) for {self.default_bounding_parameter[0]}: ",bounding_val)
            self.node.set_nodeattr(self.default_bounding_parameter[0],bounding_val)


            
        # after the optimization routines, when the final values are being applied,
        # additionally update any bounding parameters such as MW and MH to introduce
        # padding if necessary to support more folding factors.
        if final:
            # print(self.possible_values,self.bounding_value_for_each_possible_value)

            bounding_val = self.bounding_value_for_each_possible_value[self.possible_values.index(self.value)]
            print(f"new bounding val (FINAL) for {self.default_bounding_parameter[0]}: ",bounding_val)
            self.node.set_nodeattr(self.default_bounding_parameter[0],bounding_val)

            print(self.op_type)
            print(f'node index: {self.node_index}')
            # update the proto tensors

            # input shape adjustment
            if self.default_bounding_parameter[0] in ["MW", "NumChannels", "IFMChannels", "Labels"]:
                #x = np.zeros((1,bounding_val),dtype=np.float32)
                #if self.node_index != 0:
                #    self.model.set_initializer(self.model.graph.node[self.node_index].input[0],x) 
                
                new_shape = getCustomOp(self.model.graph.node[self.node_index]).get_normal_input_shape()
                self.model.set_tensor_shape(self.model.graph.node[self.node_index].input[0],new_shape)
                         
             
               # else: 
                    # delete original output and replace
                #    self.model.graph.input.remove(self.model.graph.input[0])
                #    input_x = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, bounding_val])
                #    self.model.graph.input.append(input_x)
                #    self.model.set_initializer("inp", x)

            # output shape adjustment
            if self.default_bounding_parameter[0] in ["MH", "IFMChannels", "Channels", "NumChannels"]:
                #y = np.zeros((1,bounding_val),dtype=np.float32)

                #if self.node_index < len(self.model.graph.node)-1:
                #    self.model.set_initializer(self.model.graph.node[self.node_index].output[0],y) 
                new_shape = getCustomOp(self.model.graph.node[self.node_index]).get_normal_output_shape()
                self.model.set_tensor_shape(self.model.graph.node[self.node_index].output[0],new_shape)
                    
               # else:
                    # delete original output and replace
                #    self.model.graph.output.remove(self.model.graph.output[0])
                #    output_y = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, bounding_val])
                #    self.model.graph.output.append(output_y)
                 #   self.model.set_initializer("outp", y)

                #    self.model.set_initializer(f"outp",y)
                  #  self.model.graph.node[-1].output[0] = y

                #model.set_tensor_shape("inp",input_mw_naive)

            if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
                # also update weight matrix
                # if its an mvau

                mw = self.node.get_nodeattr("MW")
                mh = self.node.get_nodeattr("MH")

                # proto W tensor
                W = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])

                # update weight matrix of the MVAU due to a padding or cropping changing the 
                if (mw,mh) != W.shape:
                    wdt = self.model.get_tensor_datatype(self.model.graph.node[self.node_index].input[1])
                    #W_new = np.zeros(wdt.min(), wdt.max() + 1, size=(mw, mh))

                    W_new = gen_finn_dt_tensor(wdt, (mw,mh))
                    W_new[...] = 0

                    W_new[:min(mw,W.shape[0]),:min(mh,W.shape[1])] = W[:min(mw,W.shape[0]),:min(mh,W.shape[1])]
                    self.model.set_initializer(self.model.graph.node[self.node_index].input[1],W_new)

                    print(f"updated weight matrix shape from {W.shape} to {W_new.shape}")

                # proto T tensor
                if len(self.model.graph.node[self.node_index].input) == 3:
                    T = self.model.get_initializer(self.model.graph.node[self.node_index].input[2])

                    adt = self.model.get_tensor_datatype(self.model.graph.node[self.node_index].input[2])
                    #W_new = np.zeros(wdt.min(), wdt.max() + 1, size=(mw, mh))

                    T_new = gen_finn_dt_tensor(adt, (mh,T.shape[1]))
                    T_new[...] = 0

                    T_new[:min(mh,T.shape[0]),:] = T[:min(mh,T.shape[0]),:]
                    self.model.set_initializer(self.model.graph.node[self.node_index].input[2],T_new)

            if self.op_type in ["Thresholding_hls", "Thresholding_rtl"]:

                mh = self.node.get_nodeattr("NumChannels")
                # thresholding nodes have a weight matrix which needs to be
                # adjusted if padding or cropping were introduced
                T = self.model.get_initializer(self.model.graph.node[self.node_index].input[1])
                adt = self.model.get_tensor_datatype(self.model.graph.node[self.node_index].input[1])
                T_new = gen_finn_dt_tensor(adt, (mh,T.shape[1]))
                T_new[...] = 0

                T_new[:min(mh,T.shape[0]),:] = T[:min(mh,T.shape[0]),:]
                self.model.set_initializer(self.model.graph.node[self.node_index].input[1],T_new)


        if self.updated:
            return True
        else:
            return False




    def get_resources(self):

        # get own and add dependent node if necessary
        pass

    def get_cycles(self):
        own_cycles = self.node.get_exp_cycles()

        # in case we have to tie in a DW input node, consider it as well
        if self.dependant_node is not None:
            own_cycles  = min(own_cycles,self.dependant_node.get_exp_cycles())

        return 


def allowed_divisors(cap,max_padding_count=0):
    # compute all possible folding factors for a given
    # upper bound variable

    #max_padding_count allows generating values with the assumption
    # that the bounding variable could be padded by up to that many
    # elements, which dramatically increases the possible folding
    # parameters with even a small amount of extra values

    all_divs = []
    all_bounding_values = []
    print(cap,max_padding_count)
    for i in range(cap,cap+max_padding_count+1):
        for x in range(1, i+1):
            if (i % x) == 0:
                if x not in all_divs and x <= cap:
                    all_divs.append(x)
                    all_bounding_values.append(i)
           
    all_divs,all_bounding_values = zip(*sorted(zip(all_divs, all_bounding_values)))

    return all_divs, all_bounding_values


def get_max_cycles(nodes):
    cycles = []
    for n in nodes:
        cycles.append(n.get_exp_cycles())
    return max(cycles)

def get_resources(nodes):
    resources = {}
    for n in nodes:
        resources[n] = n.node_res_estimation()

    resources = aggregate_dict_keys(resources)
    return resources









class Optimizer():
    def __init__(self,model,name,targets,hard_constraint_target = None, padding = 0,maxfun_per_parameter= 200):
        self.params = None
        self.targets = targets
        self.updated_nodes = []
        self.param_indexes = [] # this might require insertion!!!
        self.param_ranges = []
        self.updated_nodes = []
        self.all_nodes = []
        self.padding = padding
        self.mvau_wwidth_max = 52
        self.model = model
        self.name = name    
        self.maxiter=100 
        self.accept=-10.0
        self.maxfun_per_parameter = maxfun_per_parameter
        self.max_parameters_per_partition = 2

        #self.node_costs

        self.hard_constraint_target = hard_constraint_target


    def update_io_sizes(self):
        mw = getCustomOp(self.model.graph.node[0]).get_nodeattr("MW")
        mh = getCustomOp(self.model.graph.node[-1]).get_nodeattr("MH")
        x = np.zeros((1,mw),dtype=np.float32)
        y = np.zeros((1,mh),dtype=np.float32)
        self.model.set_initializer(self.model.graph.node[0].input[0], x)
        self.model.set_initializer(self.model.graph.node[-1].output[0], y)
        

    def execute_target_function(self, param_guess,opt):
        opt.params.set_values(param_guess)
        opt.params.apply_updates(final=False)
        
        print(f"working on nodes: {opt.params.nodes}")
        cycles = get_max_cycles(opt.params.nodes)
        resources = get_resources(opt.params.nodes)
        metrics = {**{"max_cycles":cycles}, **resources}

        cost = 0

        for value_to_minimize in opt.targets:
            if value_to_minimize != opt.hard_constraint_target:
                print(f"{value_to_minimize} value ratio to target: {metrics[value_to_minimize]/opt.targets[value_to_minimize]} ({metrics[value_to_minimize]}/{opt.targets[value_to_minimize]})")
                cost += metrics[value_to_minimize] / opt.targets[value_to_minimize]
            else:
                if metrics[value_to_minimize] > opt.targets[value_to_minimize]:
                    print(f"{value_to_minimize} value {metrics[value_to_minimize]} failed to meet  target {opt.targets[value_to_minimize]}")
                    cost = np.inf
        print(f"guess: [{param_guess}] final cost: {cost}")
        return cost


    def execute_minimizer(self, discrete_args, init_guess):

        wrapped_objective = Objective(
                self.execute_target_function,
                variables=discrete_args,
            )
        bounds = wrapped_objective.bounds

        
        encoded_init_guess = wrapped_objective.encode((init_guess))
        fixed_args = tuple([self])

        optimal_args = scipy.optimize.dual_annealing(
                func=wrapped_objective,
                x0=encoded_init_guess,
                maxiter=self.maxiter,
                accept=self.accept,
                maxfun=self.maxfun_per_parameter * len(init_guess),
                #niter=self.optimizer_ites,
                #stepsize=self.stepsize,
                #T=self.temp,
                args=(fixed_args),
                bounds=bounds)  

        #cost = execute_target_function(param_guess,opt1)
        optimized_params = optimal_args.x
        optimized_params = np.array(wrapped_objective.decode(optimized_params))
        # print("optimized params:",optimized_params)
        print("optimal params:", optimized_params)

        return optimized_params

    def optimize(self,partitions=1, initial_guess="max"):
        
        # initial guess can be "min" or "max"
        # min = least folding
        # max = maximum folding

        param_count = len(self.params.parameters)
        if param_count > self.max_parameters_per_partition:
            partitions = param_count // self.max_parameters_per_partition

       
        if partitions == 1:
            self.params.add_all_params_to_index_list()

        indexes = self.params.index_list = [x for x in range(len(self.params.parameters))]

        if initial_guess == "min":
            init_guess = self.params.get_min_vals()
        elif initial_guess == "max":
            init_guess = self.params.get_max_vals()
        self.params.set_values(init_guess)
        
        self.params.apply_updates()
        self.params.assign_involved_nodes()
        self.params.node_cache_store()
        
        params = self.params.parameters

        index_partitions = np.array_split(np.array(indexes),partitions)

        init_guess_partitions = np.array_split(np.array(init_guess),partitions)

        params_partitions = np.array_split(np.array(params),partitions)

        print(index_partitions)

        for p in range(partitions):
            print("param part:")
            print(params_partitions[p])
            index_partition = index_partitions[p]
            print(index_partition)

            param_part = []
            for i in index_partition:
                param_part.append(params[i])

            self.params.index_list = index_partition


            discrete_args = []
            for arg in params_partitions[p]:
                discrete_args.append(GridVar(tuple(arg.possible_values)))
                print(f"adding: {arg.possible_values} ")

            # get node partition
            self.params.index_list = index_partition
            self.params.assign_involved_nodes()

            init_guess = init_guess_partitions[p]
            optimized_params = self.execute_minimizer(discrete_args, init_guess)

            self.params.set_values(optimized_params)
            self.params.apply_updates(final=True)
            

            # self.model = self.model.transform(AnnotateCycles())
        print("final parameters:")
        print(self.params.parameters)
        print(self.params.get_vals())
        #assert True==False



    def update_values(self,values):

        self.updated_nodes = []
        for i in self.param_indexes:
            update = self.params[i].update_value(values[i])
            if update:
                if self.params[i].node not in self.updated_nodes:
                    self.updated_nodes.append(self.params[i].node)
                    self.params[i].perform_node_insertions_deletions()



    def get_params(self):
        
        model = self.model
        padding = self.padding
        mvau_wwidth_max = self.mvau_wwidth_max

        graph = model.graph
        pset = ParameterSet()

        # these ops use PE parallelism, up to a max value of NumChannels
        #print(arg_vals,virtual)

        pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
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
        #print(f"argvals: {arg_vals}")
        #print(len(graph.node))
        #print(get)
        for node in graph.node:
        # print(node.name,arg_indx)
            node_indx+=1
            if node.op_type == "StreamingDataWidthConverter":
                continue

            #if get == False:
            #    print(f"setting node {node_indx} arg {arg_indx} to {arg_vals[arg_indx]}")
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)

                # special case with stream width for weights being a restriction
                max_simd = node_inst.get_nodeattr("MW")
                tmp_node = copy.deepcopy(node_inst)
                tmp_node.set_nodeattr("SIMD",max_simd)
                stream_width_bound = int(mvau_wwidth_max /  tmp_node.get_weight_datatype().bitwidth())
                possible_values,bounding_values = allowed_divisors(max_simd,padding)
                # SIMD has a restriction of SIMD >= MW / 1024
                # as well as a max stream width bound

                # we are overly conservative on the minimum here by using padding for the /1024 calc
                # in the rare case this introduces too much simd and restricts us, worth adjusting
                # to use the padding for the precise possible value
               

                # for SIMD, hls has restrictions on minimum and maximum stream widths which we have to take into account
                new_bounding_values = []
                new_possible_values = []
                pairs = zip(possible_values,bounding_values)
                for pair in pairs:
                    if pair[0] <= stream_width_bound and pair[0]  > ((max_simd+padding) / 1024) :
                        new_possible_values.append(pair[0])
                        new_bounding_values.append(pair[1])
                

                #print(possible_values)

                # padding is 0 here because we have a hard stream width-defined limit on PE. So we can only add padding to 
               # node=node_inst,
               # node_indx=node_indx,
               # op_type=op_type,
               # name="SIMD",
               # index = arg_indx,
               # value=possible_values[0],
               # possible_values=possible_values,
               # padding=padding,
               # model=model,
               # default_bounding_parameter=("MW",min(stream_width_bound,max_simd)),
               # bounding_value_for_each_possible_value=bounding_values
                
                param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="SIMD",index = arg_indx,value=new_possible_values[0],
                                  possible_values=new_possible_values,padding=padding,model=model,default_bounding_parameter=("MW",max_simd),bounding_value_for_each_possible_value=new_bounding_values)
                pset.parameters.append(param)
                arg_indx+=1

                max_pe = node_inst.get_nodeattr("MH")
                possible_values, bounding_values = allowed_divisors(max_pe,padding)
               # print(possible_values)
                param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="PE",index = arg_indx,value=possible_values[0],
                                  possible_values=possible_values,padding=padding,model=model,default_bounding_parameter=("MH",max_pe),bounding_value_for_each_possible_value=bounding_values)
                pset.parameters.append(param)
                arg_indx+=1



            elif op_type in pe_ops:
                max_pe = node_inst.get_nodeattr("NumChannels")

                node_inst.set_nodeattr("PE", 1)

                possible_values,bounding_values = allowed_divisors(max_pe,padding)
                param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="PE",index = arg_indx,value=possible_values[0],
                                  possible_values=possible_values,padding=padding,model=model,default_bounding_parameter=("NumChannels",max_pe),bounding_value_for_each_possible_value=bounding_values)
                pset.parameters.append(param)
                arg_indx+=1

            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                node_inst.set_nodeattr("PE", 1)

                possible_values,bounding_values = allowed_divisors(max_pe,padding)
                param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="PE",index = arg_indx,value=possible_values[0],
                                  possible_values=possible_values,padding=padding,model=model,default_bounding_parameter=("Labels",max_pe),bounding_value_for_each_possible_value=bounding_values)
                pset.parameters.append(param)
                arg_indx+=1

            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU

                
                #if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    #node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                #self.optimize_attribute_val(node_inst, max_pe, "PE")

                possible_values,bounding_values = allowed_divisors(max_pe,padding)
                param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="PE",index = arg_indx,value=possible_values[0],
                                  possible_values=possible_values,padding=padding,model=model,default_bounding_parameter=("Channels",max_pe),bounding_value_for_each_possible_value=bounding_values)
                pset.parameters.append(param)
                arg_indx+=1

                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
            # cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                # self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                    
                    possible_values,bounding_values = allowed_divisors(max_simd,padding)
                    param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="SIMD",index = arg_indx,value=possible_values[0],
                                      possible_values=possible_values,padding=padding,model=model,default_bounding_parameter=("Kernel",node_inst.get_nodeattr("Kernel")),bounding_value_for_each_possible_value=bounding_values)
                    pset.parameters.append(param)
                    arg_indx+=1
            

            elif op_type in simd_ops:
                #print("entering simd conv exception")
                if op_type.startswith("ConvolutionInputGenerator"):
                    
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        # init/reset parallel_window mode of RTL SWG
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                            

                        node_inst.set_nodeattr("SIMD", max_simd)
                       # possible_values,bounding_values = allowed_divisors(max_simd,padding)
                        
                       # param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="SIMD",index = arg_indx,value=possible_values[0],
                       #                   possible_values=possible_values,padding=padding,model=model,default_bounding_parameter=("IFMChannels",max_simd),bounding_value_for_each_possible_value=bounding_values)
                       # pset.parameters.append(param)
                       # arg_indx+=1


                    else:
                        # depthwise SWGs are handled separately
                        continue

                    #arg_indx+=1

                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")
                    possible_values,bounding_values = allowed_divisors(max_simd,padding)
                    param = Parameter(node=node_inst,node_indx=node_indx,op_type=op_type,name="SIMD",index = arg_indx,value=possible_values[0],
                                      possible_values=possible_values,padding=padding,model=model,
                                      default_bounding_parameter=("NumChannels",max_simd),bounding_value_for_each_possible_value=bounding_values)
                    pset.parameters.append(param)
                    arg_indx+=1

            else:
                warnings.warn("Unknown op type: " + op_type)
                #if get:
                #    args.append((node_indx,"PE",1, False))
                    #constraints.append((node_indx, "PE", {'type': 'eq', 'fun': lambda x:  x[arg_indx]+1}))
                #    bounds.append((1,1))
                #    pass
                #else:
                #    pass
                #arg_indx+=1
            
            
        #if not get:
            # model is being updated so we'll also recount the cycles and such
    # model = model.transform(InsertDWC())
        #model = model.transform(InsertFIFO(create_shallow_fifos=True))
    #  model = model.transform(GiveUniqueNodeNames())
    #  model = model.transform(SpecializeDWCLayers())
        #model = model.transform(SpecializeFIFOLayers())
    # model = model.transform(AnnotateCycles())
        #print("b count: ",len(bounds))
        #print("arg count: ",arg_indx)

        self.params = pset





def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


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
    def __init__(self, target_cycles_per_frame=1000, mvau_wwidth_max=36, two_pass_relaxation=True,padding=0,effort=200):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation
        self.padding = padding
        self.effort = 200
        self.hard_constraint_target = "max_cycles"
        self.insert_dwcs = True


    def apply_optimized_folding(self, model):

        """
        Optimization algorithm-based folding transformation which employs
        padding, an iterative optimization algorithm and a target function
        to find optimal folding values for each node in the FINN graph,
        by default minimizing resource consumption while making sure to meet
        the target max_cycles (throughput) rate
        """
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SpecializeLayers())
        model = model.transform(AnnotateCycles())

        

        targets = {}
        targets["max_cycles"] = self.target_cycles_per_frame

        # fetch all parameters and bounds from the model by running the optimizer once without padding
        init_model = copy.deepcopy(model)
        opt1 = Optimizer(init_model,"defaultOPT_for_parameter_extraction",targets,self.hard_constraint_target, padding = 0)
        opt1.targets = targets
        opt1.get_params() # generate full param list
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
        param_set_min.apply_updates()
        param_set_max.apply_updates()

        param_set_min.assign_involved_nodes()
        param_set_max.assign_involved_nodes()




        # apply minimum & maximum values and fetch resource & cycle reports

      #  cycles_max = get_max_cycles(param_set_min.nodes)
      #  cycles_min = get_max_cycles(param_set_max.nodes)

       # resources_max = get_resources(param_set_min.nodes)
        resources_min = get_resources(param_set_max.nodes)

        targets["LUT"] = resources_min["LUT"]
       # targets["BRAM_18K"] = resources_min["BRAM_18K"]
       # targets["DSP48"] = resources_min["DSP48"]

        opt2 = Optimizer(model,"padded OPT",targets,self.hard_constraint_target, padding = self.padding,maxfun_per_parameter=self.effort)
        opt2.targets = targets
        opt2.get_params() # generate full param list
        opt2.optimize()
        model = opt2.model

        # necessary final transforms
        if self.insert_dwcs:
            model.transform(InsertDWC())

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        return (model, False)


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
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
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
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame and simd_val > (max_simd/1024):
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
        if self.padding == 0:
            return self.apply_naive_folding(model)
        else:
            return self.apply_optimized_folding(model)