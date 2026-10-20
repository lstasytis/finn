# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

from finn.custom_op.fpgadataflow.globalaccpool import GlobalAccPool
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class GlobalAccPool_hls(GlobalAccPool, HLSBackend):
    """Class that corresponds to finn-hlslib AccPool_Batch function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(GlobalAccPool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required GlobalAccPool_Batch attributes do not exist.""")

        # verify that input data is 2D
        if len(self.get_nodeattr("numInputVectors")) != 3:
            info_messages.append("""GlobalAccPool_Batch requires 2D data input.""")
            raise Exception

        return info_messages

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var):
        pe = self.get_nodeattr("PE")
        vecs = list(self.get_nodeattr("numInputVectors"))
        self.code_gen_dict["$DEFINES$"] = [
            "#define PE %d" % pe,
            "#define NumFolds %d" % (self.get_nodeattr("NumChannels") // pe),
            "#define NumPixels %d" % int(np.prod(vecs[1:])),
            "typedef %s TI;" % self.get_input_datatype().get_hls_datatype_str(),
            "typedef %s TO;" % self.get_output_datatype().get_hls_datatype_str(),
        ]

    def docompute(self):
        ishape = self.get_normal_input_shape()
        if ishape[1] == ishape[2]:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """AccPool_Batch<{}, {}, {}, {}, {}> (in0_V, out0_V, 1);""".format(
                    ishape[1],
                    self.get_nodeattr("NumChannels"),
                    self.get_input_datatype().get_hls_datatype_str(),
                    self.get_nodeattr("PE"),
                    self.get_output_datatype().get_hls_datatype_str(),
                )
            ]
            return
        # hlslib's AccPool_Batch<ImgDim, ...> sums over ImgDim * ImgDim pixels (a
        # square image); the same loop over the actual pixel count covers H != W
        # inputs, e.g. the 1 x tokens frames of the transformers
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """ap_uint<PE * TO::width> accumulators[NumFolds];
#pragma HLS bind_storage variable=accumulators type=RAM_2P impl=LUTRAM
for (unsigned pixel = 0; pixel < NumPixels; pixel++) {
  for (unsigned fold = 0; fold < NumFolds; fold++) {
#pragma HLS pipeline style=flp II=1
    ap_uint<PE * TI::width> const thin = in0_V.read();
    ap_uint<PE * TO::width> accbank = accumulators[fold];
    for (unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      TI const val = thin((pe + 1) * TI::width - 1, pe * TI::width);
      TO const acc = accbank((pe + 1) * TO::width - 1, pe * TO::width);
      accbank((pe + 1) * TO::width - 1, pe * TO::width) = val + (pixel == 0 ? TO(0) : acc);
    }
    accumulators[fold] = accbank;
  }
}
for (unsigned fold = 0; fold < NumFolds; fold++) {
  out0_V.write(accumulators[fold]);
}"""
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0_V,
                hls::stream<ap_uint<{}>> &out0_V)""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.get_outstream_width(),
            )
        ]
