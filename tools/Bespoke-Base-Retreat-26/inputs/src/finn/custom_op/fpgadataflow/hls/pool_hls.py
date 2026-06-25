import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.pool import Pool


class Pool_hls(Pool, HLSBackend):
    """Class that corresponds to finn-hlslib Pool_batch function.
    Requires ConvolutionInputGenerator(depthwise == 1) to format its input

    Input shape (BatchSize,OutImgDim,OutImgDim,TotalKernelSize*Channels)
    Output shape (BatchSize,OutImgDim,OutImgDim,Channels)

    Notes:

    * The input shape was chosen to be compatible with im2col (only true when there
      is not folding).
    * The actual data layout produced by the hlslib kernels is different
      for depthwise ops.

        * depthwise SWG: (1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)

    Channels can be folded using PE (SIMD from the input perspective)
    """

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Pool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "pool.hpp"']

    def defines(self, var):
        k = int(np.prod(self.get_nodeattr("KernelSize")))
        cf = int(self.get_nodeattr("Channels") / self.get_nodeattr("PE"))
        osz = np.prod(self.get_nodeattr("OutImgDims"))
        self.code_gen_dict["$DEFINES$"] = [
            "constexpr unsigned  ISIZE = {};".format(osz * cf * k),
            "constexpr unsigned  K = {};".format(k),
        ]

    def docompute(self):
        pe = self.get_nodeattr("PE")
        fxn = self.get_nodeattr("Function")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        o_hls_dt = "hls::vector<%s, %d>" % (odt.get_hls_datatype_str(), pe)

        self.code_gen_dict["$DOCOMPUTE$"] = []
        if fxn == "MaxPool":
            self.code_gen_dict["$DOCOMPUTE$"] += ["MaxPoolFunction<{}> pool_fxn;".format(o_hls_dt)]
        elif fxn == "QuantAvgPool":
            shift = self.get_nodeattr("Size")
            accum_bits = self.get_nodeattr("AccumBits")
            act_hls_dt = "hls::vector<ap_%sint<%d>, %d>" % (
                "" if idt.signed() else "u",
                accum_bits,
                pe,
            )
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "QuantAvgPoolFunction<{},{},{}> pool_fxn;".format(o_hls_dt, act_hls_dt, shift)
            ]
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        self.code_gen_dict["$DOCOMPUTE$"] += ["Pool_batch<ISIZE, K>(in0_V, out0_V, pool_fxn);"]

    def pragmas(self):
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS dataflow disable_start_propagation")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=in0_V compact=bit")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=out0_V compact=bit")

    def blackboxfunction(self):
        pe = self.get_nodeattr("PE")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_hls_dt = "hls::vector<%s, %d>" % (idt.get_hls_datatype_str(), pe)
        o_hls_dt = "hls::vector<%s, %d>" % (odt.get_hls_datatype_str(), pe)

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s> &in0_V, hls::stream<%s> &out0_V)"
            % (self.onnx_node.name, i_hls_dt, o_hls_dt)
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
