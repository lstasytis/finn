import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)

# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter_hls(StreamingDataWidthConverter, HLSBackend):
    """Class that corresponds to finn-hlslib StreamingDataWidthConverter_Batch
    function."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingDataWidthConverter.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var):
        numReps = 1
        numInWords = int(np.prod(self.get_folded_input_shape()[:-1]))
        inWidth = self.get_nodeattr("inWidth")
        outWidth = self.get_nodeattr("outWidth")
        self.code_gen_dict["$DEFINES$"] = [
            "#define InWidth %d " % inWidth,
            "#define OutWidth %d " % outWidth,
            "#define NumInWords %d " % numInWords,
            "#define numReps %d" % numReps,
        ]
        if self.needs_lcm():
            lcmWidth = self.get_iowidth_lcm()
            assert numInWords % (lcmWidth / inWidth) == 0, "Error in DWC LCM calculation"
            numLCMToOut = numInWords // (lcmWidth / inWidth)
            self.code_gen_dict["$DEFINES$"].append("#define LCMWidth %d" % lcmWidth)
            self.code_gen_dict["$DEFINES$"].append("#define NumLCMToOut %d" % (numLCMToOut))

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )

    def docompute(self):
        # TODO continue with fxns below, they are copy-pasted
        op = "StreamingDataWidthConverter_Batch"
        if self.needs_lcm():
            self.code_gen_dict["$DOCOMPUTE$"] = [
                'hls::stream<ap_uint<{}>> intermediate ("intermediate");'.format(
                    self.get_iowidth_lcm()
                ),
                "%s<InWidth, LCMWidth, NumInWords>(in0_V, intermediate, numReps);" % op,
                "%s<LCMWidth, OutWidth, NumLCMToOut>(intermediate, out0_V, numReps);" % op,
            ]
        else:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "%s<InWidth, OutWidth, NumInWords>(in0_V, out0_V, numReps);" % op
            ]

    def blackboxfunction(self):
        in_packed_bits = self.get_instream_width()
        in_packed_hls_type = "ap_uint<%d>" % in_packed_bits
        out_packed_bits = self.get_outstream_width()
        out_packed_hls_type = "ap_uint<%d>" % out_packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
            % (
                self.onnx_node.name,
                in_packed_hls_type,
                out_packed_hls_type,
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        if self.needs_lcm():
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW disable_start_propagation")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            exp_shape = self.get_normal_input_shape()
            output = context[self.onnx_node.input[0]]
            output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
            context[self.onnx_node.output[0]] = output

        elif mode == "rtlsim":
            HLSBackend.execute_node(self, context, graph)
