
from finn.custom_op.fpgadataflow.alignlabels import AlignLabels
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

import numpy as np


class AlignLabels_hls(AlignLabels, HLSBackend):
    """Class that corresponds to finn-hlslib function of the same name."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(AlignLabels.get_nodeattr_types(self))
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
            self.get_nodeattr("PE")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required AlignLabels attributes do not exist.""")

        return info_messages

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']
        
    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width(0))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in1_V ("in1_V");'.format(self.get_instream_width(1))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width(0))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out1_V ("out1_V");'.format(self.get_outstream_width(1))
        )
        

    def defines(self, var):
        pe = self.get_nodeattr("PE")
        numTotal = np.prod(self.get_folded_output_shape(1)[:-1])
        self.code_gen_dict["$DEFINES$"] = [
            "#define LabelWidth %d " % self.get_instream_width(0),
            "#define DataWidth %d " % self.get_instream_width(1),
            "#define NumTotal %d " % numTotal,
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """AlignLabels<LabelWidth, DataWidth, NumTotal>
            (in0_V, in1_V, out0_V, out1_V);"""
        ]

    def blackboxfunction(self):
        in_stream1 = "hls::stream<ap_uint<%d>> &in0_V" % (self.get_instream_width(0))
        in_stream2 = "hls::stream<ap_uint<%d>> &in1_V" % (self.get_instream_width(1))
        out_stream1 = "hls::stream<ap_uint<%d>> &out0_V" % (self.get_outstream_width(0))
        out_stream2 = "hls::stream<ap_uint<%d>> &out1_V" % (self.get_outstream_width(1))
        
        blackbox_hls = "void %s(%s, %s, %s, %s)" % (self.onnx_node.name, in_stream1, in_stream2, out_stream1, out_stream2)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self): # TODO: Same as DuplicateStreams like this?  
        pragmas = []
        pragmas.append("#pragma HLS dataflow disable_start_propagation")
        pragmas.append("#pragma HLS INTERFACE axis port=in0_V")
        pragmas.append("#pragma HLS INTERFACE axis port=in1_V")
        pragmas.append("#pragma HLS INTERFACE axis port=out0_V")
        pragmas.append("#pragma HLS INTERFACE axis port=out1_V")
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        pragmas.append("#pragma HLS aggregate variable=in0_V compact=bit")
        pragmas.append("#pragma HLS aggregate variable=in1_V compact=bit")
        pragmas.append("#pragma HLS aggregate variable=out0_V compact=bit")
        pragmas.append("#pragma HLS aggregate variable=out1_V compact=bit")
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    # TODO: Necessary?
    # def timeout_condition(self):

    # def timeout_read_stream(self):
    

    # TODO: Further methods?
