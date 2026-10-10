# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

import math
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Converts between stream widths. If in_shape and out_shape differ in their last
# (channel) dimension, the channels are zero-padded or cropped at the tail of each row.


class StreamingDataWidthConverter(HWCustomOp):
    """Abstraction layer for HW implementation of StreamingDataWidthConverter"""

    def get_nodeattr_types(self):
        my_attrs = {
            # shapes of input/output tensors, which may differ in the last
            # dimension when the converter pads or crops
            "in_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            # bit width of input and output streams
            "inWidth": ("i", True, 0),
            "outWidth": ("i", True, 0),
            # FINN DataTypes for inputs/outputs
            "dataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("dataType")]

    def get_normal_input_shape(self, ind=0):
        ishape = self.get_nodeattr("in_shape")
        return ishape

    def get_normal_output_shape(self, ind=0):
        oshape = self.get_nodeattr("out_shape")
        return oshape

    def pads_or_crops(self):
        return np.prod(self.get_normal_input_shape()) != np.prod(self.get_normal_output_shape())

    def get_row_elements(self):
        """Elements per row that are carried from input to output: the shorter of
        the input and output channel counts."""
        return min(self.get_normal_input_shape()[-1], self.get_normal_output_shape()[-1])

    def rtl_supported(self):
        """Padding/cropping is done by the RTL converter (vpc), within the last word of a
        row only: each side must carry ceil(N / elements per word) words for N row
        elements. Channel padding from folding always satisfies this."""
        if not self.pads_or_crops():
            return True
        # the same number of rows on both sides (their rank may differ)
        if np.prod(self.get_normal_input_shape()[:-1]) != np.prod(
            self.get_normal_output_shape()[:-1]
        ):
            return False
        bits = self.get_input_datatype().bitwidth()
        n = self.get_row_elements()
        for shape, width in [
            (self.get_normal_input_shape(), self.get_nodeattr("inWidth")),
            (self.get_normal_output_shape(), self.get_nodeattr("outWidth")),
        ]:
            elems = width // bits
            if shape[-1] % elems != 0 or shape[-1] != -(-n // elems) * elems:
                return False
        return True

    def _folded_shape_for_width(self, width, shape):
        """Compute folded shape for a given stream width.

        When the number of channels divides evenly by the number of elements
        per stream word, returns shape[:-1] + [channels // elems, elems].

        When channels don't divide evenly (e.g., wide streams spanning multiple
        spatial dimensions), falls back to flattening the entire tensor:
        (total_elems // elems, elems).
        """
        bits = self.get_input_datatype().bitwidth()
        assert (
            width % bits == 0
        ), """DWC stream width must be divisible by
        input element bitwidth"""
        elems = int(width // bits)
        channels = shape[-1]
        if channels % elems == 0:
            return tuple(list(shape[:-1]) + [int(channels // elems), elems])
        assert not self.pads_or_crops(), "Padding/cropping DWC needs word-aligned channels"

        total_elems = int(np.prod(shape))
        assert (
            total_elems % elems == 0
        ), "DWC stream width with %d elements does not divide tensor shape %s" % (elems, str(shape))
        return (int(total_elems // elems), elems)

    def get_folded_input_shape(self, ind=0):
        iwidth = self.get_nodeattr("inWidth")
        return self._folded_shape_for_width(iwidth, self.get_normal_input_shape())

    def get_folded_output_shape(self, ind=0):
        owidth = self.get_nodeattr("outWidth")
        return self._folded_shape_for_width(owidth, self.get_normal_output_shape())

    def get_instream_width(self, ind=0):
        in_width = self.get_nodeattr("inWidth")
        return in_width

    def get_outstream_width(self, ind=0):
        out_width = self.get_nodeattr("outWidth")
        return out_width

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""StreamingDWC needs 1 data input""")

        return info_messages

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_shape = self.get_normal_input_shape()
        out_shape = self.get_normal_output_shape()
        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == tuple(in_shape), "Input shape does not match expected shape."

        if not self.pads_or_crops():
            context[node.output[0]] = inp.reshape(out_shape)
            return
        n = self.get_row_elements()
        rows = inp.reshape(-1, in_shape[-1])
        output = np.zeros((rows.shape[0], out_shape[-1]), dtype=np.float32)
        output[:, :n] = rows[:, :n]
        context[node.output[0]] = output.reshape(out_shape)

    def get_exp_cycles(self):
        num_in_words = int(np.prod(self.get_folded_input_shape()[:-1]))
        num_out_words = int(np.prod(self.get_folded_output_shape()[:-1]))
        inw, outw = self.get_instream_width(), self.get_outstream_width()
        if max(inw, outw) % min(inw, outw) == 0 and not self.pads_or_crops():
            # (de)serialization moves one word per cycle on its busier side
            return max(num_in_words, num_out_words)
        # general conversion and padding/cropping: conservatively both sides in turn
        return num_in_words + num_out_words

    def lut_estimation(self, fpgapart):
        """Calculates resource estimations for LUTs"""
        inw = self.get_instream_width()
        outw = self.get_outstream_width()

        minw = min(inw, outw)
        maxw = max(inw, outw)

        # sometimes widths aren't directly divisible
        # this requires going up from input width to least common multiple
        # then down to output width
        intw = abs(maxw * minw) // math.gcd(maxw, minw)

        # we assume a shift-based implementation
        # even if we don't use LUTs explicitly, we make some unavailable
        # to other logic because they're tied into the DWC control sets

        cnt_luts = 0
        cset_luts = 0

        if inw != intw:
            cnt_luts += abs(math.ceil(math.log(inw / intw, 2)))
            cset_luts += intw
        if intw != outw:
            cnt_luts += abs(math.ceil(math.log(intw / outw, 2)))
            cset_luts += outw

        return int(cnt_luts + cset_luts)
