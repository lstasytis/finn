# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lane-wise split and merge of a folded stream, for splitting a large layer into k parallel
parts without throughput loss (finn.transformation.fpgadataflow.split_large_mvau).

A stream of NumChannels channels folded by L lanes carries channel f*L + l in lane l of word f.

StreamingLaneSplit: one input stream of L = SIMD lanes into k output streams of L/k lanes each,
every word at once: output i carries lanes [i*L/k, (i+1)*L/k) of every input word, i.e. the
channels f*L + i*L/k + j (f over the folds, j < L/k), NumChannels / k per vector.

StreamingLaneMerge: the inverse, k input streams of L/k lanes into one stream of L = PE lanes;
word f of the output is the side-by-side concatenation of word f of every input.

Both move one word per cycle on every port (no reordering, no buffering beyond one register
stage), so the parts behind a split run in lockstep at the original rate.
"""

import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class _Lanes(HWCustomOp):
    def get_nodeattr_types(self):
        my_attrs = {
            # channels of the wide stream
            "NumChannels": ("i", True, 0),
            # lanes of the wide stream
            "Lanes": ("i", True, 0),
            # number of narrow streams (divides Lanes)
            "NumParts": ("i", True, 0),
            "dataType": ("s", True, ""),
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def _vecs(self):
        return list(self.get_nodeattr("numInputVectors"))

    def _wide_shape(self, folded):
        c, L = self.get_nodeattr("NumChannels"), self.get_nodeattr("Lanes")
        assert c % L == 0 and L % self.get_nodeattr("NumParts") == 0
        return tuple(self._vecs() + ([c // L, L] if folded else [c]))

    def _narrow_shape(self, folded):
        c, L, k = self.get_nodeattr("NumChannels"), self.get_nodeattr("Lanes"), self.get_nodeattr("NumParts")
        return tuple(self._vecs() + ([c // L, L // k] if folded else [c // k]))

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def infer_node_datatype(self, model):
        dt = model.get_tensor_datatype(self.onnx_node.input[0])
        if dt != self.get_input_datatype():
            warnings.warn("dataType changing for %s: %s -> %s" % (self.onnx_node.name, self.get_input_datatype(), dt))
            self.set_nodeattr("dataType", dt.name)
        for o in self.onnx_node.output:
            model.set_tensor_datatype(o, dt)

    def make_shape_compatible_op(self, model):
        ret = super().make_shape_compatible_op(model)
        ret.output[:] = self.onnx_node.output
        return ret

    def get_number_output_values(self):
        # (a dict per stream for several outputs, an int for one: as rtlsim_multi_io expects)
        n = [int(np.prod(self.get_folded_output_shape(i)[1:-1])) for i in range(len(self.onnx_node.output))]
        return n[0] if len(n) == 1 else {"out%d" % i: v for i, v in enumerate(n)}

    def get_exp_cycles(self):
        return int(np.prod(self._wide_shape(True)[:-1]))

    def lut_estimation(self):
        # wires and one register stage
        return 0

    def bram_estimation(self):
        return 0

    def verify_node(self):
        return []


class StreamingLaneSplit(_Lanes):
    """One stream of Lanes lanes -> NumParts streams of Lanes / NumParts lanes (see module doc)."""

    def get_normal_input_shape(self, ind=0):
        return self._wide_shape(False)

    def get_folded_input_shape(self, ind=0):
        return self._wide_shape(True)

    def get_normal_output_shape(self, ind=0):
        return self._narrow_shape(False)

    def get_folded_output_shape(self, ind=0):
        return self._narrow_shape(True)

    def get_instream_width(self, ind=0):
        return self.get_nodeattr("Lanes") * self.get_input_datatype().bitwidth()

    def get_outstream_width(self, ind=0):
        return self.get_nodeattr("Lanes") // self.get_nodeattr("NumParts") * self.get_output_datatype().bitwidth()

    def execute_node(self, context, graph):
        node = self.onnx_node
        k, L = self.get_nodeattr("NumParts"), self.get_nodeattr("Lanes")
        x = context[node.input[0]].reshape(self._wide_shape(True))
        for i in range(k):
            y = x[..., i * (L // k) : (i + 1) * (L // k)]
            context[node.output[i]] = np.ascontiguousarray(y).reshape(context[node.output[i]].shape)


class StreamingLaneMerge(_Lanes):
    """NumParts streams of Lanes / NumParts lanes -> one stream of Lanes lanes (see module doc)."""

    def get_nodeattr_types(self):
        attrs = super().get_nodeattr_types()
        attrs["inFIFODepths"] = ("ints", False, [2] * 2)
        return attrs

    def get_normal_input_shape(self, ind=0):
        return self._narrow_shape(False)

    def get_folded_input_shape(self, ind=0):
        return self._narrow_shape(True)

    def get_normal_output_shape(self, ind=0):
        return self._wide_shape(False)

    def get_folded_output_shape(self, ind=0):
        return self._wide_shape(True)

    def get_instream_width(self, ind=0):
        return self.get_nodeattr("Lanes") // self.get_nodeattr("NumParts") * self.get_input_datatype().bitwidth()

    def get_outstream_width(self, ind=0):
        return self.get_nodeattr("Lanes") * self.get_output_datatype().bitwidth()

    def execute_node(self, context, graph):
        node = self.onnx_node
        parts = [context[i].reshape(self._narrow_shape(True)) for i in node.input]
        y = np.concatenate(parts, axis=-1)
        context[node.output[0]] = y.reshape(context[node.output[0]].shape)
