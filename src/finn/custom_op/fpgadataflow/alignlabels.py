
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class AlignLabels(HWCustomOp):
    """Abstraction layer for HW implementation of AlignLabels"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # FINN DataTypes for label and (model input) data
            "label_dtype": ("s", True, ""),
            "data_dtype": ("s", True, ""),
            
            # Shapes of the label and data
            "label_shape": ("ints", True, [1]),
            "data_shape": ("ints", True, [1]),
            
            # PE for the data stream - folds data analogous to model's first layer? 
            # TODO: Do we want this? Inherently good, because chunks of data are smaller => more fitting for AXI, but is this always given?
            "PE": ("i", True, 0),
            
            # Data to label ratio, how many "chunks" of data to produce one label
            # "label_ratio": ("i", True, 1),
            # TODO: Unnecessary? label_ratio is always np.prod(self.get_folded_output_shape(1)[:-1]), right?
            
            # TODO: Is this required like in elementwiseop?
            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there might be two inputs
            # "inFIFODepths": ("ints", False, [2, 2]),
            # "outFIFODepths": ("ints", False, [2, 2]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return [self.get_nodeattr("label_shape"), self.get_nodeattr("data_shape")][ind]

    def get_folded_input_shape(self, ind=0):
        if(ind == 0): return self.get_normal_input_shape(0) # Label always fully folded - TODO: Valid assumption?
        
        *input_vectors, input_channels = self.get_nodeattr("data_shape")
        pe = self.get_nodeattr("PE")
        assert input_channels % pe == 0, "PE must divide data shape's number of channels"
        folds = int(input_channels / pe)
        folded_ishape = tuple(input_vectors + [folds, pe])
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        # since the output shape corresponds entirely to the input shape
        return self.get_normal_input_shape(ind)

    def get_folded_output_shape(self, ind=0):
        # since the output shape corresponds entirely to the input shape
        return self.get_folded_input_shape(ind)

    def make_shape_compatible_op(self, model):
        ret = super().make_shape_compatible_op(model)
        ret.output[:] = self.onnx_node.output
        return ret

    def infer_node_datatype(self, model):
        node = self.onnx_node

        label_dtype = model.get_tensor_datatype(node.input[0])
        if label_dtype != self.get_input_datatype(0):
            warn_str = "label_dtype changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(0)),
                str(label_dtype),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("label_dtype", label_dtype.name)

        data_dtype = model.get_tensor_datatype(node.input[1])
        if data_dtype != self.get_input_datatype(1):
            warn_str = "data_dtype changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(1)),
                str(data_dtype),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("data_dtype", data_dtype.name)
        
        # Set output data types accordingly
        model.set_tensor_datatype(node.output[0], label_dtype)
        model.set_tensor_datatype(node.output[1], data_dtype)


    def get_input_datatype(self, ind=0): 
        """Returns FINN DataType of input."""
        return [self.get_nodeattr("label_dtype"), self.get_nodeattr("data_dtype")][ind]

    def get_output_datatype(self, ind=0): 
        """Returns FINN DataType of output."""
        return self.get_input_datatype(ind)

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        if (ind == 0):
            return np.prod(self.get_normal_input_shape(0)) # Label always fully folded - TODO: Valid assumption?
        
        else: 
            ibits = self.get_input_datatype(ind).bitwidth()
            pe = self.get_nodeattr("PE")
            in_width = pe * ibits
            return in_width

    def get_outstream_width(self, ind=0):
        """Returns output stream width."""
        return self.get_instream_width(ind)

    def get_exp_cycles(self):
        # [:-1] to exclude PE dimension
        # (1) to get data shape - always more data than labels => determines amount of cycles
        return np.prod(self.get_folded_output_shape(1)[:-1])

    def execute_node(self, context, graph): # TODO: Does this method have to reflect the alignment? (Shouldn't, right?)
        node = self.onnx_node
        inputs = [context[node.input[0]], context[node.input[1]]]
        context[node.output[0]] = inputs[0].astype(np.float32)
        context[node.output[1]] = inputs[1].astype(np.float32)

    # TODO: Necessary
    # def derive_characteristic_fxns(self, period):

    # def get_number_output_values(self):
        

    # TODO: Further methods?