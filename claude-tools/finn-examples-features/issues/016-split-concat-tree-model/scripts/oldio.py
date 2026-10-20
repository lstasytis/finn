# Does node-rtlsim TAV derivation work with the default (in0/out0 only) io_dict?
import sys
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from fpgadataflow.test_fpgadataflow_split_concat_characterization import (
    make_concat_hw_model, make_split_hw_model)
part = "xc7z020clg400-1"
for name, mk in (("split", make_split_hw_model), ("concat", make_concat_hw_model)):
    m = mk([8, 8, 8, 8], 2, [1])
    m = m.transform(SpecializeLayers(part)).transform(GiveUniqueNodeNames())
    m = m.transform(PrepareIP(part, 4)).transform(HLSSynthIP()).transform(PrepareRTLSim())
    inst = getCustomOp(m.graph.node[0])
    try:
        HWCustomOp.derive_token_access_vectors(inst, m, 10**6, "rtlsim", part, 4, inst.onnx_node.op_type)
        print(name, "default io_dict OK")
    except BaseException as e:
        print(name, "default io_dict FAILS:", type(e).__name__, str(e)[:200])
