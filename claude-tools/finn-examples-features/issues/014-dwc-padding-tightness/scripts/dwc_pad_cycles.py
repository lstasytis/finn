"""Issue 014: node rtlsim cycles of padding/cropping DWCs (vpc) vs get_exp_cycles,
max(in_words, out_words) and in_words + out_words. Usage: python dwc_pad_cycles.py"""
import sys

from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "tests/fpgadataflow")
from test_fpgadataflow_dwc import make_single_dwc_modelwrapper  # noqa: E402

import finn.core.onnx_exec as oxe  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim  # noqa: E402
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode  # noqa: E402
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers  # noqa: E402

# (in_shape, out_shape, inWidth, outWidth, dtype)
CONFIGS = [
    ([1, 64, 12], [1, 64, 10], 8, 10, DataType["INT2"]),  # crop, 3 -> 2 words/row
    ([1, 64, 10], [1, 64, 12], 10, 8, DataType["INT2"]),  # pad, 2 -> 3 words/row
    ([1, 64, 12], [1, 64, 15], 16, 20, DataType["INT4"]),  # pad, 3 -> 3
    ([1, 64, 24], [1, 64, 26], 4, 13, DataType["BIPOLAR"]),  # pad, 6 -> 2
    ([1, 64, 30], [1, 64, 32], 10, 8, DataType["INT2"]),  # pad, 6 -> 8
    ([1, 64, 64], [1, 64, 66], 16, 12, DataType["INT4"]),  # pad 64 -> 66, 16 -> 22 words/row
    ([1, 64, 66], [1, 64, 64], 12, 16, DataType["INT4"]),  # crop 66 -> 64, 22 -> 16
    ([1, 64, 64], [1, 64, 66], 32, 132, DataType["INT2"]),  # pad 64 -> 66, 4 -> 2 words/row
    ([1, 64, 64], [1, 64, 66], 64, 66, DataType["INT2"]),  # pad 64 -> 66, 2 -> 2 words/row
    ([1, 256, 32], [1, 256, 36], 64, 72, DataType["INT4"]),  # 32 -> 36, 2 -> 2 (vgg10-like)
    ([1, 256, 36], [1, 256, 32], 72, 64, DataType["INT4"]),  # crop 36 -> 32
]
part = "xczu3eg-sbva484-1-e"
for in_shape, out_shape, iw, ow, dt in CONFIGS:
    model = make_single_dwc_modelwrapper(in_shape, iw, ow, dt, "rtl", out_shape=out_shape)
    model = model.transform(SpecializeLayers(part)).transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(part, 5)).transform(HLSSynthIP())
    model = model.transform(SetExecMode("rtlsim")).transform(PrepareRTLSim())
    inst = getCustomOp(model.graph.node[0])
    x = gen_finn_dt_tensor(dt, in_shape)
    oxe.execute_onnx(model, {"inp": x})
    inst = getCustomOp(model.graph.node[0])
    win = int(inst.get_number_input_values())
    wout = int(inst.get_number_output_values())
    print(
        f"{in_shape}->{out_shape} {iw}->{ow}b: in_words {win} out_words {wout} "
        f"rtlsim {inst.get_nodeattr('cycles_rtlsim')} exp {inst.get_exp_cycles()} "
        f"max {max(win, wout)} sum {win + wout}",
        flush=True,
    )
