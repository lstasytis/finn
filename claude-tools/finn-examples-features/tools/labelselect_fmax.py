"""OOC synth + place & route of a single LabelSelect node: fmax vs PE."""
import json, os, sys
sys.path.insert(0, "/home/lstasytis/finn_dev_lstasytis/finn/tests/fpgadataflow")
from qonnx.core.datatype import DataType
from qonnx.transformation.general import GiveUniqueNodeNames
from test_fpgadataflow_labelselect import make_labelselect_modelwrapper
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.vivado import parse_ooc_synth_results

PART, CLK = "xczu7ev-ffvc1156-2-e", 5.0
labels, pe, dt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
model = make_labelselect_modelwrapper(labels, pe, 1, DataType[dt], "hls")
model = model.transform(SpecializeLayers(PART)).transform(GiveUniqueNodeNames())
model = model.transform(InsertFIFO(create_shallow_fifos=True)).transform(SpecializeLayers(PART))
model = model.transform(GiveUniqueNodeNames())
model = model.transform(PrepareIP(PART, CLK)).transform(HLSSynthIP())
model = model.transform(CreateStitchedIP(PART, CLK, run_synth=True, run_pnr=True))
res = parse_ooc_synth_results(model.get_metadata_prop("vivado_stitch_proj"))
row = {"labels": labels, "pe": pe, "dtype": dt, "clk_ns": CLK, **(res or {})}
with open("/tmp/finn_dev_lstasytis/labelselect_fmax/results.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("RESULT", labels, pe, dt, (res or {}).get("fmax_mhz"), (res or {}).get("LUT"))
