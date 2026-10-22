"""Issue 018: depth JSON setting to D the FIFOs of a copied build whose producer/consumer op
types match a pattern. usage: group_json.py <copy dir> <out.json> <D> <prod re> <cons re>
[EXCL=name,name] (regexes on node names, e.g. 'VVAU_hls_' 'StreamingDataWidthConverter')"""
import json
import re
import sys

from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper(sys.argv[1] + "/step_hw_ipgen.onnx")
out, d, pre, cre = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
excl = sys.argv[6].split("=")[1].split(",") if len(sys.argv) > 6 else []
res = {}
for n in m.graph.node:
    if not n.op_type.startswith("StreamingFIFO"):
        continue
    p, c = m.find_producer(n.input[0]), m.find_consumer(n.output[0])
    if p is None or c is None or p.name in excl or c.name in excl:
        continue
    if re.fullmatch(pre, p.name) and re.fullmatch(cre, c.name):
        res[n.name] = d
json.dump(res, open(out, "w"))
print(out.split("/")[-1], len(res), sorted(res)[:20])
