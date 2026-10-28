"""Issue 018: copy a build's step_hw_ipgen.onnx and every node's code_gen dir into <dest>
(so that diagnostic re-stitches never touch the original build dirs) and rewrite the paths.
usage: prep_copy.py <build dir> <dest dir>"""
import os
import shutil
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

bdir, dest = sys.argv[1], os.path.abspath(sys.argv[2])
os.makedirs(dest, exist_ok=True)
m = ModelWrapper(bdir + "/intermediate_models/step_hw_ipgen.onnx")
moved = {}
for n in m.graph.node:
    inst = getCustomOp(n)
    src = inst.get_nodeattr("code_gen_dir_ipgen")
    if src and src not in moved:
        dst = os.path.join(dest, os.path.basename(src))
        if not os.path.isdir(dst):
            shutil.copytree(src, dst, symlinks=True)
        moved[src] = dst
for n in m.graph.node:
    for a in n.attribute:
        if a.type == a.STRING:
            s = a.s.decode()
            for src, dst in moved.items():
                if src in s:
                    s = s.replace(src, dst)
            a.s = s.encode()
m.save(os.path.join(dest, "step_hw_ipgen.onnx"))
print("copied", len(moved), "dirs to", dest)
