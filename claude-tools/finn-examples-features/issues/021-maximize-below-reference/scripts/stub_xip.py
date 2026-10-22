"""Replace HLS wrappers of Xilinx IP (floating_point_v7_1, ...: encrypted, not in the source
list) by black-box stubs in a srcs.txt: python stub_xip.py <srcs.txt> <stub dir>"""
import os, re, sys
srcs, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
keep = []
for f in open(srcs).read().split():
    txt = open(f).read() if f.endswith(".v") else ""
    m = re.search(r"^\s*(floating_point|mult_gen|xbip_\w+)_v\d+_\d+_\d+\s*#", txt, re.M)
    if not m:
        keep.append(f)
        continue
    head = txt[: m.start()]
    # module header and port declarations only
    head = re.sub(r"\(\*[^*]*\*\)", "", head)
    decl = [l for l in head.splitlines() if re.match(r"\s*(module|input|output|\);|  \w+,?$|  \w+$)", l)]
    stub = os.path.join(out, os.path.basename(f))
    open(stub, "w").write("(* black_box *)\n" + "\n".join(decl) + "\nendmodule\n")
    keep.append(stub)
open(srcs, "w").write("\n".join(keep) + "\n")
