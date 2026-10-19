"""Assemble the docs section and IMPACT.md from the template and the rendered tables.
    python assemble_impact.py <template.rst> <tables.rst> <tables.md> <impact.json> <docs.rst> <IMPACT.md>
"""
import json
import sys

tpl, tables_rst, tables_md, impact_json, docs_rst, impact_md = sys.argv[1:7]
ranking, model = open(tables_rst).read().strip().split("\n\n", 1)
section = open(tpl).read().replace("RANKING_TABLE", ranking).replace("MODEL_TABLE", model)
docs = open(docs_rst).read()
marker = "Impact of each rule\n===================\n"
if marker in docs:
    docs = docs[: docs.index(marker)]
elif "Regression set\n==============\n" in docs:
    # insert before the regression-set section
    idx = docs.index("Regression set\n==============\n")
    docs = docs[:idx] + section.rstrip("\n") + "\n\n" + docs[idx:]
    open(docs_rst, "w").write(docs)
    print("inserted before Regression set")
    section = None
if section is not None:
    open(docs_rst, "w").write(docs.rstrip("\n") + "\n\n" + section)
impact = json.load(open(impact_json))
md = ["# Impact of each FIFO sizing rule (Vivado 2024.2, ZCU104, stitched-IP rtlsim, 1 frame)\n",
      "Method: full sizer vs the sizer with one rule removed (ablation), on characterized fixtures; "
      "throughput from rtlsim of the ablated depths applied with auto_fifo_depths=False.\n",
      open(tables_md).read(),
      "\n## Raw numbers (impact.json)\n", "```", json.dumps(impact, indent=1), "```"]
open(impact_md, "w").write("\n".join(md))
print("ok")
