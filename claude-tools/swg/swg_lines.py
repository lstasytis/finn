# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the size of the ConvolutionInputGenerator tree model.

Counts, by AST, every top-level definition that exists only to serve
``get_tree_model``, plus ``get_tree_model`` itself. Run it on any revision:

    python claude-tools/swg/swg_lines.py
    git show 43597d82:src/finn/... > /tmp/old.py && python swg_lines.py /tmp/old.py
"""

import ast
import sys

DEFAULT = "src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py"

# every symbol that exists only to serve get_tree_model, across revisions
# The model is one method with everything nested inside it, so counting that
# method counts the whole model. The rest are historical names, kept so a
# count taken against an older revision still measures the same thing.
TREE_SYMBOLS = {
    "get_tree_model",
    "swg_default_params",
    "swg_parallel_params",
    "swg_default_schedule",
    "swg_parallel_schedule",
    "swg_default_tree",
    "swg_params",
    "_is_int",
    "SwgController",
    "swg_nest_dims",
    "swg_default_nest",
    "swg_parallel_nest",
    "swg_demand_lead",
    "_leaf",
    "_comp",
    "_clip",
}


def main(path):
    tree = ast.parse(open(path).read())
    rows = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in TREE_SYMBOLS:
            rows.append((node.name, node.end_lineno - node.lineno + 1))
        if isinstance(node, ast.ClassDef) and node.name == "ConvolutionInputGenerator":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "get_tree_model":
                    rows.append(("get_tree_model", sub.end_lineno - sub.lineno + 1))
    width = max(len(n) for n, _ in rows)
    for name, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<{width}}  {n:>5}")
    print(f"  {'TOTAL':<{width}}  {sum(n for _, n in rows):>5}")
    print(f"  {'(file)':<{width}}  {len(open(path).readlines()):>5}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
