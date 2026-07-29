# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the token-access-vector tree models.

No Vivado, no rtlsim, no board. Every check replays a stored single-node
reference -- a schedule that rtlsim really measured, harvested out of the FIFO
sizing builds and TAV caches by ``ci/experiments/harvest_cache_refs.py`` and
``ci/experiments/gen_transformer_refs.py`` -- and compares it against what the
node's ``get_tree_model`` produces now.

Three things are asserted, in increasing order of how much they matter to FIFO
sizing:

* **token counts** -- one period must move exactly one folded input and one
  folded output. This needs no reference at all and is the check that catches
  the failure that costs the most: a schedule one token short per period makes
  the sizer's steady-state occupancy accumulate a deficit over every frame.
  It is what an MVAU_hls off-by-one turned into a FIFO of depth 784 against
  rtlsim's 37 on cnv-w2a2.
* **row-0 values** -- the schedule the sizer reads, against a recorded budget.
  A configuration that has a tree model but no budget is a hard failure by
  design: it would otherwise be checked against nothing.
* **the vectorised traversal** -- ``Characteristic_Node.cumulative`` must agree
  cycle for cycle with ``traverse_phase_tree``, which is the slow path it
  replaced.

Budgets are measurements, not aspirations. Run
``python3 ci/experiments/score_nodes.py --emit-budgets`` after a deliberate
change and paste the result back, with the reason in the commit message.

    python3 -m pytest tests/fpgadataflow/test_tav_tree_models.py -q
"""

import pytest

import glob
import json
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tests.testing_util.tav_refs import (  # noqa: E402
    REF_DIR,
    as_rows,
    build_node,
    ensure_finn_env,
    read_tav,
    tree_tavs,
)

# (row-0 input error, row-0 output error, period matches the reference)
NODE_BUDGETS = {
    "FMPadding_rtl_02604e": (2, 0, True),
    "FMPadding_rtl_055e2a": (1, 0, True),
    "FMPadding_rtl_0c4f27": (1, 0, True),
    "FMPadding_rtl_23daa7": (2, 0, True),
    "FMPadding_rtl_254fc1": (2, 0, True),
    "FMPadding_rtl_2dd75c": (2, 0, True),
    "FMPadding_rtl_32a647": (1, 0, True),
    "FMPadding_rtl_4604bf": (2, 0, True),
    "FMPadding_rtl_55fd8a": (1, 0, True),
    "FMPadding_rtl_58f2c1": (1, 0, True),
    "FMPadding_rtl_7263fe": (2, 0, True),
    "FMPadding_rtl_8a60ad": (2, 0, True),
    "FMPadding_rtl_930c27": (2, 0, True),
    "FMPadding_rtl_93e25f": (2, 0, True),
    "FMPadding_rtl_9a80d1": (1, 0, True),
    "FMPadding_rtl_a5ea72": (2, 0, True),
    "FMPadding_rtl_b8b6d8": (2, 0, True),
    "FMPadding_rtl_bdc052": (1, 0, True),
    "FMPadding_rtl_be5ecb": (2, 0, True),
    "FMPadding_rtl_e0ee9b": (2, 0, True),
    "FMPadding_rtl_e1b86a": (2, 0, True),
    "FMPadding_rtl_f79414": (2, 0, True),
    "FMPadding_rtl_f8424d": (2, 0, True),
    "LabelSelect_hls_0766c0": (3, 1, False),
    "LabelSelect_hls_3380c5": (5, 3, False),
    "LabelSelect_hls_42785b": (3, 1, False),
    "LabelSelect_hls_663a65": (3, 1, False),
    "LabelSelect_hls_7a243c": (5, 3, False),
    "LabelSelect_hls_8b84fe": (3, 1, False),
    "LabelSelect_hls_b2adb1": (3, 1, False),
    "LabelSelect_hls_b9e9e2": (3, 1, False),
    "Lookup_hls_7e8cfc": (0, 0, False),
    "Lookup_hls_8ae69d": (0, 0, False),
    "MVAU_hls_03675c": (1, 2, True),
    "MVAU_hls_08c630": (1, 2, True),
    "MVAU_hls_0c1c58": (10, 1, False),
    "MVAU_hls_0c2ac9": (0, 0, True),
    "MVAU_hls_0c7783": (1, 7, False),
    "MVAU_hls_10aa93": (0, 0, True),
    "MVAU_hls_110f43": (1, 4, False),
    "MVAU_hls_11e322": (0, 0, True),
    "MVAU_hls_16ce85": (7, 1, False),
    "MVAU_hls_1819a4": (1, 2, True),
    "MVAU_hls_1a0fc1": (0, 0, True),
    "MVAU_hls_1b7e32": (1, 0, True),
    "MVAU_hls_1d2fd8": (0, 0, True),
    "MVAU_hls_1fe0a7": (8, 1, False),
    "MVAU_hls_211653": (1, 2, True),
    "MVAU_hls_230142": (1, 0, True),
    "MVAU_hls_2e6a58": (8, 1, False),
    "MVAU_hls_33476f": (0, 0, True),
    "MVAU_hls_3410f2": (0, 0, True),
    "MVAU_hls_377480": (2, 2, False),
    "MVAU_hls_3aa90e": (0, 0, True),
    "MVAU_hls_3c4e6c": (0, 0, True),
    "MVAU_hls_442301": (0, 0, True),
    "MVAU_hls_4526cf": (0, 0, True),
    "MVAU_hls_453d3c": (0, 0, True),
    "MVAU_hls_493dbe": (0, 0, True),
    "MVAU_hls_50fa05": (0, 0, True),
    "MVAU_hls_5651ba": (1, 2, True),
    "MVAU_hls_5d4dbe": (0, 0, True),
    "MVAU_hls_5dc11e": (0, 0, True),
    "MVAU_hls_662d86": (1, 2, True),
    "MVAU_hls_66d6da": (3, 2, False),
    "MVAU_hls_6ed419": (0, 0, True),
    "MVAU_hls_6f0929": (3, 2, False),
    "MVAU_hls_6fbb42": (8, 1, False),
    "MVAU_hls_7d2a4a": (0, 0, True),
    "MVAU_hls_7d54ed": (1, 2, True),
    "MVAU_hls_807938": (0, 0, True),
    "MVAU_hls_834c96": (0, 0, True),
    "MVAU_hls_8413c1": (2, 2, False),
    "MVAU_hls_84dd8b": (0, 0, True),
    "MVAU_hls_86494c": (8, 1, False),
    "MVAU_hls_880a1e": (0, 0, True),
    "MVAU_hls_8bce26": (0, 0, True),
    "MVAU_hls_8cd3fb": (0, 0, True),
    "MVAU_hls_90f18d": (0, 0, True),
    "MVAU_hls_970628": (0, 0, True),
    "MVAU_hls_9a4d9f": (0, 0, True),
    "MVAU_hls_9c6f30": (1, 2, True),
    "MVAU_hls_9ff965": (2, 2, False),
    "MVAU_hls_a10577": (0, 0, True),
    "MVAU_hls_a32c88": (4, 2, False),
    "MVAU_hls_a65141": (0, 0, True),
    "MVAU_hls_a72f6a": (0, 0, True),
    "MVAU_hls_a7df32": (11, 2, False),
    "MVAU_hls_a89f13": (0, 0, True),
    "MVAU_hls_a91d12": (2, 2, False),
    "MVAU_hls_aa8e47": (1, 2, True),
    "MVAU_hls_ad1efa": (8, 1, False),
    "MVAU_hls_ad3140": (0, 0, True),
    "MVAU_hls_aeb554": (0, 0, True),
    "MVAU_hls_af4c15": (0, 0, True),
    "MVAU_hls_b153dc": (0, 0, True),
    "MVAU_hls_b67513": (0, 0, True),
    "MVAU_hls_cdea1d": (2, 2, False),
    "MVAU_hls_cfa20d": (0, 0, True),
    "MVAU_hls_d221d0": (1, 2, True),
    "MVAU_hls_d252a3": (2, 1, True),
    "MVAU_hls_d783df": (0, 0, True),
    "MVAU_hls_d8cbc2": (9, 1, False),
    "MVAU_hls_e091c3": (0, 0, True),
    "MVAU_hls_e0d98b": (0, 0, True),
    "MVAU_hls_e3b862": (1, 2, True),
    "MVAU_hls_e6e6c3": (0, 0, True),
    "MVAU_hls_e77cc8": (0, 0, True),
    "MVAU_hls_e87fe5": (0, 0, True),
    "MVAU_hls_e9573d": (1, 2, True),
    "MVAU_hls_f153ff": (2, 3, False),
    "MVAU_hls_f4f5e7": (0, 0, True),
    "MVAU_hls_f5d890": (0, 0, True),
    "MVAU_hls_f8053b": (0, 0, True),
    "MVAU_rtl_036bf7": (2, 6, True),
    "MVAU_rtl_0d8ec3": (1, 6, True),
    "MVAU_rtl_11eb99": (4, 1, True),
    "MVAU_rtl_22e298": (4, 1, True),
    "MVAU_rtl_25df79": (6, 1, True),
    "MVAU_rtl_277365": (4, 1, True),
    "MVAU_rtl_34bb94": (1, 6, True),
    "MVAU_rtl_3d8ce0": (4, 1, True),
    "MVAU_rtl_4dc250": (4, 1, True),
    "MVAU_rtl_4ebbc2": (4, 1, True),
    "MVAU_rtl_595cd9": (4, 1, True),
    "MVAU_rtl_78e88a": (4, 1, True),
    "MVAU_rtl_9a8548": (4, 1, True),
    "MVAU_rtl_9c7c69": (1, 6, True),
    "MVAU_rtl_a1095e": (4, 1, True),
    "MVAU_rtl_c83676": (4, 4, True),
    "MVAU_rtl_caa6a1": (4, 1, True),
    "MVAU_rtl_d39d78": (4, 1, True),
    "MVAU_rtl_d64081": (1, 6, True),
    "MVAU_rtl_dd07c6": (3, 6, True),
    "MVAU_rtl_f5d71c": (4, 1, True),
    "MVAU_rtl_f64e3d": (4, 1, True),
    "Pool_hls_1eac9f": (1, 1, False),
    "Pool_hls_2c0691": (1, 0, False),
    "Pool_hls_6404bb": (1, 0, False),
    "Pool_hls_6759ec": (1, 1, False),
    "Pool_hls_7df0bb": (1, 1, False),
    "Pool_hls_872b0b": (1, 1, False),
    "Pool_hls_8fb6ae": (1, 0, False),
    "Pool_hls_9aa206": (1, 0, False),
    "Pool_hls_b1c00b": (1, 1, False),
    "Pool_hls_b78e98": (1, 1, False),
    "Pool_hls_bcd2d5": (1, 0, False),
    "Pool_hls_c9b9fa": (1, 0, False),
    "Pool_hls_d593e2": (1, 1, False),
    "Pool_hls_f62976": (1, 0, False),
    "StreamingConcat_hls_20a5e6": (0, 0, True),
    "StreamingConcat_hls_20cfb5": (0, 0, True),
    "StreamingConcat_hls_239de4": (0, 0, True),
    "StreamingConcat_hls_5b676b": (0, 0, True),
    "StreamingConcat_hls_890cd6": (0, 0, True),
    "StreamingConcat_hls_9956bb": (0, 0, True),
    "StreamingConcat_hls_a7d971": (0, 0, True),
    "StreamingConcat_hls_c94b1c": (0, 0, True),
    "StreamingConcat_hls_d2a279": (0, 0, True),
    "StreamingDataWidthConverter_rtl_022800": (0, 0, True),
    "StreamingDataWidthConverter_rtl_023daa": (0, 0, True),
    "StreamingDataWidthConverter_rtl_02a8ea": (0, 1, True),
    "StreamingDataWidthConverter_rtl_03dee9": (0, 1, True),
    "StreamingDataWidthConverter_rtl_050a8d": (0, 0, True),
    "StreamingDataWidthConverter_rtl_07a236": (0, 0, True),
    "StreamingDataWidthConverter_rtl_099c53": (0, 1, True),
    "StreamingDataWidthConverter_rtl_09b5a0": (0, 0, True),
    "StreamingDataWidthConverter_rtl_0acb0d": (0, 1, True),
    "StreamingDataWidthConverter_rtl_0c2725": (0, 1, True),
    "StreamingDataWidthConverter_rtl_0ce8dd": (0, 0, True),
    "StreamingDataWidthConverter_rtl_0da6e8": (0, 1, True),
    "StreamingDataWidthConverter_rtl_106461": (0, 1, True),
    "StreamingDataWidthConverter_rtl_12f83b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_1497e5": (0, 0, True),
    "StreamingDataWidthConverter_rtl_17a213": (0, 1, True),
    "StreamingDataWidthConverter_rtl_188989": (0, 0, True),
    "StreamingDataWidthConverter_rtl_1a4dea": (0, 1, True),
    "StreamingDataWidthConverter_rtl_1c88f4": (0, 0, True),
    "StreamingDataWidthConverter_rtl_1d4439": (0, 0, True),
    "StreamingDataWidthConverter_rtl_1f308a": (0, 1, True),
    "StreamingDataWidthConverter_rtl_1f9d9f": (0, 0, True),
    "StreamingDataWidthConverter_rtl_202105": (0, 0, True),
    "StreamingDataWidthConverter_rtl_216adc": (0, 1, True),
    "StreamingDataWidthConverter_rtl_23b169": (0, 0, True),
    "StreamingDataWidthConverter_rtl_24c757": (0, 1, True),
    "StreamingDataWidthConverter_rtl_24f7a0": (0, 1, True),
    "StreamingDataWidthConverter_rtl_272922": (0, 1, True),
    "StreamingDataWidthConverter_rtl_28d63d": (0, 1, True),
    "StreamingDataWidthConverter_rtl_298acb": (0, 0, True),
    "StreamingDataWidthConverter_rtl_2a3085": (0, 0, True),
    "StreamingDataWidthConverter_rtl_2b9199": (0, 1, True),
    "StreamingDataWidthConverter_rtl_2c814c": (0, 0, True),
    "StreamingDataWidthConverter_rtl_2df278": (0, 0, True),
    "StreamingDataWidthConverter_rtl_2e887f": (0, 0, True),
    "StreamingDataWidthConverter_rtl_302636": (0, 1, True),
    "StreamingDataWidthConverter_rtl_31ffa7": (0, 0, True),
    "StreamingDataWidthConverter_rtl_321cfa": (0, 0, True),
    "StreamingDataWidthConverter_rtl_34c834": (0, 1, True),
    "StreamingDataWidthConverter_rtl_360060": (0, 1, True),
    "StreamingDataWidthConverter_rtl_366232": (0, 0, True),
    "StreamingDataWidthConverter_rtl_36c21e": (0, 1, True),
    "StreamingDataWidthConverter_rtl_37a8fd": (0, 0, True),
    "StreamingDataWidthConverter_rtl_3c1240": (0, 0, True),
    "StreamingDataWidthConverter_rtl_3c4dfd": (0, 0, True),
    "StreamingDataWidthConverter_rtl_3d41c9": (0, 0, True),
    "StreamingDataWidthConverter_rtl_3e56d9": (0, 0, True),
    "StreamingDataWidthConverter_rtl_400d1b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_4332d7": (0, 0, True),
    "StreamingDataWidthConverter_rtl_479d16": (0, 0, True),
    "StreamingDataWidthConverter_rtl_51db66": (0, 0, True),
    "StreamingDataWidthConverter_rtl_571325": (0, 0, True),
    "StreamingDataWidthConverter_rtl_57dd56": (0, 1, True),
    "StreamingDataWidthConverter_rtl_599836": (0, 1, True),
    "StreamingDataWidthConverter_rtl_5a35c5": (0, 1, True),
    "StreamingDataWidthConverter_rtl_602839": (0, 0, True),
    "StreamingDataWidthConverter_rtl_60618f": (0, 1, True),
    "StreamingDataWidthConverter_rtl_623f49": (0, 0, True),
    "StreamingDataWidthConverter_rtl_632eba": (0, 0, True),
    "StreamingDataWidthConverter_rtl_640e0b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_647872": (0, 0, True),
    "StreamingDataWidthConverter_rtl_64cb47": (0, 1, True),
    "StreamingDataWidthConverter_rtl_65d819": (0, 0, True),
    "StreamingDataWidthConverter_rtl_67b011": (0, 0, True),
    "StreamingDataWidthConverter_rtl_67fc4d": (0, 1, True),
    "StreamingDataWidthConverter_rtl_68865c": (0, 1, True),
    "StreamingDataWidthConverter_rtl_68d658": (0, 0, True),
    "StreamingDataWidthConverter_rtl_6ead04": (0, 1, True),
    "StreamingDataWidthConverter_rtl_6f7b00": (0, 0, True),
    "StreamingDataWidthConverter_rtl_6f9353": (0, 0, True),
    "StreamingDataWidthConverter_rtl_70281e": (0, 0, True),
    "StreamingDataWidthConverter_rtl_70985f": (0, 0, True),
    "StreamingDataWidthConverter_rtl_72594e": (0, 0, True),
    "StreamingDataWidthConverter_rtl_7432bc": (0, 1, True),
    "StreamingDataWidthConverter_rtl_748c53": (0, 1, True),
    "StreamingDataWidthConverter_rtl_7a0e1a": (0, 0, True),
    "StreamingDataWidthConverter_rtl_7b62e0": (0, 0, True),
    "StreamingDataWidthConverter_rtl_7b8018": (0, 1, True),
    "StreamingDataWidthConverter_rtl_7ea2f7": (0, 1, True),
    "StreamingDataWidthConverter_rtl_7eb26d": (0, 0, True),
    "StreamingDataWidthConverter_rtl_807086": (0, 0, True),
    "StreamingDataWidthConverter_rtl_808096": (0, 0, True),
    "StreamingDataWidthConverter_rtl_82f93c": (0, 1, True),
    "StreamingDataWidthConverter_rtl_83944c": (0, 0, True),
    "StreamingDataWidthConverter_rtl_8589a1": (0, 1, True),
    "StreamingDataWidthConverter_rtl_876d44": (0, 0, True),
    "StreamingDataWidthConverter_rtl_882d70": (0, 0, True),
    "StreamingDataWidthConverter_rtl_898d8b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_8b8290": (0, 1, True),
    "StreamingDataWidthConverter_rtl_8d394d": (0, 1, True),
    "StreamingDataWidthConverter_rtl_8f5fae": (0, 0, True),
    "StreamingDataWidthConverter_rtl_8f97b1": (0, 0, True),
    "StreamingDataWidthConverter_rtl_905c3b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_90f298": (0, 1, True),
    "StreamingDataWidthConverter_rtl_923529": (0, 0, True),
    "StreamingDataWidthConverter_rtl_92c0b7": (0, 0, True),
    "StreamingDataWidthConverter_rtl_930df7": (0, 0, True),
    "StreamingDataWidthConverter_rtl_942096": (0, 0, True),
    "StreamingDataWidthConverter_rtl_95fb94": (0, 1, True),
    "StreamingDataWidthConverter_rtl_992259": (0, 1, True),
    "StreamingDataWidthConverter_rtl_99d3e3": (0, 0, True),
    "StreamingDataWidthConverter_rtl_9a77ff": (0, 0, True),
    "StreamingDataWidthConverter_rtl_9caac4": (0, 0, True),
    "StreamingDataWidthConverter_rtl_a00493": (0, 0, True),
    "StreamingDataWidthConverter_rtl_a09853": (0, 1, True),
    "StreamingDataWidthConverter_rtl_a72bcf": (0, 0, True),
    "StreamingDataWidthConverter_rtl_a82adf": (0, 1, True),
    "StreamingDataWidthConverter_rtl_a9af41": (0, 0, True),
    "StreamingDataWidthConverter_rtl_a9dbe2": (0, 1, True),
    "StreamingDataWidthConverter_rtl_aaf657": (0, 1, True),
    "StreamingDataWidthConverter_rtl_accd4e": (0, 1, True),
    "StreamingDataWidthConverter_rtl_af089f": (0, 1, True),
    "StreamingDataWidthConverter_rtl_b18d35": (0, 0, True),
    "StreamingDataWidthConverter_rtl_b26440": (0, 0, True),
    "StreamingDataWidthConverter_rtl_b282a0": (0, 0, True),
    "StreamingDataWidthConverter_rtl_b30796": (0, 0, True),
    "StreamingDataWidthConverter_rtl_b3d85c": (0, 0, True),
    "StreamingDataWidthConverter_rtl_b4a5d6": (0, 0, True),
    "StreamingDataWidthConverter_rtl_b6331a": (0, 1, True),
    "StreamingDataWidthConverter_rtl_b78b9f": (0, 0, True),
    "StreamingDataWidthConverter_rtl_bb0e3c": (0, 0, True),
    "StreamingDataWidthConverter_rtl_bd0403": (0, 0, True),
    "StreamingDataWidthConverter_rtl_bd3b87": (0, 1, True),
    "StreamingDataWidthConverter_rtl_bd5e47": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c0ea13": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c2d09a": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c3feb9": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c50485": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c58b43": (0, 1, True),
    "StreamingDataWidthConverter_rtl_c69082": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c77ce0": (0, 1, True),
    "StreamingDataWidthConverter_rtl_c78bf8": (0, 0, True),
    "StreamingDataWidthConverter_rtl_c8b143": (0, 0, True),
    "StreamingDataWidthConverter_rtl_cc6049": (0, 1, True),
    "StreamingDataWidthConverter_rtl_cde5d0": (0, 0, True),
    "StreamingDataWidthConverter_rtl_cf112c": (0, 0, True),
    "StreamingDataWidthConverter_rtl_d18185": (0, 1, True),
    "StreamingDataWidthConverter_rtl_d1a3a0": (0, 1, True),
    "StreamingDataWidthConverter_rtl_d353e6": (0, 0, True),
    "StreamingDataWidthConverter_rtl_d886c1": (0, 1, True),
    "StreamingDataWidthConverter_rtl_daae30": (0, 0, True),
    "StreamingDataWidthConverter_rtl_dc9711": (0, 1, True),
    "StreamingDataWidthConverter_rtl_dd2ca9": (0, 0, True),
    "StreamingDataWidthConverter_rtl_ddbf3b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_df1b3f": (0, 1, True),
    "StreamingDataWidthConverter_rtl_e686b5": (0, 0, True),
    "StreamingDataWidthConverter_rtl_e6df59": (0, 1, True),
    "StreamingDataWidthConverter_rtl_e86a02": (0, 1, True),
    "StreamingDataWidthConverter_rtl_ed4d02": (0, 0, True),
    "StreamingDataWidthConverter_rtl_edace6": (0, 0, True),
    "StreamingDataWidthConverter_rtl_ee6c0c": (0, 0, True),
    "StreamingDataWidthConverter_rtl_ef4c31": (0, 1, True),
    "StreamingDataWidthConverter_rtl_f2ac57": (0, 1, True),
    "StreamingDataWidthConverter_rtl_f4274b": (0, 0, True),
    "StreamingDataWidthConverter_rtl_f4635e": (0, 1, True),
    "StreamingDataWidthConverter_rtl_f4f8d0": (0, 1, True),
    "StreamingDataWidthConverter_rtl_f74619": (0, 0, True),
    "StreamingDataWidthConverter_rtl_f780c6": (0, 1, True),
    "StreamingDataWidthConverter_rtl_f8522d": (0, 0, True),
    "StreamingDataWidthConverter_rtl_f9c5f7": (0, 1, True),
    "StreamingDataWidthConverter_rtl_f9e351": (0, 1, True),
    "StreamingDataWidthConverter_rtl_fa929b": (0, 1, True),
    "StreamingDataWidthConverter_rtl_fdcdf4": (0, 1, True),
    "StreamingDataWidthConverter_rtl_fecf16": (0, 0, True),
    "StreamingSplit_hls_0d7d49": (0, 0, True),
    "StreamingSplit_hls_132930": (0, 0, True),
    "StreamingSplit_hls_1d48a1": (0, 0, True),
    "StreamingSplit_hls_2e24cf": (0, 0, True),
    "StreamingSplit_hls_3bd45a": (0, 0, True),
    "StreamingSplit_hls_49a58b": (0, 0, True),
    "StreamingSplit_hls_80b6c6": (0, 0, True),
    "StreamingSplit_hls_8f43d4": (0, 0, True),
    "StreamingSplit_hls_a0523a": (0, 0, True),
    "StreamingSplit_hls_be37dc": (0, 0, True),
    "StreamingSplit_hls_cec89d": (0, 0, True),
    "StreamingSplit_hls_d1f6bc": (0, 0, True),
    "StreamingSplit_hls_d85e84": (0, 0, True),
    "Thresholding_rtl_04fd13": (0, 0, False),
    "Thresholding_rtl_05f19c": (0, 0, False),
    "Thresholding_rtl_064cf2": (0, 0, False),
    "Thresholding_rtl_072a42": (0, 0, False),
    "Thresholding_rtl_0790f5": (0, 0, False),
    "Thresholding_rtl_0afa15": (0, 0, False),
    "Thresholding_rtl_0bd06a": (0, 0, False),
    "Thresholding_rtl_0d4448": (0, 0, False),
    "Thresholding_rtl_127fd4": (0, 0, False),
    "Thresholding_rtl_13e6e2": (0, 0, False),
    "Thresholding_rtl_154066": (0, 0, False),
    "Thresholding_rtl_17c04c": (0, 0, False),
    "Thresholding_rtl_1de8ae": (0, 0, False),
    "Thresholding_rtl_1df526": (0, 0, False),
    "Thresholding_rtl_1f9d6d": (0, 0, False),
    "Thresholding_rtl_1fe30e": (0, 0, False),
    "Thresholding_rtl_256fb1": (0, 0, False),
    "Thresholding_rtl_27a6c0": (0, 0, False),
    "Thresholding_rtl_28c174": (0, 0, False),
    "Thresholding_rtl_2a6cbe": (0, 0, False),
    "Thresholding_rtl_3513db": (0, 0, False),
    "Thresholding_rtl_378236": (0, 0, False),
    "Thresholding_rtl_37ca83": (0, 0, False),
    "Thresholding_rtl_3b2fc1": (0, 0, False),
    "Thresholding_rtl_417e92": (0, 0, False),
    "Thresholding_rtl_444aa5": (0, 0, False),
    "Thresholding_rtl_4ad33b": (0, 0, False),
    "Thresholding_rtl_4e64bb": (0, 0, False),
    "Thresholding_rtl_4fb764": (0, 0, False),
    "Thresholding_rtl_585ceb": (0, 0, False),
    "Thresholding_rtl_5b5cee": (0, 0, False),
    "Thresholding_rtl_5e13f9": (0, 0, False),
    "Thresholding_rtl_5ee361": (0, 0, False),
    "Thresholding_rtl_5efcae": (0, 0, False),
    "Thresholding_rtl_5f2167": (0, 0, False),
    "Thresholding_rtl_60b5fc": (0, 0, False),
    "Thresholding_rtl_624224": (0, 0, False),
    "Thresholding_rtl_66aa8a": (0, 0, False),
    "Thresholding_rtl_693b8b": (0, 0, False),
    "Thresholding_rtl_6c15d8": (0, 0, False),
    "Thresholding_rtl_6d3fa9": (0, 0, False),
    "Thresholding_rtl_6f91ef": (0, 0, False),
    "Thresholding_rtl_7087bd": (0, 0, False),
    "Thresholding_rtl_72051d": (0, 0, False),
    "Thresholding_rtl_74573f": (0, 0, False),
    "Thresholding_rtl_751c41": (0, 0, False),
    "Thresholding_rtl_7682e1": (0, 0, False),
    "Thresholding_rtl_78423f": (0, 0, False),
    "Thresholding_rtl_7bed44": (0, 0, False),
    "Thresholding_rtl_7ca4fc": (0, 0, False),
    "Thresholding_rtl_81b84f": (0, 0, False),
    "Thresholding_rtl_82d867": (0, 0, False),
    "Thresholding_rtl_845b13": (0, 0, False),
    "Thresholding_rtl_873487": (0, 0, False),
    "Thresholding_rtl_8a5976": (0, 0, False),
    "Thresholding_rtl_8a7bba": (0, 0, False),
    "Thresholding_rtl_8c8cca": (0, 0, False),
    "Thresholding_rtl_8ce2c2": (0, 0, False),
    "Thresholding_rtl_8cfb12": (0, 0, False),
    "Thresholding_rtl_8e9cc7": (0, 0, False),
    "Thresholding_rtl_92ddaa": (0, 0, False),
    "Thresholding_rtl_92e351": (0, 0, False),
    "Thresholding_rtl_96e080": (0, 0, False),
    "Thresholding_rtl_990d90": (0, 0, False),
    "Thresholding_rtl_9ec817": (0, 0, False),
    "Thresholding_rtl_a32b09": (0, 0, False),
    "Thresholding_rtl_a47f51": (0, 0, False),
    "Thresholding_rtl_a7ead8": (0, 0, False),
    "Thresholding_rtl_a9343d": (0, 0, False),
    "Thresholding_rtl_abcd20": (0, 0, False),
    "Thresholding_rtl_b049d0": (0, 0, False),
    "Thresholding_rtl_b52597": (0, 0, False),
    "Thresholding_rtl_bdc152": (0, 0, False),
    "Thresholding_rtl_bf7909": (0, 0, False),
    "Thresholding_rtl_c005c4": (0, 0, False),
    "Thresholding_rtl_c04c8b": (0, 0, False),
    "Thresholding_rtl_c7d50b": (0, 0, False),
    "Thresholding_rtl_caa63c": (0, 0, False),
    "Thresholding_rtl_d045b1": (0, 0, False),
    "Thresholding_rtl_d4b0f0": (0, 0, False),
    "Thresholding_rtl_d62603": (0, 0, False),
    "Thresholding_rtl_d6e16a": (0, 0, False),
    "Thresholding_rtl_d6f285": (0, 0, False),
    "Thresholding_rtl_d94a00": (0, 0, False),
    "Thresholding_rtl_db6103": (0, 0, False),
    "Thresholding_rtl_de5f51": (0, 0, False),
    "Thresholding_rtl_e3e7b8": (0, 0, False),
    "Thresholding_rtl_ed6281": (0, 0, False),
    "Thresholding_rtl_f0ece6": (0, 0, False),
    "Thresholding_rtl_f410d1": (0, 0, False),
    "Thresholding_rtl_f913fe": (0, 0, False),
    "VVAU_hls_0d7b37": (0, 0, False),
    "VVAU_hls_2fcb61": (0, 0, False),
    "VVAU_hls_5776d2": (0, 0, False),
    "VVAU_hls_6e820d": (0, 0, False),
    "VVAU_hls_75b399": (10, 2, False),
    "VVAU_hls_a44ff2": (0, 0, False),
    "VVAU_hls_bf88d5": (10, 2, False),
    "VVAU_hls_fa4b75": (10, 2, False),
    "VVAU_hls_fbcd22": (0, 0, False),
}


def _node_refs():
    """{key: entry} over every reference file, node-replay entries only."""
    out = {}
    for name in sorted(os.listdir(REF_DIR)):
        if not name.endswith(".json"):
            continue
        refs = json.load(open(os.path.join(REF_DIR, name)))
        out.update({k: v for k, v in refs.items() if "spec" in v})
    return out


REFS = _node_refs()


def _inst(key):
    """The replayed node, or a skip if this FINN tree has no such operator.

    The reference base is shared with the finn-plus fork, whose operator set is
    not identical -- there is no ChannelwiseOp_hls or ScaledDotProductAttention
    here, and Pool_hls has no AccPool function. Those references are carried so
    that the two trees can be scored against the same evidence, and skipped
    here rather than deleted.
    """
    ensure_finn_env()
    try:
        return build_node(REFS[key]["spec"])
    except Exception as e:
        pytest.skip("%s: not available in this FINN tree (%s)" % (key, type(e).__name__))


@pytest.mark.tav_tree_model
@pytest.mark.parametrize("key", sorted(REFS) or ["__no_references__"])
def test_node_tree_model_token_counts(key):
    """One period moves exactly one folded input and one folded output.

    Independent of the recorded schedule: whatever shape a tree model gives the
    period, a node that reads or writes the wrong number of tokens in it is
    wrong, and the sizer's occupancy sum accumulates that error every frame.
    """
    if not REFS:
        pytest.skip("no references")
    inst = _inst(key)
    if inst.get_tree_model() is None:
        pytest.skip("%s: no tree model" % key)
    tav_in, tav_out = tree_tavs(inst)
    n_in = int(np.prod(inst.get_folded_input_shape()[:-1]))
    n_out = int(np.prod(inst.get_folded_output_shape()[:-1]))
    assert int(as_rows(tav_in)[0][-1]) == 2 * n_in, "%s reads %d over two periods, folded input %d" % (
        key,
        int(as_rows(tav_in)[0][-1]),
        n_in,
    )
    assert int(as_rows(tav_out)[0][-1]) == 2 * n_out, (
        "%s writes %d over two periods, folded output %d"
        % (key, int(as_rows(tav_out)[0][-1]), n_out)
    )


@pytest.mark.tav_tree_model
@pytest.mark.parametrize("key", sorted(REFS) or ["__no_references__"])
def test_node_tree_model_matches_rtlsim(key):
    """Row 0 -- the schedule that sets FIFO depths -- against its recorded budget."""
    if not REFS:
        pytest.skip("no references")
    inst = _inst(key)
    tree = inst.get_tree_model()
    if key not in NODE_BUDGETS:
        # A budget is the record of a measurement. An op type that acquires a
        # tree model without one would be checked against nothing, so make that
        # the failure rather than a silent pass.
        assert tree is None, (
            "%s has a tree model but no recorded budget -- score it with "
            "ci/experiments/score_nodes.py --emit-budgets and paste the result in" % key
        )
        pytest.skip("%s: no tree model yet" % key)
    assert tree is not None, "%s has a budget but returns no tree model" % key

    entry = REFS[key]
    errs = []
    for tav, name in zip(tree_tavs(inst), ("io_chrc_in", "io_chrc_out")):
        t = as_rows(tav)[0].astype(np.int64)
        ref = as_rows(read_tav(entry[name])).astype(np.int64)
        n = min(t.size, ref.shape[-1])
        errs.append((int(np.abs(t[:n] - ref[0, :n]).max()), int(t.size), int(ref.shape[-1])))
    (in_err, in_len, ref_len), (out_err, _, _) = errs
    budget_in, budget_out, period_exact = NODE_BUDGETS[key]
    if period_exact:
        assert in_len == ref_len, "%s: tree period %d vs rtlsim %d" % (key, in_len, ref_len)
    assert in_err <= budget_in, "%s: input TAV error %d > budget %d" % (key, in_err, budget_in)
    assert out_err <= budget_out, "%s: output TAV error %d > budget %d" % (key, out_err, budget_out)


@pytest.mark.tav_tree_model
@pytest.mark.parametrize("key", sorted(REFS) or ["__no_references__"])
def test_vectorised_traversal_matches_the_loop(key):
    """``cumulative`` must be cycle-for-cycle what ``traverse_phase_tree`` gives.

    The vectorised path is what makes a 400k-cycle period cost milliseconds
    instead of a second, and it is only safe as long as it is an optimisation
    rather than a second implementation.
    """
    if not REFS:
        pytest.skip("no references")
    inst = _inst(key)
    tree = inst.get_tree_model()
    if tree is None:
        pytest.skip("%s: no tree model" % key)
    loop_in, loop_out = [], []
    _, _, loop_in = tree.traverse_phase_tree(0, 0, 0, loop_in)
    _, _, loop_out = tree.traverse_phase_tree(1, 0, 0, loop_out)
    vec = tree.cumulative(periods=1)
    assert np.array_equal(np.array(loop_in), vec[:, 0]), "%s: input schedules differ" % key
    assert np.array_equal(np.array(loop_out), vec[:, 1]), "%s: output schedules differ" % key
