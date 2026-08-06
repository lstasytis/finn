# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The SWG configuration matrix the tree models are measured on.

Three tiers, cheapest first:

* ``pytest``  -- exactly the parametrisation of
  ``test_fpgadataflow_analytical_characterization_slidingwindow``, minus the
  combinations that test skips. Small feature maps, seconds to run, and the set
  the committed model is known to pass on.
* ``models``  -- the sliding windows of mobilenet_v1 and resnet50 as the
  benchmark builds fold them. These are the ones whose FIFO depth is being
  protected, so a regression here is the regression that matters.
* ``stress``  -- shapes that exist to break closed forms: dilation, stride >
  kernel, 1xN feature maps, SIMD < IFMCh on a depthwise window.

A configuration is a plain dict so it survives json round-trips into the
golden file.
"""


def _cfg(ifm_dim, k, stride, dilation, ifm_ch, simd, dw, pw, m=1, impl="rtl"):
    return dict(
        ifm_dim=list(ifm_dim),
        k=list(k),
        stride=list(stride),
        dilation=list(dilation),
        ifm_ch=ifm_ch,
        simd=simd,
        dw=dw,
        parallel_window=pw,
        m=m,
        impl=impl,
    )


def _legal(c):
    """The skip rules of the convinputgenerator pytest, in one place."""
    (k_h, k_w), (ifm_h, ifm_w) = c["k"], c["ifm_dim"]
    (s_h, s_w), (d_h, d_w) = c["stride"], c["dilation"]
    kh = (k_h - 1) * d_h + 1
    kw = (k_w - 1) * d_w + 1
    if c["ifm_ch"] % c["simd"] != 0:
        return False
    if kh > ifm_h or s_h > ifm_h or kw > ifm_w or s_w > ifm_w:
        return False
    if (k_h == 1 and d_h != 1) or (k_w == 1 and d_w != 1):
        return False
    if ((s_h > k_h) or (s_w > k_w)) and not (c["parallel_window"] or (k_h == 1 and k_w == 1)):
        return False
    if (
        c["parallel_window"]
        and c["simd"] != c["ifm_ch"]
        and not (c["dw"] or (k_h == 1 and k_w == 1))
    ):
        return False
    return True


def pytest_matrix():
    out = []
    for k in ([2, 2], [3, 3], [1, 5]):
        for ifm_dim in ([8, 8], [1, 21]):
            for ifm_ch in (2, 4):
                for stride in ([1, 1], [2, 2], [2, 1]):
                    for dilation in ([1, 1], [2, 2], [2, 1]):
                        for simd in (1, 2, 4):
                            for dw in (0, 1):
                                for pw in (0, 1):
                                    c = _cfg(ifm_dim, k, stride, dilation, ifm_ch, simd, dw, pw)
                                    if _legal(c):
                                        out.append(c)
    return out


def model_matrix():
    """The sliding windows of the models the FIFO totals are guarded on.

    **Read off real builds, not transcribed.** Every entry was extracted from
    the node attributes of a completed
    ``swg_model_sizes.py build`` -- mobilenet_v1 on the ZCU104 configuration
    (standalone thresholds, the folding config the deployed model uses) and
    bnn-pynq cnv-w2a2. So the padded ``IFMDim`` (113 rather than 112, because
    FMPadding runs first), the odd SIMD values the folding config actually
    assigns, and the fused 9x9/7x7 tail are all what the sizer really sees.

    Re-extract after any folding-config or streamlining change with the
    snippet in ``README.md``; a transcribed shape here would quietly guard the
    wrong thing.
    """
    out = []
    # mobilenet_v1 224x224 w4a4, ZCU104 folding: a 3x3 s2 head on 3 channels,
    # then 13 depthwise 3x3 windows whose SIMD falls as the channel count rises
    mbnet = [
        # (ifm_dim, k, stride, ifm_ch, simd, depthwise)
        ((224, 224), [3, 3], (2, 2), 3, 1, 0),
        ((113, 113), [3, 3], (1, 1), 32, 16, 1),
        ((113, 113), [3, 3], (2, 2), 64, 8, 1),
        ((58, 58), [3, 3], (1, 1), 128, 16, 1),
        ((58, 58), [3, 3], (2, 2), 128, 4, 1),
        ((30, 30), [3, 3], (1, 1), 256, 8, 1),
        ((30, 30), [3, 3], (2, 2), 256, 2, 1),
        ((16, 16), [3, 3], (1, 1), 512, 4, 1),
        ((16, 16), [3, 3], (2, 2), 512, 1, 1),
        ((9, 9), [3, 3], (1, 1), 1024, 2, 1),
        ((7, 7), [7, 7], (1, 1), 1024, 1, 1),
    ]
    # bnn-pynq cnv-w2a2: non-depthwise 3x3 windows plus the two k=2 s=2
    # depthwise ones the pooling layers lower to, all at SIMD < IFMCh
    cnv = [
        ((32, 32), [3, 3], (1, 1), 3, 3, 0),
        ((30, 30), [3, 3], (1, 1), 64, 16, 0),
        ((28, 28), [2, 2], (2, 2), 64, 1, 1),
        ((14, 14), [3, 3], (1, 1), 64, 16, 0),
        ((12, 12), [3, 3], (1, 1), 128, 16, 0),
        ((10, 10), [2, 2], (2, 2), 128, 1, 1),
        ((5, 5), [3, 3], (1, 1), 128, 8, 0),
        ((3, 3), [3, 3], (1, 1), 256, 8, 0),
    ]
    for ifm_dim, k, stride, ifm_ch, simd, dw in mbnet + cnv:
        out.append(_cfg(ifm_dim, k, stride, [1, 1], ifm_ch, simd, dw, 0))
    return [c for c in out if _legal(c)]


def stress_matrix():
    out = [
        # 1x1 windows -- the parallel style, and the pass-through closed form
        _cfg([16, 16], [1, 1], [1, 1], [1, 1], 8, 2, 0, 0),
        _cfg([16, 16], [1, 1], [2, 2], [1, 1], 8, 4, 0, 0),
        # stride > kernel, only legal with parallel_window
        _cfg([16, 16], [2, 2], [3, 3], [1, 1], 4, 4, 0, 1),
        _cfg([12, 12], [2, 2], [3, 3], [1, 1], 8, 8, 0, 1),
        # dilation
        _cfg([16, 16], [3, 3], [1, 1], [2, 2], 8, 4, 0, 0),
        _cfg([16, 16], [3, 3], [1, 1], [2, 1], 8, 8, 0, 0),
        # 1-D feature maps
        _cfg([1, 64], [1, 5], [1, 2], [1, 1], 16, 4, 0, 0),
        _cfg([1, 64], [1, 3], [1, 1], [1, 2], 16, 16, 0, 0),
        # depthwise with SIMD < IFMCh (the k2s2 closed form's domain)
        _cfg([16, 16], [2, 2], [2, 2], [1, 1], 32, 4, 1, 0),
        _cfg([32, 32], [2, 2], [2, 2], [1, 1], 64, 8, 1, 0),
        # large channel count, SIMD=1 -- the mobilenet tail case
        _cfg([7, 7], [7, 7], [1, 1], [1, 1], 1024, 1, 1, 0),
        # parallel_window depthwise
        _cfg([16, 16], [3, 3], [1, 1], [1, 1], 16, 4, 1, 1),
    ]
    return [c for c in out if _legal(c)]


MATRICES = {
    "pytest": pytest_matrix,
    "models": model_matrix,
    "stress": stress_matrix,
}


def get_matrix(name):
    if name == "all":
        seen, out = set(), []
        for fn in MATRICES.values():
            for c in fn():
                key = repr(sorted(c.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        return out
    return MATRICES[name]()
