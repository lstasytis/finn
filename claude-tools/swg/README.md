# `claude-tools/swg/` — sliding-window tree-model workbench

Tooling and context for two parallel workstreams on the sliding-window
operators' FIFO tree models. Additive; nothing here belongs in a PR.

| file | what it is |
|---|---|
| [`TREE_MODELS.md`](TREE_MODELS.md) | **read first** — how the other operators' tree models are built, what has already been measured and rejected, and where the CIG stands |
| [`HANDOFF_CIG.md`](HANDOFF_CIG.md) | workstream A: shrink the 1012-line ConvolutionInputGenerator tree model |
| [`HANDOFF_INPUT_GEN.md`](HANDOFF_INPUT_GEN.md) | workstream B: build a tree model for `finn-rtllib/mvu_tiled/input_gen.sv`, the CIG's replacement |
| [`HANDOFF_INPUT_GEN_RESULT.md`](HANDOFF_INPUT_GEN_RESULT.md) | **workstream B's answer** — the line count, the coverage, and why `input_gen` does not simply replace the CIG |
| `swg_configs.py` | the configuration matrix: pytest parametrisation, mobilenet_v1 + resnet50 windows, stress cases |
| `swg_tav.py` | fast per-config harness — 384 configs in ~6 s, no vivado/rtlsim/ipgen |
| `swg_model_sizes.py` | model-level FIFO-kB guard on mobilenet_v1 / resnet50 |
| `input_gen_ref.py` | cycle-accurate Python reference for `input_gen.sv` (ground truth for workstream B) — **corrected**: `ptr_t` widths, and a settling window that conserves tokens |
| `input_gen_model.py` | the `input_gen` tree model — 430 lines, a derived composite nest, exact against that reference, declines nothing |
| `input_gen_tav.py` | its harness: `dump` / `check` / `show` / `drift` / `elab` / `fuzz`, whole matrix in ~1 s |
| `input_gen_rtl_check.py` | `input_gen.sv` under Vivado xsim — handshake *and* data — diffed against `input_gen_ref.py` |
| `goldens/` | frozen reference TAVs and model-level FIFO reports |

## The two loops

**Fast loop — per configuration, ~6 s:**

```bash
python claude-tools/swg/swg_tav.py check -g claude-tools/swg/goldens/base.npz
# checked 384 configs in 5.67s
# worst undersize 0.0000  oversize 0.0000  period 0.0000
# 0/384 configs outside budget
```

**The same fast loop for `input_gen` (workstream B):**

```bash
python claude-tools/swg/input_gen_tav.py check              # 227 configs in ~1 s
python claude-tools/swg/input_gen_tav.py drift             # error constant across frames
python claude-tools/swg/input_gen_tav.py fuzz -n 6000      # random nests, ~4 s
python claude-tools/swg/input_gen_rtl_check.py --random 150  # real RTL, needs Vivado, ~6 min
```

**Slow loop — per model, ~23 s once the checkpoint exists:**

```bash
python claude-tools/swg/swg_model_sizes.py build --model mobilenet_v1   # ~91 s, once
python claude-tools/swg/swg_model_sizes.py size  --model mobilenet_v1 -o cand.json
python claude-tools/swg/swg_model_sizes.py compare \
       -a claude-tools/swg/goldens/mobilenet_base.json -b cand.json
```

The default `check` thresholds are strict enough to catch a **single cycle** of
shift on every one of the 18 model configurations (verified by perturbing the
model with one prepended idle cycle). That is stricter than the task needs — a
few cycles of constant wind-up error is acceptable — so use `--const N` to
ignore divergences under `N` tokens whatever the fraction, and let the
fractions govern above it:

```bash
python claude-tools/swg/swg_tav.py check -g claude-tools/swg/goldens/base.npz \
       --const 8 --fail-under 0.01 --fail-over 0.10
```

## `input_gen.sv` was deadlocking, and is fixed

Workstream B found that `finn-rtllib/mvu_tiled/input_gen.sv` hung after exactly
one frame on any nest that leaves input rows outside its last window — 15 of the
227 in-scope matrix configurations, **two of them mobilenet_v1's own**. `ptr_t`
was sized for the buffer, but `Cap` is a counter whose sign bit is `irdy`, and a
terminal free increment swings it past that. The fix is one `localparam` and one
`typedef` (`PTR_BITS`), costs no storage, and is validated under Vivado xsim on
handshakes and data. It is in the working tree, uncommitted, and
[`HANDOFF_INPUT_GEN_RESULT.md`](HANDOFF_INPUT_GEN_RESULT.md) opens with it.

## The one rule

**A FIFO total that comes out lower than the baseline is a failure.** The
baseline is the depth the hardware was validated at, so a shrink is
undersizing, not an improvement. Both harnesses report divergence signed by
direction — `undersize` (reads earlier or writes later than the reference) must
stay at zero; `oversize` is the budget, single-digit percent.

## Baselines captured 2026-08-05

On `feature/analytical-fifo-sizing` @ 43597d82:

- `goldens/base.npz` — 384 configurations, all served by `swg_default_exact`.
  The `models` tier is now **read off the real builds**, not transcribed, so it
  carries the padded IFMDim (113, not 112) and the true per-layer SIMD.
- `goldens/mobilenet_base.json` — mobilenet_v1 **ZCU104**: 141 FIFOs, **15.5 kB**
  (98 s to build the checkpoint, 74 s per sizing re-run).
- `goldens/cnv-w2a2_base.json` — bnn-pynq cnv-w2a2 Pynq-Z1, 8 CIG nodes:
  34 FIFOs, **12.5 kB** (35 s to build, **3 s** per sizing re-run). This one has
  a board reference to check against: 12.967 / 13.031 kB, so the harness's
  accounting is verified, not just self-consistent.

### Getting the recipe right (three things that were wrong once)

1. **Folded width, not normal width.** A FIFO holds folded stream words; the
   normal shape's last dimension for a windowed tensor is `k*k*C`, which
   overstates storage by the folding factor -- 3686 kB instead of 88 kB on
   mobilenet. `fifo_report` uses `get_outstream_width()`, kB = bits/8/1024.
   cnv-w2a2 reproducing its 12.967 / 13.031 kB board reference is what
   validates this.
2. **`standalone_thresholds` must match the board.** With it off,
   MatMul+MultiThreshold fuse into the MVAU and the graph stops being the one
   the folding config was written for. On ZCU104 that surfaces as
   "MH divisable by PE is violated" in codegen and a DWC with a non-integer
   ratio in sizing. The benchmark test sets it per board (ZCU102/ZCU104 on,
   U250 off); `MODELS` mirrors that.
3. **The board matters more than it looks.** mobilenet_v1 on U250 folding is
   88 kB; on ZCU104 folding it is 15.5 kB. Same model, same sizer.

To re-extract the per-node configurations after any folding change:

```python
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
m = ModelWrapper("$FINN_BUILD_DIR/swg_fifo_mobilenet_v1/intermediate_models/"
                 "step_set_fifo_depths.onnx")
for n in m.graph.node:
    if n.op_type.startswith("ConvolutionInputGenerator"):
        i = getCustomOp(n)
        print(i.get_nodeattr("IFMDim"), i.get_nodeattr("ConvKernelDim"),
              i.get_nodeattr("Stride"), i.get_nodeattr("IFMChannels"),
              i.get_nodeattr("SIMD"), i.get_nodeattr("depthwise"))
```

**resnet50 cannot be used as a guard on this tree.** Its build dies well before
FIFO sizing, in `step_create_dataflow_partition`:

    AssertionError: cycle-free graph violated: partition depends on itself

That is the known pre-existing partition blocker (see
`claude-tools/fifo-sizing/CONTEXT.md`), nothing to do with tree models, and out
of scope for this work. `cnv-w2a2` is the second sliding-window-heavy model in
its place — fewer CIG nodes than resnet50 but eight of them, and a 2.7 s
iteration. If the partition blocker is ever cleared, `--model resnet50` is
already wired up.

Models are downloaded to `tests/benchmark/models/`
(`mobilenetv1-w4a4_pre_post_tidy_opset-11.onnx`, `cnv-w2a2.onnx`,
`resnet50_w1a2_exported.onnx`).
