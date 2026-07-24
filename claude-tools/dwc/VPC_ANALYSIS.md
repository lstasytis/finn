# `vpc.sv` — analysis vs. the generalized (padding) DWC

`vpc.sv` (`finn-rtllib/dwc/hdl/vpc.sv`, Thomas B. Preußer) is a **Vector Pack
Converter**: it re-packs a stream of `N`-element vectors from `PI` elements/beat to
`PO` elements/beat, elements being `W` bits. Each vector is carried independently in
`ceil(N/PI)` input beats and `ceil(N/PO)` output beats. It is a fully elastic RTL
(no combinational `ordy`→`irdy` path) and, unlike the legacy `dwc.sv`, it supports
**non-integer / coprime `PI:PO` ratios** through `gcd`-normalization.

This is the natural RTL counterpart to our generalized HLS DWC, so the question is:
**can it replace the generalized DWC (including padding), and how does it cost out on
non-multiple jobs?**

## 1. Does it cover the generalized-DWC padding functionality?

The generalized DWC supports three things in one node
(`claude-tools/dwc/CONTEXT.md`):

| feature | generalized HLS DWC | `vpc.sv` |
|---|---|---|
| width conversion, integer ratio | ✅ | ✅ |
| width conversion, **non-multiple / coprime** ratio | ✅ | ✅ (via `gcd(PI,PO)`) |
| **zero-padding** — `out_shape` has *more* elements than `in_shape` | ✅ | ❌ (only round-up of `N` to a `PO`-multiple) |
| **cropping** — `out_shape` has *fewer* elements than `in_shape` | ✅ | ❌ **not natively** |
| partial last beat (N not divisible by PI/PO) → excess lanes zeroed | ✅ | ⚠️ contract-level (producer must zero-fill) |

**Key structural limitation:** `vpc` has a *single* vector-length parameter `N`,
shared by input and output (`TRNI = ceil(N/PI)`, `TRNO = ceil(N/PO)` both derive from
the same `N`). The generalized DWC's padding/cropping is exactly the case where the
**input and output element counts differ** (`prod(in_shape) != prod(out_shape)` — e.g.
folding pads 44 channels → 48). `vpc` cannot express that: it is a pure *re-packer* of a
fixed-length vector, not a *padder/cropper* that changes the length.

What `vpc` *does* do that looks like padding: when `N` is not a multiple of `PO`, it
still emits `ceil(N/PO)` full output beats, so the delivered stream is effectively
`N` rounded **up to a multiple of `PO`**, with the trailing lanes zero (little-endian
packing). That is the one padding shape folding might actually want — *beat-alignment
padding to the output parallelism* — and `vpc` gets it for free. But it is **not** the
generalized DWC's padding: it cannot pad to an *arbitrary* `out_shape` (e.g. 44→48 when
`PO∤48`), because there is no independent output length — only `N` and `PO`. Two further
caveats: the padded length is forced to a `PO`-multiple (not free), and the RTL drives
all `PO0` output lanes straight from the buffer (`odat[p*GCD +: GCD] = Buf[p]`) with no
valid-lane masking, so the trailing zeros are only actually zero if the **producer**
zero-fills the excess lanes of its last input beat — `vpc` propagates that contract, it
does not enforce it.

**Conclusion:** `vpc.sv` can replace the generalized DWC on the *width-conversion*
axis (integer **and** non-multiple ratios) but **not** on the *padding/cropping* axis
that motivated the generalized DWC for the folding optimizer. To cover padding it would
need to be extended to independent input/output lengths (`NI`, `NO`) — see §4.

## 2. How it works (and why it costs what it does)

`GCD = gcd(PI,PO)`; the datapath works in normalized units of `GCD` elements
(`W0 = GCD·W` bit words, `PI0 = PI/GCD`, `PO0 = PO/GCD`). A single `CAP = PI0+PO0`
word buffer (total `= inWidth+outWidth` bits, invariant of `W`) is filled at the tail
and drained from the head:

- **Output** (`Phase 1`): retire `PO0` head words, shift the buffer down by `PO0`
  (constant shift — cheap). Output lanes are wired straight from `Buf[0 … PO0-1]`.
- **Input** (`Phase 2`): write `PI0` input words into the buffer starting at a
  **variable base offset** `ofs = ORdy (+PO0)`:
  `for p in 0…PI0-1: Buf[ofs+p] = idat[p]`.

That variable-offset write is a **`PI0`-wide × `CAP`-position placement crossbar**, and
its LUT cost scales **linearly with `PI0`** (the normalized *input* lane count). This
single fact drives the whole resource profile:

- **Upscaling** (`PI<PO`, e.g. plot 2): `PI0` is small → placement is cheap; output is
  just wiring. `vpc` stays **very cheap** as output width grows.
- **Downscaling** (`PI>PO`, e.g. plot 1): `PI0` is large → the placement crossbar
  **blows up linearly with input width**. `vpc` becomes expensive fast.
- There is **no integer-ratio fast path**: even a clean `k:1` multiple goes through the
  variable crossbar, so `vpc_mult` on downscale is far above both the legacy RTL and the
  generalized HLS DWC (which *do* have constant-shift fast paths).

Contrast the generalized HLS DWC: it places the new word with a **log-depth barrel**
(cost ≈ `bufwidth·log2(OutLanes)`), so it scales *logarithmically* in lane count and has
constant-shift fast paths for multiples. Asymptotically the HLS gearbox wins on wide /
high-ratio conversions; `vpc`'s linear crossbar only competes at small lane counts.

## 3. Non-multiple performance benchmark

`vpc` was synthesized (Vivado OOC, xc7z020, `W=1`, `PI=inW`, `PO=outW`,
`N=lcm(PI,PO)`) at the same width points as every other variant and added to
`dwc_resource_analysis.tex` as two lines (`vpc RTL (multiple)` / `(non-multiple)`).
Since `vpc` normalizes by `gcd(PI,PO)`, `W=1` yields the same minimal normalized
datapath as `W=gcd(inW,outW)` — i.e. the *fairest* realization of each job.

**LUTs (Vivado OOC, xc7z020).** `rtl` = legacy `dwc.sv` (multiples only), `gen` =
generalized HLS DWC, `vpc` = this module.

_Plot 1 — downscaling (output fixed at 10 bits):_

| in→out | rtl | gen_m | **vpc_m** | ‖ | in→out | gen_nm | **vpc_nm** |
|---|---|---|---|---|---|---|---|
| 100→10  | 59   | 151  | **543**   | ‖ | 105→10  | 291  | **1148**  |
| 500→10  | 507  | 559  | **8321**  | ‖ | 505→10  | 1113 | **16789** |
| 1000→10 | 1008 | 1071 | **31440** | ‖ | 1005→10 | 2189 | **61678** |

_Plot 2 — upscaling (input fixed at 10 bits):_

| in→out | rtl | gen_m | **vpc_m** | ‖ | in→out | gen_nm | **vpc_nm** |
|---|---|---|---|---|---|---|---|
| 10→100  | 13 | 85  | **50**  | ‖ | 10→105  | 261  | **160** |
| 10→500  | 16 | 296 | **116** | ‖ | 10→505  | 970  | **625** |
| 10→1000 | 16 | 550 | **174** | ‖ | 10→1005 | 1691 | **980** |

_Plot 3 — near 1:1 (both widths grow):_

| in→out | rtl | gen_m | **vpc_m** | ‖ | in→out | gen_nm | **vpc_nm** |
|---|---|---|---|---|---|---|---|
| 256→128  | 134 | 358  | **342**  | ‖ | 176→192 | 1237 | **1661** |
| 512→256  | 262 | 678  | **662**  | ‖ | 352→384 | 2147 | **2723** |
| 1024→512 | 518 | 1318 | **1302** | ‖ | 720→768 | 4667 | **6448** |

**Reading the numbers (LUTs):**

- **Downscale (plot 1): `vpc` loses catastrophically.** Cost grows ~linearly with the
  *input* lane count (the placement crossbar), reaching **31k / 62k LUT** at 1000/1005→10
  vs. ~1k / 2.2k for the HLS gearbox — **15–30× worse**, and it has no integer-ratio fast
  path so even the multiple line explodes.
- **Upscale (plot 2): `vpc` wins.** LUTs stay nearly flat (crossbar is `PI0`-small, output
  is pure wiring): **174 LUT** for a 10→1000 multiple vs. **550** for the HLS gearbox, and
  **980** for the 10→1005 non-multiple vs. **1691**. The legacy RTL is still cheapest on
  *multiples* (it's a plain shift register), **but it cannot do the non-multiples at all**,
  so for non-multiple upscaling `vpc` is the most compact option available.
- **Near 1:1 (plot 3): `vpc` ≈ HLS gearbox.** It tracks `gen_mult` almost exactly on
  multiples (1302 vs 1318 at 1024→512) and runs ~1.3–1.4× the `gen_nomult` gearbox on
  non-multiples (6448 vs 4667 at 720→768) — comparable, slightly worse.

**FFs** scale with the `inWidth+outWidth` buffer for every variant and are unremarkable
for `vpc` (39–1545 across the whole sweep) — the LUT crossbar, not the flip-flops, is the
cost story.

## 3b. Functional verification

Before trusting the resource numbers, `vpc` was checked for **functional equivalence** on
the no-length-change (pure width-conversion) job with a self-checking SystemVerilog
testbench (`ref_vpc/vpc_tb.sv`, run via `ref_vpc/run_sim.sh` on Vivado `xsim`). It streams
`VECS` back-to-back `N`-element vectors, packs them into `PI`-lane input beats and checks
that the `PO`-lane output beats reproduce the identical little-endian element sequence
(golden model = identity). A clocking block samples the DUT handshake in the preponed
region so the check is race-free.

**All configurations PASS**, each with *and* without pseudo-random AXIS backpressure on
both sides:

| ratio class | configs (W, N, PI→PO) | result |
|---|---|---|
| multiple down / up | (8,16,4→2), (8,16,2→4) | ✅ |
| non-multiple down / up | (8,20,5→2), (8,20,2→5) | ✅ |
| coprime | (4,42,3→7) | ✅ |
| 1:1 | (8,9,3→3) | ✅ |
| partial-beat padding | (8,10,4→4), (8,7,3→2) | ✅ |
| wide / high-ratio | (4,100,25→10), (8,480,480→15) | ✅ |

So for the non-padding case `vpc` is **bit-exact equivalent** to the other DWC variants
(same flat element stream out as in), for multiple, non-multiple, coprime, up-, down- and
1:1 conversions, robust to backpressure.

## 4. Recommendation / improvement opportunities

- **Do not** drop the generalized HLS DWC in favor of `vpc` for the folding padding
  use-case: `vpc` cannot change vector length (no pad/crop).
- `vpc` is attractive as a compact **RTL** path for **upscaling** and **small-lane-count
  non-multiple** conversions, where it beats the HLS gearbox and avoids HLS synthesis
  entirely. It is a poor fit for wide **downscaling** until the placement crossbar is
  optimized.
- Concrete improvements (in rough priority):
  1. **Integer-ratio fast path**: detect `GCD == PI` (downscale multiple) / `GCD == PO`
     (upscale multiple) and replace the variable-offset write with a constant shift, as
     the HLS variant does — removes the crossbar on all multiple cases.
  2. **Bounded / log-depth placement** for the non-multiple downscale to break the
     linear-in-`PI0` blow-up.
  3. **Independent `NI`/`NO`** (or an explicit pad/crop count) to actually cover the
     generalized DWC's padding/cropping.
