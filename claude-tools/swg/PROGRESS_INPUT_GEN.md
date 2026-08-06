# Workstream B progress log — input_gen tree model

Terse status log. Newest at the bottom.

## 2026-08-05 10:50 — started
Confirmed tree: branch `feature/analytical-fifo-sizing` @ 43597d82, `finn`
resolves to `/home/lstasytis/backup/finn/src/finn`. Read handoff, TREE_MODELS,
README, `input_gen.sv`, `input_gen_ref.py`, `outer_shuffle.py`, `swg_configs.py`.

First reading of the RTL vs the reference: the pointer/counter/output-stage
transliteration in `simulate()` looks faithful line-for-line (irdy = Cap<0,
has_data = Rp-WpZ<0, advance = has_data && (!OVld||ordy), out beat = OVld&&ordy,
Cnt reload on `advance && term[i+1]`, the rp/fp mux taking the outermost
terminating level). BUF_SIZE = 1<<clog2(MAX_OCCUPANCY+3) matches.

Found already: **there is no standalone sliding-window custom op on input_gen in
this tree.** `input_gen.sv` is instantiated only inside `mvu_tiled_axi.sv`,
twice: activation replay `D=3 DIMS={NF,SF,TH} COEFS={0,1,SF} FM_SIZE=SF*TH`,
and output reorder `D=2 DIMS={TH,NF} COEFS={1,TH} FM_SIZE=NF*TH`. So the
conv -> DIMS/COEFS mapping in `input_gen_ref.py` is a *proposal*, not read off
an instantiation — that is the soft spot the handoff flagged, and it has no
authority to check against.

Next: validate the reference (esp. the 1x1 192-cycle period), then sweep.

## 2026-08-05 11:03 — reference validated, and a fast exact solver found
1. **`loop_nest_conv` is correct.** Checked against the RTL CIG's own documented
   data layouts (`convolutioninputgenerator_rtl.py` lines 49-51): non-depthwise
   `(OFMDim_H, OFMDim_W, K_H, K_W, IFMCh/SIMD, SIMD)`, depthwise
   `(OFMDim_H, OFMDim_W, IFMCh/SIMD, K_H, K_W, SIMD)` — exactly the two nests
   the function emits, and the coefficients are the matching input strides.
2. **The 192-cycle 1x1 period is real, not an off-by-one.** `MAX_OCCUPANCY`
   works out at 1 there, so `BUF_SIZE`=4 and only 3 words can be in flight,
   against a 3-cycle release round trip (accept -> WpZ -> advance -> Cap). The
   module throttles to 2 beats / 3 cycles. Same conv at SIMD=2 (buf=8) runs at
   full rate, which is the confirmation: it is a small-buffer effect, not a
   timing bug in the reference.
3. Derived the module's max-plus recurrence and solved it with numpy:
   `adv(k) = max(adv(k-1)+1, w(A(k))+2)`,
   `w(m) = max(w(m-1)+1, adv(kfree(m))+1)`.
   **Reproduces the literal transliteration exactly** — cumulative reads,
   cumulative writes and period identical on 200/200 pytest and 8/8 stress
   configurations — in 1-3 ms per config (the per-cycle simulator takes
   0.5-6 s on the big ones). 2-25 Kleene passes, 193 in the pathological
   4-entry-buffer 1x1 case.

That is the model's engine. Next: models tier validation, then wrap it as a
composite `Characteristic_Node` and count lines.

## 2026-08-05 11:11 — model landed, exact on the whole matrix
`claude-tools/swg/input_gen_model.py` + `input_gen_tav.py` written.

    python claude-tools/swg/input_gen_tav.py check
    checked 227 configs in 4.85s (0 out of the module's scope, worst tree 6542 leaves)
    worst undersize 0.0000  oversize 0.0000  period 0.0000
    0/227 configs outside budget

Not "within budget" — **exact** on cumulative reads, cumulative writes and
period, on all 227 in-scope configurations (200 pytest + 19 models + 8 stress),
including mobilenet windows at 330k cycles/frame.

**Headline line count: 160 lines of functions, 229 lines of file**; add ~35 for
the conv -> (DIMS,COEFS,FM_SIZE) mapping a real SWG op would carry, so ~195 end
to end. CIG: 1012.

The 157 remaining matrix entries are `parallel_window`, which `input_gen`
*cannot express* (one DATA_WIDTH for both ports; parallel_window's output word
is k*k times the input word). That is a module limitation, not a model one, and
it is the main argument against replacing the CIG wholesale.
`HANDOFF_INPUT_GEN_RESULT.md` written with the full verdict.

Next: lint, a per-function line audit, and a cross-check that the model's
re-derived elaboration agrees with the reference's.

## 2026-08-05 11:20 — validated against real RTL simulation
Verilator is absent but **Vivado 2023.1 (`xvlog`/`xelab`/`xsim`) is on the
path**, and `input_gen` needs no DSPs, so a two-signal testbench around it
compiles and runs in seconds. `claude-tools/swg/input_gen_rtl_check.py` does
that and diffs the per-cycle `irdy`/`ovld` trace against
`input_gen_ref.simulate()`:

    12 cases, 0 diverged

covering dilation, stride, depthwise, 1-D, the 1x1 buf=4 throttling case (the
one the handoff flagged as possibly an off-by-one — it is **real**), and the two
nests `mvu_tiled_axi.sv` actually instantiates. So the reference is no longer a
transliteration taken on trust: the chain is RTL -> reference -> tree model, and
both links are measured.

Also since the last entry:
- `input_gen_tav.py` gained `--matrix mvu` (the 192 real `mvu_tiled_axi`
  instantiations, `{NF,SF,TH}` replay and `{TH,NF}` reorder): **192/192 exact**.
- `input_gen_tav.py elab` cross-checks the model's re-derived elaboration
  against the reference's independent one: 0/576 disagree.
- `input_gen_tav.py fuzz`: 3000 random loop nests, **2428 matched exactly, 572
  declined, 0 wrong**.
- Fixed the one bug the mvu tier exposed: with a buffer larger than a feature
  map the writer runs whole frames ahead, so five simulated frames left the read
  vector short of the recorded window. Frames are now
  `5 + ceil((BUF_SIZE+2)/FM_SIZE)`. **The model now declines nothing in the
  entire matrix.**
- Lint clean (`black`, `isort`, `flake8`) on all three new files. Three
  pre-existing E501s elsewhere in this folder were left alone.

Next: fold the RTL result into `HANDOFF_INPUT_GEN_RESULT.md`, then look at
whether `mvau_tiled_params`'s fitted stall term agrees with the exact solve.

## 2026-08-05 11:22 — the tiled MVAU independently corroborates the model
`mvau_tiled_params` (matrixvectoractivation.py) carries a hand-derived buffer
and stall term for the activation replay `input_gen`:

    buf   = 1 << ceil(log2(SF*TH + 2))
    stall = max(0, ((TH-1)*SF + 4 - TH) - (buf - SF*TH))

Ran `schedule()` on `DIMS={NF,SF,TH}` over the full range that formula is
documented valid for (SF 2..64, NF 1..6, TH 2/3/6/9, 192 combinations). The
model's period is `NF*SF*TH + stall` with **that exact stall, every time**, and
its BUF_SIZE is that exact buf: `0/192 disagree, worst stall error 0`.

Two independent derivations from the same RTL agreeing to the cycle. Nothing was
changed in `matrixvectoractivation.py`; this is a measurement.

Result file updated with the RTL simulation, the mvu tier, the fuzz and this.
Next: README index, then a final read-through of the model.

## 2026-08-05 11:38 — the RTL check found a real defect in the reference
Ran `input_gen_rtl_check.py --random 60`: **one divergence**. Chased it, and it
is not a corner case.

`input_gen_ref.py` used unbounded Python integers for `Wp`, `WpZ`, `Rp` and
`Cap`. In the RTL all four are `ptr_t` = `logic signed [ADDR_BITS:0]`. For the
pointers the wrap is the intended circular arithmetic; **for `Cap` it is not** —
`Cap` is a counter whose sign bit *is* `irdy`. A nest that leaves part of its
feature map unread has to release that part in one lump at the frame boundary,
and `INIT_MAX_OCCUPANCY` does not bound that lump against `BUF_SIZE`. When the
lump is bigger than what is in the buffer, `Cap` underflows and `irdy` reads
back the wrong sign.

Confirmed against real RTL — for four such windows,
`RTL == unbounded reference: False`, `RTL == width-limited: True`.

**What the module then does, measured:**
- **deadlock** — completes exactly one frame and never accepts input again. 15 of
  the 227 in-scope matrix configurations, and **two of the eleven mobilenet_v1
  windows**: 58x58 k3 s2 dw SIMD 4 and 30x30 k3 s2 dw SIMD 2. Frame 1 ends at
  cycle 273518 / 285254 and nothing follows.
- **desynchronisation** — keeps running but accepts only 256 of each feature
  map's 512 words, so it is permanently out of step with its input stream.
  1 configuration (16x16 1x1 s2 SIMD 4).

Fixed `input_gen_ref.py` (a `ptr()` helper, three call sites). Re-ran the RTL
check with 200 random nests on top of the 12 named ones: **0 of 212 diverged**.

Re-dumped the golden: 227 -> 212 references (the 15 deadlockers never settle, so
the reference declines them, which is the honest answer). Added
`capacity_holds()` to the model, which declines the desynchronising one.

    check          211/211 exact, 1 declined, undersize 0 oversize 0 period 0
    check mvu      192/192 exact
    fuzz -n 3000   2396 exact, 604 declined, 0 wrong
    rtl_check      0 of 212 diverged

This changes the verdict materially and the result file needs rewriting: it is
no longer only that `input_gen` cannot express `parallel_window`, it is that it
**cannot run two of mobilenet_v1's eleven windows at all**.

Next: the coordinator's directive — derive the tree top-down from dims/coefs
instead of folding a materialised period.

## 2026-08-05 11:58 — the tree is now derived from the nest, not folded from a trace
Rewrote the model per the coordinator's directive. `block_pattern` reads
`dims`/`coefs` and hands back one iteration of a level; `Walk.step` solves that
one block by the same Kleene iteration, carrying only the accept times a later
block can still reach; `Walk.repeat` takes the rest of the level in one step
once a block hands on the state it was handed. **No period array exists.**

| shape | leaves | runs | depth | cycles | peak elements | blocks solved | time |
|---|---|---|---|---|---|---|---|
| [8,8] ch4 | 3 | 15 | 2 | 648 | 267 | 18 | 2 ms |
| [58,58] ch128 | 3 | 904 | 3 | 226652 | 7839 | 15 | 4 ms |
| [113,113] ch32 | 4 | 450 | 3 | 222174 | 3865 | 15 | 3 ms |
| [224,224] ch3 | 2 | 320 | 1 | 333976 | 10420 | 12 | 4 ms |

Same leaves/runs/depth as the folded version (the tree shape is preserved).
Peak intermediate array **1.35M -> 7839 elements** on [58,58], ~170x; whole
matrix **4.3 s -> 0.84 s**; per-config worst 2.84 s -> 4 ms.

One documented exception, as the directive allows: the levels *inside* a block
are still folded from that block's own cycles (`_fold`). The array is bounded at
`period / DIMS[level]`, and deriving them too needs the same carried state one
level down for blocks small enough that the bookkeeping would cost more than the
array. Everything outside that -- the loop that dominates the period -- is
derived.

The rewrite paid for itself twice over: the fuzz then found **two more real
defects in `input_gen_ref.py`**, both in the direction that undersizes.
1. `tav`'s window was fixed at four frames. `Cap` starts at `-BUF_SIZE+1`, so the
   writer holds `BUF_SIZE-1` words of credit it never gets back and spends it at
   one word per frame. A buffer bigger than a feature map takes tens of frames
   to drain that, and a window inside the transient reports **one read per frame
   too many** — a read-early bias. Worse, during that transient consecutive
   frames are bit-identical, so repetition alone does not detect it. Fixed:
   `settled_window` now requires token conservation (a settled frame reads
   exactly one feature map) as well as repetition, and `tav` grows the window
   until it holds.
2. The model itself had the mirror-image bug in its own frame-repeat test; the
   signature (accept queue included) now has to repeat too.

Full state:

    check              211/211 exact, 0 declines, undersize 0 oversize 0 period 0
    check mvu          192/192 exact
    fuzz -n 6000       4803 exact, 1197 declined, 0 wrong
    drift (4 periods)  pytest 187, mvu 192, models 17, stress 8 — 0 with drift
    elab               0/576 disagree
    rtl_check          0 of 162 diverged (12 named + 150 random)

Line count: **420 lines of file, 346 of functions** (was 229/160). The trade the
directive predicted; structure and the two defects it uncovered are worth it.
CIG: 1012.

Next: rewrite the result file with all of this.

## 2026-08-05 12:04 — done; everything green
Final sweep, all from a clean tree:

    check              211/211 exact, 0 declines, undersize 0 oversize 0 period 0   (0.88 s)
    check mvu --live   192/192 exact                                                (0.66 s)
    drift (4 periods)  pytest 187 / stress 8 / mvu 192 / models 17 — 0 with drift
    elab               0/576 disagree
    fuzz -n 6000       4803 exact, 1197 declined, 0 wrong
    rtl_check          0 of 162 diverged (12 named + 150 random, Vivado xsim)
    lint               black / isort / flake8 clean on all four files

`HANDOFF_INPUT_GEN_RESULT.md` rewritten in full. README index updated. Nothing
committed; everything is in the working tree.

Headline: **346 lines of functions / 420 of file**, CIG 1012. The model is a
derived composite nest, not a folded trace. The verdict is not the line count:
it is that `input_gen` **deadlocks on two of mobilenet_v1's eleven windows** and
cannot express `parallel_window` at all (41% of the matrix).

## 2026-08-05 12:22 — deadlock FIXED in input_gen.sv, validated under xsim
One-parameter change in `finn-rtllib/mvu_tiled/input_gen.sv`. `ptr_t` was sized
for the buffer alone:

    typedef logic signed [ADDR_BITS:0]  ptr_t;      // range [-BUF_SIZE, BUF_SIZE-1]

but `Cap` is not a pointer, it is a **counter whose sign bit is `irdy`**, and
when a nest leaves input rows unread the free pointer legitimately overtakes the
write pointer by one terminal increment, so `Cap` swings to
`-(BUF_SIZE-1) - TERMINAL_FP_INC`. `Rp - WpZ` swings the same way and drives
`has_data`. Both wrapped. Now:

    localparam int unsigned  PTR_BITS = 1 + $clog2(BUF_SIZE + MAX_ABS_INC + 1);
    typedef logic signed [PTR_BITS-1:0]  ptr_t;

**No storage cost** — the pointers still address the buffer with their low
`ADDR_BITS`, so `Buf` is unchanged; only the four pointer registers get a bit or
two wider. That was the point of chasing the right fix rather than growing
`BUF_SIZE` to swallow the lump, which would have doubled BRAM on the two
mobilenet windows.

Also strengthened the RTL check: the testbench now drives each instance's `idat`
with its own accepted-word counter and checks **`odat` against the loop nest's
own address** on every output beat. A handshake trace cannot catch a module that
hands back the right beats at the right cycles from the wrong addresses.

    0 of 20 cases diverged      (handshake AND data, all True)

and eight of those twenty are windows that previously hung after exactly one
frame — they now run 30+ frames with correct data.

Next: re-dump the golden (227 references expected, up from 211), drop the
model's `capacity_holds` decline, then work the rest of the decline paths.

## 2026-08-05 17:16 — no declines left where hardware exists
Went through every decline path and asked, for each, whether the reference says
hardware exists. Two had it, and both are now modelled:

1. **"addresses outside the feature map"** — a nest reading past `FM_SIZE`, i.e.
   into the next feature map's words. 1112 of 6000 random nests. The stream is
   continuous, so the module just reads them; it settles and conserves tokens.
   The guard was protecting a bookkeeping assumption, not a real limit. Removed
   it, and fixed the assumption it was hiding: `Walk`'s "forget accept times no
   later block needs" rule assumed the lowest future read is monotone in the
   block index. It is not — where the outer loop strides further than a feature
   map, the first block of the next frame reaches back behind the last block of
   this one. Now takes the minimum over the next `DIMS[level]` blocks, which is
   exact because a whole frame later every read is `FM_SIZE` higher.
2. **"blocks never settle"** — 571 more. Two causes. `_MAX_FRAMES` was 8 and the
   writer can take tens of frames to spend the credit `Cap` starts with (now 96;
   the fast-forward keeps it cheap). And, more interesting: **a period is not
   always one frame.** A two-beat nest on a four-entry buffer settles at five
   cycles covering *two* frames — the frames alternate 3 and 2 for ever. That is
   a perfectly good steady state. Both the model and `input_gen_ref.tav` only
   looked for one-frame periods and declined it.

   The model now detects the period by **the carried state repeating**, not by
   two frames matching, which finds multi-frame periods for free and is simpler
   than what it replaced. `settled_window` searches spans from one frame upward.
   One further fix: with a multi-frame period, *which* frame the window opens on
   is a rotation of the whole schedule, so the reference now takes the earliest
   qualifying window rather than the latest — the choice a model walking forward
   from reset can also make.

Result:

    fuzz -n 6000       6000 matched exactly, 0 declined, 0 wrong
    check              227/227 exact (was 211 before the RTL fix)
    check mvu          192/192 exact
    drift (4 periods)  pytest 200 / stress 9 / mvu 192 / models 19 — 0 with drift
    elab               0/576 disagree

The only decline left in `tree_model` is the free-pointer invariant, and it is
**unreachable**: `INIT_FP_INC` telescopes to exactly one frame of slots per
frame, and 40 000 random nests never reached it. It is kept so a future change
that broke the invariant declines rather than emitting nonsense.

On err-toward-oversizing: the model is bit-exact against a reference that is
bit-exact against the RTL, so there is no rounding choice to take a side on.
Where it does decline, the node falls back to rtlsim, which is ground truth --
the conservative direction, since a wrong tree can undersize and rtlsim cannot.

## 2026-08-05 17:25 — both directives done, everything green
Final state, all from a clean tree:

    check              227/227 exact, 0 declines, undersize 0 oversize 0 period 0
    check mvu --live   192/192 exact
    fuzz -n 6000       6000 exact, 0 declined, 0 wrong
    drift (4 periods)  pytest 200 / stress 9 / mvu 192 / models 19 — 0 with drift
    elab               0/576 disagree
    rtl_check          0 of 172 diverged, handshake AND data
    mobilenet 58x58    600000 cycles, 491125 beats, data True   (was: 1 frame, then hang)
    mobilenet 30x30    620000 cycles, 481753 beats, data True   (was: 1 frame, then hang)
    lint               black / isort / flake8 clean on all four Python files

RTL diff is 23 insertions / 4 deletions of which most is the comment explaining
why; the change itself is one `localparam` and one `typedef`. `input_gen.sv`
still parses clean and still elaborates at `mvu_tiled_axi.sv`'s own instantiation
parameters. Nothing committed.

`HANDOFF_INPUT_GEN_RESULT.md` now opens with the fix; README points at it.

---

# Workstream A assist: the SWG's coupled recurrence

## 2026-08-05 19:19 — derived, and it is cycle-exact
New file `claude-tools/swg/swg_coupled.py`. Only that file plus this log; I have
not touched `convolutioninputgenerator.py` or any of A's harnesses.

Read `swg_default_schedule` out into four inequalities. With `ivld`/`ordy` high,
`fetch(j)` the cycle beat `j` is fetched (it leaves one cycle later, `write_cmd`
being registered) and `read(m)` the cycle word `m` is accepted:

    fetch(j) >= fetch(j-1) + 1                  one fetch per cycle
    fetch(j) >= read(addr[j]) + 1               fetch_cmd needs current <= newest
    read(m)  >= read(m-1) + 1                   one read per cycle
    read(m)  >= fetch(EPW * (W(m) - 1)) + 1     read_ok needs oldest < first_next

`W(m) = min{w : T(w) > m - BUF_ELEM_TOTAL}`, `T` the `tail_incr` staircase that
`first_next` climbs at every EPW-th fetch. Kleene iteration from the never-waits
schedule; both sides monotone, so it climbs to the least fixed point.

Two things the FSM does that are not in those four lines, both measured:
- `read_ok` also wants `oldest < current`. **It never binds on its own** —
  instrumented over the whole matrix, every stalled read has `oldest <
  first_next` false and the `current` term is never the sole reason. Left out.
- A frame restarts only when *both* sides have finished it and the read side
  stops dead at `reading_done`, so where the write side finishes last,
  `read(first of f+1) >= fetch(last of f) + 2`. Without this the small configs
  diverged at the frame boundary and nothing else did.

Scored against `swg_fsm.swg_default_schedule`, four frames, every cycle:

    pytest  200 cycle-exact, 0 wrong, 154 parallel-style (out of scope)
    stress    7 cycle-exact, 0 wrong,   5 parallel-style
    models   19 cycle-exact, 0 wrong

**226/226 default-style configurations, cycle-identical.** 0.4 s for the biggest.

On A's config (`k3 ifm16 ch512 simd1 dw`) the solve reproduces the instrumented
FSM decomposition run for run:

    period 281502 = beats 225792 + gap 55710
    gap runs {511: 18, 7167: 2, 4596: 6, 4602: 1}
    head jumps 23532   row catch-up 32178

against A's hand count of 18x511 + 2x7167 = 23532 and 6x4596 + 4600 + 2 = 32178.
The 4602 is A's "4600 + 1 of 2" — one run, split by where A's window opened.

**One finding that matters for how A spends the next attempt:** the split into
two terms is not structural. I classified every gap by which constraint was
binding, and *every* gap in the frame is chain-bound — the beat is waiting for a
word that arrived as early as the throttled read stream allowed. There is no
population of gaps where the free pointer is the proximate cause; asking "free
pointer or demand?" returns "both, always". What is well defined is the run
*length*: a gap of exactly `HEAD_INCR_x - 1` is the address stepping out of
reach, anything else is the reader catching up across a row. That reproduces A's
split exactly, and it is the honest basis for scoring the two terms separately —
but it also says a sum of two independent closed forms is fitting two
populations of one phenomenon, which is why the `x epw` factor helped one family
and broke 22 others.

Next: numbers for the five failing layers, cost, and the parallel style.

## 2026-08-05 19:28 — both styles, whole matrix, and a note for A on integration
The parallel style is the same method with its own two gates — no line buffer,
so a read waits on `newest <= current` instead of `first_next`, and `write_ok`
is the transaction rather than a registered `write_cmd`. Both live in `gates()`;
everything else is shared.

    check --matrix all   384 cycle-exact (159 of them parallel-style), 0 wrong

`delta(p, style)` also reproduces `swg_fold.period_delta(aligned=True)`
**array-for-array** — 366/366 on pytest+stress, 19/19 on models — in 5.1 s where
the FSM takes 12.6 s. So it is a drop-in oracle as well as a source of terms.

413 lines. `black` / `isort` / `flake8` clean. Only `swg_coupled.py` and this
log were touched; `convolutioninputgenerator.py` and all of A's harnesses are
untouched.

### For A: what to take from this
- `swg_wait(p, style)["period"]` is the exact frame. If the nest's period is set
  from it, the models tier's 1.5-3% short periods go to zero by construction.
- `w["head"]` and `w["row"]` are the two populations the current two terms are
  predicting, split by run signature (a run of exactly `HEAD_INCR_x - 1` is the
  address stepping out of reach; anything else is the reader catching up).
  `w["runs"]` and `w["by_level"]` are the same decomposition an instrumented FSM
  gives, so a candidate law can be scored per level without re-instrumenting.
- **The caution:** the split is a labelling, not a mechanism. Every gap in the
  frame is chain-bound — I checked which constraint binds for each one, and it
  is never the free pointer directly. So `head` and `row` are two populations of
  one phenomenon, and a sum of two independent closed forms will keep behaving
  the way the last four attempts did: fixing one family by breaking another.
- Cost: 0.4 s median, 1.5 s worst on the models tier, dominated by generating
  `addr[j]` one beat at a time through `SwgController` plus 5-224 Kleene passes.
  That is cheaper than the FSM it replaces but it is not a closed form. If it is
  wanted inside `get_tree_model()`, that is the price; if it is wanted as a
  derivation aid, it is free.

### The exact targets, for the depthwise family that is failing
`period = beats + head + row`, all four exact from the coupled solve:

    layer                              period    beats     head      row  epw
    k3 ifm113 s1 ch32   simd16         222228   221778      448        2    9
    k3 ifm113 s2 ch64   simd8          231240   225792     2586     2862    9
    k3 ifm58  s1 ch128  simd16         226730   225792      936        2    9
    k3 ifm58  s2 ch128  simd4          238968   225792     5442     7734    9
    k3 ifm30  s1 ch256  simd8          228256   225792     1976      488    9
    k3 ifm30  s2 ch256  simd2          252988   225792    11230    15966    9
    k3 ifm16  s1 ch512  simd4          231620   225792     4344     1484    9
    k3 ifm16  s2 ch512  simd1          281502   225792    23532    32178    9
    k3 ifm9   s1 ch1024 simd2          239014   225792    10232     2990    9
    k7 ifm7   s1 ch1024 simd1           99282    50176    49104        2   49
    k2 ifm28  s2 ch64   simd1           55307    50176     2672     2459    4
    k2 ifm10  s2 ch128  simd1           16239    12800     1913     1526    4

On `k3 ifm16 s2` that is head 23532 (A's measurement exactly; A's law gives
15352) and row 32178 (A's `beats - epw` gives 32193, over by 15 — 7 rows, ~2
cycles each, which is the frame-wrap run being counted as a full row).

Note the stride-1 rows: `row` is 2 there, not `(rows-1) x (beats - epw)`. The
row catch-up only exists when the reader has to cross a stride gap. A law that
charges it unconditionally will over-predict every stride-1 depthwise layer,
which is the other half of why one factor cannot fix both families.

## 2026-08-05 19:46 — lead-in constructor, and the cap is not the bug
New file `claude-tools/swg/swg_lead_in.py` (321 lines). Only that and this log;
`convolutioninputgenerator.py` and A's harnesses untouched.

**Two facts, measured on all 384 configurations, 0 exceptions each:**
1. **Every write-idle cycle of a frame carries a read.** So a lead-in of
   `windup` cycles carries exactly `windup` reads — not approximately.
2. **Therefore `windup <= n_read`**, as a corollary: the idle cycles are a
   subset of the cycles that read.

So `min(windup, n_read)` **is a no-op** and capping the lead-in cannot be the
fix. That is why A's cap attempt made things worse rather than better — it was
adjusting the one quantity that was already right.

**Where the over-draw actually is.** `check --budget` reproduces A's read
accounting read-only and prints it: the rows' draws come from `TAIL_INCR_W/H`
and do not know about `windup`, so when the lead-in grows,
`budget = n_read - windup` shrinks under them and
`budget - n_first - (h-2)*n_mid` goes **negative** — the rows place more words
than the frame has. On the models tier that is **exactly six** configurations,
all stride-2 depthwise, which is exactly the 19/19 -> 13/19 A saw. Over the full
384 it is 13. The list is in `check --budget --matrix all`.

**What the module gives you.**

    lead_in_block(cycles, reads, writes) -> [(run, [rd, wr]), ...]

Exact on all three counts by construction — verified on 50 000 random triples,
0 violations — because both streams are laid on **one** cycle grid and then
run-length encoded together. That is the double-count the coordinator described:
neither stream is placed assuming the other is dense. Reads take the late edge
of their share and writes the early edge, so where the block cannot honour both
it reads later and writes sooner than a fair split, which is the over-sizing
direction.

    allocate(budget, weights, cap) -> per-row reads

Water-filling: sums to `min(budget, rows*cap)` exactly with every row inside
`cap`, spilling rather than clipping. 50 000 random cases, 0 violations. (My
first version clipped and failed 30% of them; caught by the property test, not
by the matrix.) **This is the half that fixes the six** — the rows scale to the
budget instead of to draws that ignore it.

    swg_frame_reads(p, windup, rows, per_row, cap) -> {lead_in, paced, rows, drain}

Sums to `n_read` exactly for any `windup` the solver produces. It takes A's own
row count, draw weights and cap; it does not re-derive the nest.

**Validation** — `check --matrix all`, 384 configurations:

    every write-idle cycle carries a read: 0 exceptions
    windup <= n_read (so the cap is a no-op): 0 exceptions
    lead_in_block exact on (cycles, reads, writes): 0 exceptions
    lead-in read curve == the real collected idle cycles: 0 exceptions
    swg_frame_reads sums to n_read: 0 exceptions

The fourth is the one the coordinator asked for: the candidate lead-in's
cumulative read curve is compared against **the real one** — the frame's
write-idle cycles collected in order, straight out of `swg_coupled.delta()` —
and they are identical, cycle for cycle, on every configuration.

### For A, concretely
`windup` and `paced` stay as they are. Two changes:
- `budget`, `n_first`, `n_mid`, `left` become one `allocate` call with the
  existing draws as weights and `cap = w*beats`, so the rows can never place
  more than `n_read - windup`;
- the `lead_in` composite becomes `_leaf("lead_in", lead_in_block(windup,
  windup, debt))`, which is the same block A already emits when `windup >=
  debt*(gap+1)` and a correct one when it is not.

Cost: both functions are closed-form — `allocate` is O(rows), `lead_in_block` is
O(windup) numpy and returns O(writes) runs. Neither calls the solver, so A's
"period-only via the solver, closed forms elsewhere" shape is preserved.

## 2026-08-05 20:17 — the shape defect: found, and it was mine
The composed frame's peak occupancy was wrong because **the lead-in was taking
`windup` reads.** That is right only if the whole wait sits at the head of the
frame. It does not: `windup = head + row`, and the `row` share is at the row
boundaries, spread through the frame. Lumping all of it at the head over-states
the head read burst, and the head read burst *is* the frame's peak occupancy.

I validated counts and the lead-in's own read curve and never validated the
composed frame's occupancy, which is the quantity the sizer keys on. That is the
hole the coordinator identified and it was exactly where the defect was.

**The fix, and it is exact rather than tuned.** Occupancy only jumps on a read,
so the peak is attainable at one, and it can be read straight off the solved
cycles without a period array:

    ahead[m] = (m + 1) - #{beats that left by read(m)}
    peak     = max(ahead)

`swg_coupled.swg_wait` now returns `peak` and `peak_cycle`. Verified against
`max(cum_reads - cum_writes)` over `delta()`: **384/384, 0 mismatches.**

Then, because occupancy is *non-increasing* through the rows — every row cycle
emits a beat and reads at most one word — the frame's peak **is** the value at
the end of the lead-in. So setting it there sets it exactly:

    paced   = clip(min(epw-1, beats-1), 0, min(windup, n_read) - peak)
    lead_in = peak + paced          (<= windup by construction)
    budget  = n_read - lead_in      -> allocate() over the rows

`paced` now gives way to the peak rather than the other way round: a borrowed
beat inside the lead-in cancels one of its reads, so it is capped at the room
left once the reads are placed. **A must use the returned `paced` for the
give-back too**, or the write count stops balancing.

**What it was costing.** Old lead-in peak vs the solver's exact peak, models tier:

    k3 ifm58  s2 ch128  simd4 dw    3770 -> 13168   +249%
    k3 ifm30  s2 ch256  simd2 dw    7930 -> 27188   +243%
    k3 ifm16  s2 ch512  simd1 dw   17402 -> 55702   +220%
    k2 ifm28  s2 ch64   simd1 dw    1855 ->  5128   +176%
    k3 ifm9   s1 ch1024 simd2 dw   10234 -> 13214    +29%   <- the +72% edge
    k3 ifm12  s1 ch128  simd16      146 ->     75    -49%
    k3 ifm5   s1 ch128  simd8        66 ->      0   -100%

The +29% on `ifm[9,9] k3 ch1024 simd2 dw` is the one A traced to a 51% peak error
and 93.8% of mobilenet's +72%. Note the bottom rows: several configurations were
*under*-stated, including two at -100%, which is the undersizing direction. The
fix corrects both directions; it is not a safety margin.

**New validation** (`check --matrix all`, 384 configurations):

    frame idle cycles == solver total_gap, all carrying a read: 0 exceptions
    lead_in_block exact on (cycles, reads, writes): 0 exceptions
    reads sum to n_read: 0 exceptions
    composed peak occupancy vs the solver: worst error 0
    composed cumulative reads / writes: worst 19173 / 32682

`compose()` builds the composed frame the way the nest emits it — lead-in, then
the beats one per cycle carrying what reads are left, then the drain — so the
occupancy check runs against the shape the nest really produces rather than
against the lead-in in isolation.

The remaining cumulative read/write divergence is the modelling approximation
that lumping the wait at the head *is*: the curves differ mid-frame while the
totals, the period and now the peak are exact. That is A's design choice and it
is visible rather than hidden; if it ever needs to go, the `row` share has to be
placed at the row boundaries instead of the head.

Signature change for A: `swg_frame_reads(p, windup, peak, ...)` takes the peak
now — `w = swg_wait(p, style)` gives both `w["total_gap"]` and `w["peak"]`.

**And use `lead_in_node(name, cycles, reads, writes)`, not the flat block.** A
sparse lead-in is one run per read: 34830 entries on `ifm16 s2 ch512 simd1 dw`.
`lead_in_node` builds the same block as a nest — a segment per borrowed beat,
each two repeated children — for **37 entries**, materialising to exactly the
same counts. It returns a `Characteristic_Node` ready to drop in where the
`_comp("lead_in", ...)` is now.

Two bugs the property tests caught on the way, neither of which the 384-config
matrix would have found — both were silent read/write *losses*, the undersizing
direction:
- `writes + 1 > cycles` made the segment split drop segments, losing beats;
- an even cut of the reads across segments hands a short segment more reads than
  it has cycles, and the clamp inside the segment then drops them.
Both now share out against each segment's own cycle count. 30 000 random cases
each on `lead_in_node` and `_spread_node`, 0 violations.

`swg_coupled` still 384/384 cycle-exact. `black`/`isort`/`flake8` clean on both
files. Nothing committed.
