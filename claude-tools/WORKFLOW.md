# Branch workflow: base + tools overlay + features, assembled on demand

The feature lines (FIFO sizing, finn-examples, label-aligner, generalized DWC,
folding) **genuinely conflict** when combined — they touch the same FINN code
(the FIFO characterization subsystem, MVAU codegen, …). A single ever-growing
mega-branch is therefore fragile. But the `claude-tools/` folder is **purely
additive** and never conflicts. Those two facts drive this layout.

## The layers

```
feature/expanded-finn-examples     <- BASE: our modern dev (+ benchmark harnesses)
        │
        ▼  additive overlay, merges into anything with zero conflicts
   claude-tools                     <- TOOLS: the claude-tools/ folder, hlslib
        │                              activations.hpp, benchmark harness fixes
        ▼  merged per environment, rerere replays recorded conflict resolutions
   env/<name>                       <- EPHEMERAL working env = BASE + TOOLS + feature(s)
```

Feature branches sit on the base and stay focused:

| branch | what it is |
|---|---|
| `feature/expanded-finn-examples` | base / modern dev + finn-examples + benchmarks |
| `claude-tools` | thin additive tools overlay (this folder) |
| `feature/analytical-fifo-sizing` | analytic tree-model FIFO sizer |
| `feature/label_aligner-clean` | **align node + insertion machinery only** (no MVAU/MMV edits) |
| `feature/generalized-datawidthconverter` | generalized DWC |
| `feature/set-folding-optimizer` | folding optimizer |

`env/*` branches are **disposable** — always rebuildable with `mkenv`, so never
hand-edit them for anything you want to keep; put durable work on a feature
branch or the tools overlay.

> `feature/label_aligner` (the original) also carries MVAU/thresholding MMV
> changes. `feature/label_aligner-clean` is the scoped version used for envs.

## Building an environment

```bash
# work on FIFO sizing (base + tools + fifo)
claude-tools/mkenv.sh env/fifo feature/analytical-fifo-sizing

# work on alignment (needs examples + tools + fifo + align)
claude-tools/mkenv.sh env/align \
    feature/analytical-fifo-sizing feature/label_aligner-clean

# several envs side by side, each in its own directory (git worktree)
claude-tools/mkenv.sh --worktree ../finn-fifo  env/fifo  feature/analytical-fifo-sizing
claude-tools/mkenv.sh --worktree ../finn-align env/align feature/analytical-fifo-sizing feature/label_aligner-clean
```

Every env gets the `claude-tools/` folder automatically, because the overlay is
merged first.

## Why it stays cheap: git rerere

`git rerere` ("reuse recorded resolution") records how you resolved a merge
conflict and **replays it automatically** the next time the identical conflict
appears. It is enabled repo-wide (`git config rerere.enabled true`) and has been
**pre-trained on the FIFO↔examples merge**, so `mkenv env/fifo …` resolves those
9 conflicts with no interaction.

When a feature branch changes such that a *new* conflict appears, `mkenv` stops
and asks you to resolve it once:

```bash
# ...resolve the reported files...
git add -A && git commit --no-edit     # rerere records it; next run is automatic
```

To re-teach a resolution from an existing good merge commit `<M>`:

```bash
git switch -c _train <first-parent-of-M>
git merge <second-parent-of-M>         # reproduce the conflict
git checkout <M> -- <conflicted-files> # apply the known-good result
git add -A && git commit --no-edit     # rerere records it
git switch - && git branch -D _train
```

## Updating the base / tools

- **Base** moves forward (upstream dev, new finn-examples): update
  `feature/expanded-finn-examples`, then rebuild the `env/*` you use. rerere
  replays the feature conflicts; only genuinely new ones need attention.
- **Tools** grow: commit to `claude-tools` (additive → never conflicts), then
  rebuild envs (or just `git merge claude-tools` into a live env).

## Notes

- `mkenv` assembles envs by **merge**, so rerere can do its job; it does not
  rebase. The `env/*` branches diverge from any remote of the same name — push
  them only if you deliberately want to (force). Feature branches and the tools
  overlay are the things you push normally.
- `deps/finn-hlslib` is gitignored and cloned by `fetch-repos.sh`. We do **not**
  git-track dep files (a branch that doesn't track them deletes them from the
  shared clone on checkout). Instead the overlay keeps complete reference copies
  under `claude-tools/hlslib/` and `claude-tools/patch_hlslib.sh` installs them
  into the clone (the `AlignLabels` template + the `activations.hpp` the clone
  shipped without). `mkenv.sh` runs it automatically; run it yourself after any
  fresh `fetch-repos.sh`.
