# `tav_eval` — token-access-vector evaluation harness

A small driver for AlphaEvolve-style search over FINN node `get_tree_model`
functions. Given a node and a candidate `get_tree_model`, it splices the
candidate into the node's source, runs the node's analytical-characterization
pytest inside the FINN docker container, and emits a per-test-case log of the
**parameters**, the **pass/fail verdict**, and the **element-wise delta** between
the analytically-modeled token access vector (TAV) and the rtlsim reference TAV.

## What it does

For `evaluate_tree_model(node, tree_model_path)`:

1. **Patch** — finds the first `get_tree_model` definition in `tree_model_path`
   (it may be at module level or inside a class) and splices it into
   `src/finn/custom_op/fpgadataflow/<node>.py`, re-indented to sit on the class.
   The pristine file is saved once as `<src>.tav_orig`.
2. **Run** — executes the node's characterization test via `docker exec` in the
   running `finn_dev_<user>` container, loading the `_tav_eval_plugin` pytest
   plugin. The plugin forces the **rtlsim reference to be served from cache**
   (`$FINN_BUILD_DIR`), so rtlsim runs at most once per parameter set; the
   analytical tree is recomputed every run.
3. **Log** — writes `tav_eval.log` (and the raw `pytest.out`) under the mounted
   build dir `$FINN_HOST_BUILD_DIR` (default `/tmp/finn_dev_<user>/tav_eval/...`),
   which is writable from inside the container. Returns the log path. With `-v`
   the log contents are also printed.

## Log format

One line per parametrized test case:

```
[PASS] <nodeid> | <param>=<val> ... | input: len_a=.. len_rtl=.. len_delta=.. peak=.. delta=[...] | output: ...
[FAIL] <nodeid> | ...                | input: ... delta=[0, 0, 3, 0, ...] | output: ...
[ERROR] <nodeid> | ...               | <exception>
[SKIP] <nodeid> | ...
```

* `delta` is the new vector `analytical_TAV - rtlsim_TAV` (element-wise over the
  common length). All-zero ⇒ exact match.
* `len_delta` is `len(analytical) - len(rtlsim)`.
* `ERROR` = the test raised before the comparison (e.g. no rtlsim reference and
  no simulator available); `FAIL` = the TAVs were compared and differed beyond
  the test's tolerance.

## Usage

```bash
# CLI
python tools/tav_eval/tav_eval.py FMPadding my_candidate.py        # prints log path
python tools/tav_eval/tav_eval.py FMPadding my_candidate.py -v     # also prints the log
python tools/tav_eval/tav_eval.py --list                           # list known nodes
python tools/tav_eval/tav_eval.py FMPadding --restore              # revert to baseline source

# import
from tav_eval import evaluate_tree_model
log_path = evaluate_tree_model("FMPadding", "my_candidate.py", verbose=False)
```

For a node not in the registry, pass `--src <source.py>` and
`--test <path::test_func>` explicitly.

## Requirements / notes

* A long-running `finn_dev_<user>` container. The harness starts one
  automatically (detached, `sleep infinity`) if none is running, and waits for
  the entrypoint to finish installing the editable deps. For the image it
  reuses an existing `xilinx/finn` image if one is present, otherwise it builds
  one via `run-docker.sh` (first run only). The image tag is pinned **without**
  the `git describe --dirty` suffix, because splicing in a candidate
  `get_tree_model` dirties the working tree — otherwise the tag would float and
  never match the built image. Override with `FINN_DOCKER_TAG=...` if needed.
* **rtlsim cache.** Generating the rtlsim reference needs the Xilinx simulator
  (XSI/Vivado). In an environment without it, the first run reports `ERROR`
  (`'NoneType' object has no attribute 'compile_sim_obj'`). Populate the cache by
  running the characterization test once where rtlsim works; thereafter the
  harness reuses `model_rtlsim.onnx` from `$FINN_BUILD_DIR` and only the
  analytical tree is re-evaluated.
* **No FINN source is modified by the instrumentation.** Capture is done with a
  runtime monkeypatch inside the pytest plugin, so an optimizer rewriting `src/`
  never collides with the measurement code.
* `_tav_eval_plugin` stubs out `torchvision` if it is absent (this image ships no
  torch/torchvision/CUDA stack); it is only used by unrelated image-resize
  helpers in `finn.util.test`, not by the TAV path.

## Files

* `tav_eval.py` — the harness (CLI + `evaluate_tree_model`).
* `_tav_eval_plugin.py` — pytest plugin: caching override + TAV capture.
* `examples/fmpadding_tree_model.py` — example candidate `get_tree_model`.
