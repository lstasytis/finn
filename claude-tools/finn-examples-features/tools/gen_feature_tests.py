"""Append the per-model feature tests to tests/benchmark/<model>/test_build_<model>.py."""
import re
import sys

ROOT = sys.argv[1]

# dir, test name stem, build-dir tag, multi-model?, model file expr, steps var, builds expr, extra
MODELS = [
    ("bnn-pynq", "bnnpynq", "bnn-pynq", True, "get_model_file(model)", "None", "BNN_BUILDS", ""),
    ("cybersecurity-mlp", "cybersecuritymlp", "cybersecurity-mlp", False, "model_file", "None",
     '["AUP-ZU3_8GB"]', ""),
    ("gtsrb", "gtsrb", "gtsrb", False, "model_file", "None", '["AUP-ZU3_8GB"]', ""),
    ("kws", "kws", "kws", False, "model_file", "None", '["AUP-ZU3_8GB"]', ""),
    ("vgg10-radioml", "vgg10radioml", "vgg10-radioml", False, "model_file", "build_steps",
     '["ZCU104"]', ""),
    ("mobilenet_v1", "mobilenetv1", "mobilenet_v1", False, "model_file", "build_steps",
     '["U55C"]', "# ZCU104 is left out: its stitched-IP rtlsim gives X outputs (URAM init)\n"),
    ("espcn", "espcn", "espcn", False, "model_file", "espcn_build_steps", '["KV260_SOM"]', ""),
    ("resnet18", "resnet18", "resnet18", True, "get_model_file(model)", "resnet18_build_steps",
     "RESNET18_BUILDS", ""),
    ("yolov8", "yolov8", "yolov8", True, "get_model_file(model)", "yolov8_build_steps",
     "YOLOV8_BUILDS", ""),
    ("bert", "bert", "bert", False, "model_file", "bert_build_steps", '["VCK190"]', ""),
    ("resnet50", "resnet50", "resnet50", False, "model_file", "resnet50_build_steps", '["V80"]',
     ""),
]

HELPERS = [
    "ESTIMATE_OUTPUTS",
    "FEATURE_OUTPUTS",
    "cached_build",
    "estimated_throughput",
    "feature_metrics",
    "feature_steps",
    "resume_checkpoint",
    "step_maximize_folding",
    "timed_steps",
]

BLOCK = '''

# --- feature tests: FIFO sizing, folding, DWC padding -----------------------
# Each test resumes the convert-phase checkpoint of a cached estimate-only build
# (committed folding) with the feature's settings, builds the stitched IP and
# checks it in rtlsim: output equal to the golden output and throughput within
# 2 % of the build's own analytic estimate.

{extra}FEATURE_BUILDS = {builds_param}
FOLDING_MAX_PADDING = 4
# optimizer folding instead of the committed JSON, FIFOs sized analytically
OPTIMIZER_FOLDING = dict(
    folding_config_file=None,
    folding_style="optimizer",
    auto_fifo_depths=True,
    auto_fifo_strategy="heuristic_analytical",
)


def reference_estimate({args}):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build({args_pre}output_dir, generate_outputs=ESTIMATE_OUTPUTS, verify_steps=[])
        assert build.build_dataflow_cfg({model_file}, cfg) == 0

    return cached_build(
        f"est_{tag}_{{board}}_", build_estimate, "report/estimate_network_performance.json"
    )


def run_feature_build({args}, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate({args})
    output_dir = make_build_dir(f"{{name}}_{tag}_{{board}}_")
    cfg = configure_build(
        {args_pre}output_dir,
        steps=feature_steps({steps}),
        generate_outputs=ESTIMATE_OUTPUTS if estimate_only else FEATURE_OUTPUTS,
        verify_steps=[] if estimate_only else ["stitched_ip_rtlsim"],
        stitched_ip_gen_dcp=False,
        save_intermediate_models=True,
        **overrides,
    )
    # make_benchmark_cfg drops None values: apply them afterwards (e.g. no folding JSON)
    for key, value in overrides.items():
        if value is None:
            setattr(cfg, key, None)
    with timed_steps() as step_times:
        ret = build.build_dataflow_cfg(resume_checkpoint(ref_dir, output_dir), cfg)
    metrics = feature_metrics(output_dir, ref_dir, test=name, step_times_s=step_times)
    assert ret == 0, f"build failed, see {{output_dir}}/build_dataflow.log"
    if not estimate_only:
        assert metrics["verified"], "stitched-IP rtlsim output differs from the golden output"
        assert (metrics["rtlsim_vs_est"] or 0) >= 0.98, "rtlsim %s fps, estimate %s fps" % (
            metrics["rtlsim_fps"],
            metrics["est_fps"],
        )
    return metrics


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.fifo_sizing{group}
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize({params}, FEATURE_BUILDS)
def test_{stem}_fifo_sizing({args}, strategy):
{fifo_skip}    run_feature_build(
        {args}, f"fifo-{{strategy}}", auto_fifo_depths=True, auto_fifo_strategy=strategy
    )


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.folding{group}
@pytest.mark.parametrize("mode", ["target", "maximize"])
@pytest.mark.parametrize({params}, FEATURE_BUILDS)
def test_{stem}_folding({args}, mode):
    target_fps = int(estimated_throughput(reference_estimate({args})))
    if mode == "target":
        metrics = run_feature_build(
            {args}, "fold-target", target_fps=target_fps, **OPTIMIZER_FOLDING
        )
        assert metrics["est_fps"] >= target_fps
    else:
        maximize = {{"step_target_fps_parallelization": [step_maximize_folding]}}
        metrics = run_feature_build(
            {args}, "fold-max", inject_steps_before=maximize, **OPTIMIZER_FOLDING
        )
        if metrics["est_vs_ref"] < 1:
            warnings.warn("maximized folding is %.2fx the committed one" % metrics["est_vs_ref"])


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.dwc{group}
@pytest.mark.parametrize({params}, FEATURE_BUILDS)
def test_{stem}_dwc({args}):
    target_fps = int(estimated_throughput(reference_estimate({args})))
    padded = dict(
        OPTIMIZER_FOLDING, target_fps=target_fps, folding_maximum_padding=FOLDING_MAX_PADDING
    )
    # the folding search is cheap: skip the hardware build if nothing gets padded
    if not run_feature_build({args}, "dwc-est", estimate_only=True, **padded)["pad_dwcs"]:
        pytest.skip("no padding needed: the padded folding inserts no padding DWC")
    metrics = run_feature_build({args}, "dwc", **padded)
    assert metrics["est_fps"] >= target_fps
'''

for d, stem, tag, multi, model_file, steps, builds, extra in MODELS:
    path = f"{ROOT}/tests/benchmark/{d}/test_build_{d}.py"
    src = open(path).read()
    if "feature tests: FIFO sizing" in src:
        print("skip", path)
        continue
    if multi:
        args, args_pre, params = "board, model", "board, model, ", '"board,model"'
        builds_param = (
            "[\n    pytest.param(board, model, marks=pytest.mark.xdist_group(model))\n"
            f"    for board, model in {builds}\n]"
        )
        group = ""
        tag_f = tag + "_{model}"
    else:
        args, args_pre, params = "board", "board, ", '"board"'
        builds_param = builds
        group = f'\n@pytest.mark.xdist_group("{tag}")'
        tag_f = tag
    fifo_skip = ""
    if d == "resnet18":
        fifo_skip = (
            '    if model == "resnet18_8w8a":\n'
            '        pytest.skip("n/a: runtime-writeable URAM weights are not initialized in "\n'
            '                    "stitched-IP rtlsim")\n'
        )
    block = BLOCK.format(
        extra=extra, builds_param=builds_param, args=args, args_pre=args_pre, params=params,
        model_file=model_file, tag=tag_f, steps=steps, stem=stem, group=group,
        fifo_skip=fifo_skip,
    )
    src = src.rstrip("\n") + "\n" + block
    # imports
    m = re.search(r"from benchmark_helpers import \((.*?)\)", src, re.S)
    names = [n.strip() for n in m.group(1).split(",") if n.strip()]
    names = sorted(set(names + HELPERS), key=lambda n: (not n.isupper(), n.lower()))
    src = src[: m.start()] + "from benchmark_helpers import (\n" + "".join(
        f"    {n},\n" for n in names) + ")" + src[m.end():]
    if "\nimport warnings\n" not in src:
        src = src.replace("\nimport os\n", "\nimport os\nimport warnings\n", 1)
    open(path, "w").write(src)
    print("ok", path)
