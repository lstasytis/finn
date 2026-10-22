"""Results of the issue-021 fold-max reruns: newest folding2/fold-max_<tag>_* build per model,
its feature metrics, budget verdict and the junit result.  python collect.py"""
import glob
import json
import os
import xml.etree.ElementTree as ET

FX = "/home/lstasytis/finn/build/finn_build/fx"
TAGS = {
    "bnn-pynq__AUP-ZU3_8GB-tfc-w1a1": "bnn-pynq_tfc-w1a1_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-tfc-w1a2": "bnn-pynq_tfc-w1a2_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-tfc-w2a2": "bnn-pynq_tfc-w2a2_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-cnv-w1a1": "bnn-pynq_cnv-w1a1_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-cnv-w1a2": "bnn-pynq_cnv-w1a2_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-cnv-w2a2": "bnn-pynq_cnv-w2a2_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-lfc-w1a1": "bnn-pynq_lfc-w1a1_AUP-ZU3_8GB",
    "bnn-pynq__AUP-ZU3_8GB-lfc-w1a2": "bnn-pynq_lfc-w1a2_AUP-ZU3_8GB",
    "kws__AUP-ZU3_8GB": "kws_AUP-ZU3_8GB",
    "cybersecurity-mlp__AUP-ZU3_8GB": "cybersecurity-mlp_AUP-ZU3_8GB",
    "gtsrb__AUP-ZU3_8GB": "gtsrb_AUP-ZU3_8GB",
    "bnn-pynq__KV260_SOM-cnv-w1a2": "bnn-pynq_cnv-w1a2_KV260_SOM",
    "espcn__KV260_SOM": "espcn_KV260_SOM",
}
print("| job | junit | est fps / ref | rtlsim / est | verdict | build |")
print("|---|---|---|---|---|---|")
for job, tag in TAGS.items():
    xml = f"{FX}/logs/campaign/v2__fold-max__{job}.xml"
    res = "pending"
    if os.path.isfile(xml):
        case = next(ET.parse(xml).getroot().iter("testcase"))
        res = "failed" if case.find("failure") is not None or case.find("error") is not None else (
            "skipped" if case.find("skipped") is not None else "passed")
    dirs = sorted(glob.glob(f"{FX}/folding2/fold-max_{tag}_*"), key=os.path.getmtime)
    if not dirs or res == "pending":
        print(f"| {job} | {res} | | | | |")
        continue
    d = dirs[-1]
    m = {}
    if os.path.isfile(f"{d}/report/feature_metrics.json"):
        m = json.load(open(f"{d}/report/feature_metrics.json"))
    verdict = ""
    for log in glob.glob(f"{d}/build_dataflow*.log"):
        for line in open(log, errors="ignore"):
            if "resource budget" in line:
                verdict = line.split("] ")[-1].strip()[:90]
    f = lambda k: f"{m[k]:.3f}" if isinstance(m.get(k), (int, float)) else "-"
    print(f"| {job} | {res} | {f('est_vs_ref')} | {f('rtlsim_vs_est')} | {verdict} | {os.path.basename(d)} |")
