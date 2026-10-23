#!/usr/bin/env python3
"""jobs.json for islands_driver.py: per model the island job and its Vivado baseline, all small
foldings first (small models first), then the full ones; a full island job only after its small
sibling passed, a baseline after its island job (same thread cap).

  python3 make_jobs.py <state dir>
"""

import json
import os
import sys

T = "tests/benchmark"
# (model id, test node prefix, board, threads small, threads full, verify frames)
MODELS = [
    ("tfc-w1a1", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-tfc-w1a1", 16, 16, 4),
    ("tfc-w1a2", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-tfc-w1a2", 16, 16, 4),
    ("tfc-w2a2", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-tfc-w2a2", 16, 16, 4),
    ("lfc-w1a1", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-lfc-w1a1", 16, 16, 2),
    ("lfc-w1a2", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-lfc-w1a2", 16, 16, 2),
    ("cnv-w1a1", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-cnv-w1a1", 16, 16, 1),
    ("cnv-w1a2", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-cnv-w1a2", 16, 16, 1),
    ("cnv-w2a2", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[AUP-ZU3_8GB-cnv-w2a2", 16, 16, 1),
    ("cnv-w1a2-kv260", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[KV260_SOM-cnv-w1a2", 16, 16, 1),
    ("cnv-w2a2-u55c", "bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_islands[U55C-cnv-w2a2", 16, 16, 1),
    ("cybersecurity", "cybersecurity-mlp/test_build_cybersecurity-mlp.py::test_cybersecuritymlp_islands[AUP-ZU3_8GB", 16, 16, 4),
    ("kws", "kws/test_build_kws.py::test_kws_islands[AUP-ZU3_8GB", 16, 16, 1),
    ("gtsrb", "gtsrb/test_build_gtsrb.py::test_gtsrb_islands[AUP-ZU3_8GB", 16, 16, 1),
    ("vgg10", "vgg10-radioml/test_build_vgg10-radioml.py::test_vgg10radioml_islands[ZCU104", 8, 32, 1),
    ("transformer_benchmark", "transformer/test_build_transformer.py::test_transformer_islands[ZCU104-transformer_benchmark", 16, 32, 1),
    ("mobilenet", "mobilenet_v1/test_build_mobilenet_v1.py::test_mobilenetv1_islands[U55C", 16, 32, 1),
    ("resnet18_4w4a", "resnet18/test_build_resnet18.py::test_resnet18_islands[U55C-resnet18_4w4a", 32, 32, 1),
    ("resnet18_8w8a", "resnet18/test_build_resnet18.py::test_resnet18_islands[U55C-resnet18_8w8a", 32, 32, 1),
    ("espcn", "espcn/test_build_espcn.py::test_espcn_islands[KV260_SOM", 16, 32, 1),
    ("yolov8_4w4a", "yolov8/test_build_yolov8.py::test_yolov8_islands[U55C-quantyolov8_4w4a", 32, 32, 1),
    ("yolov8_8w8a", "yolov8/test_build_yolov8.py::test_yolov8_islands[U55C-quantyolov8_8w8a", 32, 32, 1),
    ("transformer_radioml", "transformer/test_build_transformer.py::test_transformer_islands[ZCU104-transformer_radioml", 16, 32, 1),
    ("transformer_vision", "transformer/test_build_transformer.py::test_transformer_islands[ZCU104-transformer_vision", 16, 32, 1),
]


# random frames for the island-vs-RTL check next to the golden frame (cheap simulations only)
RANDOM_FRAMES = {m: 4 for m in ("tfc-w1a1", "tfc-w1a2", "tfc-w2a2", "lfc-w1a1", "lfc-w1a2",
                                "cybersecurity")}
RANDOM_FRAMES.update({m: 2 for m in ("cnv-w1a1", "cnv-w1a2", "cnv-w2a2", "cnv-w1a2-kv260",
                                     "cnv-w2a2-u55c", "kws", "gtsrb", "vgg10",
                                     "transformer_benchmark")})


# large designs: at least two lanes (MobileNet's islands timed out at 8 threads)
BIG = ("mobilenet", "resnet18_4w4a", "resnet18_8w8a", "yolov8_4w4a", "yolov8_8w8a")


# designs with HLS floating-point cores (encrypted IP): island flow with FINN_RWI_ALLOW_VENDOR_IP
VENDOR_IP = ("transformer_benchmark", "transformer_radioml", "transformer_vision", "resnet18_4w4a",
             "resnet18_8w8a", "yolov8_4w4a", "yolov8_8w8a")


def main():
    jobs = []
    for size in ("small", "full"):
        for mid, node, ts, tf, frames in MODELS:
            threads = ts if size == "small" else tf
            isl = "%s__%s__islands" % (mid, size)
            if mid == "cnv-w1a2-kv260" and size == "small":
                # (the first run shared the AUP-ZU3_8GB build's prep: a repeat of that one)
                isl += "_r"
            if mid == "transformer_benchmark" and size == "small":
                # (the run that validated FINN_RWI_ALLOW_VENDOR_IP)
                isl += "_vip2"
            after = [] if size == "small" else [["%s__small__islands" % mid, "pass"]]
            if mid == "cnv-w1a2-kv260" and size == "full":
                after = [["%s__small__islands_r" % mid, "pass"]]
            if mid == "transformer_benchmark" and size == "full":
                after = [["%s__small__islands_vip2" % mid, "pass"]]
            jobs.append(dict(id=isl, node="%s/%s-%s-islands]" % (T, node, size), threads=threads,
                             min_threads=16 if mid in BIG else 8,
                             frames=frames, after=after,
                             env=dict({"ISLANDS_VERIFY_RANDOM": RANDOM_FRAMES.get(mid, 0)},
                                      **({"FINN_RWI_ALLOW_VENDOR_IP": "1"} if mid in VENDOR_IP else {}))))
            jobs.append(dict(id="%s__%s__baseline" % (mid, size),
                             node="%s/%s-%s-baseline]" % (T, node, size), threads=threads,
                             same_threads_as=isl, after=[[isl, "done"]]))
    with open(os.path.join(sys.argv[1], "jobs.json"), "w") as f:
        json.dump(jobs, f, indent=1)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
