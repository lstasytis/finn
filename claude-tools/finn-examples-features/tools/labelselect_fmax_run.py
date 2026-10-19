import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
P = sys.argv[1]
L = "/tmp/finn_dev_lstasytis/labelselect_fmax"
cfgs = [(lab, pe, dt) for lab, pe in [(10, 1), (10, 2), (10, 5), (10, 10), (43, 1), (43, 43),
        (1000, 1), (1000, 10), (1000, 100), (1000, 1000)] for dt in ["UINT8", "INT16"]]
def run(c):
    log = "%s/log_%d_%d_%s.txt" % (L, *c)
    with open(log, "w") as f:
        subprocess.run([sys.executable, P, str(c[0]), str(c[1]), c[2]], stdout=f, stderr=subprocess.STDOUT, cwd="/tmp")
    out = [l for l in open(log) if l.startswith("RESULT")]
    print(out[0].strip() if out else "FAILED %s" % (c,), flush=True)
with ThreadPoolExecutor(6) as ex:
    list(ex.map(run, cfgs))
