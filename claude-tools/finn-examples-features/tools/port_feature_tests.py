"""Port the per-model feature tests from the integration branch to one PR branch worktree.

    python port_feature_tests.py <src worktree> <dst worktree> fifo|folding|dwc

Copies the feature-test helper block of benchmark_helpers.py, appends each model's feature
block (only the tests of that feature; only models the destination has), registers the markers,
replaces the README's feature-flow section and deletes the old feature-flow harness.
On the folding/DWC branches (no analytical FIFO sizing) the FIFO strategy becomes dev's
"characterize".
"""
import os
import re
import subprocess
import sys

src, dst, feature = sys.argv[1:4]
B = "tests/benchmark"
MARK = {"fifo": "fifo_sizing", "folding": "folding", "dwc": "dwc"}[feature]
BLOCK_START = "# --- feature tests: FIFO sizing, folding, DWC padding"


def read(root, rel):
    with open(os.path.join(root, rel)) as f:
        return f.read()


def write(root, rel, text):
    with open(os.path.join(root, rel), "w") as f:
        f.write(text)


# helpers: replace the destination's import header + insert the helper block
h_src, h_dst = read(src, f"{B}/benchmark_helpers.py"), read(dst, f"{B}/benchmark_helpers.py")
if "# --- feature tests (folding" not in h_dst:
    head_src = h_src[: h_src.index("\n\ndef benchmark_root")]
    head_dst = h_dst[: h_dst.index("\n\ndef benchmark_root")]
    block = h_src[h_src.index("# --- feature tests (folding") : h_src.index("# --- benchmark aggregation")]
    h_dst = h_dst.replace(head_dst, head_src, 1)
    h_dst = h_dst.replace("# --- benchmark aggregation", block + "# --- benchmark aggregation", 1)
    write(dst, f"{B}/benchmark_helpers.py", h_dst)

for d in sorted(os.listdir(os.path.join(dst, B))):
    rel = f"{B}/{d}/test_build_{d}.py"
    if not os.path.isfile(os.path.join(dst, rel)) or not os.path.isfile(os.path.join(src, rel)):
        continue
    s_src, s_dst = read(src, rel), read(dst, rel)
    block = s_src[s_src.index(BLOCK_START) :]
    if BLOCK_START in s_dst:
        # block of a branch below (DWC is stacked on folding): add this feature's tests
        tests = [t for t in re.split(r"\n\n\n(?=@pytest\.mark\.slow\n)", block)[1:]
                 if f"@pytest.mark.{MARK}\n" in t]
        if f"@pytest.mark.{MARK}\n" not in s_dst:
            if "FOLDING_MAX_PADDING" in "".join(tests) and "FOLDING_MAX_PADDING =" not in s_dst:
                s_dst = s_dst.replace("# optimizer folding instead",
                                      "FOLDING_MAX_PADDING = 4\n# optimizer folding instead", 1)
            s_dst = s_dst.rstrip("\n") + "\n\n\n" + "\n\n\n".join(t.rstrip("\n") for t in tests) + "\n"
            write(dst, rel, s_dst)
        continue
    # keep the shared part and only this feature's test functions
    parts = re.split(r"\n\n\n(?=@pytest\.mark\.slow\n)", block)
    shared, tests = parts[0], [t for t in parts[1:] if f"@pytest.mark.{MARK}\n" in t]
    if feature != "fifo":
        shared = shared.replace('auto_fifo_strategy="heuristic_analytical"', 'auto_fifo_strategy="characterize"')
        shared = shared.replace("FIFOs sized analytically", "FIFOs sized by characterization")
    if feature == "fifo":
        shared = re.sub(r"FOLDING_MAX_PADDING = 4\n# optimizer folding.*?\n\)\n", "", shared, flags=re.S)
    elif feature == "folding":
        shared = shared.replace("FOLDING_MAX_PADDING = 4\n", "")
    text = "\n\n\n".join([shared.rstrip("\n")] + [t.rstrip("\n") for t in tests]) + "\n"
    s_dst = s_dst.rstrip("\n") + "\n\n\n" + text
    # imports: union of the source's benchmark_helpers import list, filtered to used names
    m_src = re.search(r"from benchmark_helpers import \((.*?)\)", s_src, re.S)
    m_dst = re.search(r"from benchmark_helpers import \((.*?)\)", s_dst, re.S)
    names = {n.strip() for n in m_src.group(1).split(",") + m_dst.group(1).split(",") if n.strip()}
    body = s_dst[m_dst.end():]
    names = sorted((n for n in names if re.search(r"\b%s\b" % n, body)),
                   key=lambda n: (not n.isupper(), n.lower()))
    s_dst = s_dst[: m_dst.start()] + "from benchmark_helpers import (\n" + "".join(
        f"    {n},\n" for n in names) + ")" + body
    if "warnings.warn" in s_dst and "\nimport warnings\n" not in s_dst:
        s_dst = s_dst.replace("\nimport os\n", "\nimport os\nimport warnings\n", 1)
    if "warnings." not in s_dst.replace("import warnings", ""):
        s_dst = s_dst.replace("import warnings\n", "")
    write(dst, rel, s_dst)

# markers (all three on every branch, identical lines avoid conflicts)
cfg = read(dst, "setup.cfg")
if "fifo_sizing:" not in cfg:
    lines = [l for l in read(src, "setup.cfg").splitlines() if re.match(r"    (fifo_sizing|folding|dwc):", l)]
    cfg = re.sub(r"(    finn_examples: mark tests that run finn examples\n)", r"\1" + "\n".join(lines) + "\n", cfg)
    write(dst, "setup.cfg", cfg)

# README section
r_src, r_dst = read(src, f"{B}/README.md"), read(dst, f"{B}/README.md")
sec = r_src[r_src.index("## Feature tests (FIFO sizing"):]
rows = [l for l in sec.splitlines() if l.startswith("| `")]
keep = [l for l in rows if l.startswith(f"| `{MARK}`") or l in r_dst]
for l in rows:
    if l not in keep:
        sec = sec.replace(l + "\n", "")
if feature != "fifo":
    sec = sec.replace("`heuristic_analytical`. Numbers", "dev's `characterize` strategy. Numbers")
for head in ("## Feature-flow tests", "## Feature tests (FIFO sizing"):
    if head in r_dst:
        r_dst = r_dst[: r_dst.index(head)] + sec
        break
write(dst, f"{B}/README.md", r_dst)

subprocess.run(["git", "-C", dst, "rm", "-q", "--ignore-unmatch"] + [
    f"{B}/{f}" for f in ["feature_flow_helpers.py", "feature_flow_report.py",
                         "test_fifo_sizing_flow.py", "test_folding_flow.py",
                         "test_dwc_padding_flow.py"]], check=True)
print("ported", feature, "->", dst)
