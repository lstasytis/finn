"""Build a fresh reference estimate (committed folding, estimate only) of a feature test under a
name that the tests' cache lookup finds first, while running jobs keep using the old one.

    source tools/wtenv.sh build/wt/<wt> && python regen_reference_estimate.py <model dir> <param> <small|full>

The tests' ``cached_build`` takes the first ``$FINN_BUILD_DIR`` entry (sorted by name) whose name
starts with the model's ``est_*`` prefix. This builds the estimate as ``<prefix>00000021`` (sorted
before every random ``mkdtemp`` suffix), with the test module's own ``reference_estimate`` under
its file lock. Move the stale dirs aside once no running job uses them (issue 023).
"""
import importlib.util
import os
import sys

SUFFIX = "00000021"


def main():
    mdir, param, size = sys.argv[1:4]
    root = os.environ["FINN_ROOT"]
    build_dir = os.environ["FINN_BUILD_DIR"]
    test_dir = os.path.join(root, "tests", "benchmark", mdir)
    sys.path[:0] = [test_dir, os.path.join(root, "tests", "benchmark")]
    os.chdir(test_dir)
    spec = importlib.util.spec_from_file_location("ref_test", f"test_build_{mdir}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def make_build_dir(prefix):
        path = os.path.join(build_dir, prefix + SUFFIX)
        os.makedirs(path)
        return path

    found = []
    mod.find_cached_build = lambda prefix, build_dir=None: found.append(prefix)  # always rebuild
    mod.make_build_dir = make_build_dir
    board = next(b for b in ("AUP-ZU3_8GB", "KV260_SOM", "ZCU104", "U55C", "VCK190", "V80")
                 if param.startswith(b))
    model = param[len(board) + 1:]
    small = size == "small"
    out = mod.reference_estimate(board, model, small=small) if model else mod.reference_estimate(
        board, small=small)
    first = sorted(e for e in os.listdir(build_dir)
                   if e.startswith(found[0]) and os.path.isdir(os.path.join(build_dir, e)))[0]
    assert os.path.join(build_dir, first) == out, f"{first} sorts before {out}"
    print(out)


if __name__ == "__main__":
    main()
