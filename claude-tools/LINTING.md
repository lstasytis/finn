# Linting — ready-to-use env (read before touching src/tests)

FINN CI enforces **pinned** `black==23.3.0` (`--line-length=100`), `isort==5.12.0`,
`flake8==6.0.0` (`--max-line-length=100 --extend-ignore=E203`). The system python3
does **not** have these installed, so keep a dedicated, gitignored env ready.

## The env (persistent, local, gitignored)

Location: **`/home/lstasytis/backup/finn/.lintenv`** (repo-local, added to
`.git/info/exclude` so it never shows up in `git status` or a PR — it is NOT in the
tracked `.gitignore`).

Recreate it in one line if it's ever missing (e.g. `.lintenv/bin/flake8` gone):

```bash
python3 -m pip install --quiet --target /home/lstasytis/backup/finn/.lintenv \
    black==23.3.0 isort==5.12.0 flake8==6.0.0
# and, once, so it stays out of git:
grep -qxF '.lintenv/' .git/info/exclude || echo '.lintenv/' >> .git/info/exclude
```

Note: `/home/lstasytis/backup/` (the repo's parent) is **not writable** — the env
must live inside the repo (`.lintenv/`), not beside it.

## Run it (before every commit that touches .py)

```bash
LENV=/home/lstasytis/backup/finn/.lintenv
FILES="path/to/file_a.py path/to/file_b.py"
PYTHONPATH=$LENV python3 $LENV/bin/black  --line-length=100 --check --diff $FILES
PYTHONPATH=$LENV python3 $LENV/bin/isort  --check-only --diff $FILES
PYTHONPATH=$LENV python3 $LENV/bin/flake8 --max-line-length=100 --extend-ignore=E203 $FILES
```

Drop `--check`/`--diff` (and `--check-only`) to auto-format in place. isort settings
(`.isort.cfg`) use `profile=black`, `known_first_party=finn`, `known_test=pytest`,
and custom section order `FUTURE,STDLIB,TEST,THIRDPARTY,FIRSTPARTY,LOCALFOLDER`, i.e.
`import pytest` groups first, then stdlib, then third-party, then `finn.*`, then
local-folder imports. For a `sys.path.insert(...)` before a sibling import, tag the
import with `# noqa: E402` (isort leaves it as its own block).
