"""Fill the verbatim code listings of fifo_sizing_analytical.tex.

    python build_tex.py <worktree of feature/dev/analytical-fifo-sizing>

reads fifo_sizing_analytical.tex.in (next to this script), replaces every line

    %%CODE <path relative to the worktree> <first line> <last line>%%

by an ``lstlisting`` of exactly those source lines (numbered as in the file)
and writes ../fifo_sizing_analytical.tex. The listings in the committed .tex
were produced from commit 4ce1bd188 of that branch (worktree
build/wt/pr-fifo).
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"^%%CODE (\S+) (\d+) (\d+)%%\s*$")


def tex_escape(s):
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_")


def listing(root, path, a, b):
    with open(os.path.join(root, path)) as f:
        lines = f.read().split("\n")
    body = "\n".join(lines[a - 1 : b])
    assert "\\end{lstlisting}" not in body
    name = tex_escape(path)
    return (
        "\\begin{lstlisting}[firstnumber=%d,title={\\texttt{%s}, lines %d--%d}]\n%s\n"
        "\\end{lstlisting}" % (a, name, a, b, body)
    )


def main(root):
    src = open(os.path.join(HERE, "fifo_sizing_analytical.tex.in")).read().split("\n")
    out = []
    for line in src:
        m = PAT.match(line)
        out.append(listing(root, m.group(1), int(m.group(2)), int(m.group(3))) if m else line)
    with open(os.path.join(HERE, "..", "fifo_sizing_analytical.tex"), "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/lstasytis/finn/build/wt/pr-fifo")
