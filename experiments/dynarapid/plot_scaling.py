"""Plot the core-scaling experiment: wall time vs. cores, Vivado flow vs. DynaRapid cold.

    python plot_scaling.py <scaling dir with scaling.json> [out.png]

scaling.json is written by summarize_scaling.py. One panel per model (one y axis each).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# categorical slots 1 and 2 of the reference palette; identity also by marker, line style
# and a direct label
SERIES = {
    "dynarapid": dict(color="#2a78d6", marker="o", ls="-", label="DynaRapid (cold)"),
    "vivado": dict(color="#eb6834", marker="s", ls="--", label="Vivado flow"),
}


def main():
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "scaling.png")
    rows = json.load(open(os.path.join(d, "scaling.json")))
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.0), facecolor=SURFACE)
    axes = axes if len(models) > 1 else [axes]
    for ax, model in zip(axes, models):
        ax.set_facecolor(SURFACE)
        for mode, st in SERIES.items():
            pts = sorted(
                (r["cores"], r["total_s"])
                for r in rows
                if r["model"] == model and r["mode"] == mode and r.get("total_s")
            )
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=st["color"], ls=st["ls"], lw=2, marker=st["marker"], ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=st["label"], zorder=3)
            # direct label at the second point, value labels at the ends only
            lx, ly = pts[min(1, len(pts) - 1)]
            ax.annotate(st["label"], (lx, ly), xytext=(8, 8), textcoords="offset points",
                        color=INK2, fontsize=9)
            ax.annotate("%d s" % pts[0][1], pts[0], xytext=(10, -4), textcoords="offset points",
                        ha="left", va="top", color=INK2, fontsize=8)
            up = mode == "vivado"
            ax.annotate("%d s" % pts[-1][1], pts[-1], xytext=(0, 10 if up else -12),
                        textcoords="offset points", ha="center", va="bottom" if up else "top",
                        color=INK2, fontsize=8)
        ax.set_xscale("log", base=2)
        xs = sorted({r["cores"] for r in rows if r["model"] == model})
        ax.set_xticks(xs)
        ax.set_xticklabels([("%d\n(SMT)" % c) if c == 128 else str(c) for c in xs])
        top = max(r["total_s"] for r in rows if r["model"] == model and r.get("total_s"))
        ax.set_ylim(0, 1.1 * top)
        ax.set_title(model.upper(), color=INK, fontsize=11, loc="left")
        ax.set_xlabel("cores (taskset)", color=INK2, fontsize=9)
        ax.set_ylabel("wall time [s]", color=INK2, fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    fig.suptitle("ZCU104 bitfile build, cold DynaRapid library vs. Vivado flow",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
