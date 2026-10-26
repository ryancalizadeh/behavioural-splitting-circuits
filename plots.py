"""Figures of the summary and profiling tables written by reproduce.py."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crossbar import CELLS

TITLES = {"1R": "1R", "1D1R-ideal": "1D1R, ideal diode", "1D1R-Shockley": "1D1R, Shockley diode"}
STYLE = {"SPICE": dict(color="#eb6834", marker="o"), "Algorithm 1": dict(color="#2a78d6", marker="s")}
# stacked bottom to top; adjacent pairs checked with the dataviz palette validator (light surface)
STEPS = {"resolvent": ("resolvents", "#2a78d6"), "solve": ("triangular solve", "#eb6834"),
         "matvec": ("mat-vec", "#1baf7a"), "check": ("convergence test", "#eda100"),
         "update": ("relaxation", "#e87ba4"), "setup": ("setup (once)", "#6250d6")}
INK, GRID = "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def runtime_memory(summary, path):
    """Median runtime (top) and peak memory (bottom) against N, min-max over the runs as error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey="row", constrained_layout=True)
    for col, cell in enumerate(CELLS):
        for solver, style in STYLE.items():
            rows = [r for r in summary if r["cell"] == cell and r["solver"] == solver and not math.isnan(r["runtime_s"])]
            n = [r["N"] for r in rows]
            for ax, key in zip(axes[:, col], ("runtime_s", "memory_gb")):
                y = [r[key] for r in rows]
                err = [[a - r[key.replace("_", "_min_")] for a, r in zip(y, rows)],
                       [r[key.replace("_", "_max_")] - a for a, r in zip(y, rows)]]
                ax.errorbar(n, y, err, label=solver, lw=1.5, ms=5, capsize=2, mec="white", **style)
        axes[0, col].set_title(TITLES[cell])
        axes[1, col].set(xscale="log", yscale="log", xticks=[16, 64, 256, 1024], xticklabels=[16, 64, 256, 1024])
        axes[0, col].set_yscale("log")
    axes[0, 0].set_ylabel("runtime (s)")
    axes[1, 0].set_ylabel("peak memory (GB)")
    fig.supxlabel("N (N x N array)", color=INK, fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def profiling(profiling, path):
    """Algorithm 1 per cell: each step's share of the runtime (setup plus iterations), the iterations, and the time per
    iteration, against N."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey="row", constrained_layout=True)
    for col, cell in enumerate(CELLS):
        rows = [r for r in profiling if r["cell"] == cell]
        x = range(len(rows))
        bottom = [0.0] * len(rows)
        for step, (label, color) in STEPS.items():
            share = [100 - r["loop_pct"] if step == "setup" else r[f"{step}_pct"] * r["loop_pct"] / 100 for r in rows]
            axes[0, col].bar(x, share, 0.7, bottom, color=color, edgecolor="white", lw=1, label=label)
            bottom = [b + s for b, s in zip(bottom, share)]
        axes[1, col].plot(x, [r["iterations"] for r in rows], color=INK, marker="o", ms=4)
        axes[2, col].plot(x, [r["ms_per_iteration"] for r in rows], color=INK, marker="o", ms=4)
        axes[0, col].set_title(TITLES[cell])
        axes[2, col].set_xticks(x, [r["N"] for r in rows])
    for ax, label in zip(axes[:, 0], ("share of runtime (%)", "iterations", "time per iteration (ms)")):
        ax.set_ylabel(label)
    axes[0, 0].set_ylim(0, 100)
    axes[1, 0].set(yscale="log", ylim=(10, None))
    axes[2, 0].set_yscale("log")
    for ax in axes[0]:
        ax.grid(axis="x", visible=False)
    fig.supxlabel("N (N x N array)", color=INK, fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=6, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
