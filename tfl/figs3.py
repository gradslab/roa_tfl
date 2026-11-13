r"""Figure fig_slices3: one two-dimensional slice per TFL case.

Regions (all on the same slice grid):
  light aqua  = simulated region of attraction (reference simulation of every grid state),
  blue        = union over the four fine-tuned seeds of the outer layer {v < c2} (tested on samples),
  orange line = boundary of the verified inner layer,
  dark dashes = the path (projection of the target set),
  red         = estimate outside the simulated region of attraction (none is expected; the script warns if any appear).
Slices: unicycle at heading north and speed 1; quadrotor at the orbit velocity, level tilt and T = 1.

Run:  python figs3.py     (writes ../paper/fig_slices3.pdf and .png; the slice simulations are cached in results/)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

from certify3 import load, Inner, inner_level

OUT = Path(__file__).resolve().parents[1] / "paper"
GRID = 321

C_OUT = "#f2f1ee"
C_ROA = "#a9e3c9"
C_EST = "#2a78d6"
C_INN = "#eb6834"
C_PATH = "#1b1b1b"
C_BAD = "#e34948"
INK, INK2 = "#1b1b1b", "#5a5955"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": "#b9b7b1", "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlepad": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def slice_states(s, case, n):
    t = torch.linspace(-1, 1, n)
    A, Bm = torch.meshgrid(t, t, indexing="xy")
    if case == "quadrotor":
        x = torch.zeros(n * n, 6)
        x[:, 0], x[:, 1] = s.L * A.flatten(), s.L * Bm.flatten()
        ang = torch.atan2(x[:, 1], x[:, 0])
        x[:, 2], x[:, 3] = -s.eta_ref * torch.sin(ang), s.eta_ref * torch.cos(ang)
        x[:, 4], x[:, 5] = 0.0, 1.0
        return x, (-s.L, s.L, -s.L, s.L), ("$p_1$", "$p_2$")
    x = torch.zeros(n * n, 4)
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = s.Lx * A.flatten(), s.L * Bm.flatten(), math.pi / 2, 0.0
    return x, (-s.Lx, s.Lx, -s.L, s.L), ("$x_1$", "$x_2$")


def path_xy(case, ext):
    if case == "sinusoid":
        u = np.linspace(ext[0], ext[1], 400)
        return u, np.sin(u)
    a = np.linspace(0, 2 * np.pi, 400)
    return np.cos(a), np.sin(a)


def regions(name, case):
    reg = None
    for k in range(4):
        run = Path(f"runs/{name}_s{k}_rep3")
        res = json.loads((run / "certify_v3.json").read_text())
        prob, model, cfg = load(run)
        model.load_state_dict(torch.load(run / "snapshots" / res["snapshot"], map_location="cpu"))
        s = prob.sys
        if reg is None:
            x, ext, labels = slice_states(s, case, GRID)
            cache = Path(f"results/slice_gt_{name}_{GRID}.pt")
            if cache.exists():
                roa = torch.load(cache)
            else:
                roa = torch.cat([s.simulate(c, T=40.0 if case == "quadrotor" else 30.0) == 0 for c in x.split(25000)])
                torch.save(roa, cache)
            inner = Inner(prob, inner_level(cfg, prob))
            with torch.no_grad():
                I = torch.cat([inner(c) for c in x.split(20000)]).reshape(GRID, GRID).numpy()
            reg = torch.zeros(len(x), dtype=torch.bool)
        with torch.no_grad():
            v = torch.cat([model(c) for c in x.split(20000)])
        reg |= v < res["c2"]
    R, G = roa.reshape(GRID, GRID).numpy(), reg.reshape(GRID, GRID).numpy()
    return R, G, I, ext, labels


def main():
    cases = [("circ3", "circle", "Unicycle, circle"), ("sin6", "sinusoid", "Unicycle, sinusoid"), ("quad3", "quadrotor", "Planar quadrotor")]
    data = [regions(n, c) for n, c, _ in cases]
    widths = [d[3][1] - d[3][0] for d in data]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.9), gridspec_kw=dict(width_ratios=widths, wspace=0.1))
    for ax, (name, case, title), (R, G, I, ext, (lx, ly)) in zip(axs, cases, data):
        xs = np.linspace(ext[0], ext[1], GRID)
        ys = np.linspace(ext[2], ext[3], GRID)
        ax.set_facecolor(C_OUT)
        ax.contourf(xs, ys, R.astype(float), levels=[0.5, 1.5], colors=[C_ROA], antialiased=True)
        ax.contourf(xs, ys, (G & R).astype(float), levels=[0.5, 1.5], colors=[C_EST], antialiased=True)
        bad = G & ~R
        if bad.any():
            print(f"WARNING {name}: {int(bad.sum())} grid points of the estimate lie outside the simulated region of attraction")
            ax.contourf(xs, ys, bad.astype(float), levels=[0.5, 1.5], colors=[C_BAD])
        px, py = path_xy(case, ext)
        ax.plot(px, py, color=C_PATH, lw=0.9, ls=(0, (3.5, 2.2)), alpha=0.85, zorder=4)
        if (I < 1).any():
            ax.contour(xs, ys, I, levels=[1.0], colors=[C_INN], linewidths=1.8, zorder=5)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontweight="semibold")
        ax.set_xlabel(lx, labelpad=1)
        if case == "sinusoid":
            ax.set_xticks([-3, 0, 3])
        else:
            ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        if ax is axs[0]:
            ax.set_ylabel(ly, labelpad=7, rotation=0, va="center")
        elif case == "quadrotor":
            ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
            ax.set_ylabel(ly, labelpad=7, rotation=0, va="center")
        else:
            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(pad=1.5)
    handles = [Patch(facecolor=C_ROA, label="Simulated region of attraction"),
               Patch(facecolor=C_EST, label="Estimate"),
               Line2D([], [], color=C_INN, lw=1.8, label="Inner layer"),
               Line2D([], [], color=C_PATH, lw=0.9, ls=(0, (3.5, 2.2)), label="Path")]
    fig.canvas.draw()
    top = max(a.title.get_window_extent().y1 for a in axs) / fig.bbox.height
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, top + 0.01), ncol=4, frameon=False,
               handlelength=1.6, handleheight=0.9, columnspacing=1.4, handletextpad=0.5, fontsize=9)
    fig.savefig(OUT / "fig_slices3.pdf")
    fig.savefig(OUT / "fig_slices3.png", dpi=300)
    print("wrote", OUT / "fig_slices3.pdf")


if __name__ == "__main__":
    main()
