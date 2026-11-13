"""Renders level_sweep.{mp4,gif,jpg}: the level c of seed 0's candidate v swept on the circle slice, past the selected
level c2 to the first counterexamples and back. Run: python level_sweep.py"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

import common as C
from tfl_data import slice_data

C_OVER = 0.998


def main():
    C.style()
    d = slice_data("circ3")
    xs, ys, v, R, I = d["xs"], d["ys"], d["V"][0], d["R"].astype(bool), d["I"]
    c2 = float(d["c2"][0])
    first_cex = float(v[~R].min())
    print(f"  c2 = {c2:.5f}; smallest v outside the simulated region on the slice = {first_cex:.5f}")
    up = 3
    xu = np.linspace(xs[0], xs[-1], up * len(xs)); yu = np.linspace(ys[0], ys[-1], up * len(ys))
    Vz = -np.log1p(-np.clip(ndimage.zoom(v.astype(float), up, order=1), 0, 1 - 1e-9))
    _, _, Ns = C.smooth(xs, ys, ~R, up=up)
    Vg = -np.log1p(-np.clip(v.astype(float), 0, 1 - 1e-9))

    ext = d["ext"]
    fig, ax, y_top, y_bot = C.layout(7.2, ext)
    ax.set_facecolor(C.C_ROA)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    C.clean_axes(ax)
    ax.contourf(xu, yu, Ns, levels=[0.5, 2], colors=[C.C_OUT], zorder=3)
    if (I < 1).any():
        ax.contour(xs, ys, I, levels=[1.0], colors=[C.C_INN], linewidths=3.0, zorder=6)
    a = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(a), np.sin(a), color=C.C_PATH, lw=1.7, ls=C.PATH_DASH, alpha=0.85, zorder=5)
    fig.text(0.5, y_top + 0.02, "Selecting the outer level", ha="center", va="bottom", fontsize=22, fontweight="semibold")
    handles = [Patch(facecolor=C.C_ROA, label="Simulated region of attraction"), Patch(facecolor=C.C_EST, label="$\\{v < c\\}$"),
               Line2D([], [], color=C.C_INN, lw=3, label="Inner layer"),
               Line2D([], [], color=C.C_PATH, lw=1.7, ls=C.PATH_DASH, label="Path"),
               Patch(facecolor=C.C_BAD, label="Counterexample")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y_bot - 0.015), ncol=3, frameon=False,
               fontsize=15, handlelength=1.4, columnspacing=0.9, handletextpad=0.5)
    box = dict(boxstyle="round,pad=0.35", fc="white", ec="#d6d4ce", lw=0.8, alpha=0.93)
    hud = ax.text(-1.93, 1.9, "", ha="left", va="top", fontsize=19, zorder=12, bbox=box, linespacing=1.35,
                  family="serif")
    tag = ax.text(-1.93, 1.18, "selected level $c_2$", ha="left", va="top", fontsize=18, zorder=12, color="white",
                  bbox=dict(boxstyle="round,pad=0.3", fc=C.INK, ec="none"))
    cex = ax.annotate("counterexample", xy=(0.03, -0.36), xytext=(-1.15, -1.3), fontsize=18, color=C.C_BAD, zorder=12,
                      ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C.C_BAD, lw=1.2),
                      arrowprops=dict(arrowstyle="-|>", color=C.C_BAD, lw=1.6, shrinkB=4))

    V2, Vo = -np.log1p(-c2), -np.log1p(-C_OVER)
    ease_out = lambda u: 1 - (1 - np.clip(u, 0, 1)) ** 3
    sec = lambda s: int(round(s * C.FPS))
    plan = [("hold", 0.3, 0.0, 0.0), ("ramp_out", 4.2, 0.0, V2), ("hold", 0.7, V2, V2), ("ramp", 1.1, V2, Vo),
            ("hold", 0.8, Vo, Vo), ("ramp", 0.9, Vo, V2), ("hold", 1.4, V2, V2), ("ramp", 0.6, V2, 0.0)]
    frames = []
    for kind, dur, a0, a1 in plan:
        n = sec(dur)
        for j in range(n):
            u = j / max(n - 1, 1)
            if kind == "ramp_out":
                Vc, al = a0 + (a1 - a0) * ease_out(u), 1.0
            elif kind == "ramp":
                Vc, al = a0 + (a1 - a0) * C.smoothstep(u), 1.0
            else:
                Vc, al = a0, 1.0
            frames.append((Vc, al, kind))

    vid = C.Video("level_sweep", fig)
    arts, poster, Rn = [], None, R.sum()
    for i, (Vc, al, kind) in enumerate(frames):
        for A in arts:
            A.remove()
        arts = []
        c = -np.expm1(-Vc)
        if Vc > 0:
            arts.append(ax.contourf(xu, yu, Vz, levels=[-1, Vc], colors=[C.C_EST], alpha=al, zorder=2))
            red = np.minimum(Vc - Vz, 4 * (Ns - 0.5))
            if (red > 0).any():
                arts.append(ax.contourf(xu, yu, red, levels=[0, 1e9], colors=[C.C_BAD], alpha=al, zorder=4))
        cov = 100 * ((Vg < Vc) & R).sum() / Rn
        n_cex = int(((Vg < Vc) & ~R).sum())
        hud.set_text(f"$c$ = {c:.4f}\ncoverage {cov:5.1f} %")
        at_c2 = abs(Vc - V2) < 1e-9 and kind == "hold" and i > sec(1)
        tag.set_visible(at_c2); tag.set_alpha(al)
        cex.set_visible(n_cex > 0)
        buf = vid.grab()
        if at_c2 and poster is None and i > sec(7.5):
            poster = buf.copy()
    vid.poster(poster)
    vid.close()


if __name__ == "__main__":
    main()
