"""Unicycles launched on the Figure 1 slice (heading north, speed 1) under the true TFL closed loop."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

import common as C
from sim import integrate, pick_spread, resample
from tfl_data import load_case, slice_data

DART = np.array([[1.0, 0.0], [-0.65, 0.55], [-0.3, 0.0], [-0.65, -0.55]])


def darts(P, th, size):
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x = DART[None, :, 0] * c - DART[None, :, 1] * s
    y = DART[None, :, 0] * s + DART[None, :, 1] * c
    return np.stack([P[:, 0:1] + size * x, P[:, 1:2] + size * y], -1)


def select(name, counts, rmins, seed, T_fail):
    f = C.CACHE / f"starts_{name}.npz"
    if f.exists():
        return dict(np.load(f))
    d = slice_data(name)
    R, G = d["R"].astype(bool), d["G"]
    X, Y = np.meshgrid(d["xs"], d["ys"])
    P = np.stack([X, Y], -1)
    er = lambda m, k: ndimage.binary_erosion(m, iterations=k, border_value=1)
    r = np.hypot(X, Y)
    masks = dict(A=er(G & R, 4), B=er(R & ~G, 3), C=er(~R, 2) & (r > 0.08) & (r > 0.6),
                 D=er(~R, 2) & (r > 0.08) & (r < 0.6))
    prob, _, _ = load_case(name)
    wrap = name == "sin6"
    rng = np.random.default_rng(seed)
    chosen, cls = [], []
    for k in ("D", "C", "B", "A"):
        if counts.get(k, 0) == 0:
            continue
        cand = P[masks[k]]
        pts = pick_spread(cand, 4 * counts[k], rmins[k], rng, avoid=chosen if chosen else None)
        x0 = np.zeros((len(pts), 4)); x0[:, :2] = pts; x0[:, 2] = np.pi / 2
        ts, xs, st = integrate(prob.sys, x0, T_fail + 1, dtau=0.01, wrap_x1=wrap)
        _, tend = resample(ts, xs, st, np.array([0.0]), periodic=(2,) + ((0,) if wrap else ()))
        ok = (st == 0) if k in "AB" else ((st != 0) & (tend > 0.3) & (tend < T_fail))
        pts = pts[ok][:counts[k]]
        chosen += list(pts); cls += [k] * len(pts)
        print(f"  class {k}: {len(pts)} starts")
    out = dict(P=np.array(chosen), cls=np.array(cls))
    np.savez(f, **out)
    return out


def render(name, out, title, counts, rmins, seed=0, T_phys=14.0, width=7.2, dart=0.09, legend_cols=3):
    C.style()
    wrap = name == "sin6"
    d = slice_data(name)
    sel = select(name, counts, rmins, seed, T_fail=T_phys - 2)
    prob, _, _ = load_case(name)
    N = len(sel["P"])
    x0 = np.zeros((N, 4)); x0[:, :2] = sel["P"]; x0[:, 2] = np.pi / 2

    n_in, n_run, n_out = int(0.6 * C.FPS), int(7.6 * C.FPS), int(1.0 * C.FPS)
    F = n_in + n_run + n_out
    tq = np.linspace(0, T_phys, n_run)
    ts, xs, st = integrate(prob.sys, x0, T_phys, dtau=0.005, wrap_x1=wrap)
    S, t_end = resample(ts, xs, st, tq, periodic=(2,) + ((0,) if wrap else ()))
    print(f"  {N} vehicles, failures {int((st != 0).sum())}")

    ext = d["ext"]
    fig, ax, y_top, y_bot = C.layout(width, ext)
    R, G, I = d["R"].astype(float), (d["G"] & d["R"].astype(bool)).astype(float), d["I"]
    ax.set_facecolor(C.C_OUT)
    ax.contourf(*C.smooth(d["xs"], d["ys"], R), levels=[0.5, 2], colors=[C.C_ROA], antialiased=True)
    ax.contourf(*C.smooth(d["xs"], d["ys"], G), levels=[0.5, 2], colors=[C.C_EST], antialiased=True)
    if (I < 1).any():
        ax.contour(d["xs"], d["ys"], I, levels=[1.0], colors=[C.C_INN], linewidths=3.0, zorder=5)
    if wrap:
        u = np.linspace(ext[0], ext[1], 400); px, py = u, np.sin(u)
    else:
        a = np.linspace(0, 2 * np.pi, 400); px, py = np.cos(a), np.sin(a)
    ax.plot(px, py, color=C.C_PATH, lw=1.7, ls=C.PATH_DASH, alpha=0.85, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    C.clean_axes(ax)
    fig.text(0.5, y_top + 0.02, title, ha="center", va="bottom", fontsize=22, fontweight="semibold")
    handles = [Patch(facecolor=C.C_ROA, label="Simulated region of attraction"), Patch(facecolor=C.C_EST, label="Estimate"),
               Line2D([], [], color=C.C_INN, lw=3, label="Inner layer"),
               Line2D([], [], color=C.C_PATH, lw=1.7, ls=C.PATH_DASH, label="Path"),
               Line2D([], [], ls="none", marker="X", ms=12, mfc=C.C_BAD, mec="white", label="Failure")]
    if not (I < 1).any():
        handles.pop(2)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y_bot - 0.015), ncol=legend_cols, frameon=False,
               fontsize=16, handlelength=1.5, columnspacing=1.2, handletextpad=0.5)

    trails = LineCollection([], linewidths=2.0, capstyle="round", zorder=6)
    ax.add_collection(trails)
    bodies = PolyCollection([], facecolors=C.INK, edgecolors="white", linewidths=1.1, zorder=8)
    ax.add_collection(bodies)
    marks = ax.scatter([], [], marker="X", s=190, c=C.C_BAD, edgecolors="white", linewidths=1.3, zorder=10, clip_on=False)
    bursts = C.Bursts(ax, N, rmax=0.22)
    ink = np.array(C.matplotlib.colors.to_rgb(C.INK))
    L = int(1.3 / (tq[1] - tq[0]))
    k_end = np.where(np.isfinite(t_end), np.searchsorted(tq, t_end), n_run + 10)

    vid = C.Video(out, fig)
    poster = None
    for f in range(F):
        if f < n_in:
            k, fade = 0, C.smoothstep(f / (n_in - 1))
        elif f < n_in + n_run:
            k, fade = f - n_in, 1.0
        else:
            k, fade = n_run - 1, 1.0 - C.smoothstep((f - n_in - n_run + 1) / (n_out * 0.8))
        Hk = S[:, :, :2]
        seg, w, who = C.trail_segments(Hk, k, L, wrap=np.pi if wrap else None)
        rgba = np.zeros((len(seg), 4)); rgba[:, :3] = ink; rgba[:, 3] = 0.75 * w ** 1.4 * fade
        trails.set_segments(seg); trails.set_colors(rgba)
        alive = k < k_end
        P = S[k, alive, :2]; th = S[k, alive, 2]
        bodies.set_verts(darts(P, th, dart))
        bodies.set_alpha(fade)
        dead = ~alive
        marks.set_offsets(S[np.minimum(k_end[dead], n_run - 1), np.nonzero(dead)[0], :2] if dead.any() else np.zeros((0, 2)))
        marks.set_alpha(fade)
        for i in np.nonzero(np.isfinite(t_end))[0]:
            bursts.update(i, S[min(k_end[i], n_run - 1), i, :2], (k - k_end[i]) / C.FPS if f < n_in + n_run else -1)
        buf = vid.grab()
        if f == n_in + int(0.55 * n_run):
            poster = buf.copy()
    vid.poster(poster)
    vid.close()

