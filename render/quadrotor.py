"""Renders quadrotor.{mp4,gif,jpg}: planar quadrotors on the vertical loop, from reference states inside the estimate
(converge and orbit) and outside the simulated region (thrust reaches zero). Run: python quadrotor.py"""
from __future__ import annotations

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import common as C
from sim import integrate, pick_spread, resample
from tfl_data import load_case

T_PHYS = 20.0
N_IN, N_OUT = 6, 5


def select(prob, seeds):
    f = C.CACHE / "starts_quad3.npz"
    if f.exists():
        return dict(np.load(f))
    gt = torch.load(C.TFL / "results" / "gt_quadrotor_50k.pt")
    x, lab = gt["xg"], gt["label"]
    with torch.no_grad():
        E = torch.zeros(len(x), dtype=torch.bool)
        for m, c in seeds:
            E |= torch.cat([m(c_) for c_ in x.split(20000)]) < c
    rng = np.random.default_rng(7)
    s = prob.sys
    r = x[:, :2].norm(dim=-1)
    ia = torch.nonzero(E & (lab == 0) & ((r - 1).abs() > 0.45) & (x[:, :2].abs().max(-1).values < 1.75)).flatten().numpy()
    pa = pick_spread(x[ia, :2].numpy(), 40, 0.7, rng)
    ia = np.array([ia[np.argmin(np.abs(x[ia, :2].numpy() - p).sum(1))] for p in pa])
    ts, xs, st = integrate(s, x[ia].double(), T_PHYS, dtau=0.01, thrust=True)
    ia = ia[st == 0][:N_IN]
    ic = torch.nonzero(lab == 1).flatten().numpy()
    ts, xs, st = integrate(s, x[ic].double(), 10.0, dtau=0.01, thrust=True)
    _, tend = resample(ts, xs, st, np.array([0.0]), periodic=(4,))
    K = len(ts)
    last = np.array([xs[min(K - 1, int(np.searchsorted(ts[:, i], tend[i])) - 1), i] if np.isfinite(tend[i]) else xs[-1, i]
                     for i in range(len(ic))])
    ok = (st == 1) & (last[:, 5] < 0.1) & (tend > 1.5) & (tend < 8.5) & (np.abs(last[:, :2]).max(1) < 1.8) & \
         (np.abs(x[ic, :2].numpy()).max(1) < 1.8)
    cand = ic[ok]
    pc = pick_spread(x[cand, :2].numpy(), N_OUT, 0.6, rng, avoid=list(x[ia, :2].numpy()))
    ic = np.array([cand[np.argmin(np.abs(x[cand, :2].numpy() - p).sum(1))] for p in pc])
    print(f"  {len(ia)} starts inside the estimate, {len(ic)} zero-thrust failures (of {ok.sum()} candidates)")
    out = dict(x0=np.concatenate([x[ia].numpy(), x[ic].numpy()]), cls=np.array([0] * len(ia) + [1] * len(ic)))
    np.savez(f, **out)
    return out


def glyph(P, th, a=0.17):
    rt = np.stack([np.cos(th), np.sin(th)], -1)
    up = np.stack([-np.sin(th), np.cos(th)], -1)
    bars = np.stack([P - a * rt, P + a * rt], 1)
    rot = []
    for sgn in (-1, 1):
        m = P + sgn * a * rt
        rot.append(np.stack([m, m + 0.055 * up], 1))
        h = m + 0.065 * up
        rot.append(np.stack([h - 0.075 * rt, h + 0.075 * rt], 1))
    return bars, np.concatenate(rot)


def main():
    C.style()
    prob, seeds, _ = load_case("quad3")
    sel = select(prob, seeds)
    x0, cls = sel["x0"], sel["cls"]
    N = len(x0)
    n_in, n_run, n_out = int(0.6 * C.FPS), int(8.2 * C.FPS), int(1.0 * C.FPS)
    F = n_in + n_run + n_out
    tq = np.linspace(0, T_PHYS, n_run)
    ts, xs, st = integrate(prob.sys, torch.as_tensor(x0).double(), T_PHYS, dtau=0.004, thrust=True)
    S, t_end = resample(ts, xs, st, tq, periodic=(4,))
    print(f"  {N} quadrotors, failures {int((st != 0).sum())}, all failures from the outside class: "
          f"{bool(np.all((st != 0) == (cls == 1)))}")

    L = prob.sys.L
    fig, ax, y_top, y_bot = C.layout(7.2, (-L, L, -L, L))
    ax.set_facecolor(C.C_OUT)
    a = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(a), np.sin(a), color=C.C_PATH, lw=1.7, ls=C.PATH_DASH, alpha=0.85, zorder=2)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    C.clean_axes(ax)
    ax.annotate("", xy=(1.8, -1.82), xytext=(1.8, -1.45), arrowprops=dict(arrowstyle="-|>", color=C.INK2, lw=1.6))
    ax.text(1.72, -1.63, "$g$", fontsize=19, color=C.INK2, va="center", ha="right")
    fig.text(0.5, y_top + 0.02, "Planar quadrotor, vertical loop", ha="center", va="bottom", fontsize=22, fontweight="semibold")
    handles = [Line2D([], [], color=C.C_EST, lw=4.5, label="Starts in the estimate"),
               Line2D([], [], color=C.INK, lw=4.5, label="Starts outside the simulated region"),
               Line2D([], [], ls="none", marker="X", ms=12, mfc=C.C_BAD, mec="white", label="Thrust reaches zero"),
               Line2D([], [], color=C.C_PATH, lw=1.7, ls=C.PATH_DASH, label="Path")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y_bot - 0.015), ncol=2, frameon=False,
               fontsize=16, handlelength=1.5, columnspacing=1.2, handletextpad=0.5)

    rgb = {0: np.array(C.matplotlib.colors.to_rgb(C.C_EST)), 1: np.array(C.matplotlib.colors.to_rgb(C.INK))}
    red = np.array(C.matplotlib.colors.to_rgb(C.C_BAD))
    trails = LineCollection([], linewidths=2.2, capstyle="round", zorder=4)
    bars = LineCollection([], linewidths=5.0, capstyle="round", zorder=7)
    rotors = LineCollection([], linewidths=3.0, capstyle="round", zorder=7)
    for c in (trails, bars, rotors):
        ax.add_collection(c)
    arrows = ax.quiver(S[0, :, 0], S[0, :, 1], np.zeros(N), np.zeros(N), angles="xy", scale_units="xy", scale=1,
                       width=0.0065, headwidth=3.6, headlength=4.0, headaxislength=3.6, color=C.INK2, zorder=6)
    marks = ax.scatter([], [], marker="X", s=200, c=C.C_BAD, edgecolors="white", linewidths=1.3, zorder=10)
    bursts = C.Bursts(ax, N, rmax=0.3)
    Ltr = int(2.4 / (tq[1] - tq[0]))
    k_end = np.where(np.isfinite(t_end), np.searchsorted(tq, t_end), n_run + 10)
    col = np.array([rgb[c] for c in cls])

    vid = C.Video("quadrotor", fig)
    poster = None
    for f in range(F):
        if f < n_in:
            k, fade = 0, C.smoothstep(f / (n_in - 1))
        elif f < n_in + n_run:
            k, fade = f - n_in, 1.0
        else:
            k, fade = n_run - 1, 1.0 - C.smoothstep((f - n_in - n_run + 1) / (n_out * 0.8))
        seg, w, who = C.trail_segments(S[:, :, :2], k, Ltr)
        rgba = np.zeros((len(seg), 4)); rgba[:, :3] = col[who]; rgba[:, 3] = 0.6 * w ** 1.4 * fade
        trails.set_segments(seg); trails.set_colors(rgba)
        kk = np.minimum(k, np.minimum(k_end, n_run - 1))
        P = S[kk, np.arange(N), :2]; th = S[kk, np.arange(N), 4]; T = S[kk, np.arange(N), 5]
        dead = k >= k_end
        c = np.where(dead[:, None], red, col)
        b, r = glyph(P, th)
        bars.set_segments(b); bars.set_colors(np.c_[c, np.full(N, fade)])
        rotors.set_segments(r); rotors.set_colors(np.c_[np.tile(c, (4, 1)), np.full(4 * N, fade)])
        up = np.stack([-np.sin(th), np.cos(th)], -1) * (0.16 * np.where(dead, 0, T))[:, None]
        arrows.set_offsets(P); arrows.set_UVC(up[:, 0], up[:, 1]); arrows.set_alpha(fade)
        marks.set_offsets(P[dead] if dead.any() else np.zeros((0, 2))); marks.set_alpha(fade)
        for i in np.nonzero(np.isfinite(t_end))[0]:
            bursts.update(i, P[i], (k - k_end[i]) / C.FPS if f < n_in + n_run else -1)
        buf = vid.grab()
        if f == n_in + int(0.33 * n_run):
            poster = buf.copy()
    vid.poster(poster)
    vid.close()


if __name__ == "__main__":
    main()
