"""Renders obstacles.{mp4,gif,jpg}: the reversed Van der Pol oscillator with two obstacles (benchmarks system vdp_obs2), with
the union estimate of the four fine-tuned seeds under the relaxed scoring. Run: python obstacles.py"""
from __future__ import annotations

import json

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy import ndimage

import common as C
from sim import pick_spread

NAME = "vdp_obs2"
NX, NY = 251, 351
T_PHYS = 12.0


def systems():
    with C.research(C.BENCH):
        import systems as S
        return S


@torch.no_grad()
def simulate(sysm, x0, T, dt=0.01, record=False):
    """RK4 of the physical field; reason 0 converged/running, 1 obstacle, 2 box; label 0 converged as in systems.simulate."""
    x = torch.as_tensor(x0, dtype=torch.float32).clone()
    N = len(x)
    reason = np.zeros(N, dtype=int)
    held = torch.zeros(N)
    conv = torch.zeros(N, dtype=torch.bool)
    alive = torch.ones(N, dtype=torch.bool)
    k_stop = np.full(N, -1)
    hist = [x.numpy().copy()] if record else None
    nc = len(sysm.cons)
    f = sysm.f
    for k in range(int(round(T / dt)) + 1):
        d = sysm.distances(x)
        obs = (d[:, :nc] <= 0).any(-1) & alive
        box = (d[:, nc:] <= 0).any(-1) & alive & ~obs
        for m, r in ((obs, 1), (box, 2)):
            if m.any():
                reason[m.numpy()] = r
                k_stop[m.numpy()] = k
                alive &= ~m
        near = (x * sysm.s).norm(dim=-1) < sysm.tol
        held = torch.where(near, held + dt, torch.zeros_like(held))
        conv |= held >= 1.0 - 1e-9
        if not record and not (alive & ~conv).any():
            break
        a = alive & (~conv if not record else torch.ones_like(conv))
        xa = x[a]
        k1 = f(xa); k2 = f(xa + 0.5 * dt * k1); k3 = f(xa + 0.5 * dt * k2); k4 = f(xa + dt * k3)
        x[a] = xa + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if record:
            hist.append(x.numpy().copy())
    label = np.where(conv.numpy(), 0, np.where(reason > 0, 1, 3))
    return label, reason, k_stop, (np.array(hist) if record else None)


def grid_data(S):
    f = C.CACHE / f"grid_{NAME}.npz"
    if f.exists():
        return dict(np.load(f))
    sysm = S.get(NAME)
    lo, hi = sysm.lo.numpy(), sysm.hi.numpy()
    xs, ys = np.linspace(lo[0], hi[0], NX), np.linspace(lo[1], hi[1], NY)
    X, Y = np.meshgrid(xs, ys)
    x = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
    label, reason, _, _ = simulate(sysm, x, sysm.sim_T)
    with C.research(C.BENCH):
        import core
        c0 = json.loads((C.BENCH / "results" / f"core_{NAME}.json").read_text())["c0"]
        est = torch.zeros(len(x), dtype=torch.bool)
        for k in range(4):
            run = C.BENCH / "runs" / f"{NAME}_s{k}_rep"
            snap = json.loads((run / "certify.json").read_text())["snapshot"]
            c2 = json.loads((run / "certify_core_relaxed.json").read_text())["c2"]
            prob, model, _ = core.load(run, run / "snapshots" / snap)
            prob.inner_c = float(c0)
            with torch.no_grad():
                est |= torch.cat([(model(c) < c2) | (prob.inner(c) < 1) for c in x.split(50000)])
                if k == 0:
                    Eq = torch.cat([prob.inner(c) for c in x.split(50000)])
    d = dict(xs=xs, ys=ys, R=(label == 0).reshape(NY, NX), reason=reason.reshape(NY, NX),
             G=est.numpy().reshape(NY, NX), E=Eq.numpy().reshape(NY, NX))
    np.savez_compressed(f, **d)
    return d


def main():
    C.style()
    S = systems()
    sysm = S.get(NAME)
    d = grid_data(S)
    xs, ys, R, G, E, reason = d["xs"], d["ys"], d["R"], d["G"], d["E"], d["reason"]
    print(f"  grid: simulated region {100 * R.mean():.1f} % of the box, union estimate covers {100 * (G & R).sum() / R.sum():.1f} %"
          f" of it, estimate outside the simulated region: {int((G & ~R).sum())} grid states")

    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], -1)
    er = lambda m, k: ndimage.binary_erosion(m, iterations=k, border_value=1)
    inb = (np.abs(X) < 2.3) & (np.abs(Y) < 3.3)
    masks = dict(A=er(G & R, 3) & inb, B=er(R & ~G, 2) & inb, C=er(~R & (reason == 1), 2) & inb, D=er(~R & (reason == 2), 3) & inb)
    counts = dict(C=6, B=9, A=20, D=7)
    rmins = dict(C=0.3, B=0.3, A=0.55, D=0.75)
    rng = np.random.default_rng(5)
    chosen, cls = [], []
    for k in ("C", "B", "A", "D"):
        pts = pick_spread(P[masks[k]], 6 * counts[k], rmins[k], rng, avoid=chosen if chosen else None)
        lab, rea, ks, _ = simulate(sysm, pts, T_PHYS - 1.5, record=False)
        want = {"A": lambda: (lab == 0) | (rea == 0), "B": lambda: (lab == 0) | (rea == 0),
                "C": lambda: (rea == 1) & (ks > 50), "D": lambda: (rea == 2) & (ks > 30)}[k]()
        pts = pts[want][:counts[k]]
        chosen += list(pts); cls += [k] * len(pts)
        print(f"  class {k}: {len(pts)} starts")
    x0 = np.array(chosen); cls = np.array(cls)
    N = len(x0)
    dt = 0.01
    _, rea, kst, H = simulate(sysm, x0, T_PHYS, dt=dt, record=True)
    n_in, n_run, n_out = int(0.5 * C.FPS), int(7.6 * C.FPS), int(1.0 * C.FPS)
    F = n_in + n_run + n_out
    tq = np.linspace(0, T_PHYS, n_run)
    kq = np.minimum(np.round(tq / dt).astype(int), len(H) - 1)
    Sx = np.stack([H[np.minimum(kq, np.where(kst[i] >= 0, kst[i], 10 ** 9)), i] for i in range(N)], 1)
    k_end = np.where(kst >= 0, np.searchsorted(tq, kst * dt), n_run + 10)
    print(f"  {N} trajectories: obstacle {int((rea == 1).sum())}, box {int((rea == 2).sum())}")

    lo, hi = sysm.lo.numpy(), sysm.hi.numpy()
    ext = (lo[0], hi[0], lo[1], hi[1])
    W, Hh, dpi = 9.6, 7.8, 150
    fig = C.plt.figure(figsize=(W, Hh), dpi=dpi)
    ah = 6.9
    aw = ah * (ext[1] - ext[0]) / (ext[3] - ext[2])
    ax = fig.add_axes([0.45 / W, 0.45 / Hh, aw / W, ah / Hh])
    ax.set_facecolor(C.C_OUT)
    ax.contourf(*C.smooth(xs, ys, R), levels=[0.5, 2], colors=[C.C_ROA])
    ax.contourf(*C.smooth(xs, ys, G & R), levels=[0.5, 2], colors=[C.C_EST])
    ax.contour(xs, ys, E, levels=[1.0], colors=[C.C_INN], linewidths=3.0, zorder=5)
    for c in [(1.0, 1.0), (-1.0, -1.0)]:
        ax.add_patch(Circle(c, 0.25, fc=C.C_OBS, ec="none", zorder=7))
    ax.plot([0], [0], marker="o", ms=9, mfc="white", mec=C.INK, mew=1.8, zorder=9)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    C.clean_axes(ax)
    xl = (0.45 + aw + 0.4) / W
    fig.text(xl, 0.93, "Van der Pol oscillator,\ntwo obstacles", ha="left", va="top", fontsize=22, fontweight="semibold",
             linespacing=1.15)
    handles = [Patch(facecolor=C.C_ROA, label="Simulated region\nof attraction"), Patch(facecolor=C.C_EST, label="Estimate"),
               Line2D([], [], color=C.C_INN, lw=3, label="Inner layer $E$"),
               Line2D([], [], ls="none", marker="o", ms=17, mfc=C.C_OBS, mec="none", label="Obstacle"),
               Line2D([], [], ls="none", marker="o", ms=9, mfc="white", mec=C.INK, mew=1.8, label="Equilibrium"),
               Line2D([], [], ls="none", marker="X", ms=13, mfc=C.C_BAD, mec="white", label="Hits an obstacle\nor leaves the box")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(xl - 0.01, 0.76), ncol=1, frameon=False, fontsize=17,
               handlelength=1.5, handletextpad=0.6, labelspacing=0.9)

    ink = np.array(C.matplotlib.colors.to_rgb(C.INK))
    trails = LineCollection([], linewidths=2.0, capstyle="round", zorder=6)
    ax.add_collection(trails)
    dots = ax.scatter([], [], s=70, c=C.INK, edgecolors="white", linewidths=1.2, zorder=8)
    marks = ax.scatter([], [], marker="X", s=190, c=C.C_BAD, edgecolors="white", linewidths=1.3, zorder=10, clip_on=False)
    bursts = C.Bursts(ax, N, rmax=0.28)
    L = int(1.6 / (tq[1] - tq[0]))

    vid = C.Video("obstacles", fig, gif_width=760)
    poster = None
    for f in range(F):
        if f < n_in:
            k, fade = 0, C.smoothstep(f / (n_in - 1))
        elif f < n_in + n_run:
            k, fade = f - n_in, 1.0
        else:
            k, fade = n_run - 1, 1.0 - C.smoothstep((f - n_in - n_run + 1) / (n_out * 0.8))
        seg, w, who = C.trail_segments(Sx, k, L)
        rgba = np.zeros((len(seg), 4)); rgba[:, :3] = ink; rgba[:, 3] = 0.75 * w ** 1.4 * fade
        trails.set_segments(seg); trails.set_colors(rgba)
        alive = k < k_end
        dots.set_offsets(Sx[k, alive]); dots.set_alpha(fade)
        dead = ~alive
        marks.set_offsets(Sx[k, dead] if dead.any() else np.zeros((0, 2))); marks.set_alpha(fade)
        for i in np.nonzero(kst >= 0)[0]:
            bursts.update(i, Sx[k, i], (k - k_end[i]) / C.FPS if f < n_in + n_run else -1)
        buf = vid.grab()
        if f == n_in + int(0.2 * n_run):
            poster = buf.copy()
    vid.poster(poster)
    vid.close()


if __name__ == "__main__":
    main()
