"""RK4 of the TFL closed loops in normalized time, with the physical time recorded for playback."""
from __future__ import annotations

import numpy as np
import torch


def _status(sys, x, Tprev=None):
    e = sys.evaluate(x)
    bad = (e["margin"] <= 0) | ~torch.isfinite(x).all(-1) | ~torch.isfinite(e["f_cl"]).all(-1)
    if Tprev is not None:
        bad = bad | (Tprev * x[:, 5] <= 0)
    out = ~sys.in_box(x)
    return bad, out


@torch.no_grad()
def integrate(sys, x0, T_phys, dtau=0.004, max_steps=40000, wrap_x1=False, thrust=False):
    x = torch.as_tensor(x0, dtype=torch.float64).clone()
    N, n = x.shape
    s = sys.s.double()

    def fh(y):
        f = torch.nan_to_num(sys.evaluate(y)["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.sqrt(1.0 + ((f * s) ** 2).sum(-1, keepdim=True))
        return f / g, g[:, 0]

    status = np.zeros(N, dtype=int)
    t = torch.zeros(N, dtype=torch.float64)
    ts, xs = [t.numpy().copy()], [x.numpy().copy()]
    alive = torch.ones(N, dtype=torch.bool)
    Tprev = x[:, 5].clone() if thrust else None
    bad, out = _status(sys, x, Tprev)
    status[bad.numpy()] = 1
    status[(out & ~bad).numpy()] = 2
    alive &= ~(bad | out)
    for _ in range(max_steps):
        if not alive.any() or t[alive].min() >= T_phys:
            break
        a = alive & (t < T_phys)
        y = x[a]
        k1, g1 = fh(y); k2, g2 = fh(y + 0.5 * dtau * k1); k3, g3 = fh(y + 0.5 * dtau * k2); k4, g4 = fh(y + dtau * k3)
        y1 = y + dtau / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        dt = dtau / 6 * (g1 + 2 * g2 + 2 * g3 + g4)
        if wrap_x1:
            y1[:, 0] = torch.remainder(y1[:, 0] + np.pi, 2 * np.pi) - np.pi
        if n == 6:
            y1[:, 4] = torch.remainder(y1[:, 4] + np.pi, 2 * np.pi) - np.pi
        bad, out = _status(sys, y1, Tprev[a] if thrust else None)
        idx = torch.nonzero(a).flatten()
        stop = bad | out
        keep = ~stop
        x[idx[keep]] = y1[keep]
        t[idx] = t[idx] + dt
        if thrust:
            Tprev[idx[keep]] = y1[keep, 5]
        st = idx[stop].numpy()
        status[st] = np.where(bad[stop].numpy(), 1, 2)
        alive[idx[stop]] = False
        ts.append(t.numpy().copy()); xs.append(x.numpy().copy())
    return np.array(ts), np.array(xs), status


def resample(ts, xs, status, tq, periodic=()):
    K, N, n = xs.shape
    out = np.zeros((len(tq), N, n))
    t_end = np.full(N, np.inf)
    for i in range(N):
        t = ts[:, i]
        moved = np.nonzero(np.any(np.diff(xs[:, i], axis=0) != 0, axis=1))[0]
        last = (moved[-1] + 1) if len(moved) else 0
        if status[i] != 0:
            t_end[i] = t[min(last + 1, K - 1)]
        tt, xx = t[:last + 1], xs[:last + 1, i].copy()
        for d in periodic:
            xx[:, d] = np.unwrap(xx[:, d])
        for d in range(n):
            out[:, i, d] = np.interp(tq, tt, xx[:, d]) if len(tt) > 1 else xx[0, d]
        for d in periodic:
            out[:, i, d] = np.remainder(out[:, i, d] + np.pi, 2 * np.pi) - np.pi
    return out, t_end


def pick_spread(cands, k, rmin, rng, avoid=None):
    sel = [] if avoid is None else list(avoid)
    n0 = len(sel)
    for j in rng.permutation(len(cands)):
        p = cands[j]
        if all(np.hypot(*(p - q)) >= rmin for q in sel):
            sel.append(p)
            if len(sel) - n0 >= k:
                break
    return np.array(sel[n0:])
