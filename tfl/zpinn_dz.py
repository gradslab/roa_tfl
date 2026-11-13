r"""
Dirichlet-Zubov PINN baseline of Meng & Liu (ECC 2026, arXiv 2511.09523) on the TFL closed loops, with the level selection and final test of certify3.py like the other baselines.

  gamma(x) = min_i d_i(x) / median d_i, > 0 on Omega_ad (their gamma = 1 - h with h = max_i h_i); lambda = 0.1 (their value);
  rescaled field f~ = lambda gamma f (their Theorem 3, with theta = 1 / lambda), residual
      DW.f~ + a c (1 - W^2) = 0,   W = tanh(a V~),   c = zeta^T Q zeta (their |x|^2, with the distance to the target set in place of |x|),
  W = 0 on the target set, W = 1 on the layer of admissible states closest to the boundary of Omega_ad (the Dirichlet condition; f is not defined beyond it),
  data loss on 3000 simulated states, W^ = tanh(a V~) if the state converges and 1 otherwise, with V~ = int c / (lambda gamma) dt along the physical trajectory.
  Two hidden layers of 30 tanh units, 1e5 collocation points, 2e4 Adam steps (batch 2048, cosine lr 1e-3 to 1e-5), as for the Zubov PINN baseline.  The residual is divided by
  sqrt(1 + |S f~|^2 + (a c)^2), a positive weight, because f and c are large at some states.  a = alpha / factor; the paper tunes this scale per example.

Run:  python -u zpinn_dz.py <case> <seed> <out> <factor>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from certify3 import band_eps
from problem import ConstrainedZubov
from zpinn_liu import PinnModel, case_cfg, write_common

LAM = 0.1
STEPS = int(os.environ.get("DZ_STEPS", 20000))


def gamma(prob, x):
    return (prob.distances(x) / prob.scale).min(-1).values


@torch.no_grad()
def labels(prob, x, a, T, dt=0.02):
    s = prob.sys
    conv = s.simulate(x) == 0
    y = x.clone()
    I = torch.zeros(len(x), dtype=x.dtype)
    F = lambda z: torch.nan_to_num(s.evaluate(z)["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(int(round(T / dt))):
        k1 = F(y); k2 = F(y + 0.5 * dt * k1); k3 = F(y + 0.5 * dt * k2); k4 = F(y + dt * k3)
        I = I + dt * prob.g(y, 1.0) / (LAM * gamma(prob, y).clamp_min(1e-6))
        y = torch.nan_to_num(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), nan=1e3).clamp(-1e3, 1e3)
    return torch.where(conv, torch.tanh(a * torch.nan_to_num(I, nan=1e6, posinf=1e6)), torch.ones(len(x), dtype=x.dtype))


def train(case, seed, out, factor, n_c=100000, n_d=3000, n_b=2000, batch=2048):
    t0 = time.time()
    cfg = case_cfg(case, seed, out)
    out = write_common(cfg, out, "pinn.flag")
    (out / "dz.json").write_text(json.dumps(dict(method="dirichlet_zubov", lam=LAM, factor=factor)))
    torch.manual_seed(seed)
    prob = ConstrainedZubov(cfg)
    s = prob.sys
    rich = os.environ.get("PINN_RICH") == "1"
    if rich:
        (out / "rich.flag").write_text("rich\n")
    m = PinnModel(prob, rich=rich)
    a = float(prob.alpha) / factor
    xc = s.sample_box(n_c, seed=seed + 1)
    xc = xc[prob.admissible(xc)]
    with torch.no_grad():
        gc = gamma(prob, xc)
        f = LAM * gc[:, None] * torch.nan_to_num(s.evaluate(xc)["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
        om = prob.g(xc, 1.0)
        nf = torch.sqrt(1.0 + ((f * s.s) ** 2).sum(-1) + (a * om) ** 2)
    xd = s.sample_box(n_d, seed=seed + 2)
    xd = xd[prob.admissible(xd)]
    T = 40.0 if case == "quadrotor" else 30.0
    Wd = labels(prob, xd, a, T)
    with torch.enable_grad():
        xt = s.target_points(2000, seed=seed + 3)
        band = s.sample_band(20000, "singular", 3.0, 0.1, seed=seed + 5, eps=band_eps(s))
    pool = torch.cat([xc, s.sample_edge(20000, seed=seed + 4), band.detach()])
    pool = pool[prob.admissible(pool) & s.in_box(pool)]
    with torch.no_grad():
        xb = pool[torch.argsort(gamma(prob, pool))[:n_b]]
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
    g = torch.Generator().manual_seed(seed)
    for it in range(STEPS):
        idx = torch.randint(len(xc), (batch,), generator=g)
        x = xc[idx].detach().requires_grad_(True)
        W = m.W(x)
        (dW,) = torch.autograd.grad(W.sum(), x, create_graph=True)
        res = ((dW * f[idx]).sum(-1) + a * om[idx] * (1 - W ** 2)) / nf[idx]
        loss = (res ** 2).mean() + (m.W(xt) ** 2).mean() + ((m.W(xb) - 1) ** 2).mean() + ((m.W(xd) - Wd) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % (STEPS // 4) == 0:
            torch.save(m.state_dict(), out / "snapshots" / f"epoch_{25 * (it + 1) // (STEPS // 4):03d}.pt")
            print(f"step {it + 1} loss {loss.item():.3e} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
