r"""
Dirichlet-Zubov PINN baseline of Meng & Liu (ECC 2026, arXiv 2511.09523) on the equilibrium benchmarks, tested with the same selection and i.i.d. final test as the other baselines.
gamma = min_i d_i / median d_i, f~ = lambda gamma f with lambda = 0.1, residual DW.f~ + a |x~|^2 (1 - W^2) = 0, W(0) = 0, W = 1 on the admissible states closest to the boundary of the
admissible set, data W^ = tanh(a V~) with V~ = int |x~|^2 / (lambda gamma) dt on 3000 simulated states (1 if not converged).  Two hidden layers of 30 tanh units.  a = alpha / factor, the factor
chosen on seed 0 by coverage over a grid of factors (1, 3, 10, 30, 100, and 300, 1000 where the best was at the edge).   Run:  python -u baseline_dz.py vdp vdp3 pendulum con3d power
Writes results/baseline_dz_<name>.json.  (certify_iid.py is not imported because it runs at import.)"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

import baselines as B
import core
import systems as S
from certify import EXTRA, region
from gt import ground_truth
from iid import iid_test, seed_of

R = Path(__file__).parent / "results"
LAM, STEPS = 0.1, int(os.environ.get("DZ_STEPS", 20000))


class DzClip(torch.nn.Module):
    def __init__(self, m, s):
        super().__init__()
        self.m, self.sys = m, s

    def forward(self, x):
        return torch.where(self.sys.admissible(x), self.m(torch.nan_to_num(x)).clamp(0.0, 1.0), torch.ones(len(x), device=x.device, dtype=x.dtype))


def scale_of(s):
    x = s.sample_box(50000, seed=123)
    d = s.distances(x)
    return torch.quantile(d[(d > 0).all(-1)], 0.5, dim=0)


def gam(s, sc, x):
    return (s.distances(x) / sc).min(-1).values


@torch.no_grad()
def dz_data(s, sc, x, a, dt=0.01):
    lab = s.simulate(x)
    y = x.clone()
    I = torch.zeros(len(x), device=S.DEV)
    for _ in range(int(round(s.sim_T / dt))):
        k1 = s.f(y); k2 = s.f(y + 0.5 * dt * k1); k3 = s.f(y + 0.5 * dt * k2); k4 = s.f(y + dt * k3)
        I = I + dt * ((y * s.s) ** 2).sum(-1) / (LAM * gam(s, sc, y).clamp_min(1e-6))
        y = torch.nan_to_num(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), nan=1e3).clamp(-1e3, 1e3)
    return torch.where(lab == 0, torch.tanh(a * torch.nan_to_num(I, nan=1e6, posinf=1e6)), torch.ones_like(I))


def train(s, prob, seed, factor, n_c=100000, n_d=3000, n_b=2000):
    torch.manual_seed(seed)
    m = B.PINN(s.n).to(S.DEV)
    m.s = s.s
    a = float(prob.alpha) / factor
    sc = scale_of(s)
    xc = s.sample_box(n_c, seed=seed + 1)
    xc = xc[s.admissible(xc)]
    xd = s.sample_box(n_d, seed=seed + 2)
    xd = xd[s.admissible(xd)]
    Wd = dz_data(s, sc, xd, a)
    pool = torch.cat([xc, s.sample_edge(20000, seed=seed + 4)])
    pool = pool[s.admissible(pool)]
    xb = pool[torch.argsort(gam(s, sc, pool))[:n_b]].detach()
    with torch.no_grad():
        f = LAM * gam(s, sc, xc)[:, None] * s.f(xc)
        om = ((xc * s.s) ** 2).sum(-1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
    g = torch.Generator().manual_seed(seed)
    zero = torch.zeros(1, s.n, device=S.DEV)
    for _ in range(STEPS):
        idx = torch.randint(len(xc), (2048,), generator=g).to(S.DEV)
        x = xc[idx].detach().requires_grad_(True)
        W = m(x)
        (dW,) = torch.autograd.grad(W.sum(), x, create_graph=True)
        res = (dW * f[idx]).sum(-1) + a * om[idx] * (1 - W ** 2)
        loss = (res ** 2).mean() + m(zero).pow(2).mean() + ((m(xb) - 1) ** 2).mean() + ((m(xd) - Wd) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def test(model, prob, name, c1, c2):
    s = prob.sys

    def step(x, T):
        y, ok = core.flow(s, x, T)
        return y, ok & s.admissible(x)

    r = iid_test(model, prob.inner, lambda u: s._map(u.to(S.DEV, S.DT)), step, lambda x: prob.xt(x).norm(dim=-1), s.n, c1, c2, seed_of(name), horizons=EXTRA)
    xg, roa = ground_truth(s)
    reg = region(model, prob, r["c2"], xg)
    r.update(coverage=100 * (reg & roa).sum().item() / roa.sum().item(), false_positive=100 * (reg & ~roa).sum().item() / len(roa))
    return r


def run(s, prob, name, seed, factor):
    m = train(s, prob, seed, factor)
    model = DzClip(m, s)
    c = B.certify_fn(model, prob, f"dz{factor:g}")
    r = test(model, prob, f"dz_{name}_s{seed}", c["c1"], c["c2"])
    r.update(seed=seed, factor=factor)
    torch.save(m.state_dict(), R / f"dz_{name}_s{seed}_f{factor:g}.pt")
    print(f"{name} seed {seed} factor {factor:g}: c1 {r['c1']:.4f} c2 {r['c2']:.4f} coverage {r['coverage']:.2f} FP {r['false_positive']:.3f}", flush=True)
    return r


def main():
    extra = [float(f) for f in os.environ.get("DZ_EXTRA", "30,100").split(",")]
    for name in sys.argv[1:]:
        t0 = time.time()
        s = S.get(name)
        prob = core.Problem(s, core.Config(system=name))
        prob.build_inner(verbose=False)
        old = json.loads((R / f"baseline_dz_{name}.json").read_text()) if (R / f"baseline_dz_{name}.json").exists() else None
        done = {r["factor"] for r in old["grid"]} if old else set()
        grid = (old["grid"] if old else []) + [run(s, prob, name, 0, f) for f in extra if f not in done]
        best = max(grid, key=lambda r: r["coverage"])
        res = old["runs"] if old and best["factor"] == old["runs"][0]["factor"] else [best] + [run(s, prob, name, k, best["factor"]) for k in (1, 2, 3)]
        (R / f"baseline_dz_{name}.json").write_text(json.dumps(dict(grid=grid, runs=res), indent=1))
        print(f"{name}: factor {best['factor']:g}  coverage {[round(r['coverage'], 2) for r in res]}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
