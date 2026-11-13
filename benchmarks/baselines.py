r"""
Quadratic and Zubov PINN baselines on the benchmarks, with the same level selection (certify.two_stage) and the same final
test (certify_iid.py) as the candidate, so only the function differs:

  quad   the quadratic Lyapunov function of the linearization, v = 1 - e^{-q}, q = x~^T P x~ (A~^T P + P A~ = -I)
  pinn   Zubov PINN of Liu, Meng, Fitzsimmons & Zhou (Automatica 2025; LyZNet): W = tanh(a V), PDE residual
         DW.f + a omega (1 - W^2) = 0, omega = |x~|^2, boundary W(0) = 0, and their data loss on simulated values
         W^(z) = tanh(a int omega dt) (trajectories that leave the domain or do not converge: W^ = 1).
         Two hidden layers of 30 tanh units, as in the paper.  Note: it uses simulation data, the proposed method none.

Run:  python -u baselines.py quad vdp pendulum con3d power ;  python -u baselines.py pinn vdp ...
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

import core
import systems as S
from certify import two_stage, region
from gt import ground_truth

OUT = Path(__file__).parent / "results"


class Quad(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        return 1.0 - torch.exp(-self.prob.q(x))


class PINN(nn.Module):
    def __init__(self, n, width=30, depth=2):
        super().__init__()
        dims = [n, *([width] * depth), 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])
        self.s = None

    def forward(self, x):
        return self.net(x * self.s)[:, 0]


@torch.no_grad()
def zubov_data(sys_, x, a, dt=0.01):
    """W^ = tanh(a int_0^T |x~|^2 dt) along simulated trajectories; 1 if the trajectory leaves or does not converge."""
    lab = sys_.simulate(x)
    y = x.clone()
    I = torch.zeros(len(x), device=S.DEV)
    for _ in range(int(round(sys_.sim_T / dt))):
        f = sys_.f
        k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        I = I + dt * ((y * sys_.s) ** 2).sum(-1)
        y = torch.nan_to_num(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), nan=1e3).clamp(-1e3, 1e3)
    return torch.where(lab == 0, torch.tanh(a * I), torch.ones_like(I))


def train_pinn(sys_, prob, seed=0, steps=20000, n_c=100000, n_d=3000, lam_b=1.0, lam_d=1.0, factor=1.0):
    torch.manual_seed(seed)
    m = PINN(sys_.n).to(S.DEV)
    m.s = sys_.s
    a = prob.alpha / factor
    xc = sys_.sample_box(n_c, seed=seed + 1)
    xd = sys_.sample_box(n_d, seed=seed + 2)
    Wd = zubov_data(sys_, xd, a)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    g = torch.Generator().manual_seed(seed)
    zero = torch.zeros(1, sys_.n, device=S.DEV)
    for it in range(steps):
        idx = torch.randint(n_c, (2048,), generator=g).to(S.DEV)
        x = xc[idx].requires_grad_(True)
        W = m(x)
        (dW,) = torch.autograd.grad(W.sum(), x, create_graph=True)
        om = ((x * sys_.s) ** 2).sum(-1)
        res = (dW * sys_.f(x)).sum(-1) + a * om * (1 - W ** 2)
        loss = (res ** 2).mean() + lam_b * m(zero).pow(2).mean() + lam_d * ((m(xd) - Wd) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return m


class Clip(nn.Module):
    """v = clamp(W, 0, 1) for the certificate (sublevel sets of W below 1 are unchanged)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).clamp(0.0, 1.0)


def certify_fn(model, prob, tag):
    c1a, c2a, _, _ = two_stage(model, prob, 100000, 5, 9000)
    c1b, c2b, _, NB = two_stage(model, prob, 100000, 5, 23000)
    c1, c2 = min(c1a, c1b), min(c2a, c2b)
    xg, roa = ground_truth(prob.sys)
    reg = region(model, prob, c2, xg)
    nr = roa.sum().item()
    res = dict(method=tag, system=prob.sys.name, c1=c1, c2=c2, coverage=100 * (reg & roa).sum().item() / nr,
               false_positive=100 * (reg & ~roa).sum().item() / len(roa), scenario_eps=math.log(1e6) / NB)
    print(f"{tag:6s} {prob.sys.name:9s} c2={c2:.4f}  coverage {res['coverage']:6.2f}%  FP {res['false_positive']:.2f}%", flush=True)
    return res


def main():
    kind, names = sys.argv[1], sys.argv[2:]
    for name in names:
        t0 = time.time()
        sys_ = S.get(name)
        prob = core.Problem(sys_, core.Config(system=name))
        prob.build_inner(verbose=False)
        if kind == "quad":
            res = [certify_fn(Quad(prob), prob, "quad")]
        else:
            res = []
            for seed in range(4):
                m = train_pinn(sys_, prob, seed)
                r = certify_fn(Clip(m), prob, "pinn")
                r["seed"] = seed
                res.append(r)
                torch.save(m.state_dict(), OUT / f"pinn_{name}_s{seed}.pt")
        (OUT / f"baseline_{kind}_{name}.json").write_text(json.dumps(res, indent=1))
        print(f"  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
