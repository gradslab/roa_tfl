r"""
Baselines on the TFL closed loops, with the level selection and final test of certify3.py exactly like the candidate.

  pinn   Zubov PINN of Liu, Meng, Fitzsimmons & Zhou (Automatica 2025; LyZNet) with these settings: W = tanh(a V), residual
         DW.f + a omega (1 - W^2) = 0 (divided by N_f = sqrt(1 + |S f|^2), a positive weight, because f is unbounded near the singular set),
         boundary W = 0 on target points, data loss on 3000 simulated states, W^ = tanh(a q_ex) if the state converges and 1 otherwise.
         Two hidden layers of 30 tanh units on the normalized state features, 1e5 collocation points, 2e4 Adam steps (batch 2048, cosine lr 1e-3 to 1e-5).
         It uses simulation labels; the candidate uses none.  Its sublevel sets are intersected with Omega_ad at evaluation (v = 1 off Omega_ad).
  quad   the transverse quadratic alone, v = 1 - exp(-alpha q_ex) (1 off Omega_ad), with the level chosen by the same rule.

Run:  python -u zpinn_liu.py pinn|quad circle|sinusoid|quadrotor <seed> <out>
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from problem import ConstrainedZubov
from zpinn import TrainConfig


def case_cfg(case, seed, out):
    extra = {"circle": {}, "sinusoid": dict(path="y-sin(x)", pi="x", periodic_x1=True), "quadrotor": dict(system="quadrotor")}[case]
    return TrainConfig(beta=0.1, h_pow=2.0, kappa=0.0, device="cpu", seed=seed, out=str(out), **extra)


class PinnModel(nn.Module):
    """rich: the input features and the 64-64-64 network of the candidate (sensitivity check of the PINN baselines; PINN_RICH=1, marker rich.flag)."""
    def __init__(self, prob, width=30, depth=2, rich=False):
        super().__init__()
        self.prob = prob
        self.rich = rich
        if rich:
            width, depth = 64, 3
            self.sing = list(getattr(prob.sys, "sing_idx", (0, 2)))
        n_in = prob.sys.n_features + (prob.sys.r + len(self.sing) if rich else 0)
        dims = [n_in, *([width] * depth), 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])

    def input(self, x):
        F = self.prob.sys.features(x)
        if self.rich:
            d = self.prob.distances(x)[:, self.sing].clamp_min(1e-12) / self.prob.scale[self.sing]
            F = torch.cat([F, torch.tanh(self.prob.zeta(x) / 3.0), torch.tanh(0.5 * torch.log(d))], -1)
        return F

    def W(self, x):
        return self.net(self.input(x))[:, 0]

    def forward(self, x):
        ok = self.prob.admissible(x)
        w = self.W(torch.nan_to_num(x)).clamp(max=1.0)
        return torch.where(ok, w, torch.ones(len(x), dtype=x.dtype, device=x.device))


class QuadModel(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self.register_buffer("unused", torch.zeros(1))

    def forward(self, x):
        ok = self.prob.admissible(x)
        v = 1.0 - torch.exp(-self.prob.alpha * self.prob.q_exact(x))
        return torch.where(ok, torch.nan_to_num(v, nan=1.0), torch.ones(len(x), dtype=x.dtype, device=x.device))


def load_baseline(run):
    d = json.loads((Path(run) / "config.json").read_text())
    cfg = TrainConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    cfg.device = "cpu"
    prob = ConstrainedZubov(cfg)
    model = PinnModel(prob, rich=(Path(run) / "rich.flag").exists()) if (Path(run) / "pinn.flag").exists() else QuadModel(prob)
    return prob, model, cfg


def write_common(cfg, out, flag):
    out = Path(out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / flag).write_text("baseline\n")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return out


def train_pinn(case, seed, out, steps=20000, n_c=100000, n_d=3000, batch=2048, factor=1.0):
    t0 = time.time()
    cfg = case_cfg(case, seed, out)
    out = write_common(cfg, out, "pinn.flag")
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
        f = torch.nan_to_num(s.evaluate(xc)["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
        nf = torch.sqrt(1.0 + ((f * s.s) ** 2).sum(-1))
        om = prob.g(xc, 1.0)
    xd = s.sample_box(n_d, seed=seed + 2)
    with torch.no_grad():
        lab = s.simulate(xd)
        Wd = torch.where(lab == 0, torch.tanh(a * prob.q_exact(xd)), torch.ones(len(xd)))
    xd = torch.nan_to_num(xd)
    with torch.enable_grad():
        xt = s.target_points(2000, seed=seed + 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    g = torch.Generator().manual_seed(seed)
    for it in range(steps):
        idx = torch.randint(len(xc), (batch,), generator=g)
        x = xc[idx].detach().requires_grad_(True)
        W = m.W(x)
        (dW,) = torch.autograd.grad(W.sum(), x, create_graph=True)
        res = ((dW * f[idx]).sum(-1) + a * om[idx] * (1 - W ** 2)) / nf[idx]
        loss = (res ** 2).mean() + (m.W(xt) ** 2).mean() + ((m.W(xd) - Wd) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 5000 == 0:
            torch.save(m.state_dict(), out / "snapshots" / f"epoch_{(it + 1) // 200:03d}.pt")
            print(f"step {it + 1} loss {loss.item():.3e} [{time.time() - t0:.0f} s]", flush=True)


def make_quad(case, out):
    cfg = case_cfg(case, 0, out)
    out = write_common(cfg, out, "quad.flag")
    prob = ConstrainedZubov(cfg)
    torch.save(QuadModel(prob).state_dict(), out / "snapshots" / "epoch_000.pt")


if __name__ == "__main__":
    kind, case, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    train_pinn(case, seed, out, factor=float(sys.argv[5]) if len(sys.argv) > 5 else 1.0) if kind == "pinn" else make_quad(case, out)
