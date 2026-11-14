r"""
Training of the candidate on the TFL closed loops (paper, Section 3: "Neural candidate" and "Training and level selection").

Value function (problem.py):  V = alpha q_ex + Phi,  v = 1 - exp(-V)       (alpha in the code is gamma in the paper)
    q_ex   exact quadratic cost-to-go, D q_ex . f = -zeta^T Q zeta on Omega_ad
    psi    = relu(h - alpha q_ex)                                (margin function)
    Phi    = sup_t psi(y(t))                                     (safety value function, paper, Lemma 1)
Candidate (--ansatz max):  Phi_theta = max(psi, Psi_theta),  Psi_theta = tanh(q_ex / q0) softplus(sharp N_theta(F)) / sharp,
    so V_theta >= max(h, alpha q_ex) and every finite sublevel set lies in Omega_ad.
Training (--loss vi):  fitted semi-Lagrangian value iteration over steps of normalized length tau (sub RK4 substeps of
    f / N_f, N_f = sqrt(1 + |S f|^2)) with a frozen target from the averaged weights; the finite-step, decrease and
    obstacle penalties; the exponential loss weight e^{2 min(V, w_cap)}; counterexamples to the finite-step penalty;
    the factor 1 + w_exit on the regression loss and the obstacle penalty of points whose step leaves Omega_ad.
    Step end points, running maxima and running-cost integrals do not depend on the network and are computed once.
    Constants: paper/table_config3.tex and paper/table_config3b.tex.

The paper's runs use the flags in run_stream.sh.  Step setting: long = --tau 2.0 --sub 80 --gamma-num 1 --w-exit 10,
short = the defaults (tau 0.2, 8 substeps, gamma_num 3, w_exit 0).  Options used only for the ablations of the paper: --loss residual
(PDE residual loss), --ansatz sum (Phi = psi + Psi_theta), --mu-step 0, --w-cap 0.

Run:  python -u zpinn.py <flags of run_stream.sh> --seed 0 --out runs/circ3_s0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from problem import ConstrainedZubov, ProblemConfig


@dataclass
class TrainConfig(ProblemConfig):
    hidden: tuple = (64, 64, 64)
    n_colloc: int = 400000
    n_edge: int = 100000
    n_band: int = 100000
    epochs: int = 100
    batch: int = 4096
    lr: float = 1e-3
    lr_final: float = 1e-5
    v_cap: float = 5.0
    loss: str = "vi"
    lam: float = 0.1
    tau: float = 0.2
    mu: float = 10.0
    sharp: float = 10.0
    ema: float = 0.0
    mu_obs: float = 10.0
    obs_eps: float = 0.01
    ansatz: str = "max"
    cex_every: int = 0
    cex_pool: int = 200000
    cex_max: int = 10000
    cex_jitter: int = 4
    cex_sigma: float = 0.01
    cex_buffer: int = 200000
    sub: int = 8
    step_tau: float = 0.0
    step_sub: int = 40
    mu_step: float = 10.0
    step_margin: float = 0.1
    step_detach: bool = False
    cex_step: bool = False
    cex_rolling: bool = False
    auto_settings: bool = False
    cex_worst: bool = False
    w_exit: float = 0.0
    expectile: float = 0.5
    w_cap: float = 0.0
    q0_quantile: float = 0.01
    snap_every: int = 5
    seed: int = 0
    out: str = "runs/b0.1_s0"


class ZidaniNet(nn.Module):
    def __init__(self, prob: ConstrainedZubov, cfg: TrainConfig, q0: float):
        super().__init__()
        self.prob, self.sys = prob, prob.sys
        self.register_buffer("alpha", torch.tensor(float(prob.alpha)))
        self.register_buffer("beta", torch.tensor(float(cfg.beta)))
        self.register_buffer("q0", torch.tensor(float(q0)))
        self.register_buffer("v_cap", torch.tensor(float(cfg.v_cap)))
        self.register_buffer("sharp", torch.tensor(float(cfg.sharp)))
        self.register_buffer("kappa", torch.tensor(float(cfg.kappa)))
        self.ansatz = cfg.ansatz
        self.sing_idx = tuple(getattr(self.sys, "sing_idx", (0, 2)))
        dims = [self.sys.n_features + self.sys.r + len(self.sing_idx), *cfg.hidden, 1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        with torch.no_grad():
            self.lins[-1].weight.mul_(0.1 / cfg.sharp)
            self.lins[-1].bias.fill_(-4.0 / cfg.sharp)
        self.barrier, self.barrier_delta, self.barrier_tau, self.barrier_eps = "singular", 3.0, 0.1, (0.01, 0.01, 0.05)

    def features(self, x, d=None):
        p = self.prob
        d = p.distances(x) if d is None else d
        z = p.zeta(x)
        idx = list(self.sing_idx)
        sing = d[:, idx].clamp_min(1e-12) / p.scale[idx]
        return torch.cat([self.sys.features(x), torch.tanh(z / 3.0), torch.tanh(0.5 * torch.log(sing))], -1)

    def sp(self, N):
        """softplus(k N) / k: nonnegative, with floor and ripples ~ e^{kN} / k, so Phi - psi can be ~0 where
        the exact Phi equals psi (there the true decrease rate alpha omega is small and the floor's ripples
        along f would otherwise exceed it).  Derivative: sigmoid(k N)."""
        return nn.functional.softplus(self.sharp * N) / self.sharp

    def net(self, F, dF=None):
        """MLP and, if dF is given, its forward-mode derivative along dF."""
        for i, l in enumerate(self.lins):
            F = l(F)
            if dF is not None:
                dF = dF @ l.weight.T
            if i < len(self.lins) - 1:
                F = torch.tanh(F)
                if dF is not None:
                    dF = (1 - F ** 2) * dF
        return (F[:, 0], None if dF is None else dF[:, 0])

    def parts(self, x):
        p = self.prob
        d = p.distances(x)
        h = p.h_from_d(d, self.beta.item())
        qe = p.q_exact(x)
        psi = torch.relu(h - self.alpha * qe)
        s = torch.tanh(qe / self.q0)
        N, _ = self.net(self.features(x, d))
        Psi = s * self.sp(N)
        Phi = torch.maximum(psi, Psi) if self.ansatz == "max" else psi + Psi
        return self.alpha * qe + Phi, Phi, psi

    def V(self, x):
        return self.parts(x)[0]

    def forward(self, x):
        """v = 1 - exp(-V) (1 off Omega_ad)."""
        return 1.0 - torch.exp(-self.V(x))


def collocation(prob: ConstrainedZubov, cfg: TrainConfig):
    s = prob.sys
    xs = [s.sample_box(cfg.n_colloc, seed=cfg.seed + 11), s.sample_edge(cfg.n_edge, seed=cfg.seed + 12)]
    if cfg.n_band:
        xs.append(s.sample_band(cfg.n_band, "singular", 3.0, 0.1, seed=cfg.seed + 13, eps=(0.01, 0.01, 0.05)))
    x = torch.cat(xs)
    with torch.no_grad():
        x = x[prob.admissible(x) & s.in_box(x)]
    return x


def precompute(model: ZidaniNet, x, chunk=20000):
    """Network-independent tensors: features F and dF = DF . f, psi, D psi . f, q_ex, zeta^T Q zeta, N_f."""
    p, s = model.prob, model.sys
    out = {k: [] for k in ("F", "dF", "psi", "dpsi", "qe", "om", "nf", "rho")}
    for xc in x.split(chunk):
        with torch.no_grad():
            f = s.evaluate(xc)["f_cl"]
            nf = torch.sqrt(1.0 + ((f * s.s) ** 2).sum(-1))
            om = p.g(xc, 1.0)
            rho = p.rho(xc) if model.kappa.item() > 0 else torch.zeros_like(om)
        xr = xc.detach().requires_grad_(True)
        F = model.features(xr)
        dF = torch.stack([(torch.autograd.grad(F[:, j].sum(), xr, retain_graph=True)[0] * f).sum(-1)
                          for j in range(F.shape[1])], -1)
        h = p.h(xr, model.beta.item())
        (gh,) = torch.autograd.grad(h.sum(), xr)
        with torch.no_grad():
            qe = p.q_exact(xc)
            act = (h.detach() > model.alpha * qe).float()
            psi = torch.relu(h.detach() - model.alpha * qe)
            dpsi = act * ((gh * f).sum(-1) + model.alpha * om)
        for k, v in zip(out, (F.detach(), dF.detach(), psi, dpsi, qe, om, nf, rho)):
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def phi_terms(model: ZidaniNet, b):
    """Psi_theta = s softplus_k(N), Phi, D Phi . f, V, and where the network part is active."""
    N, dN = model.net(b["F"], b["dF"])
    sp = model.sp(N)
    s = torch.tanh(b["qe"] / model.q0)
    Psi = s * sp
    dPsi = -(1 - s ** 2) * b["om"] / model.q0 * sp + s * torch.sigmoid(model.sharp * N) * dN
    if model.ansatz == "max":
        net_act = Psi >= b["psi"]
        Phi = torch.where(net_act, Psi, b["psi"])
        dPhi = torch.where(net_act, dPsi, b["dpsi"])
    else:
        net_act = torch.ones_like(Psi, dtype=torch.bool)
        Phi, dPhi = b["psi"] + Psi, b["dpsi"] + dPsi
    return Psi, Phi, dPhi, model.alpha * b["qe"] + Phi, net_act


def loss_terms(model: ZidaniNet, b):
    Psi, Phi, dPhi, V, net_act = phi_terms(model, b)
    margin = model.kappa * b["rho"]
    R = torch.minimum(-(dPhi + margin) / b["nf"], Phi - b["psi"])
    viol = torch.relu((dPhi + margin) / b["nf"]) * torch.exp(-V.detach()) * net_act
    return R, viol, V


def cert_violation(model, b):
    """True where the decrease condition fails: DV . f = -alpha omega + DPhi . f >= 0."""
    _, _, dPhi, V, _ = phi_terms(model, b)
    return (dPhi >= model.alpha * b["om"]) & (V < model.v_cap)


@torch.no_grad()
def find_cex(model, cfg, ep):
    """Counterexamples (no labels): fresh box + singular-band samples where DV . f >= 0 and v < 1 - e^{-V_cap},
    plus Gaussian neighbours (sigma = cex_sigma in box-normalized coordinates)."""
    prob, s = model.prob, model.sys
    with torch.enable_grad():
        pool = torch.cat([s.sample_box(cfg.cex_pool, seed=10000 + ep),
                          s.sample_band(cfg.cex_pool // 4, "singular", 3.0, 0.1, seed=20000 + ep,
                                        eps=(0.01, 0.01, 0.05))]).detach()
    pool = pool[prob.admissible(pool) & s.in_box(pool)]
    with torch.enable_grad():
        bp = precompute(model, pool)
    bad = torch.cat([cert_violation(model, {k: v[j] for k, v in bp.items()})
                     for j in torch.arange(len(pool), device=pool.device).split(50000)])
    xb = pool[bad][: cfg.cex_max]
    g = torch.Generator().manual_seed(ep)
    half = half_widths(s, pool.device)
    nb = torch.randn((len(xb) * cfg.cex_jitter, len(half)), generator=g).to(pool) * cfg.cex_sigma * half
    xn = torch.cat([xb, xb.repeat(cfg.cex_jitter, 1) + nb])
    return xn[prob.admissible(xn) & s.in_box(xn)], int(bad.sum().item()), len(pool)


def half_widths(s, device):
    """Half-widths of the box in every coordinate (Gaussian jitter scale), for either system."""
    if hasattr(s, "half"):
        return s.half.to(device)
    return torch.tensor([s.Lx, s.L, math.pi, s.h4], device=device)


def loss_fn(model, b, cfg):
    """loss "super" (default): the true V is the smallest supersolution (GZ Prop. 4.6-4.7), so penalize
    D Phi . f > 0 and push v down: mean(viol^2) + lam mean(v).  Phi >= psi holds by construction.
    loss "residual": mean of the reduced GZ residual squared, faded above V_cap.  It has a trivial
    near-minimizer (Phi large and constant away from M, where the fade switches the loss off), so it
    is used only in the ablation."""
    R, viol, V = loss_terms(model, b)
    if cfg.loss == "residual":
        return ((R * torch.exp(-torch.relu(V.detach() - model.v_cap))) ** 2).mean()
    return (viol ** 2).mean() + cfg.lam * (1 - torch.exp(-V)).mean()


@torch.no_grad()
def precompute_next(model: ZidaniNet, x, tau, sub, chunk=20000):
    """Semi-Lagrangian step (GZ Sec. 5, eq. (5.1)): RK4 over normalized time tau (sub substeps) of the
    normalized field f / N_f (a time reparametrization: Phi = sup psi along the orbit is unchanged).
    Returns the running max M of psi over the step (psi(x) included), and at the end point x': features,
    psi, s.  A step that leaves Omega_ad or the box gets M = inf (v = 1)."""
    p, s = model.prob, model.sys
    out = {k: [] for k in ("M", "C", "F1", "psi1", "s1", "qe1")}
    hs = tau / sub
    fh = lambda z: s.field(z)[0]
    kap = model.kappa.item()

    def rate(z):
        if kap <= 0:
            return torch.zeros(len(z), device=z.device, dtype=z.dtype)
        f = torch.nan_to_num(s.evaluate(z)["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
        return kap * p.rho(z) / torch.sqrt(1.0 + ((f * s.s) ** 2).sum(-1))

    for xc in x.split(chunk):
        y = xc.clone()
        M = torch.relu(p.h(y, model.beta.item()) - model.alpha * p.q_exact(y))
        C = torch.zeros_like(M)
        r0 = rate(y)
        for _ in range(sub):
            k1 = fh(y); k2 = fh(y + 0.5 * hs * k1); k3 = fh(y + 0.5 * hs * k2); k4 = fh(y + hs * k3)
            y = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            ok = p.admissible(y) & s.in_box(y) & torch.isfinite(y).all(-1)
            r1 = rate(torch.nan_to_num(y))
            C = C + 0.5 * hs * (r0 + r1)
            r0 = r1
            psi = torch.relu(p.h(y, model.beta.item()) - model.alpha * p.q_exact(y))
            M = torch.where(ok, torch.maximum(M, C + psi), torch.full_like(M, math.inf))
        y = torch.nan_to_num(y)
        qe1 = p.q_exact(y)
        psi1 = torch.relu(p.h(y, model.beta.item()) - model.alpha * qe1)
        for k, v in zip(out, (M, C, model.features(y), torch.nan_to_num(psi1, posinf=1e30), torch.tanh(qe1 / model.q0),
                              torch.nan_to_num(qe1, posinf=1e30))):
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def precompute_step(model, x, tau, sub):
    """End point of the finite-step test (same RK4 of the normalized field as precompute_next): features, psi,
    s, q_ex there, kappa int rho over the step, and whether every substep stayed admissible and in the box."""
    nx = precompute_next(model, x, tau, sub)
    return dict(F1=nx["F1"], psi1=nx["psi1"], s1=nx["s1"], qe1=nx["qe1"], C=nx["C"], ok=torch.isfinite(nx["M"]))


def step_violation(model, b, st, margin, detach_up=False):
    """Finite-step decrease violation in v units: relu(e^{-(V(x) - margin * exact decrease)} - e^{-V(x_tau)}),
    on steps that stay admissible and where v < 1 - e^{-V_cap}.  Also returns V(x).  With detach_up the upstream
    value V(x) carries no gradient."""
    V = phi_terms(model, b)[3]
    Vu = V.detach() if detach_up else V
    N1, _ = model.net(st["F1"])
    Psi1 = st["s1"] * model.sp(N1)
    Phi1 = torch.maximum(st["psi1"], Psi1) if model.ansatz == "max" else st["psi1"] + Psi1
    V1 = model.alpha * st["qe1"] + Phi1
    req = margin * ((model.alpha * (b["qe"] - st["qe1"])).clamp_min(0) + st["C"])
    live = st["ok"] & (V.detach() < model.v_cap)
    return torch.relu(torch.exp(-(Vu - req)) - torch.exp(-V1)) * live, V


@torch.no_grad()
def find_cex_step(model, cfg, ep):
    """Counterexamples of the finite-step condition (no labels): fresh box + singular-band samples whose
    finite-step test step stays admissible but does not decrease v, plus Gaussian neighbours."""
    prob, s = model.prob, model.sys
    with torch.enable_grad():
        pool = torch.cat([s.sample_box(cfg.cex_pool, seed=10000 + ep),
                          s.sample_band(cfg.cex_pool // 4, "singular", 3.0, 0.1, seed=20000 + ep,
                                        eps=(0.01, 0.01, 0.05))]).detach()
    pool = pool[prob.admissible(pool) & s.in_box(pool)]
    with torch.enable_grad():
        bp = precompute(model, pool)
    sp = precompute_step(model, pool, cfg.step_tau, cfg.step_sub)
    viol = torch.cat([step_violation(model, {k: v[j] for k, v in bp.items()}, {k: v[j] for k, v in sp.items()}, 0.0)[0]
                      for j in torch.arange(len(pool), device=pool.device).split(50000)])
    bad = viol > 0
    if cfg.cex_worst:
        xb = pool[torch.argsort(viol, descending=True)[: min(cfg.cex_max, int(bad.sum().item()))]]
    else:
        xb = pool[bad][: cfg.cex_max]
    g = torch.Generator().manual_seed(ep)
    half = half_widths(s, pool.device)
    nb = torch.randn((len(xb) * cfg.cex_jitter, len(half)), generator=g).to(pool) * cfg.cex_sigma * half
    xn = torch.cat([xb, xb.repeat(cfg.cex_jitter, 1) + nb])
    return xn[prob.admissible(xn) & s.in_box(xn)], int(bad.sum().item()), len(pool)


def train_vi(cfg: TrainConfig, prob, model, x, data, out, t0):
    """Fitted semi-Lagrangian value iteration: regress e^{-Phi(x)} on e^{-max(M(x), Phi_tgt(x'))}, with
    Phi_tgt a copy of the network refreshed every epoch.  Starting from Phi = psi (below the solution),
    the iteration rises monotonically to the smallest fixed point, which is the GZ value function."""
    import copy
    nxt = precompute_next(model, x, cfg.tau, cfg.sub)
    print(f"  semi-Lagrangian step tau {cfg.tau} ({cfg.sub} RK4 substeps): {100 * torch.isinf(nxt['M']).float().mean():.2f} % "
          f"of steps leave Omega_ad  [{time.time() - t0:.0f} s]", flush=True)
    stp = None
    if cfg.step_tau > 0:
        stp = precompute_step(model, x, cfg.step_tau, cfg.step_sub)
        print(f"  finite-step test tau {cfg.step_tau} ({cfg.step_sub} RK4 substeps): {100 * (~stp['ok']).float().mean():.2f} % "
              f"leave Omega_ad / the box  [{time.time() - t0:.0f} s]", flush=True)
    tgt = copy.deepcopy(model)
    ema = copy.deepcopy(model) if cfg.ema > 0 else model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(x)
    steps = cfg.epochs * math.ceil(n / cfg.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=cfg.lr_final)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 5)
    hist = []
    n0 = n
    for ep in range(1, cfg.epochs + 1):
        if cfg.cex_every and ep % cfg.cex_every == 0 and ep <= int(0.9 * cfg.epochs) and (cfg.cex_rolling or n - n0 < cfg.cex_buffer):
            xc, nbad, npool = (find_cex_step if cfg.cex_step else find_cex)(ema, cfg, ep)
            if len(xc):
                dnew = precompute(model, xc)
                nnew = precompute_next(model, xc, cfg.tau, cfg.sub)
                data = {k: torch.cat([data[k], dnew[k]]) for k in data}
                nxt = {k: torch.cat([nxt[k], nnew[k]]) for k in nxt}
                if stp is not None:
                    snew = precompute_step(model, xc, cfg.step_tau, cfg.step_sub)
                    stp = {k: torch.cat([stp[k], snew[k]]) for k in stp}
                n = len(data["qe"])
                if cfg.cex_rolling and n - n0 > cfg.cex_buffer:
                    keep = torch.cat([torch.arange(n0), torch.arange(n - cfg.cex_buffer, n)]).to(cfg.device)
                    data = {k: v[keep] for k, v in data.items()}
                    nxt = {k: v[keep] for k, v in nxt.items()}
                    if stp is not None:
                        stp = {k: v[keep] for k, v in stp.items()}
                    n = len(data["qe"])
            cond = "the finite-step decrease" if cfg.cex_step else "DV.f < 0"
            print(f"  cex epoch {ep}: {nbad} / {npool} pool samples violate {cond}; added {len(xc)}, "
                  f"buffer {n - n0}", flush=True)
        with torch.no_grad():
            Psi1 = torch.cat([nxt["s1"][j] * tgt.sp(tgt.net(nxt["F1"][j])[0])
                              for j in torch.arange(n, device=cfg.device).split(100000)])
            Phi1 = torch.maximum(nxt["psi1"], Psi1) if model.ansatz == "max" else nxt["psi1"] + Psi1
            u_t = torch.exp(-torch.maximum(nxt["M"], nxt["C"] + Phi1))
        perm = torch.randperm(n, generator=g).to(cfg.device)[:n0]
        acc, nb = 0.0, 0
        for i in range(0, n0, cfg.batch):
            idx = perm[i:i + cfg.batch]
            bt = {k: v[idx] for k, v in data.items()}
            _, viol, V = loss_terms(model, bt)
            ut = u_t[idx]
            if model.ansatz == "max":
                Psi, _, _, _, net_act = phi_terms(model, bt)
                above = ut < torch.exp(-bt["psi"]) * (1 - 1e-6)
                eP = torch.exp(-Psi)
                wq = torch.where(eP < ut, 2 * cfg.expectile, 2 * (1 - cfg.expectile))
                fit = torch.where(above, wq * (eP - ut) ** 2, torch.relu(ut - eP) ** 2)
                need = ~net_act & (bt["dpsi"] + model.kappa * bt["rho"] > 0)
                obs = torch.relu(bt["psi"] + cfg.obs_eps - Psi) * need * torch.exp(-V.detach())
                fit = fit + cfg.mu_obs * obs ** 2
            else:
                fit = (torch.exp(-(V - model.alpha * bt["qe"])) - ut) ** 2
            w2 = torch.exp(2 * V.detach().clamp(0, cfg.w_cap)) if cfg.w_cap > 0 else 1.0
            if getattr(cfg, "w_exit", 0.0) > 0:
                fit = fit * (1.0 + cfg.w_exit * torch.isinf(nxt["M"][idx]).float())
            loss = (w2 * fit).mean() + cfg.mu * (w2 * viol ** 2).mean()
            if stp is not None:
                sv, _ = step_violation(model, bt, {k: v[idx] for k, v in stp.items()}, cfg.step_margin, cfg.step_detach)
                loss = loss + cfg.mu_step * (w2 * sv ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if cfg.ema > 0:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(cfg.ema).add_(pm, alpha=1 - cfg.ema)
            acc += loss.item()
            nb += 1
        tgt.load_state_dict(ema.state_dict())
        with torch.no_grad():
            Rs, Ws, Vs = zip(*(loss_terms(ema, {k: v[j] for k, v in data.items()})
                               for j in torch.arange(min(n, 200000), device=cfg.device).split(50000)))
            R, viol, V = torch.cat(Rs), torch.cat(Ws), torch.cat(Vs)
            vv = 1 - torch.exp(-V)
            fs = float("nan")
            if stp is not None:
                fs = torch.cat([step_violation(ema, {k: v[j] for k, v in data.items()}, {k: v[j] for k, v in stp.items()}, 0.0)[0] > 0
                                for j in torch.arange(min(n, 200000), device=cfg.device).split(50000)]).float().mean().item()
        rec = dict(epoch=ep, loss=acc / nb, res_p99=torch.quantile(R.abs(), 0.99).item(), frac_step_viol=fs,
                   frac_viol=(viol > 0).float().mean().item(), frac_v_lt_0_9=(vv < 0.9).float().mean().item(),
                   frac_v_lt_0_5=(vv < 0.5).float().mean().item(), time=time.time() - t0)
        hist.append(rec)
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:4d}  fit {rec['loss']:.3e}  |R| p99 {rec['res_p99']:.3e}  DPhi.f>0 on "
                  f"{100 * rec['frac_viol']:5.2f}%  adm. samples with v<0.5 {100 * rec['frac_v_lt_0_5']:5.1f}%  "
                  f"v<0.9 {100 * rec['frac_v_lt_0_9']:5.1f}%  step viol. {100 * fs:5.2f}%  [{rec['time']:6.0f} s]", flush=True)
        if ep % cfg.snap_every == 0:
            torch.save(ema.state_dict(), out / "snapshots" / f"epoch_{ep:03d}.pt")
    return hist


def build(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    prob = ConstrainedZubov(cfg)
    x = collocation(prob, cfg)
    with torch.no_grad():
        q0 = torch.quantile(prob.q_exact(x[:200000]), cfg.q0_quantile).item()
    model = ZidaniNet(prob, cfg, q0).to(cfg.device)
    return prob, model, x


def train(cfg: TrainConfig):
    if cfg.c_inner <= 0:
        from certify3 import case_name, inner_level
        f = Path(f"results/inner_{case_name(cfg)}.json")
        if f.exists():
            inner = json.loads(f.read_text())
            cfg.c_inner = inner_level(cfg, ConstrainedZubov(cfg))
            if cfg.kappa > 0 and inner["volume_of_box"] < 0.01:
                cfg.kappa = 0.0
            print(f"verified inner level c_inner {cfg.c_inner:.5f} (region {100 * inner['volume_of_box']:.2f} % of box) "
                  f"-> kappa {cfg.kappa:g}", flush=True)
    prob, model, x = build(cfg)
    out = Path(cfg.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = precompute(model, x)
    print(f"alpha {model.alpha.item():.4f}  beta {cfg.beta}  q0 {model.q0.item():.3e}  collocation {len(x)}  "
          f"precompute {time.time() - t0:.0f} s  device {cfg.device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(x)
    steps = cfg.epochs * math.ceil(n / cfg.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=cfg.lr_final)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 5)
    hist = []
    torch.save(model.state_dict(), out / "snapshots" / "epoch_000.pt")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if cfg.loss == "vi":
        hist = train_vi(cfg, prob, model, x, data, out, t0)
        if cfg.ema > 0:
            model.load_state_dict(torch.load(sorted((out / "snapshots").glob("epoch_*.pt"))[-1]))
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out / "model.pt")
        return model
    for ep in range(1, cfg.epochs + 1):
        perm = torch.randperm(n, generator=g).to(cfg.device)
        acc, nb = 0.0, 0
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            loss = loss_fn(model, {k: v[idx] for k, v in data.items()}, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            acc += loss.item()
            nb += 1
        with torch.no_grad():
            Rs, Ws, Vs = zip(*(loss_terms(model, {k: v[j] for k, v in data.items()})
                               for j in torch.arange(min(n, 200000), device=cfg.device).split(50000)))
            R, viol, V = torch.cat(Rs), torch.cat(Ws), torch.cat(Vs)
            vv = 1 - torch.exp(-V)
        rec = dict(epoch=ep, loss=acc / nb, res_p99=torch.quantile(R.abs(), 0.99).item(),
                   frac_viol=(viol > 0).float().mean().item(),
                   frac_v_lt_0_9=(vv < 0.9).float().mean().item(), frac_v_lt_0_5=(vv < 0.5).float().mean().item(),
                   time=time.time() - t0)
        hist.append(rec)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:4d}  loss {rec['loss']:.3e}  |R| p99 {rec['res_p99']:.3e}  "
                  f"DPhi.f>0 on {100 * rec['frac_viol']:5.2f}%  adm. samples with "
                  f"v<0.5 {100 * rec['frac_v_lt_0_5']:5.1f}%  v<0.9 {100 * rec['frac_v_lt_0_9']:5.1f}%  "
                  f"[{rec['time']:6.0f} s]", flush=True)
        if ep % cfg.snap_every == 0:
            torch.save(model.state_dict(), out / "snapshots" / f"epoch_{ep:03d}.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out / "model.pt")
    return model


def load(run):
    d = json.loads((Path(run) / "config.json").read_text())
    cfg = TrainConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    prob = ConstrainedZubov(cfg)
    sd = torch.load(Path(run) / "snapshots" / "epoch_000.pt", map_location=cfg.device)
    model = ZidaniNet(prob, cfg, sd["q0"].item()).to(cfg.device)
    model.load_state_dict(sd)
    return prob, model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f_ in TrainConfig.__dataclass_fields__.values():
        d = f_.default
        if isinstance(d, bool):
            ap.add_argument(f"--{f_.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"),
                            default=d)
        elif isinstance(d, tuple):
            ap.add_argument(f"--{f_.name.replace('_', '-')}", type=type(d[0]), nargs=len(d), default=d)
        else:
            ap.add_argument(f"--{f_.name.replace('_', '-')}", type=type(d), default=d)
    a = vars(ap.parse_args())
    train(TrainConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in a.items()}))


if __name__ == "__main__":
    main()
