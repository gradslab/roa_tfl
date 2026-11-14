r"""
Candidate for the equilibrium benchmarks (paper, Appendix D, "Closed loops with an equilibrium"): state-constrained Zubov value function
(Gruene & Zidani 2015) learned by fitted semi-Lagrangian value iteration.
The same code and the same configuration (Config defaults) for every system; a system only supplies problem data.
Step setting: short = the defaults (tau 0.2, 8 substeps, gamma_num 3, w_exit 0), long = --tau 2 --sub 80 --gamma-num 1
--w-exit 10, with alpha = gamma_num / (90th percentile of q over admissible samples); step_rule.py keeps one per system.

    V(x) = sup_{t >= 0} [ int_0^t g(y) ds + h(y(t)) ],   v = 1 - e^{-V},   ROA = {v < 1}
    g = alpha |x~|^2 + kappa rho,  x~ = S x (box-normalized),  h = sum_i relu(b_i / d_i - 1)^2  (inf off Omega_ad)

Model (as for the TFL unicycle, with the exact transverse cost replaced by the linearization quadratic):
    V = alpha q + Phi,  q = x~^T P x~  (A~^T P + P A~ = -I),  Phi = max(psi, Psi),  psi = relu(h - alpha q),
    Psi = tanh(q / q0) softplus(k N(F)) / k,  F = (normalized x, tanh(x~ / 3), tanh(log(d_j / scale_j) / 2) for the
    non-box constraints).  V >= h by construction, so {v < c} lies in Omega_ad for every c < 1.

Training (unsupervised: only the model equations):
    value iteration  Phi(x) = max( M(x), C(x) + Phi(x_tau) )   over a step tau of the normalized field, with
                     C = int g + alpha (q(x_tau) - q(x)),  M = running max of C + psi along the step (inf if it leaves)
    GZ inequality    DV.f <= -g (supersolution penalty) and obstacle-slope penalty
    finite-step      V(x_1) <= V(x) - margin int_0^1 g   (the condition of level selection, tau = 1), with counterexamples
    loss in V units up to level w_cap (weights e^{2 min(V, w_cap)}), EMA weights.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import systems as S


@dataclass
class Config:
    beta: float = 0.1
    h_pow: float = 2.0
    kappa: float = 0.3
    hidden: tuple = (64, 64, 64)
    n_colloc: int = 400000
    n_edge: int = 100000
    n_band: int = 100000
    epochs: int = 100
    batch: int = 4096
    lr: float = 1e-3
    lr_final: float = 1e-5
    tau: float = 0.2
    sub: int = 8
    gamma_num: float = 3.0
    w_exit: float = 0.0
    mu: float = 10.0
    mu_obs: float = 10.0
    obs_eps: float = 0.01
    sharp: float = 10.0
    ema: float = 0.999
    step_tau: float = 1.0
    step_sub: int = 40
    mu_step: float = 100.0
    step_margin: float = 0.2
    cex_every: int = 5
    cex_pool: int = 200000
    cex_max: int = 10000
    cex_jitter: int = 4
    cex_sigma: float = 0.01
    cex_buffer: int = 100000
    w_cap: float = 3.0
    v_cap: float = 8.0
    q0_quantile: float = 0.01
    snap_every: int = 5
    system: str = "vdp"
    seed: int = 0
    out: str = "runs/vdp_s0"


class Problem:
    def __init__(self, sys_: S.System, cfg: Config):
        from scipy.linalg import solve_continuous_lyapunov
        self.sys, self.cfg = sys_, cfg
        s = sys_.s.double().cpu().numpy()
        A = sys_.jacobian0()
        At = np.diag(s) @ A @ np.diag(1 / s)
        if np.max(np.linalg.eigvals(At).real) >= 0:
            raise ValueError("equilibrium is not exponentially stable")
        P = solve_continuous_lyapunov(At.T, -np.eye(sys_.n))
        self.P = torch.as_tensor(P, dtype=S.DT, device=S.DEV)
        x = sys_.sample_box(50000, seed=123)
        with torch.no_grad():
            d = sys_.distances(x)
            adm = (d > 0).all(-1)
            self.scale = torch.quantile(d[adm], 0.5, dim=0)
            self.alpha = cfg.gamma_num / torch.quantile(self.q(x[adm]), 0.9).item()
        self.kappa = cfg.kappa
        self.inner_c = None

    def xt(self, x):
        return x * self.sys.s

    def q(self, x):
        z = self.xt(x)
        return torch.einsum("ni,ij,nj->n", z, self.P, z)

    def h_from_d(self, d):
        b = self.cfg.beta * self.scale
        phi = torch.where(d > 0, torch.relu(b / d.clamp_min(1e-30) - 1.0) ** self.cfg.h_pow, torch.full_like(d, math.inf))
        return phi.sum(-1)

    def h(self, x):
        return self.h_from_d(self.sys.distances(x))

    def inner(self, x):
        return self.q(x) / self.inner_c

    def rho(self, x):
        t = ((self.inner(x) - 0.5) / 0.5).clamp(0.0, 1.0)
        return t * t * (3 - 2 * t)

    def g(self, x):
        k = self.kappa * self.rho(x) if self.kappa > 0 else 0.0
        return self.alpha * (self.xt(x) ** 2).sum(-1) + k

    @torch.no_grad()
    def build_inner(self, n=200000, rounds=5, seed=4321, verbose=True):
        sys_ = self.sys
        x = sys_.sample_box(n, seed=seed)
        xb, nb = sample_boundary(sys_, n // 2, seed + 1)
        xo = xb[(torch.nan_to_num(sys_.f(xb)) * nb).sum(-1) >= 0]
        xe = torch.cat([x[~sys_.admissible(x)], xo])
        self.inner_c = self.q(xe).min().item() * (1 - 1e-4)
        for r in range(rounds):
            x = torch.cat([sys_.sample_box(n, seed=seed + 10 + r), sys_.sample_edge(n, seed=seed + 60 + r)])
            I = self.inner(x)
            x, I = x[I < 1], I[I < 1]
            bad = torch.zeros(len(x), dtype=torch.bool, device=x.device)
            for j in torch.arange(len(x), device=x.device).split(20000):
                y, ok = flow(sys_, x[j], 1.0)
                small = (self.xt(x[j])).norm(dim=-1) <= 1e-3
                bad[j] = ~ok | ((self.inner(y) >= I[j]) & ~small)
            if not bad.any():
                break
            self.inner_c *= I[bad].min().item() * (1 - 1e-4)
        vol = (self.inner(sys_.sample_box(100000, seed=31337)) < 1).float().mean().item()
        if self.cfg.kappa > 0 and vol < 0.01:
            self.kappa = 0.0
        if verbose:
            print(f"  inner region {100 * vol:.2f} % of the box -> kappa {self.kappa:g}", flush=True)
        return vol


def sample_boundary(sys_, n, seed):
    """Uniform samples on the box faces with outward normals (face chosen in proportion to its area)."""
    g = torch.Generator().manual_seed(int(seed))
    span = (sys_.hi - sys_.lo).cpu()
    areas = torch.tensor([float(torch.prod(torch.cat([span[:i], span[i + 1:]]))) for i in range(sys_.n)] * 2)
    face = torch.multinomial(areas / areas.sum(), n, replacement=True, generator=g).to(S.DEV)
    x = sys_._map(torch.rand(n, sys_.n, generator=g).to(S.DEV, S.DT))
    nrm = torch.zeros_like(x)
    for k in range(2 * sys_.n):
        i, hi = k % sys_.n, k >= sys_.n
        m = face == k
        x[m, i] = sys_.hi[i] if hi else sys_.lo[i]
        nrm[m, i] = 1.0 if hi else -1.0
    return x, nrm


@torch.no_grad()
def flow(sys_, x, tau, h=0.025):
    """RK4 of the normalized field over normalized time tau; also whether every substep stays admissible."""
    sub = max(1, math.ceil(tau / h))
    hs = tau / sub
    y = x.clone()
    ok = torch.ones(len(x), dtype=torch.bool, device=x.device)
    F = sys_.field
    for _ in range(sub):
        k1 = F(y); k2 = F(y + 0.5 * hs * k1); k3 = F(y + 0.5 * hs * k2); k4 = F(y + hs * k3)
        y1 = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ok = ok & torch.isfinite(y1).all(-1)
        y = torch.where(ok[:, None], y1, y)
        ok = ok & sys_.admissible(y)
    return y, ok


class Net(nn.Module):
    def __init__(self, prob: Problem, cfg: Config, q0: float):
        super().__init__()
        self.prob, self.sys = prob, prob.sys
        self.register_buffer("alpha", torch.tensor(float(prob.alpha)))
        self.register_buffer("q0", torch.tensor(float(q0)))
        self.register_buffer("v_cap", torch.tensor(float(cfg.v_cap)))
        self.register_buffer("sharp", torch.tensor(float(cfg.sharp)))
        self.register_buffer("kappa", torch.tensor(float(prob.kappa)))
        self.register_buffer("inner_c", torch.tensor(float(prob.inner_c)))
        self.m = len(self.sys.cons)
        dims = [2 * self.sys.n + self.m, *cfg.hidden, 1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        with torch.no_grad():
            self.lins[-1].weight.mul_(0.1 / cfg.sharp)
            self.lins[-1].bias.fill_(-4.0 / cfg.sharp)

    def features(self, x, d=None):
        s, p = self.sys, self.prob
        F = [(x - s.center) / s.half, torch.tanh(p.xt(x) / 3.0)]
        if self.m:
            d = s.distances(x) if d is None else d
            F.append(torch.tanh(0.5 * torch.log(d[:, :self.m].clamp_min(1e-12) / p.scale[:self.m])))
        return torch.cat(F, -1)

    def sp(self, N):
        return nn.functional.softplus(self.sharp * N) / self.sharp

    def net(self, F, dF=None):
        for i, l in enumerate(self.lins):
            F = l(F)
            if dF is not None:
                dF = dF @ l.weight.T
            if i < len(self.lins) - 1:
                F = torch.tanh(F)
                if dF is not None:
                    dF = (1 - F ** 2) * dF
        return F[:, 0], None if dF is None else dF[:, 0]

    def V(self, x):
        p = self.prob
        d = self.sys.distances(x)
        h = p.h_from_d(d)
        q = p.q(x)
        psi = torch.relu(h - self.alpha * q)
        Psi = torch.tanh(q / self.q0) * self.sp(self.net(self.features(x, d))[0])
        return self.alpha * q + torch.maximum(psi, Psi)

    def forward(self, x):
        return 1.0 - torch.exp(-self.V(x))


def collocation(prob, cfg):
    s = prob.sys
    xs = [s.sample_box(cfg.n_colloc, seed=cfg.seed + 11), s.sample_edge(cfg.n_edge, seed=cfg.seed + 12)]
    if cfg.n_band and s.cons:
        xs.append(s.sample_band(cfg.n_band, seed=cfg.seed + 13))
    x = torch.cat(xs)
    with torch.no_grad():
        return x[s.admissible(x)]


def precompute(model, x, chunk=20000):
    """Network-independent tensors: F, dF = DF.f, psi, D psi.f, q, D q.f, g, N_f, rho."""
    p, s = model.prob, model.sys
    out = {k: [] for k in ("F", "dF", "psi", "dpsi", "q", "dq", "g", "nf", "rho")}
    for xc in x.split(chunk):
        with torch.no_grad():
            f = s.f(xc)
            nf = torch.sqrt(1.0 + ((f * s.s) ** 2).sum(-1))
            g = p.g(xc)
            rho = p.rho(xc) if p.kappa > 0 else torch.zeros_like(g)
        xr = xc.detach().requires_grad_(True)
        F = model.features(xr)
        dF = torch.stack([(torch.autograd.grad(F[:, j].sum(), xr, retain_graph=True)[0] * f).sum(-1)
                          for j in range(F.shape[1])], -1)
        q = p.q(xr)
        (gq,) = torch.autograd.grad(q.sum(), xr, retain_graph=True)
        h = p.h(xr)
        (gh,) = torch.autograd.grad(h.sum(), xr)
        with torch.no_grad():
            dq = (gq * f).sum(-1)
            act = (h > model.alpha * q).float()
            psi = torch.relu(h - model.alpha * q)
            dpsi = act * ((gh * f).sum(-1) - model.alpha * dq)
        for k, v in zip(out, (F.detach(), dF.detach(), psi.detach(), dpsi, q.detach(), dq, g, nf, rho)):
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def phi_terms(model, b):
    N, dN = model.net(b["F"], b["dF"])
    sp = model.sp(N)
    s = torch.tanh(b["q"] / model.q0)
    Psi = s * sp
    dPsi = (1 - s ** 2) * b["dq"] / model.q0 * sp + s * torch.sigmoid(model.sharp * N) * dN
    act = Psi >= b["psi"]
    Phi = torch.where(act, Psi, b["psi"])
    dPhi = torch.where(act, dPsi, b["dpsi"])
    return Psi, Phi, dPhi, model.alpha * b["q"] + Phi, act


@torch.no_grad()
def precompute_next(model, x, tau, sub, chunk=20000):
    """Semi-Lagrangian step: running max M of C + psi along the step (inf if it leaves Omega_ad), C at the end
    (C = int g dt + alpha (q(y) - q(x)), dt = ds / N_f), and the end point's features, psi, s, q."""
    p, s = model.prob, model.sys
    out = {k: [] for k in ("M", "C", "G", "F1", "psi1", "s1", "q1")}
    hs = tau / sub
    F = s.field
    rate = lambda z: p.g(z) / torch.sqrt(1.0 + ((torch.nan_to_num(s.f(z)) * s.s) ** 2).sum(-1))
    a = model.alpha
    for xc in x.split(chunk):
        y = xc.clone()
        q0 = p.q(y)
        M = torch.relu(p.h(y) - a * q0)
        G = torch.zeros_like(M)
        r0 = rate(y)
        for _ in range(sub):
            k1 = F(y); k2 = F(y + 0.5 * hs * k1); k3 = F(y + 0.5 * hs * k2); k4 = F(y + hs * k3)
            y = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            ok = s.admissible(y) & torch.isfinite(y).all(-1)
            yz = torch.nan_to_num(y)
            r1 = rate(yz)
            G = G + 0.5 * hs * (r0 + r1)
            r0 = r1
            qy = p.q(yz)
            C = G + a * (qy - q0)
            psi = torch.relu(p.h(yz) - a * qy)
            M = torch.where(ok, torch.maximum(M, C + psi), torch.full_like(M, math.inf))
        y = torch.nan_to_num(y)
        q1 = p.q(y)
        psi1 = torch.relu(p.h(y) - a * q1)
        C = G + a * (q1 - q0)
        for k, v in zip(out, (M, C, G, model.features(y), torch.nan_to_num(psi1, posinf=1e30), torch.tanh(q1 / model.q0), q1)):
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def precompute_step(model, x, tau, sub):
    nx = precompute_next(model, x, tau, sub)
    return dict(F1=nx["F1"], psi1=nx["psi1"], s1=nx["s1"], q1=nx["q1"], G=nx["G"], ok=torch.isfinite(nx["M"]))


def step_violation(model, b, st, margin):
    """relu(e^{-(V(x) - margin int g)} - e^{-V(x_tau)}) on admissible steps below v_cap (v units), and V(x)."""
    V = phi_terms(model, b)[3]
    N1, _ = model.net(st["F1"])
    Phi1 = torch.maximum(st["psi1"], st["s1"] * model.sp(N1))
    V1 = model.alpha * st["q1"] + Phi1
    live = st["ok"] & (V.detach() < model.v_cap)
    return torch.relu(torch.exp(-(V - margin * st["G"])) - torch.exp(-V1)) * live, V


def loss_terms(model, b):
    Psi, Phi, dPhi, V, act = phi_terms(model, b)
    gap = (model.alpha * b["dq"] + dPhi + b["g"]) / b["nf"]
    viol = torch.relu(gap) * torch.exp(-V.detach()) * act
    return viol, V


@torch.no_grad()
def find_cex(model, cfg, ep):
    prob, s = model.prob, model.sys
    with torch.enable_grad():
        xs = [s.sample_box(cfg.cex_pool, seed=10000 + ep)]
        if s.cons:
            xs.append(s.sample_band(cfg.cex_pool // 4, seed=20000 + ep))
        pool = torch.cat(xs).detach()
    pool = pool[s.admissible(pool)]
    with torch.enable_grad():
        bp = precompute(model, pool)
    sp = precompute_step(model, pool, cfg.step_tau, cfg.step_sub)
    viol = torch.cat([step_violation(model, {k: v[j] for k, v in bp.items()}, {k: v[j] for k, v in sp.items()}, 0.0)[0]
                      for j in torch.arange(len(pool), device=pool.device).split(50000)])
    nbad = int((viol > 0).sum())
    xb = pool[torch.argsort(viol, descending=True)[:min(cfg.cex_max, nbad)]]
    g = torch.Generator().manual_seed(ep)
    nb = torch.randn((len(xb) * cfg.cex_jitter, s.n), generator=g).to(pool) * cfg.cex_sigma * s.half
    xn = torch.cat([xb, xb.repeat(cfg.cex_jitter, 1) + nb])
    return xn[s.admissible(xn)], nbad, len(pool)


def build(cfg):
    torch.manual_seed(cfg.seed)
    sys_ = S.get(cfg.system)
    prob = Problem(sys_, cfg)
    prob.build_inner()
    x = collocation(prob, cfg)
    with torch.no_grad():
        q0 = torch.quantile(prob.q(x[:200000]), cfg.q0_quantile).item()
    return prob, Net(prob, cfg, q0).to(S.DEV), x


def train(cfg: Config):
    t0 = time.time()
    out = Path(cfg.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    prob, model, x = build(cfg)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    print(f"[{cfg.system} seed {cfg.seed}] {len(x)} collocation points, alpha {prob.alpha:.4f}, kappa {prob.kappa:g}, "
          f"inner c {prob.inner_c:.4g}", flush=True)
    data = precompute(model, x)
    nxt = precompute_next(model, x, cfg.tau, cfg.sub)
    stp = precompute_step(model, x, cfg.step_tau, cfg.step_sub)
    print(f"  precomputed [{time.time() - t0:.0f} s]; {100 * torch.isinf(nxt['M']).float().mean():.2f} % of VI steps and "
          f"{100 * (~stp['ok']).float().mean():.2f} % of certificate steps leave Omega_ad", flush=True)
    tgt, ema = copy.deepcopy(model), copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = n0 = len(x)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs * math.ceil(n0 / cfg.batch), eta_min=cfg.lr_final)
    g = torch.Generator().manual_seed(cfg.seed + 5)
    hist = []
    for ep in range(1, cfg.epochs + 1):
        if cfg.cex_every and ep % cfg.cex_every == 0 and ep <= int(0.9 * cfg.epochs):
            xc, nbad, npool = find_cex(ema, cfg, ep)
            if len(xc):
                dn, nn_, sn = precompute(model, xc), precompute_next(model, xc, cfg.tau, cfg.sub), \
                    precompute_step(model, xc, cfg.step_tau, cfg.step_sub)
                data = {k: torch.cat([data[k], dn[k]]) for k in data}
                nxt = {k: torch.cat([nxt[k], nn_[k]]) for k in nxt}
                stp = {k: torch.cat([stp[k], sn[k]]) for k in stp}
                n = len(data["q"])
                if n - n0 > cfg.cex_buffer:
                    keep = torch.cat([torch.arange(n0), torch.arange(n - cfg.cex_buffer, n)]).to(S.DEV)
                    data = {k: v[keep] for k, v in data.items()}
                    nxt = {k: v[keep] for k, v in nxt.items()}
                    stp = {k: v[keep] for k, v in stp.items()}
                    n = len(data["q"])
        with torch.no_grad():
            Psi1 = torch.cat([nxt["s1"][j] * tgt.sp(tgt.net(nxt["F1"][j])[0]) for j in torch.arange(n, device=S.DEV).split(100000)])
            u_t = torch.exp(-torch.maximum(nxt["M"], nxt["C"] + torch.maximum(nxt["psi1"], Psi1)))
        perm = torch.randperm(n, generator=g).to(S.DEV)[:n0]
        acc, nb = 0.0, 0
        for i in range(0, n0, cfg.batch):
            idx = perm[i:i + cfg.batch]
            bt = {k: v[idx] for k, v in data.items()}
            viol, V = loss_terms(model, bt)
            Psi, _, _, _, act = phi_terms(model, bt)
            ut = u_t[idx]
            above = ut < torch.exp(-bt["psi"]) * (1 - 1e-6)
            eP = torch.exp(-Psi)
            fit = torch.where(above, (eP - ut) ** 2, torch.relu(ut - eP) ** 2)
            need = ~act & ((model.alpha * bt["dq"] + bt["dpsi"] + bt["g"]) > 0)
            fit = fit + cfg.mu_obs * (torch.relu(bt["psi"] + cfg.obs_eps - Psi) * need * torch.exp(-V.detach())) ** 2
            w2 = torch.exp(2 * V.detach().clamp(0, cfg.w_cap))
            if cfg.w_exit > 0:
                fit = fit * (1.0 + cfg.w_exit * torch.isinf(nxt["M"][idx]).float())
            sv, _ = step_violation(model, bt, {k: v[idx] for k, v in stp.items()}, cfg.step_margin)
            loss = (w2 * fit).mean() + cfg.mu * (w2 * viol ** 2).mean() + cfg.mu_step * (w2 * sv ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(cfg.ema).add_(pm, alpha=1 - cfg.ema)
            acc += loss.item()
            nb += 1
        tgt.load_state_dict(ema.state_dict())
        rec = dict(epoch=ep, loss=acc / nb, time=time.time() - t0)
        hist.append(rec)
        if ep % 10 == 0 or ep == 1:
            with torch.no_grad():
                v = torch.cat([ema(c) for c in x[:200000].split(50000)])
            print(f"epoch {ep:4d}  loss {rec['loss']:.3e}  admissible samples with v<0.9 {100 * (v < 0.9).float().mean():5.1f}%  "
                  f"[{rec['time']:6.0f} s]", flush=True)
        if ep % cfg.snap_every == 0:
            torch.save(ema.state_dict(), out / "snapshots" / f"epoch_{ep:03d}.pt")
    (out / "history.json").write_text(json.dumps(hist))
    return out


def load(run, snap=None):
    d = json.loads((Path(run) / "config.json").read_text())
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    sys_ = S.get(cfg.system)
    prob = Problem(sys_, cfg)
    snaps = sorted((Path(run) / "snapshots").glob("epoch_*.pt"))
    sd = torch.load(snap or snaps[-1], map_location=S.DEV)
    prob.inner_c, prob.kappa = sd["inner_c"].item(), sd["kappa"].item()
    model = Net(prob, cfg, sd["q0"].item()).to(S.DEV)
    model.load_state_dict(sd)
    return prob, model, cfg


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--tau", type=float, default=Config.tau)
    ap.add_argument("--sub", type=int, default=Config.sub)
    ap.add_argument("--gamma-num", type=float, default=Config.gamma_num)
    ap.add_argument("--w-exit", type=float, default=Config.w_exit)
    a = ap.parse_args()
    train(Config(system=a.system, seed=a.seed, out=a.out, epochs=a.epochs, tau=a.tau, sub=a.sub, gamma_num=a.gamma_num, w_exit=a.w_exit))


if __name__ == "__main__":
    main()
