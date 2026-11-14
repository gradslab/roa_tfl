r"""
State-constrained Zubov problem (Grune & Zidani 2015) for the TFL closed loops, without any learning
(paper, Section 2 and Section 3, "Decomposition of the value function").

    V(x) = sup_{t >= 0} [ int_0^t g(y(s)) ds + h(y(t)) ],     v = 1 - exp(-V)          (GZ (3.4)-(3.5))
    D(M) = {V < inf} = {v < 1}                                                         (paper, Theorem 2)

Paper setting (target "z", the default):
  * Omega_ad: |det D| > eps_D, |det J_T| > eps_T, 1/||grad pi|| > eps_pi (the regularity constraints), and the open box;
  * target M = {z = 0}; for the unicycle z = (xi1, xi2, eta2 - 1), the quadrotor is in quad_system.py;
  * running cost g = alpha zeta^T Q zeta (+ kappa rho), zeta = z / z_scale; alpha is the paper's gamma, and 0 selects
    gamma_num / (90th percentile of zeta^T P zeta over admissible samples), gamma_num = 3 (short step) or 1 (long step);
  * h = sum_i relu(b_i / d_i - 1)^h_pow, d_i the signed distance to the i-th constraint, b_i = beta * median d_i
    (the paper uses beta = 0.1 and h_pow = 2);
  * q_exact = zeta^T P zeta is the exact cost-to-go because zdot = A z on Omega_ad, so V = alpha q_exact + Phi;
  * rho is a smooth step of alpha q_exact from c_inner / 2 to c_inner, c_inner the verified inner level.
The target "xi_eta" (dead zone in eta2, with eta_c and eta_c_hi) is an option and is not used in the paper.
"""
from __future__ import annotations

import math
import sys as _sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_sys.path.insert(0, str(Path(__file__).resolve().parent / "unicycle"))
from config import Config as TConfig, build_system


@dataclass
class ProblemConfig:
    path: str = "x^2 + y^2 - 1 = 0"
    pi: str = "atan2(y, x)"
    periodic_x1: bool = False
    box_x4: tuple = (-3.0, 1.0)
    eta_c: float = 0.4
    eta_c_hi: float = 1.8
    Q_diag: tuple = (1.742, 0.249, 3.281)
    alpha: float = 0.0
    gamma_num: float = 3.0
    beta: float = 0.1
    kappa: float = 0.0
    h_pow: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    system: str = "unicycle"
    target: str = "z"
    c_inner: float = 0.0


class ConstrainedZubov:
    def __init__(self, cfg: ProblemConfig):
        self.cfg = cfg
        from scipy.linalg import solve_continuous_lyapunov
        if cfg.system == "quadrotor":
            from quad_system import QuadTFL
            self.sys = sys = QuadTFL(device=cfg.device)
            self.tc = None
            Q = np.eye(sys.r) if len(cfg.Q_diag) < sys.r else np.diag(np.array(cfg.Q_diag[:sys.r], dtype=float))
        else:
            tc = TConfig(path=cfg.path, pi=cfg.pi, periodic_x1=cfg.periodic_x1, box_x4=tuple(cfg.box_x4),
                         target=cfg.target, eta_c=cfg.eta_c, eta_c_hi=cfg.eta_c_hi, init_Q_diag=tuple(cfg.Q_diag),
                         device=cfg.device)
            self.sys = sys = build_system(tc)
            self.tc = tc
            if cfg.target == "z":
                sys.linear_target = True
            Q = np.diag(np.array(cfg.Q_diag[:sys.r], dtype=float))
        self.Q = torch.as_tensor(Q, dtype=sys.dtype, device=sys.device)
        self.P = torch.as_tensor(solve_continuous_lyapunov(sys.A_tilde.T, -Q), dtype=sys.dtype, device=sys.device)
        self.linear_target = bool(getattr(sys, "linear_target", False))
        if not self.linear_target:
            self.P_xi = torch.as_tensor(solve_continuous_lyapunov(sys.A_tilde[:2, :2].T, -Q[:2, :2]), dtype=sys.dtype,
                                        device=sys.device)
        x = sys.sample_box(50000, seed=123)
        with torch.no_grad():
            d = self.distances(x)
            adm = torch.isfinite(d).all(-1) & (d > 0).all(-1)
            self.scale = torch.quantile(d[adm], 0.5, dim=0)
            self.alpha = cfg.alpha if cfg.alpha > 0 else getattr(cfg, "gamma_num", 3.0) / torch.quantile(self.q(x[adm]), 0.9).item()

    def zeta(self, x):
        return torch.nan_to_num(self.sys.z(x) / self.sys.z_scale, nan=1e3, posinf=1e3, neginf=-1e3)

    def q(self, x):
        """Quadratic of the exact transverse dynamics, q_dot <= -zeta^T Q zeta on Omega_ad."""
        z = self.zeta(x)
        return torch.einsum("ni,ij,nj->n", z, self.P, z)

    def q_exact(self, x):
        """Exact cost-to-go int_0^inf zeta^T Q zeta dt, valid on Omega_ad.  For target "z" (the paper) it is zeta^T P zeta.
        Target "xi_eta": xi block zeta_xi^T P_xi zeta_xi (A_xi^T P_xi + P_xi A_xi = -Q_xi), plus the dead-zone integral of
        Q_3 (w / s_3)^2 along eta2(t) = 1 + (eta2_0 - 1) e^{k3 t}: with b = 1 - eta_c (w > 0) or
        eta_c_hi - 1 (w < 0), a = b + |w|, u = |w| / b,
            int w^2 dt = b^2 / kappa * (u^2 / 2 - u + log(1 + u)),   kappa = -k3.
        So D q_exact . f = -zeta^T Q zeta exactly, and V = alpha q_exact + Phi with
        Phi = sup_t relu(h - alpha q_exact)(y(t)) (the constraint-induced part)."""
        s = self.sys
        if self.linear_target:
            return self.q(x)
        z = self.zeta(x)
        qx = torch.einsum("ni,ij,nj->n", z[:, :2], self.P_xi, z[:, :2])
        w = (z[:, 2] * s.z_scale[2]).double()
        b = torch.where(w > 0, torch.full_like(w, 1 - self.cfg.eta_c), torch.full_like(w, self.cfg.eta_c_hi - 1))
        u = w.abs() / b
        F = b ** 2 / (-s.k3) * (0.5 * u ** 2 - u + torch.log1p(u))
        return qx + (self.Q[2, 2] / s.z_scale[2] ** 2) * F.to(qx.dtype)

    def g(self, x, alpha=None):
        z = self.zeta(x)
        return (self.alpha if alpha is None else alpha) * torch.einsum("ni,ij,nj->n", z, self.Q, z)

    def distances(self, x):
        """Signed distances to the constraints, > 0 inside Omega_ad: |TFL regularity functions| minus their
        thresholds (circle: |det D| = 2|v|, |det DT| = 4|v|, 1/|grad pi| = r: 2|v| - 0.01, 4|v| - 0.01, r - 0.05), then
        the box faces."""
        s = self.sys
        if hasattr(s, "distances"):
            return s.distances(x)
        reg = torch.nan_to_num(s._reg(x), nan=0.0, posinf=1e6, neginf=-1e6)
        eD, eT, eG = s.eps
        d = [reg[:, 0].abs() - eD, reg[:, 1].abs() - eT, reg[:, 2] - eG]
        if not s.periodic_x1:
            d += [s.L - x[:, 0].abs()]
        d += [s.L - x[:, 1].abs(), x[:, 3] - s.x4_lo, s.x4_hi - x[:, 3]]
        return torch.stack(d, -1)

    def h(self, x, beta=None):
        """GZ (A4) penalty: locally Lipschitz on Omega_ad, +inf outside, -> inf at its boundary, 0 on M."""
        return self.h_from_d(self.distances(x), beta)

    def h_from_d(self, d, beta=None):
        beta = self.cfg.beta if beta is None else beta
        if beta <= 0:
            return torch.where((d > 0).all(-1), torch.zeros_like(d[:, 0]), torch.full_like(d[:, 0], math.inf))
        b = beta * self.scale
        phi = torch.where(d > 0, torch.relu(b / d.clamp_min(1e-30) - 1.0) ** self.cfg.h_pow, torch.full_like(d, math.inf))
        return phi.sum(-1)

    def obstacle(self, x, beta=None):
        return 1.0 - torch.exp(-self.h(x, beta))

    def rho(self, x):
        """Weight rho of the shaping cost: a smooth step of alpha q_ex from c_inner / 2 to c_inner."""
        if self.cfg.c_inner <= 0:
            raise RuntimeError("rho needs the verified inner level c_inner > 0")
        t = ((self.alpha * self.q_exact(x) - 0.5 * self.cfg.c_inner) / (0.5 * self.cfg.c_inner)).clamp(0.0, 1.0)
        return t * t * (3 - 2 * t)

    def admissible(self, x):
        return (self.distances(x) > 0).all(-1)
