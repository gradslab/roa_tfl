r"""
Planar quadrotor under a transverse-feedback-linearizing (TFL) path-following controller with thrust
dynamic extension (Akhtar, Waslander & Nielsen, CDC 2012, planar version; the thrust singularity is the one of
Al Lawati & Akhtar 2026, Assumption A1).  Built symbolically (SymPy) and compiled to PyTorch, with the same
interface as unicycle/tfl_system.TFLClosedLoop so that problem.py / zpinn.py can use either system.

Units: m = 1, g = 1, path radius 1 (time unit sqrt(r / g)).

    state  x = (p1, p2, v1, v2, theta, T)        theta on S^1, T the thrust (a controller state)
    pdot = v,   vdot = T (-sin theta, cos theta) - (0, 1),   thetadot = u2,   Tdot = u1
    outputs alpha(p) = p1^2 + p2^2 - 1 (path),  pi(p) = atan2(p2, p1) (tangential)
    vector relative degree {3, 3}:  xi = (alpha, Lf alpha, Lf^2 alpha),  eta = (pi, Lf pi, Lf^2 pi)
    u = D(x)^{-1} ( -[Lf^3 pi, Lf^3 alpha] + [v_par, v_perp] ),
    v_perp = k . xi (poles of (s+2)^3),  v_par = a1 (eta2 - eta_ref) + a2 eta3 (poles of (s+1)^2)
    det D = -2 T (and det DT = -8 T):  the controller is singular at zero thrust.

Linear coordinates z = (xi1, xi2, xi3, eta2 - eta_ref, eta3), zdot = A z exactly on the regularity domain;
target set M = {z = 0}: the circle traversed counter-clockwise at angular rate eta_ref.

Admissible set Omega_ad = {|det D| > eps_D, |det DT| > eps_T, r > eps_geom} inside the open box
    |p_i| < L = 2,  |v_i| < v_max = 2,  0 < T < T_max = 4   (theta periodic).
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
import torch


def _compile(exprs, syms):
    fn = sp.lambdify(syms, list(exprs), modules="torch", cse=True)

    def call(x: torch.Tensor) -> torch.Tensor:
        cols = fn(*x.unbind(-1))
        ref = x[..., 0]
        return torch.stack([c if torch.is_tensor(c) and c.shape == ref.shape
                            else torch.full_like(ref, float(c)) if not torch.is_tensor(c)
                            else c.expand_as(ref) for c in cols], dim=-1)

    return call


class QuadTFL:
    LABELS = ("converged", "left TFL regularity domain", "left box", "inconclusive")
    n = 6
    periodic_x1 = False
    linear_target = True
    sing_idx = (0, 2)
    k3 = -1.0

    def __init__(self, eta_ref=0.5, xi_poles=(-2.0, -2.0, -2.0), eta_poles=(-1.0, -1.0), box_p=2.0, v_max=2.0,
                 T_max=4.0, eps_D=1e-2, eps_T=1e-2, eps_geom=5e-2, normalize_field=True, device="cpu",
                 dtype=torch.float32, z_quantile=0.5, z_sat=10.0):
        self.device, self.dtype = torch.device(device), dtype
        self.eta_ref, self.L, self.v_max, self.T_max = eta_ref, box_p, v_max, T_max
        self.eta2_ref = eta_ref
        self.eps = (eps_D, eps_T, eps_geom)
        self.normalize_field = normalize_field
        self.z_quantile, self.z_sat = z_quantile, z_sat
        self.path_str, self.pi_str = "p1^2 + p2^2 - 1 = 0", "atan2(p2, p1)"
        cx = np.poly(xi_poles)[1:]
        self.kx = (-cx[2], -cx[1], -cx[0])
        ce = np.poly(eta_poles)[1:]
        self.ke = (-ce[1], -ce[0])

        X = sp.symbols("p1 p2 v1 v2 th T", real=True)
        p1, p2, v1, v2, th, T = X
        f = [v1, v2, -T * sp.sin(th), T * sp.cos(th) - 1, 0, 0]
        G = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 0]]
        alpha = p1 ** 2 + p2 ** 2 - 1
        phase = sp.atan2(p2, p1)
        lie = lambda h, v: sum(sp.diff(h, s) * vi for s, vi in zip(X, v))
        La, Lp = lie(alpha, f), lie(phase, f)
        L2a, L2p = lie(La, f), lie(Lp, f)
        L3a, L3p = lie(L2a, f), lie(L2p, f)
        col = lambda j: [G[i][j] for i in range(6)]
        for h, name in ((alpha, "alpha"), (La, "Lf alpha"), (phase, "pi"), (Lp, "Lf pi")):
            if any(sp.simplify(lie(h, col(j))) != 0 for j in range(2)):
                raise NotImplementedError(f"L_g {name} != 0: relative degree is not 3")
        D = [[lie(L2p, col(0)), lie(L2p, col(1))], [lie(L2a, col(0)), lie(L2a, col(1))]]
        Tmap = [phase, Lp, L2p, alpha, La, L2a]
        DT = sp.Matrix(6, 6, [sp.diff(t, s) for t in Tmap for s in X])
        gp = [sp.diff(phase, p1), sp.diff(phase, p2)]
        detD_s = sp.simplify(D[0][0] * D[1][1] - D[0][1] * D[1][0])
        detDT_s = sp.factor(sp.simplify(DT.det(method="berkowitz")))
        self.detD_str = str(detD_s)
        self._reg = _compile([detD_s, detDT_s, 1 / sp.sqrt(gp[0] ** 2 + gp[1] ** 2)], X)
        self.z_exprs = [alpha, La, L2a, Lp - eta_ref, L2p]
        self._z = _compile(self.z_exprs, X)
        self._all = _compile([alpha, La, L2a, phase, Lp, L2p, L3a, L3p, D[0][0], D[0][1], D[1][0], D[1][1], *f], X)
        r1s = -L3p + self.ke[0] * (Lp - eta_ref) + self.ke[1] * L2p
        r2s = -L3a + self.kx[0] * alpha + self.kx[1] * La + self.kx[2] * L2a
        u1s = (D[1][1] * r1s - D[0][1] * r2s) / detD_s
        u2s = (-D[1][0] * r1s + D[0][0] * r2s) / detD_s
        fcl_s = [f[0], f[1], f[2], f[3], u2s, u1s]
        self.sym = dict(X=X, f=f, alpha=alpha, phase=phase, z=self.z_exprs, reg=[detD_s, detDT_s,
                        1 / sp.sqrt(gp[0] ** 2 + gp[1] ** 2)], D=D, L3=(L3p, L3a), fcl=fcl_s)
        self.r = 5
        Axi = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [self.kx[0], self.kx[1], self.kx[2]]])
        Aeta = np.array([[0.0, 1.0], [self.ke[0], self.ke[1]]])
        self.A_z = np.block([[Axi, np.zeros((3, 2))], [np.zeros((2, 3)), Aeta]])
        if np.max(np.linalg.eigvals(self.A_z).real) >= 0:
            raise ValueError("A_z is not Hurwitz")
        self.cT, self.hT = 0.5 * T_max, 0.5 * T_max
        self.half = torch.tensor([box_p, box_p, v_max, v_max, math.pi, self.hT], device=self.device, dtype=dtype)
        self.s = 1.0 / self.half
        self.n_features = 7
        self.z_scale = torch.ones(self.r, device=self.device, dtype=dtype)
        self._calibrate_z_scale()

    def tensor(self, a):
        return torch.as_tensor(a, device=self.device, dtype=self.dtype)

    def features(self, x):
        return torch.stack([x[:, 0] / self.L, x[:, 1] / self.L, x[:, 2] / self.v_max, x[:, 3] / self.v_max,
                            torch.cos(x[:, 4]), torch.sin(x[:, 4]), (x[:, 5] - self.cT) / self.hT], dim=-1)

    def z(self, x):
        return self._z(x)

    def z_tilde(self, x):
        zt = torch.nan_to_num(self._z(x) / self.z_scale, nan=1e3, posinf=1e3, neginf=-1e3)
        return self.z_sat * torch.tanh(zt / self.z_sat)

    @property
    def A_tilde(self) -> np.ndarray:
        S = np.diag(1.0 / self.z_scale.cpu().numpy().astype(float))
        return S @ self.A_z @ np.linalg.inv(S)

    def evaluate(self, x):
        v = self._all(x)
        alpha, La, L2a, phase, Lp, L2p, L3a, L3p, D11, D12, D21, D22 = v[:, :12].unbind(-1)
        f = v[:, 12:18]
        r1 = -L3p + self.ke[0] * (Lp - self.eta_ref) + self.ke[1] * L2p
        r2 = -L3a + self.kx[0] * alpha + self.kx[1] * La + self.kx[2] * L2a
        detD = D11 * D22 - D12 * D21
        u1 = (D22 * r1 - D12 * r2) / detD
        u2 = (-D21 * r1 + D11 * r2) / detD
        fcl = f.clone()
        fcl[:, 5] = fcl[:, 5] + u1
        fcl[:, 4] = fcl[:, 4] + u2
        reg = self._reg(x)
        eD, eT, eG = self.eps
        margin = torch.minimum(torch.minimum(reg[:, 0].abs() / eD, reg[:, 1].abs() / eT), reg[:, 2] / eG) - 1.0
        margin = torch.nan_to_num(margin, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return dict(f_cl=fcl, margin=margin, detD=detD, detDT=reg[:, 1], geom=reg[:, 2],
                    xi=torch.stack([alpha, La, L2a], -1), eta=torch.stack([phase, Lp, L2p], -1))

    def distances(self, x):
        """Signed distances to the constraints (> 0 inside Omega_ad): regularity functions minus their
        thresholds, then the box faces."""
        reg = torch.nan_to_num(self._reg(x), nan=0.0, posinf=1e6, neginf=-1e6)
        eD, eT, eG = self.eps
        d = [reg[:, 0].abs() - eD, reg[:, 1].abs() - eT, reg[:, 2] - eG,
             self.L - x[:, 0].abs(), self.L - x[:, 1].abs(), self.v_max - x[:, 2].abs(), self.v_max - x[:, 3].abs(),
             x[:, 5], self.T_max - x[:, 5]]
        return torch.stack(d, -1)

    CONSTRAINTS = ("|det D| (thrust)", "|det DT|", "r (centre)", "p1 face", "p2 face", "v1 face", "v2 face",
                   "T = 0 face", "T max face")

    def field(self, x):
        e = self.evaluate(x)
        f = torch.nan_to_num(e["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize_field:
            f = f / torch.sqrt(1.0 + ((f * self.s) ** 2).sum(-1, keepdim=True))
        return f, e["margin"]

    def in_box(self, x):
        return (x[:, 0].abs() <= self.L) & (x[:, 1].abs() <= self.L) & (x[:, 2].abs() <= self.v_max) & \
               (x[:, 3].abs() <= self.v_max) & (x[:, 5] >= 0.0) & (x[:, 5] <= self.T_max)

    def _box_map(self, u):
        lo = self.tensor([-self.L, -self.L, -self.v_max, -self.v_max, -math.pi, 0.0])
        hi = self.tensor([self.L, self.L, self.v_max, self.v_max, math.pi, self.T_max])
        return lo + (hi - lo) * u

    def sample_box(self, n, seed=0):
        u = torch.quasirandom.SobolEngine(6, scramble=True, seed=int(seed)).draw(n)
        return self._box_map(u.to(self.device, self.dtype))

    def sample_edge(self, n, seed=0, width=0.1):
        """Samples concentrated near the faces of the box (every non-periodic coordinate, with probability
        1/2, in the slab of relative width `width` next to a random face)."""
        u = torch.quasirandom.SobolEngine(6, scramble=True, seed=int(seed)).draw(n)
        g = torch.Generator().manual_seed(int(seed) + 3)
        for k in (0, 1, 2, 3, 5):
            near = torch.rand(n, generator=g) < 0.5
            hi = torch.rand(n, generator=g) < 0.5
            w = width * u[:, k]
            u[:, k] = torch.where(near, torch.where(hi, 1 - w, w), u[:, k])
        return self._box_map(u.to(self.device, self.dtype))

    def sample_box_boundary(self, n, seed=0):
        g = torch.Generator().manual_seed(int(seed))
        faces = [(0, self.L, 1), (0, -self.L, -1), (1, self.L, 1), (1, -self.L, -1), (2, self.v_max, 1),
                 (2, -self.v_max, -1), (3, self.v_max, 1), (3, -self.v_max, -1), (5, self.T_max, 1), (5, 0.0, -1)]
        widths = torch.tensor([2 * self.L, 2 * self.L, 2 * self.v_max, 2 * self.v_max, 2 * math.pi, self.T_max])
        vol = widths.prod()
        w = torch.tensor([float(vol / widths[d]) for d, _, _ in faces])
        face = torch.multinomial(w / w.sum(), n, replacement=True, generator=g)
        x = self._box_map(torch.rand(n, 6, generator=g).to(self.device, self.dtype))
        nrm = torch.zeros(n, 6, device=self.device, dtype=self.dtype)
        face = face.to(self.device)
        for k, (dim, val, sgn) in enumerate(faces):
            m = face == k
            x[m, dim] = val
            nrm[m, dim] = sgn
        return x, nrm

    def sample_band(self, n, mode="singular", delta=3.0, tau=0.1, seed=0, steps=4, eps=None):
        """Samples next to the singular sets (thrust |det D| -> eps_D, centre r -> eps_geom): Newton steps on
        one normalized regularity margin h_i = reg_i / eps_i - 1 towards a uniform level in (0, delta)."""
        eD, eT, eG = self.eps if eps is None else eps
        x = self.sample_box(n, seed=seed)
        g = torch.Generator(device="cpu").manual_seed(int(seed) + 7)
        target = (delta * torch.rand(n, generator=g)).to(self.device, self.dtype)
        out = []
        for i, idx in enumerate(torch.arange(n, device=self.device).chunk(2)):
            xi, ti = x[idx], target[idx]
            which = (0, eD) if i == 0 else (2, eG)
            for _ in range(steps):
                xi = xi.detach().requires_grad_(True)
                reg = self._reg(xi)
                h = (reg[:, which[0]].abs() if which[0] == 0 else reg[:, which[0]]) / which[1] - 1
                (gh,) = torch.autograd.grad(h.sum(), xi)
                step = ((ti - h) / (gh ** 2).sum(-1).clamp(min=1e-12))[:, None] * gh
                xi = (xi + step).detach()
            out.append(xi)
        x = torch.cat(out)
        x[:, 4] = torch.remainder(x[:, 4] + math.pi, 2 * math.pi) - math.pi
        with torch.no_grad():
            d = self.distances(x)
            hmin = torch.minimum(d[:, 0] / eD, d[:, 2] / eG)
            keep = (d > 0).all(-1) & (hmin < delta) & self.in_box(x) & torch.isfinite(d).all(-1)
        return x[keep]

    def _calibrate_z_scale(self, n=20000):
        x = self.sample_box(n, seed=123)
        e = self.evaluate(x)
        z = self._z(x)[e["margin"] > 0]
        z = z[torch.isfinite(z).all(-1)]
        self.z_scale = torch.quantile(z.abs(), self.z_quantile, dim=0).clamp_min(1e-3)

    @torch.no_grad()
    def simulate(self, x0, T=40.0, dt=0.01, tol=1e-3, hold=1.0, return_final=False, eta_tol=None):
        """Batched RK4 of the physical closed loop.  Labels: 0 converged (||z|| / z_scale < tol for `hold` time
        units), 1 left the regularity domain, 2 left the box, 3 inconclusive."""
        x = x0.clone()
        N = len(x)
        label = torch.full((N,), 3, dtype=torch.long, device=self.device)
        held = torch.zeros(N, device=self.device, dtype=self.dtype)
        Tprev = x[:, 5].clone()
        act = torch.arange(N, device=self.device)
        rhs = lambda y: self.evaluate(y)["f_cl"]
        for _ in range(int(round(T / dt)) + 1):
            xa = x[act]
            e = self.evaluate(xa)
            bad = (e["margin"] <= 0) | ~torch.isfinite(xa).all(-1) | ~torch.isfinite(e["f_cl"]).all(-1)
            bad = bad | (Tprev[act] * xa[:, 5] <= 0)
            Tprev[act] = xa[:, 5]
            out = ~self.in_box(xa)
            near = (self._z(xa) / self.z_scale).norm(dim=-1) < tol
            held[act] = torch.where(near, held[act] + dt, torch.zeros_like(held[act]))
            conv = held[act] >= hold - 1e-9
            lab = torch.full_like(act, -1)
            lab[conv] = 0
            lab[out] = 2
            lab[bad] = 1
            done = lab >= 0
            label[act[done]] = lab[done]
            act, xa, k1 = act[~done], xa[~done], e["f_cl"][~done]
            if len(act) == 0:
                break
            k2 = rhs(xa + 0.5 * dt * k1)
            k3 = rhs(xa + 0.5 * dt * k2)
            k4 = rhs(xa + dt * k3)
            x[act] = xa + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (label, x) if return_final else label

    def target_points(self, n, seed=0, iters=30, tol=1e-6):
        """Points of M = {z = 0} by Gauss-Newton from box samples (no simulation)."""
        x = self.sample_box(4 * n, seed=seed + 7)
        for _ in range(iters):
            x = x.detach().requires_grad_(True)
            z = self._z(x)
            J = torch.stack([torch.autograd.grad(z[:, i].sum(), x, retain_graph=i < self.r - 1)[0]
                             for i in range(self.r)], 1)
            step = (torch.linalg.pinv(torch.nan_to_num(J)) @ torch.nan_to_num(z).unsqueeze(-1)).squeeze(-1)
            x = (x - step.clamp(-0.5, 0.5)).detach()
            x[:, 4] = torch.remainder(x[:, 4] + math.pi, 2 * math.pi) - math.pi
        with torch.no_grad():
            ok = (self._z(x).norm(dim=-1) < tol) & self.in_box(x) & (self.evaluate(x)["margin"] > 0)
        pts = x[ok]
        if len(pts) == 0:
            raise RuntimeError("Gauss-Newton found no admissible point on z = 0")
        idx = torch.randint(len(pts), (n,), generator=torch.Generator().manual_seed(seed))
        return pts[idx.to(self.device)]
