r"""
TFL closed loop of the dynamically extended unicycle, built symbolically
(SymPy) and compiled to PyTorch so it runs batched on the GPU.

    x_dot = f(x) + g1(x) u1 + g2(x) u2,      x = (x1, x2, x3, x4)
    u     = D(x)^{-1} ( -[Lf^2 pi, Lf^2 alpha] + [v_par, v_perp] )
    v_perp = k1 xi1 + k2 xi2,   v_par = k3 (eta2 - eta2_ref)

Transverse coordinates (the target set is {z = 0}):

    target "z" : z = (xi1, xi2, eta2 - eta2_ref),  z_dot = A_z z,  A_z = blkdiag(A_perp, k3)
    target "xi": z = (xi1, xi2),                   z_dot = A_perp z

Admissible (TFL regularity) domain:

    D_adm = { |det D| > eps_D, |det DT| > eps_T, 1/||grad pi|| > eps_geom } inside the box
            |x1|, |x2| <= L,  x4 in [x4_lo, x4_hi],  x3 periodic.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
import torch

X_SYM, Y_SYM = sp.symbols("x y", real=True)


def parse_planar(s: str) -> sp.Expr:
    local = {"x": X_SYM, "y": Y_SYM, "atan2": sp.atan2, "pi": sp.pi}
    s = s.replace("^", "**")
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        return sp.sympify(lhs, locals=local) - sp.sympify(rhs, locals=local)
    return sp.sympify(s, locals=local)


def _compile(exprs, syms):
    """Lambdify a list of expressions to torch; constants are broadcast."""
    fn = sp.lambdify(syms, list(exprs), modules="torch")

    def call(x: torch.Tensor) -> torch.Tensor:
        cols = fn(*x.unbind(-1))
        ref = x[..., 0]
        return torch.stack([c if torch.is_tensor(c) and c.shape == ref.shape
                            else torch.full_like(ref, float(c)) if not torch.is_tensor(c)
                            else c.expand_as(ref) for c in cols], dim=-1)

    return call


class TFLClosedLoop:
    LABELS = ("converged", "left TFL regularity domain", "left box", "inconclusive")

    def __init__(self, path="x^2 + y^2 - 1 = 0", pi="atan2(y, x)", vbar=1.0,
                 k1=-1.0, k2=-2.0, k3=-1.0, eta2_ref=1.0, box_xy=2.0, box_x4=(-1.5, 1.5),
                 eps_D=1e-2, eps_T=1e-2, eps_geom=5e-2, target="z", normalize_field=True,
                 device="cpu", dtype=torch.float32, z_quantile=0.5, z_sat=10.0, periodic_x1=False, eta_c=0.5, eta_c_hi=float("inf")):
        if target not in ("z", "xi", "xi_eta"):
            raise ValueError("target must be 'z', 'xi' or 'xi_eta'")
        self.eta_c = eta_c
        self.device, self.dtype = torch.device(device), dtype
        self.k1, self.k2, self.k3, self.eta2_ref, self.vbar = k1, k2, k3, eta2_ref, vbar
        self.L, self.x4_lo, self.x4_hi = box_xy, *box_x4
        self.periodic_x1 = periodic_x1
        self.Lx = math.pi if periodic_x1 else box_xy
        self.eps = (eps_D, eps_T, eps_geom)
        self.target, self.normalize_field = target, normalize_field
        self.path_str, self.pi_str = path, pi
        self.z_quantile, self.z_sat = z_quantile, z_sat

        X = sp.symbols("x1 x2 x3 x4", real=True)
        x1, x2, x3, x4 = X
        f = [(vbar + x4) * sp.cos(x3), (vbar + x4) * sp.sin(x3), 0, 0]
        G = [[0, 0], [0, 0], [0, 1], [1, 0]]
        to_x = {X_SYM: x1, Y_SYM: x2}
        alpha = parse_planar(path).subs(to_x)
        phase = parse_planar(pi).subs(to_x)
        lie = lambda h, v: sum(sp.diff(h, s) * vi for s, vi in zip(X, v))
        La, Lp = lie(alpha, f), lie(phase, f)
        L2a, L2p = lie(La, f), lie(Lp, f)
        col = lambda j: [G[i][j] for i in range(4)]
        D = [[lie(Lp, col(0)), lie(Lp, col(1))], [lie(La, col(0)), lie(La, col(1))]]
        T = [phase, Lp, alpha, La]
        DT = [sp.diff(t, s) for t in T for s in X]
        gp = [sp.diff(phase, x1), sp.diff(phase, x2)]
        for h, name in ((alpha, "alpha"), (phase, "pi")):
            if any(sp.simplify(lie(h, col(j))) != 0 for j in range(2)):
                raise NotImplementedError(f"L_g {name} != 0: relative degree is not 2")

        detD_s = D[0][0] * D[1][1] - D[0][1] * D[1][0]
        detDT_s = sp.factor(sp.simplify(sp.Matrix(4, 4, DT).det(method="berkowitz")))
        self._reg = _compile([detD_s, detDT_s, 1 / sp.sqrt(gp[0] ** 2 + gp[1] ** 2)], X)

        z_exprs = [alpha, La] + ([Lp - eta2_ref] if target == "z" else [Lp] if target == "xi_eta" else [])
        self._z = _compile(z_exprs, X)
        if target == "xi_eta":
            raw = self._z
            self._z = lambda x: (lambda r: torch.cat([r[:, :2], torch.relu(eta_c - r[:, 2:3])
                                                      - torch.relu(r[:, 2:3] - eta_c_hi)], -1))(raw(x))
        self._all = _compile([alpha, La, phase, Lp, L2a, L2p, D[0][0], D[0][1], D[1][0], D[1][1],
                              *gp, *f, *DT], X)
        self.r = len(z_exprs)
        Aperp = np.array([[0.0, 1.0], [k1, k2]])
        self.A_z = Aperp if target == "xi" else np.block([[Aperp, np.zeros((2, 1))],
                                                         [np.zeros((1, 2)), np.array([[k3]])]])
        if np.max(np.linalg.eigvals(self.A_z).real) >= 0:
            raise ValueError("A_z is not Hurwitz: check k1, k2, k3")

        self.c4, self.h4 = 0.5 * (self.x4_lo + self.x4_hi), 0.5 * (self.x4_hi - self.x4_lo)
        self.s = torch.tensor([1 / self.Lx, 1 / self.L, 1 / math.pi, 1 / self.h4],
                              device=self.device, dtype=dtype)
        self.n_features = 6 if periodic_x1 else 5
        self.z_scale = torch.ones(self.r, device=self.device, dtype=dtype)
        self._calibrate_z_scale()

    def tensor(self, a):
        return torch.as_tensor(a, device=self.device, dtype=self.dtype)

    def features(self, x):
        """Network input: normalized position, heading on S^1, normalized speed offset."""
        if self.periodic_x1:
            return torch.stack([torch.cos(x[:, 0]), torch.sin(x[:, 0]), x[:, 1] / self.L, torch.cos(x[:, 2]),
                                torch.sin(x[:, 2]), (x[:, 3] - self.c4) / self.h4], dim=-1)
        return torch.stack([x[:, 0] / self.L, x[:, 1] / self.L, torch.cos(x[:, 2]),
                            torch.sin(x[:, 2]), (x[:, 3] - self.c4) / self.h4], dim=-1)

    def z(self, x):
        return self._z(x)

    def z_tilde(self, x):
        """Normalized transverse coordinates, softly saturated at +-z_sat:
        z~ = z_sat tanh(z / (scale z_sat)).  Identity to first order near the
        target (so V = 0 <=> z = 0 and the local quadratic part are unchanged),
        but bounded near the circle centre where eta2 ~ 1/r."""
        zt = torch.nan_to_num(self._z(x) / self.z_scale, nan=1e3, posinf=1e3, neginf=-1e3)
        return self.z_sat * torch.tanh(zt / self.z_sat)

    @property
    def A_tilde(self) -> np.ndarray:
        """Linear transverse dynamics in normalized coordinates z~ = S z."""
        S = np.diag(1.0 / self.z_scale.cpu().numpy().astype(float))
        return S @ self.A_z @ np.linalg.inv(S)

    def evaluate(self, x):
        """Closed-loop vector field, regularity margin and diagnostics."""
        v = self._all(x)
        alpha, La, phase, Lp, L2a, L2p, D11, D12, D21, D22 = v[:, :10].unbind(-1)
        gp, f, DT = v[:, 10:12], v[:, 12:16], v[:, 16:32].reshape(-1, 4, 4)
        r1 = -L2p + self.k3 * (Lp - self.eta2_ref)
        r2 = -L2a + self.k1 * alpha + self.k2 * La
        detD = D11 * D22 - D12 * D21
        u1 = (D22 * r1 - D12 * r2) / detD
        u2 = (-D21 * r1 + D11 * r2) / detD
        fcl = f.clone()
        fcl[:, 3] = fcl[:, 3] + u1
        fcl[:, 2] = fcl[:, 2] + u2
        detDT = torch.linalg.det(torch.nan_to_num(DT))
        geom = 1.0 / gp.norm(dim=-1)
        eD, eT, eG = self.eps
        margin = torch.minimum(torch.minimum(detD.abs() / eD, detDT.abs() / eT), geom / eG) - 1.0
        margin = torch.nan_to_num(margin, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return dict(f_cl=fcl, margin=margin, detD=detD, detDT=detDT, geom=geom,
                    xi=torch.stack([alpha, La], -1), eta=torch.stack([phase, Lp], -1))

    def constraint_h(self, x, mode="singular", tau=0.1, eps_box=0.05, which=None, eps=None):
        """Smooth constraint function for the barrier: h > 0 admissible,
        h <= 0 on the constrained set.  h_i are the regularity margins in
        units of their eps (h_i = 0 exactly where margin = 0) and, for mode
        "all", the distances to the box faces in units of eps_box.  The
        soft-min  -tau log sum exp(-h_i / tau)  is <= min_i h_i, so it never
        declares a constrained point admissible.  eps overrides the regularity
        thresholds (for the circle |det D| = 2|v|, |det DT| = 4|v|, 1/||grad pi|| = r,
        so eps = (2 v_min, 4 v_min, r_min) excludes |v| <= v_min and r <= r_min)."""
        reg = self._reg(x)
        eD, eT, eG = self.eps if eps is None else eps
        hs = [reg[:, 0].abs() / eD - 1, reg[:, 1].abs() / eT - 1, reg[:, 2] / eG - 1]
        if mode == "all":
            hs += ([] if self.periodic_x1 else [(self.L - x[:, 0]) / eps_box, (self.L + x[:, 0]) / eps_box]) +                   [(self.L - x[:, 1]) / eps_box, (self.L + x[:, 1]) / eps_box,
                   (x[:, 3] - self.x4_lo) / eps_box, (self.x4_hi - x[:, 3]) / eps_box]
        h = torch.stack(hs, -1)
        h = torch.nan_to_num(h, nan=-1.0, posinf=1e3, neginf=-1.0).clamp(max=1e3)
        if which is not None:
            return h[:, which]
        return -tau * torch.logsumexp(-h / tau, dim=-1)

    def sample_band(self, n, mode="singular", delta=1.0, tau=0.1, seed=0, steps=4, eps=None):
        """Samples inside the barrier band 0 < h < delta, where the barrier
        acts on the decrease condition.  The band is thin (about 0.25 % of the
        box for delta = 1), so uniform samples barely hit it: start from box
        samples and take Newton steps on one constraint function h_i (equal
        share per constraint) towards a uniform level in the band."""
        k = 3 if mode == "singular" else 9
        x = self.sample_box(n, seed=seed)
        g = torch.Generator(device="cpu").manual_seed(int(seed) + 7)
        target = (delta * torch.rand(n, generator=g)).to(self.device, self.dtype)
        out = []
        for i, idx in enumerate(torch.arange(n, device=self.device).chunk(k)):
            xi, ti = x[idx], target[idx]
            for _ in range(steps):
                xi = xi.detach().requires_grad_(True)
                h = self.constraint_h(xi, mode, tau, which=i, eps=eps)
                (gh,) = torch.autograd.grad(h.sum(), xi)
                step = ((ti - h) / (gh ** 2).sum(-1).clamp(min=1e-12))[:, None] * gh
                xi = (xi + step).detach()
            out.append(xi)
        x = torch.cat(out)
        h = self.constraint_h(x, mode, tau, eps=eps)
        keep = (h > 0) & (h < delta) & self.in_box(x) & torch.isfinite(h)
        return x[keep]

    def field(self, x):
        """Closed-loop field used in the Lyapunov conditions.  With
        normalize_field the field is divided by sqrt(1 + ||S f_cl||^2): a
        positive time rescaling, so orbits and the ROA are unchanged, but the
        field stays bounded near the TFL singularities where u ~ 1/det D."""
        e = self.evaluate(x)
        f = torch.nan_to_num(e["f_cl"], nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize_field:
            f = f / torch.sqrt(1.0 + ((f * self.s) ** 2).sum(-1, keepdim=True))
        return f, e["margin"]

    def in_box(self, x):
        return ((x[:, 0].abs() <= self.L) | self.periodic_x1) & (x[:, 1].abs() <= self.L) & \
               (x[:, 3] >= self.x4_lo) & (x[:, 3] <= self.x4_hi)

    def _box_map(self, u):
        lo = self.tensor([-self.Lx, -self.L, -math.pi, self.x4_lo])
        hi = self.tensor([self.Lx, self.L, math.pi, self.x4_hi])
        return lo + (hi - lo) * u

    def sample_box(self, n, seed=0):
        """Scrambled Sobol (quasi-Monte Carlo) samples in the box."""
        u = torch.quasirandom.SobolEngine(4, scramble=True, seed=int(seed)).draw(n)
        return self._box_map(u.to(self.device, self.dtype))

    def sample_edge(self, n, seed=0, width=0.1):
        """Samples concentrated near the faces, edges and corners of the box:
        each of x1, x2, x4 is, with probability 1/2, drawn from the slab of
        relative width `width` next to a random face (so 1/8 of the points are
        near a corner).  Uniform samples rarely land there."""
        u = torch.quasirandom.SobolEngine(4, scramble=True, seed=int(seed)).draw(n)
        g = torch.Generator().manual_seed(int(seed) + 3)
        for k in ((1, 3) if self.periodic_x1 else (0, 1, 3)):
            near = torch.rand(n, generator=g) < 0.5
            hi = torch.rand(n, generator=g) < 0.5
            w = width * u[:, k]
            u[:, k] = torch.where(near, torch.where(hi, 1 - w, w), u[:, k])
        return self._box_map(u.to(self.device, self.dtype))

    def sample_box_boundary(self, n, seed=0):
        """Uniform samples on the faces x1 = +-L, x2 = +-L, x4 = lo/hi with outward normals."""
        g = torch.Generator().manual_seed(int(seed))
        a_x, a_y, a_top = 2 * self.L * 2 * self.h4, 2 * self.Lx * 2 * self.h4, 2 * self.Lx * 2 * self.L
        w = torch.tensor([0.0 if self.periodic_x1 else a_x] * 2 + [a_y] * 2 + [a_top] * 2)
        face = torch.multinomial(w / w.sum(), n, replacement=True, generator=g)
        x = self._box_map(torch.rand(n, 4, generator=g).to(self.device, self.dtype))
        nrm = torch.zeros(n, 4, device=self.device, dtype=self.dtype)
        face = face.to(self.device)
        for k, (dim, val, sgn) in enumerate([(0, self.L, 1), (0, -self.L, -1), (1, self.L, 1),
                                             (1, -self.L, -1), (3, self.x4_hi, 1), (3, self.x4_lo, -1)]):
            m = face == k
            x[m, dim] = val
            nrm[m, dim] = sgn
        return x, nrm

    def _calibrate_z_scale(self, n=20000):
        """Median |z| over admissible samples (robust to the 1/r growth of eta2)."""
        x = self.sample_box(n, seed=123)
        e = self.evaluate(x)
        z = self._z(x)[e["margin"] > 0]
        z = z[torch.isfinite(z).all(-1)]
        self.z_scale = torch.quantile(z.abs(), self.z_quantile, dim=0).clamp_min(1e-3)
        if self.target == "xi_eta":
            w = z[:, 2][z[:, 2] != 0].abs()
            self.z_scale[2] = torch.quantile(w, self.z_quantile).clamp_min(1e-3)

    @torch.no_grad()
    def simulate(self, x0, T=30.0, dt=0.01, tol=1e-3, hold=1.0, return_final=False, eta_tol=0.05):
        """Batched RK4 ground truth.  Labels: 0 converged (||xi|| < tol and |eta2 - eta2_ref| < eta_tol
        for `hold` seconds), 1 left the TFL regularity domain, 2 left the box, 3 inconclusive.  Uses the
        physical (unnormalized) closed loop.  The eta2 condition matters: on the path, a robot moving the
        wrong way (eta2 < 0) has xi = 0 but must still pass eta2 = 0, i.e. v = 0 (singular), because the
        speed loop drives eta2 -> eta2_ref monotonically.  eta_tol = inf drops the eta2 condition."""
        x = x0.clone()
        N = len(x)
        label = torch.full((N,), 3, dtype=torch.long, device=self.device)
        held = torch.zeros(N, device=self.device, dtype=self.dtype)
        act = torch.arange(N, device=self.device)
        rhs = lambda y: self.evaluate(y)["f_cl"]
        for _ in range(int(round(T / dt)) + 1):
            xa = x[act]
            e = self.evaluate(xa)
            bad = (e["margin"] <= 0) | ~torch.isfinite(xa).all(-1) | ~torch.isfinite(e["f_cl"]).all(-1)
            out = ~self.in_box(xa)
            near = (e["xi"].norm(dim=-1) < tol) & ((e["eta"][:, 1] - self.eta2_ref).abs() < eta_tol)
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
        """Points on the attracting set {z = 0}, without simulation: Gauss-Newton
        x <- x - J^+ z(x) from box samples (minimum-norm steps onto z = 0),
        keeping converged points that are admissible and inside the box."""
        x = self.sample_box(4 * n, seed=seed + 7)
        for _ in range(iters):
            x = x.detach().requires_grad_(True)
            z = self._z(x)
            J = torch.stack([torch.autograd.grad(z[:, i].sum(), x, retain_graph=i < self.r - 1)[0]
                             for i in range(self.r)], 1)
            step = (torch.linalg.pinv(torch.nan_to_num(J)) @ torch.nan_to_num(z).unsqueeze(-1)).squeeze(-1)
            x = (x - step.clamp(-0.5, 0.5)).detach()
            x[:, 2] = torch.remainder(x[:, 2] + math.pi, 2 * math.pi) - math.pi
            if self.periodic_x1:
                x[:, 0] = torch.remainder(x[:, 0] + math.pi, 2 * math.pi) - math.pi
        with torch.no_grad():
            ok = (self._z(x).norm(dim=-1) < tol) & self.in_box(x) & (self.evaluate(x)["margin"] > 0)
        pts = x[ok]
        if len(pts) == 0:
            raise RuntimeError("Gauss-Newton found no admissible point on z = 0")
        idx = torch.randint(len(pts), (n,), generator=torch.Generator().manual_seed(seed))
        return pts[idx.to(self.device)]

    def target_points_sim(self, n, seed=0):
        """Target points from simulation: final states of converged runs."""
        x0 = self.sample_box(4 * n, seed=seed + 7)
        label, xf = self.simulate(x0, T=25.0, return_final=True)
        pts = xf[label == 0]
        if len(pts) == 0:
            raise RuntimeError("no trajectory converged; check gains / box")
        pts[:, 2] = torch.remainder(pts[:, 2] + math.pi, 2 * math.pi) - math.pi
        idx = torch.randint(len(pts), (n,), generator=torch.Generator().manual_seed(seed))
        return pts[idx.to(self.device)]
