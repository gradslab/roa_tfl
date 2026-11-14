r"""
Benchmark systems for the general state-constrained Zubov method.  A system is pure problem data:

  f(x)         closed-loop vector field (batched torch), equilibrium at the origin
  box          lo, hi: the computational domain (leaving it counts as leaving the admissible set)
  constraints  extra state constraints d_i(x) > 0 (admissible where all are positive), with names

Everything else (normalization, sampling, distances, the simulated ground truth) is generic.

Systems
  vdp       reversed Van der Pol, mu = 1, on [-2.5, 2.5] x [-3.5, 3.5]          (Liu et al. 2025, Example 6)
  vdp3      reversed Van der Pol, mu = 3, on [-3, 3] x [-6, 6]                  (Liu et al. 2025, Example 6)
  pendulum  inverted pendulum with the linear gains of Liu et al. (2025, Example 5), torque saturated
            at |u| <= 0.8, on [-pi, pi] x [-4, 4]  (gravity can exceed the torque: bounded ROA)
  con3d     the Gruene-Zidani (2015, Example 5.1) dynamics (unperturbed) with a driven third state
            x3' = -x3 + x1 x2, the state constraint x3 < 1 and a cylindrical keep-out region
            (x1 + 1)^2 + (x2 - 1)^2 > 0.3^2 as in their Example 5.2, on [-3, 3]^2 x [-2, 2]
  power     two-machine power system (Vannelli & Vidyasagar 1985; Liu et al. 2025, Example 7), shifted to
            the stable equilibrium, on [-2, 3] x [-3, 3]

Obstacle benchmarks of Meng & Liu (ECC 2026, arXiv 2511.09523, Section IV): their obstacles U_i = {h_i >= 1} are discs, written
here as d_i = |x - c_i| - r_i > 0; their region of interest Omega is the box (leaving it counts as leaving the admissible set)
  vdp_obs1    reversed Van der Pol, mu = 1, on [-2.5, 2.5] x [-3.5, 3.5], disc centre (1, 1), radius 0.25      (Example 3)
  vdp_obs2    as vdp_obs1 with a second disc, centre (-1, -1), radius 0.25                                     (Example 3)
  power_obs2  the power system on [-1, 1.5] x [-1.5, 1], discs of radius 1/8 at (0.25, 0.25), (0.25, -0.25)  (Example 4, case 1)
  power_obs4  the power system on [-2, 3] x [-3, 1.5], discs of radius 1/4 at (2, -1.5), (-1, 0.5), (-1, -1.5)
              and (pi/3, 0) (the last one covers the saddle)                                                (Example 4, case 2)
"""
from __future__ import annotations

import math

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


class System:
    def __init__(self, name, f, lo, hi, constraints=(), cnames=(), sim_T=40.0, tol=1e-3):
        self.name = name
        self._f = f
        self.lo = torch.tensor(lo, dtype=DT, device=DEV)
        self.hi = torch.tensor(hi, dtype=DT, device=DEV)
        self.n = len(lo)
        self.cons = list(constraints)
        self.cnames = list(cnames)
        self.center = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        self.s = 1.0 / self.half
        self.sim_T, self.tol = sim_T, tol
        self.device, self.dtype = DEV, DT

    def f(self, x):
        return self._f(x)

    def field(self, x):
        """Normalized field f / sqrt(1 + ||S f||^2): a positive time rescaling (same orbits and ROA), bounded."""
        f = torch.nan_to_num(self.f(x), nan=0.0, posinf=0.0, neginf=0.0)
        return f / torch.sqrt(1.0 + ((f * self.s) ** 2).sum(-1, keepdim=True))

    def jacobian0(self):
        x = torch.zeros(1, self.n, device=DEV, dtype=torch.float64, requires_grad=True)
        J = torch.autograd.functional.jacobian(lambda z: self._f(z.to(DT)).to(torch.float64)[0], x)
        return J[:, 0, :].detach().cpu().double().numpy()

    def distances(self, x):
        """Signed distances, > 0 admissible: the extra constraints, then the box faces (lo side, hi side)."""
        d = [c(x) for c in self.cons]
        d += [x[:, i] - self.lo[i] for i in range(self.n)] + [self.hi[i] - x[:, i] for i in range(self.n)]
        return torch.stack(d, -1)

    def dist_names(self):
        return self.cnames + [f"x{i + 1} low face" for i in range(self.n)] + [f"x{i + 1} high face" for i in range(self.n)]

    def admissible(self, x):
        return (torch.nan_to_num(self.distances(x), nan=-1.0) > 0).all(-1)

    def in_box(self, x):
        return ((x >= self.lo) & (x <= self.hi)).all(-1)

    def _map(self, u):
        return self.lo + (self.hi - self.lo) * u

    def sample_box(self, n, seed=0):
        u = torch.quasirandom.SobolEngine(self.n, scramble=True, seed=int(seed)).draw(n)
        return self._map(u.to(DEV, DT))

    def sample_edge(self, n, seed=0, width=0.1):
        """Each coordinate, with probability 1/2, from the slab of relative width `width` next to a random face."""
        u = torch.quasirandom.SobolEngine(self.n, scramble=True, seed=int(seed)).draw(n)
        g = torch.Generator().manual_seed(int(seed) + 3)
        for k in range(self.n):
            near = torch.rand(n, generator=g) < 0.5
            hi = torch.rand(n, generator=g) < 0.5
            w = width * u[:, k]
            u[:, k] = torch.where(near, torch.where(hi, 1 - w, w), u[:, k])
        return self._map(u.to(DEV, DT))

    def sample_band(self, n, seed=0, rel=0.1, steps=4):
        """Samples next to the extra constraints: Newton steps on d_i towards a uniform level in
        (0, rel * median d_i) (equal share per constraint).  Empty if there are no extra constraints."""
        if not self.cons:
            return torch.zeros(0, self.n, device=DEV, dtype=DT)
        x = self.sample_box(n, seed=seed)
        with torch.no_grad():
            med = torch.stack([c(x) for c in self.cons], -1).clamp_min(0).median(0).values
        g = torch.Generator().manual_seed(int(seed) + 7)
        tgt = torch.rand(n, generator=g).to(DEV, DT)
        out = []
        for i, idx in enumerate(torch.arange(n, device=DEV).chunk(len(self.cons))):
            xi, ti = x[idx], tgt[idx] * rel * med[i]
            for _ in range(steps):
                xi = xi.detach().requires_grad_(True)
                d = self.cons[i](xi)
                (gd,) = torch.autograd.grad(d.sum(), xi)
                xi = (xi + ((ti - d) / (gd ** 2).sum(-1).clamp_min(1e-12))[:, None] * gd).detach()
            out.append(xi)
        x = torch.cat(out)
        return x[self.admissible(x)]

    @torch.no_grad()
    def simulate(self, x0, dt=0.01, hold=1.0):
        """RK4 of the physical field.  Labels: 0 converged (normalized |x| < tol for `hold`), 1 left the admissible
        set (constraints or box), 3 inconclusive after sim_T."""
        x = x0.clone()
        N = len(x)
        label = torch.full((N,), 3, dtype=torch.long, device=DEV)
        held = torch.zeros(N, device=DEV, dtype=DT)
        act = torch.arange(N, device=DEV)
        for _ in range(int(round(self.sim_T / dt)) + 1):
            xa = x[act]
            bad = ~self.admissible(xa) | ~torch.isfinite(xa).all(-1)
            near = (xa * self.s).norm(dim=-1) < self.tol
            held[act] = torch.where(near, held[act] + dt, torch.zeros_like(held[act]))
            lab = torch.full_like(act, -1)
            lab[held[act] >= hold - 1e-9] = 0
            lab[bad] = 1
            done = lab >= 0
            label[act[done]] = lab[done]
            act, xa = act[~done], xa[~done]
            if len(act) == 0:
                break
            k1 = self.f(xa); k2 = self.f(xa + 0.5 * dt * k1); k3 = self.f(xa + 0.5 * dt * k2); k4 = self.f(xa + dt * k3)
            x[act] = xa + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return label


def vdp(mu=1.0):
    def f(x):
        return torch.stack([-x[:, 1], x[:, 0] - mu * (1 - x[:, 0] ** 2) * x[:, 1]], -1)
    lo, hi = ([-2.5, -3.5], [2.5, 3.5]) if mu == 1.0 else ([-3.0, -6.0], [3.0, 6.0])
    return System(f"vdp{mu:g}", f, lo, hi, sim_T=40.0)


def pendulum(umax=0.8, k1=4.4142, k2=2.3163):
    def f(x):
        u = (-(k1 * x[:, 0] + k2 * x[:, 1])).clamp(-umax, umax)
        return torch.stack([x[:, 1], torch.sin(x[:, 0]) - x[:, 1] + u], -1)
    return System("pendulum", f, [-math.pi, -4.0], [math.pi, 4.0], sim_T=40.0)


def con3d():
    def f(x):
        x1, x2, x3 = x.unbind(-1)
        return torch.stack([-x1 + x2, 0.1 * x1 - 2 * x2 - x1 ** 2 - 0.1 * x1 ** 3, -x3 + x1 * x2], -1)
    cons = [lambda x: 1.0 - x[:, 2],
            lambda x: torch.sqrt((x[:, 0] + 1.0) ** 2 + (x[:, 1] - 1.0) ** 2) - 0.3]
    return System("con3d", f, [-3.0, -3.0, -2.0], [3.0, 3.0, 2.0], cons, ["x3 limit", "obstacle"], sim_T=40.0)


def power(delta=math.pi / 3):
    def f(x):
        return torch.stack([x[:, 1], -0.5 * x[:, 1] - (torch.sin(x[:, 0] + delta) - math.sin(delta))], -1)
    return System("power", f, [-2.0, -3.0], [3.0, 3.0], sim_T=60.0)


def _disc(c, r):
    return lambda x: torch.sqrt((x[:, 0] - c[0]) ** 2 + (x[:, 1] - c[1]) ** 2) - r


def vdp_obs(k):
    """Meng & Liu (ECC 2026), Example 3: h_i = 1 + 1/4 - |x - c_i|^2 / 0.5^2, i.e. discs of radius 0.25."""
    base = vdp()
    cs = [(1.0, 1.0), (-1.0, -1.0)][:k]
    return System(f"vdp_obs{k}", base._f, [-2.5, -3.5], [2.5, 3.5], [_disc(c, 0.25) for c in cs],
                  [f"obstacle {i + 1}" for i in range(k)], sim_T=40.0)


def power_obs(k, delta=math.pi / 3):
    """Meng & Liu (ECC 2026), Example 4: h_i = 1 + r^2 - |x - c_i|^2, case 1 (two discs) and case 2 (four discs)."""
    base = power(delta)
    if k == 2:
        cs, r, lo, hi = [(0.25, 0.25), (0.25, -0.25)], 1 / 8, [-1.0, -1.5], [1.5, 1.0]
    else:
        cs, r, lo, hi = [(2.0, -1.5), (-1.0, 0.5), (-1.0, -1.5), (delta, 0.0)], 1 / 4, [-2.0, -3.0], [3.0, 1.5]
    return System(f"power_obs{k}", base._f, lo, hi, [_disc(c, r) for c in cs], [f"obstacle {i + 1}" for i in range(k)], sim_T=60.0)


SYSTEMS = {"vdp": vdp, "vdp3": lambda: vdp(3.0), "pendulum": pendulum, "con3d": con3d, "power": power,
           "vdp_obs1": lambda: vdp_obs(1), "vdp_obs2": lambda: vdp_obs(2),
           "power_obs2": lambda: power_obs(2), "power_obs4": lambda: power_obs(4)}


def get(name):
    return SYSTEMS[name]()
