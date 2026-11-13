r"""
Verified core of the equilibrium benchmarks (floating-point interval arithmetic with outward rounding).

The quadratic of the pipeline is q(x) = x~^T P x~, x~ = S x (box-normalized, S = diag(sys.s)), A~^T P + P A~ = -I for the
linearization (core.Problem).  P and S are used exactly as the pipeline stores them (float32 values, exact in float64 and
in rational arithmetic), so q(x) = x^T M x with the exact rational matrix M = S P S.  For a level c the claim verified is

  (a) E_c = {q <= c} lies in the admissible set: the open box and every extra constraint (the x3 limit and the cylinder
      of con3d, the discs of the obstacle systems);
  (b) dq/dt = grad q . f < 0 on E_c \ {0} for the exact closed-loop field f (sin, the saturation of the pendulum and all
      decimal constants enclosed; pi is enclosed, no irrational constant is left unenclosed).

Then E_c is compact, forward invariant and admissible, and q is a strict Lyapunov function on it, so every trajectory
from E_c stays in E_c and converges to the origin: E_c lies in the constrained region of attraction.  The sign of dq/dt
is the same for the physical and the normalized field (a positive time rescaling).

How (a) and (b) are verified.
  * M is positive definite (Sylvester's criterion, exact rationals).  E_c is the ellipsoid with the tight bounding box
    |x_i| <= sqrt(c (M^-1)_ii); (a) for the box faces is the exact rational test c (M^-1)_ii < m_i^2, where m_i is a lower
    bound of the distance from 0 to the nearer face i, taken over the exact box (pi enclosed) and the float32 box of the
    code (the smaller of the two).
  * Local part (near 0, where dq/dt vanishes): on a box N = {|x_i| <= rho half_i} around 0 the field is C^1 and
    f(x) = B(x) x with B(x) = int_0^1 Df(t x) dt, which lies in the interval hull [Df](N) (N is convex and contains 0).
    dq/dt = x^T (M B + B^T M) x.  The symmetric interval matrix C = M [Df](N) + [Df](N)^T M is formed in exact rationals
    from the float endpoints of [Df](N); with midpoint C_m and radius R (entrywise), every symmetric C_m + E with
    |E| <= R satisfies lambda_max <= lambda_max(C_m) + ||R||_2 <= lambda_max(C_m) + ||R||_F, so it suffices that
    -C_m - r I is positive definite for a rational r >= ||R||_F (Sylvester, exact rationals).  Then dq/dt < 0 on N \ {0}.
  * Branch and bound over the bounding box of E_c: a cell is discarded when q > c on it, or when all extra constraints
    are positive on it and either dq/dt < 0 on it or it lies in N.  dq/dt on a cell is bounded by the natural interval
    extension of 2 (M x) . f(x), intersected with the mean-value form g(m) + sum_k [dg/dx_k](X)(X_k - m_k) where f is
    smooth on a neighbourhood of the cell (for the pendulum: the saturation strictly active or strictly inactive on the
    whole cell).  Remaining cells are bisected along their widest normalized side; a cell below the minimum width, or a
    cell whose midpoint is a definite counterexample, ends the verification with "not verified".
  * Interval arithmetic: float64 with outward rounding by two units in the last place after every operation (as in
    tfl/inner_region.py); sin and cos from numpy (about one ulp), widened by a further 1e-15, with the extrema
    located with a margin; sat has
    an exact interval extension (it is monotone, only min and max are needed).  Decimal constants are enclosed by the
    neighbouring floats unless the float is exact; pi by [prev(pi_64), next(pi_64)].

The largest verified level is found by bisection (12 rounds below the sampled first failure level), then a 1 % safety
factor is applied and the final level c0 is verified again.  Self-tests: the checker must fail for a level above a
sampled failure, for the field f + kappa q(x) x (decrease false on part of E_c0, true near 0) and for f + gamma x
(decrease false near 0).

    python -u core_proof.py vdp vdp3 pendulum con3d power          (writes results/core_<sys>.json)
    python -u core_proof.py vdp_obs1 vdp_obs2 power_obs2 power_obs4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import torch

INF = float("inf")
RES = Path(__file__).parent / "results"


def _out(lo, hi):
    """Outward rounding by two ulps on each side."""
    lo = np.nextafter(np.nextafter(lo, -INF), -INF)
    hi = np.nextafter(np.nextafter(hi, INF), INF)
    return lo, hi


class IV:
    """Closed intervals [lo, hi] (numpy float64 arrays, broadcasting; scalar intervals are constants)."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    @staticmethod
    def _lift(o):
        if isinstance(o, IV):
            return o
        c = float(o)
        return IV(c, c)

    def __add__(self, o):
        o = IV._lift(o)
        return IV(*_out(self.lo + o.lo, self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return IV(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-IV._lift(o))

    def __rsub__(self, o):
        return IV._lift(o) - self

    def __mul__(self, o):
        o = IV._lift(o)
        with np.errstate(invalid="ignore"):
            p = np.stack(np.broadcast_arrays(self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi))
        ok = ~np.broadcast_to(np.isnan(self.lo) | np.isnan(self.hi) | np.isnan(o.lo) | np.isnan(o.hi), p.shape[1:])
        p = np.where(np.isnan(p) & ok[None], 0.0, p)
        return IV(*_out(p.min(0), p.max(0)))

    __rmul__ = __mul__

    def sq(self):
        a, b = self.lo ** 2, self.hi ** 2
        lo = np.where(self.lo >= 0, a, np.where(self.hi <= 0, b, 0.0))
        lo2, hi2 = _out(lo, np.maximum(a, b))
        return IV(np.maximum(lo2, 0.0), hi2)

    def cube(self):
        return IV(*_out(self.lo ** 3, self.hi ** 3))

    def mid(self):
        return 0.5 * (self.lo + self.hi)


def hull(*vals):
    lo = min(float(v) for v in vals)
    hi = max(float(v) for v in vals)
    return IV(lo, hi)


def K(dec, *extra):
    """Enclosure of the decimal constant `dec` (string), widened to contain the floats in `extra` (e.g. the float32
    value the code uses)."""
    F = Fr(dec)
    f = float(F)
    lo = f if Fr(f) <= F else float(np.nextafter(f, -INF))
    hi = f if Fr(f) >= F else float(np.nextafter(f, INF))
    for e in extra:
        lo, hi = min(lo, float(e)), max(hi, float(e))
    return IV(lo, hi)


PI = IV(float(np.nextafter(math.pi, 0.0)), float(np.nextafter(math.pi, 4.0)))


def _contains_point(lo, hi, a, period):
    """True where a + k period may lie in [lo, hi] (checked with a margin, so false positives only)."""
    m = 1e-9
    return np.ceil((lo - m - a) / period) <= np.floor((hi + m - a) / period)


LIBM_ERR = 1e-15


def isin(x):
    s = np.stack([np.sin(x.lo), np.sin(x.hi)])
    smin, smax = s.min(0), s.max(0)
    two_pi = 2 * math.pi
    wide = x.hi - x.lo >= 6.0
    smax = np.where(_contains_point(x.lo, x.hi, math.pi / 2, two_pi) | wide, 1.0, smax)
    smin = np.where(_contains_point(x.lo, x.hi, -math.pi / 2, two_pi) | wide, -1.0, smin)
    lo, hi = _out(smin - LIBM_ERR, smax + LIBM_ERR)
    return IV(np.maximum(lo, -1.0), np.minimum(hi, 1.0))


def icos(x):
    c = np.stack([np.cos(x.lo), np.cos(x.hi)])
    cmin, cmax = c.min(0), c.max(0)
    two_pi = 2 * math.pi
    wide = x.hi - x.lo >= 6.0
    cmax = np.where(_contains_point(x.lo, x.hi, 0.0, two_pi) | wide, 1.0, cmax)
    cmin = np.where(_contains_point(x.lo, x.hi, math.pi, two_pi) | wide, -1.0, cmin)
    lo, hi = _out(cmin - LIBM_ERR, cmax + LIBM_ERR)
    return IV(np.maximum(lo, -1.0), np.minimum(hi, 1.0))


def isat(a, u):
    """Exact interval extension of sat(a) = min(max(a, -u), u) over a in [a], u in [u] (u > 0): monotone in a."""
    lo = np.where(a.lo >= 0, np.minimum(a.lo, u.lo), np.maximum(a.lo, -u.hi))
    hi = np.where(a.hi <= 0, np.maximum(a.hi, -u.lo), np.minimum(a.hi, u.hi))
    return IV(lo, hi)


def fr_iv(F):
    """Float interval enclosing the rational F."""
    f = float(F)
    lo = f if Fr(f) <= F else float(np.nextafter(f, -INF))
    hi = f if Fr(f) >= F else float(np.nextafter(f, INF))
    return IV(lo, hi)


def fr_up(F):
    f = float(F)
    return f if Fr(f) >= F else float(np.nextafter(f, INF))


def fr_down(F):
    f = float(F)
    return f if Fr(f) <= F else float(np.nextafter(f, -INF))


def sqrt_up(F):
    """Float r with r >= sqrt(F) (F rational >= 0), checked exactly."""
    r = math.sqrt(float(F)) if F > 0 else 0.0
    while Fr(r) ** 2 < F:
        r = float(np.nextafter(r, INF))
    return r


def fr_det(A):
    """Determinant of a small rational matrix (fraction-exact Gaussian elimination)."""
    A = [row[:] for row in A]
    n, d = len(A), Fr(1)
    for k in range(n):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None:
            return Fr(0)
        if p != k:
            A[k], A[p] = A[p], A[k]
            d = -d
        d *= A[k][k]
        for i in range(k + 1, n):
            r = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= r * A[k][j]
    return d


def fr_posdef(A):
    """Sylvester's criterion: all leading principal minors > 0 (exact)."""
    return all(fr_det([row[:k] for row in A[:k]]) > 0 for k in range(1, len(A) + 1))


def fr_inv(A):
    n = len(A)
    G = [A[i][:] + [Fr(int(i == j)) for j in range(n)] for i in range(n)]
    for k in range(n):
        p = next(i for i in range(k, n) if G[i][k] != 0)
        G[k], G[p] = G[p], G[k]
        piv = G[k][k]
        G[k] = [v / piv for v in G[k]]
        for i in range(n):
            if i != k and G[i][k] != 0:
                r = G[i][k]
                G[i] = [a - r * b for a, b in zip(G[i], G[k])]
    return [row[n:] for row in G]


def fr_ldl(A):
    """A = L D L^T, L unit lower triangular (exact); requires positive leading minors."""
    n = len(A)
    L = [[Fr(int(i == j)) for j in range(n)] for i in range(n)]
    D = [Fr(0)] * n
    for j in range(n):
        D[j] = A[j][j] - sum(L[j][k] ** 2 * D[k] for k in range(j))
        for i in range(j + 1, n):
            L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))) / D[j]
    return L, D


class Spec:
    """Interval form of a benchmark of systems.py.  field(X) -> [f_i], jac(X) -> ([[df_i/dx_j]], smooth), cons(X) ->
    [(name, d_j)] with d_j > 0 admissible (extra constraints only), box: exact bounds as intervals."""

    def __init__(self, name, n, box_lo, box_hi, field, jac, cons=lambda X: [], f32_box=None):
        self.name, self.n = name, n
        self.box_lo, self.box_hi = box_lo, box_hi
        self.field, self.jac, self.cons = field, jac, cons


def _always_smooth(X):
    return np.ones(np.broadcast(X[0].lo, X[-1].lo).shape, dtype=bool)


def vdp_spec(name, mu, lo, hi, discs=()):
    mu = float(mu)

    def field(X):
        x1, x2 = X
        return [-x2, x1 - (1 - x1.sq()) * x2 * mu]

    def jac(X):
        x1, x2 = X
        z = IV(np.zeros_like(x1.lo), np.zeros_like(x1.lo))
        return [[z, z - 1.0], [1 + x1 * x2 * (2 * mu), (1 - x1.sq()) * (-mu)]], _always_smooth(X)

    return Spec(name, 2, [K(str(v)) for v in lo], [K(str(v)) for v in hi], field, jac, _disc_cons(discs))


def _disc_cons(discs):
    """Discs (centre cx, cy and radius r as intervals): d = (x1 - cx)^2 + (x2 - cy)^2 - r^2 > 0 (same sign as the
    code's |x - c| - r for r > 0; the enclosures contain the exact and the float32 constants)."""
    def cons(X):
        x1, x2 = X
        return [(f"obstacle {i + 1}", (x1 - cx).sq() + (x2 - cy).sq() - r.sq()) for i, (cx, cy, r) in enumerate(discs)]
    return cons


def _f32(v):
    return float(np.float32(v))


def pendulum_spec():
    k1, k2, um = K("4.4142", _f32(4.4142)), K("2.3163", _f32(2.3163)), K("0.8", _f32(0.8))

    def field(X):
        x1, x2 = X
        u = isat(-(k1 * x1 + k2 * x2), um)
        return [x2, isin(x1) - x2 + u]

    def jac(X):
        x1, x2 = X
        a = -(k1 * x1 + k2 * x2)
        lin = (a.hi < um.lo) & (a.lo > -um.lo)
        satd = (a.lo > um.hi) | (a.hi < -um.hi)
        sig = IV(np.where(lin, 1.0, 0.0), np.where(satd, 0.0, 1.0))
        z = IV(np.zeros_like(x1.lo), np.zeros_like(x1.lo))
        return [[z, z + 1.0], [icos(x1) - k1 * sig, -1.0 - k2 * sig]], lin | satd

    return Spec("pendulum", 2, [-PI, K("-4")], [PI, K("4")], field, jac)


def con3d_spec():
    c = K("0.1")

    def field(X):
        x1, x2, x3 = X
        return [-x1 + x2, c * x1 - 2 * x2 - x1.sq() - c * x1.cube(), -x3 + x1 * x2]

    def jac(X):
        x1, x2, x3 = X
        z = IV(np.zeros_like(x1.lo), np.zeros_like(x1.lo))
        return [[z - 1.0, z + 1.0, z], [c - 2 * x1 - 3 * c * x1.sq(), z - 2.0, z], [x2, x1, z - 1.0]], _always_smooth(X)

    r = K("0.3", _f32(0.3))

    def cons(X):
        x1, x2, x3 = X
        return [("x3 limit", 1.0 - x3), ("obstacle", (x1 + 1.0).sq() + (x2 - 1.0).sq() - r.sq())]

    return Spec("con3d", 3, [K("-3"), K("-3"), K("-2")], [K("3"), K("3"), K("2")], field, jac, cons)


def _third(x):
    """x / 3 for an interval x, with outward rounding."""
    return IV(*_out(x.lo / 3.0, x.hi / 3.0))


DELTA = _third(PI)
SIN_DELTA = isin(DELTA)


def power_spec(name="power", lo=(-2, -3), hi=(3, 3), discs=()):
    half = K("0.5")

    def field(X):
        x1, x2 = X
        return [x2, -(half * x2) - (isin(x1 + DELTA) - SIN_DELTA)]

    def jac(X):
        x1, x2 = X
        z = IV(np.zeros_like(x1.lo), np.zeros_like(x1.lo))
        return [[z, z + 1.0], [-icos(x1 + DELTA), z - 0.5]], _always_smooth(X)

    return Spec(name, 2, [K(str(v)) for v in lo], [K(str(v)) for v in hi], field, jac, _disc_cons(discs))


def _disc(cx, cy, r):
    return (cx if isinstance(cx, IV) else K(cx, _f32(float(Fr(cx)))), cy if isinstance(cy, IV) else K(cy, _f32(float(Fr(cy)))),
            K(r, _f32(float(Fr(r)))))


_D64 = math.pi / 3
DELTA_CODE = hull(DELTA.lo, DELTA.hi, _D64, _f32(_D64))

SPECS = {
    "vdp": lambda: vdp_spec("vdp", 1, (-2.5, -3.5), (2.5, 3.5)),
    "vdp3": lambda: vdp_spec("vdp3", 3, (-3, -6), (3, 6)),
    "pendulum": pendulum_spec,
    "con3d": con3d_spec,
    "power": power_spec,
    "vdp_obs1": lambda: vdp_spec("vdp_obs1", 1, (-2.5, -3.5), (2.5, 3.5), [_disc("1", "1", "0.25")]),
    "vdp_obs2": lambda: vdp_spec("vdp_obs2", 1, (-2.5, -3.5), (2.5, 3.5), [_disc("1", "1", "0.25"), _disc("-1", "-1", "0.25")]),
    "power_obs2": lambda: power_spec("power_obs2", (-1, -1.5), (1.5, 1), [_disc("0.25", "0.25", "1/8"), _disc("0.25", "-0.25", "1/8")]),
    "power_obs4": lambda: power_spec("power_obs4", (-2, -3), (3, 1.5),
                                     [_disc("2", "-1.5", "1/4"), _disc("-1", "0.5", "1/4"), _disc("-1", "-1.5", "1/4"),
                                      _disc(DELTA_CODE, K("0"), "1/4")]),
}


class Quad:
    """q(x) = x^T M x with the exact rational M = S P S from the stored float32 P and s."""

    def __init__(self, P32, s32):
        n = len(s32)
        self.P = [[Fr(float(P32[i][j])) for j in range(n)] for i in range(n)]
        self.s = [Fr(float(v)) for v in s32]
        self.n = n
        self.M = [[self.s[i] * (self.P[i][j] + self.P[j][i]) / 2 * self.s[j] for j in range(n)] for i in range(n)]
        self.posdef = fr_posdef(self.M)
        self.Minv = fr_inv(self.M)
        L, D = fr_ldl(self.M)
        self.Liv = [[fr_iv(L[i][j]) for j in range(n)] for i in range(n)]
        self.Div = [fr_iv(d) for d in D]
        self.Miv = [[fr_iv(self.M[i][j]) for j in range(n)] for i in range(n)]
        self.Mf = np.array([[float(self.M[i][j]) for j in range(n)] for i in range(n)])

    def q(self, X):
        n = self.n
        ldl = None
        for k in range(n):
            lin = X[k]
            for j in range(k + 1, n):
                lin = lin + self.Liv[j][k] * X[j]
            t = lin.sq() * self.Div[k]
            ldl = t if ldl is None else ldl + t
        nat = None
        for i in range(n):
            t = X[i].sq() * self.Miv[i][i]
            for j in range(i + 1, n):
                t = t + X[i] * X[j] * (self.Miv[i][j] * 2.0)
            nat = t if nat is None else nat + t
        return IV(np.maximum(ldl.lo, nat.lo), np.minimum(ldl.hi, nat.hi))

    def Mx(self, X):
        out = []
        for i in range(self.n):
            t = None
            for j in range(self.n):
                u = self.Miv[i][j] * X[j]
                t = u if t is None else t + u
            out.append(t)
        return out

    def qf(self, x):
        """Float64 value of q at points x (n_pts, n): for sampling only."""
        return np.einsum("ni,ij,nj->n", x, self.Mf, x)

    def bbox(self, c):
        """Upper bounds r_i >= sqrt(c (M^-1)_ii): the tight bounding box of {q <= c}."""
        return np.array([sqrt_up(Fr(c) * self.Minv[i][i]) for i in range(self.n)])


def dq_bounds(spec, Q, X, smooth_needed=True):
    """Upper/lower enclosure of dq/dt = 2 (M x) . f(x) on cells X: natural extension intersected with the mean-value
    form where f is smooth on the cell."""
    f = spec.field(X)
    Mx = Q.Mx(X)
    nat = None
    for i in range(spec.n):
        t = Mx[i] * f[i]
        nat = t if nat is None else nat + t
    nat = nat * 2.0
    J, smooth = spec.jac(X)
    m = [IV(x.mid(), x.mid()) for x in X]
    fm = spec.field(m)
    Mm = Q.Mx(m)
    gm = None
    for i in range(spec.n):
        t = Mm[i] * fm[i]
        gm = t if gm is None else gm + t
    gm = gm * 2.0
    MF = Q.Mx(f)
    cen = gm
    for k in range(spec.n):
        G = MF[k]
        for i in range(spec.n):
            G = G + Mx[i] * J[i][k]
        G = G * 2.0
        cen = cen + G * (X[k] - m[k])
    lo = np.where(smooth, np.maximum(nat.lo, cen.lo), nat.lo)
    hi = np.where(smooth, np.minimum(nat.hi, cen.hi), nat.hi)
    return IV(lo, hi), gm


def local_test(spec, Q, rho, half):
    """Verified negative definiteness of M B + B^T M for every B in [Df](N), N = {|x_i| <= rho half_i}."""
    r = rho * half
    X = [IV(np.array([-r[i]]), np.array([r[i]])) for i in range(spec.n)]
    J, smooth = spec.jac(X)
    if not bool(np.all(smooth)):
        return False, dict(reason="field not smooth on N")
    adm = all(bool(np.all(d.lo > 0)) for _, d in spec.cons(X))
    n = spec.n
    Bl = [[Fr(float(J[i][j].lo[0])) for j in range(n)] for i in range(n)]
    Bh = [[Fr(float(J[i][j].hi[0])) for j in range(n)] for i in range(n)]
    if any(not math.isfinite(float(J[i][j].lo[0])) or not math.isfinite(float(J[i][j].hi[0])) for i in range(n) for j in range(n)):
        return False, dict(reason="unbounded Jacobian")
    M = Q.M

    def prod_iv(m, lo, hi):
        a, b = m * lo, m * hi
        return (min(a, b), max(a, b))

    Cl = [[Fr(0)] * n for _ in range(n)]
    Ch = [[Fr(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            lo = hi = Fr(0)
            for k in range(n):
                a = prod_iv(M[i][k], Bl[k][j], Bh[k][j])
                b = prod_iv(M[k][j], Bl[k][i], Bh[k][i])
                lo += a[0] + b[0]
                hi += a[1] + b[1]
            Cl[i][j], Ch[i][j] = lo, hi
    Cm = [[(Cl[i][j] + Ch[i][j]) / 2 for j in range(n)] for i in range(n)]
    R2 = sum(((Ch[i][j] - Cl[i][j]) / 2) ** 2 for i in range(n) for j in range(n))
    rad = Fr(sqrt_up(R2))
    T = [[-Cm[i][j] - (rad if i == j else 0) for j in range(n)] for i in range(n)]
    ok = fr_posdef(T)
    lam = float(np.linalg.eigvalsh(np.array([[float(v) for v in row] for row in Cm])).max())
    return ok and adm, dict(rho=rho, N_half_widths=r.tolist(), lambda_max_midpoint=lam, radius_frobenius=float(rad),
                            admissible=adm, negative_definite=ok)


def best_local(spec, Q, half):
    for rho in (0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        ok, info = local_test(spec, Q, rho, half)
        if ok:
            return rho, info
    return None, info


def box_ok(spec, Q, c, f32_lo, f32_hi):
    """(a) for the box faces: c (M^-1)_ii < m_i^2 exactly, m_i = distance from 0 to the nearer face (lower bound over
    the exact and the float32 box)."""
    out = []
    for i in range(spec.n):
        m = min(-float(spec.box_lo[i].hi), float(spec.box_hi[i].lo), -float(f32_lo[i]), float(f32_hi[i]))
        out.append(m > 0 and Fr(c) * Q.Minv[i][i] < Fr(m) ** 2)
    return all(out), out


def verify(spec, Q, c, rho, half, f32_lo, f32_hi, min_rel_width=2.0 ** -18, max_cells=30_000_000, batch=100000):
    """True if (a) and (b) hold on {q <= c}.  Returns (verified, info)."""
    t0 = time.time()
    if not Q.posdef:
        return False, dict(reason="M not positive definite")
    lok, linfo = local_test(spec, Q, rho, half)
    if not lok:
        return False, dict(reason="local test failed on N", local=linfo)
    bok, per = box_ok(spec, Q, c, f32_lo, f32_hi)
    if not bok:
        return False, dict(reason="ellipsoid leaves the open box", faces=per)
    rN = rho * half
    rb = Q.bbox(c)
    lo, hi = -rb[None, :].copy(), rb[None, :].copy()
    processed = 0
    while len(lo):
        if processed > max_cells:
            return False, dict(reason="cell budget exhausted", processed=processed)
        nlo, nhi = [], []
        for j in range(0, len(lo), batch):
            L, H = lo[j:j + batch], hi[j:j + batch]
            X = [IV(L[:, i], H[:, i]) for i in range(spec.n)]
            q = Q.q(X)
            out = q.lo > c
            processed += len(L)
            cons = spec.cons(X)
            adm = np.ones(len(L), dtype=bool)
            for _, d in cons:
                adm &= d.lo > 0
            inN = np.all((L >= -rN) & (H <= rN), axis=1)
            dq, gm = dq_bounds(spec, Q, X)
            dec = dq.hi < 0
            keep = ~out & ~(adm & (dec | inN))
            if not keep.any():
                continue
            m = [IV(x.mid(), x.mid()) for x in X]
            qm = Q.q(m)
            bad_m = (qm.hi <= c) & (((gm.lo >= 0) & ~inN))
            for _, d in spec.cons(m):
                bad_m |= (qm.hi <= c) & (d.hi <= 0)
            bad_m &= keep
            if bad_m.any():
                i = int(np.nonzero(bad_m)[0][0])
                return False, dict(reason="counterexample", processed=processed, x=[float(v.lo[i]) for v in m],
                                   q=float(qm.lo[i]), dq=float(gm.lo[i]))
            L, H = L[keep], H[keep]
            rel = (H - L) / (2 * half)
            if (rel.max(1) < min_rel_width).any():
                k = int(np.argmax(rel.max(1) < min_rel_width))
                return False, dict(reason="undecided cell below the minimum width", processed=processed,
                                   cell=[L[k].tolist(), H[k].tolist()])
            dim = rel.argmax(1)
            mid = 0.5 * (L + H)
            A, B = L.copy(), H.copy()
            idx = np.arange(len(L))
            A[idx, dim] = mid[idx, dim]
            B[idx, dim] = mid[idx, dim]
            nlo.append(np.concatenate([L, A]))
            nhi.append(np.concatenate([B, H]))
        lo = np.concatenate(nlo) if nlo else np.zeros((0, spec.n))
        hi = np.concatenate(nhi) if nhi else np.zeros((0, spec.n))
    return True, dict(processed=processed, seconds=time.time() - t0)


def sampled_failure_level(spec, Q, half, f32_lo, f32_hi, n=400000, seed=777):
    """Smallest q over box samples that violate (a) or (b) (point intervals), and the box limit min_i m_i^2/(M^-1)_ii."""
    rng = np.random.default_rng(seed)
    lo = np.maximum([float(b.hi) for b in spec.box_lo], f32_lo)
    hi = np.minimum([float(b.lo) for b in spec.box_hi], f32_hi)
    x = lo + (hi - lo) * rng.random((n, spec.n))
    X = [IV(x[:, i], x[:, i]) for i in range(spec.n)]
    q = Q.qf(x)
    dq, _ = dq_bounds(spec, Q, X)
    znorm = np.linalg.norm(x / half, axis=1)
    bad = (dq.lo >= 0) & (znorm > 1e-3)
    for _, d in spec.cons(X):
        bad |= d.hi <= 0
    c_samp = float(q[bad].min()) if bad.any() else INF
    c_box = min(float(Fr(min(-float(spec.box_lo[i].hi), float(spec.box_hi[i].lo), -float(f32_lo[i]), float(f32_hi[i]))) ** 2
                      / Q.Minv[i][i]) for i in range(spec.n))
    return c_samp, c_box, (x[bad][np.argmin(q[bad])].tolist() if bad.any() else None)


def largest_level(spec, Q, rho, half, f32_lo, f32_hi, c_hi, rounds=12, verbose=True):
    a, b = 0.0, c_hi
    ok, info = verify(spec, Q, c_hi * 1e-4, rho, half, f32_lo, f32_hi)
    if not ok:
        return 0.0, info
    total = info["processed"]
    last_fail = None
    for _ in range(rounds):
        m = 0.5 * (a + b)
        ok, inf_m = verify(spec, Q, m, rho, half, f32_lo, f32_hi)
        total += inf_m.get("processed", 0)
        if verbose:
            print(f"    c = {m:.6g}: {'verified' if ok else 'not verified (' + inf_m['reason'] + ')'}  "
                  f"({inf_m.get('processed')} cells)", flush=True)
        if ok:
            a = m
        else:
            b = m
            last_fail = dict(level=m, **{k: v for k, v in inf_m.items() if k in ("reason", "x", "q", "dq", "faces", "cell")})
    if last_fail and last_fail.get("x") is not None:
        X = [IV(np.array([v]), np.array([v])) for v in last_fail["x"]]
        viol = [nm for nm, d in spec.cons(X) if float(d.hi[0]) <= 0]
        last_fail["limited_by"] = ("constraint " + ", ".join(viol)) if viol else "decrease of q"
    elif last_fail:
        last_fail["limited_by"] = "box" if last_fail.get("reason", "").startswith("ellipsoid") else last_fail.get("reason")
    return a, dict(bisection_cells=total, first_failure_above=last_fail)


def crosscheck(spec, sys_, n=20000, seed=5):
    """Transcription check (not part of the proof): the interval field at point cells contains the float64 value of
    the code's field, and the extra constraints have the code's sign away from their zero sets."""
    rng = np.random.default_rng(seed)
    lo, hi = sys_.lo.double().cpu().numpy(), sys_.hi.double().cpu().numpy()
    x = lo + (hi - lo) * rng.random((n, spec.n))
    X = [IV(x[:, i], x[:, i]) for i in range(spec.n)]
    f = spec.field(X)
    with torch.no_grad():
        ft = sys_._f(torch.as_tensor(x, dtype=torch.float64)).numpy()
        dt = torch.stack([c(torch.as_tensor(x, dtype=torch.float64)) for c in sys_.cons], -1).numpy() if sys_.cons else np.zeros((n, 0))
    tol = 1e-12 * (1 + np.abs(ft))
    field_ok = all(bool(np.all((f[i].lo - tol[:, i] <= ft[:, i]) & (ft[:, i] <= f[i].hi + tol[:, i]))) for i in range(spec.n))
    cons = spec.cons(X)
    cons_ok = len(cons) == dt.shape[1]
    if cons_ok:
        for j, (_, d) in enumerate(cons):
            far = np.abs(dt[:, j]) > 1e-6
            cons_ok &= bool(np.all(((d.lo > 0) == (dt[:, j] > 0))[far] | ((d.lo <= 0) & (d.hi >= 0))[far]))
    lo_ok = bool(np.all([abs(float(spec.box_lo[i].mid()) - lo[i]) < 1e-6 and abs(float(spec.box_hi[i].mid()) - hi[i]) < 1e-6
                         for i in range(spec.n)]))
    return dict(field_contains_code_value=field_ok, constraint_signs_agree=bool(cons_ok), box_matches=lo_ok)


class Perturbed(Spec):
    """Self-test fields: f + kappa q(x) x  (decrease false on part of E, true near 0) or f + gamma x (false near 0)."""

    def __init__(self, base, Q, kappa=0.0, gamma=0.0):
        self.__dict__.update(base.__dict__)
        self.base, self.Q, self.kappa, self.gamma = base, Q, float(kappa), float(gamma)

        def field(X):
            f = base.field(X)
            q = Q.q(X)
            return [f[i] + q * X[i] * self.kappa + X[i] * self.gamma for i in range(self.n)]

        def jac(X):
            J, sm = base.jac(X)
            q, Mx = Q.q(X), Q.Mx(X)
            out = [[J[i][j] + (X[i] * Mx[j] * (2.0 * self.kappa)) + ((q * self.kappa + self.gamma) if i == j else 0.0)
                    for j in range(self.n)] for i in range(self.n)]
            return out, sm

        self.field, self.jac = field, jac


def load_quad(name):
    import core
    import systems as S
    sys_ = S.get(name)
    prob = core.Problem(sys_, core.Config(system=name))
    return sys_, prob


def stored_inner_level(name):
    vals = []
    for k in range(4):
        run = Path(__file__).parent / "runs" / f"{name}_s{k}_rep"
        if not (run / "certify.json").exists():
            continue
        snap = json.loads((run / "certify.json").read_text())["snapshot"]
        sd = torch.load(run / "snapshots" / snap, map_location="cpu")
        vals.append(float(sd["inner_c"]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--outdir", default=str(RES))
    a = ap.parse_args()
    torch.set_num_threads(2)
    from gt import ground_truth
    for name in a.systems:
        t0 = time.time()
        spec = SPECS[name]()
        sys_, prob = load_quad(name)
        P32 = prob.P.cpu().numpy().astype(np.float64)
        s32 = sys_.s.cpu().numpy().astype(np.float64)
        f32_lo, f32_hi = sys_.lo.cpu().numpy().astype(np.float64), sys_.hi.cpu().numpy().astype(np.float64)
        half = 0.5 * (f32_hi - f32_lo)
        Q = Quad(P32, s32)
        xc = crosscheck(spec, sys_)
        print(f"[{name}] M positive definite: {Q.posdef}; transcription check {xc}", flush=True)
        rho, linfo = best_local(spec, Q, half)
        if rho is None:
            raise SystemExit(f"{name}: local test failed for every rho: {linfo}")
        print(f"  local part verified on N = {{|x_i| <= {rho} half_i}}: lambda_max(C_m) {linfo['lambda_max_midpoint']:.4g}, "
              f"radius {linfo['radius_frobenius']:.3g}", flush=True)
        c_samp, c_box, x_samp = sampled_failure_level(spec, Q, half, f32_lo, f32_hi)
        c_hi = min(c_samp, c_box)
        print(f"  sampled first failure level {c_samp:.6g}, box limit {c_box:.6g}", flush=True)
        c_bis, binfo = largest_level(spec, Q, rho, half, f32_lo, f32_hi, c_hi, rounds=a.rounds)
        c0 = 0.99 * c_bis
        t1 = time.time()
        ok, info = verify(spec, Q, c0, rho, half, f32_lo, f32_hi)
        t_final = time.time() - t1
        if not ok or c0 <= 0:
            raise SystemExit(f"{name}: final level not verified: {info}")
        st = {}
        c_bad = 1.02 * c_hi
        ok_bad, inf_bad = verify(spec, Q, c_bad, rho, half, f32_lo, f32_hi)
        st["level_above_sampled_failure"] = dict(level=c_bad, verified=ok_bad, reason=inf_bad.get("reason"))
        rng = np.random.default_rng(11)
        rb = Q.bbox(c0)
        xs = rb * (2 * rng.random((400000, spec.n)) - 1)
        qs = Q.qf(xs)
        sel = (qs <= c0) & (qs >= 0.5 * c0)
        xs, qs = xs[sel], qs[sel]
        dq, _ = dq_bounds(spec, Q, [IV(xs[:, i], xs[:, i]) for i in range(spec.n)])
        ratio = -dq.hi / (2 * qs ** 2)
        kappa = 2.0 * float(ratio.min())
        pk = Perturbed(spec, Q, kappa=kappa)
        wit = int(np.argmin(ratio))
        dq_w = float(dq_bounds(pk, Q, [IV(xs[wit:wit + 1, i], xs[wit:wit + 1, i]) for i in range(spec.n)])[0].lo[0])
        ok_k, inf_k = verify(pk, Q, c0, rho, half, f32_lo, f32_hi)
        st["field_plus_kappa_q_x"] = dict(kappa=kappa, witness=xs[wit].tolist(), witness_q=float(qs[wit]),
                                          witness_dq_lower_bound=dq_w, local_test_passes=local_test(pk, Q, rho, half)[0],
                                          verified=ok_k, reason=inf_k.get("reason"))
        lamP = float(np.linalg.eigvalsh(0.5 * (P32 + P32.T)).max())
        gamma = 1.0 / lamP
        pg = Perturbed(spec, Q, gamma=gamma)
        ok_g, inf_g = verify(pg, Q, c0, rho, half, f32_lo, f32_hi)
        lg = local_test(pg, Q, rho, half)
        st["field_plus_gamma_x"] = dict(gamma=gamma, local_test_passes=lg[0], verified=bool(ok_g and lg[0]),
                                        reason=inf_g.get("reason"), note="the proof needs both the local test and the branch and bound")
        selftest_pass = (not ok_bad) and (not ok_k) and dq_w > 0 and (not (ok_g and lg[0]))
        print(f"  self-tests: too large level -> {'not verified' if not ok_bad else 'VERIFIED (BAD)'}; "
              f"f + kappa q x -> {'not verified' if not ok_k else 'VERIFIED (BAD)'} (witness dq >= {dq_w:.3g}); "
              f"f + gamma x -> local {lg[0]}, bb {ok_g}", flush=True)
        n = spec.n
        detM = fr_det(Q.M)
        vol_E = math.pi ** (n / 2) / math.gamma(n / 2 + 1) * c0 ** (n / 2) / math.sqrt(float(detM))
        box_vol = float(np.prod(f32_hi - f32_lo))
        xg, roa = ground_truth(sys_)
        xg64 = xg.cpu().double().numpy()
        roa = roa.cpu().numpy()
        inE = Q.qf(xg64) < c0
        c_in = stored_inner_level(name)
        cin = max(c_in) if c_in else None
        inner_chk = None
        if cin is not None:
            rbI = Q.bbox(cin)
            rng = np.random.default_rng(13)
            cnt = dict(in_I=0, in_I_outside_E=0, in_I_dq_nonneg=0, in_I_inadmissible=0)
            n_draw = 4_000_000
            for _ in range(8):
                xs_ = rbI * (2 * rng.random((n_draw // 8, n)) - 1)
                m_ = Q.qf(xs_) < cin
                xs_ = xs_[m_]
                XI = [IV(xs_[:, i], xs_[:, i]) for i in range(n)]
                dqI, _ = dq_bounds(spec, Q, XI)
                badc = np.zeros(len(xs_), dtype=bool)
                for _, d in spec.cons(XI):
                    badc |= d.hi <= 0
                cnt["in_I"] += int(m_.sum())
                cnt["in_I_outside_E"] += int((Q.qf(xs_) >= c0).sum())
                cnt["in_I_dq_nonneg"] += int(((dqI.lo >= 0) & (np.linalg.norm(xs_ / half, axis=1) > 1e-3)).sum())
                cnt["in_I_inadmissible"] += int(badc.sum())
            scale = float(np.prod(2 * rbI)) / box_vol / n_draw * 100
            inner_chk = dict(samples=n_draw, **cnt, **{"share_of_box_" + k: v * scale for k, v in cnt.items()})
        res = dict(system=name, c0=c0, c_bisection=c_bis, safety_factor=0.99, rho_local=rho, local=linfo,
                   share_of_box=100 * vol_E / box_vol, share_of_box_sampled=100 * float(inE.mean()),
                   share_of_reference_roa=100 * float((inE & roa).sum() / roa.sum()),
                   reference_states_in_E_outside_roa=int((inE & ~roa).sum()),
                   cells_final=info["processed"], seconds_final=t_final, cells_bisection=binfo.get("bisection_cells"),
                   seconds_total=time.time() - t0, sampled_failure_level=c_samp, sampled_failure_point=x_samp, box_limit=c_box,
                   c_inner_sampled=c_in, inner_inside_core=(None if cin is None else cin <= c0),
                   inner_over_core_level=(cin / c0 if cin else None),
                   inner_over_core_volume=((cin / c0) ** (n / 2) if cin else None), inner_sampled_check=inner_chk,
                   first_failure_above=binfo.get("first_failure_above"),
                   selftests=st, selftests_pass=selftest_pass, crosscheck=xc,
                   constants=dict(P_float32=P32.tolist(), s_float32=s32.tolist(), box_lo_float32=f32_lo.tolist(),
                                  box_hi_float32=f32_hi.tolist()),
                   arithmetic="float64 intervals, outward rounding by 2 ulps per operation; pi and decimal constants enclosed; "
                              "exact rationals for M, the bounding box and the local matrix test")
        outdir = Path(a.outdir)
        outdir.mkdir(exist_ok=True)
        (outdir / f"core_{name}.json").write_text(json.dumps(res, indent=1))
        print(f"VERIFIED core {name}: c0 = {c0:.6g} ({100 * vol_E / box_vol:.2f} % of the box, "
              f"{res['share_of_reference_roa']:.2f} % of the reference ROA), final check {info['processed']} cells "
              f"{t_final:.1f} s; sampled inner level {cin} -> inside core: {res['inner_inside_core']}  "
              f"[{res['seconds_total']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
