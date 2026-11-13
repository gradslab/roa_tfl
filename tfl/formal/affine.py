r"""
Batched first-order affine arithmetic on the GPU (the bounding engine of the formal check).

A quantity over a cell is kept as   q = c + a . r + [-e, e],   r in [-1, 1]^K
(c centre, a the coefficients of the K shared cell symbols, e >= 0 the radius of everything nonlinear).
Shared symbols keep the linear dependence on the starting point exact, so v(x_tau) - v(x) cancels its
common linear part instead of adding two interval widths.  An interval is the special case a = 0.

Every elementary function f uses the mean-value form f(q) in f(c) + s (q - c) + [-E, E] with s the
midpoint of an enclosure [dl, du] of f' over the range of q and E = (du - dl) / 2 * rad(q) (sound for any
enclosure of f'); relu / abs use the exact triangle relaxation.  Invalid operations (1/q with 0 in the
range, log or sqrt of a range reaching <= 0) give e = inf, which every caller treats as "not proven".

Floating point: float64 with round-to-nearest; the radii are inflated by REL * |value| per operation as a
stand-in for directed rounding (not yet a rigorous rounding model).
"""
from __future__ import annotations

import math

import torch

REL = 1e-13
INF = float("inf")


def _num(x):
    return isinstance(x, (int, float)) or (torch.is_tensor(x) and not isinstance(x, AF))


class AF:
    __slots__ = ("c", "a", "e")

    def __init__(self, c, a, e):
        self.c, self.a, self.e = c, a, e

    @staticmethod
    def cell(lo, hi):
        """x_i = mid_i + rad_i r_i for a batch of boxes lo, hi of shape (N, K): returns K AFs of shape (N,)."""
        mid, rad = 0.5 * (lo + hi), 0.5 * (hi - lo)
        K = lo.shape[-1]
        eye = torch.eye(K, dtype=lo.dtype, device=lo.device)
        return [AF(mid[:, i], rad[:, i:i + 1] * eye[i], torch.zeros_like(mid[:, i])) for i in range(K)]

    @staticmethod
    def interval(lo, hi, K):
        c = 0.5 * (lo + hi)
        return AF(c, torch.zeros(*c.shape, K, dtype=c.dtype, device=c.device), 0.5 * (hi - lo))

    @staticmethod
    def const(v, like):
        c = torch.as_tensor(v, dtype=like.c.dtype, device=like.c.device).expand_as(like.c)
        return AF(c, torch.zeros_like(like.a), torch.zeros_like(like.e))

    @staticmethod
    def stack(qs):
        return AF(torch.stack([q.c for q in qs], -1), torch.stack([q.a for q in qs], -2),
                  torch.stack([q.e for q in qs], -1))

    def __getitem__(self, i):
        return AF(self.c[..., i], self.a[..., i, :], self.e[..., i])

    @property
    def rad(self):
        return self.a.abs().sum(-1) + self.e

    @property
    def lo(self):
        return self.c - self.rad

    @property
    def hi(self):
        return self.c + self.rad

    def hull(self):
        return AF(self.c, torch.zeros_like(self.a), self.rad)

    def _fix(self):
        self.e = self.e + REL * (self.c.abs() + self.rad)
        bad = ~torch.isfinite(self.c) | ~torch.isfinite(self.e) | ~torch.isfinite(self.a).all(-1)
        self.c = torch.where(bad, 0.0, self.c)
        self.a = torch.where(bad[..., None], 0.0, self.a)
        self.e = torch.where(bad, INF, self.e)
        return self

    def __add__(self, o):
        if _num(o):
            return AF(self.c + o, self.a, self.e)._fix()
        return AF(self.c + o.c, self.a + o.a, self.e + o.e)._fix()

    __radd__ = __add__

    def __neg__(self):
        return AF(-self.c, -self.a, self.e)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if _num(o):
            o_abs = abs(o) if not torch.is_tensor(o) else o.abs()
            return AF(self.c * o, self.a * (o[..., None] if torch.is_tensor(o) and o.dim() else o),
                      self.e * o_abs)._fix()
        if self is o:
            return self.square()
        rx, ry = self.rad, o.rad
        return AF(self.c * o.c, o.c[..., None] * self.a + self.c[..., None] * o.a,
                  self.c.abs() * o.e + o.c.abs() * self.e + rx * ry)._fix()

    __rmul__ = __mul__

    def square(self):
        r = self.rad
        return AF(self.c ** 2 + 0.5 * r ** 2, 2 * self.c[..., None] * self.a, 2 * self.c.abs() * self.e + 0.5 * r ** 2)._fix()

    def __truediv__(self, o):
        if _num(o):
            return self * (1.0 / o)
        return self * o.recip()

    def __rtruediv__(self, o):
        return self.recip() * o

    def __pow__(self, p):
        p = float(p)
        if p == 1:
            return self
        if p == 2:
            return self.square()
        if p == 0.5:
            return sqrt(self)
        if p == -0.5:
            return sqrt(self).recip()
        if p.is_integer():
            n = int(p)
            if n < 0:
                return self.recip() ** (-n)
            if n % 2 == 0:
                return (self ** (n // 2)).square()
            return _unary(self, lambda c: c ** n, lambda l, u: _ipow(l, u, n - 1, n))
        def dp(l, u):
            a, b = p * l.clamp_min(0) ** (p - 1), p * u.clamp_min(0) ** (p - 1)
            return _pos(l, (torch.minimum(a, b), torch.maximum(a, b)))
        return _unary(self, lambda c: c.clamp_min(0) ** p, dp)

    def recip(self):
        l, u = self.lo, self.hi
        ok = (l > 0) | (u < 0)

        def dr(l, u):
            m2 = torch.minimum(l * l, u * u)
            M2 = torch.maximum(l * l, u * u)
            return torch.where(ok, -1 / m2, -INF), torch.where(ok, -1 / M2, INF)
        out = _unary(self, lambda c: 1 / c, dr)
        out.e = torch.where(ok, out.e, torch.full_like(out.e, INF))
        return out

    def matmul_T(self, W, b=None):
        """q @ W.T + b for a constant matrix W (out, in); q has a trailing feature dimension."""
        c = self.c @ W.T + (0 if b is None else b)
        a = torch.einsum("...ik,oi->...ok", self.a, W)
        return AF(c, a, self.e @ W.abs().T)._fix()


def _pos(l, d):
    """Derivative enclosure valid only on ranges with l > 0; otherwise unbounded."""
    dl, du = d
    return torch.where(l > 0, dl, -INF), torch.where(l > 0, du, INF)


def _ipow(l, u, m, scale=1.0):
    """scale * [l, u]^m (integer m >= 0)."""
    if m == 0:
        o = torch.full_like(l, float(scale))
        return o, o
    pl, pu = l ** m, u ** m
    if m % 2 == 1:
        lo, hi = pl, pu
    else:
        lo = torch.where((l <= 0) & (u >= 0), torch.zeros_like(l), torch.minimum(pl, pu))
        hi = torch.maximum(pl, pu)
    return (scale * lo, scale * hi) if scale >= 0 else (scale * hi, scale * lo)


def _unary(x: AF, f, drange):
    l, u = x.lo, x.hi
    dl, du = drange(l, u)
    s = 0.5 * (dl + du)
    E = 0.5 * (du - dl) * x.rad
    fc = f(x.c)
    s0 = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    out = AF(fc, s0[..., None] * x.a, s0.abs() * x.e + E)
    bad = ~torch.isfinite(s) | ~torch.isfinite(E)
    out.e = torch.where(bad, torch.full_like(out.e, INF), out.e)
    return out._fix()


def _sin_range(l, u):
    two = 2 * math.pi
    full = (u - l) >= two
    kmax = torch.ceil((l - math.pi / 2) / two)
    has_max = (math.pi / 2 + two * kmax) <= u
    kmin = torch.ceil((l + math.pi / 2) / two)
    has_min = (-math.pi / 2 + two * kmin) <= u
    sl, su = torch.sin(l), torch.sin(u)
    lo = torch.where(has_min | full, -torch.ones_like(l), torch.minimum(sl, su))
    hi = torch.where(has_max | full, torch.ones_like(l), torch.maximum(sl, su))
    return lo, hi


def _cos_range(l, u):
    return _sin_range(l + math.pi / 2, u + math.pi / 2)


def sin(x):
    return _unary(x, torch.sin, _cos_range) if isinstance(x, AF) else math.sin(x)


def cos(x):
    if not isinstance(x, AF):
        return math.cos(x)
    return _unary(x, torch.cos, lambda l, u: tuple(-t for t in reversed(_sin_range(l, u))))


def tanh(x):
    def d(l, u):
        tl, tu = torch.tanh(l) ** 2, torch.tanh(u) ** 2
        mn = torch.where((l <= 0) & (u >= 0), torch.zeros_like(l), torch.minimum(tl, tu))
        return 1 - torch.maximum(tl, tu), 1 - mn
    return _unary(x, torch.tanh, d)


def exp(x):
    return _unary(x, torch.exp, lambda l, u: (torch.exp(l), torch.exp(u)))


def log(x):
    out = _unary(x, lambda c: torch.log(c.clamp_min(1e-300)), lambda l, u: _pos(l, (1 / u, 1 / l)))
    out.e = torch.where(x.lo > 0, out.e, torch.full_like(out.e, INF))
    return out


def sqrt(x):
    out = _unary(x, lambda c: torch.sqrt(c.clamp_min(0)), lambda l, u: _pos(l, (0.5 / torch.sqrt(u), 0.5 / torch.sqrt(l))))
    out.e = torch.where(x.lo > 0, out.e, torch.full_like(out.e, INF))
    return out


def softplus(x, k):
    """softplus(k x) / k."""
    return _unary(x, lambda c: torch.nn.functional.softplus(k * c) / k,
                  lambda l, u: (torch.sigmoid(k * l), torch.sigmoid(k * u)))


def relu(x):
    l, u = x.lo, x.hi
    pos, neg = l >= 0, u <= 0
    s = torch.where(pos, torch.ones_like(l), torch.where(neg, torch.zeros_like(l), u / (u - l).clamp_min(1e-300)))
    g = torch.where(pos | neg, torch.zeros_like(l), -s * l)
    out = AF(s * x.c + 0.5 * g, s[..., None] * x.a, s * x.e + 0.5 * g)
    return out._fix()


def absf(x):
    l, u = x.lo, x.hi
    pos, neg = l >= 0, u <= 0
    s = torch.where(pos, torch.ones_like(l), torch.where(neg, -torch.ones_like(l), (u + l) / (u - l).clamp_min(1e-300)))
    g = torch.where(pos | neg, torch.zeros_like(l), -l * (1 + s))
    out = AF(s * x.c + 0.5 * g, s[..., None] * x.a, s.abs() * x.e + 0.5 * g)
    return out._fix()


def maximum(x, y):
    return y + relu(x - y)


def mono(x, f, df):
    """Monotone-derivative function on x >= 0 (f' increasing): f' range [df(l), df(u)]."""
    return _unary(x, f, lambda l, u: (df(l.clamp_min(0)), df(u.clamp_min(0))))


MODULE = {"sin": sin, "cos": cos, "sqrt": sqrt, "tanh": tanh, "exp": exp, "log": log, "Abs": absf, "abs": absf,
          "pi": math.pi, "E": math.e}
