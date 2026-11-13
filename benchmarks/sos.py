r"""
SOS baseline: V-s iteration with an expanding interior (Jarvis-Wloszek et al. 2003; Topcu, Packard & Seiler 2008;
Packard et al. 2010), polynomial V of degree 6 (2-D) or 4 (3-D), in box-normalized coordinates x~ = S x.

    V - eps |x~|^2                                in SOS                     (V positive definite, V(0) = 0)
    -[(beta - p) s1 + (V - 1)]                     in SOS, s1 in SOS          ({p <= beta} inside {V <= 1})
    -[(1 - V) s2 + DV.f + eps |x~|^2]              in SOS, s2 in SOS          (V decreases on {V <= 1})
    c_j - eps - s3j (1 - V)                         in SOS, s3j in SOS         ({V <= 1} inside every constraint c_j > 0)
with p the linearization quadratic (scaled to unit largest eigenvalue).  Alternation: (multiplier step) fix V, find s2, s3 for the largest level gamma (bisection), rescale V,
then the largest beta with s1 (bisection);  (V step) fix the multipliers, maximize beta over V (one SDP).
Non-polynomial terms: Taylor expansion of sin to degree 5 (as in Liu et al. 2025).  Pendulum saturation: the region is
additionally required to lie in the unsaturated set |k.x| <= u_max, where the dynamics are polynomial.
The certified set {V <= 1} is evaluated against the simulated ground truth (coverage, false positives).

Run:  python -u sos.py vdp pendulum con3d power
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import sympy as sp
import torch

import systems as S
from gt import ground_truth

EPS = 1e-6
LAM_DEG = 2
S2_EXTRA = 2
OUT = Path(__file__).parent / "results"


def monos(n, lo, hi):
    return [e for d in range(lo, hi + 1) for e in itertools.product(range(d + 1), repeat=n) if sum(e) == d]


def padd(*ps):
    out = {}
    for p in ps:
        for e, c in p.items():
            out[e] = out[e] + c if e in out else c
    return out


def pscale(p, a):
    return {e: a * c for e, c in p.items()}


def pmul(p, q):
    out = {}
    for (e1, c1), (e2, c2) in itertools.product(p.items(), q.items()):
        e = tuple(a + b for a, b in zip(e1, e2))
        t = c1 * c2
        out[e] = out[e] + t if e in out else t
    return out


def pdiff(p, i):
    out = {}
    for e, c in p.items():
        if e[i] > 0:
            e2 = list(e); e2[i] -= 1
            out[tuple(e2)] = out.get(tuple(e2), 0) + e[i] * c
    return out


def deg(p):
    return max((sum(e) for e in p), default=0)


def from_sympy(expr, X):
    P = sp.Poly(sp.expand(expr), *X)
    return {tuple(int(k) for k in m): float(c) for m, c in zip(P.monoms(), P.coeffs())}


def var_poly(n, lo, hi, name):
    ms = monos(n, lo, hi)
    v = cp.Variable(len(ms), name=name)
    return {e: v[i] for i, e in enumerate(ms)}, v


def evaluate(p, x):
    """Numeric polynomial on a (N, n) array."""
    out = np.zeros(len(x))
    for e, c in p.items():
        out += float(c) * np.prod(x ** np.array(e), axis=1)
    return out


class SOSProgram:
    def __init__(self, n):
        self.n, self.cons = n, []

    def sos(self, p, lo=0):
        """p in SOS via a Gram matrix on monomials of degree lo .. deg(p)/2."""
        hi = (deg(p) + 1) // 2
        z = monos(self.n, lo, hi)
        Q = cp.Variable((len(z), len(z)), symmetric=True)
        self.cons.append(Q >> 0)
        gram = {}
        for i, j in itertools.product(range(len(z)), repeat=2):
            e = tuple(a + b for a, b in zip(z[i], z[j]))
            gram.setdefault(e, []).append((i, j))
        for e in set(gram) | set(p):
            lhs = p.get(e, 0)
            rhs = sum(Q[i, j] for i, j in gram.get(e, [])) if e in gram else 0
            self.cons.append(lhs == rhs)

    def sos_var(self, lo, hi, name):
        p, v = var_poly(self.n, lo, hi, name)
        self.sos(p, lo=lo // 2 if lo % 2 == 0 else (lo + 1) // 2)
        return p

    def solve(self, obj=None):
        prob = cp.Problem(cp.Maximize(obj) if obj is not None else cp.Minimize(0), self.cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, max_iters=20000)
            except Exception:
                return False
        return prob.status in ("optimal", "optimal_inaccurate")


def symbolic(name):
    sys_ = S.get(name)
    n = sys_.n
    X = sp.symbols(f"y1:{n + 1}", real=True)
    aux_g, aux_h = [], []
    s = [float(v) for v in sys_.s.cpu()]
    x = [X[i] / s[i] for i in range(n)]
    T5 = lambda a: a - a ** 3 / 6 + a ** 5 / 120
    cons = []
    if name in ("vdp", "vdp3"):
        mu = 1.0 if name == "vdp" else 3.0
        f = [-x[1], x[0] - mu * (1 - x[0] ** 2) * x[1]]
    elif name == "pendulum":
        k1, k2, um = 4.4142, 2.3163, 0.8
        u = -(k1 * x[0] + k2 * x[1])
        w = sp.Symbol("w", real=True)
        X = tuple(X) + (w,)
        f = [x[1], T5(x[0]) - x[1] + w]
        aux_g += [um ** 2 - w ** 2, (u - w) * w]
        aux_h += [(u - w) * (um ** 2 - w ** 2)]
    elif name == "con3d":
        f = [-x[0] + x[1], 0.1 * x[0] - 2 * x[1] - x[0] ** 2 - 0.1 * x[0] ** 3, -x[2] + x[0] * x[1]]
        cons += [1 - x[2], (x[0] + 1) ** 2 + (x[1] - 1) ** 2 - 0.09]
    elif name == "power":
        d = math.pi / 3
        sinT = sum(((-1) ** k) * (x[0] ** (2 * k + 1)) / math.factorial(2 * k + 1) for k in range(3))
        cosT = sum(((-1) ** k) * (x[0] ** (2 * k)) / math.factorial(2 * k) for k in range(3))
        f = [x[1], -0.5 * x[1] - (math.sin(d) * (cosT - 1) + math.cos(d) * sinT)]
    lo, hi = [float(v) for v in sys_.lo.cpu()], [float(v) for v in sys_.hi.cpu()]
    cons += [x[i] - lo[i] for i in range(n)] + [hi[i] - x[i] for i in range(n)]
    fy = [from_sympy(s[i] * f[i], X) for i in range(n)]
    cy = [from_sympy(c, X) for c in cons]
    return sys_, X, fy, cy, [from_sympy(g, X) for g in aux_g], [from_sympy(h, X) for h in aux_h]


def vdot(V, fy, n):
    return padd(*[pmul(pdiff(V, i), fy[i]) for i in range(n)])


def run(name, dV=None, iters=60, verbose=True):
    sys_, X, fy, cy, ag, ah = symbolic(name)
    nx, n = sys_.n, len(X)
    dV = dV or (6 if n == 2 else 4)
    p = {tuple(2 if j == i else 0 for j in range(n)): 1.0 for i in range(nx)}
    nrm = p
    from scipy.linalg import solve_continuous_lyapunov
    sc = sys_.s.double().cpu().numpy()
    A = np.diag(sc) @ sys_.jacobian0() @ np.diag(1 / sc)
    P = solve_continuous_lyapunov(A.T, -np.eye(nx))
    V = {}
    for i, j in itertools.product(range(nx), repeat=2):
        e = tuple((1 if k == i else 0) + (1 if k == j else 0) for k in range(n))
        V[e] = V.get(e, 0) + P[i, j]
    p = pscale(V, 1.0 / max(np.linalg.eigvalsh(P)))
    ddeg = max(deg(f) for f in fy)
    d2 = max(0, dV - 1 + ddeg - dV) + S2_EXTRA
    d2 += d2 % 2

    def aux_terms(pr):
        """sum sigma_k g_k + lambda_j h_j (sigma SOS, lambda free), added inside the negated decrease expression."""
        out = []
        for k, gk in enumerate(ag):
            out.append(pmul(pr.sos_var(0, 2, f"sg{k}"), gk))
        for j, hj in enumerate(ah):
            lam, _ = var_poly(n, 0, LAM_DEG, f"lh{j}")
            out.append(pmul(lam, hj))
        return out

    def level_feasible(V, g):
        pr = SOSProgram(n)
        s2 = pr.sos_var(2, max(2, d2), "s2")
        pr.sos(pscale(padd(pmul(padd({tuple([0] * n): g}, pscale(V, -1)), s2), vdot(V, fy, nx), pscale(nrm, EPS), *aux_terms(pr)), -1), lo=0 if ag else 1)
        for j, c in enumerate(cy):
            s3 = pr.sos_var(0, 2, f"s3{j}")
            pr.sos(padd(c, {tuple([0] * n): -EPS}, pscale(pmul(s3, padd({tuple([0] * n): g}, pscale(V, -1))), -1)))
        ok = pr.solve()
        if not ok:
            return None
        return ({e: float(c.value) if hasattr(c, "value") else c for e, c in s2.items()},
                None)

    def max_level(V, hi=None):
        lo_, hi_ = 0.0, hi or 1.0
        while level_feasible(V, hi_) is not None and hi_ < 1e6:
            lo_, hi_ = hi_, 2 * hi_
        best = None
        for _ in range(14):
            m = 0.5 * (lo_ + hi_)
            r = level_feasible(V, m)
            if r is not None:
                lo_, best = m, r
            else:
                hi_ = m
        return lo_

    def multipliers(V):
        """s2, s3 at level 1 and s1 for the largest beta (V scaled so that level 1 is feasible)."""
        pr = SOSProgram(n)
        s2 = pr.sos_var(2, max(2, d2), "s2")
        pr.sos(pscale(padd(pmul(padd({tuple([0] * n): 1.0}, pscale(V, -1)), s2), vdot(V, fy, nx), pscale(nrm, EPS), *aux_terms(pr)), -1), lo=0 if ag else 1)
        s3s = []
        for j, c in enumerate(cy):
            s3 = pr.sos_var(0, 2, f"s3{j}")
            pr.sos(padd(c, {tuple([0] * n): -EPS}, pscale(pmul(s3, padd({tuple([0] * n): 1.0}, pscale(V, -1))), -1)))
            s3s.append(s3)
        if not pr.solve():
            return None
        num = lambda q: {e: float(c.value) for e, c in q.items()}
        s2n, s3n = num(s2), [num(s) for s in s3s]
        lo_, hi_, s1n = 0.0, 1.0, None
        def beta_ok(b):
            pr = SOSProgram(n)
            s1 = pr.sos_var(0, dV - 2, "s1")
            pr.sos(pscale(padd(pmul(padd({tuple([0] * n): b}, pscale(p, -1)), s1), padd(V, {tuple([0] * n): -1.0})), -1))
            return num(s1) if pr.solve() else None
        while beta_ok(hi_) is not None and hi_ < 1e3:
            lo_, hi_ = hi_, 2 * hi_
        for _ in range(14):
            m = 0.5 * (lo_ + hi_)
            r = beta_ok(m)
            if r is not None:
                lo_, s1n = m, r
            else:
                hi_ = m
        s1n = beta_ok(0.98 * lo_) or s1n
        return s1n, s2n, s3n, lo_

    def V_step(s1, s2, s3):
        pr = SOSProgram(n)
        Vv, _ = var_poly(n, 2, dV, "V")
        Vv = {e: c for e, c in Vv.items() if all(k == 0 for k in e[nx:])}
        b = cp.Variable()
        pr.sos(padd(Vv, pscale(nrm, -EPS)), lo=1)
        one = {tuple([0] * n): 1.0}
        pr.sos(pscale(padd({e: b * c for e, c in s1.items()}, pscale(pmul(p, s1), -1), Vv, {tuple([0] * n): -1.0}), -1))
        pr.sos(pscale(padd(pmul(padd(one, pscale(Vv, -1)), s2), vdot(Vv, fy, nx), pscale(nrm, EPS), *aux_terms(pr)), -1), lo=0 if ag else 1)
        for c, s in zip(cy, s3):
            pr.sos(padd(c, {tuple([0] * n): -EPS}, pscale(pmul(s, padd(one, pscale(Vv, -1))), -1)))
        if not pr.solve(b):
            return None, None
        return {e: float(c.value) for e, c in Vv.items()}, float(b.value)

    t0 = time.time()
    g = max_level(V)
    V = pscale(V, 1.0 / (0.98 * g))
    best_beta, hist, stall = 0.0, [], 0
    for it in range(iters):
        m = multipliers(V)
        for _ in range(6):
            if m is not None:
                break
            V = pscale(V, 1 / 0.9)
            m = multipliers(V)
        if m is None:
            if verbose:
                print(f"  {name} iter {it}: multiplier step infeasible (level normalization {g:.4g})", flush=True)
            break
        s1, s2, s3, beta = m
        Vn, bn = V_step(s1, s2, s3) if s1 is not None else (None, None)
        hist.append(dict(iter=it, beta=beta, beta_V=bn))
        if verbose:
            print(f"  {name} iter {it}: beta {beta:.4f}  (V step beta {bn})  [{time.time() - t0:.0f} s]", flush=True)
        if Vn is None:
            break
        g = max_level(Vn, hi=1.0)
        V = pscale(Vn, 1.0 / (0.98 * g))
        stall = stall + 1 if beta <= best_beta * (1 + 1e-3) else 0
        best_beta = max(best_beta, beta)
        if stall >= 3:
            break
    return sys_, pscale(V, 0.98), hist


def evaluate_region(sys_, V):
    xg, roa = ground_truth(sys_)
    y = (xg * sys_.s).double().cpu().numpy()
    na = len(next(iter(V)))
    if na > y.shape[1]:
        y = np.concatenate([y, np.zeros((len(y), na - y.shape[1]))], 1)
    reg = torch.as_tensor(evaluate(V, y) <= 1.0, device=roa.device)
    nr = roa.sum().item()
    return 100 * (reg & roa).sum().item() / nr, 100 * (reg & ~roa).sum().item() / len(roa)


def main():
    for name in sys.argv[1:]:
        t0 = time.time()
        sys_, V, hist = run(name)
        cov, fp = evaluate_region(sys_, V)
        res = dict(method="sos", system=sys_.name, degree=max(sum(e) for e in V), coverage=cov, false_positive=fp, iterations=hist,
                   seconds=time.time() - t0, V=[[list(e), float(c)] for e, c in V.items()],
                   saturation="exact (auxiliary variable w = sat(u))" if name == "pendulum" else None)
        (OUT / f"baseline_sos_{name}.json").write_text(json.dumps(res, indent=1))
        print(f"sos    {sys_.name:9s} degree {res['degree']}  coverage {cov:6.2f}%  FP {fp:.2f}%  [{res['seconds']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
