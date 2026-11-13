r"""
Sound sum-of-squares baseline for the non-polynomial benchmarks (saturated pendulum, power system), and the same
V-s iteration with selectable settings for the polynomial benchmarks (used for the Van der Pol mu = 3 comparison).

Non-polynomial terms.  Each sine is replaced by its Taylor polynomial of degree 5 plus a remainder r with the
Lagrange bound |r| <= t^6 / 720 (t the argument of the Taylor expansion, in the original coordinates; the bound
holds for every real t because every derivative of sin is bounded by 1):
    pendulum  sin(x1)            = T5(x1) + r,      T5(t) = t - t^3/6 + t^5/120
    power     sin(x1 + delta)    = T5d(x1) + r,     T5d the degree-5 Taylor polynomial of sin(. + delta) at 0
The vector field is affine in r, x' = f0(x) + g r, so dV/dt = a(x) + c(x) r and, for every |r| <= b(x),
    dV/dt <= max(a + c b, a - c b).
Default (--rem vertex): the decrease condition is imposed at the two vertices r = +b(x) and r = -b(x), each with its own
S-procedure multipliers; this is the vertex form of Topcu & Packard (2009, Prop. 4.4) and the sign splitting of
Chesi (2005, Thm 1) with the global bound |theta| <= 1 on the sixth derivative, and it is equivalent to the condition
for every admissible r.  Alternative (--rem sproc): r is an extra indeterminate and the bound enters through SOS
multipliers on b - r >= 0 and b + r >= 0.  Either way the certificate holds for the true sine, hence for the true
system.

Saturation (pendulum).  u = -(k1 x1 + k2 x2), w = sat(u) is an auxiliary indeterminate described by
    um^2 - w^2 >= 0,   (u - w) w >= 0,   (u - w)(um^2 - w^2) = 0,
whose real solution set is exactly the graph w = sat(u) (|w| <= um; the equality forces w = u or |w| = um; the
sector condition fixes the sign of w when |w| = um).  The inequalities get SOS multipliers, the equality a free one.

Conditions (y = S x box-normalized, V(0) = 0, level 1, p the shape function, eps = 1e-6):
    V - eps |y|^2                                          SOS
    -[(beta - p) s1 + (V - 1)]                             SOS,  s1 SOS     ({p <= beta} inside {V <= 1})
    -[(1 - V) s2_k + dV/dt|_k + eps |y|^2 + aux_k]         SOS,  s2_k SOS   (decrease, k over the remainder vertices)
    c_j - eps - s3_j (1 - V)                               SOS,  s3_j SOS   ({V <= 1} inside every constraint c_j > 0)
V-s iteration with an expanding interior (Jarvis-Wloszek et al. 2003; Topcu, Packard & Seiler 2008; Packard et al.
2010), exactly as in sos.py: (multiplier step) fix V, bisect the largest level gamma with s2, s3, rescale V so that
the certified level is 1/0.98, then bisect the largest beta with s1; (V step) fix s1, s2, s3, maximize beta over V and
the auxiliary multipliers; re-normalize.  Stop after `--stall` consecutive iterations without a 0.1 % gain in beta,
after `--iters` iterations, on an infeasible step, or at the time cap.

Every SDP is solved with Clarabel alone, and a solve counts as feasible only with status "optimal" (no SCS, no
"optimal_inaccurate").  The final level is re-certified in separate solves (Clarabel, status "optimal"; levels 1,
0.999, 0.99, ... with default and 1e-10 tolerances and facial reduction) and checked a posteriori in float64
(Program.gram_check): multipliers are replaced by z'Q+z, residuals are absorbed into the Gram matrices, and the
smallest eigenvalues are reported.  The returned set is {V <= level} at the first level that passes.  A sample check
on the true dynamics follows.  The iterate returned is the one with the largest beta.

Run:  python -u sos_sound.py power pendulum                    # main runs -> results/baseline_sos_sound_<sys>.json
      python -u sos_sound.py vdp3 --shape id --init x --s1deg 2 --s2deg 4 --box 0 --iters 30 --stall 0 --out FILE
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy.sparse as spr
import sympy as sp
import torch

import sos as SOS0
import systems as S

EPS = 1e-6
OUT = Path(__file__).parent / "results"
STATUS = Counter()
TIMES = Counter()
SOLVER_OPTS = {}


monos, padd, pscale, pmul, pdiff, deg, from_sympy, evaluate = (SOS0.monos, SOS0.padd, SOS0.pscale, SOS0.pmul,
                                                              SOS0.pdiff, SOS0.deg, SOS0.from_sympy, SOS0.evaluate)


class LP:
    """c(x) + sum_k M_k(x) v_k: const {mono: float}, terms {name: (cvxpy var, {mono: {col: coef}})}."""

    def __init__(self, const=None, terms=None):
        self.c = dict(const or {})
        self.t = dict(terms or {})

    @staticmethod
    def var(n, lo, hi, name, keep=None):
        ms = [e for e in monos(n, lo, hi) if keep is None or keep(e)]
        v = cp.Variable(len(ms), name=name)
        return LP({}, {name: (v, {e: {i: 1.0} for i, e in enumerate(ms)})}), ms

    @staticmethod
    def scalar_times(b, name, q):
        """b * q for a scalar cvxpy variable b and a numeric polynomial q."""
        return LP({}, {name: (b, {e: {0: c} for e, c in q.items()})})

    def __add__(self, o):
        if isinstance(o, dict):
            o = LP(o)
        out = LP(padd(self.c, o.c), {k: (v, {e: dict(m) for e, m in mp.items()}) for k, (v, mp) in self.t.items()})
        for k, (v, mp) in o.t.items():
            if k not in out.t:
                out.t[k] = (v, {})
            tgt = out.t[k][1]
            for e, m in mp.items():
                d = tgt.setdefault(e, {})
                for j, a in m.items():
                    d[j] = d.get(j, 0.0) + a
        return out

    def scale(self, a):
        return LP(pscale(self.c, a), {k: (v, {e: {j: a * c for j, c in m.items()} for e, m in mp.items()})
                                      for k, (v, mp) in self.t.items()})

    def mul(self, q):
        """times a numeric polynomial q."""
        out_t = {}
        for k, (v, mp) in self.t.items():
            new = {}
            for (e1, m), (e2, c2) in itertools.product(mp.items(), q.items()):
                e = tuple(a + b for a, b in zip(e1, e2))
                d = new.setdefault(e, {})
                for j, a in m.items():
                    d[j] = d.get(j, 0.0) + a * c2
            out_t[k] = (v, new)
        return LP(pmul(self.c, q), out_t)

    def diff(self, i):
        out_t = {}
        for k, (v, mp) in self.t.items():
            new = {}
            for e, m in mp.items():
                if e[i] > 0:
                    e2 = list(e); e2[i] -= 1
                    new[tuple(e2)] = {j: e[i] * a for j, a in m.items()}
            out_t[k] = (v, new)
        return LP(pdiff(self.c, i), out_t)

    def support(self):
        s = set(self.c)
        for _, mp in self.t.values():
            s |= set(mp)
        return s

    def value(self):
        out = dict(self.c)
        for _, (v, mp) in self.t.items():
            val = np.atleast_1d(v.value)
            for e, m in mp.items():
                out[e] = out.get(e, 0.0) + sum(a * float(val[j]) for j, a in m.items())
        return out


def as_lp(p):
    return p if isinstance(p, LP) else LP(p)


NEWTON = True
LENIENT = False


def newton_filter(z, sup):
    """Keep the monomials m with 2m in conv(sup) (one small LP each)."""
    from scipy.optimize import linprog
    pts = np.array(sorted(sup), dtype=float)
    keep = []
    for m in z:
        t = 2 * np.array(m, dtype=float)
        if any((pts == t).all(1)):
            keep.append(m); continue
        A = np.vstack([pts.T, np.ones(len(pts))])
        r = linprog(np.zeros(len(pts)), A_eq=A, b_eq=np.append(t, 1.0), bounds=(0, None), method="highs")
        if r.status == 0:
            keep.append(m)
    return keep


class Program:
    def __init__(self, n, drop=None):
        self.n, self.cons, self.grams = n, [], []
        self.drop = drop or {}

    def sos(self, p, lo=0, label="", drop=()):
        """p in SOS via a Gram matrix on the monomials of degree lo .. ceil(deg p / 2) that lie in half the Newton
        polytope of the (structural) support of p (lossless: Reznick 1978), minus `drop`."""
        p = as_lp(p)
        sup = p.support()
        hi = (max(sum(e) for e in sup) + 1) // 2
        z = [e for e in monos(self.n, lo, hi) if e not in drop]
        if NEWTON and self.n > 1:
            z = newton_filter(z, sup)
        drop = set(drop) | set(self.drop.get(label, ()))
        z = [e for e in z if e not in drop]
        N = len(z)
        Q = cp.Variable((N, N), symmetric=True)
        gram = {}
        for i, j in itertools.product(range(N), repeat=2):
            gram.setdefault(tuple(a + b for a, b in zip(z[i], z[j])), []).append((i, j))
        M = sorted(sup | set(gram))
        idx = {e: r for r, e in enumerate(M)}
        rows, cols = [], []
        for e, ij in gram.items():
            for i, j in ij:
                rows.append(idx[e]); cols.append(i + j * N)
        G = spr.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(M), N * N))
        expr = -(G @ cp.vec(Q, order="F"))
        cvec = np.zeros(len(M))
        for e, c in p.c.items():
            cvec[idx[e]] += c
        for k, (v, mp) in p.t.items():
            r_, c_, d_ = [], [], []
            for e, m in mp.items():
                for j, a in m.items():
                    r_.append(idx[e]); c_.append(j); d_.append(a)
            A = spr.csr_matrix((d_, (r_, c_)), shape=(len(M), int(np.prod(v.shape)) if v.shape else 1))
            expr = expr + A @ (v if v.shape else cp.reshape(v, (1,), order="F"))
        self.cons += [Q >> 0, expr + cvec == 0]
        self.grams.append(dict(label=label, p=p, Q=Q, z=z, M=M, gram=gram))

    def sos_var(self, lo, hi, name, keep=None):
        p, _ = LP.var(self.n, lo, hi, name, keep)
        self.sos(p, lo=lo // 2 if lo % 2 == 0 else (lo + 1) // 2, label=f"multiplier {name}")
        return p

    def solve(self, obj=None, tag=""):
        prob = cp.Problem(cp.Maximize(obj) if obj is not None else cp.Minimize(0), self.cons)
        t0 = time.time()
        try:
            prob.solve(solver=cp.CLARABEL, **SOLVER_OPTS)
            st = prob.status
        except Exception as ex:
            st = f"error:{type(ex).__name__}"
        STATUS[f"{tag}:{st}"] += 1
        TIMES[tag] += time.time() - t0
        if os.environ.get("SOS_VERBOSE"):
            print(f"    solve {tag}: {st} {time.time() - t0:.1f} s", flush=True)
        self.status = st
        return st == "optimal" or (LENIENT and tag != "recert" and st == "optimal_inaccurate")

    def gram_check(self, R=None):
        """A posteriori check of the solved program (after Loefberg 2009).
        1. Every SOS multiplier is replaced by z'Q+z, Q+ the PSD part of its Gram matrix (eigenvalues clipped at 0):
           the multipliers used below are SOS by construction.
        2. For every constraint block p (evaluated with these multipliers), E places the coefficient residual of
           p - z'Qz on one entry pair, so z'(Q+E)z = p except for residuals e_m on monomials m outside the Gram
           support.  `exact`: lambda_min(Q+E) > tol (float tolerance) and no such residual: p is SOS.
           `certified`: exact, or, for the decrease blocks, which are used only on the compact set
           K = {|X_k| <= R_k} (states of {V <= 1}, inside the box by the exact containment certificates, and
           |w| <= um): every m with e_m != 0 has a basis monomial a with 2a <= m, and
           lambda_min(Q+E) > tol + sum_m |e_m| prod_k R_k^(m - 2a)_k; then p >= 0 on K because
           |X^m| <= (X^a)^2 prod R^(m - 2a) <= |z|^2 prod R^(m - 2a) there.
           Decrease blocks only: residuals on monomials that no basis monomial dominates, of size <= 1e-15 max|coef|
           (the float64 resolution of the data), are reported as "up to float rounding".  This is not an exact
           argument; for the pendulum result these residuals (~2e-16 on y1, y2, w) are proportional to the linear
           part of h and vanish when the constant of the free multiplier of h is shifted by ~2e-17 (independent audit).
        All checks are float64 a-posteriori checks (Loefberg 2009), not rational or interval verification."""
        out = []
        for g in self.grams:
            if not g["label"].startswith("multiplier"):
                continue
            Q = np.asarray(g["Q"].value, dtype=float)
            w_, U = np.linalg.eigh(0.5 * (Q + Q.T))
            Qp = (U * np.clip(w_, 0.0, None)) @ U.T
            ((v, mp),) = g["p"].t.values()
            new = np.zeros(int(np.prod(v.shape)))
            for e, m in mp.items():
                (j, _), = m.items()
                new[j] = sum(Qp[i, k] for i, k in g["gram"].get(e, []))
            g["moved"] = float(np.abs(new - np.asarray(v.value)).max())
            v.value = new
            out.append(dict(label=g["label"], size=len(Q), lam_min_Q=float(w_[0]), lam_min_Q_plus_E=float(max(w_[0], 0.0)),
                            max_residual=g["moved"], unabsorbed_residual=0.0, unabsorbed_bound_on_K=0.0, float_tol=0.0,
                            exact=True, certified=True, by="SOS by construction (z'Q+z)"))
        for g in self.grams:
            if g["label"].startswith("multiplier"):
                continue
            Q = np.asarray(g["Q"].value, dtype=float)
            Q = 0.5 * (Q + Q.T)
            pv = g["p"].value()
            E = np.zeros_like(Q)
            unabsorbed, rmax, bound, unbounded = 0.0, 0.0, 0.0, 0.0
            for e in g["M"]:
                res = pv.get(e, 0.0) - sum(Q[i, j] for i, j in g["gram"].get(e, []))
                rmax = max(rmax, abs(res))
                if e not in g["gram"]:
                    if res != 0.0:
                        unabsorbed = max(unabsorbed, abs(res))
                        cs = [np.prod([R[k] ** (e[k] - 2 * a[k]) for k in range(len(e))]) for a in g["z"]
                              if R is not None and all(2 * a[k] <= e[k] for k in range(len(e)))]
                        if cs and np.isfinite(min(cs)):
                            bound += abs(res) * float(min(cs))
                        else:
                            unbounded = max(unbounded, abs(res))
                    continue
                i, j = g["gram"][e][0]
                if i == j:
                    E[i, i] += res
                else:
                    E[i, j] += res / 2; E[j, i] += res / 2
            lam_q = float(np.linalg.eigvalsh(Q)[0])
            lam = float(np.linalg.eigvalsh(Q + E)[0])
            tol = len(Q) * np.finfo(float).eps * max(1.0, float(np.abs(Q).max()))
            exact = bool(lam > tol and unabsorbed == 0.0)
            on_K = "decrease" in g["label"] and R is not None and all(np.isfinite(R))
            noise = 1e-15 * max(1.0, max((abs(c) for c in pv.values()), default=1.0))
            certified = exact or bool(on_K and unbounded == 0.0 and lam > tol + bound)
            rounding = (not certified) and on_K and bool(lam > tol + bound and unbounded <= noise)
            certified = certified or rounding
            out.append(dict(label=g["label"], size=len(Q), lam_min_Q=lam_q, lam_min_Q_plus_E=lam,
                            max_residual=rmax, unabsorbed_residual=unabsorbed, unabsorbed_bound_on_K=bound,
                            float_tol=tol, exact=exact, certified=certified, residual_without_bound=unbounded,
                            data_resolution=noise,
                            by=("exact SOS" if exact else (f"up to float rounding of the data (residual {max(unbounded, unabsorbed):.1e} "
                                f"<= {noise:.1e})" if rounding else ("positive on K" if certified else "not certified")))))
        return out


def symbolic(name, rem="vertex"):
    """Returns dict: sys, X (indeterminates), nx, fy (nominal field of y = S x), vertices (list of fields),
    cy (constraints > 0), ag (g >= 0), ah (h = 0), remainder description."""
    if name not in ("pendulum", "power"):
        sys_, X, fy, cy, ag, ah = SOS0.symbolic(name)
        return dict(sys=sys_, X=X, nx=sys_.n, fy=fy, vertices=[fy], cy=cy, ag=ag, ah=ah, remainder=None, n_sat=len(ag))
    sys_ = S.get(name)
    n = sys_.n
    Y = sp.symbols(f"y1:{n + 1}", real=True)
    s = [float(v) for v in sys_.s.cpu()]
    x = [Y[i] / s[i] for i in range(n)]
    X = list(Y)
    ag, ah, cons = [], [], []
    r = sp.Symbol("r", real=True)
    t = x[0]
    if name == "pendulum":
        k1, k2, um = 4.4142, 2.3163, 0.8
        u = -(k1 * x[0] + k2 * x[1])
        w = sp.Symbol("w", real=True)
        X.append(w)
        T5 = t - t ** 3 / 6 + t ** 5 / 120
        f0 = [x[1], T5 - x[1] + w]
        gr = [0, 1]
        ag += [um ** 2 - w ** 2, (u - w) * w]
        ah += [(u - w) * (um ** 2 - w ** 2)]
        rdesc = "sin(x1) = T5(x1) + r, |r| <= x1^6/720"
    else:
        d = math.pi / 3
        T5d = sum((math.sin(d) if k % 2 == 0 else math.cos(d)) * (-1) ** (k // 2) * t ** k / math.factorial(k)
                  for k in range(6))
        f0 = [x[1], -0.5 * x[1] - (T5d - math.sin(d))]
        gr = [0, -1]
        rdesc = "sin(x1 + pi/3) = T5d(x1) + r, |r| <= x1^6/720"
    n_sat = len(ag)
    if rem == "sproc":
        X.append(r)
    X = tuple(X)
    bnd = t ** 6 / 720
    lo, hi = [float(v) for v in sys_.lo.cpu()], [float(v) for v in sys_.hi.cpu()]
    cons += [x[i] - lo[i] for i in range(n)] + [hi[i] - x[i] for i in range(n)]
    fy = [from_sympy(s[i] * f0[i], X) for i in range(n)]
    bp = from_sympy(bnd, X)
    gy = [s[i] * gr[i] for i in range(n)]
    if rem == "none":
        vertices = [fy]
    elif rem == "vertex":
        vertices = [[padd(fy[i], pscale(bp, sg * gy[i])) for i in range(n)] for sg in (+1, -1)]
    else:
        rp = from_sympy(r, X)
        vertices = [[padd(fy[i], pscale(rp, gy[i])) for i in range(n)]]
        ag = ag + [padd(bp, pscale(rp, -1)), padd(bp, rp)]
    return dict(sys=sys_, X=X, nx=n, fy=fy, vertices=vertices, cy=[from_sympy(c, X) for c in cons],
                ag=[from_sympy(g, X) if not isinstance(g, dict) else g for g in ag],
                ah=[from_sympy(h, X) for h in ah], remainder=dict(desc=rdesc, mode=rem, bound="x1^6/720"), n_sat=n_sat)


def vdot(V, fy, nx):
    """grad V . f; V numeric dict or LP."""
    if isinstance(V, LP):
        out = LP()
        for i in range(nx):
            out = out + V.diff(i).mul(fy[i])
        return out
    return padd(*[pmul(pdiff(V, i), fy[i]) for i in range(nx)])


def run(name, a, log=print):
    P = symbolic(name, a.rem)
    sys_, X, nx, vert, cy, ag, ah = P["sys"], P["X"], P["nx"], P["vertices"], P["cy"], P["ag"], P["ah"]
    n = len(X)
    zero = tuple([0] * n)
    one = {zero: 1.0}
    state_only = lambda e: all(k == 0 for k in e[nx:])
    dV = a.dV or (6 if nx == 2 else 4)
    nrm = {tuple(2 if j == i else 0 for j in range(n)): 1.0 for i in range(nx)}
    sc = sys_.s.double().cpu().numpy()
    from scipy.linalg import solve_continuous_lyapunov
    A = np.diag(sc) @ sys_.jacobian0() @ np.diag(1 / sc)
    if a.init == "y":
        Pm = solve_continuous_lyapunov(A.T, -np.eye(nx))
    else:
        Ax = sys_.jacobian0()
        Px = solve_continuous_lyapunov(Ax.T, -np.eye(nx))
        Pm = np.diag(1 / sc) @ Px @ np.diag(1 / sc)
    V0 = {}
    for i, j in itertools.product(range(nx), repeat=2):
        e = tuple((1 if k == i else 0) + (1 if k == j else 0) for k in range(n))
        V0[e] = V0.get(e, 0.0) + Pm[i, j]
    if a.shape == "lin":
        p = pscale(V0, 1.0 / max(np.linalg.eigvalsh(Pm)))
        shape_desc = "linearization quadratic y'Py (A'P+PA=-I in y), scaled to unit largest eigenvalue"
    else:
        p = {tuple(2 if j == i else 0 for j in range(n)): 1.0 / sc[i] ** 2 for i in range(nx)}
        shape_desc = "x'x in the original coordinates"
    ddeg = max(deg(f) for f in P["fy"])
    d2 = a.s2deg or (max(0, ddeg - 1) + 2 + (max(0, ddeg - 1) % 2))
    s1deg = a.s1deg if a.s1deg is not None else dV - 2
    has_aux = n > nx
    settings = dict(V_degree=dV, V_monomials=f"degrees 2..{dV} in the state", shape=shape_desc,
                    init=("quadratic Lyapunov function of the linearization, A'P+PA=-I in "
                          + ("box-normalized y" if a.init == "y" else "original x")),
                    s1=f"SOS, degrees 0..{s1deg}", s2=f"SOS, degrees 2..{d2}" + ((" in (state, aux)" if a.s2aux else " in the state only") if has_aux else ""),
                    s3=f"SOS, degrees 0..{a.s3deg} in the state" if (a.box and cy) else None,
                    aux_g_multipliers=(f"SOS, degrees 0..{a.auxdeg}" if P["n_sat"] else None),
                    remainder_multipliers=(f"SOS in (state, r), degrees 0..{a.remdeg}" if a.rem == "sproc" and P["remainder"] else None),
                    aux_h_multipliers=f"free, degrees 0..{a.lamdeg}" if ah else None,
                    eps=EPS, level_backoff=0.98, bisection_steps=14, beta_slack=0.98,
                    stall=a.stall, max_iter=a.iters, time_cap_s=a.cap, solver="Clarabel only, status 'optimal'",
                    box_containment=bool(a.box), remainder=P["remainder"],
                    vertices=len(vert))
    cons_used = cy if a.box else [c for c in cy[:len(cy) - 2 * nx]]
    lo_, hi_ = sys_.lo.double().cpu().numpy(), sys_.hi.double().cpu().numpy()
    bounds = None
    if a.box:
        bounds = [float(max(abs(lo_[i]), abs(hi_[i])) * sc[i]) for i in range(nx)]
        bounds += [0.8 if str(v) == "w" else float("inf") for v in X[nx:]]

    def aux_terms(pr, k):
        out = []
        for j, gj in enumerate(ag):
            dj = a.auxdeg if j < P["n_sat"] else a.remdeg
            out.append(pr.sos_var(0, dj, f"sg{k}_{j}").mul(gj))
        for j, hj in enumerate(ah):
            lam, _ = LP.var(n, 0, a.lamdeg, f"lh{k}_{j}")
            out.append(lam.mul(hj))
        return out

    def decrease(pr, V, level, s2s, label):
        """-[(level - V) s2 + dV/dt + eps|y|^2 + aux] SOS at every vertex; V numeric or LP; s2s numeric or None."""
        made = []
        for k, fv in enumerate(vert):
            s2 = s2s[k] if s2s is not None else pr.sos_var(2, max(2, d2), f"s2_{k}", keep=None if a.s2aux else state_only)
            made.append(s2)
            if isinstance(V, LP):
                lev = (V.scale(-1) + {zero: level}).mul(s2)
            else:
                lev = as_lp(s2).mul(padd({zero: level}, pscale(V, -1))) if isinstance(s2, LP) else LP(pmul(s2, padd({zero: level}, pscale(V, -1))))
            e = lev + as_lp(vdot(V, fv, nx)) + LP(pscale(nrm, EPS))
            for t_ in aux_terms(pr, k):
                e = e + t_
            pr.sos(e.scale(-1), lo=0 if ag else 1, label=f"{label} decrease vertex {k}")
        return made

    def contain(pr, V, level, s3s, label):
        made = []
        for j, c in enumerate(cons_used):
            s3 = s3s[j] if s3s is not None else pr.sos_var(0, a.s3deg, f"s3_{j}", keep=state_only)
            made.append(s3)
            if isinstance(V, LP):
                e = (V.scale(-1) + {zero: level}).mul(s3).scale(-1) + padd(c, {zero: -EPS})
            elif isinstance(s3, LP):
                e = s3.mul(padd({zero: level}, pscale(V, -1))).scale(-1) + padd(c, {zero: -EPS})
            else:
                e = LP(padd(c, {zero: -EPS}, pscale(pmul(s3, padd({zero: level}, pscale(V, -1))), -1)))
            pr.sos(e, lo=0, label=f"{label} containment {j}")
        return made

    def level_prog(V, g):
        pr = Program(n)
        s2 = decrease(pr, V, g, None, "level")
        s3 = contain(pr, V, g, None, "level")
        return pr, s2, s3

    def level_ok(V, g):
        pr, _, _ = level_prog(V, g)
        return pr.solve(tag="level")

    def max_level(V, hi=1.0):
        lo_, hi_ = 0.0, hi
        while level_ok(V, hi_) and hi_ < 1e6:
            lo_, hi_ = hi_, 2 * hi_
        for _ in range(14):
            m = 0.5 * (lo_ + hi_)
            if level_ok(V, m):
                lo_ = m
            else:
                hi_ = m
        return lo_

    def multipliers(V):
        pr, s2, s3 = level_prog(V, 1.0)
        if not pr.solve(tag="mult"):
            return None
        s2n, s3n = [q.value() for q in s2], [q.value() for q in s3]

        def beta_ok(b):
            pr = Program(n)
            s1 = pr.sos_var(0, s1deg, "s1", keep=state_only)
            e = s1.mul(padd({zero: b}, pscale(p, -1))) + padd(V, {zero: -1.0})
            pr.sos(e.scale(-1), lo=0, label="shape")
            return s1.value() if pr.solve(tag="beta") else None
        lo_, hi_, s1n = 0.0, 1.0, None
        while beta_ok(hi_) is not None and hi_ < 1e3:
            lo_, hi_ = hi_, 2 * hi_
        for _ in range(14):
            m = 0.5 * (lo_ + hi_)
            r_ = beta_ok(m)
            if r_ is not None:
                lo_, s1n = m, r_
            else:
                hi_ = m
        s1n = beta_ok(0.98 * lo_) or s1n
        return s1n, s2n, s3n, lo_

    def V_step(s1, s2, s3):
        pr = Program(n)
        Vv, _ = LP.var(n, 2, dV, "V", keep=state_only)
        b = cp.Variable(name="beta")
        pr.sos(Vv + LP(pscale(nrm, -EPS)), lo=1, label="V positive")
        e = LP.scalar_times(b, "beta", s1) + LP(pscale(pmul(p, s1), -1)) + Vv + {zero: -1.0}
        pr.sos(e.scale(-1), lo=0, label="shape")
        decrease(pr, Vv, 1.0, s2, "Vstep")
        contain(pr, Vv, 1.0, s3, "Vstep")
        if not pr.solve(b, tag="V"):
            return None, None
        return Vv.value(), float(b.value)

    t0 = time.time()
    if a.fixedV:
        Vf = {tuple(e): float(c) for e, c in json.loads(Path(a.fixedV).read_text())["V"]}
        settings["fixedV"] = str(a.fixedV)
        return dict(V=Vf, hist=[], settings=settings, stop="fixed V (no iteration)", seconds=0.0,
                    program=dict(n=n, nx=nx, X=[str(v) for v in X]), P=P,
                    funcs=dict(decrease=decrease, contain=contain, nrm=nrm, zero=zero), bounds=bounds)
    g = max_level(V0)
    log(f"  {name}: initial quadratic certified level {g:.5g}  [{time.time() - t0:.0f} s]")
    if g <= 0:
        return dict(V=None, hist=[], settings=settings, stop="initial level not certified", seconds=time.time() - t0)
    V = pscale(V0, 1.0 / (0.98 * g))
    best_beta, hist, stall, stop = 0.0, [], 0, f"max iterations ({a.iters})"
    best = (-1.0, pscale(V, 0.98), -1)
    for it in range(a.iters):
        m = multipliers(V)
        retries = 0
        while m is None and retries < 6:
            V = pscale(V, 1 / 0.9); retries += 1
            m = multipliers(V)
        if m is None or m[0] is None:
            stop = "multiplier step infeasible"
            log(f"  {name} iter {it}: multiplier step infeasible")
            break
        s1, s2, s3, beta = m
        if beta > best[0]:
            best = (beta, pscale(V, 0.98), it)
        Vn, bn = V_step(s1, s2, s3)
        hist.append(dict(iter=it, beta=beta, beta_V=bn, retries=retries, seconds=time.time() - t0))
        log(f"  {name} iter {it}: beta {beta:.5f}  (V step beta {bn})  [{time.time() - t0:.0f} s]")
        if Vn is None:
            stop = "V step infeasible"
            break
        g = max_level(Vn, hi=1.0)
        if g <= 0:
            stop = "new V: no certified level"
            break
        V = pscale(Vn, 1.0 / (0.98 * g))
        stall = stall + 1 if beta <= best_beta * (1 + 1e-3) else 0
        best_beta = max(best_beta, beta)
        if a.stall and stall >= a.stall:
            stop = f"stall: {a.stall} iterations without a 0.1 % gain in beta"
            break
        if time.time() - t0 > a.cap:
            stop = f"time cap {a.cap:.0f} s"
            break
    if best[2] < 0:
        best = (0.0, pscale(V, 0.98), -1)
    settings["returned"] = "the iterate with the largest beta (multiplier step), scaled so that level 1 is its certified level"
    return dict(V=best[1], best_iter=best[2], best_beta=best[0], hist=hist, settings=settings, stop=stop, seconds=time.time() - t0,
                program=dict(n=n, nx=nx, X=[str(v) for v in X]), P=P,
                funcs=dict(decrease=decrease, contain=contain, nrm=nrm, zero=zero), bounds=bounds)


TIGHT = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


def recertify(res, n, nx, levels=(1.0, 0.999, 0.99, 0.98, 0.95, 0.9)):
    """Fixed V: positivity, decrease at every vertex and containment at level c.  For each level from 1 down: Clarabel
    with default tolerances, then with tolerances 1e-10, then (tight) with facial reduction (basis monomials whose
    Gram diagonal is numerically zero are removed, two rounds).  The first level with status 'optimal' and every Gram
    block certified (see Program.gram_check) is the certified level."""
    V = res["V"]
    f = res["funcs"]
    R = res["bounds"]
    tried = []
    saved = dict(SOLVER_OPTS)
    r = None
    for c in levels:
        for mode in ("default", "tight", "tight+facial", "tight+facial2"):
            SOLVER_OPTS.clear(); SOLVER_OPTS.update(saved)
            if mode != "default":
                SOLVER_OPTS.update(TIGHT)
            if mode == "tight+facial" or mode == "tight+facial2":
                if r is None or not r["accepted"]:
                    continue
                for g in last.grams:
                    Q = np.asarray(g["Q"].value)
                    dmax = max(1e-300, float(np.abs(np.diag(Q)).max()))
                    tiny = [g["z"][i] for i in range(len(Q)) if Q[i, i] < 1e-9 * dmax]
                    if tiny:
                        drop[g["label"]] = sorted(set(drop.get(g["label"], ())) | set(tiny))
            else:
                drop = {}
            pr = Program(n, drop)
            pr.sos(LP(padd(V, pscale(f["nrm"], -EPS))), lo=1, label="V positive")
            f["decrease"](pr, V, c, None, "recert")
            f["contain"](pr, V, c, None, "recert")
            ok = pr.solve(tag="recert")
            last = pr
            checks = pr.gram_check(R) if ok else []
            r = dict(level=c, mode=mode, status=pr.status, accepted=ok, gram_checks=checks,
                     dropped={k: [list(e) for e in v] for k, v in drop.items()},
                     all_exact=bool(ok and all(g["exact"] for g in checks)),
                     all_certified=bool(ok and all(g["certified"] for g in checks)),
                     min_lam_Q_plus_E=min((g["lam_min_Q_plus_E"] for g in checks if not g["label"].startswith("multiplier")), default=None))
            tried.append({k: v for k, v in r.items() if k not in ("gram_checks", "dropped")})
            if r["all_certified"]:
                SOLVER_OPTS.clear(); SOLVER_OPTS.update(saved)
                r["tried"] = tried
                return r
    SOLVER_OPTS.clear(); SOLVER_OPTS.update(saved)
    r["tried"] = tried
    r["level"] = None
    return r


def sample_check(name, sys_, V, nx, N=4_000_000, seed=7, enlarge=1.25):
    """True dynamics on N uniform samples of the box enlarged by `enlarge` about its center: in {V <= 1}, count
    dV/dt >= 0 (away from the origin), V <= 0 away from the origin, and points outside the admissible set."""
    g = np.random.default_rng(seed)
    lo, hi = sys_.lo.double().cpu().numpy(), sys_.hi.double().cpu().numpy()
    cen, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    sc = sys_.s.double().cpu().numpy()
    Vs = {e[:nx]: c for e, c in V.items()}
    grads = [{e[:nx]: c for e, c in pdiff(V, i).items()} for i in range(nx)]
    cnt = Counter()
    vmax_dot = -np.inf
    for _ in range(N // 500_000):
        x = cen + enlarge * half * (2 * g.random((500_000, nx)) - 1)
        y = x * sc
        v = evaluate(Vs, y)
        inside = v <= 1.0
        cnt["samples"] += len(x)
        cnt["in_set"] += int(inside.sum())
        xi, yi = x[inside], y[inside]
        fx = sys_.f(torch.as_tensor(xi, dtype=torch.float64)).numpy()
        vd = sum(evaluate(grads[i], yi) * sc[i] * fx[:, i] for i in range(nx))
        away = np.linalg.norm(yi, axis=1) > 1e-6
        cnt["decrease_violations"] += int((vd[away] >= 0).sum())
        vmax_dot = max(vmax_dot, float((vd[away] / (yi[away] ** 2).sum(1)).max()) if away.any() else -np.inf)
        cnt["nonpositive_V"] += int(((v <= 0) & (np.linalg.norm(y, axis=1) > 1e-6)).sum())
        adm = sys_.admissible(torch.as_tensor(xi, dtype=torch.float32)).numpy()
        cnt["outside_admissible"] += int((~adm).sum())
        t = xi[:, 0]
        if name == "pendulum":
            err = np.abs(np.sin(t) - (t - t ** 3 / 6 + t ** 5 / 120))
        elif name == "power":
            d = math.pi / 3
            T5d = sum((math.sin(d) if k % 2 == 0 else math.cos(d)) * (-1) ** (k // 2) * t ** k / math.factorial(k)
                      for k in range(6))
            err = np.abs(np.sin(t + d) - T5d)
        else:
            err = np.zeros_like(t)
        cnt["remainder_bound_violations"] += int((err > t ** 6 / 720 + 1e-15).sum()) if name in ("pendulum", "power") else 0
    out = dict(cnt)
    out["max_Vdot_over_|y|^2"] = vmax_dot
    out["region"] = f"box enlarged x{enlarge} about its center, uniform, seed {seed}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--rem", default="vertex", choices=["vertex", "sproc", "none"])
    ap.add_argument("--dV", type=int, default=None)
    ap.add_argument("--s1deg", type=int, default=None)
    ap.add_argument("--s2deg", type=int, default=None)
    ap.add_argument("--s3deg", type=int, default=2)
    ap.add_argument("--auxdeg", type=int, default=2)
    ap.add_argument("--lamdeg", type=int, default=2)
    ap.add_argument("--s2aux", type=int, default=0, help="1: the decrease multiplier s2 also depends on w (and r)")
    ap.add_argument("--remdeg", type=int, default=6, help="degree of the remainder-bound multipliers (--rem sproc)")
    ap.add_argument("--shape", default="lin", choices=["lin", "id"])
    ap.add_argument("--init", default="y", choices=["y", "x"])
    ap.add_argument("--box", type=int, default=1)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--stall", type=int, default=3)
    ap.add_argument("--cap", type=float, default=6 * 3600.0)
    ap.add_argument("--samples", type=int, default=4_000_000)
    ap.add_argument("--fixedV", default=None, help="re-certify the V stored in this result file instead of iterating")
    ap.add_argument("--update", type=int, default=0, help="with --fixedV FILE --out FILE: replace the certification, "
                    "evaluation and sample check of FILE, keep its iteration history")
    ap.add_argument("--out", default=None, help="output file (default results/baseline_sos_sound_<sys>.json)")
    ap.add_argument("--solve-limit", type=float, default=0.0, help="Clarabel time_limit per solve in s (0: none); "
                    "a solve that hits it counts as infeasible")
    ap.add_argument("--lenient", type=int, default=0, help="1: the iteration (not the re-certification) also accepts "
                    "'optimal_inaccurate'")
    a = ap.parse_args()
    global LENIENT
    LENIENT = bool(a.lenient)
    if a.solve_limit:
        SOLVER_OPTS["time_limit"] = a.solve_limit
    torch.set_num_threads(1)
    for name in a.systems:
        STATUS.clear(); TIMES.clear()
        t0 = time.time()
        res = run(name, a)
        sys_ = res["P"]["sys"] if "P" in res else S.get(name)
        out = dict(method="sos_sound", system=sys_.name, settings=res["settings"], stop=res["stop"],
                   iterations=res["hist"], n_iterations=len(res["hist"]), iteration_seconds=res["seconds"],
                   argv=vars(a))
        if res["V"] is not None:
            n, nx = res["program"]["n"], res["program"]["nx"]
            t1 = time.time()
            rc = recertify(res, n, nx)
            out["recertification"] = rc
            out["recert_seconds"] = time.time() - t1
            print(f"  recert: level {rc['level']} ({rc.get('mode')}) status {rc['status']}, certified {rc['all_certified']} "
                  f"(exact SOS {rc['all_exact']}), min eig(Q+E) {rc['min_lam_Q_plus_E']}", flush=True)
            if rc["level"] is None:
                print("  recert FAILED at every level; coverage of the uncertified level 1 reported", flush=True)
            elif rc["level"] < 1.0:
                res["V"] = pscale(res["V"], 1.0 / rc["level"])
            out["best_iter"], out["best_beta"] = res.get("best_iter"), res.get("best_beta")
            cov, fp = SOS0.evaluate_region(sys_, res["V"])
            out.update(coverage=cov, false_positive=fp, degree=max(sum(e) for e in res["V"]),
                       V=[[list(e), float(c)] for e, c in res["V"].items()], indeterminates=res["program"]["X"])
            if a.samples:
                out["sample_check"] = sample_check(name, sys_, res["V"], nx, N=a.samples)
                print(f"  sample check: {out['sample_check']}", flush=True)
        out["solver_status_counts"] = dict(STATUS)
        out["solver_seconds"] = dict(TIMES)
        out["seconds"] = time.time() - t0
        path = Path(a.out) if a.out else OUT / f"baseline_sos_sound_{name}.json"
        if a.update:
            old = json.loads(path.read_text())
            for k in ("recertification", "recert_seconds", "coverage", "false_positive", "V", "sample_check"):
                if k in out:
                    old[k] = out[k]
            old["recert_status_counts"] = out["solver_status_counts"]
            old.setdefault("recert_history", []).append(f"re-certified in place by sos_sound.py --fixedV --update ({time.ctime()})")
            out = old
        path.write_text(json.dumps(out, indent=1))
        print(f"sos_sound {sys_.name:9s} coverage {out.get('coverage', float('nan')):6.2f}%  FP {out.get('false_positive', float('nan')):.3f}%  "
              f"iters {len(res['hist'])}  stop: {res['stop']}  [{out['seconds']:.0f} s] -> {path}", flush=True)


if __name__ == "__main__":
    main()
