r"""
Symbolic pieces of the TFL closed loop for the formal check, compiled (with common-subexpression
elimination) to functions on affine forms:

  F = f_cl / sqrt(1 + ||S f_cl||^2)        the normalized closed-loop field the certificate steps with
  G = DF . F,  R = DG . F                  second and third time derivatives of a solution (Taylor terms)
  reg = (det D, det DT, 1 / ||grad pi||)   the regularity functions of Omega_ad
  zs  = (alpha, L_f alpha, L_f pi)         transverse / tangential coordinates (xi1, xi2, eta2)

Same construction as tfl_system.TFLClosedLoop, from the same strings and constants.
"""
from __future__ import annotations

import math
import sys as _sys
from pathlib import Path

import sympy as sp

_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unicycle"))
from tfl_system import X_SYM, Y_SYM, parse_planar

import affine


def build(sys):
    X = sp.symbols("x1 x2 x3 x4", real=True)
    x1, x2, x3, x4 = X
    f = [(sys.vbar + x4) * sp.cos(x3), (sys.vbar + x4) * sp.sin(x3), 0, 0]
    Gm = [[0, 0], [0, 0], [0, 1], [1, 0]]
    to_x = {X_SYM: x1, Y_SYM: x2}
    al = parse_planar(sys.path_str).subs(to_x)
    ph = parse_planar(sys.pi_str).subs(to_x)
    lie = lambda h, v: sum(sp.diff(h, s) * vi for s, vi in zip(X, v))
    La, Lp = lie(al, f), lie(ph, f)
    L2a, L2p = lie(La, f), lie(Lp, f)
    col = lambda j: [Gm[i][j] for i in range(4)]
    D = [[lie(Lp, col(0)), lie(Lp, col(1))], [lie(La, col(0)), lie(La, col(1))]]
    r1 = -L2p + sys.k3 * (Lp - sys.eta2_ref)
    r2 = -L2a + sys.k1 * al + sys.k2 * La
    det = D[0][0] * D[1][1] - D[0][1] * D[1][0]
    u1 = (D[1][1] * r1 - D[0][1] * r2) / det
    u2 = (-D[1][0] * r1 + D[0][0] * r2) / det
    fc = [sp.simplify(e) for e in (f[0], f[1], u2, u1)]
    s = [sp.nsimplify(float(v)) if abs(float(v) - 1 / math.pi) > 1e-9 else 1 / sp.pi for v in sys.s.cpu()]
    nrm = sp.sqrt(1 + sum((fi * si) ** 2 for fi, si in zip(fc, s)))
    F = [fi / nrm for fi in fc]
    G = [lie(e, F) for e in F]
    R = [lie(e, F) for e in G]
    T = [ph, Lp, al, La]
    DT = sp.Matrix(4, 4, [sp.diff(t, v) for t in T for v in X])
    gp = [sp.diff(ph, x1), sp.diff(ph, x2)]
    reg = [sp.simplify(det), sp.simplify(DT.det(method="berkowitz")), sp.simplify(1 / sp.sqrt(gp[0] ** 2 + gp[1] ** 2))]
    zs = [sp.simplify(al), sp.simplify(La), sp.simplify(Lp)]
    return dict(F=F, G=G, R=R, reg=reg, z=zs), X


def compile_af(exprs, X):
    fn = sp.lambdify(X, list(exprs), modules=[affine.MODULE], cse=True)

    def call(xs):
        out = fn(*xs)
        return [o if isinstance(o, affine.AF) else affine.AF.const(float(o), xs[0]) for o in out]
    return call


def compile_torch(exprs, X):
    fn = sp.lambdify(X, list(exprs), modules="torch", cse=True)

    def call(x):
        import torch
        out = fn(*x.unbind(-1))
        return torch.stack([o if torch.is_tensor(o) else torch.full_like(x[:, 0], float(o)) for o in out], -1)
    return call
