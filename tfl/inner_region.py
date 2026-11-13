r"""
Interval verification of the inner layer (interval branch and bound; paper, Appendix "Interval Verification").

Claim verified:   every point x of the closed box with alpha q_ex(x) < c violates no regularity
constraint (thresholds on det D, det DT, 1/||grad pi||) and meets the boundary of the box only where the closed-loop
field points strictly into the box.  Since D q_ex . f = -zeta^T Q zeta < 0 off the target set M on Omega_ad (the
transverse dynamics are exactly linear there), a trajectory that starts in I_c and in the open box stays in I_c, cannot
reach the box boundary (it could only leave through a point where the field points outward) and converges to M.  So
I_c (intersected with the open box) is an inner approximation of the constrained region of attraction, and
{v < 1 - e^{-c}} = {V < c} lies inside it because V >= alpha q_ex.

Method: the closed box is split into cells; a cell is discarded when interval arithmetic shows that alpha q_ex > c on it
(it holds no point of I_c), or that it holds no violating point: all regularity distances > 0 and, if it touches a face
of the box, the field component normal to that face points inward on the whole cell.  Remaining cells are bisected along their widest normalized side.  If every cell is
discarded, the claim is verified.  Interval arithmetic is implemented in float64 with outward rounding by two units in
the last place after every operation (numpy's sin, cos and log1p are accurate to about one unit in the last place);
the scale factors are enclosed by outward rounding, and the verified expressions contain no irrational constant (a symbolic pi raises an error).  Residual float effects (the eigen decomposition of P, float32 storage
of P in the pipeline) are at the 1e-6 relative level and are absorbed by the 1 % safety factor on the level.

    python -u inner_region.py --system unicycle            # circle
    python -u inner_region.py --system unicycle --path "y-sin(x)" --pi x --periodic-x1 true
    python -u inner_region.py --system quadrotor
The largest verified c is found by bisection (12 rounds); results go to results/inner_<case>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys as _sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from problem import ConstrainedZubov, ProblemConfig

_sys.path.insert(0, str(Path(__file__).resolve().parent / "formal"))

INF = float("inf")


def _out(lo, hi):
    """Outward rounding by two ulps on each side."""
    lo = np.nextafter(np.nextafter(lo, -INF), -INF)
    hi = np.nextafter(np.nextafter(hi, INF), INF)
    return lo, hi


class IV:
    """Vectorized closed intervals [lo, hi] (numpy float64 arrays of equal shape)."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    @staticmethod
    def const(c, like):
        c = float(c)
        return IV(np.full_like(like.lo, c), np.full_like(like.lo, c))

    def _lift(self, o):
        return o if isinstance(o, IV) else IV.const(o, self)

    def __add__(self, o):
        o = self._lift(o)
        return IV(*_out(self.lo + o.lo, self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return IV(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        if not isinstance(o, IV):
            c = float(o)
            if c == 0.0:
                return IV(np.zeros_like(self.lo), np.zeros_like(self.lo))
            a, b = self.lo * c, self.hi * c
            return IV(*_out(np.minimum(a, b), np.maximum(a, b)))
        with np.errstate(invalid="ignore"):
            p = np.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
        inputs_ok = ~(np.isnan(self.lo) | np.isnan(self.hi) | np.isnan(o.lo) | np.isnan(o.hi))
        p = np.where(np.isnan(p) & inputs_ok[None], 0.0, p)
        return IV(*_out(p.min(0), p.max(0)))

    __rmul__ = __mul__

    def recip(self):
        bad = (self.lo <= 0) & (self.hi >= 0)
        with np.errstate(divide="ignore"):
            lo, hi = 1.0 / self.hi, 1.0 / self.lo
        lo, hi = _out(lo, hi)
        return IV(np.where(bad, -INF, lo), np.where(bad, INF, hi))

    def __truediv__(self, o):
        if not isinstance(o, IV):
            c = float(o)
            a, b = self.lo / c, self.hi / c
            return IV(*_out(np.minimum(a, b), np.maximum(a, b)))
        return self * o.recip()

    def __rtruediv__(self, o):
        return self._lift(o) * self.recip()

    def __pow__(self, n):
        if isinstance(n, IV):
            raise NotImplementedError("interval exponent")
        if float(n) == 0.5:
            return isqrt(self)
        if float(n) != int(n):
            raise NotImplementedError(f"non-integer exponent {n}")
        n = int(n)
        if n < 0:
            return (self ** (-n)).recip()
        if n == 0:
            return IV.const(1.0, self)
        if n % 2 == 1:
            return IV(*_out(self.lo ** n, self.hi ** n))
        a, b = self.lo ** n, self.hi ** n
        lo = np.where(self.lo >= 0, a, np.where(self.hi <= 0, b, 0.0))
        hi = np.maximum(a, b)
        return IV(*_out(lo, hi))

    def width(self):
        return self.hi - self.lo


def isqrt(x):
    lo, hi = np.sqrt(np.maximum(x.lo, 0.0)), np.sqrt(np.maximum(x.hi, 0.0))
    return IV(*_out(lo, hi))


def iabs(x):
    lo = np.where(x.lo >= 0, x.lo, np.where(x.hi <= 0, -x.hi, 0.0))
    hi = np.maximum(-x.lo, x.hi)
    return IV(lo, hi)


def irelu(x):
    return IV(np.maximum(x.lo, 0.0), np.maximum(x.hi, 0.0))


def ilog1p(x):
    return IV(*_out(np.log1p(np.maximum(x.lo, -1.0)), np.log1p(np.maximum(x.hi, -1.0))))


def isin(x):
    lo, hi = x.lo, x.hi
    s = np.stack([np.sin(lo), np.sin(hi)])
    smin, smax = s.min(0), s.max(0)
    has_max = np.ceil((lo - math.pi / 2) / (2 * math.pi)) <= np.floor((hi - math.pi / 2) / (2 * math.pi))
    has_min = np.ceil((lo + math.pi / 2) / (2 * math.pi)) <= np.floor((hi + math.pi / 2) / (2 * math.pi))
    wide = hi - lo >= 2 * math.pi
    smin = np.where(has_min | wide, -1.0, smin)
    smax = np.where(has_max | wide, 1.0, smax)
    lo2, hi2 = _out(smin, smax)
    return IV(np.maximum(lo2, -1.0), np.minimum(hi2, 1.0))


def icos(x):
    lo, hi = x.lo, x.hi
    c = np.stack([np.cos(lo), np.cos(hi)])
    cmin, cmax = c.min(0), c.max(0)
    has_max = np.ceil(lo / (2 * math.pi)) <= np.floor(hi / (2 * math.pi))
    has_min = np.ceil((lo - math.pi) / (2 * math.pi)) <= np.floor((hi - math.pi) / (2 * math.pi))
    wide = hi - lo >= 2 * math.pi
    cmin = np.where(has_min | wide, -1.0, cmin)
    cmax = np.where(has_max | wide, 1.0, cmax)
    lo2, hi2 = _out(cmin, cmax)
    return IV(np.maximum(lo2, -1.0), np.minimum(hi2, 1.0))


PI_IV = (np.nextafter(math.pi, 0.0), np.nextafter(math.pi, 4.0))


def _atan2_unsupported(*a):
    raise NotImplementedError("atan2 does not occur in the verified expressions")


IV_MODULE = {"sin": isin, "cos": icos, "sqrt": isqrt, "Abs": iabs, "pi": None, "atan2": _atan2_unsupported,
             "log1p": ilog1p, "Max": lambda a, b: IV(np.maximum(a.lo, b.lo), np.maximum(a.hi, b.hi)),
             "Min": lambda a, b: IV(np.minimum(a.lo, b.lo), np.minimum(a.hi, b.hi))}


def compile_af(exprs, X):
    import affine
    fn = sp.lambdify(X, list(exprs), modules=[affine.MODULE], cse=True)

    def call(cells):
        out = fn(*cells)
        return [o if isinstance(o, affine.AF) else affine.AF.const(float(o), cells[0]) for o in out]
    return call


def compile_iv(exprs, X):
    mod = dict(IV_MODULE)
    fn = sp.lambdify(X, list(exprs), modules=[mod], cse=True)

    def call(cells):
        mod["pi"] = IV(np.full_like(cells[0].lo, PI_IV[0]), np.full_like(cells[0].lo, PI_IV[1]))
        out = fn(*cells)
        return [o if isinstance(o, IV) else IV.const(float(o), cells[0]) for o in out]
    return call


class Pieces:
    """Interval versions of the signed constraint distances and of alpha q_ex for a ConstrainedZubov problem.
    engine = "interval" (plain intervals with outward rounding) or "affine" (first-order affine arithmetic of
    formal/affine.py: the linear dependence on the cell coordinates is kept exact, so products of correlated
    quantities are bounded much more tightly; its rounding model is a relative inflation of 1e-13 per operation)."""

    def __init__(self, prob: ConstrainedZubov, engine="interval"):
        self.prob, s = prob, prob.sys
        self.engine = engine
        self.alpha = float(prob.alpha)
        self.zs = s.z_scale.cpu().double().numpy()
        comp = compile_iv if engine == "interval" else compile_af
        if prob.cfg.system == "quadrotor":
            self.X = s.sym["X"]
            self.reg = comp(s.sym["reg"], self.X)
            self.zf = comp(s.sym["z"], self.X)
            self.field = comp(s.sym["fcl"], self.X)
            self.faces = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (5, -1), (5, 1)]
            self.split_groups = [[0, 1, 2, 3], [4, 5]]
            self.n = 6
            self.lo = np.array([-s.L, -s.L, -s.v_max, -s.v_max, -math.pi, 0.0])
            self.hi = np.array([s.L, s.L, s.v_max, s.v_max, math.pi, s.T_max])
            self.periodic = (4,)
            P = prob.P.cpu().double().numpy()
            self.evals, self.evecs = np.linalg.eigh(0.5 * (P + P.T))
            self.kind = "quadratic"
            self.offsets = [0.0] * 5
        else:
            import tflsym
            d, X = tflsym.build(s)
            self.X = X
            self.reg = comp(d["reg"], X)
            self.zf = comp(d["z"], X)
            self.field = comp(d["F"], X)
            self.faces = ([] if s.periodic_x1 else [(0, -1), (0, 1)]) + [(1, -1), (1, 1), (3, -1), (3, 1)]
            self.split_groups = [[0, 1, 2, 3]]
            self.n = 4
            self.lo = np.array([-s.Lx, -s.L, -math.pi, s.x4_lo])
            self.hi = np.array([s.Lx, s.L, math.pi, s.x4_hi])
            self.periodic = (0, 2) if s.periodic_x1 else (2,)
            if prob.linear_target:
                P = prob.P.cpu().double().numpy()
                self.evals, self.evecs = np.linalg.eigh(0.5 * (P + P.T))
                self.kind = "quadratic"
                self.offsets = [0.0, 0.0, float(s.eta2_ref)]
            else:
                Pxi = prob.P_xi.cpu().double().numpy()
                self.evals, self.evecs = np.linalg.eigh(0.5 * (Pxi + Pxi.T))
                self.kind = "deadzone"
                self.eta_c, self.eta_hi = prob.cfg.eta_c, prob.cfg.eta_c_hi
                self.cw = float(prob.Q[2, 2].item()) / self.zs[2] ** 2
                self.kap = -s.k3
        self.eps = s.eps

    def cells(self, lo, hi):
        if self.engine == "interval":
            return [IV(lo[:, i], hi[:, i]) for i in range(self.n)]
        import affine
        L = torch.as_tensor(lo, dtype=torch.float64)
        H = torch.as_tensor(hi, dtype=torch.float64)
        return affine.AF.cell(L, H)

    @staticmethod
    def LO(q):
        return q.lo if isinstance(q, IV) else q.lo.numpy()

    @staticmethod
    def HI(q):
        return q.hi if isinstance(q, IV) else q.hi.numpy()

    def const(self, v, like):
        return IV.const(v, like) if isinstance(like, IV) else like.__class__.const(float(v), like)

    def reg_distances(self, cells):
        reg = self.reg(cells)
        eD, eT, eG = self.eps
        return [iabs(reg[0]) - eD, iabs(reg[1]) - eT, reg[2] - eG]

    def may_violate(self, lo, hi, cells):
        """True for cells that may hold a violating point: a regularity distance may be <= 0, or the cell touches a
        box face on which the field is not proved to point strictly inward."""
        bad = np.zeros(len(lo), dtype=bool)
        for di in self.reg_distances(cells):
            bad |= ~(self.LO(di) > 0)
        touch_any = np.zeros(len(lo), dtype=bool)
        for dim, side in self.faces:
            touch = (lo[:, dim] <= self.lo[dim]) if side < 0 else (hi[:, dim] >= self.hi[dim])
            touch_any |= touch
        if touch_any.any():
            f = self.field(cells)
            for dim, side in self.faces:
                touch = (lo[:, dim] <= self.lo[dim]) if side < 0 else (hi[:, dim] >= self.hi[dim])
                inflow = (self.LO(f[dim]) > 0) if side < 0 else (self.HI(f[dim]) < 0)
                bad |= touch & ~inflow
        return bad

    def quad_form(self, zetas):
        """zeta^T P zeta: the eigen form sum_k lambda_k (u_k . zeta)^2 (each square bounded below by 0), with the lower
        bound sharpened by lambda_min sum_i zeta_i^2 (both are valid bounds; the second does not mix coordinates)."""
        q = None
        for k in range(len(self.evals)):
            lin = None
            for i, z in enumerate(zetas):
                if self.evecs[i, k] == 0.0:
                    continue
                t = z * float(self.evecs[i, k])
                lin = t if lin is None else lin + t
            term = (lin ** 2) * float(self.evals[k])
            q = term if q is None else q + term
        sq = None
        for z in zetas:
            t = z ** 2
            sq = t if sq is None else sq + t
        lo2 = self.LO(sq * float(self.evals.min()))
        return IV(np.maximum(self.LO(q), lo2), self.HI(q))

    def alpha_q(self, cells):
        z = self.zf(cells)
        if self.kind == "quadratic":
            zetas = [(z[i] - self.offsets[i] if self.offsets[i] else z[i]) / float(self.zs[i]) for i in range(len(z))]
            return self.quad_form(zetas) * self.alpha
        if self.engine != "interval":
            raise NotImplementedError("dead-zone part only in the interval engine")
        zetas = [z[0] / float(self.zs[0]), z[1] / float(self.zs[1])]
        q = self.quad_form(zetas)
        eta2 = z[2]
        def F(wabs, b):
            u = wabs / float(b)
            return (u ** 2 * 0.5 - u + ilog1p(u)) * (b * b / self.kap)
        w_lo = irelu(IV.const(self.eta_c, eta2) - eta2)
        w_hi = irelu(eta2 - IV.const(self.eta_hi, eta2))
        Flo, Fhi = F(w_lo, 1.0 - self.eta_c), F(w_hi, self.eta_hi - 1.0)
        Fw = IV(np.maximum(np.maximum(Flo.lo, Fhi.lo), 0.0), np.maximum(Flo.hi, Fhi.hi))
        return (q + Fw * self.cw) * self.alpha


def verify(pieces: Pieces, c, min_rel_width=1e-4, max_cells=20_000_000, batch=200000, verbose=False, return_open=False):
    """True if every cell is discarded.  Returns (proved, info)."""
    lo, hi = pieces.lo[None, :].copy(), pieces.hi[None, :].copy()
    half = 0.5 * (pieces.hi - pieces.lo)
    processed, t0 = 0, time.time()
    undecided_small = 0
    min_q_on_constraints = INF
    while len(lo):
        if processed > max_cells:
            info = dict(reason="cell budget exhausted", processed=processed)
            if return_open:
                info["open"] = (lo, hi)
            return False, info
        nlo, nhi = [], []
        for j in range(0, len(lo), batch):
            L, H = lo[j:j + batch], hi[j:j + batch]
            cells = pieces.cells(L, H)
            touch = pieces.may_violate(L, H, cells)
            q = pieces.alpha_q(cells)
            if touch.any():
                min_q_on_constraints = min(min_q_on_constraints, float(np.nan_to_num(q.lo[touch], nan=INF).min()))
            keep = touch & ~(q.lo >= c)
            processed += len(L)
            if not keep.any():
                continue
            L, H = L[keep], H[keep]
            rel = (H - L) / (2 * half)
            small = rel.max(1) < min_rel_width
            if small.any():
                undecided_small += int(small.sum())
                L, H = L[~small], H[~small]
            if len(L) == 0:
                continue
            dim = rel.argmax(1)
            mid = 0.5 * (L + H)
            A, B = L.copy(), H.copy()
            idx = np.arange(len(L))
            A[idx, dim] = mid[idx, dim]
            B[idx, dim] = mid[idx, dim]
            nlo.append(np.concatenate([L, A]))
            nhi.append(np.concatenate([B, H]))
        lo = np.concatenate(nlo) if nlo else np.zeros((0, pieces.n))
        hi = np.concatenate(nhi) if nhi else np.zeros((0, pieces.n))
        if verbose:
            print(f"    c {c:.4f}: {len(lo)} cells open, {processed} processed, {undecided_small} undecided  [{time.time() - t0:.0f} s]", flush=True)
        if undecided_small:
            return False, dict(reason="undecided cells below the minimum width", processed=processed,
                               min_q_lower_bound_on_constraint_cells=min_q_on_constraints)
    return True, dict(processed=processed, seconds=time.time() - t0,
                      min_q_lower_bound_on_constraint_cells=min_q_on_constraints)


def largest_level(pieces: Pieces, c_hi=None, rounds=12, verbose=True, **kw):
    """Bisection for the largest proved c (returned with a 1 % safety factor applied)."""
    if c_hi is None:
        prob, s = pieces.prob, pieces.prob.sys
        x = s.sample_box(200000, seed=99)
        with torch.no_grad():
            d = prob.distances(x)
            bad = ~(d > 0).all(-1)
            aq = pieces.alpha * prob.q_exact(x)
            xb, nb = s.sample_box_boundary(100000, seed=98)
            outflow = torch.nan_to_num((s.evaluate(xb)["f_cl"] * nb).sum(-1), nan=1.0) >= 0
            aqb = pieces.alpha * prob.q_exact(xb[outflow])
            c_hi = float(min(aq[bad].min().item() if bad.any() else INF, aqb.min().item() if len(aqb) else INF))
        if verbose:
            print(f"  sampled minimum of alpha q_ex over inadmissible and outflow-face samples: {c_hi:.4f}", flush=True)
    a, b = 0.0, c_hi
    ok0, info = verify(pieces, 1e-6, **kw)
    if not ok0:
        return 0.0, info
    for _ in range(rounds):
        m = 0.5 * (a + b)
        ok, info_m = verify(pieces, m, **kw)
        if verbose:
            print(f"  c = {m:.5f}: {'proved' if ok else 'not proved'}  ({info_m.get('processed')} cells)", flush=True)
        if ok:
            a, info = m, info_m
        else:
            b = m
    return a * 0.99, info


def problem_signature(prob):
    """Constants that the proved level depends on; certify3.inner_level checks them against the run."""
    cfg, s = prob.cfg, prob.sys
    box = [float(v) for v in (([-s.L, s.L, -s.L, s.L, -s.v_max, s.v_max, -s.v_max, s.v_max, 0.0, s.T_max]) if cfg.system == "quadrotor"
                              else [-s.Lx, s.Lx, -s.L, s.L, s.x4_lo, s.x4_hi])]
    return dict(system=cfg.system, target=cfg.target, path=cfg.path, pi=cfg.pi, periodic_x1=cfg.periodic_x1, eta_c=cfg.eta_c, eta_c_hi=cfg.eta_c_hi,
                Q_diag=list(cfg.Q_diag), alpha=float(prob.alpha), z_scale=[float(v) for v in s.z_scale.tolist()], box=box,
                eps=[float(e) for e in s.eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="unicycle")
    ap.add_argument("--path", default=ProblemConfig.path)
    ap.add_argument("--pi", default=ProblemConfig.pi)
    ap.add_argument("--periodic-x1", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--eta-c-hi", type=float, default=ProblemConfig.eta_c_hi)
    ap.add_argument("--target", default=ProblemConfig.target)
    ap.add_argument("--case", default="")
    ap.add_argument("--min-rel-width", type=float, default=1e-4)
    ap.add_argument("--max-cells", type=int, default=20_000_000)
    ap.add_argument("--engine", default="interval", choices=["interval", "affine"])
    a = ap.parse_args()
    cfg = ProblemConfig(system=a.system, target=a.target, path=a.path, pi=a.pi, periodic_x1=a.periodic_x1, eta_c_hi=a.eta_c_hi,
                        beta=0.1, h_pow=2.0, device="cpu")
    prob = ConstrainedZubov(cfg)
    pieces = Pieces(prob, a.engine)
    t0 = time.time()
    c, info = largest_level(pieces, min_rel_width=a.min_rel_width, max_cells=a.max_cells)
    ok, info2 = verify(pieces, c, min_rel_width=a.min_rel_width, max_cells=a.max_cells)
    if not ok or c <= 0:
        raise SystemExit(f"no level proved: {info}")
    with torch.no_grad():
        x = prob.sys.sample_box(200000, seed=5)
        vol = (pieces.alpha * prob.q_exact(x) < c).float().mean().item()
    res = dict(system=a.system, path=a.path, pi=a.pi, eta_c_hi=a.eta_c_hi, alpha=pieces.alpha, c_inner=c,
               c1=1 - math.exp(-c), volume_of_box=vol, cells_processed=info2["processed"], seconds=time.time() - t0,
               min_rel_width=a.min_rel_width, verify_info=info2, problem=problem_signature(prob), engine=a.engine)
    case = a.case or (a.system if a.system == "quadrotor" else ("sinusoid" if "sin" in a.path else "circle"))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/inner_{case}.json").write_text(json.dumps(res, indent=1))
    print(f"PROVED inner level c = {c:.5f} (c1 = {res['c1']:.5f}), region holds {100 * vol:.2f} % of the box, "
          f"{info2['processed']} cells, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
