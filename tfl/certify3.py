r"""
Level selection and final test of a learned function on the TFL closed loops (paper, Section 3, "Level selection").

Inner layer (verified, inner_region.py):  I = {alpha q_ex < c_I} intersected with the open box.  Since V >= alpha q_ex,
    {v < c1} with c1 = 1 - e^{-c_I} lies inside I (paper, Theorem "Two-layer estimate" (ii)); this needs no samples.
Finite-step decrease condition (sampled):  a sample x with v(x) >= c1 passes if, for some horizon H in HORIZONS, the
    trajectory of the normalized field stays in Omega_ad and the box at every RK4 substep (spacing STEP_H) over length H
    and v at its end point is below v(x).  A sample that fails is a counterexample (this includes samples whose unit
    step leaves Omega_ad or the box, and outflow samples on the box faces).  c2 = the lowest v among counterexamples.
Protocol:  every snapshot of a run is screened (3 x 50k samples) and ranked by the share of box samples in
    I u {v < c2}; the best `top` snapshots are retested on fresh samples (5 x 100k); the largest is kept with the
    lowest level it received.  Final test: N i.i.d. uniform box samples with disjoint seeds (benchmarks/iid.py) lower c2 to
    the lowest counterexample; with confidence 1 - 1e-6 the uniform measure of counterexamples in the annulus
    {c1 <= v < c2} is at most ln(1e6) / N (paper, Proposition "Scenario bound").
Evaluation (simulation, never used for selection):  coverage of the simulated region of attraction by I u {v < c2}
    and the false-positive share, on the reference states of the case.

Run:  python -u certify3.py runs/circ3_s0 runs/circ3_s1 --top 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys as _sys
import time
from pathlib import Path

import torch

from zpinn import load as _load


def load(run):
    """zpinn.load, with the no-decomposition problem class for runs of zpinn_whole.py (marker file whole.flag)."""
    if (Path(run) / "pinn.flag").exists() or (Path(run) / "quad.flag").exists():
        from zpinn_liu import load_baseline
        return load_baseline(run)
    prob, model, cfg = _load(run)
    if (Path(run) / "whole.flag").exists():
        from zpinn_whole import WholeProblem
        prob.__class__ = WholeProblem
    return prob, model, cfg


def is_whole(run):
    return (Path(run) / "whole.flag").exists() or (Path(run) / "pinn.flag").exists()

_sys.path.append(str(Path(__file__).resolve().parent.parent / "benchmarks"))
from iid import iid_test, seed_of

HORIZONS = (1.0, 2.0, 4.0, 8.0)
STEP_H = 0.025
EDGE = 0.5
Z_TOL = 1e-3
N_IID = 1000000


def case_name(cfg):
    if cfg.system == "quadrotor":
        return "quadrotor"
    return "sinusoid" if "sin" in cfg.path else "circle"


def inner_level(cfg, prob=None):
    """Verified inner level c_I of the case (results/inner_<case>.json), checked against the problem constants it was
    verified for when `prob` is given; falls back to cfg.c_inner (> 0) when no verification file exists."""
    f = Path(f"results/inner_{case_name(cfg)}.json")
    if f.exists():
        d = json.loads(f.read_text())
        if prob is not None and "problem" in d:
            from inner_region import problem_signature
            sig = problem_signature(prob)
            for k, v in d["problem"].items():
                mine = sig[k]
                same = (abs(float(v) - float(mine)) <= 1e-6 * max(1.0, abs(float(mine)))) if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    else (all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(v, mine)) and len(v) == len(mine)) if isinstance(v, list) \
                    else v == mine
                if not same:
                    raise ValueError(f"verified inner level of {case_name(cfg)} was computed for {k}={v}, run has {mine}")
        return float(d["c_inner"])
    if cfg.c_inner > 0:
        return float(cfg.c_inner)
    raise FileNotFoundError(f"no verified inner level for {case_name(cfg)}: run inner_region.py first")


class Inner(torch.nn.Module):
    """inner(x) < 1  <=>  x in I = {alpha q_ex < c_I}."""

    def __init__(self, prob, c_I):
        super().__init__()
        self.prob, self.c_I = prob, float(c_I)

    def forward(self, x):
        return self.prob.alpha * self.prob.q_exact(x) / self.c_I


def _batched(fn, x, bs=20000):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def dim_of(sys):
    return int(getattr(sys, "n", 4))


@torch.no_grad()
def flow_step(prob, x, tau, h=STEP_H):
    """RK4 of the normalized closed-loop field over normalized time tau; ok = every substep admissible and in the box."""
    sys = prob.sys
    sub = max(1, math.ceil(tau / h))
    hs = tau / sub
    fh = lambda z: sys.field(z)[0]
    y = x.clone()
    ok = torch.ones(len(x), dtype=torch.bool, device=x.device)
    for _ in range(sub):
        k1 = fh(y); k2 = fh(y + 0.5 * hs * k1); k3 = fh(y + 0.5 * hs * k2); k4 = fh(y + hs * k3)
        y1 = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ok = ok & torch.isfinite(y1).all(-1)
        y1 = torch.where(ok[:, None], y1, y)
        ok = ok & prob.admissible(y1) & sys.in_box(y1)
        y = y1
    return y, ok


def band_eps(sys):
    return (0.01, 0.01, 0.05) if not hasattr(sys, "distances") else sys.eps


def cert_samples(prob, n, seed, edge=EDGE):
    """Box samples, samples near the faces (n * edge), samples next to the singular sets (n / 5), and the outflow
    samples of the box faces."""
    sys = prob.sys
    x = sys.sample_box(n, seed=seed)
    if edge > 0:
        x = torch.cat([x, sys.sample_edge(int(n * edge), seed=seed + 300)])
    with torch.enable_grad():
        x = torch.cat([x, sys.sample_band(n // 5, "singular", 3.0, 0.1, seed=seed + 500, eps=band_eps(sys))])
    x = x.detach()
    xb, nb = sys.sample_box_boundary(n // 2, seed=seed + 99)
    xo = xb[torch.nan_to_num((sys.evaluate(xb)["f_cl"] * nb).sum(-1), nan=1.0) >= 0]
    return x[prob.admissible(x) & sys.in_box(x)], xo


@torch.no_grad()
def band_failures(model, prob, x, W, horizons=HORIZONS):
    """kind: 1 = unit step leaves Omega_ad / the box (exit), 0 = no decrease at any horizon, -1 = passes."""
    sys = prob.sys
    y, ok = flow_step(prob, x, horizons[0])
    zn = sys.z_tilde(x).norm(dim=-1)
    dec = ok & (torch.nan_to_num(model(y), nan=1.0) >= W) & (zn > Z_TOL)
    for T in horizons[1:]:
        r = torch.nonzero(dec).squeeze(-1)
        if len(r) == 0:
            break
        yT, okT = flow_step(prob, x[r], T)
        dec[r] = ~(okT & (torch.nan_to_num(model(yT), nan=1.0) < W[r]))
    kind = torch.full_like(W, -1, dtype=torch.long)
    kind[dec] = 0
    kind[~ok] = 1
    return kind


@torch.no_grad()
def two_stage(model, prob, c1, n, rounds, seed, keep=200, batch=20000):
    """c2 for the fixed c1 from the verified inner level.  Lazy: failures are only sought among the lowest-v band samples, in order of v,
    stopping once a failure is found (the same c2 as testing every sample).  Returns c2 and the lowest failures."""
    Xs, Ws, Os = [], [], []
    for r in range(rounds):
        x, xo = cert_samples(prob, n, seed + 17 * r)
        Xs.append(x); Ws.append(torch.nan_to_num(_batched(model, x), nan=1.0))
        Os.append(torch.nan_to_num(_batched(model, xo), nan=1.0))
    X, W, Wo = torch.cat(Xs), torch.cat(Ws), torch.cat(Os)
    c2 = min(Wo[Wo >= c1].min().item() if (Wo >= c1).any() else 1.0, 1.0)
    cand = torch.nonzero((W >= c1) & (W < c2)).squeeze(-1)
    cand = cand[torch.argsort(W[cand])]
    fails = []
    n_tested = 0
    for b in cand.split(batch):
        if not len(b) or W[b[0]].item() >= c2:
            break
        kind = band_failures(model, prob, X[b], W[b])
        n_tested += len(b)
        bad = torch.nonzero(kind >= 0).squeeze(-1)
        if len(bad):
            c2 = min(c2, W[b[bad[0]]].item())
            for i in bad[:keep]:
                fails.append(dict(v=W[b[i]].item(), kind=int(kind[i]), x=[round(t, 5) for t in X[b[i]].tolist()]))
            break
    c2 = max(c2, c1)
    return c2, dict(n_band=int(((W >= c1) & (W < 1.0)).sum()), n_tested=n_tested, lowest_failures=fails[:10])


@torch.no_grad()
def sampled_c1(model, prob, inner, n, rounds, seed):
    """Largest level with {v < c1} inside I on samples: the two-stage rule of Liu et al. (2025, Prop. 5), used only for
    the no-decomposition baseline, whose function has no lower bound alpha q_ex."""
    lo = 1.0
    for r in range(rounds):
        x, xo = cert_samples(prob, n, seed + 17 * r)
        x = torch.cat([x, xo])
        out = _batched(inner, x) >= 1
        if out.any():
            lo = min(lo, torch.nan_to_num(_batched(model, x[out]), nan=1.0).min().item())
    return lo - 1e-4 * abs(lo)


def region(model, inner, c2, x):
    with torch.no_grad():
        return (_batched(model, x) < c2) | (_batched(inner, x) < 1)


def ground_truth(prob, n=50000, seed=4242):
    sys = prob.sys
    cache = Path(f"results/gt_{case_name(prob.cfg)}_{n // 1000}k.pt")
    if cache.exists():
        d = torch.load(cache, map_location=sys.device)
        return d["xg"], d["roa"], d["label"]
    xg = sys.sample_box(n, seed=seed)
    label = sys.simulate(xg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(xg=xg, roa=label == 0, label=label), cache)
    return xg, label == 0, label


def certify_run(run, top=3, tag="_v3", n_iid=N_IID, snapshots=None, verbose=True, seed_tag=None):
    t0 = time.time()
    prob, model, cfg = load(run)
    sys = prob.sys
    c_I = inner_level(cfg, prob)
    c1 = 1.0 - math.exp(-c_I)
    inner = Inner(prob, c_I)
    whole = is_whole(run)
    x_sel = sys.sample_box(50000, seed=31337)
    rows = []
    snaps = snapshots or sorted((Path(run) / "snapshots").glob("epoch_*.pt"))
    for sp in snaps:
        model.load_state_dict(torch.load(sp, map_location=sys.device))
        if whole:
            c1 = sampled_c1(model, prob, inner, 50000, 3, 5000)
        c2, _ = two_stage(model, prob, c1, 50000, 3, 5000)
        vol = region(model, inner, c2, x_sel).float().mean().item()
        rows.append(dict(snapshot=Path(sp).name, c2=c2, volume=vol))
        if verbose:
            print(f"  {Path(sp).name}  c2={c2:.4f}  candidate region {100 * vol:5.2f} % of box  [{time.time() - t0:.0f} s]", flush=True)
    ranked = sorted(rows, key=lambda r_: -r_["volume"])
    recert, best = [], (-1.0, None)
    for r_ in ranked[:top]:
        model.load_state_dict(torch.load(Path(run) / "snapshots" / r_["snapshot"], map_location=sys.device))
        if whole:
            c1 = sampled_c1(model, prob, inner, 100000, 5, 9000)
        c2, _ = two_stage(model, prob, c1, 100000, 5, 9000)
        vol = region(model, inner, c2, x_sel).float().mean().item()
        recert.append(dict(snapshot=r_["snapshot"], c2=c2, volume=vol))
        if verbose:
            print(f"  re-tested {r_['snapshot']}  c2={c2:.4f}  region {100 * vol:5.2f} % of box", flush=True)
        if vol > best[0]:
            best = (vol, r_["snapshot"])
    snap = best[1]
    model.load_state_dict(torch.load(Path(run) / "snapshots" / snap, map_location=sys.device))
    if whole:
        c1 = sampled_c1(model, prob, inner, 100000, 5, 9000)
    c2_sets = [r_["c2"] for r_ in recert + rows if r_["snapshot"] == snap]
    c2_sel = max(min(c2_sets), c1)
    def step(x, T):
        y, ok = flow_step(prob, x, T)
        return y, ok & (sys.field(x)[1] > 0)
    iid = iid_test(model, inner, lambda u: sys._box_map(u.to(sys.device, sys.dtype)), step,
                   lambda x: sys.z_tilde(x).norm(dim=-1), dim_of(sys), c1, c2_sel, seed_of(Path(run).name + (seed_tag or tag)),
                   n=n_iid, horizons=HORIZONS[1:])
    c2 = iid["c2"]
    xg, roa, label = ground_truth(prob)
    reg = region(model, inner, c2, xg)
    nr = roa.sum().item()
    with torch.no_grad():
        inner_only = _batched(inner, xg) < 1
    res = dict(run=str(run), case=case_name(cfg), snapshot=snap, alpha=float(prob.alpha), c_inner=c_I, c1=c1,
               c2_screening_and_retest=c2_sets, c2_selected=c2_sel, c2=c2, iid=iid, screening=rows, retest=recert,
               horizons=list(HORIZONS), step_h=STEP_H, c1_sampled=whole,
               coverage=100 * (reg & roa).sum().item() / nr, false_positive=100 * (reg & ~roa).sum().item() / len(roa),
               false_positive_of_region=100 * (reg & ~roa).sum().item() / max(reg.sum().item(), 1),
               inner_coverage=100 * (inner_only & roa).sum().item() / nr, roa_fraction_of_box=100 * nr / len(roa),
               region_fraction_of_box=100 * reg.float().mean().item(), n_ground_truth=len(roa), seconds=time.time() - t0)
    (Path(run) / f"certify{tag}.json").write_text(json.dumps(res, indent=1))
    if verbose:
        print(f"{Path(run).name:18s} {snap}  c1={c1:.4f}  c2={c2:.4f} (selected {c2_sel:.4f})  coverage {res['coverage']:6.2f} %  "
              f"FP {res['false_positive']:.3f} %  inner alone {res['inner_coverage']:.2f} %  [{time.time() - t0:.0f} s]", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--tag", default="_v3")
    ap.add_argument("--n-iid", type=int, default=N_IID)
    ap.add_argument("--seed-tag", default=None)
    a = ap.parse_args()
    for run in a.runs:
        certify_run(run, a.top, a.tag, a.n_iid, seed_tag=a.seed_tag)


if __name__ == "__main__":
    main()
