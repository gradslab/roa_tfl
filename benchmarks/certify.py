r"""
Level selection on the equilibrium benchmarks (no simulation labels); the estimate is tested on samples, not verified:

  * inner region I < 1: the linearization quadratic, scaled label-free with the finite-step condition (Problem.build_inner);
  * c1 = min v outside the inner region; c2 = min v over the violators with v >= c1, where a sample violates if its
    closed-loop step of normalized length 1 leaves Omega_ad (kind 1) or v does not decrease after it nor after any
    longer admissible step of length 2, 4 or 8 (kind 0; multi-horizon), plus box-outflow samples (kind 2);
    samples: box + 50 % face samples + 20 % next to the extra constraints + box-boundary outflow samples;
  * every snapshot screened (3 x 50k), the 3 largest re-certified on fresh sets A (5 x 100k), the largest kept and
    certified again on an independent set B (5 x 100k); the reported c2 is the lowest found on A and B.
  * set B is not i.i.d. uniform (it mixes box, face and constraint samples); the i.i.d. final test of iid.py is the one of
    the TFL cases (tfl/certify3.py).
Coverage / false positives against the simulated ground truth are evaluation only.

Run:  python -u certify.py runs/vdp_s0 runs/vdp_s1 ...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import core
import systems as S
from gt import ground_truth

EXTRA = (2.0, 4.0, 8.0)
EDGE, BAND = 0.5, 0.2


def cert_samples(sys_, n, seed):
    xs = [sys_.sample_box(n, seed=seed), sys_.sample_edge(int(EDGE * n), seed=seed + 300)]
    if sys_.cons:
        with torch.enable_grad():
            xs.append(sys_.sample_band(int(BAND * n), seed=seed + 500).detach())
    xb, nb = core.sample_boundary(sys_, n // 2, seed + 99)
    xo = xb[(torch.nan_to_num(sys_.f(xb)) * nb).sum(-1) >= 0]
    return torch.cat(xs), xo


@torch.no_grad()
def two_stage(model, prob, n, rounds, seed, z_tol=1e-3):
    """c1, c2 and the violators (v, kind, x) on rounds x n samples (multi-horizon, lazy retries)."""
    sys_ = prob.sys
    Ws, viol, retry, outq, kind, X = [], [], [], [], [], []
    N = 0
    for r in range(rounds):
        x, xo = cert_samples(sys_, n, seed + 17 * r)
        N += len(x) + len(xo)
        for xc in x.split(20000):
            W = model(xc)
            adm = sys_.admissible(xc)
            y, ok = core.flow(sys_, xc, 1.0)
            ok = ok & adm
            dec = (model(y) >= W) & ok & (prob.xt(xc).norm(dim=-1) > z_tol)
            Ws.append(W); viol.append(dec | ~ok); retry.append(dec); outq.append(prob.inner(xc) >= 1)
            kind.append(torch.where(ok, 0, 1)); X.append(xc)
        Wo = model(xo)
        Ws.append(Wo); viol.append(torch.ones_like(Wo, dtype=torch.bool)); retry.append(torch.zeros_like(Wo, dtype=torch.bool))
        outq.append(prob.inner(xo) >= 1); kind.append(torch.full_like(Wo, 2, dtype=torch.long)); X.append(xo)
    W, viol, retry, outq, kind, X = map(torch.cat, (Ws, viol, retry, outq, kind, X))
    W = torch.nan_to_num(W, nan=1.0)
    c1 = W[outq].min().item() * (1 - 1e-4) if outq.any() else 1.0
    hard = viol & ~retry & (W >= c1)
    vb = W[hard].min().item() if hard.any() else math.inf
    cand = torch.nonzero(retry & (W >= c1) & (W < vb)).squeeze(-1)
    cand = cand[torch.argsort(W[cand])]
    for b in cand.split(2000):
        passed = torch.zeros(len(b), dtype=torch.bool, device=W.device)
        for T in EXTRA:
            rr = torch.nonzero(~passed).squeeze(-1)
            if len(rr) == 0:
                break
            yT, okT = core.flow(sys_, X[b[rr]], T)
            passed[rr] = okT & (model(yT) < W[b[rr]])
        viol[b[passed]] = False
        if (~passed).any():
            break
    sv = viol & (W >= c1)
    c2 = min(W[sv].min().item() if sv.any() else 1.0, 1.0)
    order = torch.argsort(W[sv])[:50]
    return c1, max(c2, c1), dict(v=W[sv][order], kind=kind[sv][order], x=X[sv][order]), N


@torch.no_grad()
def region(model, prob, c2, x):
    return torch.cat([(model(c) < c2) | (prob.inner(c) < 1) for c in x.split(50000)])


def certify(run, top=3):
    run = Path(run)
    snaps = sorted((run / "snapshots").glob("epoch_*.pt"))
    prob, model, cfg = core.load(run, snaps[-1])
    sys_ = prob.sys
    x_sel = sys_.sample_box(50000, seed=31337)
    rows = []
    for sp in snaps:
        model.load_state_dict(torch.load(sp, map_location=S.DEV))
        c1, c2, _, _ = two_stage(model, prob, 50000, 3, 5000)
        vol = region(model, prob, c2, x_sel).float().mean().item()
        rows.append(dict(snapshot=sp.name, c1=c1, c2=c2, volume=vol))
        print(f"  {sp.name}  c1={c1:.4f}  c2={c2:.4f}  certified {100 * vol:5.2f} % of box", flush=True)
    best = sorted(rows, key=lambda r: -r["volume"])[:top]
    rec = []
    for r in best:
        model.load_state_dict(torch.load(run / "snapshots" / r["snapshot"], map_location=S.DEV))
        c1, c2, _, _ = two_stage(model, prob, 100000, 5, 9000)
        rec.append(dict(snapshot=r["snapshot"], c1=c1, c2=c2, volume=region(model, prob, c2, x_sel).float().mean().item()))
    sel = max(rec, key=lambda r: r["volume"])
    model.load_state_dict(torch.load(run / "snapshots" / sel["snapshot"], map_location=S.DEV))
    c1b, c2b, lim, NB = two_stage(model, prob, 100000, 5, 23000)
    c1, c2 = min(sel["c1"], c1b), min(sel["c2"], c2b)
    xg, roa = ground_truth(sys_)
    reg = region(model, prob, c2, xg)
    nr = roa.sum().item()
    names = ["step in Omega_ad, no decrease", "step leaves Omega_ad", "box outflow"]
    res = dict(run=str(run), system=cfg.system, snapshot=sel["snapshot"], c1=c1, c2=c2,
               coverage=100 * (reg & roa).sum().item() / nr, false_positive=100 * (reg & ~roa).sum().item() / len(roa),
               inner_coverage=100 * (torch.cat([prob.inner(c) < 1 for c in xg.split(50000)]) & roa).sum().item() / nr,
               box_volume=100 * reg.float().mean().item(), roa_share=100 * roa.float().mean().item(),
               scenario=dict(N=NB, eps=math.log(1e6) / NB, confidence=1 - 1e-6),
               limiting={names[k]: int((lim["kind"] == k).sum()) for k in range(3)},
               limiting_in_roa=None, screening=rows, recertified=rec)
    lab = sys_.simulate(lim["x"]) if len(lim["x"]) else torch.zeros(0)
    res["limiting_in_roa"] = int((lab == 0).sum()) if len(lab) else 0
    (run / "certify.json").write_text(json.dumps(res, indent=1))
    print(f"{run.name:18s} {sel['snapshot']}  c1={c1:.4f}  c2={c2:.4f}  coverage {res['coverage']:6.2f}%  FP {res['false_positive']:.2f}%  "
          f"(inner alone {res['inner_coverage']:.2f}%; scenario eps {res['scenario']['eps']:.1e} at 1-1e-6)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for r in a.runs:
        certify(r)


if __name__ == "__main__":
    main()
