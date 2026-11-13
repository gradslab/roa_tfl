r"""
Counterexample-guided fine-tuning on the TFL closed loops (paper, Section 3; the code calls it "repair").

The selected snapshot of a run (certify_v3.json) is copied to <run>_rep3/snapshots/epoch_000.pt.  Each round
  * computes the current c2 (2 x 50k samples, c1 fixed by the verified inner level);
  * draws fresh samples (seeds disjoint from screening, retest and final test) and keeps the counterexamples with
    v in [c1, c2 + band], plus Gaussian neighbours, in rolling sets (newest `keep`);
  * fine-tunes for `steps` Adam steps on
        anchor  mean w (e^{-Psi} - e^{-Psi_ref})^2 on fresh box samples        (hold Psi_theta elsewhere)
      + mu_dec  mean w step_violation(margin)^2 on the counterexamples         (decrease with a margin)
      + mu_out  mean w e^{-2 Phi} on counterexamples whose unit step leaves     (push v towards 1)
    with w = e^{2 min(V, w_cap)};
  * saves epoch_{round:03d}.pt.
certify3.py then screens the rounds like training snapshots, with epoch_000 (the trained network) among them.

Run:  python -u repair3.py runs/circ3_s0 [--rounds 4 --steps 300]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch

from certify3 import HORIZONS, Inner, band_failures, cert_samples, inner_level, is_whole, load, sampled_c1, two_stage, _batched
from zpinn import half_widths, phi_terms, precompute, precompute_step, step_violation


@torch.no_grad()
def blockers(model, prob, n, seed, c1, v_hi):
    x, _ = cert_samples(prob, n, seed)
    v = _batched(model, x)
    m = (v >= c1) & (v < v_hi)
    x, v = x[m], v[m]
    kinds = torch.cat([band_failures(model, prob, x[j], v[j]) for j in torch.arange(len(x), device=x.device).split(20000)]) \
        if len(x) else torch.zeros(0, dtype=torch.long, device=x.device)
    return x[kinds == 1], x[kinds == 0], v[kinds >= 0]


def repair(run, tag="_v3", rounds=4, steps=300, lr=5e-5, mu_dec=100.0, mu_out=10.0, margin=0.2, band=0.05,
           n_pool=1000000, jitter=8, sigma=0.01, n_anchor=200000, keep=50000):
    t0 = time.time()
    run = Path(run)
    cert = json.loads((run / f"certify{tag}.json").read_text())
    snap = Path(cert["snapshot"]).name
    out = Path(str(run) + "_rep3")
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "config.json", out / "config.json")
    if is_whole(run):
        shutil.copy(run / "whole.flag", out / "whole.flag")
    shutil.copy(run / "snapshots" / snap, out / "snapshots" / "epoch_000.pt")
    prob, model, cfg = load(out)
    sys = prob.sys
    c1 = cert["c1"] if cert.get("c1_sampled") else 1.0 - math.exp(-inner_level(cfg, prob))
    tau, sub, w_cap = cfg.step_tau, cfg.step_sub, (cfg.w_cap if cfg.w_cap > 0 else 3.0)
    xa = sys.sample_box(n_anchor, seed=60001)
    xa = xa[prob.admissible(xa) & sys.in_box(xa)]
    with torch.enable_grad():
        ba = precompute(model, xa)
    with torch.no_grad():
        Psi_ref, _, _, Va, _ = phi_terms(model, ba)
        wa = torch.exp(2 * Va.clamp(0, w_cap))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(7)
    half = half_widths(sys, xa.device)
    pool_d = pool_s = pool_o = None
    log = []

    def jit(x):
        nb = torch.randn((len(x) * jitter, len(half)), generator=g).to(x) * sigma * half
        xn = torch.cat([x, x.repeat(jitter, 1) + nb])
        return xn[prob.admissible(xn) & sys.in_box(xn)]

    for r in range(1, rounds + 1):
        model.eval()
        c2, _ = two_stage(model, prob, c1, 50000, 2, 70000 + 101 * r)
        xo, xd, vb = blockers(model, prob, n_pool, 80000 + 101 * r, c1, min(c2 + band, 0.999))
        if len(xd):
            xd = jit(xd)
            with torch.enable_grad():
                bd = precompute(model, xd)
            sd = precompute_step(model, xd, tau, sub)
            pool_d = bd if pool_d is None else {k: torch.cat([pool_d[k], bd[k]]) for k in bd}
            pool_s = sd if pool_s is None else {k: torch.cat([pool_s[k], sd[k]]) for k in sd}
        if len(xo):
            xo = jit(xo)
            with torch.enable_grad():
                bo = precompute(model, xo)
            pool_o = bo if pool_o is None else {k: torch.cat([pool_o[k], bo[k]]) for k in bo}
        if pool_d is not None and len(pool_d["qe"]) > keep:
            pool_d = {k: v[-keep:] for k, v in pool_d.items()}
            pool_s = {k: v[-keep:] for k, v in pool_s.items()}
        if pool_o is not None and len(pool_o["qe"]) > keep:
            pool_o = {k: v[-keep:] for k, v in pool_o.items()}
        nd = 0 if pool_d is None else len(pool_d["qe"])
        no = 0 if pool_o is None else len(pool_o["qe"])
        vmin = vb.min().item() if len(vb) else float("nan")
        print(f"round {r}: c1 {c1:.4f} c2 {c2:.4f} (2 x 50k); blockers in [c1, c2 + {band}]: lowest v {vmin:.4f}; "
              f"repair set: {nd} decrease, {no} exit  [{time.time() - t0:.0f} s]", flush=True)
        log.append(dict(round=r, c1=c1, c2=c2, v_min_blocker=vmin, n_dec=nd, n_out=no))
        if nd + no == 0:
            break
        model.train()
        for it in range(steps):
            ia = torch.randint(len(xa), (4096,), generator=g).to(xa.device)
            Psi = phi_terms(model, {k: v[ia] for k, v in ba.items()})[0]
            loss = (wa[ia] * (torch.exp(-Psi) - torch.exp(-Psi_ref[ia])) ** 2).mean()
            ld = lo = torch.zeros((), device=xa.device)
            if nd:
                iv = torch.randint(nd, (min(nd, 4096),), generator=g).to(xa.device)
                sv, V = step_violation(model, {k: v[iv] for k, v in pool_d.items()}, {k: v[iv] for k, v in pool_s.items()},
                                       margin, getattr(cfg, "step_detach", False))
                ld = (torch.exp(2 * V.detach().clamp(0, w_cap)) * sv ** 2).mean()
            if no:
                io = torch.randint(no, (min(no, 4096),), generator=g).to(xa.device)
                _, Phi, _, V, _ = phi_terms(model, {k: v[io] for k, v in pool_o.items()})
                lo = (torch.exp(2 * V.detach().clamp(0, w_cap)) * torch.exp(-2 * Phi)).mean()
            loss = loss + mu_dec * ld + mu_out * lo
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  {steps} steps: loss {loss.item():.3e}  decrease {ld.item():.3e}  exit {lo.item():.3e}", flush=True)
        torch.save(model.state_dict(), out / "snapshots" / f"epoch_{r:03d}.pt")
    (out / "repair.json").write_text(json.dumps(dict(source=str(run), snapshot=snap, rounds=log, seconds=time.time() - t0), indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default="_v3")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    for run in a.runs:
        print(f"== repair {run}", flush=True)
        repair(run, a.tag, a.rounds, a.steps, a.lr)


if __name__ == "__main__":
    main()
