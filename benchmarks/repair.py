r"""
Counterexample-guided fine-tuning on the equilibrium benchmarks (the code calls it "repair"), same settings as on the TFL loops: the selected snapshot is copied to
<run>_rep as epoch_000.pt; each of 4 rounds finds the samples that block c2 (v in [c1, c2 + 0.05] whose step leaves
Omega_ad, or that decrease after no horizon 1, 2, 4, 8) in a 1M certification-style pool, adds Gaussian neighbours to
rolling repair sets (newest 50k), and fine-tunes 300 Adam steps (lr 5e-5) on
    anchor  w (e^{-Psi} - e^{-Psi_ref})^2 on fresh box samples
  + 100 w step_violation(margin 0.2)^2 on the decrease blockers  + 10 w e^{-2 Phi} on the exit blockers,
w = e^{2 min(V, 3)}.  The rounds are then screened by certify.py like training snapshots (epoch_000 included).

Run:  python -u repair.py runs/vdp_s0 ...
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch

import core
import systems as S
from certify import cert_samples, two_stage, EXTRA


def blockers(model, prob, n, seed, c1, v_hi):
    sys_ = prob.sys
    x, _ = cert_samples(sys_, n, seed)
    with torch.no_grad():
        x = x[sys_.admissible(x)]
        v = torch.cat([model(c) for c in x.split(50000)])
        m = (v >= c1) & (v < v_hi)
        x, v = x[m], v[m]
        out = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        dec = torch.zeros_like(out)
        for j in torch.arange(len(x), device=x.device).split(20000):
            y, ok = core.flow(sys_, x[j], 1.0)
            fail = ok & (model(y) >= v[j]) & (prob.xt(x[j]).norm(dim=-1) > 1e-3)
            for T in EXTRA:
                r = torch.nonzero(fail).squeeze(-1)
                if len(r) == 0:
                    break
                yT, okT = core.flow(sys_, x[j][r], T)
                fail[r] = ~(okT & (model(yT) < v[j][r]))
            out[j], dec[j] = ~ok, fail
    return x[out], x[dec], v[out | dec]


def repair(run, rounds=4, steps=300, lr=5e-5, mu_dec=100.0, mu_out=10.0, margin=0.2, band=0.05, n_pool=1000000,
           jitter=8, sigma=0.01, n_anchor=200000, keep=50000):
    t0 = time.time()
    run = Path(run)
    cert = json.loads((run / "certify.json").read_text())
    out = Path(str(run) + "_rep")
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "config.json", out / "config.json")
    shutil.copy(run / "snapshots" / cert["snapshot"], out / "snapshots" / "epoch_000.pt")
    prob, model, cfg = core.load(out)
    sys_ = prob.sys
    xa = sys_.sample_box(n_anchor, seed=60001)
    xa = xa[sys_.admissible(xa)]
    ba = core.precompute(model, xa)
    with torch.no_grad():
        Psi_ref, _, _, Va, _ = core.phi_terms(model, ba)
        wa = torch.exp(2 * Va.clamp(0, cfg.w_cap))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(7)
    pool_d = pool_s = pool_o = None
    log = []

    def jit(x):
        if len(x) * (jitter + 1) > keep:
            x = x[torch.randperm(len(x), generator=g)[:max(1, keep // (jitter + 1))].to(x.device)]
        nb = torch.randn((len(x) * jitter, sys_.n), generator=g).to(x) * sigma * sys_.half
        xn = torch.cat([x, x.repeat(jitter, 1) + nb])
        return xn[sys_.admissible(xn)]

    for r in range(1, rounds + 1):
        c1, c2, _, _ = two_stage(model, prob, 50000, 2, 70000 + 101 * r)
        xo, xd, vb = blockers(model, prob, n_pool, 80000 + 101 * r, c1, min(c2 + band, 0.9999))
        if len(xd):
            xd = jit(xd)
            bd, sd = core.precompute(model, xd), core.precompute_step(model, xd, cfg.step_tau, cfg.step_sub)
            pool_d = {k: (bd[k] if pool_d is None else torch.cat([pool_d[k], bd[k]]))[-keep:] for k in bd}
            pool_s = {k: (sd[k] if pool_s is None else torch.cat([pool_s[k], sd[k]]))[-keep:] for k in sd}
        if len(xo):
            bo = core.precompute(model, jit(xo))
            pool_o = {k: (bo[k] if pool_o is None else torch.cat([pool_o[k], bo[k]]))[-keep:] for k in bo}
        nd = 0 if pool_d is None else len(pool_d["q"])
        no = 0 if pool_o is None else len(pool_o["q"])
        vmin = vb.min().item() if len(vb) else float("nan")
        print(f"round {r}: c1 {c1:.4f} c2 {c2:.4f} (2 x 50k); lowest blocker {vmin:.4f}; repair set {nd} decrease, {no} exit  "
              f"[{time.time() - t0:.0f} s]", flush=True)
        log.append(dict(round=r, c1=c1, c2=c2, v_min_blocker=vmin, n_dec=nd, n_out=no))
        if nd + no == 0:
            break
        for _ in range(steps):
            ia = torch.randint(len(xa), (4096,), generator=g).to(S.DEV)
            Psi = core.phi_terms(model, {k: v[ia] for k, v in ba.items()})[0]
            loss = (wa[ia] * (torch.exp(-Psi) - torch.exp(-Psi_ref[ia])) ** 2).mean()
            if nd:
                iv = torch.randint(nd, (min(nd, 4096),), generator=g).to(S.DEV)
                sv, V = core.step_violation(model, {k: v[iv] for k, v in pool_d.items()}, {k: v[iv] for k, v in pool_s.items()}, margin)
                loss = loss + mu_dec * (torch.exp(2 * V.detach().clamp(0, cfg.w_cap)) * sv ** 2).mean()
            if no:
                io = torch.randint(no, (min(no, 4096),), generator=g).to(S.DEV)
                _, Phi, _, V, _ = core.phi_terms(model, {k: v[io] for k, v in pool_o.items()})
                loss = loss + mu_out * (torch.exp(2 * V.detach().clamp(0, cfg.w_cap)) * torch.exp(-2 * Phi)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), out / "snapshots" / f"epoch_{r:03d}.pt")
    (out / "repair.json").write_text(json.dumps(dict(source=str(run), snapshot=cert["snapshot"], rounds=log), indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for r in a.runs:
        print(f"== repair {r}", flush=True)
        repair(r)


if __name__ == "__main__":
    main()
