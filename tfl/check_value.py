r"""
Exact value function by simulation: integrates the closed loop from each sample and tracks
V(x) = sup_t [ int_0^t g + h(y(t)) ].  ceiling3.py uses value_sim; the command line compares {v < 1} with the simulated
region of attraction for several (alpha, beta).

Run:  python -u check_value.py [--n 20000] [--out results/check]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from problem import ConstrainedZubov, ProblemConfig

BETAS = (0.0, 0.03, 0.1, 0.3)
ALPHA_MULT = (0.3, 1.0, 3.0)
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)


@torch.no_grad()
def value_sim(P: ConstrainedZubov, x0, alphas, betas, T=30.0, dt=0.01, tol=1e-3, hold=1.0, eta_tol=0.05,
              chunk=20000, extra=None):
    """Returns (label, V) with V of shape (N, len(alphas), len(betas)); labels as TFLClosedLoop.simulate.
    extra: optional additional running cost x -> rate (e.g. kappa rho), integrated in physical time and
    assumed to vanish once the run has converged (rho = 0 near the target)."""
    if len(x0) > chunk:
        parts = [value_sim(P, xc, alphas, betas, T, dt, tol, hold, eta_tol, chunk, extra) for xc in x0.split(chunk)]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    s = P.sys
    a = torch.as_tensor(alphas, device=s.device, dtype=s.dtype)
    x = x0.clone()
    N, A, B = len(x), len(alphas), len(betas)
    label = torch.full((N,), 3, dtype=torch.long, device=s.device)
    held = torch.zeros(N, device=s.device, dtype=s.dtype)
    G = torch.zeros(N, device=s.device, dtype=s.dtype)
    w_prev = torch.zeros(N, device=s.device, dtype=s.dtype)
    E = torch.zeros(N, device=s.device, dtype=s.dtype)
    e_prev = torch.zeros(N, device=s.device, dtype=s.dtype)
    S = torch.full((N, A, B), -math.inf, device=s.device, dtype=s.dtype)
    V = torch.full((N, A, B), math.nan, device=s.device, dtype=s.dtype)
    act = torch.arange(N, device=s.device)
    Tprev = x[:, -1].clone()
    rhs = lambda y: s.evaluate(y)["f_cl"]
    for k in range(int(round(T / dt)) + 1):
        xa = x[act]
        e = s.evaluate(xa)
        bad = (e["margin"] <= 0) | ~torch.isfinite(xa).all(-1) | ~torch.isfinite(e["f_cl"]).all(-1)
        out = ~s.in_box(xa)
        w = P.g(xa, 1.0)
        ex = extra(xa) if extra is not None else torch.zeros_like(w)
        if k > 0:
            G[act] += 0.5 * dt * (w_prev[act] + w)
            E[act] += 0.5 * dt * (e_prev[act] + ex)
        w_prev[act] = w
        e_prev[act] = ex
        d = P.distances(xa)
        hb = torch.stack([P.h_from_d(d, b) for b in betas], -1)
        S[act] = torch.maximum(S[act], a[None, :, None] * G[act][:, None, None] + E[act][:, None, None] + hb[:, None, :])
        if hasattr(s, "T_max"):
            near = (s._z(xa) / s.z_scale).norm(dim=-1) < tol
            bad = bad | (Tprev[act] * xa[:, 5] <= 0)
            Tprev[act] = xa[:, 5]
        else:
            near = (e["xi"].norm(dim=-1) < tol) & ((e["eta"][:, 1] - s.eta2_ref).abs() < eta_tol)
        held[act] = torch.where(near, held[act] + dt, torch.zeros_like(held[act]))
        conv = held[act] >= hold - 1e-9
        lab = torch.full_like(act, -1)
        lab[conv] = 0
        lab[out] = 2
        lab[bad] = 1
        done = lab >= 0
        if done.any():
            idx = act[done]
            label[idx] = lab[done]
            tail = G[idx] + P.q(xa[done])
            Vc = torch.maximum(S[idx], a[None, :, None] * tail[:, None, None] + E[idx][:, None, None])
            V[idx] = torch.where((lab[done] == 0)[:, None, None], Vc, torch.full_like(Vc, math.inf))
        act, xa, k1 = act[~done], xa[~done], e["f_cl"][~done]
        if len(act) == 0:
            break
        k2 = rhs(xa + 0.5 * dt * k1)
        k3 = rhs(xa + 0.5 * dt * k2)
        k4 = rhs(xa + dt * k3)
        x[act] = xa + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return label, V


def perturb(s, x, eps, seed):
    """x + eps * (unit random direction) in box-normalized coordinates."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(x.shape, generator=g).to(x)
    u = u / u.norm(dim=-1, keepdim=True)
    half = torch.tensor([s.Lx, s.L, math.pi, s.h4], device=x.device, dtype=x.dtype)
    return x + eps * u * half


def slice_grid(s, which, n):
    t = torch.linspace(-1, 1, n, device=s.device, dtype=s.dtype)
    A, Bm = torch.meshgrid(t, t, indexing="xy")
    x = torch.zeros(n * n, 4, device=s.device, dtype=s.dtype)
    if which == "x1x2":
        x[:, 0], x[:, 1], x[:, 2], x[:, 3] = s.L * A.flatten(), s.L * Bm.flatten(), math.pi / 2, 0.0
        ext, lab = (-s.L, s.L, -s.L, s.L), ("x1", "x2")
    else:
        x[:, 0], x[:, 1], x[:, 2] = s.L * A.flatten(), 0.0, math.pi / 2
        x[:, 3] = s.c4 + s.h4 * Bm.flatten()
        ext, lab = (-s.L, s.L, s.x4_lo, s.x4_hi), ("x1", "x4")
    return x, ext, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-pairs", type=int, default=8000)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--out", default="results/check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = ConstrainedZubov(ProblemConfig())
    s = P.sys
    alphas = [m * P.alpha for m in ALPHA_MULT]
    ia = ALPHA_MULT.index(1.0)
    print(f"alpha (label-free 3/q90) = {P.alpha:.4f}; constraint scales {P.scale.cpu().numpy().round(3)}", flush=True)
    res = dict(alpha=P.alpha, alphas=alphas, betas=BETAS, scales=P.scale.tolist())

    t0 = time.time()
    xg = s.sample_box(a.n, seed=4242)
    ref = s.simulate(xg)
    lab, V = value_sim(P, xg, alphas, BETAS)
    v = 1 - torch.exp(-V)
    roa = ref == 0
    res["true_roa_percent_of_box"] = 100 * roa.float().mean().item()
    res["labels_identical_to_simulator"] = bool((lab == ref).all().item())
    res["inconclusive"] = int((lab == 3).sum().item())
    fin = torch.isfinite(V[:, ia, 2])
    res["finite_V_equals_roa"] = bool((fin == roa).all().item())
    print(f"[1] true ROA {res['true_roa_percent_of_box']:.2f} % of box; labels identical to simulator: "
          f"{res['labels_identical_to_simulator']}; inconclusive {res['inconclusive']}; "
          f"{{v<1}} == ROA: {res['finite_V_equals_roa']}  [{time.time() - t0:.0f} s]", flush=True)

    cover = {}
    for i, al in enumerate(alphas):
        for j, b in enumerate(BETAS):
            key = f"alpha x{ALPHA_MULT[i]} beta {b}"
            cover[key] = {str(c): 100 * ((v[:, i, j] < c) & roa).sum().item() / roa.sum().item() for c in LEVELS}
    res["roa_share_in_sublevel"] = cover
    print("[3] % of true ROA inside {v < c}:", flush=True)
    print("    " + " " * 22 + "".join(f"c={c:<7}" for c in LEVELS))
    for k_, row in cover.items():
        print(f"    {k_:22s}" + "".join(f"{row[str(c)]:7.2f}  " for c in LEVELS), flush=True)

    t0 = time.time()
    base = xg[: a.n_pairs]
    cont = {}
    for eps in (0.003, 0.01, 0.03):
        xp = perturb(s, base, eps, seed=int(1e4 * eps))
        lp, Vp = value_sim(P, xp, alphas, BETAS)
        ok = (lab[: a.n_pairs] != 3) & (lp != 3)
        inb, inp = (lab[: a.n_pairs] == 0) & ok, (lp == 0) & ok
        strad = inb ^ inp
        vin = torch.where(inb[:, None, None], 1 - torch.exp(-V[: a.n_pairs]), 1 - torch.exp(-Vp))[strad][:, ia, :]
        both = inb & inp
        dv = ((1 - torch.exp(-V[: a.n_pairs])) - (1 - torch.exp(-Vp)))[both][:, ia, :].abs()
        row = dict(n_straddling=int(strad.sum().item()))
        for j, b in enumerate(BETAS):
            vb = vin[:, j]
            row[f"beta {b}"] = dict(median_v_inside=vb.median().item() if len(vb) else None,
                                    frac_v_inside_gt_0_9=(vb > 0.9).float().mean().item() if len(vb) else None,
                                    frac_v_inside_gt_0_99=(vb > 0.99).float().mean().item() if len(vb) else None,
                                    p99_dv_both_inside=torch.quantile(dv[:, j], 0.99).item())
        cont[str(eps)] = row
        print(f"[2] eps {eps}: {row['n_straddling']} straddling pairs; v of the inside point "
              "(median / >0.9 / >0.99), 99th pct |dv| for pairs inside:", flush=True)
        for j, b in enumerate(BETAS):
            r = row[f"beta {b}"]
            print(f"      beta {b:<5} {r['median_v_inside']:.3f} / {100 * r['frac_v_inside_gt_0_9']:5.1f} % / "
                  f"{100 * r['frac_v_inside_gt_0_99']:5.1f} %    {r['p99_dv_both_inside']:.3f}", flush=True)
    res["continuity"] = cont
    print(f"    [{time.time() - t0:.0f} s]", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = (0, 2)
        fig, axs = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        for r_, which in enumerate(("x1x2", "x1x4")):
            xs, ext, (lx, ly) = slice_grid(s, which, a.grid)
            ls, Vs = value_sim(P, xs, alphas, BETAS)
            vs = (1 - torch.exp(-Vs[:, ia, :])).reshape(a.grid, a.grid, -1).cpu()
            inroa = (ls == 0).reshape(a.grid, a.grid).float().cpu()
            for c_, j in enumerate(show):
                ax = axs[r_, c_]
                im = ax.imshow(vs[:, :, j], origin="lower", extent=ext, vmin=0, vmax=1, cmap="viridis",
                               aspect="auto")
                ax.contour(inroa, levels=[0.5], extent=ext, colors="w", linewidths=1.0)
                ax.set_title(f"v, beta = {BETAS[j]}" + (" (hard exit)" if BETAS[j] == 0 else ""))
                ax.set_xlabel(lx)
                ax.set_ylabel(ly)
        fig.colorbar(im, ax=axs, shrink=0.6, label="v (white: true ROA boundary)")
        fig.savefig(out / "slices.png", dpi=120)
    except Exception as e_:
        print(f"(slices skipped: {e_})")
    (out / "check.json").write_text(json.dumps(res, indent=2))
    print(f"saved {out / 'check.json'}", flush=True)


if __name__ == "__main__":
    main()
