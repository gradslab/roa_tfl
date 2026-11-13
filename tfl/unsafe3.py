r"""
Direct test of unsafe states in the estimates of a TFL case (evaluation only; nothing here changes an estimate).

S is the union of the four fine-tuned estimates of the case, I u {v_k < c2_k}, fixed before any sample is drawn.  N i.i.d.
uniform samples of the box are drawn with seeds disjoint from every earlier test.  A sample in S counts as safe if its
trajectory of the normalized field reaches the verified inner layer I within normalized length T_MAX while every RK4
substep (length STEP) is finite, admissible, inside the box, and keeps the sign of det D (a sign change between substeps
would cross the singular set unnoticed).  I is forward invariant and lies in D(M) (paper, Theorem "Two-layer estimate"
(i)), so a safe sample lies in D(M) up to the integration error.  The other samples of S form the set U, which contains
S minus D(M).  With k samples in U, the one-sided Clopper-Pearson bound eps_k satisfies P^N[ P(U) > eps_k ] <= 1e-6.
U contains the corresponding set of every seed, since each seed's estimate lies in S.

Run:  python -u unsafe3.py circ3 sin6 quad3     (writes results/unsafe_<name>.json)
"""
import json
import math
import sys
import time
import zlib
from pathlib import Path

import torch
from scipy.stats import beta as beta_dist

from certify3 import Inner, _batched, dim_of, inner_level, load

N = 1000000
CHUNK = 200000
STEP = 0.025
T_MAX = 200.0
DELTA = 1e-6
SIGN_INDEX = {"unicycle": 3, "quadrotor": 5}


def cp_upper(k, n, delta=DELTA):
    return 1.0 if k >= n else float(beta_dist.ppf(1 - delta, k + 1, n - k))


def sign_of_det(prob, x):
    i = SIGN_INDEX[prob.cfg.system]
    return torch.sign(1 + x[:, 3]) if i == 3 else torch.sign(x[:, 5])


@torch.no_grad()
def classify(prob, inner, x):
    """0: reaches I safely; 1: a substep is not finite, not admissible, outside the box or changes the sign of det D;
    2: does not reach I within T_MAX.  Also returns the normalized length needed."""
    sys = prob.sys
    fh = lambda z: sys.field(z)[0]
    lab = torch.full((len(x),), 2, dtype=torch.long)
    t_need = torch.full((len(x),), float("nan"))
    act = torch.arange(len(x))
    y = x.clone()
    s0 = sign_of_det(prob, y)
    ok0 = sys.field(y)[1] > 0
    lab[~ok0] = 1
    reached = (inner(y) < 1) & ok0
    lab[reached] = 0
    t_need[reached] = 0.0
    keep = ok0 & ~reached
    act, y, s0 = act[keep], y[keep], s0[keep]
    t, per_unit = 0.0, int(round(1.0 / STEP))
    while len(act) and t < T_MAX - 1e-9:
        for _ in range(per_unit):
            k1 = fh(y); k2 = fh(y + 0.5 * STEP * k1); k3 = fh(y + 0.5 * STEP * k2); k4 = fh(y + STEP * k3)
            y1 = y + STEP / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            good = torch.isfinite(y1).all(-1)
            y1 = torch.where(good[:, None], y1, y)
            good = good & prob.admissible(y1) & sys.in_box(y1) & (sign_of_det(prob, y1) == s0)
            lab[act[~good]] = 1
            act, y, s0 = act[good], y1[good], s0[good]
            if not len(act):
                break
        t += 1.0
        if len(act):
            reached = inner(y) < 1
            lab[act[reached]] = 0
            t_need[act[reached]] = t
            act, y, s0 = act[~reached], y[~reached], s0[~reached]
    return lab, t_need


def main():
    for name in sys.argv[1:]:
        t0 = time.time()
        models, levels = [], []
        for k in range(4):
            run = Path(f"runs/{name}_s{k}_rep3")
            res = json.loads((run / "certify_v3.json").read_text())
            prob, model, cfg = load(run)
            model.load_state_dict(torch.load(run / "snapshots" / res["snapshot"], map_location=cfg.device))
            models.append(model)
            levels.append(res["c2"])
        inner = Inner(prob, inner_level(cfg, prob))
        sysm = prob.sys
        seed = 9000000 + zlib.crc32(("unsafe_" + name).encode()) % 1000000
        g = torch.Generator().manual_seed(seed)
        n_in, n_inner, counts, t_max = 0, 0, [0, 0, 0], 0.0
        examples = []
        for i in range(0, N, CHUNK):
            m = min(CHUNK, N - i)
            x = sysm._box_map(torch.rand(m, dim_of(sysm), generator=g).to(sysm.device, sysm.dtype))
            with torch.no_grad():
                inI = _batched(inner, x) < 1
                inS = inI.clone()
                for model, c2 in zip(models, levels):
                    inS |= _batched(model, x) < c2
            xs = x[inS]
            n_in += len(xs)
            n_inner += int(inI.sum())
            lab, tn = classify(prob, inner, xs)
            for c in range(3):
                counts[c] += int((lab == c).sum())
            if (lab == 0).any():
                t_max = max(t_max, float(tn[lab == 0].max()))
            examples += xs[lab > 0][:5].tolist()
            print(f"{name}: {i + m} samples, {n_in} in S, unsafe so far {counts[1] + counts[2]}  [{time.time() - t0:.0f} s]", flush=True)
        k = counts[1] + counts[2]
        out = dict(case=name, N=N, seed=seed, distribution="uniform on the box, i.i.d.", step=STEP, t_max=T_MAX,
                   levels=levels, c_inner=inner.c_I, samples_in_S=n_in, samples_in_inner=n_inner,
                   share_of_box_in_S=100 * n_in / N, reached_inner=counts[0], left_or_crossed=counts[1],
                   not_reached=counts[2], unsafe=k, eps_k=cp_upper(k, N), confidence=1 - DELTA,
                   longest_time_to_inner=t_max, unsafe_examples=examples[:10], seconds=time.time() - t0)
        Path(f"results/unsafe_{name}.json").write_text(json.dumps(out, indent=1))
        print(f"{name}: N {N}, in S {n_in}, unsafe {k} (left {counts[1]}, not reached {counts[2]}), eps_k {out['eps_k']:.3g}, "
              f"longest time to I {t_max:.0f}  [{out['seconds']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
