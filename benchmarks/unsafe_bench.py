r"""
Direct test of unsafe states in the estimates of the equilibrium benchmarks (evaluation only; nothing here changes an
estimate).  The benchmark version of tfl/unsafe3.py.

S is the union over the four fine-tuned seeds of the reported estimate I u {v_k < c2_k}: the snapshot of certify.json,
the level c2_k after the final test (certify_iid.json) and the sampled inner region I = {q < c_in}, exactly the sets of
the reported union coverage (certify_iid.py, results/iid_union_<sys>.json).  S is fixed before any sample is drawn.
N = 10^6 i.i.d. uniform samples of the box are drawn with seeds disjoint from every other test (11,000,000 + crc32 %
10^6; the final tests use 7,000,000 + ..., unsafe3.py 9,000,000 + ...).  A sample in S counts as safe if its trajectory of
the normalized field (RK4, substep STEP = 0.025 as in the final test) reaches the verified core E = {q < c0}
(core_proof.py) within normalized length T_MAX while every substep is finite, admissible and inside the box.  E is
forward invariant, admissible and lies in the region of attraction, so a safe sample lies in the constrained region of
attraction up to the integration error.  The other samples of S form the set U, which contains S minus the region of
attraction.  With k samples in U, the one-sided Clopper-Pearson bound eps_k satisfies P^N[ P(U) > eps_k ] <= 1e-6.
U contains the corresponding set of every seed, since each seed's estimate lies in S.

The keys *_inner of the output keep the names of the TFL files (results of unsafe3.py); here they refer to the
verified core E, the target set of the test.

Run:  python -u unsafe_bench.py vdp vdp3 pendulum con3d power      (writes results/unsafe_<sys>.json, after core_proof.py)
      python -u unsafe_bench.py --core vdp ...    (rescored estimates of certify_core.py; results/unsafe_core_<sys>.json)
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_dist

import core
import systems as S
from certify import region

ROOT = Path(__file__).parent
N = 1000000
CHUNK = 200000
STEP = 0.025
T_MAX = 200.0
DELTA = 1e-6
MAX_EXAMPLES = 2000


def cp_upper(k, n, delta=DELTA):
    return 1.0 if k >= n else float(beta_dist.ppf(1 - delta, k + 1, n - k))


MODE = "sampled"


def seed_of(name):
    if MODE == "core_relaxed":
        return 15000000 + zlib.crc32(("unsafe_core_relaxed_" + name).encode()) % 1000000
    if MODE == "core":
        return 13000000 + zlib.crc32(("unsafe_core_" + name).encode()) % 1000000
    return 11000000 + zlib.crc32(("unsafe_bench_" + name).encode()) % 1000000


def out_name(name):
    return {"core": f"unsafe_core_{name}.json", "core_relaxed": f"unsafe_core_relaxed_{name}.json"}.get(MODE, f"unsafe_{name}.json")


def parse_args(argv):
    global MODE
    names = [a for a in argv if not a.startswith("--")]
    if "--core" in argv:
        MODE = "core_relaxed" if "--relaxed" in argv else "core"
    return names


class Core:
    """E = {q < c0} with q(x) = x^T M x evaluated in float64 from the float32 constants of the pipeline."""

    def __init__(self, name, prob):
        rec = json.loads((ROOT / "results" / f"core_{name}.json").read_text())
        P = np.array(rec["constants"]["P_float32"])
        s = np.array(rec["constants"]["s_float32"])
        if not (np.array_equal(P, prob.P.cpu().double().numpy()) and np.array_equal(s, prob.sys.s.cpu().double().numpy())):
            raise SystemExit(f"{name}: the quadratic of the runs differs from the one of core_{name}.json")
        Ps = 0.5 * (P + P.T)
        self.M = torch.as_tensor(s[:, None] * Ps * s[None, :], dtype=torch.float64)
        self.c0 = float(rec["c0"])

    def q(self, x):
        z = x.double()
        return torch.einsum("ni,ij,nj->n", z, self.M.to(z.device), z)

    def inside(self, x):
        return self.q(x) < self.c0


def load_union(name):
    """The four fine-tuned seeds: (model, prob, c2) with the snapshot of certify.json and c2 of certify_iid.json."""
    out = []
    for k in range(4):
        run = ROOT / "runs" / f"{name}_s{k}_rep"
        old = json.loads((run / "certify.json").read_text())
        iid = json.loads((run / "certify_iid.json").read_text())
        if iid.get("snapshot", old["snapshot"]) != old["snapshot"]:
            raise SystemExit(f"{run}: snapshot of certify_iid.json differs from certify.json")
        prob, model, cfg = core.load(run, run / "snapshots" / old["snapshot"])
        c2 = float(iid["c2"])
        if MODE in ("core", "core_relaxed"):
            cc = json.loads((run / ("certify_core_relaxed.json" if MODE == "core_relaxed" else "certify_core.json")).read_text())
            if cc["snapshot"] != old["snapshot"]:
                raise SystemExit(f"{run}: snapshot of certify_core.json differs from certify.json")
            prob.inner_c = float(json.loads((ROOT / "results" / f"core_{name}.json").read_text())["c0"])
            c2 = float(cc["c2"])
        out.append(dict(run=run.name, snapshot=old["snapshot"], c2=c2, model=model, prob=prob))
    return out


@torch.no_grad()
def in_S(members, x, batch=50000):
    inS = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for m in members:
        inS |= region(m["model"], m["prob"], m["c2"], x)
    return inS


@torch.no_grad()
def seeds_containing(members, x):
    return [[k for k, m in enumerate(members) if bool(region(m["model"], m["prob"], m["c2"], x[i:i + 1])[0])]
            for i in range(len(x))]


@torch.no_grad()
def classify(sys_, E, x, step=STEP, t_max=T_MAX):
    """0: reaches E safely; 1: a substep is not finite, not admissible or outside the box; 2: does not reach E within
    t_max.  Returns the label, the normalized length needed to reach E and the failing constraint (index into
    sys_.dist_names(), -1 for a non-finite state, -2 if none)."""
    F = sys_.field
    n = len(x)
    lab = torch.full((n,), 2, dtype=torch.long)
    t_need = torch.full((n,), float("nan"), dtype=torch.float64)
    fail = torch.full((n,), -2, dtype=torch.long)
    y = x.clone()
    act = torch.arange(n)

    def bad_of(z):
        fin = torch.isfinite(z).all(-1)
        viol = torch.nan_to_num(sys_.distances(z), nan=-1.0) <= 0
        inbox = sys_.in_box(z)
        ok = fin & ~viol.any(-1) & inbox
        first = torch.where(viol.any(-1), viol.float().argmax(-1), torch.full((len(z),), -2, dtype=torch.long))
        first = torch.where(~inbox & ~viol.any(-1), torch.full_like(first, -3), first)
        first = torch.where(fin, first, torch.full_like(first, -1))
        return ok, first

    ok, first = bad_of(y)
    lab[~ok], fail[~ok] = 1, first[~ok]
    reached = ok & E.inside(y)
    lab[reached] = 0
    t_need[reached] = 0.0
    keep = ok & ~reached
    act, y = act[keep], y[keep]
    nsub = int(round(t_max / step))
    for i in range(1, nsub + 1):
        if not len(act):
            break
        k1 = F(y); k2 = F(y + 0.5 * step * k1); k3 = F(y + 0.5 * step * k2); k4 = F(y + step * k3)
        y = y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ok, first = bad_of(y)
        lab[act[~ok]], fail[act[~ok]] = 1, first[~ok]
        act, y = act[ok], y[ok]
        reached = E.inside(y)
        lab[act[reached]] = 0
        t_need[act[reached]] = i * step
        act, y = act[~reached], y[~reached]
    return lab, t_need, fail


def main():
    torch.set_num_threads(2)
    for name in parse_args(sys.argv[1:]):
        t0 = time.time()
        members = load_union(name)
        sys_ = members[0]["prob"].sys
        E = Core(name, members[0]["prob"])
        from gt import ground_truth
        xg, roa = ground_truth(sys_)
        u = in_S(members, xg)
        union_cov = 100 * (u & roa).sum().item() / roa.sum().item()
        stored = json.loads((ROOT / "results" / ({"core": f"core_union_{name}.json", "core_relaxed": f"core_union_relaxed_{name}.json"}.get(MODE, f"iid_union_{name}.json"))).read_text())["union"]
        print(f"{name}: union coverage of S {union_cov:.4f} % (stored {stored:.4f} %), core level c0 {E.c0:.6g}", flush=True)
        seed = seed_of(name)
        g = torch.Generator().manual_seed(seed)
        n_in, n_core, n_inner, counts, t_max = 0, 0, 0, [0, 0, 0], 0.0
        fails = {}
        examples, ex_lab, ex_fail = [], [], []
        names = sys_.dist_names()
        for i in range(0, N, CHUNK):
            m = min(CHUNK, N - i)
            x = sys_._map(torch.rand(m, sys_.n, generator=g).to(S.DEV, S.DT))
            inS = in_S(members, x)
            with torch.no_grad():
                inI = torch.cat([members[0]["prob"].q(c) < float(members[0]["model"].inner_c) for c in x.split(50000)])
            xs = x[inS]
            n_in += len(xs)
            n_inner += int(inI.sum())
            n_core += int(E.inside(xs).sum())
            lab, tn, fl = classify(sys_, E, xs)
            for c in range(3):
                counts[c] += int((lab == c).sum())
            if (lab == 0).any():
                t_max = max(t_max, float(tn[lab == 0].max()))
            bad = lab > 0
            for j in torch.nonzero(bad).squeeze(-1).tolist():
                key = ("not reached" if lab[j] == 2 else
                       {-1: "not finite", -3: "outside the closed box"}.get(int(fl[j]), names[int(fl[j])] if fl[j] >= 0 else "?"))
                fails[key] = fails.get(key, 0) + 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append(xs[j].tolist())
                    ex_lab.append(int(lab[j]))
                    ex_fail.append(key)
            print(f"{name}: {i + m} samples, {n_in} in S, unsafe so far {counts[1] + counts[2]}  [{time.time() - t0:.0f} s]", flush=True)
        k = counts[1] + counts[2]
        ex_t = torch.tensor(examples, dtype=S.DT, device=S.DEV).reshape(-1, sys_.n)
        seeds = seeds_containing(members, ex_t) if len(examples) else []
        inI_ex = (members[0]["prob"].q(ex_t) < float(members[0]["model"].inner_c)).tolist() if len(examples) else []
        out = dict(case=name, N=N, seed=seed, distribution="uniform on the box, i.i.d.", step=STEP, t_max=T_MAX,
                   levels=[m["c2"] for m in members], c_inner=E.c0, samples_in_S=n_in, samples_in_inner=n_core,
                   share_of_box_in_S=100 * n_in / N, reached_inner=counts[0], left_or_crossed=counts[1],
                   not_reached=counts[2], unsafe=k, eps_k=cp_upper(k, N), confidence=1 - DELTA,
                   longest_time_to_inner=t_max,
                   unsafe_examples=examples[:10], seconds=time.time() - t0,
                   target_set="verified core E = {q < c0} of core_proof.py (the *_inner keys refer to E)",
                   c_core=E.c0, samples_in_core=n_core, longest_time_to_core=t_max,
                   c_inner_sampled=float(members[0]["model"].inner_c), samples_in_sampled_inner=n_inner,
                   runs=[m["run"] for m in members], snapshots=[m["snapshot"] for m in members],
                   union_coverage_of_S=union_cov, union_coverage_stored=stored,
                   failing_constraint=fails,
                   unsafe_all=[dict(state=examples[j], label=ex_lab[j], failing=ex_fail[j], seeds_containing=seeds[j],
                                    in_sampled_inner=bool(inI_ex[j])) for j in range(len(examples))])
        out["mode"] = MODE
        (ROOT / "results" / out_name(name)).write_text(json.dumps(out, indent=1))
        print(f"{name}: N {N}, in S {n_in}, unsafe {k} (left {counts[1]}, not reached {counts[2]}; {fails}), "
              f"eps_k {out['eps_k']:.3g}, longest time to E {t_max:.2f}  [{out['seconds']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
