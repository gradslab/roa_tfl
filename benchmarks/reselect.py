r"""Scale selection of the Zubov PINN (pinn2) and the Dirichlet-Zubov PINN (dz) under the reported scoring
(certify_core.py --relaxed: verified inner layer E, relaxed rule, no lower clamp).  The seed-0 grid is scored, a
Dirichlet-Zubov grid whose best factor is its largest is extended to 300 and 1000, ties go to the smallest factor, and
seeds 1 to 3 are trained at the selected factor.  Every seed is scored with the relaxed and with the strict rule.
Writes results/grid_relaxed_<kind>_<sys>.json, results/baseline_<kind>_core_relaxed_<sys>.json and
results/baseline_<kind>_core_noclamp_<sys>.json.
Run:  python -u reselect.py vdp vdp3 ..."""
import json
import shutil
import sys
import time

import torch

import baseline_dz as DZ
import baselines as B
import certify_core as C

R = C.R


def network(kind, name, sys_, prob, seed, factor):
    ck = R / f"{kind}_{name}_s{seed}_f{factor:g}.pt"
    if ck.exists():
        return C.pinn_net(sys_, ck), 0.0
    t0 = time.time()
    m = B.train_pinn(sys_, prob, seed, factor=factor) if kind == "pinn2" else DZ.train(sys_, prob, seed, factor)
    torch.save(m.state_dict(), ck)
    return m, time.time() - t0


def score(m, sys_, prob, c0, run, relaxed):
    C.RELAXED, C.NOCLAMP = relaxed, True
    C.set_core(prob, c0)
    r, _ = C.rescore(C.wrap_dz(m, sys_), prob, run)
    return r


def select(kind, name):
    sys_, prob = C.base_prob(name)
    c_samp, c0 = prob.inner_c, C.core_level(name, prob)
    old = json.loads((R / f"baseline_{kind}_{name}.json").read_text())
    factors = [1.0, 3.0, 10.0, 30.0, 100.0] + ([300.0] if kind == "pinn2" else [])
    rows = {}

    def grid_row(f):
        C.set_core(prob, c_samp)
        m, sec = network(kind, name, sys_, prob, 0, f)
        r = score(m, sys_, prob, c0, f"{kind}_{name}_s0", True)
        rows[f] = dict(factor=f, coverage=r["coverage"], c1=r["c1"], c2_old=r["c2_old"], c2=r["c2"],
                       false_positive=r["false_positive"], train_seconds=sec)

    for f in factors:
        grid_row(f)
    best = lambda: min(rows, key=lambda f: (-round(rows[f]["coverage"], 6), f))
    if kind == "dz":
        for f in (300.0, 1000.0):
            if best() == max(rows) and f not in rows:
                grid_row(f)
    sel = best()
    relaxed, strict = [], []
    for seed in range(4):
        C.set_core(prob, c_samp)
        m, sec = network(kind, name, sys_, prob, seed, sel)
        for rule, out in ((True, relaxed), (False, strict)):
            r = score(m, sys_, prob, c0, f"{kind}_{name}_s{seed}", rule)
            r.update(seed=seed, factor=sel, train_seconds=sec,
                     scoring="W = min(net, 1), no lower clamp", rule="relaxed" if rule else "strict")
            out.append(r)
    for tag, res in (("_relaxed", relaxed), ("_noclamp", strict)):
        f_ = R / f"baseline_{kind}_core{tag}_{name}.json"
        prev = R / f"baseline_{kind}_core{tag}_{name}_prev.json"
        if f_.exists() and not prev.exists():
            shutil.copy(f_, prev)
        f_.write_text(json.dumps(res, indent=1))
    out = dict(system=name, kind=kind, selected_earlier=old["runs"][0]["factor"], selected=sel,
               grid=[rows[f] for f in sorted(rows)])
    (R / f"grid_relaxed_{kind}_{name}.json").write_text(json.dumps(out, indent=1))
    print(f"SELECT {kind} {name}: earlier {out['selected_earlier']:g} now {sel:g}  seeds " +
          " ".join(f"{r['coverage']:.1f}" for r in relaxed), flush=True)


def main():
    torch.set_num_threads(1)
    for name in sys.argv[1:]:
        for kind in ("pinn2", "dz"):
            select(kind, name)


if __name__ == "__main__":
    main()
