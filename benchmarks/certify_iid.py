r"""Final test (iid.py, i.i.d. uniform samples) of every selected function and baseline; the existing certify.json files are not changed.
Writes runs/<run>/certify_iid.json and results/iid_{quad,pinn,union}_<system>.json.
Run:  python -u certify_iid.py vdp vdp3 pendulum con3d power"""
import json
import sys
import time
from pathlib import Path

import torch

import baselines as B
import core
import systems as S
from certify import EXTRA, region
from gt import ground_truth
from iid import iid_test, seed_of

R = Path("results")


def test(model, prob, name, c1, c2_old):
    s = prob.sys

    def step(x, T):
        y, ok = core.flow(s, x, T)
        return y, ok & s.admissible(x)

    r = iid_test(model, prob.inner, lambda u: s._map(u.to(S.DEV, S.DT)), step, lambda x: prob.xt(x).norm(dim=-1), s.n,
                 c1, c2_old, seed_of(name), horizons=EXTRA)
    xg, roa = ground_truth(s)
    nr = roa.sum().item()
    old, new = region(model, prob, c2_old, xg), region(model, prob, r["c2"], xg)
    r.update(coverage_old=100 * (old & roa).sum().item() / nr, coverage=100 * (new & roa).sum().item() / nr,
             false_positive=100 * (new & ~roa).sum().item() / len(roa))
    if r["first_failure"]:
        x0 = torch.tensor([r["first_failure"]["x"]], device=S.DEV, dtype=S.DT)
        r["first_failure"]["in_roa"] = bool(s.simulate(x0)[0] == 0)
    print(f"{name:22s} c2 {c2_old:.5f} -> {r['c2']:.5f}  coverage {r['coverage_old']:6.2f} -> {r['coverage']:6.2f} %  "
          f"FP {r['false_positive']:.3f} %  band {r['n_band']}  tested {r['n_tested']}  below-c1 "
          f"{r['n_below_c1_outside_inner']}/{r['n_below_c1_outside_inner_failed']}", flush=True)
    return r, new


for name in sys.argv[1:]:
    t0 = time.time()
    regs = []
    for suf in ("", "_rep"):
        for k in range(4):
            run = Path(f"runs/{name}_s{k}{suf}")
            old = json.loads((run / "certify.json").read_text())
            prob, model, cfg = core.load(run, run / "snapshots" / old["snapshot"])
            r, reg = test(model, prob, run.name, old["c1"], old["c2"])
            r.update(snapshot=old["snapshot"], coverage_stored=old["coverage"])
            (run / "certify_iid.json").write_text(json.dumps(r, indent=1))
            if suf:
                regs.append(reg)
    xg, roa = ground_truth(prob.sys)
    u = torch.stack(regs).any(0)
    un = dict(union=100 * (u & roa).sum().item() / roa.sum().item(), false_positive=100 * (u & ~roa).sum().item() / len(roa))
    (R / f"iid_union_{name}.json").write_text(json.dumps(un, indent=1))
    print(f"{name}: union {un['union']:.2f} % (FP {un['false_positive']:.3f} %)", flush=True)
    sys_ = S.get(name)
    prob = core.Problem(sys_, core.Config(system=name))
    prob.build_inner(verbose=False)
    q = json.loads((R / f"baseline_quad_{name}.json").read_text())[0]
    r, _ = test(B.Quad(prob), prob, f"quad_{name}", q["c1"], q["c2"])
    r["coverage_stored"] = q["coverage"]
    (R / f"iid_quad_{name}.json").write_text(json.dumps([r], indent=1))
    res = []
    for b in json.loads((R / f"baseline_pinn_{name}.json").read_text()):
        m = B.PINN(sys_.n).to(S.DEV)
        m.s = sys_.s
        m.load_state_dict(torch.load(R / f"pinn_{name}_s{b['seed']}.pt", map_location=S.DEV))
        r, _ = test(B.Clip(m), prob, f"pinn_{name}_s{b['seed']}", b["c1"], b["c2"])
        r.update(seed=b["seed"], coverage_stored=b["coverage"])
        res.append(r)
    (R / f"iid_pinn_{name}.json").write_text(json.dumps(res, indent=1))
    print(f"  [{time.time() - t0:.0f} s]", flush=True)
