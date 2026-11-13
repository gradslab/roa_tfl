r"""
Sensitivity of the rescoring to the stored quadratic (evaluation only).

core_proof.py verifies E = {q <= c0} with the float32 P and s of core.Problem on this machine, written to
results/core_<sys>.json; certify_core.py and unsafe_bench.py check that core.Problem gives exactly these constants before
using them (so every network evaluation here uses the verified q).  The runs were trained on another machine; P there
comes from the same solver but may differ in the last float32 bits (for instance with a Jacobian evaluated in float64,
the "exact" A).  This script forms that alternative P' = float32(lyap(A_exact)), reports the largest relative difference
from the stored P, and counts the samples whose membership in E or in {v_k < c1} (fine-tuned seeds, rescored c1) changes
when q is evaluated with P' instead of P.

Run:  python -u core_q_check.py vdp vdp3 pendulum con3d power      (writes results/core_q_check.json; merges systems)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_continuous_lyapunov

import core
import systems as S
from gt import ground_truth

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "core_q_check.json"


def main():
    torch.set_num_threads(2)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in sys.argv[1:]:
        rec = json.loads((ROOT / "results" / f"core_{name}.json").read_text())
        P = np.array(rec["constants"]["P_float32"])
        s = np.array(rec["constants"]["s_float32"])
        c0 = float(rec["c0"])
        sys_ = S.get(name)
        prob = core.Problem(sys_, core.Config(system=name))
        same_here = bool(np.array_equal(P, prob.P.cpu().double().numpy()) and np.array_equal(s, sys_.s.cpu().double().numpy()))
        x0 = torch.zeros(1, sys_.n, dtype=torch.float64, requires_grad=True)
        A = torch.autograd.functional.jacobian(lambda z: sys_._f(z)[0], x0)[:, 0, :].numpy()
        At = np.diag(s) @ A @ np.diag(1 / s)
        Palt = solve_continuous_lyapunov(At.T, -np.eye(sys_.n)).astype(np.float32).astype(np.float64)
        rel = float(np.max(np.abs(Palt - P) / np.maximum(np.abs(P), 1e-30) * (np.abs(P) > 0)))
        nz_diff = int(np.sum(Palt != P))
        g = torch.Generator().manual_seed(4711)
        xs = torch.cat([ground_truth(sys_)[0], sys_._map(torch.rand(1000000, sys_.n, generator=g))])
        P32, Pa32 = prob.P.clone(), torch.as_tensor(Palt, dtype=S.DT)
        with torch.no_grad():
            qa = torch.einsum("ni,ij,nj->n", xs * sys_.s, Pa32, xs * sys_.s)
            qp = torch.einsum("ni,ij,nj->n", xs * sys_.s, P32, xs * sys_.s)
        E_changes = int(((qa < c0) != (qp < c0)).sum())
        v_changes = []
        for k in range(4):
            run = ROOT / "runs" / f"{name}_s{k}_rep"
            cc = run / "certify_core.json"
            if not cc.exists():
                continue
            r = json.loads(cc.read_text())
            pr, model, _ = core.load(run, run / "snapshots" / r["snapshot"])
            with torch.no_grad():
                v1 = torch.cat([model(c) for c in xs.split(100000)])
                pr.P = Pa32
                v2 = torch.cat([model(c) for c in xs.split(100000)])
                pr.P = P32
            v_changes.append(dict(run=run.name, c1=r["c1"], c2=r["c2"], c1_changes=int(((v1 < r["c1"]) != (v2 < r["c1"])).sum()),
                                  c2_changes=int(((v1 < r["c2"]) != (v2 < r["c2"])).sum()),
                                  max_abs_v_difference=float((v1 - v2).abs().max())))
        out[name] = dict(stored_P_equals_core_Problem_here=same_here, A_exact=A.tolist(), P_alt=Palt.tolist(), P_stored=P.tolist(),
                         entries_differing=nz_diff, max_relative_difference=rel, n_samples=len(xs),
                         E_membership_changes=E_changes, v_membership=v_changes)
        print(name, {k: v for k, v in out[name].items() if k not in ("A_exact", "P_alt", "P_stored")}, flush=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
