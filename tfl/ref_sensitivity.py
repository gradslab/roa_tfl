r"""Sensitivity of the reference labels to the integration step: 5000 of the 50k reference states, RK4 step 0.0025 against the reference 0.01.
Run:  python -u ref_sensitivity.py   (writes results/ref_sensitivity.json)"""
import json
from pathlib import Path
import torch
from certify3 import case_name, ground_truth
from problem import ConstrainedZubov, ProblemConfig

CFGS = [ProblemConfig(beta=0.1, h_pow=2.0, device="cpu"), ProblemConfig(path="y-sin(x)", pi="x", periodic_x1=True, beta=0.1, h_pow=2.0, device="cpu"),
        ProblemConfig(system="quadrotor", beta=0.1, h_pow=2.0, device="cpu")]
out = {}
for cfg in CFGS:
    prob = ConstrainedZubov(cfg)
    xg, roa, label = ground_truth(prob)
    idx = torch.linspace(0, len(xg) - 1, 5000).long()
    T = 40.0 if cfg.system == "quadrotor" else 30.0
    fine = prob.sys.simulate(xg[idx], T=T, dt=0.0025) == 0
    ch = (fine != roa[idx]).float().mean().item()
    out[case_name(cfg)] = dict(n=len(idx), changed_pct=100 * ch)
    print(case_name(cfg), out[case_name(cfg)], flush=True)
Path("results/ref_sensitivity.json").write_text(json.dumps(out, indent=1))
