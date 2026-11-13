r"""Union of the estimates of several runs, each {v_k < c2_k} u inner region.  Like each estimate, the union is tested on
samples, not verified.  Coverage against the reference simulation is evaluation only.
Run:  python -u union.py runs/vdp_s0_rep runs/vdp_s1_rep ...   (writes results/union_<system>.json)"""
import json
import sys
from pathlib import Path

import torch

import core
from certify import region
from gt import ground_truth

runs = [Path(r) for r in sys.argv[1:]]
regs = []
for r in runs:
    res = json.loads((r / "certify.json").read_text())
    prob, model, cfg = core.load(r, r / "snapshots" / res["snapshot"])
    if not regs:
        xg, roa = ground_truth(prob.sys)
    regs.append(region(model, prob, res["c2"], xg))
nr = roa.sum().item()
cov = lambda m: 100 * (m & roa).sum().item() / nr
u = torch.stack(regs).any(0)
out = dict(runs=[str(r) for r in runs], singles=[cov(m) for m in regs], union=cov(u),
           false_positive=100 * (u & ~roa).sum().item() / len(roa))
Path(f"results/union_{cfg.system}{'_rep' if runs[0].name.endswith('_rep') else ''}.json").write_text(json.dumps(out, indent=1))
print(f"{cfg.system}: singles {' / '.join(f'{c:.2f}' for c in out['singles'])} -> union {out['union']:.2f} % (FP {out['false_positive']:.2f} %)", flush=True)
