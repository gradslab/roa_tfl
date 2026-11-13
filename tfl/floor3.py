r"""
Share of the simulated region of attraction that the bound V_theta >= alpha q_ex alone excludes at the levels c2 of the
fine-tuned runs (paper, "What limits the outer layer?").  Evaluation only.
Run:  python -u floor3.py   (writes results/floor.json)
"""
import json, math, sys, torch
sys.path.insert(0,'.')
from pathlib import Path
from certify3 import load, ground_truth
out={}
for name,case in (('circ3','circle'),('sin6','sinusoid'),('quad3','quadrotor')):
    shares=[]
    for k in range(4):
        run=Path(f'runs/{name}_s{k}_rep3')
        res=json.loads((run/'certify_v3.json').read_text())
        prob,model,cfg=load(run)
        xg,roa,label=ground_truth(prob)
        with torch.no_grad():
            q=torch.cat([prob.alpha*prob.q_exact(c) for c in xg[roa].split(20000)])
        floor=1-torch.exp(-q)
        shares.append(100*(floor>=res['c2']).float().mean().item())
    out[case]=dict(share_of_roa_with_floor_above_c2=shares,mean=sum(shares)/4)
    print(case,out[case],flush=True)
Path('results/floor.json').write_text(json.dumps(out,indent=1))
