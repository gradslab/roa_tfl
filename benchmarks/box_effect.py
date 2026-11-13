"""Evaluation only (python box_effect.py vdp_obs1 vdp_obs2 power_obs2 power_obs4 -> results/box_effect.json): how much of each reference ROA depends on counting box exits as failures.  On the same 50k reference samples,
simulate with the box replaced by [-1e3, 1e3]^2 (obstacles kept; Meng-Liu treat Omega as a region of interest) and
without the obstacles (box kept)."""
import json, sys, torch
from pathlib import Path
import systems as S
torch.set_num_threads(2)
out = {}
for name in sys.argv[1:]:
    s = S.get(name)
    d = torch.load(Path("results") / f"gt_{s.name}.pt")
    x, roa = d["x"], d["roa"]
    lo, hi = s.lo.clone(), s.hi.clone()
    s.lo, s.hi = torch.full_like(lo, -1e3), torch.full_like(hi, 1e3)
    nobox = torch.cat([s.simulate(c) for c in x.split(25000)]) == 0
    s.lo, s.hi = lo, hi
    cons = s.cons; s.cons = []
    noobs = torch.cat([s.simulate(c) for c in x.split(25000)]) == 0
    s.cons = cons
    r = dict(roa=100 * roa.float().mean().item(), roa_no_box_exit=100 * nobox.float().mean().item(),
             roa_no_obstacles=100 * noobs.float().mean().item(), ref_in_nobox=bool((roa & ~nobox).sum() == 0))
    out[name] = r
    print(name, r, flush=True)
(Path("results") / "box_effect.json").write_text(json.dumps(out, indent=1))
