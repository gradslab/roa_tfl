r"""Simulated ground truth (evaluation only, never used in training or selection): 50k Sobol samples, seed 4242.
Run:  python -u gt.py vdp pendulum con3d power"""
import sys
import time
from pathlib import Path

import torch

import systems

N, SEED = 50000, 4242


def ground_truth(sys_, cache=True):
    p = Path(__file__).parent / "results" / f"gt_{sys_.name}.pt"
    if cache and p.exists():
        d = torch.load(p, map_location=sys_.device)
        return d["x"], d["roa"]
    x = sys_.sample_box(N, seed=SEED)
    lab = torch.cat([sys_.simulate(c) for c in x.split(25000)])
    roa = lab == 0
    torch.save(dict(x=x, roa=roa, label=lab), p)
    return x, roa


if __name__ == "__main__":
    for name in sys.argv[1:]:
        s = systems.get(name)
        t = time.time()
        x, roa = ground_truth(s, cache=False)
        lab = torch.load(Path(__file__).parent / "results" / f"gt_{s.name}.pt")["label"]
        print(f"{name}: ROA {100 * roa.float().mean():.2f} % of the box; labels {torch.bincount(lab.cpu(), minlength=4).tolist()} "
              f"[{time.time() - t:.0f} s]", flush=True)
        if name == "con3d":
            s.cons = []
            lab2 = torch.cat([s.simulate(c) for c in x.split(25000)])
            print(f"   without the x3 limit and obstacle: ROA {100 * (lab2 == 0).float().mean():.2f} %", flush=True)
