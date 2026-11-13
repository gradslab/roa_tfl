"""Slice data of the TFL cases (as in tfl/figs3.py), cached in _cache/. Run: python tfl_data.py"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from common import CACHE, TFL, research

GRID = 321
CASES = {"circ3": "circle", "sin6": "sinusoid", "quad3": "quadrotor"}


def load_case(name):
    with research(TFL):
        from certify3 import load, Inner, inner_level
        seeds = []
        for k in range(4):
            run = Path(f"runs/{name}_s{k}_rep3")
            res = json.loads((run / "certify_v3.json").read_text())
            prob, model, cfg = load(run)
            model.load_state_dict(torch.load(run / "snapshots" / res["snapshot"], map_location="cpu"))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            seeds.append((model, float(res["c2"])))
            if k == 0:
                prob0, inner = prob, Inner(prob, inner_level(cfg, prob))
    return prob0, seeds, inner


def slice_states(s, case, n=GRID):
    t = torch.linspace(-1, 1, n)
    A, Bm = torch.meshgrid(t, t, indexing="xy")
    if case == "quadrotor":
        x = torch.zeros(n * n, 6)
        x[:, 0], x[:, 1] = s.L * A.flatten(), s.L * Bm.flatten()
        ang = torch.atan2(x[:, 1], x[:, 0])
        x[:, 2], x[:, 3] = -s.eta_ref * torch.sin(ang), s.eta_ref * torch.cos(ang)
        x[:, 4], x[:, 5] = 0.0, 1.0
        return x, (-s.L, s.L, -s.L, s.L)
    x = torch.zeros(n * n, 4)
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = s.Lx * A.flatten(), s.L * Bm.flatten(), math.pi / 2, 0.0
    return x, (-s.Lx, s.Lx, -s.L, s.L)


def slice_data(name):
    f = CACHE / f"slice_{name}.npz"
    if f.exists():
        return dict(np.load(f))
    case = CASES[name]
    prob, seeds, inner = load_case(name)
    s = prob.sys
    x, ext = slice_states(s, case)
    roa = torch.load(TFL / "results" / f"slice_gt_{name}_{GRID}.pt")
    with torch.no_grad():
        I = torch.cat([inner(c) for c in x.split(20000)]).reshape(GRID, GRID).numpy()
        V = np.stack([torch.cat([m(c) for c in x.split(20000)]).reshape(GRID, GRID).numpy() for m, _ in seeds])
    c2 = np.array([c for _, c in seeds])
    G = (V < c2[:, None, None]).any(0)
    d = dict(R=roa.reshape(GRID, GRID).numpy(), G=G, I=I, V=V, c2=c2, ext=np.array(ext),
             xs=np.linspace(ext[0], ext[1], GRID), ys=np.linspace(ext[2], ext[3], GRID))
    np.savez_compressed(f, **d)
    return d


if __name__ == "__main__":
    for n in CASES:
        d = slice_data(n)
        R, G = d["R"], d["G"]
        print(n, "ROA share", R.mean().round(3), "estimate share", G.mean().round(3), "estimate outside ROA", int((G & ~R).sum()),
              "c2", d["c2"].round(4))
