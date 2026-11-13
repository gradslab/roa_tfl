r"""Data of the interactive slice explorer of the project page (docs/assets/data/slices_<case>.{json,bin}).

For the unicycle cases (circle, sinusoid) and 12 headings x 3 speeds, on a 121 x 121 grid of positions:
  bit 0  simulated region of attraction (reference simulation of every grid state, RK4, horizon 30),
  bit 1  estimate: union over the four fine-tuned seeds of I u {v < c2},
  bit 2  inner layer I = {gamma q_ex < c_I} (verified),
  uint16 V of seed 0 (V = -ln(1 - v), x 1000, capped at 60), for the level slider.
Also docs/assets/data/constants.json: the constants the page needs to evaluate q_ex, h and psi along a simulated trajectory.

Run from the repository root:  python web/export_slices.py      (about 30 minutes on a laptop CPU)
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TFL = Path(os.environ.get("ROA_TFL_DIR", ROOT / "tfl"))
sys.path.insert(0, str(TFL))
os.chdir(TFL)
from certify3 import load, Inner, inner_level  # noqa: E402

OUT = ROOT / "docs" / "assets" / "data"
N = 121
HEADINGS = [k * 30 for k in range(12)]          # degrees
SPEEDS = [0.5, 1.0, 1.5]                          # speed 1 + x4
torch.set_num_threads(int(os.environ.get("THREADS", "4")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    consts = {}
    for name, case in (("circ3", "circle"), ("sin6", "sinusoid")):
        runs = []
        for k in range(4):
            run = Path(f"runs/{name}_s{k}_rep3")
            res = json.loads((run / "certify_v3.json").read_text())
            prob, model, cfg = load(run)
            model.load_state_dict(torch.load(run / "snapshots" / res["snapshot"], map_location="cpu"))
            model.eval()
            runs.append((model, res["c2"]))
        s = prob.sys
        cI = inner_level(cfg, prob)
        inner = Inner(prob, cI)
        P = prob.P.cpu().numpy().tolist()
        consts[case] = dict(
            Lx=float(s.Lx), L=float(s.L), x4_lo=float(s.x4_lo), x4_hi=float(s.x4_hi), eps=[float(e) for e in s.eps],
            z_scale=s.z_scale.cpu().numpy().tolist(), P=P, gamma=float(prob.alpha), scale=prob.scale.cpu().numpy().tolist(),
            beta=float(cfg.beta), h_pow=float(cfg.h_pow), c_I=float(cI), c1=1 - math.exp(-float(cI)),
            c2=[float(c) for _, c in runs], gains=dict(k1=float(s.k1), k2=float(s.k2), k3=float(s.k3), eta2_ref=float(s.eta2_ref)),
            periodic_x1=bool(s.periodic_x1))
        t = torch.linspace(-1, 1, N)
        A, Bm = torch.meshgrid(t, t, indexing="xy")
        X1, X2 = (s.Lx * A).flatten(), (s.L * Bm).flatten()
        bits, vals, meta = [], [], []
        for hdg in HEADINGS:
            for sp in SPEEDS:
                x = torch.zeros(N * N, 4)
                x[:, 0], x[:, 1], x[:, 2], x[:, 3] = X1, X2, math.radians(hdg), sp - 1.0
                roa = torch.cat([s.simulate(c, T=30.0) == 0 for c in x.split(25000)])
                with torch.no_grad():
                    I = torch.cat([inner(c) for c in x.split(20000)]) < 1
                    est = I.clone()
                    V0 = None
                    for j, (m, c2) in enumerate(runs):
                        v = torch.cat([m(c) for c in x.split(20000)])
                        est |= v < c2
                        if j == 0:
                            V0 = -torch.log1p(-v.clamp(max=1 - 1e-12))
                b = roa.to(torch.uint8) | (est.to(torch.uint8) << 1) | (I.to(torch.uint8) << 2)
                bits.append(b.numpy().astype(np.uint8))
                vals.append((np.minimum(np.nan_to_num(V0.numpy(), nan=60.0, posinf=60.0), 60.0) * 1000).astype(np.uint16))
                cov = float((est & roa).sum() / max(1, roa.sum()))
                meta.append(dict(heading=hdg, speed=sp, roa=float(roa.float().mean()), coverage=cov,
                                 fp=int((est & ~roa).sum())))
                print(case, hdg, sp, f"roa {meta[-1]['roa']:.3f} cov {cov:.3f} fp {meta[-1]['fp']}", flush=True)
        blob = np.concatenate([np.stack(bits).ravel(), np.stack(vals).ravel().view(np.uint8)])
        (OUT / f"slices_{case}.bin").write_bytes(blob.tobytes())
        (OUT / f"slices_{case}.json").write_text(json.dumps(dict(
            n=N, headings=HEADINGS, speeds=SPEEDS, extent=[-float(s.Lx), float(s.Lx), -float(s.L), float(s.L)],
            layout="uint8 bits[slice][row][col] (bit0 roa, bit1 estimate, bit2 inner), then uint16 V*1000 of seed 0 [slice][row][col], little endian",
            c2_seed0=runs[0][1], slices=meta), indent=1))
    (OUT / "constants.json").write_text(json.dumps(consts, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
