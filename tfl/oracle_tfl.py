r"""Oracle level of the PINN baselines on the TFL closed loops: for every run with pinn.flag and every snapshot, the
largest coverage of I u {v < c} over all levels c at which no reference state outside the simulated region of
attraction lies inside.  An upper bound for any level rule.  Writes results/oracle_tfl.json.
Run:  python -u oracle_tfl.py"""
import json
from pathlib import Path

import torch

from certify3 import Inner, _batched, ground_truth, inner_level, load


def oracle(run):
    prob, model, cfg = load(run)
    xg, roa, _ = ground_truth(prob)
    with torch.no_grad():
        in_i = _batched(Inner(prob, inner_level(cfg, prob)), xg) < 1
    bad = ~roa & ~in_i
    nr = roa.sum().item()
    best = dict(coverage=100 * (in_i & roa).sum().item() / nr, snapshot=None, level=None)
    for sp in sorted((Path(run) / "snapshots").glob("epoch_*.pt")):
        model.load_state_dict(torch.load(sp, map_location=prob.sys.device))
        with torch.no_grad():
            v = torch.nan_to_num(_batched(model, xg), nan=1.0)
        c = v[bad].min().item() if bad.any() else 1.0
        cov = 100 * ((in_i | (v < c)) & roa).sum().item() / nr
        if cov > best["coverage"]:
            best = dict(coverage=cov, snapshot=sp.name, level=c)
    best["inner_coverage"] = 100 * (in_i & roa).sum().item() / nr
    return best


def main():
    torch.set_num_threads(2)
    runs = sorted(d for d in Path("runs").iterdir() if (d / "pinn.flag").exists())
    out = {}
    for run in runs:
        out[run.name] = oracle(run)
        print(f"{run.name:26s} oracle {out[run.name]['coverage']:6.2f} %", flush=True)
    Path("results/oracle_tfl.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
