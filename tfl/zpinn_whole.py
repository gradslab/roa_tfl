r"""
No-decomposition baseline.  The same network, data, losses, counterexamples, level selection and fine-tuning as zpinn.py,
without the term alpha q_ex in the value function: V_theta = max{h, Psi_theta}, and the running cost g is integrated
numerically along the short steps and enters the target, as in a Zubov solver that treats the running-cost integral as
unknown.  Implementation: alpha = 0 in the network (so V = Phi, psi = h) and kappa rho replaced by the full running cost
g + kappa rho, integrated in physical time by the trapezoid rule over the RK4 substeps (zpinn.precompute_next).  The factor
tanh(q_ex / q0) and the input features of the candidate are kept, so V = 0 on M.
With ZW_NO_COST=1 the running cost is dropped and the network learns the safety value function alone (safety-value baseline).
Run:  python -u zpinn_whole.py <zpinn flags>   (writes whole.flag into the run directory)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from problem import ConstrainedZubov
from zpinn import TrainConfig, ZidaniNet, collocation, precompute, train_vi


class WholeProblem(ConstrainedZubov):
    no_cost = False

    def rho(self, x):
        if self.no_cost or os.environ.get("ZW_NO_COST") == "1":
            return torch.zeros(len(x), device=x.device, dtype=x.dtype)
        extra = super().rho(x) * self.cfg.kappa if self.cfg.kappa > 0 else 0.0
        return self.g(x) + extra


def train(cfg: TrainConfig):
    if cfg.c_inner <= 0:
        from certify3 import case_name, inner_level
        f = Path(f"results/inner_{case_name(cfg)}.json")
        if f.exists():
            inner = json.loads(f.read_text())
            cfg.c_inner = inner_level(cfg, ConstrainedZubov(cfg))
            if cfg.kappa > 0 and inner["volume_of_box"] < 0.01:
                cfg.kappa = 0.0
    torch.manual_seed(cfg.seed)
    prob = WholeProblem(cfg)
    x = collocation(prob, cfg)
    with torch.no_grad():
        q0 = torch.quantile(prob.q_exact(x[:200000]), cfg.q0_quantile).item()
    model = ZidaniNet(prob, cfg, q0).to(cfg.device)
    model.alpha.fill_(0.0)
    model.kappa.fill_(1.0)
    out = Path(cfg.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "whole.flag").write_text("whole-value baseline: alpha = 0 in the network, running cost in the target\n")
    t0 = time.time()
    data = precompute(model, x)
    torch.save(model.state_dict(), out / "snapshots" / "epoch_000.pt")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    hist = train_vi(cfg, prob, model, x, data, out, t0)
    if cfg.ema > 0:
        model.load_state_dict(torch.load(sorted((out / "snapshots").glob("epoch_*.pt"))[-1]))
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out / "model.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f_ in TrainConfig.__dataclass_fields__.values():
        d = f_.default
        name = f"--{f_.name.replace('_', '-')}"
        if isinstance(d, bool):
            ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=d)
        elif isinstance(d, tuple):
            ap.add_argument(name, type=type(d[0]), nargs=len(d), default=d)
        else:
            ap.add_argument(name, type=type(d), default=d)
    a = vars(ap.parse_args())
    train(TrainConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in a.items()}))


if __name__ == "__main__":
    main()
