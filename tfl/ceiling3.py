r"""
Exact value function v by simulation on the reference states (evaluation only): the share of the simulated region of
attraction inside I u {v_exact < c2} for the level c2 of each fine-tuned run, i.e. what an exact fit would cover at that
level (column "Exact v" of Table 1).
Run:  python -u ceiling3.py circ3 sin6 quad3
"""
import json, math, sys, time
from pathlib import Path
import torch
from certify3 import load, Inner, ground_truth, inner_level
from check_value import value_sim

for name in sys.argv[1:]:
    t0 = time.time()
    out, cache = {}, None
    for k in range(4):
        run = Path(f"runs/{name}_s{k}_rep3")
        res = json.loads((run / "certify_v3.json").read_text())
        prob, model, cfg = load(run)
        s = prob.sys
        xg, roa, label = ground_truth(prob)
        nr = roa.sum().item()
        if cache is None:
            cfile = Path(f"results/vexact3_{name}.pt")
            if cfile.exists():
                ve = torch.load(cfile)
            else:
                extra = (lambda z: cfg.kappa * prob.rho(z)) if cfg.kappa > 0 else None
                lab, V = value_sim(prob, xg, [prob.alpha], [cfg.beta], T=40.0 if cfg.system == "quadrotor" else 30.0, extra=extra)
                ve = V[:, 0, 0].double()
                torch.save(ve, cfile)
            cache = ve
            inner = Inner(prob, inner_level(cfg, prob))
            with torch.no_grad():
                I = torch.cat([inner(c) for c in xg.split(20000)]) < 1
            agree = (torch.isfinite(ve) == roa).float().mean().item()
        reg = (cache < -math.log(1 - res["c2"])) | I
        out[run.name] = dict(c2=res["c2"], ceiling=100 * (reg & roa).sum().item() / nr, learned=res["coverage"])
    out["mean_ceiling"] = sum(v["ceiling"] for k, v in out.items() if k != "mean_ceiling") / 4
    out["mean_learned"] = sum(v["learned"] for k, v in out.items() if k not in ("mean_ceiling", "mean_learned")) / 4
    out["v_exact_lt_1_agrees_with_simulated_roa"] = agree
    out["ceiling_at"] = {str(c): 100 * (((cache < -math.log(1 - c)) | I) & roa).sum().item() / nr for c in (0.5, 0.9, 0.99, 0.999)}
    Path(f"results/ceiling_{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: ceiling at c2 {[round(v['ceiling'], 2) for k, v in out.items() if isinstance(v, dict) and 'ceiling' in v]} mean {out['mean_ceiling']:.2f}  "
          f"learned mean {out['mean_learned']:.2f}  {{v<1}} = ROA on {100 * agree:.2f} % of states  ceiling at 0.9/0.99/0.999: "
          f"{out['ceiling_at']['0.9']:.1f}/{out['ceiling_at']['0.99']:.1f}/{out['ceiling_at']['0.999']:.1f}  [{time.time() - t0:.0f} s]")
