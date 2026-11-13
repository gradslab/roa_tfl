r"""
Per-seed summary and union of the four fine-tuned estimates of a case, scored against the reference simulation
(evaluation only).  Like each estimate, the union is tested on samples, not verified.
Run:  python -u union3.py circ3
"""
import json, sys
from pathlib import Path
import torch
from certify3 import load, Inner, ground_truth, inner_level, region

for name in sys.argv[1:]:
    rows, regs, regs_raw = [], [], []
    for k in range(4):
        for suf, store in (("", regs_raw), ("_rep3", regs)):
            run = Path(f"runs/{name}_s{k}{suf}")
            res = json.loads((run / "certify_v3.json").read_text())
            prob, model, cfg = load(run)
            model.load_state_dict(torch.load(run / "snapshots" / res["snapshot"], map_location=cfg.device))
            if not rows:
                inner = Inner(prob, inner_level(cfg, prob))
                xg, roa, label = ground_truth(prob)
                nr = roa.sum().item()
            store.append(region(model, inner, res["c2"], xg))
            rows.append(dict(run=run.name, c2=res["c2"], coverage=res["coverage"], false_positive=res["false_positive"],
                             iid_eps=res["iid"]["eps"], iid_N=res["iid"]["N"]))
    cov = lambda m: 100 * (m & roa).sum().item() / nr
    fp = lambda m: 100 * (m & ~roa).sum().item() / len(roa)
    u, u_raw = torch.stack(regs).any(0), torch.stack(regs_raw).any(0)
    out = dict(case=res["case"], rows=rows, inner_coverage=res["inner_coverage"], roa_fraction_of_box=res["roa_fraction_of_box"],
               union_repaired=cov(u), union_repaired_fp=fp(u), union_unrepaired=cov(u_raw), union_unrepaired_fp=fp(u_raw),
               mean_repaired=sum(r["coverage"] for r in rows if r["run"].endswith("_rep3")) / 4,
               mean_unrepaired=sum(r["coverage"] for r in rows if not r["run"].endswith("_rep3")) / 4)
    Path(f"results/summary_{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: unrepaired {[round(r['coverage'], 2) for r in rows if not r['run'].endswith('_rep3')]} mean {out['mean_unrepaired']:.2f}  "
          f"repaired {[round(r['coverage'], 2) for r in rows if r['run'].endswith('_rep3')]} mean {out['mean_repaired']:.2f}  "
          f"union {out['union_repaired']:.2f} % (FP {out['union_repaired_fp']:.3f} %)  inner alone {out['inner_coverage']:.2f} %")
