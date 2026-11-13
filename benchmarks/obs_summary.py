r"""Summary of the obstacle benchmarks of Meng & Liu (ECC 2026) under the final test (iid.py); writes results/obs_summary.json.
Coverage = share of the reference ROA inside the estimate, false positive (FP) = share of the box inside the estimate but outside the
reference ROA (both on the 50k reference samples, evaluation only).  Run from benchmarks/ after run_obs.sh."""
import json
import statistics as st
import sys
from pathlib import Path

import torch

js = lambda p: json.loads(Path(p).read_text())
ms = lambda v: dict(mean=st.mean(v), std=st.stdev(v) if len(v) > 1 else 0.0)
fmt = lambda v: f"{st.mean(v):5.1f} +- {st.stdev(v):4.1f}"
names = sys.argv[1:] or ["vdp_obs1", "vdp_obs2", "power_obs2", "power_obs4"]
out = {}
print(f"{'system':11s} {'ref ROA':>7s} {'quad':>5s} {'Zubov PINN':>14s} {'DZ PINN':>14s} {'ours':>14s} {'ours ft':>14s} {'union':>6s} {'max FP':>6s}")
for name in names:
    gt = torch.load(f"results/gt_{name}.pt")
    o = [js(f"runs/{name}_s{k}/certify_iid.json") for k in range(4)]
    r = [js(f"runs/{name}_s{k}_rep/certify_iid.json") for k in range(4)]
    q = js(f"results/iid_quad_{name}.json")[0]
    p1 = js(f"results/iid_pinn_{name}.json")
    p2, dz = js(f"results/baseline_pinn2_{name}.json"), js(f"results/baseline_dz_{name}.json")
    u, u0 = js(f"results/iid_union_{name}.json"), js(f"results/union_{name}.json")
    c = lambda L, k="coverage": [x[k] for x in L]
    lv = lambda L: [dict(c1=x["c1"], c2_old=x["c2_old"], c2=x["c2"]) for x in L]
    grid = lambda d: [dict(factor=g["factor"], coverage=g["coverage"], false_positive=g["false_positive"]) for g in d["grid"]]
    fps = c(o, "false_positive") + c(r, "false_positive") + [u["false_positive"], q["false_positive"]] + c(p1, "false_positive") \
        + c(p2["runs"], "false_positive") + c(dz["runs"], "false_positive")
    out[name] = dict(
        roa_share=100 * gt["roa"].float().mean().item(),
        ours=dict(coverage=c(o), coverage_selection=c(o, "coverage_old"), false_positive=c(o, "false_positive"), levels=lv(o), **ms(c(o))),
        ours_finetuned=dict(coverage=c(r), coverage_selection=c(r, "coverage_old"), false_positive=c(r, "false_positive"), levels=lv(r), **ms(c(r))),
        union_finetuned=dict(coverage=u["union"], false_positive=u["false_positive"], coverage_selection=u0["union"]),
        quad=dict(coverage=q["coverage"], false_positive=q["false_positive"], levels=lv([q])[0]),
        zubov_pinn_default=dict(coverage=c(p1), false_positive=c(p1, "false_positive"), levels=lv(p1), **ms(c(p1))),
        zubov_pinn_search=dict(factor=p2["runs"][0]["factor"], grid=grid(p2), coverage=c(p2["runs"]), false_positive=c(p2["runs"], "false_positive"),
                               levels=lv(p2["runs"]), **ms(c(p2["runs"]))),
        dirichlet_zubov_pinn=dict(factor=dz["runs"][0]["factor"], grid=grid(dz), coverage=c(dz["runs"]), false_positive=c(dz["runs"], "false_positive"),
                                  levels=lv(dz["runs"]), **ms(c(dz["runs"]))),
        max_false_positive=dict(ours=max(c(o, "false_positive") + c(r, "false_positive") + [u["false_positive"]]), all=max(fps)),
        n_lowered_by_final_test=sum(x["c2"] < x["c2_old"] for x in o + r))
    d = out[name]
    print(f"{name:11s} {d['roa_share']:7.2f} {q['coverage']:5.1f} {fmt(c(p2['runs'])):>14s} {fmt(c(dz['runs'])):>14s} {fmt(c(o)):>14s} "
          f"{fmt(c(r)):>14s} {u['union']:6.1f} {d['max_false_positive']['all']:6.3f}   (factors: Zubov {d['zubov_pinn_search']['factor']:g}, "
          f"DZ {d['dirichlet_zubov_pinn']['factor']:g}; default Zubov {fmt(c(p1))})")
Path("results/obs_summary.json").write_text(json.dumps(out, indent=1))
