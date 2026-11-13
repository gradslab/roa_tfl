r"""Coverage before and after fine-tuning under the final test (iid.py); writes results/iid_summary.json.  Run from benchmarks/ after certify_iid.py."""
import json
import statistics as st
from pathlib import Path

H, Z = Path("."), Path("../tfl")
js = lambda p: json.loads(Path(p).read_text())
ms = lambda v: f"{st.mean(v):5.1f} +- {st.stdev(v):4.1f}"
out = {}
print(f"{'system':10s} {'ours old':>14s} {'ours iid':>14s} {'repair old':>14s} {'repair iid':>14s} {'union':>6s} {'quad':>11s} {'pinn old':>14s} {'pinn iid':>14s}")
for name in ("vdp", "vdp3", "pendulum", "con3d", "power"):
    o = [js(H / f"runs/{name}_s{k}/certify_iid.json") for k in range(4)]
    r = [js(H / f"runs/{name}_s{k}_rep/certify_iid.json") for k in range(4)]
    q = js(H / f"results/iid_quad_{name}.json")[0]
    p = js(H / f"results/iid_pinn_{name}.json")
    u = js(H / f"results/iid_union_{name}.json")
    c = lambda L, k="coverage": [x[k] for x in L]
    out[name] = dict(ours=c(o), repair=c(r), union=u["union"], quad=q["coverage"], pinn=c(p), n_lowered=sum(x["c2"] < x["c2_old"] for x in o + r))
    print(f"{name:10s} {ms(c(o, 'coverage_old')):>14s} {ms(c(o)):>14s} {ms(c(r, 'coverage_old')):>14s} {ms(c(r)):>14s} {u['union']:6.1f} "
          f"{q['coverage_old']:5.1f}->{q['coverage']:4.1f} {ms(c(p, 'coverage_old')):>14s} {ms(c(p)):>14s}")
for name, rep in (("gen2_circle", "_rep2"), ("sin4", "_rep2")):
    if not (Z / f"runs/{name}_s0").exists():
        continue
    o = [js(Z / f"runs/{name}_s{k}/certify_iid.json") for k in range(4)]
    r = [js(Z / f"runs/{name}_s{k}{rep}/certify_iid.json") for k in range(4)]
    u = js(Z / f"results/iid_union_{name}.json")
    c = lambda L, k="coverage": [x[k] for x in L]
    out[name] = dict(ours=c(o), repair=c(r), union=u["union"], n_lowered=sum(x["c2"] < x["c2_old"] for x in o + r))
    print(f"{name:10s} {ms(c(o, 'coverage_old')):>14s} {ms(c(o)):>14s} {ms(c(r, 'coverage_old')):>14s} {ms(c(r)):>14s} {u['union']:6.1f}")
Path("results/iid_summary.json").write_text(json.dumps(out, indent=1))
