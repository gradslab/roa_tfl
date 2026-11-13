r"""
Rescoring of the equilibrium benchmarks with the verified core E = {q <= c0} (core_proof.py) as the inner layer, for the
candidate and for every baseline, without retraining (the scale-searched Zubov PINN, whose networks were not stored, is
retrained deterministically with its recorded scale and checked against its stored levels).  certify.py, iid.py and the
stored result files are not changed.

Inner layer.  prob.inner_c is set to c0, so prob.inner(x) = q(x) / c0 and certify.region gives E u {v < c2}.
Candidate (V_theta = alpha q + Phi, Phi >= 0):  c1 = (1 - e^{-alpha c0}) (1 - 1e-4), so {v < c1} lies in E by the floor
  V_theta >= alpha q (the factor absorbs float32 rounding).  Baselines (no floor): c1 is set on the samples of each set as
  the largest level with {W < c1} inside E, as certify.two_stage does (min of W over the samples outside E).
Level selection.  The two_stage rule of certify.py on the same sample sets and seeds (set A: 5 x 100k, seed 9000; set B:
  5 x 100k, seed 23000), the same counterexample definition (the step of length 1 leaves Omega_ad, or v does not decrease
  after it nor after any longer admissible step of length 2, 4, 8; box-outflow samples), but counterexamples are sought
  only among samples with v >= c1 outside E (states in E are safe by the verified core).  c2_old = min over A and B.
Final test.  iid.py's test with the same N = 10^6 and the same seed rule as certify_iid.py (seed_of(run name)), the band
  {c1 <= v < c2_old} restricted to outside E in the same way (the failure test is iid.failures, unchanged).
For the candidate, the selected snapshot of certify.json is kept, for the fine-tuned runs <sys>_s<k>_rep and for the
runs before fine-tuning <sys>_s<k>.

Run (from benchmarks/):
    python -u certify_core.py ours  vdp vdp3 pendulum con3d power    # runs/<run>/certify_core.json, results/core_union_<sys>.json
    python -u certify_core.py quad  vdp ...                          # results/baseline_quad_core_<sys>.json
    python -u certify_core.py pinn  vdp ...                          # default Zubov PINN, results/baseline_pinn_core_<sys>.json
    python -u certify_core.py pinn2 vdp ...                          # scale-searched Zubov PINN, results/baseline_pinn2_core_<sys>.json
    python -u certify_core.py dz    vdp ...                          # Dirichlet-Zubov PINN, results/baseline_dz_core_<sys>.json
    python -u certify_core.py --noclamp pinn|pinn2|dz vdp ...        # baselines scored with W = min(net, 1) (no lower clamp at 0),
                                                                     #   results/baseline_<kind>_core_noclamp_<sys>.json
    python -u certify_core.py --relaxed ours|quad|pinn|pinn2|dz vdp ... # relaxed counterexample rule (see RELAXED below),
                                                                     #   baselines unclamped; files with the suffix _relaxed
    python -u certify_core.py --relaxed summary vdp ...              # core_summary_relaxed_<sys>.json, block "relaxed"
    python -u certify_core.py summary vdp ...                        # results/core_summary_<sys>.json, results/core_iid_summary.json
                                                                     #   (pinn, pinn2, dz from the _noclamp files when present;
                                                                     #    the clamped scoring under <kind>_clamped)
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch

import baseline_dz as DZ
import baselines as B
import core
import systems as S
from certify import EXTRA, cert_samples, region
from gt import ground_truth
from iid import N_IID, failures, seed_of

ROOT = Path(__file__).parent
R = ROOT / "results"
C1_FACTOR = 1 - 1e-4


def core_level(name, prob):
    rec = json.loads((R / f"core_{name}.json").read_text())
    import numpy as np
    if not (np.array_equal(np.array(rec["constants"]["P_float32"]), prob.P.cpu().double().numpy())
            and np.array_equal(np.array(rec["constants"]["s_float32"]), prob.sys.s.cpu().double().numpy())):
        raise SystemExit(f"{name}: the quadratic differs from the one of core_{name}.json")
    return float(rec["c0"])


def set_core(prob, c0):
    prob.inner_c = float(c0)
    return prob


RELAXED = False
Q_MARGIN = 1e-12


def in_open_core(prob, y):
    z = y.double() * prob.sys.s.double()
    q = torch.einsum("ni,ij,nj->n", z, prob.P.double(), z)
    return q <= prob.inner_c * (1 - Q_MARGIN)


@torch.no_grad()
def failures_relaxed(v, step, znorm, x, W, horizons, z_tol, prob=None):
    """iid.failures with the relaxed rule: 1 = the step of length 1 leaves Omega_ad, 0 = for no H the trajectory ends in
    {q < c0} or below v(x), -1 = passes."""
    y, ok = step(x, 1.0)
    dec = ok & (torch.nan_to_num(v(y), nan=1.0) >= W) & (znorm(x) > z_tol) & ~in_open_core(prob, y)
    for T in horizons:
        r = torch.nonzero(dec).squeeze(-1)
        if len(r) == 0:
            break
        yT, okT = step(x[r], T)
        dec[r] = ~(okT & ((torch.nan_to_num(v(yT), nan=1.0) < W[r]) | in_open_core(prob, yT)))
    kind = torch.full_like(W, -1, dtype=torch.long)
    kind[dec] = 0
    kind[~ok] = 1
    return kind


def sfx():
    return "_relaxed" if RELAXED else ""


@torch.no_grad()
def two_stage_core(model, prob, n, rounds, seed, c1=None, z_tol=1e-3):
    """certify.two_stage with the inner layer E: c1 given (candidate) or the minimum of v over the samples outside E
    (baselines); violators are counted only outside E.  Returns c1, c2, the lowest violators, the sample count."""
    sys_ = prob.sys
    Ws, viol, retry, outq, kind, X = [], [], [], [], [], []
    N = 0
    for r in range(rounds):
        x, xo = cert_samples(sys_, n, seed + 17 * r)
        N += len(x) + len(xo)
        for xc in x.split(20000):
            W = model(xc)
            adm = sys_.admissible(xc)
            y, ok = core.flow(sys_, xc, 1.0)
            ok = ok & adm
            dec = (model(y) >= W) & ok & (prob.xt(xc).norm(dim=-1) > z_tol)
            if RELAXED:
                dec = dec & ~in_open_core(prob, y)
            Ws.append(W); viol.append(dec | ~ok); retry.append(dec); outq.append(prob.inner(xc) >= 1)
            kind.append(torch.where(ok, 0, 1)); X.append(xc)
        Wo = model(xo)
        Ws.append(Wo); viol.append(torch.ones_like(Wo, dtype=torch.bool)); retry.append(torch.zeros_like(Wo, dtype=torch.bool))
        outq.append(prob.inner(xo) >= 1); kind.append(torch.full_like(Wo, 2, dtype=torch.long)); X.append(xo)
    W, viol, retry, outq, kind, X = map(torch.cat, (Ws, viol, retry, outq, kind, X))
    W = torch.nan_to_num(W, nan=1.0)
    if c1 is None:
        wmin = W[outq].min().item() if outq.any() else 1.0
        c1 = wmin - (1 - C1_FACTOR) * abs(wmin)
    viol = viol & outq
    retry = retry & outq
    hard = viol & ~retry & (W >= c1)
    vb = W[hard].min().item() if hard.any() else math.inf
    cand = torch.nonzero(retry & (W >= c1) & (W < vb)).squeeze(-1)
    cand = cand[torch.argsort(W[cand])]
    for b in cand.split(2000):
        passed = torch.zeros(len(b), dtype=torch.bool, device=W.device)
        for T in EXTRA:
            rr = torch.nonzero(~passed).squeeze(-1)
            if len(rr) == 0:
                break
            yT, okT = core.flow(sys_, X[b[rr]], T)
            passed[rr] = okT & ((model(yT) < W[b[rr]]) | (in_open_core(prob, yT) if RELAXED else False))
        viol[b[passed]] = False
        if (~passed).any():
            break
    sv = viol & (W >= c1)
    c2 = min(W[sv].min().item() if sv.any() else 1.0, 1.0)
    order = torch.argsort(W[sv])[:50]
    n_low = int(((W < c1) & outq).sum())
    return c1, max(c2, c1), dict(v=W[sv][order], kind=kind[sv][order], x=X[sv][order], n_below_c1_outside_E=n_low), N


@torch.no_grad()
def iid_test_core(v, inner, draw, step, znorm, dim, c1, c2_old, seed, n=N_IID, horizons=EXTRA, chunk=200000,
                  batch=20000, z_tol=1e-3, prob=None):
    """iid.iid_test with the band {c1 <= v < c2_old} restricted to outside the inner layer (inner >= 1)."""
    g = torch.Generator().manual_seed(int(seed))
    band_x, band_w, low_x, low_w = [], [], [], []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = draw(torch.rand(m, dim, generator=g))
        W = torch.cat([torch.nan_to_num(v(c), nan=1.0) for c in x.split(batch)])
        I = torch.cat([inner(c) for c in x.split(batch)])
        b = (W >= c1) & (W < c2_old) & (I >= 1)
        band_x.append(x[b])
        band_w.append(W[b])
        low = (W < c1) & (I >= 1)
        low_x.append(x[low])
        low_w.append(W[low])
        done += m
    X, W = torch.cat(band_x), torch.cat(band_w)
    o = torch.argsort(W)
    X, W = X[o], W[o]
    c2_iid, n_tested, first = 1.0, 0, None
    for j in torch.arange(len(W), device=W.device).split(batch):
        kind = failures_relaxed(v, step, znorm, X[j], W[j], horizons, z_tol, prob) if RELAXED else \
            failures(v, step, znorm, X[j], W[j], horizons, z_tol)
        n_tested += len(j)
        bad = torch.nonzero(kind >= 0).squeeze(-1)
        if len(bad):
            i = bad[0]
            c2_iid = W[j][i].item()
            first = dict(v=c2_iid, kind=int(kind[i]), x=[float(t) for t in X[j][i].tolist()])
            break
    XL, WL = torch.cat(low_x), torch.cat(low_w)
    low_fail = 0
    for j in torch.arange(len(WL), device=WL.device).split(batch):
        low_fail += int(((failures_relaxed(v, step, znorm, XL[j], WL[j], horizons, z_tol, prob) if RELAXED else
                          failures(v, step, znorm, XL[j], WL[j], horizons, z_tol)) >= 0).sum())
    return dict(N=n, seed=int(seed), distribution="uniform on the box, i.i.d.", eps=math.log(1e6) / n, confidence=1 - 1e-6,
                c1=c1, c2_old=c2_old, c2_iid=c2_iid, c2=max(min(c2_old, c2_iid), c1), n_band=len(W), n_tested=n_tested,
                first_failure=first, n_below_c1_outside_inner=len(WL), n_below_c1_outside_inner_failed=low_fail,
                band="{c1 <= v < c2_old} outside E", rule="relaxed (end in {q < c0} or v decrease)" if RELAXED else "strict (v decrease)")


def final_test(model, prob, name, c1, c2_old):
    s = prob.sys

    def step(x, T):
        y, ok = core.flow(s, x, T)
        return y, ok & s.admissible(x)

    r = iid_test_core(model, prob.inner, lambda u: s._map(u.to(S.DEV, S.DT)), step, lambda x: prob.xt(x).norm(dim=-1), s.n,
                      c1, c2_old, seed_of(name), prob=prob)
    xg, roa = ground_truth(s)
    nr = roa.sum().item()
    old, new = region(model, prob, c2_old, xg), region(model, prob, r["c2"], xg)
    inE = torch.cat([prob.inner(c) < 1 for c in xg.split(50000)])
    r.update(coverage_old=100 * (old & roa).sum().item() / nr, coverage=100 * (new & roa).sum().item() / nr,
             false_positive=100 * (new & ~roa).sum().item() / len(roa), core_coverage=100 * (inE & roa).sum().item() / nr)
    if r["first_failure"]:
        x0 = torch.tensor([r["first_failure"]["x"]], device=S.DEV, dtype=S.DT)
        r["first_failure"]["in_roa"] = bool(s.simulate(x0)[0] == 0)
    return r, new


def rescore(model, prob, name, c1_fixed=None):
    """Sets A and B (certify.py / baselines.certify_fn), then the final test."""
    t0 = time.time()
    c1a, c2a, _, _ = two_stage_core(model, prob, 100000, 5, 9000, c1_fixed)
    c1b, c2b, lim, NB = two_stage_core(model, prob, 100000, 5, 23000, c1_fixed)
    c1, c2_old = min(c1a, c1b), min(c2a, c2b)
    r, reg = final_test(model, prob, name, c1, c2_old)
    names = ["step in Omega_ad, no decrease", "step leaves Omega_ad", "box outflow"]
    r.update(c1_A=c1a, c2_A=c2a, c1_B=c1b, c2_B=c2b, n_set_B=NB, c0=prob.inner_c,
             limiting={names[k]: int((lim["kind"] == k).sum()) for k in range(3)},
             n_below_c1_outside_E_setB=lim["n_below_c1_outside_E"], seconds=time.time() - t0)
    print(f"{name:24s} c1 {c1:.5f}  c2 A {c2a:.5f} B {c2b:.5f} -> final {r['c2']:.5f}  coverage {r['coverage']:6.2f} %  "
          f"(E alone {r['core_coverage']:.2f} %)  FP {r['false_positive']:.3f} %  band {r['n_band']}  [{r['seconds']:.0f} s]", flush=True)
    return r, reg


def do_ours(name):
    regs = []
    for suf in ("", "_rep"):
        for k in range(4):
            run = ROOT / "runs" / f"{name}_s{k}{suf}"
            old = json.loads((run / "certify.json").read_text())
            prob, model, cfg = core.load(run, run / "snapshots" / old["snapshot"])
            c0 = core_level(name, prob)
            set_core(prob, c0)
            c1 = (1 - math.exp(-float(model.alpha) * c0)) * C1_FACTOR
            r, reg = rescore(model, prob, run.name, c1_fixed=c1)
            oi = json.loads((run / "certify_iid.json").read_text())
            r.update(snapshot=old["snapshot"], alpha=float(model.alpha), c1_rule="(1 - exp(-alpha c0)) (1 - 1e-4)",
                     sampled_inner_c=float(model.inner_c), old=dict(c1=oi["c1"], c2=oi["c2"], coverage=oi["coverage"],
                                                                     false_positive=oi["false_positive"]))
            (run / f"certify_core{sfx()}.json").write_text(json.dumps(r, indent=1))
            if suf:
                regs.append(reg)
    xg, roa = ground_truth(prob.sys)
    u = torch.stack(regs).any(0)
    un = dict(union=100 * (u & roa).sum().item() / roa.sum().item(), false_positive=100 * (u & ~roa).sum().item() / len(roa),
              old=json.loads((R / f"iid_union_{name}.json").read_text()))
    (R / f"core_union{sfx()}_{name}.json").write_text(json.dumps(un, indent=1))
    print(f"{name}: union {un['union']:.2f} % (FP {un['false_positive']:.3f} %); old {un['old']['union']:.2f} %", flush=True)


NOCLAMP = False


class UpClip(torch.nn.Module):
    """W = min(net, 1): the scoring of the Zubov PINN without the lower clamp of baselines.Clip.  Sublevel sets
    {W < c}, c > 0, are those of Clip; a region where the network is <= 0 is no longer a plateau W = 0."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).clamp(max=1.0)


class DzUpClip(torch.nn.Module):
    """baseline_dz.DzClip without the lower clamp: min(net, 1) on Omega_ad, 1 off it."""
    def __init__(self, m, s):
        super().__init__()
        self.m, self.sys = m, s

    def forward(self, x):
        return torch.where(self.sys.admissible(x), self.m(torch.nan_to_num(x)).clamp(max=1.0),
                           torch.ones(len(x), device=x.device, dtype=x.dtype))


def wrap_pinn(m):
    return UpClip(m) if NOCLAMP else B.Clip(m)


def wrap_dz(m, s):
    return DzUpClip(m, s) if NOCLAMP else DZ.DzClip(m, s)


def base_prob(name):
    sys_ = S.get(name)
    prob = core.Problem(sys_, core.Config(system=name))
    prob.build_inner(verbose=False)
    return sys_, prob


def pinn_net(sys_, path):
    m = B.PINN(sys_.n).to(S.DEV)
    m.s = sys_.s
    m.load_state_dict(torch.load(path, map_location=S.DEV))
    return m


def do_baseline(kind, name):
    sys_, prob = base_prob(name)
    c_samp = prob.inner_c
    c0 = core_level(name, prob)
    res = []
    if kind == "quad":
        old = json.loads((R / f"iid_quad_{name}.json").read_text())[0]
        set_core(prob, c0)
        r, _ = rescore(B.Quad(prob), prob, f"quad_{name}")
        r.update(old=dict(c1=old["c1"], c2=old["c2"], coverage=old["coverage"], false_positive=old["false_positive"]))
        res.append(r)
    elif kind == "pinn":
        olds = {o_["seed"]: o_ for o_ in json.loads((R / f"iid_pinn_{name}.json").read_text())}
        for b in json.loads((R / f"baseline_pinn_{name}.json").read_text()):
            m = pinn_net(sys_, R / f"pinn_{name}_s{b['seed']}.pt")
            set_core(prob, c0)
            r, _ = rescore(wrap_pinn(m), prob, f"pinn_{name}_s{b['seed']}")
            o_ = olds[b["seed"]]
            r.update(seed=b["seed"], factor=1.0, old=dict(c1=o_["c1"], c2=o_["c2"], coverage=o_["coverage"], false_positive=o_["false_positive"]))
            res.append(r)
    elif kind in ("pinn2", "dz"):
        for b in json.loads((R / f"baseline_{kind}_{name}.json").read_text())["runs"]:
            seed, factor = b["seed"], b["factor"]
            extra = {}
            if kind == "dz":
                m = pinn_net(sys_, R / f"dz_{name}_s{seed}_f{factor:g}.pt")
            else:
                ck = R / f"pinn2_{name}_s{seed}_f{factor:g}.pt"
                if ck.exists():
                    m = pinn_net(sys_, ck)
                else:
                    t0 = time.time()
                    m = B.train_pinn(sys_, prob, seed, factor=factor)
                    torch.save(m.state_dict(), ck)
                    extra["retrain_seconds"] = time.time() - t0
            if kind == "pinn2" and (not NOCLAMP or "retrain_seconds" in extra):
                set_core(prob, c_samp)
                c1a, c2a, _, _ = _two_stage_orig(DZ.DzClip(m, sys_), prob, 100000, 5, 9000)
                c1b, c2b, _, _ = _two_stage_orig(DZ.DzClip(m, sys_), prob, 100000, 5, 23000)
                extra["reproduction"] = dict(c1=min(c1a, c1b), c2_old=min(c2a, c2b), stored_c1=b["c1"], stored_c2_old=b["c2_old"],
                                             match=abs(min(c1a, c1b) - b["c1"]) < 1e-5 and abs(min(c2a, c2b) - b["c2_old"]) < 1e-5)
                print(f"  pinn2 {name} s{seed}: reproduction {extra['reproduction']}", flush=True)
            set_core(prob, c0)
            r, _ = rescore(wrap_dz(m, sys_), prob, f"{kind}_{name}_s{seed}")
            r.update(seed=seed, factor=factor, old=dict(c1=b["c1"], c2=b["c2"], coverage=b["coverage"],
                                                        false_positive=b["false_positive"]), **extra)
            res.append(r)
    for r in res:
        r["scoring"] = ("quadratic form" if kind == "quad" else
                        "W = min(net, 1), no lower clamp" if NOCLAMP else "clamped to [0, 1] (baselines.Clip / baseline_dz.DzClip)")
    tag = "_relaxed" if RELAXED else ("_noclamp" if NOCLAMP and kind != "quad" else "")
    (R / f"baseline_{kind}_core{tag}_{name}.json").write_text(json.dumps(res, indent=1))


def _two_stage_orig(model, prob, n, rounds, seed):
    from certify import two_stage
    c1, c2, lim, N = two_stage(model, prob, n, rounds, seed)
    return c1, c2, lim, N


def do_summary(names):
    js = lambda p: json.loads(Path(p).read_text())
    ms = lambda v: dict(mean=st.mean(v), std=st.stdev(v)) if len(v) > 1 else dict(mean=v[0], std=0.0)
    allsum = {}
    for name in names:
        o = [js(ROOT / f"runs/{name}_s{k}/certify_core.json") for k in range(4)]
        r = [js(ROOT / f"runs/{name}_s{k}_rep/certify_core.json") for k in range(4)]
        u = js(R / f"core_union_{name}.json")
        cov = lambda L, k="coverage": [x[k] for x in L]
        d = dict(ours=cov(o), repair=cov(r), union=u["union"], n_lowered=sum(x["c2"] < x["c2_old"] for x in o + r))
        for kind in ("quad", "pinn", "pinn2", "dz"):
            p = R / f"baseline_{kind}_core_{name}.json"
            pn = R / f"baseline_{kind}_core_noclamp_{name}.json"
            if p.exists():
                b = js(p)
                d[kind + "_old"] = b[0]["old"]["coverage"] if kind == "quad" else [x["old"]["coverage"] for x in b]
                if kind != "quad":
                    d[kind + "_clamped"] = cov(b)
                    d[kind + "_clamped_false_positive"] = [x["false_positive"] for x in b]
                    d[kind + "_clamped_levels"] = [dict(seed=x.get("seed"), c1=x["c1"], c2=x["c2"]) for x in b]
                if kind != "quad" and pn.exists():
                    b = js(pn)
                d[kind] = b[0]["coverage"] if kind == "quad" else cov(b)
                d[kind + "_false_positive"] = [x["false_positive"] for x in b]
                d[kind + "_levels"] = [dict(seed=x.get("seed"), c1=x["c1"], c2=x["c2"]) for x in b]
                d[kind + "_scoring"] = b[0].get("scoring", "clamped to [0, 1] (baselines.Clip / baseline_dz.DzClip)")
        d.update(ours_stats=ms(d["ours"]), repair_stats=ms(d["repair"]),
                 false_positive=max([x["false_positive"] for x in r] + [u["false_positive"]]),
                 false_positive_union=u["false_positive"], c0=r[0]["c0"],
                 levels=dict(before=[dict(c1=x["c1"], c2=x["c2"]) for x in o], after=[dict(c1=x["c1"], c2=x["c2"]) for x in r]),
                 core_coverage=r[0]["core_coverage"],
                 old=dict(ours=[x["old"]["coverage"] for x in o], repair=[x["old"]["coverage"] for x in r],
                          union=u["old"]["union"], false_positive=max([x["old"]["false_positive"] for x in r] + [u["old"]["false_positive"]])))
        (R / f"core_summary_{name}.json").write_text(json.dumps(d, indent=1))
        allsum[name] = {k: d[k] for k in ("ours", "repair", "union", "quad", "pinn", "n_lowered") if k in d}
        allsum[name].update({k: d[k] for k in ("pinn2", "dz", "false_positive", "pinn_clamped", "pinn2_clamped", "dz_clamped",
                                               "pinn_scoring", "pinn2_scoring", "dz_scoring") if k in d})
        print(f"{name}: before {d['ours_stats']['mean']:.1f} after {d['repair_stats']['mean']:.1f} union {d['union']:.1f} "
              f"(old {st.mean(d['old']['ours']):.1f} / {st.mean(d['old']['repair']):.1f} / {d['old']['union']:.1f})", flush=True)
    p = R / "core_iid_summary.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(allsum)
    p.write_text(json.dumps(prev, indent=1))


def do_summary_relaxed(names):
    """results/core_summary_relaxed_<sys>.json and the block "relaxed" of results/core_iid_summary.json (same keys as the
    strict entries; baselines unclamped); the strict numbers are kept next to them for comparison."""
    js = lambda p: json.loads(Path(p).read_text())
    cov = lambda L: [x["coverage"] for x in L]
    p = R / "core_iid_summary.json"
    allsum = json.loads(p.read_text()) if p.exists() else {}
    block = allsum.get("relaxed", {})
    for name in names:
        o = [js(ROOT / f"runs/{name}_s{k}/certify_core_relaxed.json") for k in range(4)]
        r = [js(ROOT / f"runs/{name}_s{k}_rep/certify_core_relaxed.json") for k in range(4)]
        u = js(R / f"core_union_relaxed_{name}.json")
        d = dict(ours=cov(o), repair=cov(r), union=u["union"], n_lowered=sum(x["c2"] < x["c2_old"] for x in o + r),
                 false_positive=max([x["false_positive"] for x in r] + [u["false_positive"]]))
        for kind in ("quad", "pinn", "pinn2", "dz"):
            f = R / f"baseline_{kind}_core_relaxed_{name}.json"
            if f.exists():
                b = js(f)
                d[kind] = b[0]["coverage"] if kind == "quad" else cov(b)
                d[kind + "_false_positive"] = [x["false_positive"] for x in b]
                d[kind + "_levels"] = [dict(seed=x.get("seed"), c1=x["c1"], c2=x["c2"]) for x in b]
        d["levels"] = dict(before=[dict(c1=x["c1"], c2=x["c2"]) for x in o], after=[dict(c1=x["c1"], c2=x["c2"]) for x in r])
        d["rule"] = "relaxed: pass if for some H in {1,2,4,8} the step stays in Omega_ad and ends in {q < c0} or below v(x)"
        (R / f"core_summary_relaxed_{name}.json").write_text(json.dumps(d, indent=1))
        block[name] = {k: d[k] for k in ("ours", "repair", "union", "quad", "pinn", "pinn2", "dz", "false_positive", "n_lowered") if k in d}
        print(f"{name} relaxed: before {st.mean(d['ours']):.2f} after {st.mean(d['repair']):.2f} union {d['union']:.2f}", flush=True)
    allsum["relaxed"] = block
    p.write_text(json.dumps(allsum, indent=1))


def main():
    torch.set_num_threads(2)
    global NOCLAMP
    global RELAXED
    args = [a for a in sys.argv[1:] if a not in ("--noclamp", "--relaxed")]
    NOCLAMP = "--noclamp" in sys.argv[1:]
    RELAXED = "--relaxed" in sys.argv[1:]
    if RELAXED and args[0] in ("pinn", "pinn2", "dz"):
        NOCLAMP = True
    if args[0] == "summary" and RELAXED:
        return do_summary_relaxed(args[1:])
    kind, names = args[0], args[1:]
    if kind == "summary":
        return do_summary(names)
    for name in names:
        t0 = time.time()
        do_ours(name) if kind == "ours" else do_baseline(kind, name)
        print(f"  {kind} {name} done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
