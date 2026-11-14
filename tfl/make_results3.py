r"""Paper ledger: results3.tex (macros) and the tables of main.tex from the run files.  Every number in the paper
comes from here.  Run:  python make_results3.py  (writes to ../paper/)"""
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "paper"
CASES = [("circ3", "circle", "Circle"), ("sin6", "sinusoid", "Sinusoid"), ("quad3", "quadrotor", "Quadrotor")]
MAC = {"circ3": "Circ", "sin6": "Sin", "quad3": "Quad"}


def load(p):
    return json.loads(Path(p).read_text())


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def pct(x, d=1):
    return f"{x:.{d}f}\\,\\%"


def main():
    macros, main_rows, seed_rows, inner_rows, const_rows, abl_rows = [], [], [], [], [], []
    collapse_extra = []
    gains, kept, t_train, t_cert, t_rep, t_inner, iid_fail = [], 0, [], [], [], [], 0
    for name, case, label in CASES:
        S = load(f"results/summary_{name}.json")
        I = load(f"results/inner_{case}.json")
        C = load(f"results/ceiling_{name}.json") if Path(f"results/ceiling_{name}.json").exists() else None
        A = load(f"results/ablation_{name}.json") if Path(f"results/ablation_{name}.json").exists() else None
        W = load(f"results/summary_{name}w.json") if Path(f"results/summary_{name}w.json").exists() else None
        Qd = load(f"runs/{name}_quadratic/certify_v3.json")["coverage"]
        Pc = [load(f"runs/{name}_pinn_s{k}/certify_v3.json")["coverage"] for k in range(4)]
        Dz = [load(f"runs/{name}_dz_s{k}/certify_v3.json")["coverage"] for k in range(4)]
        rep = [r for r in S["rows"] if r["run"].endswith("_rep3")]
        raw = [r for r in S["rows"] if not r["run"].endswith("_rep3")]
        cov_rep = [r["coverage"] for r in rep]
        cov_raw = [r["coverage"] for r in raw]
        m = MAC[name]
        macros += [f"\\newcommand{{\\resAbs{m}}}{{{pct(S['union_repaired'])}}}",
                   f"\\newcommand{{\\resAbsInner{m}}}{{{pct(S['inner_coverage'])}}}",
                   f"\\newcommand{{\\resRoa{m}}}{{{pct(S['roa_fraction_of_box'])}}}",
                   f"\\newcommand{{\\resCov{m}Mean}}{{{pct(statistics.mean(cov_rep))}}}"]
        if C:
            macros += [f"\\newcommand{{\\resCeil{m}}}{{{pct(C['mean_ceiling'])}}}",
                       f"\\newcommand{{\\resCeilFrac{m}}}{{{pct(100 * C['mean_learned'] / C['mean_ceiling'], 0)}}}"]
        macros += [f"\\newcommand{{\\resQuad{m}}}{{{pct(Qd)}}}", f"\\newcommand{{\\resPinnTFL{m}}}{{{pct(statistics.mean(Pc))}}}", f"\\newcommand{{\\resDzTFL{m}}}{{{pct(statistics.mean(Dz))}}}"]
        if A and "safety_value" in A:
            macros += [f"\\newcommand{{\\resSafety{m}}}{{{pct(A['safety_value']['coverage'])}}}"]
        if W:
            wrep = [r["coverage"] for r in W["rows"] if r["run"].endswith("_rep3")]
            macros += [f"\\newcommand{{\\resGap{m}}}{{{round(statistics.mean(cov_rep), 1) - round(statistics.mean(wrep), 1):.1f}}}", f"\\newcommand{{\\resWholeLevelMax{m}}}{{{max(r['c2'] for r in W['rows'] if r['run'].endswith('_rep3')):.4f}}}"]
            macros += [f"\\newcommand{{\\resPinnBothMax{m}}}{{{pct(max(statistics.mean(Pc), statistics.mean(Dz)))}}}",
                       f"\\newcommand{{\\resZubovMax{m}}}{{{pct(max(Qd, statistics.mean(Pc), statistics.mean(Dz)))}}}",
                       f"\\newcommand{{\\resOtherMax{m}}}{{{pct(max(Qd, statistics.mean(Pc), statistics.mean(Dz), statistics.mean(wrep)))}}}"]
            macros += [f"\\newcommand{{\\resWhole{m}}}{{{pct(statistics.mean(wrep))}}}",
                       f"\\newcommand{{\\resWholeUnion{m}}}{{{pct(W['union_repaired'])}}}"]
        if A and "pinn_style" in A:
            macros += [f"\\newcommand{{\\resPinn{m}}}{{{pct(A['pinn_style']['coverage'])}}}"]
        gains += [a - b for a, b in zip(cov_rep, cov_raw)]
        macros += [f"\\newcommand{{\\resRaw{m}Mean}}{{{pct(statistics.mean(cov_raw))}}}",
                   f"\\newcommand{{\\resRepairGain{m}}}{{{statistics.mean(cov_rep) - statistics.mean(cov_raw):+.1f}}}"]
        c1 = 1 - math.exp(-I["c_inner"])
        c2s = [r["c2"] for r in rep]
        fps = max(max(r["false_positive"] for r in S["rows"]), S["union_repaired_fp"], S["union_unrepaired_fp"])
        ceil = f"{C['mean_ceiling']:.1f}" if C else "--"
        whole = "--"
        if W:
            wrep = [r["coverage"] for r in W["rows"] if r["run"].endswith("_rep3")]
            whole = f"${statistics.mean(wrep):.1f}\\pm{statistics.stdev(wrep):.1f}$"
        main_rows.append(f"{label} & {S['roa_fraction_of_box']:.1f} & {S['inner_coverage']:.1f} & {Qd:.1f} & "
                         f"${statistics.mean(Pc):.1f}\\pm{statistics.stdev(Pc):.1f}$ & ${statistics.mean(Dz):.1f}\\pm{statistics.stdev(Dz):.1f}$ & {whole} & "
                         f"${statistics.mean(cov_rep):.1f}\\pm{statistics.stdev(cov_rep):.1f}$ & {S['union_repaired']:.1f} & {ceil} & {fps:.2f}\\\\")
        rich = [load(f"runs/{name}_rich{k_}_f{f_}_s0/certify_v3.json")["coverage"] for k_ in ("pinn", "dz") for f_ in (1, 10)]
        macros.append(f"\\newcommand{{\\resRichMax{m}}}{{{pct(max(rich))}}}")
        kept_case, draw_diff = 0, []
        for kind_, names_ in (("pinn", ("circ3", "quad3")), ("dz", ("sin6", "quad3"))):
            if name in names_:
                rs_ = [load(f"runs/{name}_{kind_}_s{k}/certify_v3.json") for k in range(4)]
                extra_ = [round((r_["coverage"] - r_["inner_coverage"]) / 100 * r_["roa_fraction_of_box"] / 100 * r_["n_ground_truth"]) for r_ in rs_]
                assert all(r_["c2"] - r_["c1"] <= 2e-4 * abs(r_["c1"]) + 1e-12 for r_ in rs_) and max(extra_) <= 1
                collapse_extra.append(max(extra_))
        for k in range(4):
            r3 = load(f"runs/{name}_s{k}_rep3/certify_v3.json")
            r0 = load(f"runs/{name}_s{k}/certify_v3.json")
            rp = load(f"runs/{name}_s{k}_rep3/repair.json")
            hist = load(f"runs/{name}_s{k}/history.json")
            if k > 0:
                t_train.append(hist[-1]["time"] / 60)
            t_cert.append(r3["seconds"] / 60)
            t_rep.append(rp["seconds"] / 60)
            kept += r3["snapshot"] == "epoch_000.pt"
            kept_case += r3["snapshot"] == "epoch_000.pt"
            if r3["snapshot"] == "epoch_000.pt":
                draw_diff.append(abs(r3["coverage"] - r0["coverage"]))
            iid_fail += r3["iid"]["first_failure"] is not None
            seed_rows.append(f"{label} & {k} & {r0['c2']:.4f} & {r0['coverage']:.2f} & {int(r3['snapshot'].replace('epoch_', '').replace('.pt', ''))} & "
                             f"{r3['c2_selected']:.4f} & {r3['c2']:.4f} & {r3['iid']['n_band']} & {r3['coverage']:.2f} & {r3['false_positive']:.3f}\\\\")
        t_inner.append(I["seconds"])
        macros.append(f"\\newcommand{{\\resKept{m}}}{{{['zero','one','two','three','four'][kept_case]}}}")
        if draw_diff:
            macros.append(f"\\newcommand{{\\resDrawMax{m}}}{{{max(draw_diff):.1f}}}")
        if C:
            macros.append(f"\\newcommand{{\\resCeilNine{m}}}{{{pct(C['ceiling_at']['0.9'])}}}")
        macros += [f"\\newcommand{{\\resCtwoMin{m}}}{{{min(c2s):.2f}}}", f"\\newcommand{{\\resCtwoMax{m}}}{{{max(c2s):.2f}}}"]
        info = I.get("verify_info", {})
        const_rows.append(f"{label} & {I['problem']['alpha']:.4f} & " + ", ".join(f"{v:.3f}" for v in I['problem']['z_scale']) + "\\\\")
        inner_rows.append(f"{label} & {I['c_inner']:.4f} & {c1:.4f} & {100 * I['volume_of_box']:.2f} & {I['cells_processed']} & {I['seconds']:.0f}\\\\")
        if A:
            r00 = next(r for r in S["rows"] if r["run"] == f"{name}_s0")
            macros.append(f"\\newcommand{{\\resSeedZero{m}}}{{{pct(r00['coverage'])}}}")
            abl_rows.append(f"{label} & full method & {r00['c2']:.3f} & {r00['coverage']:.1f} & {r00['false_positive']:.2f}\\\\")
            for key, lab in (("pinn_style", "PDE residual loss alone"), 
                             ("safety_value", "safety-value baseline"), ("no_step", "no finite-step penalty, no counterexamples"),
                             ("no_wcap", "no exponential loss weight"), ("sum_ansatz", "sum form $\\Phi=\\psi+\\Psi_\\theta$")):
                if key in A:
                    abl_rows.append(f"{label} & {lab} & {A[key]['c2']:.3f} & {A[key]['coverage']:.1f} & {A[key]['false_positive']:.2f}\\\\")
    for n_, m_ in MAC.items():
        f_ = Path(f"results/unsafe_{n_}.json")
        if f_.exists():
            u_ = load(f_)
            macros += [f"\\newcommand{{\\resUnsafeK{m_}}}{{{u_['unsafe']}}}",
                       f"\\newcommand{{\\resUnsafeIn{m_}}}{{{u_['samples_in_S']:,}}}".replace(",", "\\,"),
                       f"\\newcommand{{\\resUnsafeEps{m_}}}{{{sci(u_['eps_k'])}}}",
                       f"\\newcommand{{\\resUnsafeTime{m_}}}{{{u_['longest_time_to_inner']:.0f}}}"]
    if Path("results/unsafe_check.json").exists():
        uc = load("results/unsafe_check.json")
        us = [load(f"results/unsafe_{n_}.json") for n_ in MAC]
        macros += [f"\\newcommand{{\\resUnsafeDis}}{{{sum(v['disagreements'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeAgree}}{{{sum(v['agree_after_finer_reference'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeCmp}}{{{sum(v['n_compared'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeCons}}{{{sum(v['remaining_conservative'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeDisInS}}{{{'none' if sum(v['disagreements_in_S'] for v in uc.values()) == 0 else sum(v['disagreements_in_S'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeTotal}}{{{sum(v['unsafe'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeConfirmed}}{{{sum(v['unsafe_confirmed'] for v in uc.values())}}}",
                   f"\\newcommand{{\\resUnsafeEpsMax}}{{{sci(max(u['eps_k'] for u in us))}}}",
                   f"\\newcommand{{\\resUnsafeTimeMax}}{{{max(u['longest_time_to_inner'] for u in us):.0f}}}"]
    noise = load("results/ref_sensitivity.json")
    contact, floor_ = load("results/inner_contact.json"), load("results/floor.json")
    for key_, m_ in (("circle", "Circ"), ("sinusoid", "Sin"), ("quadrotor", "Quad")):
        macros += [f"\\newcommand{{\\resContact{m_}}}{{{contact[key_]:.4f}}}",
                   f"\\newcommand{{\\resCI{m_}}}{{{load(f'results/inner_{key_}.json')['c_inner']:.4f}}}",
                   f"\\newcommand{{\\resFloor{m_}}}{{{pct(floor_[key_]['mean'])}}}"]
    macros.append(f"\\newcommand{{\\resRefNoiseMax}}{{{pct(max(v['changed_pct'] for v in noise.values()), 2)}}}")
    for key_, m_ in (("circle", "Circ"), ("sinusoid", "Sin"), ("quadrotor", "Quad")):
        macros.append(f"\\newcommand{{\\resRefNoise{m_}}}{{{pct(noise[key_]['changed_pct'], 2)}}}")
    probes = {n: (load(f"runs/cfg{n}/config.json"), load(f"runs/cfg{n}/certify_probe.json"))
              for n in ("O_circle", "M10W1_circle", "O_sinusoid", "M10W1_sinusoid", "O_quadrotor", "M0_quadrotor", "M10_quadrotor",
                        "W1_quadrotor", "M10W1_quadrotor", "M30W1_quadrotor", "M3W1_quadrotor", "M10W0_quadrotor", "M10W1H128_quadrotor")}
    assert all(c["seed"] == 0 for c, _ in probes.values())
    for n_ in ("circ3", "sin6", "quad3"):
        cs = [load(f"runs/{n_}_s{k}/config.json") for k in range(4)]
        tag = {"circ3": "O_circle", "sin6": "O_sinusoid", "quad3": "M10W0_quadrotor"}[n_]
        assert all((c["mu_step"], c["w_cap"], c["hidden"]) == (probes[tag][0]["mu_step"], probes[tag][0]["w_cap"], probes[tag][0]["hidden"]) for c in cs)
    pc = {n: r["coverage"] for n, (_, r) in probes.items()}
    assert pc["O_circle"] > pc["M10W1_circle"] and pc["O_sinusoid"] > pc["M10W1_sinusoid"]
    quad_pairs = [n for n in pc if n.endswith("_quadrotor") and "H128" not in n]
    assert max(quad_pairs, key=pc.get) == "M10W0_quadrotor"
    assert {probes[n][1]["iid"]["N"] for n in pc} == {200000}
    macros += [f"\\newcommand{{\\resProbeKeepCirc}}{{{pct(pc['O_circle'])}}}",
               f"\\newcommand{{\\resProbeAltCirc}}{{{pct(pc['M10W1_circle'])}}}",
               f"\\newcommand{{\\resProbeKeepSin}}{{{pct(pc['O_sinusoid'])}}}",
               f"\\newcommand{{\\resProbeAltSin}}{{{pct(pc['M10W1_sinusoid'])}}}",
               f"\\newcommand{{\\resProbeOrigQuad}}{{{pct(pc['O_quadrotor'])}}}",
               f"\\newcommand{{\\resProbeBestQuad}}{{{pct(pc['M10W0_quadrotor'])}}}",
               f"\\newcommand{{\\resProbeWideQuad}}{{{pct(pc['M10W1H128_quadrotor'])}}}",
               f"\\newcommand{{\\resProbeQuadN}}{{{['zero','one','two','three','four','five','six','seven','eight','nine','ten'][len(quad_pairs)]}}}"]
    macros.append(f"\\newcommand{{\\resCollapseExtra}}{{{['no', 'one'][max(collapse_extra)]}}}")
    orc = load("results/oracle_tfl.json")
    for n_, m_ in MAC.items():
        ob_ = max(v["coverage"] for k, v in orc.items() if k.startswith(n_ + "_"))
        assert ob_ < statistics.mean(load(f"runs/{n_}_s{k}_rep3/certify_v3.json")["coverage"] for k in range(4))
        macros.append(f"\\newcommand{{\\resOracleMax{m_}}}{{{pct(ob_)}}}")
    fp_all = {}
    for n, _, _ in CASES:
        S_ = load(f"results/summary_{n}.json")
        fp_all[n] = max(max(r["false_positive"] for r in S_["rows"]), S_["union_repaired_fp"], S_["union_unrepaired_fp"])
    macros.append(f"\\newcommand{{\\resFPMax}}{{{pct(max(fp_all.values()), 3)}}}")
    def fp_sentence(fp):
        n = round(fp * 5e4 / 100)
        return "No reference state lies" if n == 0 else f"At most {n} of the $5\\times10^4$ reference states lie"
    macros.append(f"\\newcommand{{\\resFPSentence}}{{{fp_sentence(max(fp_all.values()))}}}")
    macros += [
               f"\\newcommand{{\\resIidFailures}}{{{'no' if iid_fail == 0 else ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][iid_fail]}}}",
               f"\\newcommand{{\\resRepairGain}}{{{statistics.mean(gains):.1f}}}",
               f"\\newcommand{{\\resRepairKept}}{{{kept}}}",
               f"\\newcommand{{\\resTimeTrain}}{{{min(t_train):.0f} to {max(t_train):.0f} minutes}}",
               f"\\newcommand{{\\resTimeCert}}{{{min(t_cert):.0f} to {max(t_cert):.0f} minutes}}",
               f"\\newcommand{{\\resTimeRepair}}{{{min(t_rep):.0f} to {max(t_rep):.0f} minutes}}",
               f"\\newcommand{{\\resTimeInner}}{{{min(t_inner):.0f} to {max(t_inner):.0f} seconds per case}}"]
    G = Path("../benchmarks")
    bench = [("vdp", "vdp1", "Van der Pol, $\\mu=1$", True), ("vdp3", "vdp3", "Van der Pol, $\\mu=3$", True), ("pendulum", "pendulum", "Pendulum, $|u|\\le0.8$", False),
             ("con3d", "con3d", "3-D, constraint and obstacle", True), ("power", "power", "Power system", False)]
    Sm = load(G / "results/core_iid_summary.json")["relaxed"]
    b_rows, b_means, b_unions, b_fps, sos_gap = [], [], [], [], []
    import torch
    for key, gt, lab, poly in bench:
        d = Sm[key]
        roa = 100 * torch.load(G / f"results/gt_{gt}.pt", map_location="cpu")["roa"].float().mean().item()
        so = load(G / (f"results/sos_recert_{key}.json" if poly else f"results/baseline_sos_sound_{key}.json"))
        if poly:
            rc_ = so["recertification"]
            assert rc_["status"] == "optimal" and rc_["all_certified"] and rc_["level"] is not None
        else:
            rc_ = load(G / "results/sos_settings.json")[key]["recertification_clarabel"]
            assert rc_["status"] == "optimal" and rc_["all_gram_certified"] and rc_["level"] is not None and so["false_positive"] == 0.0
        assert load(G / "results/sos_settings.json")[key]["sample_check"]["decrease_violations"] == 0
        sos = f"{so['coverage']:.1f}"
        fp = max([load(G / f"runs/{key}_s{k}_rep/certify_core_relaxed.json")["false_positive"] for k in range(4)] + [load(G / f"results/core_union_relaxed_{key}.json")["false_positive"]])
        m_, s_ = statistics.mean(d["repair"]), statistics.stdev(d["repair"])
        p2 = d["pinn2"]
        assert statistics.mean(p2) > statistics.mean(d["pinn"])
        pick = p2
        pm, ps = statistics.mean(pick), statistics.stdev(pick)
        dz = d["dz"]
        dm, ds = statistics.mean(dz), statistics.stdev(dz)
        tag = {"vdp": "VdpOne", "vdp3": "VdpThree", "pendulum": "Pend", "con3d": "ConThree", "power": "Power"}[key]
        macros += [f"\\newcommand{{\\resPinn{tag}}}{{{pct(pm)}}}", f"\\newcommand{{\\resDz{tag}}}{{{pct(dm)}}}", f"\\newcommand{{\\resSeed{tag}}}{{{pct(m_)}}}",
                   f"\\newcommand{{\\resPinnBest{tag}}}{{{pct(max(d['pinn'] + p2))}}}", f"\\newcommand{{\\resPinnWorst{tag}}}{{{pct(min(p2))}}}"]
        b_means.append(m_); b_unions.append(d["union"]); b_fps.append(fp)
        if key in ("vdp", "vdp3"):
            sos_gap.append(abs(so["coverage"] - d["union"]))
        macros += [f"\\newcommand{{\\resSos{tag}}}{{{pct(so['coverage'])}}}"]
        b_rows.append(f"{lab} & {roa:.1f} & {d['quad']:.1f} & {sos} & ${pm:.1f}\\pm{ps:.1f}$ & ${dm:.1f}\\pm{ds:.1f}$ & ${statistics.mean(d['ours']):.1f}\\pm{statistics.stdev(d['ours']):.1f}$ & ${m_:.1f}\\pm{s_:.1f}$ & {d['union']:.1f} & {fp:.2f}\\\\")
    obs = [("vdp_obs1", "Van der Pol, 1 obstacle", "ObsVdpOne"), ("vdp_obs2", "Van der Pol, 2 obstacles", "ObsVdpTwo"),
           ("power_obs2", "Power, 2 obstacles", "ObsPowTwo"), ("power_obs4", "Power, 4 obstacles", "ObsPowFour")]
    o_rows, o_best = [], []
    bx = load(G / "results/box_effect.json")
    for key, lab, tag in obs:
        d = Sm[key]
        roa = 100 * torch.load(G / f"results/gt_{key}.pt", map_location="cpu")["roa"].float().mean().item()
        fp = max([load(G / f"runs/{key}_s{k}_rep/certify_core_relaxed.json")["false_positive"] for k in range(4)] + [load(G / f"results/core_union_relaxed_{key}.json")["false_positive"]])
        m_, s_ = statistics.mean(d["repair"]), statistics.stdev(d["repair"])
        assert statistics.mean(d["pinn2"]) > statistics.mean(d["pinn"])
        pick = d["pinn2"]
        pm, ps = statistics.mean(pick), statistics.stdev(pick)
        dm, ds = statistics.mean(d["dz"]), statistics.stdev(d["dz"])
        b_fps.append(fp); b_unions.append(d["union"])
        assert m_ > d["quad"] and bx[key]["ref_in_nobox"] and abs(bx[key]["roa_no_box_exit"] - roa) < 1e-9
        macros += [f"\\newcommand{{\\resSeed{tag}}}{{{pct(m_)}}}", f"\\newcommand{{\\resDz{tag}}}{{{pct(dm)}}}",
                   f"\\newcommand{{\\resPinn{tag}}}{{{pct(pm)}}}", f"\\newcommand{{\\resUnion{tag}}}{{{pct(d['union'])}}}"]
        o_rows.append(f"{lab} & {roa:.1f} & {d['quad']:.1f} & n/a & ${pm:.1f}\\pm{ps:.1f}$ & ${dm:.1f}\\pm{ds:.1f}$ & ${statistics.mean(d['ours']):.1f}\\pm{statistics.stdev(d['ours']):.1f}$ & ${m_:.1f}\\pm{s_:.1f}$ & {d['union']:.1f} & {fp:.2f}\\\\")
    mn_ = statistics.mean
    allk = [b[0] for b in bench] + [o[0] for o in obs]
    lead = [k for k in allk if mn_(Sm[k]["repair"]) > max(mn_(Sm[k]["pinn2"]), mn_(Sm[k]["dz"]))]
    assert lead == ["con3d"], lead
    assert all(mn_(Sm[k]["dz"]) > max(mn_(Sm[k]["pinn2"]), mn_(Sm[k]["repair"])) for k, *_ in obs)
    assert all(mn_(Sm[k]["repair"]) > Sm[k]["quad"] for k in allk)
    grids = {(kd, k): load(G / f"results/grid_relaxed_{kd}_{k}.json") for kd in ("pinn2", "dz") for k in allk}
    edges = {key_ for key_, g in grids.items() if g["selected"] == max(r["factor"] for r in g["grid"])}
    assert edges == {("pinn2", "vdp"), ("pinn2", "vdp_obs2"), ("dz", "pendulum"), ("dz", "vdp_obs2")}, edges
    assert all(300.0 in [r["factor"] for r in grids[("dz", k)]["grid"]] for k in allk)
    for (kd, k), g in grids.items():
        cov_ = {r["factor"]: r["coverage"] for r in g["grid"]}
        reach = [1.0, 3.0, 10.0, 30.0, 100.0] + ([300.0] if kd == "pinn2" else [])
        best_ = lambda: min((f for f in reach), key=lambda f: (-round(cov_[f], 6), f))
        if kd == "dz":
            for f in (300.0, 1000.0):
                if best_() == max(reach):
                    reach.append(f)
        assert sorted(cov_) == reach and g["selected"] == best_(), (kd, k)
        assert all(abs(c_ - g["selected"]) < 1e-9 for c_ in [r_.get("factor", g["selected"]) for r_ in load(G / f"results/baseline_{kd}_core_relaxed_{k}.json")])
    macros += [f"\\newcommand{{\\resObsNoObsMin}}{{{pct(min(bx[k]['roa_no_obstacles'] - bx[k]['roa'] for k, *_ in obs))}}}",
               f"\\newcommand{{\\resObsNoObsMax}}{{{pct(max(bx[k]['roa_no_obstacles'] - bx[k]['roa'] for k, *_ in obs))}}}"]
    st = {r["run"]: r for r in load(G / "results/sos_vdp_study.json")["runs"]}
    sset = load(G / "results/sos_settings.json")
    rc = load(G / "results/sos_recert_con3d.json")["recertification"]["level"]
    macros += [f"\\newcommand{{\\resSosLiuSetVdpOne}}{{{pct(st['mu_vdp_liu']['coverage'])}}}",
               f"\\newcommand{{\\resSosLiuSetVdpThree}}{{{pct(st['mu_vdp3_liu']['coverage'])}}}",
               f"\\newcommand{{\\resSosStudyOurs}}{{{pct(st['mu_vdp3_ours']['coverage'])}}}",
               f"\\newcommand{{\\resSosStudyShape}}{{{pct(st['mu_vdp3_ours_shapeid']['coverage'])}}}",
               f"\\newcommand{{\\resSosStudyInitx}}{{{pct(st['mu_vdp3_ours_initx']['coverage'])}}}",
               f"\\newcommand{{\\resSosStudySone}}{{{pct(st['mu_vdp3_ours_s1deg2']['coverage'])}}}",
               f"\\newcommand{{\\resSosStudyBoxGap}}{{{st['mu_vdp3_ours_nobox']['coverage'] - st['mu_vdp3_ours']['coverage']:.1f}}}",
               f"\\newcommand{{\\resSosStudyNobox}}{{{pct(st['mu_vdp3_ours_nobox']['coverage'])}}}",
               f"\\newcommand{{\\resSosLevelConThree}}{{{rc:g}}}",
               f"\\newcommand{{\\resSosTimePend}}{{{sset['pendulum']['seconds']/60:.0f}}}",
               f"\\newcommand{{\\resSosTimePower}}{{{sset['power']['seconds']/60:.0f}}}"]
    macros.append(f"\\newcommand{{\\resBenchFPSentence}}{{{fp_sentence(max(b_fps))}}}")
    p3 = sorted(Sm["vdp3"]["pinn"])
    macros += [f"\\newcommand{{\\resPinnVdpOneMean}}{{{pct(statistics.mean(Sm['vdp']['pinn']))}}}", f"\\newcommand{{\\resPinnVdpThreeLow}}{{{pct(p3[0])}}}",
               f"\\newcommand{{\\resPinnVdpThreeMin}}{{{p3[1]:.1f}}}", f"\\newcommand{{\\resPinnVdpThreeMax}}{{{pct(p3[3])}}}"]
    macros += [f"\\newcommand{{\\resBenchFPMax}}{{{pct(max(b_fps), 3)}}}", f"\\newcommand{{\\resSosGapMax}}{{{math.ceil(10 * max(sos_gap)) / 10:.1f}}}"]
    macros += [f"\\newcommand{{\\resBenchMin}}{{{pct(min(b_unions))}}}", f"\\newcommand{{\\resBenchMax}}{{{pct(max(b_unions))}}}",
               f"\\newcommand{{\\resBenchSeedMin}}{{{pct(min(b_means))}}}", f"\\newcommand{{\\resBenchSeedMax}}{{{pct(max(b_means))}}}"]
    (OUT / "table_bench3.tex").write_text(r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\caption{Coverage (\%) on the equilibrium benchmarks, mean $\pm$ std over four seeds where shown. Below the line: the obstacle benchmarks of \citet{mengliu2026}. Zubov PINN: searched scale. n/a: not run. FP as in Table~\ref{tab:main}.}
\label{tab:bench}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
System & \shortstack{Share\\of box} & Quadratic & SOS & \shortstack{Zubov\\PINN} & \shortstack{Dirichlet-\\Zubov PINN} & \shortstack{Ours,\\before} & \shortstack{Ours,\\seed} & \shortstack{Ours,\\union} & FP\\
\midrule
""" + "\n".join(b_rows) + "\n\\midrule\n" + "\n".join(o_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    core_rows, core_roa, core_t = [], [], []
    for key, lab in [(b[0], b[2]) for b in bench] + [(o[0], o[1]) for o in obs]:
        cj = load(G / f"results/core_{key}.json")
        assert cj["selftests_pass"] and cj["reference_states_in_E_outside_roa"] == 0
        core_rows.append(f"{lab} & {cj['c0']:.4g} & {cj['share_of_box']:.2f} & {cj['share_of_reference_roa']:.1f} & {cj['cells_final']} & {cj['seconds_total']:.0f}\\\\")
        core_roa.append(cj["share_of_reference_roa"]); core_t.append(cj["seconds_total"])
    macros += [f"\\newcommand{{\\resCoreRoaMin}}{{{pct(min(core_roa))}}}", f"\\newcommand{{\\resCoreRoaMax}}{{{pct(max(core_roa))}}}",
               f"\\newcommand{{\\resCoreTimeMax}}{{{max(core_t):.0f}}}"]
    (OUT / "table_core3.tex").write_text(r"""\begin{table}[ht]
\centering\small
\caption{Verified inner layers $E=\{q\le c_0\}$ of the equilibrium benchmarks. ROA: simulated region of attraction.}
\label{tab:core}
\begin{tabular}{lccccc}
\toprule
System & $c_0$ & $E$ (\% box) & $E$ (\% ROA) & Cells & Time (s)\\
\midrule
""" + "\n".join(core_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    Cs = load(G / "results/core_iid_summary.json")
    red, rec, rest, nseeds = [], [], [], 0
    for key, *_ in bench:
        e_share = load(G / f"results/core_{key}.json")["share_of_reference_roa"]
        for m in ("pinn", "pinn2", "dz"):
            for st_, rl_ in zip(Cs[key][m], Sm[key][m]):
                nseeds += 1
                if st_ - e_share < 0.5:
                    red.append(key)
                    (rec if rl_ - e_share > 5 else rest).append((key, m, rl_ - e_share, rl_))
    W_ = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    assert {(r[0], r[1]) for r in rest} == {("con3d", "pinn")} and all(r[2] < 0.05 for r in rest), (rec, rest)
    macros += [f"\\newcommand{{\\resRedN}}{{{len(red)}}}", f"\\newcommand{{\\resSeedsN}}{{{nseeds}}}", f"\\newcommand{{\\resRecN}}{{{len(rec)}}}",
               f"\\newcommand{{\\resRecLo}}{{{min(r[3] for r in rec):.1f}}}", f"\\newcommand{{\\resRecHi}}{{{max(r[3] for r in rec):.1f}}}",
               f"\\newcommand{{\\resRestN}}{{{W_[len(rest)]}}}"]
    gl = [r["c2_old"] - r["c2"] for g in grids.values() for r in g["grid"]]
    macros += [f"\\newcommand{{\\resGridLowerMax}}{{{math.ceil(1000 * max(gl)) / 1000:.3f}}}"]
    from scipy.stats import beta as beta_dist

    def pool(k):
        a, b = load(G / f"results/unsafe_core_{k}.json"), load(G / f"results/unsafe_core_relaxed_{k}.json")
        assert a["levels"] == b["levels"] and a["seed"] != b["seed"] and abs(a["union_coverage_of_S"] - b["union_coverage_of_S"]) < 1e-9
        assert a["N"] == b["N"] == 1000000
        k_ = max(a["unsafe"], b["unsafe"])
        eps_ = max(a["eps_k"], b["eps_k"])
        assert abs(eps_ - float(beta_dist.ppf(1 - 1e-6, k_ + 1, a["N"] - k_))) < 1e-12
        return dict(a, unsafe=k_, eps_k=eps_, longest_time_to_inner=max(a["longest_time_to_inner"], b["longest_time_to_inner"]))
    ca, cb = load(G / "results/unsafe_core_check.json"), load(G / "results/unsafe_core_relaxed_check.json")
    bc = {k: {**{f: ca[k][f] + cb[k][f] for f in ("n_compared", "disagreements", "unsafe_confirmed")},
              "unsafe_checks": ca[k]["unsafe_checks"] + cb[k]["unsafe_checks"]} for k in ca}
    bu = [pool(key) for key, *_ in bench]
    assert all(u["mode"] == "core" and abs(u["union_coverage_of_S"] - u["union_coverage_stored"]) < 1e-9 for u in bu)
    assert all(abs(u["union_coverage_of_S"] - Sm[k]["union"]) < 0.01 for u, (k, *_) in zip(bu, bench))
    assert all(u["levels"] == [load(G / f"runs/{k}_s{i}_rep/certify_core_relaxed.json")["c2"] for i in range(4)] for u, (k, *_) in zip(bu, bench))
    ks = [str(u["unsafe"]) for u in bu]
    ou = [pool(k) for k, *_ in obs]
    assert all(load(G / f"runs/{k}_s{i}_rep/certify_core.json")["c2"] == load(G / f"runs/{k}_s{i}_rep/certify_core_relaxed.json")["c2"]
               for k, *_ in bench + obs for i in range(4))
    assert all(u["levels"] == [load(G / f"runs/{k}_s{i}_rep/certify_core_relaxed.json")["c2"] for i in range(4)] for u, (k, *_) in zip(ou, obs))
    W12 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    mnn = statistics.mean
    assert all(load(G / f"results/sos_recert_{k}.json")["coverage"] > mnn(Sm[k]["repair"]) for k in ("vdp", "vdp3"))
    macros += [f"\\newcommand{{\\resPinnWinsN}}{{{W12[len(allk) - len(lead)]}}}"]
    oc = bc
    ko = [str(u["unsafe"]) for u in ou]
    macros += [f"\\newcommand{{\\resAllUnsafeMax}}{{{W12[max(u['unsafe'] for u in bu + ou)]}}}"]
    macros += [f"\\newcommand{{\\resOUnsafeList}}{{{', '.join(ko[:-1])} and {ko[-1]}}}",
               f"\\newcommand{{\\resOUnsafeEpsMax}}{{{sci(max(u['eps_k'] for u in ou))}}}",
               f"\\newcommand{{\\resOUnsafeDis}}{{{sum(oc[k]['disagreements'] for k, *_ in obs)}}}"]
    macros += [f"\\newcommand{{\\resBUnsafeList}}{{{', '.join(ks[:-1])} and {ks[-1]}}}",
               f"\\newcommand{{\\resBUnsafeEpsMax}}{{{sci(max(u['eps_k'] for u in bu))}}}",
               f"\\newcommand{{\\resBUnsafeTimeMax}}{{{max(u['longest_time_to_inner'] for u in bu):g}}}",
               f"\\newcommand{{\\resBUnsafeCmp}}{{{sum(bc[k]['n_compared'] for k, *_ in bench):,}}}".replace(",", "\\,"),
               f"\\newcommand{{\\resBUnsafeDis}}{{{sum(bc[k]['disagreements'] for k, *_ in bench)}}}",
               f"\\newcommand{{\\resBUnsafeTotal}}{{{sum(u['unsafe'] for u in bu)}}}",
               f"\\newcommand{{\\resBUnsafePend}}{{{['zero','one','two','three','four','five'][bu[[b[0] for b in bench].index('pendulum')]['unsafe']]}}}",
               f"\\newcommand{{\\resBUnsafePendSeeds}}{{{['no','one','two','three','four'][len({s_ for u_ in bc['pendulum']['unsafe_checks'] for s_ in u_['seeds_containing']})]}}}",
               f"\\newcommand{{\\resBUnsafeDisWord}}{{{'in no case' if sum(bc[k]['disagreements'] for k, *_ in bench) == 0 else 'in ' + str(sum(bc[k]['disagreements'] for k, *_ in bench)) + ' cases'}}}",
               f"\\newcommand{{\\resBUnsafeConfirmed}}{{{sum(bc[k]['unsafe_confirmed'] for k, *_ in bench)}}}"]
    import re
    defined = {re.match(r"\\newcommand\{\\(res\w+)\}", m_).group(1) for m_ in macros}
    used = set()
    for f in ("main.tex", "appendix3.tex"):
        used |= set(re.findall(r"\\(res[A-Z]\w*)", (OUT / f).read_text()))
    pending = sorted(used - defined)
    macros += [f"\\newcommand{{\\{n}}}{{\\TBD{{pending}}}}" for n in pending]
    if pending:
        print("PENDING macros:", pending)
    (OUT / "results3.tex").write_text("% written by make_results3.py from the run files; do not edit\n\\newcommand{\\TBD}[1]{\\textcolor{red}{[#1]}}\n" + "\n".join(macros) + "\n")
    (OUT / "table_main3.tex").write_text(r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Coverage (\%) of the simulated region of attraction on the TFL cases, mean $\pm$ std over four seeds where shown. Exact $v$: exact value function cut at the level of the candidate. FP: largest share of reference states inside an estimate and outside the simulated region of attraction.}
\label{tab:main}
\begin{tabular}{@{}lcccccccccc@{}}
\toprule
Case & \shortstack{Share\\of box} & Inner & Quadratic & \shortstack{Zubov\\PINN} & \shortstack{Dirichlet-\\Zubov PINN} & \shortstack{No\\decomp.} & \shortstack{Ours,\\seed} & \shortstack{Ours,\\union} & Exact $v$ & FP\\
\midrule
""" + "\n".join(main_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (OUT / "table_seeds3.tex").write_text(r"""\begin{table}[ht]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Per-seed results. Before: the trained network. Round: selected fine-tuning round (0 is the trained network). Sel.\ and Final: level after the retest and after the final test. Annulus: final-test samples between $c_1$ and the selected level. FP as in Table~\ref{tab:main}.}
\label{tab:seeds}
\begin{tabular}{llcccccccc}
\toprule
Case & Seed & $c_2$ before & Cov.\ before & Round & $c_2$ sel. & $c_2$ final & Annulus & Cov. & FP\\
\midrule
""" + "\n".join(seed_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (OUT / "table_consts3.tex").write_text(r"""\begin{table}[ht]
\centering\small
\caption{Constants computed before training: the cost scale $\gamma$ and the diagonal of $S_z^{-1}$.}
\label{tab:consts}
\begin{tabular}{@{}lcl@{}}
\toprule
Case & $\gamma$ & $S_z^{-1}$\\
\midrule
""" + "\n".join(const_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (OUT / "table_inner3.tex").write_text(r"""\begin{table}[ht]
\centering\small
\caption{Verified inner layers of the TFL cases, with the cells and time of the final verification.}
\label{tab:inner}
\begin{tabular}{lccccc}
\toprule
Case & $c_I$ & $c_1$ & $\I$ (\% box) & Cells & Time (s)\\
\midrule
""" + "\n".join(inner_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (OUT / "table_ablation3.tex").write_text(r"""\begin{table}[ht]
\centering\small
\caption{Ablations on seed 0 before fine-tuning, with the same level selection and final test. FP as in Table~\ref{tab:main}.}
\label{tab:ablation}
\begin{tabular}{llccc}
\toprule
Case & Variant & $c_2$ & Coverage & FP\\
\midrule
""" + "\n".join(abl_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""" if abl_rows else "\\begin{table}[ht]\\centering\\caption{\\TBD{ablations pending}}\\label{tab:ablation}\\end{table}\n")
    print("wrote", OUT / "results3.tex", "and tables")


if __name__ == "__main__":
    main()
