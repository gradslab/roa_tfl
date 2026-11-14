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


def method_cfg(run):
    """Config of a run of the long-step setting; fails if the run was trained with another step, cost scale or exit weight.
    The inner layers were verified at gamma = 3/q90 (results/inner_*.json, inner_contact.json); the same sets at
    gamma = gamma_num/q90 have the levels times gamma_num/3."""
    c = load(Path(run) / "config.json")
    c = c.get("cfg", c)
    assert (c["tau"], c["sub"], c["gamma_num"], c["w_exit"]) == (2.0, 80, 1.0, 10.0), f"{run}: not the reported method"
    return c


def newer(f, *deps):
    """Fails if a result file is older than a file it was computed from.  Checked on reruns only (reproduce.sh tfl
    moves the stored runs to runs.stored): a checkout does not keep the file times of the stored runs."""
    if not Path("runs.stored").exists():
        return
    for d in deps:
        assert Path(f).stat().st_mtime >= Path(d).stat().st_mtime, f"stale {f}: older than {d}"


def rows_match(name):
    """Fails if a summary does not hold the level, coverage and false-positive share of the test result of each of its runs."""
    for r in load(f"results/summary_{name}.json")["rows"]:
        c = load(f"runs/{r['run']}/certify_v3.json")
        assert (r["c2"], r["coverage"], r["false_positive"]) == (c["c2"], c["coverage"], c["false_positive"]), f"summary_{name}: {r['run']} differs from its run"


def main():
    macros, main_rows, seed_rows, inner_rows, const_rows, abl_rows = [], [], [], [], [], []
    collapse_extra = []
    gains, kept, t_train, t_cert, t_rep, t_inner, iid_fail = [], 0, [], [], [], [], 0
    gscale = {}
    for name, case, label in CASES:
        S = load(f"results/summary_{name}.json")
        I = load(f"results/inner_{case}.json")
        cfgs = [method_cfg(f"runs/{name}_s{k}") for k in range(4)]
        assert [c["seed"] for c in cfgs] == [0, 1, 2, 3] and len({(c["mu_step"], c["w_cap"], c["kappa"], tuple(c["hidden"])) for c in cfgs}) == 1
        certs = [f"runs/{name}_s{k}_rep3/certify_v3.json" for k in range(4)]
        newer(f"results/summary_{name}.json", *certs)
        rows_match(name)
        gscale[case] = cfgs[0]["gamma_num"] / 3.0
        gamma_case, c_I = I["problem"]["alpha"] * gscale[case], I["c_inner"] * gscale[case]
        assert all(abs(c["c_inner"] - c_I) <= 1e-6 * max(1.0, c_I) for c in cfgs)
        for f_ in (f"results/ceiling_{name}.json", f"results/ablation_{name}.json"):
            if Path(f_).exists():
                newer(f_, *certs)
        if Path(f"results/summary_{name}w.json").exists():
            [method_cfg(f"runs/{name}w_s{k}") for k in range(4)]
            newer(f"results/summary_{name}w.json", *[f"runs/{name}w_s{k}_rep3/certify_v3.json" for k in range(4)])
            rows_match(name + "w")
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
        macros += [f"\\newcommand{{\\resPinnBothMax{m}}}{{{pct(max(statistics.mean(Pc), statistics.mean(Dz)))}}}",
                   f"\\newcommand{{\\resZubovMax{m}}}{{{pct(max(Qd, statistics.mean(Pc), statistics.mean(Dz)))}}}"]
        if A and "safety_value" in A:
            macros += [f"\\newcommand{{\\resSafety{m}}}{{{pct(A['safety_value']['coverage'])}}}"]
        if W:
            wrep = [r["coverage"] for r in W["rows"] if r["run"].endswith("_rep3")]
            macros += [f"\\newcommand{{\\resGap{m}}}{{{round(statistics.mean(cov_rep), 1) - round(statistics.mean(wrep), 1):.1f}}}", f"\\newcommand{{\\resWholeLevelMax{m}}}{{{max(r['c2'] for r in W['rows'] if r['run'].endswith('_rep3')):.4f}}}"]
            macros += [f"\\newcommand{{\\resOtherMax{m}}}{{{pct(max(Qd, statistics.mean(Pc), statistics.mean(Dz), statistics.mean(wrep)))}}}"]
            macros += [f"\\newcommand{{\\resWhole{m}}}{{{pct(statistics.mean(wrep))}}}",
                       f"\\newcommand{{\\resWholeUnion{m}}}{{{pct(W['union_repaired'])}}}"]
        if A and "pinn_style" in A:
            macros += [f"\\newcommand{{\\resPinn{m}}}{{{pct(A['pinn_style']['coverage'])}}}"]
        gains += [a - b for a, b in zip(cov_rep, cov_raw)]
        macros += [f"\\newcommand{{\\resRaw{m}Mean}}{{{pct(statistics.mean(cov_raw))}}}",
                   f"\\newcommand{{\\resRepairGain{m}}}{{{statistics.mean(cov_rep) - statistics.mean(cov_raw):+.1f}}}"]
        c1 = 1 - math.exp(-c_I)
        if C:
            assert all(C[f"{name}_s{k}_rep3"]["c2"] == load(certs[k])["c2"] for k in range(4)), f"ceiling_{name}: other levels"
        if A:
            for key_a, v_a in A.items():
                method_cfg(f"runs/{name}_abl_{key_a}")
                ra = load(f"runs/{name}_abl_{key_a}/certify_v3.json")
                assert (ra["c2"], ra["coverage"]) == (v_a["c2"], v_a["coverage"]), f"ablation_{name}: {key_a} differs from its run"
        c2s = [r["c2"] for r in rep]
        fps = max(max(r["false_positive"] for r in S["rows"]), S["union_repaired_fp"], S["union_unrepaired_fp"])
        ceil = f"{C['mean_ceiling']:.1f}" if C else "--"
        whole = "--"
        if W:
            wrep = [r["coverage"] for r in W["rows"] if r["run"].endswith("_rep3")]
            whole = f"${statistics.mean(wrep):.1f}\\pm{statistics.stdev(wrep):.1f}$"
        main_rows.append(f"{label} & {S['roa_fraction_of_box']:.1f} & {S['inner_coverage']:.1f} & {Qd:.1f} & "
                         f"${statistics.mean(Pc):.1f}\\pm{statistics.stdev(Pc):.1f}$ & ${statistics.mean(Dz):.1f}\\pm{statistics.stdev(Dz):.1f}$ & {whole} & "
                         f"${statistics.mean(cov_rep):.1f}\\pm{statistics.stdev(cov_rep):.1f}$ & {ceil} & {fps:.2f}\\\\")
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
            t_train.append(hist[-1]["time"] / 60)
            t_cert.append(r3["seconds"] / 60)
            t_rep.append(rp["seconds"] / 60)
            kept += r3["snapshot"] == "epoch_000.pt"
            kept_case += r3["snapshot"] == "epoch_000.pt"
            if r3["snapshot"] == "epoch_000.pt":
                draw_diff.append(abs(r3.get("second_final_test", r3)["coverage"] - r0["coverage"]))
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
        const_rows.append(f"{label} & {gamma_case:.4f} & " + ", ".join(f"{v:.3f}" for v in I['problem']['z_scale']) + "\\\\")
        inner_rows.append(f"{label} & {c_I:.4f} & {c1:.4f} & {100 * I['volume_of_box']:.2f} & {I['cells_processed']} & {I['seconds']:.0f}\\\\")
        if A:
            r00 = next(r for r in S["rows"] if r["run"] == f"{name}_s0")
            macros.append(f"\\newcommand{{\\resSeedZero{m}}}{{{pct(r00['coverage'])}}}")
            abl_rows.append(f"{label} & full method & {r00['c2']:.3f} & {r00['coverage']:.1f} & {r00['false_positive']:.3f}\\\\")
            for key, lab in (("pinn_style", "PDE residual loss alone"), 
                             ("safety_value", "safety-value baseline"), ("no_step", "no finite-step penalty, no counterexamples"),
                             ("no_wcap", "no exponential loss weight"), ("sum_ansatz", "sum form $\\Phi=\\psi+\\Psi_\\theta$")):
                if key in A:
                    abl_rows.append(f"{label} & {lab} & {A[key]['c2']:.3f} & {A[key]['coverage']:.1f} & {A[key]['false_positive']:.3f}\\\\")
    for n_, m_ in MAC.items():
        f_ = Path(f"results/unsafe_{n_}.json")
        if f_.exists():
            u_ = load(f_)
            cs_ = [f"runs/{n_}_s{k}_rep3/certify_v3.json" for k in range(4)]
            newer(f_, *cs_)
            assert u_["levels"] == [load(c_)["c2"] for c_ in cs_], f"unsafe_{n_}: other levels"
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
    ks_ = [load(f"results/unsafe_{n_}.json")["unsafe"] for n_ in MAC]
    word_ = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    if len(set(ks_)) == 1:
        macros.append(f"\\newcommand{{\\resUnsafeKPhrase}}{{{word_[ks_[0]]} unsafe sample{'s' if ks_[0] != 1 else ''} in the union of four estimates of each case}}")
    else:
        macros.append(f"\\newcommand{{\\resUnsafeKPhrase}}{{{ks_[0]}, {ks_[1]} and {ks_[2]} unsafe samples in the unions of four estimates}}")
    noise = load("results/ref_sensitivity.json")
    contact, floor_ = load("results/inner_contact.json"), load("results/floor.json")
    newer("results/floor.json", *[f"runs/{n_}_s{k}_rep3/certify_v3.json" for n_ in MAC for k in range(4)])
    if Path("results/unsafe_check.json").exists():
        newer("results/unsafe_check.json", *[f"results/unsafe_{n_}.json" for n_ in MAC])
    for key_, m_ in (("circle", "Circ"), ("sinusoid", "Sin"), ("quadrotor", "Quad")):
        macros += [f"\\newcommand{{\\resContact{m_}}}{{{contact[key_] * gscale[key_]:.4f}}}",
                   f"\\newcommand{{\\resCI{m_}}}{{{load(f'results/inner_{key_}.json')['c_inner'] * gscale[key_]:.4f}}}",
                   f"\\newcommand{{\\resFloor{m_}}}{{{pct(floor_[key_]['mean'])}}}"]
    macros.append(f"\\newcommand{{\\resRefNoiseMax}}{{{pct(max(v['changed_pct'] for v in noise.values()), 2)}}}")
    for key_, m_ in (("circle", "Circ"), ("sinusoid", "Sin"), ("quadrotor", "Quad")):
        macros.append(f"\\newcommand{{\\resRefNoise{m_}}}{{{pct(noise[key_]['changed_pct'], 2)}}}")
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
    step_rows = []
    for n_ in MAC:
        cs_ = load(f"runs_short_seed0/{n_}_s0/config.json")
        cs_ = cs_.get("cfg", cs_)
        assert cs_["seed"] == 0 and cs_["tau"] == 0.2 and not cs_.get("w_exit") and (cs_.get("gamma_num") or 3.0) == 3.0, f"runs_short_seed0/{n_}_s0: not the short-step setting"
        lo_ = load(f"runs/{n_}_s0/certify_v3.json")["region_fraction_of_box"]
        sh_ = load(f"runs_short_seed0/{n_}_s0/certify_v3.json")["region_fraction_of_box"]
        assert lo_ > sh_, f"{n_}: the step setting keeps the short step ({sh_:.2f} > {lo_:.2f}), but the reported runs use the long step"
        cl_, cs2_ = load(f"runs/{n_}_s0/certify_v3.json")["coverage"], load(f"runs_short_seed0/{n_}_s0/certify_v3.json")["coverage"]
        assert cl_ > cs2_
        step_rows.append(f"{dict(circ3='Circle', sin6='Sinusoid', quad3='Quadrotor')[n_]} & {lo_:.1f} & {sh_:.1f} & long & {cl_:.1f} & {cs2_:.1f} & {load(f'runs/{n_}_s0_rep3/certify_v3.json')['coverage']:.1f}\\\\")
    macros.append("\\newcommand{\\resStepTFL}{all three cases}")
    step_rows.append("\\midrule")
    sr_ = load(G / "results/step_rule.json")
    bn_ = {"vdp": "Van der Pol, $\\mu=1$", "vdp3": "Van der Pol, $\\mu=3$", "pendulum": "Pendulum", "con3d": "3-D system", "power": "Power system"}
    assert list(sr_) == list(bn_)
    for k_, r_ in sr_.items():
        assert r_["kept"] == ("long" if r_["long"] >= r_["short"] else "short")
        cl_, cs2_ = load(G / f"runs_long_seed0/{k_}_s0/certify.json")["coverage"], load(G / f"runs/{k_}_s0/certify.json")["coverage"]
        assert (cl_ > cs2_) == (r_["kept"] == "long"), f"{k_}: the label-free share and the coverage rank the settings differently"
        root_ = "runs_long_seed0" if r_["kept"] == "long" else "runs"
        step_rows.append(f"{bn_[k_]} & {r_['long']:.1f} & {r_['short']:.1f} & {r_['kept']} & {cl_:.1f} & {cs2_:.1f} & {load(G / f'{root_}/{k_}_s0_rep/certify.json')['coverage']:.1f}\\\\")
    kept_ = [load(G / f"{'runs_long_seed0' if r_['kept'] == 'long' else 'runs'}/{k_}_s0_rep/certify.json") for k_, r_ in sr_.items()]
    assert all(c_["false_positive"] == 0.0 for c_ in kept_)
    macros += [f"\\newcommand{{\\resStepBenchCovMin}}{{{pct(min(c_['coverage'] for c_ in kept_))}}}",
               f"\\newcommand{{\\resStepBenchCovMax}}{{{pct(max(c_['coverage'] for c_ in kept_))}}}"]
    macros.append("\\newcommand{\\resStepBenchLong}{" + ["no", "one", "two", "three", "four", "five"][sum(r_["kept"] == "long" for r_ in sr_.values())] + "}")
    (OUT / "table_step3.tex").write_text(r"""\begin{table}[ht]
\centering\small
\caption{Step setting of Algorithm~\ref{alg:method} on seed 0. Share: share of box samples in the estimate (\%), before fine-tuning, which decides the setting without reference labels. Coverage (\%) of the simulated region of attraction is shown for comparison, before fine-tuning for both settings and after fine-tuning for the kept one.}
\label{tab:step}
\begin{tabular}{@{}lcccccc@{}}
\toprule
System & \shortstack{Share,\\long} & \shortstack{Share,\\short} & Kept & \shortstack{Coverage,\\long} & \shortstack{Coverage,\\short} & \shortstack{Coverage,\\kept, fine-tuned}\\
\midrule
""" + "\n".join(step_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
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
\caption{Coverage (\%) of the simulated region of attraction on the TFL cases, mean $\pm$ std over four seeds. Exact $v$: exact value function cut at the level of the candidate. FP: largest share of reference states inside an estimate and outside that region.}
\label{tab:main}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
Case & \shortstack{Share\\of box} & Inner & Quadratic & \shortstack{Zubov\\PINN} & \shortstack{Dirichlet-\\Zubov PINN} & \shortstack{No\\decomp.} & Ours & Exact $v$ & FP\\
\midrule
""" + "\n".join(main_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (OUT / "table_seeds3.tex").write_text(r"""\begin{table}[t]
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
\caption{Constants computed before training: the cost scale $\gamma$ of the long-step setting and the diagonal of $S_z^{-1}$.}
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
    (OUT / "table_ablation3.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Ablations on seed 0 before fine-tuning, with the same level selection and final test. FP: share (\%) of reference states inside the estimate and outside the simulated region of attraction.}
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
