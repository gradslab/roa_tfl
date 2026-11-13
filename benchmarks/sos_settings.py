r"""
Settings record of the sum-of-squares baselines -> results/sos_settings.json.

vdp, vdp3, con3d: sos.py and its stored results (results/baseline_sos_<sys>.json); the settings are read from the
code of sos.py (constants and degree rules), the iteration counts and the stopping reason from the stored iteration
history.  sos.py did not store solver statuses; results/sos_recert_<sys>.json (sos_sound.py --fixedV) re-solves the
certificate of each stored V at its level with Clarabel alone.
pendulum, power: sos_sound.py and results/baseline_sos_sound_<sys>.json (settings, statuses and stopping reason stored).
The stored sos.py results for pendulum and power (not sound, not reported) are listed under "superseded".

Run:  python sos_settings.py
"""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
load = lambda p: json.loads(p.read_text()) if p.exists() else None


def sos_py_stop(hist, max_iter=60):
    """Re-trace the stopping rule of sos.run from the stored history."""
    if not hist:
        return "no iteration stored"
    if hist[-1].get("beta_V") is None:
        return "V step infeasible (sos.py returns the last certified V)"
    best, stall = 0.0, 0
    for h in hist:
        stall = stall + 1 if h["beta"] <= best * (1 + 1e-3) else 0
        best = max(best, h["beta"])
        if stall >= 3:
            return "stall: 3 iterations without a 0.1 % gain in beta"
    if len(hist) >= max_iter:
        return f"max iterations ({max_iter})"
    return "multiplier step infeasible after 6 retries (not stored in the history)"


def sos_py_settings(name, res):
    nx = 3 if name == "con3d" else 2
    dV = 6 if nx == 2 else 4
    ddeg = {"vdp": 3, "vdp3": 3, "con3d": 3, "pendulum": 5, "power": 5}[name]
    d2 = ddeg - 1 + 2
    d2 += d2 % 2
    cons = {"vdp": "box faces (4)", "vdp3": "box faces (4)", "pendulum": "box faces (4)",
            "power": "box faces (4)", "con3d": "x3 limit, obstacle, box faces (6)"}[name]
    s = dict(code="sos.py", coordinates="box-normalized y = S x, S = diag(2 / (hi - lo))",
             V_degree=dV, V_monomials=f"degrees 2..{dV} in the state",
             shape="linearization quadratic y'Py (A'P + PA = -I in y), scaled to unit largest eigenvalue",
             init="the same quadratic, rescaled to its largest certified level",
             s1=f"SOS, degrees 0..{dV - 2}", s2=f"SOS, degrees 2..{d2}", s3=f"SOS, degrees 0..2, one per constraint",
             constraints=cons, eps=1e-6, level_backoff=0.98, bisection_steps=14,
             alternation=("multiplier step: bisect the largest level with s2, s3 and rescale V; bisect the largest "
                          "beta with s1; V step: maximize beta over V with s1, s2, s3 fixed; re-normalize"),
             stopping_rule="3 iterations without a 0.1 % gain in beta, or 60 iterations, or an infeasible step",
             solver=("Clarabel (CVXPY), SCS (max_iters 20000) if Clarabel raises; a solve is accepted with status "
                     "'optimal' or 'optimal_inaccurate'; statuses were not stored"))
    if name == "pendulum":
        s.update(nonpolynomial="sin(x1) -> T5(x1), no remainder bound",
                 saturation=("w = sat(u) auxiliary: um^2 - w^2 >= 0 and (u - w) w >= 0 with SOS multipliers "
                             "(degrees 0..2 in (y, w)), (u - w)(um^2 - w^2) = 0 with a free multiplier (degrees 0..2)"))
    if name == "power":
        s.update(nonpolynomial="sin(x1 + pi/3) -> degree-5 Taylor polynomial, no remainder bound")
    hist = res["iterations"]
    s.update(iterations=len(hist), stop=sos_py_stop(hist), seconds=res["seconds"], coverage=res["coverage"],
             false_positive=res["false_positive"], degree_of_returned_V=res["degree"],
             beta_history=[h["beta"] for h in hist])
    rc = load(R / f"sos_recert_{name}.json")
    if rc and "recertification" in rc:
        c = rc["recertification"]
        s["recertification_clarabel"] = dict(status=c["status"], accepted=c["accepted"], all_gram_certified=c["all_certified"], all_gram_exact_sos=c["all_exact"], mode=c.get("mode"), level=c["level"],
                                             min_eig_Q_plus_E=c["min_lam_Q_plus_E"], file=f"results/sos_recert_{name}.json",
                                             formulation=rc["settings"].get("remainder"))
        if "sample_check" in rc:
            s["sample_check"] = rc["sample_check"]
    return s


def sound_settings(name, res):
    s = dict(code="sos_sound.py", coordinates="box-normalized y = S x, S = diag(2 / (hi - lo))", **res["settings"])
    s.update(iterations=res["n_iterations"], stop=res["stop"], seconds=res["seconds"],
             iteration_seconds=res["iteration_seconds"], coverage=res.get("coverage"),
             false_positive=res.get("false_positive"), solver_status_counts=res["solver_status_counts"],
             beta_history=[h["beta"] for h in res["iterations"]])
    c = res.get("recertification")
    if c:
        s["recertification_clarabel"] = dict(status=c["status"], accepted=c["accepted"], all_gram_certified=c["all_certified"], all_gram_exact_sos=c["all_exact"], mode=c.get("mode"), level=c["level"],
                                             min_eig_Q_plus_E=c["min_lam_Q_plus_E"])
    if "sample_check" in res:
        s["sample_check"] = res["sample_check"]
    if name == "pendulum":
        s["saturation"] = ("w = sat(u) auxiliary indeterminate: um^2 - w^2 >= 0 and (u - w) w >= 0 with SOS "
                           "multipliers, (u - w)(um^2 - w^2) = 0 with a free multiplier; the real solution set is "
                           "exactly the graph of sat")
    return s


out = {}
for name in ("vdp", "vdp3", "con3d"):
    out[name] = sos_py_settings(name, load(R / f"baseline_sos_{name}.json"))
for name in ("pendulum", "power"):
    r = load(R / f"baseline_sos_sound_{name}.json")
    out[name] = sound_settings(name, r) if r else None
    old = load(R / f"baseline_sos_{name}.json")
    if old:
        out[name + "_superseded_sos_py"] = sos_py_settings(name, old)
(R / "sos_settings.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if v:
        rc = v.get("recertification_clarabel", {})
        print(f"{k:28s} V deg {v['V_degree']}  iters {v['iterations']:3d}  cov {v['coverage']:6.2f}  FP {v['false_positive']:.3f}  "
              f"stop: {v['stop'][:40]:40s} recert {rc.get('status')} certified {rc.get('all_gram_certified')} level {rc.get('level')}")
