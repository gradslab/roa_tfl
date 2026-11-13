r"""
Van der Pol (mu = 1 and 3): why the sum-of-squares coverage of sos.py differs from the SOS numbers of Liu et al. (2025,
Table 2: 94.17 % and 70.88 % "SOS volume", degree-6 V, "standard interior expanding algorithm" of Packard et al. 2010).
Packard et al. (2010) state the settings of that algorithm for Van der Pol: shape h(x) = x'x, V of degree 6
initialized with the quadratic Lyapunov function of the linearization (Q = I), multipliers s1 of degree 2 and s2 of
degree 4, 30 iterations, no state constraint.  Each run below is sos_sound.py with the options in `argv`; this script
collects the run files (given as arguments) into results/sos_vdp_study.json.

Run:  python sos_vdp_study.py <run files>.json
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "results" / "sos_vdp_study.json"
LIU = {"vdp1": 94.17, "vdp3": 70.88}
rows = []
for f in sys.argv[1:]:
    d = json.loads(Path(f).read_text())
    a = d["argv"]
    rc = d.get("recertification", {})
    rows.append(dict(run=Path(f).stem, system=d["system"], coverage=d.get("coverage"), false_positive=d.get("false_positive"),
                     liu_sos_volume=LIU.get(d["system"]), iterations=d["n_iterations"], best_iter=d.get("best_iter"),
                     stop=d["stop"], shape=a["shape"], init=a["init"], s1deg=a["s1deg"], s2deg=a["s2deg"], box=a["box"],
                     max_iter=a["iters"], stall=a["stall"], lenient=a.get("lenient", 0),
                     recert_level=rc.get("level"), recert_certified=rc.get("all_certified"), recert_mode=rc.get("mode"),
                     argv=a))
OUT.write_text(json.dumps(dict(
    note=("coverage = share of the reference ROA samples (gt.py) inside the certified set, false_positive = share of "
          "box samples in the set but outside the reference ROA (sos.evaluate_region); Liu et al. do not define "
          "'SOS volume' in the paper"),
    defaults=("shape 'lin' = linearization quadratic in box-normalized y; init 'y' = A'P + PA = -I in y; s1deg None = "
              "dV - 2 = 4; s2deg None = 4 for Van der Pol; box 1 = {V <= 1} inside the box; stall 3"),
    runs=rows), indent=1))
for r in rows:
    print(f"{r['run']:28s} {r['system']:5s} cov {r['coverage']:6.2f}  shape {r['shape']:3s} init {r['init']} s1 {r['s1deg']} "
          f"s2 {r['s2deg']} box {r['box']} it {r['iterations']:3d}/{r['max_iter']} stall {r['stall']} len {r['lenient']}  "
          f"recert {r['recert_level']} {r['recert_certified']}")
