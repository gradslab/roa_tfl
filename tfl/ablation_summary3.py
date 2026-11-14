"""Collect the finished seed-0 ablations and safety-value baselines into results/ablation_<name>.json (make_results3.py reads them)."""
import json
from pathlib import Path

for n in ("circ3", "sin6", "quad3"):
    out = {}
    for d in sorted(Path("runs").glob(f"{n}_abl_*")):
        f = d / "certify_v3.json"
        if f.exists() and f.stat().st_mtime >= (d / "model.pt").stat().st_mtime:
            r = json.loads(f.read_text())
            out[d.name.split("_abl_")[1]] = {k: r[k] for k in ("c2", "coverage", "false_positive")}
    if out:
        Path(f"results/ablation_{n}.json").write_text(json.dumps(out, indent=1))
    print(n, out)
