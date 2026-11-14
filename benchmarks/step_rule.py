"""Step setting of a benchmark: compares the seed-0 estimates of the long and the short step, after level selection and
the final test and before fine-tuning, by their share of box samples (no reference labels).
Run:  python step_rule.py <system> ...   (prints the kept setting per system; writes results/step_rule.json for all given)"""
import json
import sys
from pathlib import Path

LONG = dict(tau=2.0, sub=80, gamma_num=1.0, w_exit=10.0)
SHORT = dict(tau=0.2, sub=8, gamma_num=3.0, w_exit=0.0)


def seed0(roots, name, want):
    for root in roots:
        d = Path(root) / f"{name}_s0"
        if (d / "certify.json").exists():
            c = json.loads((d / "config.json").read_text())
            c = c.get("cfg", c)
            got = {k: (c.get(k) if c.get(k) is not None else SHORT[k]) for k in want}
            if all(abs(float(got[k]) - float(want[k])) < 1e-9 for k in want):
                return json.loads((d / "certify.json").read_text())["box_volume"]
    return None


def rule(name):
    lo = seed0(("runs_long_seed0",), name, LONG)
    sh = seed0(("runs",), name, SHORT)
    assert lo is not None and sh is not None, f"{name}: seed 0 of both settings is needed"
    return dict(long=lo, short=sh, kept="long" if lo >= sh else "short")


if __name__ == "__main__":
    out = {n: rule(n) for n in sys.argv[1:]}
    if len(out) == 1:
        print(next(iter(out.values()))["kept"])
    else:
        Path("results/step_rule.json").write_text(json.dumps(out, indent=1))
        for n, r in out.items():
            print(f"{n:11s} long {r['long']:.2f}  short {r['short']:.2f}  kept {r['kept']}")
