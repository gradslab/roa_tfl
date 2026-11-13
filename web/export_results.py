r"""Numbers of the project page (docs/assets/data/results.json), read from the paper's generated macros and tables.

Every number on the page comes from paper/results3.tex (written by tfl/make_results3.py) or from the generated
tables, so the page and the paper cannot disagree.
Run from the repository root:  python web/export_results.py
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("ROA_PAPER_DIR", ROOT / "paper"))
OUT = ROOT / "docs" / "assets" / "data" / "results.json"


def clean(v):
    v = v.replace("\\,", "").replace("{,}", ",").replace("\\%", "%").strip()
    m = re.fullmatch(r"\$?([0-9.]+)\\times10\^\{(-?[0-9]+)\}\$?", v)
    if m:
        return f"{m.group(1)}e{m.group(2)}"
    return v.replace("$", "")


def macros():
    tex = (PAPER / "results3.tex").read_text()
    out = {}
    for name, val in re.findall(r"\\newcommand\{\\(res\w+)\}\{(.*)\}\s*$", tex, re.M):
        out[name] = clean(val)
    return out


def table(fname):
    rows = []
    body = (PAPER / fname).read_text().split("\\midrule", 1)[1].split("\\bottomrule")[0]
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\midrule"):
            if line.startswith("\\midrule"):
                rows.append(None)
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        rows.append(cells)
    return rows


def pm(c):
    c = c.replace("$", "")
    if "\\pm" in c:
        a, b = c.split("\\pm")
        return [float(a), float(b)]
    try:
        return [float(c), None]
    except ValueError:
        return None


def main():
    M = macros()
    systems = []
    for r in table("table_main3.tex"):
        if r is None:
            continue
        name, box, inner, quad, pinn, dz, nod, seed, union, exact, fp = r
        systems.append(dict(name={"Circle": "Unicycle, circle", "Sinusoid": "Unicycle, sinusoid", "Quadrotor": "Planar quadrotor"}.get(name, name),
                            group="TFL", roa=float(box), quad=pm(quad), pinn=pm(pinn), dz=pm(dz), nodecomp=pm(nod),
                            ours=pm(seed), union=float(union), exact=float(exact), sos=None, before=None))
    group = "Equilibrium"
    for r in table("table_bench3.tex"):
        if r is None:
            group = "Obstacles (Meng and Liu)"
            continue
        name, box, quad, sos, pinn, dz, before, seed, union, fp = r
        name = name.replace("$\\mu=1$", "μ = 1").replace("$\\mu=3$", "μ = 3").replace("$|u|\\le0.8$", "|u| ≤ 0.8")
        systems.append(dict(name=name, group=group, roa=float(box), quad=pm(quad), pinn=pm(pinn), dz=pm(dz), nodecomp=None,
                            ours=pm(seed), before=pm(before), union=float(union), sos=pm(sos) if sos != "n/a" else None))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(macros=M, systems=systems), indent=1, ensure_ascii=False))
    print("wrote", OUT, len(M), "macros", len(systems), "systems")


if __name__ == "__main__":
    main()
