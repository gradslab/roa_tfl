r"""
Integration checks of the direct test of unsafe states on the benchmarks (unsafe_bench.py); evaluation only.  The
benchmark version of tfl/unsafe_check3.py.

1. Agreement with the reference simulation: 3000 reference states per system (results/gt_<sys>.pt) are classified by
   unsafe_bench.classify.  Where the two disagree, the reference simulator is run again with a 4x finer step (0.0025), and
   we count how often it then agrees with the test, how many remaining disagreements are conservative (test unsafe, finer
   reference converged), and how many disagreeing states lie in S.
2. Every sample that unsafe_bench.py counted as unsafe is integrated again with 4x and 16x finer substeps and with the
   reference simulator at step 0.001; the seeds whose estimate contains it and the failing constraint are listed.

Run:  python -u unsafe_bench_check.py vdp vdp3 pendulum con3d power
      (after unsafe_bench.py; results/unsafe_bench_check.json, entries of other systems already in the file are kept)
      python -u unsafe_bench_check.py --core vdp ...   (after unsafe_bench.py --core; results/unsafe_core_check.json)
"""
import json
import sys
from pathlib import Path

import torch

import systems as S
import unsafe_bench as ub
from gt import ground_truth

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "unsafe_bench_check.json"
OUT_CORE = ROOT / "results" / "unsafe_core_check.json"


def main():
    torch.set_num_threads(2)
    names = ub.parse_args(sys.argv[1:])
    dest = {"core": OUT_CORE, "core_relaxed": ROOT / "results" / "unsafe_core_relaxed_check.json"}.get(ub.MODE, OUT)
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for name in names:
        members = ub.load_union(name)
        sys_ = members[0]["prob"].sys
        E = ub.Core(name, members[0]["prob"])
        xg, roa = ground_truth(sys_)
        idx = torch.randperm(len(xg), generator=torch.Generator().manual_seed(0))[:3000]
        xr, r = xg[idx], roa[idx]
        lab, _, _ = ub.classify(sys_, E, xr)
        dis = (r & (lab != 0)) | (~r & (lab == 0))
        fine = sys_.simulate(xr[dis], dt=0.0025) == 0 if dis.any() else torch.zeros(0, dtype=torch.bool)
        agree_after = int(((lab[dis] == 0) == fine).sum())
        conservative = int(((lab[dis] != 0) & fine).sum())
        inS = ub.in_S(members, xr)
        dis_in_S = int((dis & inS).sum())
        test_safe_ref_not = int((~r & (lab == 0)).sum())
        test_unsafe_ref_conv = int((r & (lab != 0)).sum())
        u = json.loads((ROOT / "results" / ub.out_name(name)).read_text())
        recs = u.get("unsafe_all", [])
        ex = torch.tensor([c["state"] for c in recs], dtype=S.DT, device=S.DEV).reshape(-1, sys_.n)
        checks = []
        if len(ex):
            labs = {st: ub.classify(sys_, E, ex, step=st)[0].tolist() for st in (0.00625, 0.0015625)}
            ref = sys_.simulate(ex, dt=0.001).tolist()
            for i, c in enumerate(recs):
                checks.append(dict(state=c["state"], label=c["label"], failing_constraint=c["failing"],
                                   label_step_4x=labs[0.00625][i], label_step_16x=labs[0.0015625][i],
                                   reference_label_dt_0001=ref[i], seeds_containing=c["seeds_containing"],
                                   in_sampled_inner=c["in_sampled_inner"]))
        confirmed = sum(c["label_step_4x"] != 0 and c["label_step_16x"] != 0 and c["reference_label_dt_0001"] != 0 for c in checks)
        out[name] = dict(n_compared=len(idx), disagreements=int(dis.sum()), test_safe_reference_not_converged=test_safe_ref_not,
                         test_unsafe_reference_converged=test_unsafe_ref_conv,
                         agree_after_finer_reference=agree_after, remaining_conservative=conservative,
                         remaining_nonconservative=int(dis.sum()) - agree_after - conservative,
                         disagreements_in_S=dis_in_S, unsafe=u["unsafe"], unsafe_listed=len(recs),
                         unsafe_confirmed=confirmed, unsafe_checks=checks)
        print(name, {k: v for k, v in out[name].items() if k != "unsafe_checks"}, flush=True)
        dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
