r"""
Integration checks of the direct test of unsafe states (unsafe3.py); evaluation only.

1. Agreement with the reference simulation: 3000 reference states per case are classified by unsafe3.classify.  Where the
   two disagree, the reference simulator is run again with a 4x finer step (0.0025), and we count how often it then
   agrees with the test, how many remaining disagreements are conservative (test unsafe, finer reference converged), and
   how many disagreeing states lie in the union of the estimates.
2. Every sample that unsafe3.py counted as unsafe is integrated again with 4x and 16x finer substeps and with the reference
   simulator at step 0.001, and the seeds whose estimate contains it are listed.

Run:  python -u unsafe_check3.py circ3 sin6 quad3     (writes results/unsafe_check.json, after unsafe3.py)
"""
import json
import sys
from pathlib import Path

import torch

import unsafe3
from certify3 import Inner, ground_truth, inner_level, load


def main():
    out = {}
    for name in sys.argv[1:]:
        models = []
        for k in range(4):
            run = Path(f"runs/{name}_s{k}_rep3")
            res = json.loads((run / "certify_v3.json").read_text())
            prob, model, cfg = load(run)
            model.load_state_dict(torch.load(run / "snapshots" / res["snapshot"], map_location=cfg.device))
            models.append((model, res["c2"]))
        inner = Inner(prob, inner_level(cfg, prob))
        sysm = prob.sys
        xg, roa, label = ground_truth(prob)
        idx = torch.randperm(len(xg), generator=torch.Generator().manual_seed(0))[:3000]
        unsafe3.STEP = 0.025
        lab, _ = unsafe3.classify(prob, inner, xg[idx])
        r = roa[idx]
        dis = (r & (lab != 0)) | (~r & (lab == 0))
        fine = sysm.simulate(xg[idx][dis], dt=0.0025) == 0 if dis.any() else torch.zeros(0, dtype=torch.bool)
        agree_after = int(((lab[dis] == 0) == fine).sum())
        conservative = int(((lab[dis] != 0) & fine).sum())
        with torch.no_grad():
            inS = (inner(xg[idx]) < 1)
            for m_, c2_ in models:
                inS |= m_(xg[idx]) < c2_
        dis_in_S = int((dis & inS).sum())
        u = json.loads(Path(f"results/unsafe_{name}.json").read_text())
        ex = torch.tensor(u["unsafe_examples"], dtype=sysm.dtype, device=sysm.device).reshape(-1, xg.shape[1])
        checks = []
        if len(ex):
            labs = {}
            for st in (0.00625, 0.0015625):
                unsafe3.STEP = st
                labs[st] = unsafe3.classify(prob, inner, ex)[0].tolist()
            unsafe3.STEP = 0.025
            ref = sysm.simulate(ex, dt=0.001).tolist()
            with torch.no_grad():
                seeds = [[k for k, (m, c2) in enumerate(models) if m(ex[i:i + 1]).item() < c2] for i in range(len(ex))]
            for i in range(len(ex)):
                checks.append(dict(state=ex[i].tolist(), label_step_4x=labs[0.00625][i], label_step_16x=labs[0.0015625][i],
                                   reference_label_dt_0001=ref[i], seeds_containing=seeds[i]))
        confirmed = sum(c["label_step_4x"] != 0 and c["label_step_16x"] != 0 and c["reference_label_dt_0001"] != 0 for c in checks)
        out[name] = dict(n_compared=len(idx), disagreements=int(dis.sum()), agree_after_finer_reference=agree_after,
                         remaining_conservative=conservative, disagreements_in_S=dis_in_S,
                         unsafe=u["unsafe"], unsafe_confirmed=confirmed, unsafe_checks=checks)
        print(name, {k: v for k, v in out[name].items() if k != "unsafe_checks"}, flush=True)
    Path("results/unsafe_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
