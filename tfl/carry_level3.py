"""When screening after fine-tuning keeps the trained network (round 0), the reported level of a run is the lower of its
two final tests: the one before fine-tuning and the one after.  A counterexample found for a network stays one.
Run:  python carry_level3.py circ3 sin6 quad3 ...   (edits runs/<name>_s<k>_rep3/certify_v3.json; idempotent)"""
import json
import sys
from pathlib import Path

KEYS = ("c2", "coverage", "false_positive", "region_fraction_of_box", "iid")
for name in sys.argv[1:]:
    for k in range(4):
        f0, f3 = Path(f"runs/{name}_s{k}/certify_v3.json"), Path(f"runs/{name}_s{k}_rep3/certify_v3.json")
        r0, r3 = json.loads(f0.read_text()), json.loads(f3.read_text())
        if r3["snapshot"] != "epoch_000.pt" or "second_final_test" in r3:
            continue
        if r0["c2"] < r3["c2"]:
            r3["second_final_test"] = {key: r3[key] for key in KEYS if key in r3}
            for key in KEYS:
                if key in r0:
                    r3[key] = r0[key]
            r3["level_from"] = "final test before fine-tuning (same network, lower level)"
            f3.write_text(json.dumps(r3, indent=1))
            print(f"{name}_s{k}: level {r3['second_final_test']['c2']:.4f} -> {r3['c2']:.4f}, coverage {r3['second_final_test']['coverage']:.2f} -> {r3['coverage']:.2f}")
        else:
            print(f"{name}_s{k}: kept ({r3['c2']:.4f} <= {r0['c2']:.4f})")
