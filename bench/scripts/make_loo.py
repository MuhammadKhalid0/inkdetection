"""Generate LOO 8-fold manifests + configs for the r2 out-of-fold refiner.

For each fold i, hold out TRAIN_SEGMENTS[i]: the fold model trains on the other
7 train segments (manifest train_reg_t192_loo<i>.csv) and NEVER sees the
held-out segment (not in training, not in selection -- selection stays on the
designated val segment, same protocol as the full model).  The fold model then
predicts the held-out segment, giving an honestly out-of-fold probability map
whose error statistics match deployment.  The val segment is excluded from
training in every model, as always.

Usage:  python -m scripts.make_loo
"""
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import MANIFEST_ROOT, TRAIN_SEGMENTS               # noqa: E402

BASE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "a2bs_t192.yaml")
BASE_MANIFEST = os.path.join(MANIFEST_ROOT, "train_reg_t192.csv")


def main():
    with open(BASE_MANIFEST) as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)

    cfg_dir = os.path.join(os.path.dirname(BASE_CONFIG))
    for i, held in enumerate(TRAIN_SEGMENTS):
        # manifest without the held-out segment
        mpath = os.path.join(MANIFEST_ROOT, f"train_reg_t192_loo{i}.csv")
        kept = [r for r in rows if r["segment"] != held]
        segs = sorted({r["segment"] for r in kept})
        assert held not in segs and len(segs) == 7, (held, segs)
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)

        # fold config = a2bs_t192 recipe, only manifest + run_name differ
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy
        cfg["data"]["manifest_suffix"] = f"_reg_t192_loo{i}"
        cfg["output"]["run_name"] = f"a2bs_t192_f{i}"
        cpath = os.path.join(cfg_dir, f"loo_t192_f{i}.yaml")
        with open(cpath, "w") as f:
            f.write(f"# LOO fold {i}: holds out {held} (OOF map generator for r2).\n"
                    f"# Identical to a2bs_t192.yaml except manifest/run_name.\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"fold {i}: held-out {held}  manifest {len(kept)} tiles -> {mpath}")
    print("done.")


if __name__ == "__main__":
    main()
