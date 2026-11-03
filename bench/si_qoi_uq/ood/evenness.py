"""How even the per-atom force coverage is across GAP-18 configuration types: one hold-out conformal quantile against the
learned continuous scale (results/learned_scale_<tag>_atoms.npz).

Per set and method: coverage per configuration type with at least MIN_ATOMS atoms, and the RMS deviation of those
per-type coverages from the nominal 1 - alpha (types weighted equally; learned: mean over the split seeds).  Max - min is
not used: one outlying type sets it.

    python -m bench.si_qoi_uq.ood.evenness <tag> [<tag> ...]     -> results/evenness.json (merged per tag)
"""
import json, sys

import numpy as np

from ..run import RESULTS

MIN_ATOMS = 300


def main():
    path = RESULTS / "evenness.json"
    out = json.load(open(path)) if path.exists() else {}
    for tag in sys.argv[1:]:
        z = np.load(RESULTS / f"learned_scale_{tag}_atoms.npz", allow_pickle=True)
        nom, q1 = 1 - float(z["alpha"]), float(z["q_one"])
        seeds = sorted({int(n.rsplit("seed", 1)[1]) for n in z.files if "_radius_seed" in n})
        sets = sorted({n.split("_s")[0] for n in z.files if n.endswith("_s")})
        out[tag] = {}
        for st in sets:
            s, t = z[f"{st}_s"], z[f"{st}_t"]
            types, cnt = np.unique(t, return_counts=True)
            big = [tt for tt, c in zip(types, cnt) if c >= MIN_ATOMS]
            per = lambda cov: {tt: float(cov[t == tt].mean()) for tt in big}
            rms = lambda p: float(np.sqrt(np.mean([(v - nom) ** 2 for v in p.values()])))
            one = per(s <= q1)
            learned = [per(s <= z[f"{st}_radius_seed{k}"]) for k in seeds]
            lmean = {tt: float(np.mean([p[tt] for p in learned])) for tt in big}
            out[tag][st] = {"min_atoms": MIN_ATOMS, "n_types": len(big), "nominal": nom,
                            "one_group": {"per_type": one, "rms_dev": rms(one)},
                            "learned": {"per_type": lmean, "rms_dev": float(np.mean([rms(p) for p in learned])),
                                        "rms_dev_seed_range": [float(min(rms(p) for p in learned)),
                                                               float(max(rms(p) for p in learned))]}}
            print(tag, st, f"types {len(big)}: RMS deviation from {nom:.2f}: one quantile {rms(one):.3f}, learned "
                  f"{out[tag][st]['learned']['rms_dev']:.3f} {np.round(out[tag][st]['learned']['rms_dev_seed_range'], 3)}")
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
