"""Recompute the Gaussian reading of the learned scale from saved per-atom arrays (results/learned_scale_<tag>_atoms.npz):
sigma_F,i = (q / chi_3^-1(1 - alpha)) e^{f/2} sqrt(v) = radius_i sqrt(v_i) / chi_3^-1(1 - alpha).  learned_scale.py first used
an RMS of the normalised hold-out scores, which a few symmetry-zero calibration atoms blow up; the quantile is robust.
Rewrites rms_e_over_sigma_learned (and the median ratio, overall) in results/learned_scale_<tag>.json (all, per_type, per_group; mean over seeds).

    python -m bench.si_qoi_uq.ood.learned_sigma <tag> [<tag> ...]
"""
import json, sys

import numpy as np
from scipy.stats import chi

from ..run import RESULTS


def main():
    for tag in sys.argv[1:]:
        z = np.load(RESULTS / f"learned_scale_{tag}_atoms.npz", allow_pickle=True)
        res = json.load(open(RESULTS / f"learned_scale_{tag}.json"))
        k = chi(3).ppf(1 - float(z["alpha"]))
        seeds = sorted({int(n.rsplit("seed", 1)[1]) for n in z.files if "_radius_seed" in n})
        for name, o in res["sets"].items():
            e, v, t, g = z[f"{name}_e"], z[f"{name}_v"], z[f"{name}_t"], z[f"{name}_g"]
            sig = [z[f"{name}_radius_seed{sd}"] * np.sqrt(v) / k for sd in seeds]
            rms = lambda mk: float(np.mean([np.sqrt(np.mean((e[mk] / sg[mk]) ** 2)) for sg in sig]))
            o["all"]["rms_e_over_sigma_learned"] = rms(np.ones(len(e), bool))
            o["all"]["median_e_over_sigma_learned"] = float(np.mean([np.median(e / sg) for sg in sig]))
            for tt, r in o["per_type"].items():
                r["rms_e_over_sigma_learned"] = rms(t == tt)
            for gg, r in o["per_group"].items():
                r["rms_e_over_sigma_learned"] = rms(g == int(gg))
            print(tag, name, round(o["all"]["rms_e_over_sigma_learned"], 3))
        json.dump(res, open(RESULTS / f"learned_scale_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
