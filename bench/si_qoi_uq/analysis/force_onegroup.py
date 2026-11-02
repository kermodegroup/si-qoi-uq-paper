"""One conformal quantile for all atoms, from the same training hold-out as the Mondrian groups: the fair single-scale
comparison for the per-atom forces (the global lam_F of the propagation is fitted on the test split itself, so its
per-atom coverage there is in sample).

For each model: q_1, one quantile of the ARD posterior's calibration scores at the same alpha, atom-weighted (the
Mondrian per-group weights, each configuration 1 per group it populates, pooled over groups give the same quantile to
0.01), and its coverage on the test atoms of results/force_calibration_<m>.npz (same anisotropic scores), overall and per
Mondrian group, beside the Mondrian coverage.  q_1_cfgw, a configuration-weighted pooled quantile (each configuration 1
in total), is ~12 % lower and under-covers: the weighting, not the groups, is most of that difference.

    ARD_RUNS=<ARD root> python -m bench.si_qoi_uq.analysis.force_onegroup     -> results/force_onegroup.json
"""
import json, os, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "defect_uq" / "scoring"))
ARD_RUNS = pathlib.Path(os.environ.get("ARD_RUNS", "/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_ard137"))


def main():
    from ace_jax.fit.ard import ARDPosterior
    from novelty_groups import quantiles
    from ..run import RESULTS
    out = {}
    for m in ("o3d12", "o4d20"):
        post = ARDPosterior.load(ARD_RUNS / "fits" / m / "cr3" / "posterior.npz")
        alpha = float(post.group_table["alpha"])
        cal = post.cal
        sc = np.asarray(cal["scores"], float)
        q1 = [float(np.quantile(sc, 1 - alpha, method="higher"))]
        q1c, _ = quantiles(sc, np.zeros(len(sc), int), np.asarray(cal["cfg"]), 1, alpha, 1, [np.inf])
        z = np.load(RESULTS / f"force_calibration_{m}.npz", allow_pickle=True)
        s, g, qg = z["score"], z["group"], z["q"]
        G = int(g.max()) + 1
        cov = lambda c: {"all": float(c.mean()), "per_group": [float(c[g == k].mean()) for k in range(G)]}
        out[m] = {"alpha": alpha, "q_one": float(q1[0]), "q_one_cfgw": float(q1c[0]), "one_group": cov(s <= q1[0]),
                  "one_group_cfgw": cov(s <= q1c[0]), "mondrian": cov(s <= qg)}
        print(m, json.dumps(out[m]))
    json.dump(out, open(RESULTS / "force_onegroup.json", "w"), indent=1)


if __name__ == "__main__":
    main()
