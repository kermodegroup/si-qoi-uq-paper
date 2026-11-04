"""Defect-cell energy residual against temperature, with a temperature-matched bulk reference.
Regressor x = F_rms^2 (thermal potential energy and a stiffness-mismatch error both scale with <u^2> ~ F^2).
Bulk: dia cells within 0.6 A^3/atom of the defect cells' mean volume per atom, anchored at x = 0 by the perfect
crystals (F_rms < 1e-3).  Defect: r - n e_bulk(x), regressed on x; the intercept is the predicted error of the
model's formation energy at a relaxed (x -> 0) defect.  Training cells only (the residuals file also holds the test
split's, which are not used).

    python -m bench.si_qoi_uq.analysis.defect_trend      -> results/defect_trend.json (both models)"""
import json, sys, pathlib
import numpy as np
from ace_jax.fit.data import load_configs
H = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/defect_check")
D = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
TYPES = ("interstitial", "vacancy", "divacancy", "dia")
meta = []
for split in ("train", "test"):
    cs = [c for c in load_configs(f"{D}/{split}.xyz", virial_key="virial") if c.config_type in TYPES and c.energy is not None]
    meta += [(float(np.mean(c.forces ** 2)), abs(np.linalg.det(c.cell)) / len(c.numbers)) for c in cs]


def fit(x, y):
    A = np.stack([np.ones_like(x), x], 1)
    k, *_ = np.linalg.lstsq(A, y, rcond=None)
    cov = np.sum((y - A @ k) ** 2) / max(len(y) - 2, 1) * np.linalg.inv(A.T @ A)
    return k, np.sqrt(np.diag(cov))


def trend(m):
    out = json.load(open(H / f"{m}.json"))
    assert len(out) == len(meta)
    for o, (x, v) in zip(out, meta):
        o["x"], o["v"] = x, v
    out = [o for o in out if o["split"] == "train"]
    res = {}
    for t in ("interstitial", "vacancy", "divacancy"):
        s = [o for o in out if o["type"] == t]
        v0 = np.mean([o["v"] for o in s])
        dia = [o for o in out if o["type"] == "dia" and abs(o["v"] - v0) < 0.6]
        xb = np.array([o["x"] for o in dia]); eb = np.array([o["resid"] / o["n"] for o in dia])
        kb, sb = fit(xb, eb)
        x = np.array([o["x"] for o in s]); n = np.array([o["n"] for o in s]); r = np.array([o["resid"] for o in s])
        rf = r - n * (kb[0] + kb[1] * x)
        k, se = fit(x, rf)
        res[t] = {"n": len(s), "v0": v0, "n_bulk": len(dia), "n_bulk_perfect": int(np.sum(xb < 1e-6)),
                  "bulk_per_atom_meV": [1e3 * kb[0], 1e3 * kb[1]], "bulk_se_meV": [1e3 * sb[0], 1e3 * sb[1]],
                  "defect_intercept_eV": k[0], "defect_intercept_se": se[0], "defect_slope_eV_per_eV2A2": k[1],
                  "defect_mean_eV": float(rf.mean()), "x_range": [float(x.min()), float(x.max())]}
        print(f"{m} {t:12s} n={len(s):3d}, F_rms^2 {x.min():.3f}-{x.max():.3f}; bulk ({len(dia)} cells, {int(np.sum(xb < 1e-6))} perfect): "
              f"{1e3 * kb[0]:+.2f}±{1e3 * sb[0]:.2f} + {1e3 * kb[1]:+.2f} x meV/atom | defect - bulk(x): mean {rf.mean():+.3f} eV, "
              f"at x=0 {k[0]:+.3f} ± {se[0]:.3f} eV (slope {k[1]:+.3f})")
    return res


from bench.si_qoi_uq.run import RESULTS
json.dump({m: trend(m) for m in ("o3d12", "o4d20")}, open(RESULTS / "defect_trend.json", "w"), indent=1, default=float)
