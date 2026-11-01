"""Which r* estimator extrapolates the defect-bulk contrast better?  Evidence that uses no DFT QoI value.

Estimators of the zero-thermal-amplitude defect formation bias (x = F_rms^2 of the defect cell):
  paired      each defect snapshot minus (n_def / n_b) x its nearest bulk 'dia' snapshot in (volume, x) (diff_rows.pair),
              OLS of that difference on x; intercept = r*                                    (analysis/bias_term.py)
  regression  bulk reference e_b(x) = per-atom OLS trend over bulk cells within 0.6 A^3/atom of the defect volume;
              OLS of (defect residual - n e_b(x)) on x; intercept = r*                       (analysis/defect_trend.py)
Both on the training cells only (the discrepancy term's data).

A. Model against model.  rho = E_o4d20 - E_o3d12 on every training cell (= resid_o3d12 - resid_o4d20) is a pseudo-residual
   whose relaxed-defect answer is known without DFT: the two models' own harness QoIs, Q_o4d20 - Q_o3d12.  Each
   estimator applied to rho predicts it.
B. Cold hold-out (DFT training labels).  Fit on all but the coldest HOLD of each defect type, predict the held-out cells'
   absolute defect residual (paired: trend + (n_def/n_b) x partner residual; regression: trend + n e_b(x)).
C. Noise structure of the paired form: n_def / n_b, the x mismatch within pairs, intercept standard errors.

    python -m bench.si_qoi_uq.analysis.estimator_check      (SIQ_RUNS: the BLR run root)
"""
import json, pathlib

import numpy as np

from .diff_rows import DEFECT, pair

HOLD = 0.2
H = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/defect_check")
D = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
QOI = {"interstitial": ("dumbbell_interstitial_energy", "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy"),
       "vacancy": ("vacancy_energy",)}


def ols(x, y):
    X = np.stack([np.ones_like(x), x], 1)
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = np.sum((y - X @ k) ** 2) / max(len(y) - 2, 1)
    return k, float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])), float(np.sqrt(s2))


def main():
    from ace_jax.fit.data import load_configs
    from ..propagate import KEYS, harness_values
    from ..run import RESULTS
    TYPES = DEFECT + ("dia",)
    cs = [c for c in load_configs(f"{D}/train.xyz", virial_key="virial") if c.config_type in TYPES and c.energy is not None]
    n = np.array([len(c.numbers) for c in cs])
    x = np.array([float(np.mean(c.forces ** 2)) for c in cs])
    v = np.array([abs(np.linalg.det(c.cell)) / len(c.numbers) for c in cs])
    t = np.array([c.config_type for c in cs], dtype=object)
    res = {m: np.array([o["resid"] for o in json.load(open(H / f"{m}.json")) if o["split"] == "train"]) for m in ("o3d12", "o4d20")}
    assert all(len(r) == len(cs) for r in res.values())
    pairs = pair(cs)
    pd_ = {d: b for d, b in pairs}

    def paired(r, mask):
        """(intercept, se, scatter, predict(idx) of the absolute defect residual) from defect cells `mask`."""
        d = np.flatnonzero(mask)
        b = np.array([pd_[i] for i in d])
        y = r[d] - n[d] / n[b] * r[b]
        k, se, sc = ols(x[d], y)
        return k[0], se, sc, (lambda idx: k[0] + k[1] * x[idx] + n[idx] / n[[pd_[i] for i in idx]] * r[[pd_[i] for i in idx]])

    def regression(r, mask, ty):
        d = np.flatnonzero(mask)
        v0 = v[t == ty].mean()
        bulk = np.flatnonzero((t == "dia") & (np.abs(v - v0) < 0.6))
        kb, _, _ = ols(x[bulk], r[bulk] / n[bulk])
        y = r[d] - n[d] * (kb[0] + kb[1] * x[d])
        k, se, sc = ols(x[d], y)
        return k[0], se, sc, (lambda idx: k[0] + k[1] * x[idx] + n[idx] * (kb[0] + kb[1] * x[idx]))

    out = {"A_model_vs_model": {}, "B_cold_holdout": {}, "C_noise": {}}
    # A. model against model: truth from the two models' harness QoIs (no DFT)
    rho = res["o3d12"] - res["o4d20"]                       # E_o4d20 - E_o3d12 per cell
    Q3, Q4 = harness_values("SIQ_o3d12_cr3"), harness_values("SIQ_o4d20_cr3")
    for ty, keys in QOI.items():
        mask = t == ty
        truth = [float(Q4[KEYS.index(k)] - Q3[KEYS.index(k)]) for k in keys]
        a, sa, _, _ = paired(rho, mask)
        b, sb, _, _ = regression(rho, mask, ty)
        # sign: rho = E4 - E3 on a cell; a formation-energy difference Q4 - Q3 has the same sign as the contrast in rho
        out["A_model_vs_model"][ty] = {"truth_Q_o4d20_minus_Q_o3d12_eV": truth, "paired_eV": [float(a), sa],
                                       "regression_eV": [float(b), sb],
                                       "abs_err_paired": [abs(a - q) for q in truth], "abs_err_regression": [abs(b - q) for q in truth]}
    # B. cold hold-out on the DFT training labels
    for m, r in res.items():
        out["B_cold_holdout"][m] = {}
        for ty in DEFECT:
            idx = np.flatnonzero(t == ty)
            cut = np.quantile(x[idx], HOLD)
            fit_m = (t == ty) & (x > cut)
            held = idx[x[idx] <= cut]
            _, _, _, fp = paired(r, fit_m)
            _, _, _, fr = regression(r, fit_m, ty)
            e_p, e_r = r[held] - fp(held), r[held] - fr(held)
            out["B_cold_holdout"][m][ty] = {"n_held": int(len(held)), "x_cut": float(cut),
                                            "rmse_paired_eV": float(np.sqrt(np.mean(e_p ** 2))), "bias_paired_eV": float(e_p.mean()),
                                            "rmse_regression_eV": float(np.sqrt(np.mean(e_r ** 2))), "bias_regression_eV": float(e_r.mean())}
    # C. noise structure of the paired form
    for ty in DEFECT:
        d = np.array([i for i, _ in pairs if t[i] == ty])
        b = np.array([pd_[i] for i in d])
        ratio = n[d] / n[b]
        out["C_noise"][ty] = {"ratio_quantiles": np.percentile(ratio, [10, 50, 90]).tolist(),
                              "partner_sizes": sorted(set(int(k) for k in n[b])),
                              "x_mismatch_rel_median": float(np.median(np.abs(x[b] - x[d]) / x[d])),
                              "v_mismatch_median_A3": float(np.median(np.abs(v[b] - v[d])))}
        for m, r in res.items():
            _, sa, sca, _ = paired(r, t == ty)
            _, sb, scb, _ = regression(r, t == ty, ty)
            out["C_noise"][ty][m] = {"se_paired": sa, "se_regression": sb, "scatter_paired": sca, "scatter_regression": scb,
                                     "partner_term_rms_eV": float(np.sqrt(np.mean((ratio * r[b]) ** 2)))}
    json.dump(out, open(RESULTS / "estimator_check.json", "w"), indent=1, default=float)
    A = out["A_model_vs_model"]
    for ty in QOI:
        print(f"A {ty:12s}: truth (Q_o4d20 - Q_o3d12) {np.round(A[ty]['truth_Q_o4d20_minus_Q_o3d12_eV'], 3)}; "
              f"paired {A[ty]['paired_eV'][0]:+.3f} ± {A[ty]['paired_eV'][1]:.3f}; regression {A[ty]['regression_eV'][0]:+.3f} ± {A[ty]['regression_eV'][1]:.3f}")
    for m in res:
        for ty in DEFECT:
            b = out["B_cold_holdout"][m][ty]
            print(f"B {m} {ty:12s} (n={b['n_held']}): RMSE paired {b['rmse_paired_eV']:.3f} (bias {b['bias_paired_eV']:+.3f}), "
                  f"regression {b['rmse_regression_eV']:.3f} (bias {b['bias_regression_eV']:+.3f}) eV")
    for ty in DEFECT:
        c = out["C_noise"][ty]
        print(f"C {ty:12s}: n_def/n_b p10/50/90 {np.round(c['ratio_quantiles'], 1)}, partners {c['partner_sizes']}, x mismatch {c['x_mismatch_rel_median']:.2f}; "
              + "; ".join(f"{m}: se {c[m]['se_paired']:.3f} vs {c[m]['se_regression']:.3f}, partner term rms {c[m]['partner_term_rms_eV']:.3f}" for m in res))


if __name__ == "__main__":
    main()
