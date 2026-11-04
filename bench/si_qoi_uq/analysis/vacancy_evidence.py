"""Bayesian evidence for how the vacancy bias depends on training-cell size.  DFT training labels only: no QoI value or tolerance enters.

Data: the bulk-subtracted residual y_i = r_i - n_i e_bulk(x_i) of every training vacancy cell (the test split's are not used) (the regression form of
analysis/vacancy_check.py, per size: the bulk trend is fitted over dia cells within 0.6 A^3/atom of that size's volume),
against x_i = F_rms,i^2.  Sizes s in {63, 215} atoms.  Linear-Gaussian models y = X beta + e:

    pool    one intercept, one slope                      (the pooled estimators)
    int     per-size intercepts, one slope
    slope   one intercept, per-size slopes                (the sizes share the relaxed-defect bias)
    sep     per-size intercepts and slopes                (the per-size regression)

Priors: intercepts N(0, s_a^2), slopes N(0, (s_a / x_max)^2) (a slope moves y by at most ~s_a over the data), s_a = 1 eV;
per-size noise sigma_s, log-uniform on [1e-3, 10] eV (the same for every model).  beta is integrated analytically, the two
log sigma on a grid, so log p(y | M) is exact to the grid.  Prior sensitivity: s_a x 0.3 and x 3.

The zero-temperature bias of the 215-atom relaxed vacancy (the harness QoI's cell) under each model is the posterior of its
215-atom intercept; "bma" mixes the models by their posterior probabilities (equal prior odds).

    python -m bench.si_qoi_uq.analysis.vacancy_evidence     -> results/vacancy_evidence.json
"""
import json, pathlib

import numpy as np

H = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/defect_check")
D = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
SIZES = (63, 215)
MODELS = ("pool", "int", "slope", "sep")
S_A = (1.0, 0.3, 3.0)                                   # headline first
LOGSIG = np.linspace(np.log(1e-3), np.log(10.0), 241)


def design(name, x, s):
    one, a, b = np.ones_like(x), (s == SIZES[0]).astype(float), (s == SIZES[1]).astype(float)
    cols = {"pool": [one, x], "int": [a, b, x], "slope": [one, a * x, b * x], "sep": [a, b, a * x, b * x]}[name]
    kinds = {"pool": "ab", "int": "aab", "slope": "abb", "sep": "aabb"}[name]
    return np.stack(cols, 1), kinds


def intercept_215(name):
    """Coefficient vector c with c @ beta = the 215-atom intercept."""
    return np.array({"pool": [1, 0], "int": [0, 1, 0], "slope": [1, 0, 0], "sep": [0, 1, 0, 0]}[name], float)


def run(y, x, s, s_a):
    x_max = float(x.max())
    out = {}
    dl = LOGSIG[1] - LOGSIG[0]
    lprior = -2 * np.log(LOGSIG[-1] - LOGSIG[0])                       # log-uniform prior density on the grid, per size
    for name in MODELS:
        X, kinds = design(name, x, s)
        c = intercept_215(name)
        p0 = np.array([s_a ** 2 if k == "a" else (s_a / x_max) ** 2 for k in kinds])
        g = s == SIZES[0]
        lp = np.empty((len(LOGSIG), len(LOGSIG)))
        m1 = np.empty_like(lp); v1 = np.empty_like(lp)
        for i, l1 in enumerate(LOGSIG):
            for j, l2 in enumerate(LOGSIG):
                w = np.where(g, np.exp(-2 * l1), np.exp(-2 * l2))
                A = X.T @ (w[:, None] * X) + np.diag(1 / p0)
                L = np.linalg.cholesky(A)
                bv = X.T @ (w * y)
                mu = np.linalg.solve(A, bv)
                lp[i, j] = (-0.5 * (np.sum(w * y ** 2) - bv @ mu) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(np.log(p0))
                            + 0.5 * np.sum(np.log(w)) - 0.5 * len(y) * np.log(2 * np.pi))
                m1[i, j] = c @ mu
                v1[i, j] = c @ np.linalg.solve(A, c)
        t = lp + lprior + 2 * np.log(dl)
        logZ = float(np.logaddexp.reduce(t.ravel()))
        wpost = np.exp(t - logZ)
        mean = float(np.sum(wpost * m1))
        var = float(np.sum(wpost * (v1 + m1 ** 2)) - mean ** 2)
        i, j = np.unravel_index(np.argmax(t), t.shape)
        out[name] = {"log_evidence": logZ, "r215_mean_eV": mean, "r215_sd_eV": float(np.sqrt(var)),
                     "sigma_map_eV": {str(SIZES[0]): float(np.exp(LOGSIG[i])), str(SIZES[1]): float(np.exp(LOGSIG[j]))},
                     "n_params": X.shape[1]}
    lz = np.array([out[k]["log_evidence"] for k in MODELS])
    pm = np.exp(lz - np.logaddexp.reduce(lz))
    best = MODELS[int(np.argmax(lz))]
    for k, p in zip(MODELS, pm):
        out[k]["posterior_prob"] = float(p)
        out[k]["log_bf_vs_best"] = float(out[k]["log_evidence"] - out[best]["log_evidence"])
    mu = np.array([out[k]["r215_mean_eV"] for k in MODELS]); sd = np.array([out[k]["r215_sd_eV"] for k in MODELS])
    bm = float(pm @ mu)
    out["bma"] = {"r215_mean_eV": bm, "r215_sd_eV": float(np.sqrt(pm @ (sd ** 2 + mu ** 2) - bm ** 2)), "best": best}
    return out


def residuals(m, cfgs):
    """(y, x, size) of the vacancy cells, bulk-subtracted per size as in analysis/vacancy_check.py."""
    from .vacancy_check import ols
    n = np.array([len(c.numbers) for c in cfgs]); x = np.array([float(np.mean(c.forces ** 2)) for c in cfgs])
    v = np.array([abs(np.linalg.det(c.cell)) / len(c.numbers) for c in cfgs]); t = np.array([c.config_type for c in cfgs], dtype=object)
    js = json.load(open(H / f"{m}.json"))
    r = np.array([o["resid"] for o in js])
    assert len(r) == len(cfgs)
    tr = np.array([o["split"] == "train" for o in js])              # training labels only
    n, x, v, t, r = n[tr], x[tr], v[tr], t[tr], r[tr]
    ys, xs, ss = [], [], []
    for size in SIZES:
        mask = (t == "vacancy") & (n == size)
        bulk = (t == "dia") & (np.abs(v - v[mask].mean()) < 0.6)
        kb, _ = ols(x[bulk], r[bulk] / n[bulk])
        ys.append(r[mask] - n[mask] * (kb[0] + kb[1] * x[mask])); xs.append(x[mask]); ss.append(np.full(mask.sum(), size))
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(ss)


def main():
    from ace_jax.fit.data import load_configs
    from ..run import RESULTS
    types = ("interstitial", "vacancy", "divacancy", "dia")
    cfgs = [c for split in ("train", "test") for c in load_configs(f"{D}/{split}.xyz", virial_key="virial")
            if c.config_type in types and c.energy is not None]
    res = {"prior_intercept_sd_eV": list(S_A), "models": list(MODELS)}
    ys = {}
    for m in ("o3d12", "o4d20"):
        ys[m] = residuals(m, cfgs)
    ys["model_vs_model (o4d20 - o3d12)"] = (ys["o3d12"][0] - ys["o4d20"][0], ys["o3d12"][1], ys["o3d12"][2])
    for m, (y, x, s) in ys.items():
        res[m] = {f"s_a={sa:g}": run(y, x, s, sa) for sa in S_A}
        res[m]["n_cells"] = {str(k): int(np.sum(s == k)) for k in SIZES}
        h = res[m][f"s_a={S_A[0]:g}"]
        print(m, " ".join(f"{k}: logZ {h[k]['log_evidence']:.1f} p {h[k]['posterior_prob']:.3f} r215 {h[k]['r215_mean_eV']:+.3f}±{h[k]['r215_sd_eV']:.3f}"
                          for k in MODELS), f"| bma {h['bma']['r215_mean_eV']:+.3f}±{h['bma']['r215_sd_eV']:.3f}")
        for sa in S_A[1:]:
            hh = res[m][f"s_a={sa:g}"]
            print(f"   s_a={sa:g}:", " ".join(f"{k} {hh[k]['log_bf_vs_best']:+.1f}" for k in MODELS), f"bma {hh['bma']['r215_mean_eV']:+.3f}")
    json.dump(res, open(RESULTS / "vacancy_evidence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
