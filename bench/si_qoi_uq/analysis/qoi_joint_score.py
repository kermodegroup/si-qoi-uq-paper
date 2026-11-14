"""Joint (correlation-aware) QoI scores beside the diagonal soft scores of qoi_soft_score.py, for every method on the
same 17 QoIs (the delta set: the coverage QoIs with DFT values and delta rows).

Inputs, all in tolerance units (x / tol, tolerances of ace-jax bench/si_qoi/score.py):
  committee and delta covariances S and errors d of CR3 (lam_F), B1 (q_b1), B2 (lev50, lev5: committee only) and B3
  from results/joint_tests.json (assemble.joint); the leverage-cut POPS ellipsoid (delta only) as lam_e^2 G Sigma_e G^T
  from its saved fit (RUNS/fits/<m>/pops_ellipse<sfx>.npz, lam_e = its per-atom RMS |e| / sd on the test split) and the
  QoI rows RUNS/delta/<m>.npz, its error as qoi_uq.csv's (dft - mean) about theta_hat.
  "lam_mv": CR3 and B1 delta covariances rescaled to the whole-configuration conformal scale of mv_conformal.py (§4).
  "+disc": the paper's discrepancy term (analysis/discrepancy.py, estimator headline) as a covariance: one shared bias
  per defect class, so the dumbbell, hexagonal and tetrahedral energies get a fully correlated block s s^T and the vacancy
  its own variance (the paper adds only the diagonal, which is what each QoI alone sees).
  tau: a floor tau^2 I on every covariance (tau = 0.05 tolerance by default, the floor at which score.py clips errors:
  errors below 5 % of a tolerance do not count). S is nearly singular (shared parameters), so a joint score is only
  defined with a floor; tau = 0.01 and 0.2 are reported as sensitivities.

Scores (n = 17 QoIs, Sigma = S [+ D] + tau^2 I):
  diagonal  gm sigma/t, gm |z|, gm CRPS/t (as qoi_soft_score.py, from diag Sigma), k95
  joint     sharp_j = det(Sigma)^(1/2n): the geometric mean of the principal widths (= gm sigma/t when uncorrelated)
            gm |w|, w = Sigma^-1/2 d (symmetric root): 17 whitened errors, N(0, 1) when calibrated, reference 0.530,
            band 0.31-0.90 as gm |z|
            d2/n = |w|^2 / n, chi^2_n / n when calibrated (17: 95 % band 0.44-1.78; one draw per model, weak evidence)
            LS/n: the Gaussian log score 0.5 (d^2 + log det(2 pi Sigma)) / n (proper; lower is better; sensitive to tau)
            ES/sqrt n: the energy score E|X - y| - 0.5 E|X - X'|, X ~ N(0, Sigma), y = d (the multivariate CRPS, proper,
            robust to thin directions; Monte Carlo, 20000 draws, fixed seed)

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.qoi_joint_score -> results/qoi_joint_score.json
"""
import csv, json, sys

import numpy as np
from scipy.stats import chi2, norm

from ..harness import RUNS
from ..run import RESULTS

TAUS = (0.05, 0.01, 0.2)
Z_REF = float(np.exp(-(np.euler_gamma + np.log(2.0)) / 2.0))
GM_FLOOR, GM_CAP = 0.05, 10.0
ELLIPSOIDS = {"o3d12": {"ellipsoid L-BFGS": "_sp_lev50_it20000", "ellipsoid reweighting": "_sp_lev50_cf100_fp2000"},
              "o4d20": {"ellipsoid L-BFGS": "_sp_lev50_it5000", "ellipsoid reweighting": "_sp_lev50_cf100_fp1000"}}
JT = {"CR3/lam_F": "CR3", "B1/q_b1": "B1", "B2lev50/lam_1": "B2lev50", "B2lev5/lam_1": "B2lev5", "B3/lam_1": "B3"}
INTERSTITIAL = ("dumbbell_interstitial_energy", "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy")


def gm(v, lo=GM_FLOOR, hi=GM_CAP):
    return float(np.exp(np.mean(np.log(np.clip(v, lo, hi)))))


def crps(d, s):
    z = d / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def energy_score(d, Sig, m=20000, seed=0):
    rng = np.random.default_rng(seed)
    w, V = np.linalg.eigh(Sig)
    R = V * np.sqrt(np.maximum(w, 0))
    X, X2 = rng.standard_normal((m, len(d))) @ R.T, rng.standard_normal((m, len(d))) @ R.T
    return float(np.mean(np.linalg.norm(X - d, axis=1)) - 0.5 * np.mean(np.linalg.norm(X - X2, axis=1)))


def scores(d, S, tau):
    n = len(d)
    Sig = S + tau ** 2 * np.eye(n)
    s = np.sqrt(np.diag(Sig))
    w_, V = np.linalg.eigh(Sig)
    wh = V @ ((V.T @ d) / np.sqrt(w_))
    d2 = float(wh @ wh)
    logdet = float(np.sum(np.log(w_)))
    return {"k95": int(np.sum(np.abs(d / s) <= 1.96)), "gm_sigma": gm(s), "gm_absz": gm(np.abs(d / s), 1e-3, 1e3),
            "gm_crps": gm(crps(d, s)), "sharp_j": float(np.exp(logdet / (2 * n))), "gm_absw": gm(np.abs(wh), 1e-3, 1e3),
            "d2_over_n": d2 / n, "p_chi2": float(chi2.sf(d2, n)),
            "ls_over_n": 0.5 * (d2 + logdet + n * np.log(2 * np.pi)) / n, "es_over_sqrt_n": energy_score(d, Sig) / np.sqrt(n)}


def main():
    jt = json.load(open(RESULTS / "joint_tests.json"))
    rows = list(csv.DictReader(open(RESULTS / "qoi_uq.csv")))
    disc = list(csv.DictReader(open(RESULTS / "discrepancy.csv")))
    out = {"taus": TAUS, "z_ref": Z_REF, "band_absw": [Z_REF * np.exp(-1.96 * np.sqrt(np.pi ** 2 / 8 / 17)),
                                                      Z_REF * np.exp(1.96 * np.sqrt(np.pi ** 2 / 8 / 17))],
           "band_d2_over_n": [chi2.ppf(0.025, 17) / 17, chi2.ppf(0.975, 17) / 17], "models": {}}
    for m in sys.argv[1:] or ("o3d12", "o4d20"):        # other fits (weights_qoi: o4d16, o4d16_Ex10) by argument
        keys = jt[m]["CR3/lam_F"]["delta"]["qois"]
        n = len(keys)
        hl = {r["qoi"]: r for r in rows if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
        tol = np.array([float(hl[k]["tol"]) for k in keys])
        sd = {r["qoi"]: float(r["sigma_disc"]) for r in disc if r["model"] == m and r["estimator"] == "headline"
              and r["method"] == "CR3-headline+disc" and r["scale"] == "lam_F"}
        sdt = np.array([sd.get(k, 0.0) for k in keys]) / tol
        ii = np.array([k in INTERSTITIAL for k in keys])
        D = np.outer(sdt * ii, sdt * ii) + np.diag(np.where(ii, 0.0, sdt ** 2))   # shared interstitial bias; vacancy alone
        cases = {}
        for jname, lab in JT.items():
            for kind, r in jt[m].get(jname, {}).items():
                idx = [r["qois"].index(k) for k in keys]
                cases[f"{lab} ({kind})"] = (np.array(r["d_tol"])[idx], np.array(r["S_tol"])[np.ix_(idx, idx)])
        mvf = RESULTS / f"mv_conformal_{m}.json"
        mv = json.load(open(mvf))["methods"] if mvf.exists() else None     # §4: the whole-configuration scale
        for lab, mk, jn in (("CR3 lam_mv", "cr3", "CR3/lam_F"), ("B1 lam_mv", "post", "B1/q_b1")) if mv else ():
            r = jt[m][jn]["delta"]
            idx = [r["qois"].index(k) for k in keys]
            f2 = (mv[mk]["lam_mv_all"] / mv[mk]["ref_scale"]) ** 2
            cases[f"{lab} (delta)"] = (np.array(r["d_tol"])[idx], f2 * np.array(r["S_tol"])[np.ix_(idx, idx)])
        dz = np.load(RUNS / "delta" / f"{m}.npz")
        G = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
        d_hl = np.array([(float(hl[k]["dft"]) - float(hl[k]["mean"])) for k in keys]) / tol
        for lab, sfx in ELLIPSOIDS.get(m, {}).items():
            z = np.load(RUNS / "fits" / m / f"pops_ellipse{sfx}.npz")
            Se = np.asarray(z["Sigma_e"])
            lam = json.load(open(RESULTS / f"pops_ellipse_{m}{sfx}.json"))["lam_per_atom_rms"]["ellipsoid"]
            Gt = np.stack([G[k][:Se.shape[0]] for k in keys]) / tol[:, None]
            cases[f"{lab} (delta)"] = (d_hl, lam ** 2 * Gt @ Se @ Gt.T)
        res = {}
        for name, (d, S) in cases.items():
            S = 0.5 * (S + S.T)
            res[name] = {f"tau{t:g}": scores(d, S, t) for t in TAUS}
            res[name + " +disc"] = {f"tau{t:g}": scores(d, S + D, t) for t in TAUS}
        out["models"][m] = {"qois": keys, "results": res}
        print(f"\n{m}: {n} QoIs, tau = 0.05 tol (diagonal | joint); calibrated: gm|z|, gm|w| in "
              f"{out['band_absw'][0]:.2f}-{out['band_absw'][1]:.2f}, d2/n in {out['band_d2_over_n'][0]:.2f}-{out['band_d2_over_n'][1]:.2f}")
        print(f"{'method':34s} {'k95':>5s} {'gmσ':>6s} {'gm|z|':>6s} {'CRPS':>6s} | {'sharpJ':>6s} {'gm|w|':>6s} {'d2/n':>6s} "
              f"{'LS/n':>6s} {'ES/√n':>6s} | {'d2/n τ.01':>9s} {'d2/n τ.2':>8s}")
        for name, v in res.items():
            a, lo, hi = v["tau0.05"], v["tau0.01"], v["tau0.2"]
            print(f"{name:34s} {a['k95']:2d}/{n} {a['gm_sigma']:6.2f} {a['gm_absz']:6.2f} {a['gm_crps']:6.3f} | "
                  f"{a['sharp_j']:6.3f} {a['gm_absw']:6.2f} {a['d2_over_n']:6.2f} {a['ls_over_n']:6.2f} {a['es_over_sqrt_n']:6.3f} | "
                  f"{lo['d2_over_n']:9.2f} {hi['d2_over_n']:8.2f}")
    json.dump(out, open(RESULTS / "qoi_joint_score.json", "w"), indent=1)   # SIQ_RESULTS keeps other fits' apart


if __name__ == "__main__":
    main()
