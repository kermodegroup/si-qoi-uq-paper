"""B2 against POPSRegression, from the dumped design (analysis/pops_dump.py): which step makes them differ.

For the stored B2 box (pops_box_lev50.npz: mean c*, ridge, leverage cut) the POPS box is rebuilt from the same rows
three ways and compared with the stored one (box covariance, modes, members' bulk-energy spread):
  eig      the port's route: A from the eigendecomposition of X^T X (PopsRidgePath), corrections, PCA in theta
  svd      the same quantities from the SVD of X (stable at a tiny ridge: no squared condition number)
  package  POPSRegression 0.4.0's own _fit_hypercube on the svd corrections, in its input coordinates (beta = D theta)
            and in theta
plus the conditioning (cond X, and of X^T X + lam).

    python -m bench.si_qoi_uq.analysis.pops_compare o3d12     -> results/pops_compare_<m>.json
"""
import json, sys

import numpy as np

from .pops_dump import OUT


def box_cov(S, lo, hi):
    """Second moment of the uniform box on the axes S (hypercube_cov)."""
    mid = (lo + hi) / 2
    return S @ (np.diag((hi - lo) ** 2 / 12) + np.outer(mid, mid)) @ S.T


def pca_box(dt, thr=1.0e-8):
    """Box axes and bounds as both codes form them: eigenvectors of sum delta delta^T above thr * max, min/max."""
    ev, U = np.linalg.eigh(dt.T @ dt)
    S = U[:, ev > thr * ev.max()]
    p = dt @ S
    return S, p.min(0), p.max(0)


def main():
    from popsregression import POPSRegression
    from ase.build import bulk
    from ..linear import observable_rows
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    z = np.load(OUT / f"design_{m}.npz")
    X, y, Dinv = z["X"], z["y"], z["Dinv"]
    box = np.load(fit_dir(m, "pops") / "pops_box_lev50.npz")
    c, ridge, lev = np.asarray(box["c_star"]), float(box["ridge"]), float(box["leverage_pct"])
    D = 1.0 / Dinv
    at = bulk("Si", "diamond", a=5.46, cubic=True)
    bE = observable_rows(at, lin_for(m, "cr3"))[0][: len(c)] / len(at)
    sdE = lambda C: float(1e3 * np.sqrt(max(bE @ C @ bE, 0.0)))                 # meV/atom, box second moment
    C_b2 = box_cov(np.asarray(box["support"]), np.asarray(box["lo"]), np.asarray(box["hi"]))
    res = {"model": m, "ridge_rel": ridge, "leverage_pct": lev, "n_rows": int(X.shape[0]), "n_par": int(X.shape[1]),
           "b2": {"n_modes": int(box["support"].shape[1]), "sd_E_meV_atom": sdE(C_b2)}}
    G = X.T @ X
    ev, U = np.linalg.eigh(G)
    Us, s, Vt = np.linalg.svd(X, full_matrices=False)
    lam = ridge * float(ev.max())
    res["cond"] = {"cond_X": float(s.max() / s.min()), "cond_G_plus_lam": float((ev.max() + lam) / (max(ev.min(), 0) + lam)),
                   "eig_min_rel": float(ev.min() / ev.max()), "n_neg_eig": int((ev < 0).sum()), "lam_abs": lam,
                   "s2_min_rel": float(s.min() ** 2 / s.max() ** 2)}
    r = y - X @ (D * c)                                                        # residuals of B2's own mean
    # eig route (the port): Sb = (X^T X + lam)^-1 from eigh, corrections delta_beta = Sb x r / h
    Sb = (U / (np.maximum(ev, 0.0) + lam)) @ U.T
    pe = X @ Sb
    he = np.sum(pe * X, axis=1)
    # svd route: Sb x_i = V diag(s / (s^2 + lam)) u_i, h_i = sum_k u_ik^2 s_k^2 / (s_k^2 + lam)
    f = s / (s ** 2 + lam)
    ps = (Us * f) @ Vt
    hs = np.sum(Us ** 2 * (s ** 2 / (s ** 2 + lam)), axis=1)
    res["leverage"] = {"eig_vs_svd_max_rel": float(np.max(np.abs(he - hs)) / hs.max()), "eig_negative": int((he < 0).sum()),
                       "median_svd": float(np.median(hs)), "min_svd": float(hs.min())}
    for tag, p, h in (("eig", pe, he), ("svd", ps, hs)):
        keep = h >= np.percentile(h, lev)
        dbeta = p[keep] * (r[keep] / np.where(h[keep] > 0, h[keep], np.inf))[:, None]
        dth = dbeta * Dinv[None, :]
        S, lo, hi = pca_box(dth)
        C = box_cov(S, lo, hi)
        res[tag] = {"n_members": int(keep.sum()), "n_modes": int(S.shape[1]), "sd_E_meV_atom": sdE(C),
                    "cov_rel_diff_vs_b2": float(np.linalg.norm(C - C_b2) / np.linalg.norm(C_b2))}
        if tag == "svd":                                                       # the package's own hypercube code
            pk = POPSRegression(fit_intercept=False, leverage_percentile=lev)
            for coords, dd, back in (("beta", dbeta, Dinv), ("theta", dth, np.ones_like(Dinv))):
                P, (a, b) = pk._fit_hypercube(dd)
                Cp = box_cov(back[:, None] * P, a, b)
                res[f"package_{coords}"] = {"n_modes": int(P.shape[1]), "sd_E_meV_atom": sdE(Cp),
                                            "cov_rel_diff_vs_b2": float(np.linalg.norm(Cp - C_b2) / np.linalg.norm(C_b2))}
        print(tag, res[tag], flush=True)
    # ridge sweep: where is the box numerically defined?  eig and svd routes at each relative ridge, same mean and cut
    res["sweep"] = []
    for rr in (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        lam_ = rr * float(ev.max())
        row = {"ridge_rel": rr}
        for tag, p_, h_ in (("eig", X @ ((U / (np.maximum(ev, 0.0) + lam_)) @ U.T), None), ("svd", (Us * (s / (s ** 2 + lam_))) @ Vt, None)):
            h_ = np.sum(p_ * X, axis=1) if tag == "eig" else np.sum(Us ** 2 * (s ** 2 / (s ** 2 + lam_)), axis=1)
            keep = h_ >= np.percentile(h_, lev)
            dth = p_[keep] * (r[keep] / np.where(h_[keep] > 0, h_[keep], np.inf))[:, None] * Dinv[None, :]
            S_, lo_, hi_ = pca_box(dth)
            row[f"sd_E_{tag}"] = sdE(box_cov(S_, lo_, hi_)); row[f"modes_{tag}"] = int(S_.shape[1])
        res["sweep"].append(row)
        print("sweep", row, flush=True)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(RESULTS / f"pops_compare_{m}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
