"""B2 check: the reference POPSRegression package on the same loss-weighted design as the ace-jax port.

The ace-jax POPS (pops_fit.py, ace_jax.fit.predict.PopsRidgePath) minimises sum_q sum_i (w_i/sigma_q)^2 (y_i - phi_i.c)^2
+ lam c^T Gamma^2 c.  With X = diag(w/sigma_q) Phi D^-1 (D = diag Gamma) and beta = D c that is an isotropic ridge on
X, which is what POPSRegression (a BayesianRidge) fits; its ridge rho = lambda_/alpha_ is chosen by evidence.  The rows
are ace-jax's own cached loss-weighted rows (stats.HostRows: w*phi, w*y), so the design is identical.

  parity    B2's own settings (the port's selected ridge, the BLR mean, the 50 % cut) put through the package's POPS step
            (its corrections, leverages and hypercube): leverages, box covariance and member spread against the B2 box.
            (The package's evidence ridge is ~1e-21 of max eigenvalue here, where no solve is meaningful, so the port is
            not compared at that ridge.)
  defaults  the package's own default (evidence ridge, leverage_percentile 50) and every row (0): members screened on
            three rattled diamond cells (pops_fit.member_screen) and the bulk-energy spread of the members.

    python -m bench.si_qoi_uq.analysis.pops_reference o3d12|o4d20      -> results/pops_reference_<m>.json
Needs popsregression (0.4.0: uv pip install popsregression==0.4.0); o4d20 holds ~0.5M x 5476 rows (~100 GB peak).
"""
import json, sys

import numpy as np

K = 200


def box_members(c, P, lo, hi, seed):
    """theta_k = c + P u_k, u_k uniform in [lo, hi]."""
    u = np.random.default_rng(seed).uniform(lo, hi, size=(K, len(lo)))
    return c, u @ P.T


def spread(m, c, d):
    """Members' bulk-energy spread (meV/atom, sd over members) on the equilibrium diamond cell, relative to c."""
    from ase.build import bulk
    from ..linear import observable_rows
    from ..run import lin_for
    at = bulk("Si", "diamond", a=5.46, cubic=True)
    bE = observable_rows(at, lin_for(m, "cr3"))[0][: d.shape[1]]
    return float(1e3 * np.std(d @ bE) / len(at))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import popsregression
    from popsregression import POPSRegression
    from ace_jax import cli
    from ace_jax.fit.pipeline import predict as ppred
    from ace_jax.fit.stats import HostRows
    from ..pops_fit import BASIS, DATA, member_screen
    from ..run import RESULTS, SEEDS, fit_dir
    m = sys.argv[1]
    out_dir = fit_dir(m, "pops").parent / "pops_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, orig = {}, ppred._pops_setup

    def keep(cfg, d, b, stats, theta, log):
        r = orig(cfg, d, b, stats, theta, log)
        kept.update(path=r["path"], ridge=r["ridge"], prob=b.prob, ds=d.ds_train)
        return r
    ppred._pops_setup = keep
    try:
        cli.main(["fit", "--train", f"{DATA}/train.xyz", "--test", f"{DATA}/test.xyz", *BASIS[m], "--rcut", "5.0",
              "--virial-key", "virial", "--weights", f"{DATA}/weights_acejax.json", "--e0", "lsq", "--rungs", "map",
              "--m-per-species", "0", "--noise", "shared", "--configs-per-batch", "1", "--seed", "0", "--uq", "pops",
              "--lml-solver", "cholesky", "--no-save-model", "--out", str(out_dir)])   # pops_fit.py's arguments
    except ValueError as e:                    # the model-file writer refuses a non-finite readout; the path is kept by then
        if "path" not in kept:
            raise
        print(f"fit output not written ({e}); using the captured POPS path", flush=True)
    path = kept["path"]
    rows = path.rows if isinstance(path.rows, HostRows) else HostRows(kept["prob"].model, kept["prob"].cfg, kept["ds"], path.qs)
    path.rows = rows
    Dinv = np.asarray(path.Dinv)
    X = np.concatenate(rows.blocks)[: rows.n_rows] * Dinv[None, :]
    y = np.concatenate(rows.targets)[: rows.n_rows]
    print(f"{m}: design {X.shape}", flush=True)
    ref = POPSRegression(fit_intercept=False, leverage_percentile=50.0, copy_X=False).fit(X, y)
    rho = float(ref.lambda_ / ref.alpha_)
    ridge_rel = rho / float(np.max(np.asarray(path.Lam)))
    c_ref = Dinv * ref.coef_
    assert np.all(np.isfinite(c_ref)), "reference coefficients not finite"
    res = {"model": m, "popsregression": popsregression.__version__, "design": list(X.shape), "ridge_abs_ref": rho,
           "ridge_rel_ref": ridge_rel, "ridge_port_selected": kept["ridge"], "K": K}
    # --- the package's own defaults: its evidence ridge, the 50 % cut and every row
    for lev, key in ((50.0, "reference_default"), (0.0, "reference_every_row")):
        if lev != 50.0:                                   # the same fit, only the member mask changes (as fit() sets it)
            ref._leverage_mask = ref._leverage_scores >= np.percentile(ref._leverage_scores, lev)
            ref.posterior_samples_, ref.misspecification_sigma_ = ref._build_posterior()
        P, (lo, hi) = ref._hypercube_support, ref._hypercube_bounds
        res[key] = {"leverage_pct": lev, "n_modes": int(P.shape[1]), "n_members": int(ref._leverage_mask.sum()),
                    "sd_E_meV_atom": spread(m, c_ref, box_members(c_ref, Dinv[:, None] * P, lo, hi, SEEDS["pops"])[1]),
                    **member_screen(m, c_ref, Dinv[:, None] * P, lo, hi, K=K)}
        print(key, res[key], flush=True)
    # --- parity at B2's own settings: the port's selected ridge, the BLR mean (POPS_MEAN), the 50 % cut.  The package's
    # POPS step is run on the port's regression: coef_ = D c, alpha_ sigma_ = D A D (= (X^T X + lam)^-1 on X), then the
    # corrections and leverages exactly as POPSRegression.fit computes them after its BayesianRidge step (popsregression
    # 0.4.0 _pops.py, fit: errors, pointwise_correction, leverage_scores, leverage_mask) and its own _fit_hypercube.
    rF = kept["ridge"]["F"] if isinstance(kept["ridge"], dict) else kept["ridge"]
    box = np.load(fit_dir(m, "pops") / "pops_box_lev50.npz")            # the B2 box the committees used
    path.use_mean(ppred.POPS_MEAN)
    c_now = np.asarray(path.c_star)                 # this ace-jax's BLR mean (its Cholesky can fail here: recorded only)
    c_port = np.asarray(box["c_star"])                                  # B2's own mean, stored with its box
    D = 1.0 / Dinv
    G = X.T @ X                                    # = D^-1 M D^-1, the port's scaled Gram, from the same rows
    ev, U = np.linalg.eigh(G)
    lam = float(rF) * float(ev.max())              # the port's relative ridge (PopsRidgePath.ridge_abs)
    Sb = (U / (np.maximum(ev, 0.0) + lam)) @ U.T   # (X^T X + lam)^-1 = D A D
    diag = {"port_gram_finite": bool(np.all(np.isfinite(np.asarray(path._Ms)))),
            "port_A_finite": bool(np.all(np.isfinite(np.asarray(path.A(rF))))),
            "gram_rel_diff_vs_port": float(np.linalg.norm(G - np.asarray(path._Ms)) / np.linalg.norm(G))}
    print("diagnostics", diag, flush=True)
    pc = X @ Sb
    h = np.sum(pc * X, axis=1)
    pc *= ((y - X @ (D * c_port)) / np.where(h > 0, h, np.inf))[:, None]
    keep = h >= np.percentile(h, 50.0)
    pk = POPSRegression(fit_intercept=False, leverage_percentile=50.0)
    P, (lo, hi) = pk._fit_hypercube(pc[keep])
    S_p, lo_p, hi_p = np.asarray(box["support"]), np.asarray(box["lo"]), np.asarray(box["hi"])
    cov = lambda S, a, b: S @ (np.diag((b - a) ** 2 / 12) + np.outer((a + b) / 2, (a + b) / 2)) @ S.T
    C_ref, C_port = cov(Dinv[:, None] * P, lo, hi), cov(S_p, lo_p, hi_p)
    # the same corrections in theta = D^-1 beta, the coordinates the port takes its box axes in: a box is not invariant
    # under the non-orthogonal D, so the package's box (axes from PCA in its input coordinates beta) and the port's differ
    P_t, (lo_t, hi_t) = pk._fit_hypercube((pc * Dinv[None, :])[keep])
    C_t = cov(P_t, lo_t, hi_t)
    res["parity"] = {
        "ridge_rel": float(rF), "mean": ppred.POPS_MEAN, "current_blr_mean_finite": bool(np.all(np.isfinite(c_now))),
        "n_modes_package": int(P.shape[1]), "n_modes_port": int(S_p.shape[1]), "n_members_package": int(keep.sum()),
        **diag,
        "box_cov_rel_diff": float(np.linalg.norm(C_ref - C_port) / np.linalg.norm(C_port)),
        "sd_E_meV_atom_package": spread(m, c_port, box_members(c_port, Dinv[:, None] * P, lo, hi, SEEDS["pops"])[1]),
        "sd_E_meV_atom_port": spread(m, c_port, box_members(c_port, S_p, lo_p, hi_p, SEEDS["pops"])[1]),
        "theta_axes": {"n_modes": int(P_t.shape[1]), "box_cov_rel_diff": float(np.linalg.norm(C_t - C_port) / np.linalg.norm(C_port)),
                       "sd_E_meV_atom_package": spread(m, c_port, box_members(c_port, P_t, lo_t, hi_t, SEEDS["pops"])[1])}}
    print("parity", res["parity"], flush=True)
    path.use_mean(ppred.POPS_MEAN)
    print(json.dumps({k: v for k, v in res.items() if k != "parity"}, indent=1))
    json.dump(res, open(RESULTS / f"pops_reference_{m}.json", "w"), indent=1)


def jnp_(a):
    import jax.numpy as jnp
    return jnp.asarray(a)


if __name__ == "__main__":
    main()
