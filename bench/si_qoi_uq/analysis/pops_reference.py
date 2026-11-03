"""B2 check: the reference POPSRegression package on the same loss-weighted design as the ace-jax port.

The ace-jax POPS (pops_fit.py, ace_jax.fit.predict.PopsRidgePath) minimises sum_q sum_i (w_i/sigma_q)^2 (y_i - phi_i.c)^2
+ lam c^T Gamma^2 c.  With X = diag(w/sigma_q) Phi D^-1 (D = diag Gamma) and beta = D c that is an isotropic ridge on
X, which is what POPSRegression (a BayesianRidge) fits; its ridge rho = lambda_/alpha_ is chosen by evidence.  The rows
are ace-jax's own cached loss-weighted rows (stats.HostRows: w*phi, w*y), so the design is identical.

  parity    the port at the reference's ridge and leverage cut, with the mean unpinned (the port pins the BLR mean by
            default; the package uses its own ridge solution): coefficients, leverages, and the box's member spread.
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
    from ace_jax.fit.pops import hypercube_support
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
    # --- parity: the port at the reference ridge, own mean, same cut
    path.use_mean(None)
    c_port = np.asarray(path.c_star_at(ridge_rel))
    res["port_cholesky_finite"] = bool(np.all(np.isfinite(c_port)))
    if not res["port_cholesky_finite"]:      # the port's Cholesky mean fails at a tiny ridge: use its eigendecomposition
        from ace_jax.fit.predict import _rkey
        c_port = np.asarray(path.A(ridge_rel) @ (path._Db / path.Dinv))
        path._means[_rkey(ridge_rel)] = jnp_(c_port)
    print(f"ridge rel {ridge_rel:.3e} (abs {rho:.3e}); port Cholesky mean finite: {res['port_cholesky_finite']}", flush=True)
    assert np.all(np.isfinite(c_port)), "port coefficients not finite at the reference ridge"
    res["parity"] = {"coef_rel_diff": float(np.linalg.norm(c_port - c_ref) / np.linalg.norm(c_ref))}
    for lev in (50.0, 0.0):
        if lev != 50.0:                                   # the same fit, only the member mask changes (as fit() sets it)
            ref._leverage_mask = ref._leverage_scores >= np.percentile(ref._leverage_scores, lev)
            ref.posterior_samples_, ref.misspecification_sigma_ = ref._build_posterior()
        P_ref, (lo_r, hi_r) = ref._hypercube_support, ref._hypercube_bounds
        A, coef, keepm = path._coef(ridge_rel, lev)
        W, _ = rows.moment_sums(coef)
        S = hypercube_support(A @ W @ A, 1.0e-8)
        lo_p, hi_p = rows.projection_bounds(coef, keepm, A @ S)
        h_port = np.asarray(rows.leverage_residual(jnp_(c_port), A)[0]).ravel()[: rows.n_rows]
        tag = f"lev{lev:g}"
        dr = box_members(c_ref, Dinv[:, None] * P_ref, lo_r, hi_r, SEEDS["pops"])[1]
        dp = box_members(c_port, np.asarray(S), np.asarray(lo_p), np.asarray(hi_p), SEEDS["pops"])[1]
        res["parity"][tag] = {
            "n_modes_ref": int(P_ref.shape[1]), "n_modes_port": int(np.asarray(S).shape[1]),
            "n_members_ref": int(ref._leverage_mask.sum()), "n_members_port": int(np.asarray(keepm).sum()),
            "leverage_max_rel_diff": float(np.max(np.abs(h_port - ref._leverage_scores)) / np.max(ref._leverage_scores)),
            "sd_E_meV_atom_ref": spread(m, c_ref, dr), "sd_E_meV_atom_port": spread(m, c_port, dp)}
        res["reference_default" if lev == 50.0 else "reference_every_row"] = {
            "leverage_pct": lev, "sd_E_meV_atom": res["parity"][tag]["sd_E_meV_atom_ref"],
            **member_screen(m, c_ref, Dinv[:, None] * P_ref, lo_r, hi_r, K=K)}
        print(tag, res["parity"][tag], flush=True)
    path.use_mean(ppred.POPS_MEAN)
    print(json.dumps({k: v for k, v in res.items() if k != "parity"}, indent=1))
    json.dump(res, open(RESULTS / f"pops_reference_{m}.json", "w"), indent=1)


def jnp_(a):
    import jax.numpy as jnp
    return jnp.asarray(a)


if __name__ == "__main__":
    main()
