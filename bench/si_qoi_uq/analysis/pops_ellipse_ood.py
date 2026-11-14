"""The leverage-cut POPS ellipsoid (analysis/pops_ellipse.py `--smooth-prior --leverage-pct 50`) on the out-of-distribution
refit: per-atom force coverage on the held-out cracks and screw dislocations (the paper's held-out test, force_methods.py).

The same recipe, re-run on the refit (ood/make_split.py: the training set without cracks and screw dislocations,
OOD_RUNS/fits/<m>, its resolved fit.yaml and theta_map.json): every training E, F, V row weighted as that fit weights
them, targets the weighted residuals about the refit's theta_hat, columns rescaled by Lambda^-1/2 (the refit's own prior
precision at its MAP), rows at or above the 50th percentile of posterior leverage (two-stage QR), popsregression
`ellipse-ev` defaults (rank 32, whiten_ridge 1/N).  Nothing is chosen on the held-out set.
Scored as force_methods.py scores the paper's methods (same atoms: CR3 sd > 1e-3 x median; same eps regularisation;
d^2 = e^T C^-1 e <= chi2_3(0.9)):
  ellipsoid       Sigma_e = B / (P + 2), no scale
  ellipsoid+lam   lam_e^2 Sigma_e, lam_e the per-atom RMS |e| / sd on the refit's in-distribution test split
                  (OOD_DATA/test.xyz), as force_methods.py refits lam_F and q_b1 there
  ellipsoid_span  Sigma_e projected on the refit's data span (G eigenvalues above 1e-8 x the largest), and its +lam
and set beside the headline / B1 / B2 / B3 rows of results/force_methods_<m>.json (in and out of distribution).

    SIQ_RUNS=<BLR root> OOD_RUNS=<ood_holdout_blr> PYTHONPATH=<popsregression checkout> \\
        python -m bench.si_qoi_uq.analysis.pops_ellipse_ood o3d12 [--max-iter k]
        -> OOD_RUNS/fits/<m>/pops_ellipse_sp_lev50<sfx>.npz, results/pops_ellipse_ood_<m><sfx>.json
--max-iter k: L-BFGS iterations per continuation stage (as pops_ellipse.py; sfx "_it<k>", none by default).
"""
import json, sys, time

import numpy as np

LEV = 50.0


def leverage(X):
    """h_i = x_i^T (X^T X + I)^-1 x_i in two QR stages (pops_ellipse.py: the normal equations fail at this scaling)."""
    from scipy.linalg import solve_triangular
    n = X.shape[1]
    R = np.linalg.qr(X, mode="r")
    R2 = np.linalg.qr(np.vstack([R, np.eye(n)]), mode="r")
    h = np.empty(X.shape[0])
    for j in range(0, X.shape[0], 32768):
        h[j:j + 32768] = np.sum(solve_triangular(R2, X[j:j + 32768].T, trans="T") ** 2, axis=0)
    assert h.sum() <= n + 1e-6, h.sum()
    return h


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from popsregression import POPSRegression
    from scipy.stats import chi
    from ace_jax import cli
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.objective import linear_prior_diag
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.rows import chunked_rows_fn
    from ..linear import LinearACE
    from ..run import RESULTS
    from .diff_rows import rows_of
    from .force_methods import OOD_DATA, OOD_RUNS
    from .pops_ellipse import factor_of
    args = sys.argv[1:]
    mxit = None
    if "--max-iter" in args:
        i = args.index("--max-iter"); mxit = int(args[i + 1]); del args[i:i + 2]
    m = args[0]
    sfx = "" if mxit is None else f"_it{mxit}"
    t0 = time.time()
    d = OOD_RUNS / "fits" / m
    lo = LinearACE(d / "model.npz", d / "posterior.npz")
    th = np.asarray(lo.theta_hat, np.float64)
    n = len(th)
    fout = d / f"pops_ellipse_sp_lev50{sfx}.npz"
    if fout.exists():
        z = dict(np.load(fout))
        print(f"{m}: ellipsoid from {fout}", flush=True)
    else:
        a = cli._parse(["fit", "--config", str(d / "fit.yaml"), "--out", "/tmp/unused"])
        cfg = cli._fit_config(a)
        data = load_fit_data(cfg, train=a.train, test=a.test)
        prob = build_problem(cfg, data).prob
        theta = Hypers(**json.load(open(d / "theta_map.json")))
        lam_p = np.asarray(linear_prior_diag(theta, prob), np.float64)
        assert lam_p.shape == (n,) and np.all(lam_p > 0), lam_p.shape
        sig = {q: float(np.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
        tr = rows_of(prob, data.ds_train, data.train, None)
        assert tr["E"].shape[1] == n
        X = np.concatenate([tr["E"] * (tr["wE"] / sig["E"])[:, None], tr["F"] * (tr["wF"] / sig["F"])[:, None],
                            tr["V"] * (tr["wV"] / sig["V"])[:, None]])
        y = np.concatenate([tr["yE"] * tr["wE"] / sig["E"], tr["yF"] * tr["wF"] / sig["F"], tr["yV"] * tr["wV"] / sig["V"]])
        del tr
        live = np.any(X != 0, axis=1)
        X, y = X[live], y[live] - X[live] @ th
        print(f"{m} ood refit: train {a.train}; design {X.shape}, {time.time() - t0:.0f} s; weighted residual RMS "
              f"{np.sqrt(np.mean(y ** 2)):.3f}", flush=True)
        ge, gv = np.linalg.eigh(X.T @ X / X.shape[0])
        dinv = 1.0 / np.sqrt(lam_p)
        X *= dinv[None, :]
        h = leverage(X)
        keep_r = h >= np.percentile(h, LEV)
        X, y = X[keep_r], y[keep_r]
        print(f"{m}: leverage cut {LEV:g} %: {int(keep_r.sum())} of {keep_r.size} rows; h sum {h.sum():.1f}", flush=True)
        est = POPSRegression(posterior="ellipsoid", fit_intercept=False, random_state=0,
                             posterior_options={"whiten_ridge": 1.0 / X.shape[0], **({} if mxit is None else {"max_iter": mxit})})
        est.fit(X, y)
        eng = est.ellipsoid_
        P = int(eng._ball_dim)
        Dm = np.diag(dinv)
        info = {"P": P, "N": int(X.shape[0]), "rank": int(eng.U_.shape[1]), "coverage_fraction": float(eng.coverage_fraction_),
                "objective": float(eng.objective_), "n_iter": int(eng.n_iter_), "fit_s": time.time() - t0, "leverage_pct": LEV,
                "delta_rel": float(np.linalg.norm(dinv * np.asarray(est.coef_)) / np.linalg.norm(th)), "max_iter": mxit,
                "stages": getattr(eng, "stage_info_", None)}
        z = dict(Sigma_e=Dm @ (np.asarray(eng.ellipsoid_B_) / (P + 2.0)) @ Dm, G_evals=ge, G_evecs=gv, info=json.dumps(info, default=str))
        del X, y
        np.savez(fout, **z)
        print(f"{m}: ellipsoid fitted: {info}", flush=True)
    Se, info = z["Sigma_e"], json.loads(str(z["info"]))
    Vr = z["G_evecs"][:, z["G_evals"] > 1e-8 * z["G_evals"].max()]
    Sp = Vr @ Vr.T @ Se @ Vr @ Vr.T
    info["span_rank"], info["trace_outside_span"] = int(Vr.shape[1]), float(1 - np.trace(Sp) / np.trace(Se))
    facs = {"cr3": jnp.asarray(lo.factor("cr3")), "ellipsoid": jnp.asarray(factor_of(Se)), "ellipsoid_span": jnp.asarray(factor_of(Sp))}
    eps = float(ARDPosterior.load(d / "posterior.npz").eps)
    q90 = chi(3).ppf(0.9)

    def scores(xyz):
        """Per scored atom: error e (3,), 3x3 covariances per factor, config type (force_methods.atom_scores' atoms)."""
        cfgs = load_configs(str(xyz), virial_key="virial")
        ds = build_dataset(cfgs, {"rcut": lo.rc, "elements": lo.z}, lo.E0_pre, 1, log=lambda *_: None)
        rows = chunked_rows_fn(lo.model, GPConfig(r0=2.4, rcut=lo.rc, n_B=lo.n_B, n_pair=lo.n_pair, NZ=lo.NZ,
                                                  C=int(ds.y_E.shape[1]), e0_cols=lo.e0_cols))
        E, T, C, gc = [], [], {k: [] for k in facs}, 0
        for i in range(ds.n_batches):
            bt = jax.tree.map(lambda x, i=i: x[i], ds)
            F = np.asarray(rows(bt).F)
            a = np.flatnonzero(np.asarray(bt.node_mask) & (np.asarray(bt.w_F) > 0))
            live = np.flatnonzero(np.asarray(bt.cfg_mask))
            idx = {c: gc + j for j, c in enumerate(live)}
            E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
            T += [cfgs[idx[int(c)]].config_type for c in np.asarray(bt.node_cfg)[a]]
            gc += len(live)
            Fa = jnp.asarray(F[a])
            for k, L in facs.items():
                M = Fa[..., :L.shape[0]] @ L
                C[k].append(np.asarray(M @ jnp.swapaxes(M, 1, 2)))
        E = np.concatenate(E)
        C = {k: np.concatenate(v) for k, v in C.items()}
        sd = np.sqrt(np.maximum(np.trace(C["cr3"], axis1=1, axis2=2), 0))
        keep = sd > 1e-3 * np.median(sd)
        return E[keep], {k: v[keep] for k, v in C.items()}, np.array(T, dtype=object)[keep]

    def d2(E, Ca, s):
        Ca = Ca * s ** 2
        reg = Ca + eps * (np.trace(Ca, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
        return np.einsum("na,nab,nb->n", E, np.linalg.inv(reg), E)

    def lam_of(E, Ca):
        return float(np.sqrt(np.mean(np.sum(E ** 2, 1) / np.maximum(np.trace(Ca, axis1=1, axis2=2), 1e-300))))

    Ei, Ci, Ti = scores(OOD_DATA / "test.xyz")
    lam = {k: lam_of(Ei, Ci[k]) for k in ("ellipsoid", "ellipsoid_span")}
    meth = {"ellipsoid": ("ellipsoid", 1.0), "ellipsoid+lam": ("ellipsoid", lam["ellipsoid"]),
            "ellipsoid_span": ("ellipsoid_span", 1.0), "ellipsoid_span+lam": ("ellipsoid_span", lam["ellipsoid_span"])}
    Eo, Co, To = scores(OOD_DATA / "ood.xyz")
    fm = json.load(open(RESULTS / f"force_methods_{m}.json"))["ood"]
    assert len(Eo) == fm["all"]["atoms"], (len(Eo), fm["all"]["atoms"])

    def summ(E, C, T, mask=None):
        mask = np.ones(len(E), bool) if mask is None else mask
        r = {"atoms": int(mask.sum())}
        for k, (f, s) in meth.items():
            dd = d2(E[mask], C[f][mask], s)
            r[k] = {"coverage": float(np.mean(dd <= q90 ** 2)), "rms_d_over_sqrt3": float(np.sqrt(np.mean(dd) / 3))}
        return r
    out = {"model": m, "info": info, "lam_id_test": lam,
           "id_test": summ(Ei, Ci, Ti), "ood": summ(Eo, Co, To),
           "ood_per_type": {t: summ(Eo, Co, To, To == t) for t in np.unique(To.astype(str))},
           "paper_methods_ood": {"all": {k: fm["all"][k]["coverage"] for k in ("headline", "B1", "B2", "B3")},
                                 "per_type": {t: {k: v[k]["coverage"] for k in ("headline", "B1", "B2", "B3")} for t, v in fm["per_type"].items()},
                                 "id_test": {k: fm["id_test"][k]["coverage"] for k in ("headline", "B1", "B2", "B3")}}}
    print(f"{m}: lam on the refit's test split {lam}; trace outside span {info['trace_outside_span']:.3f}", flush=True)
    for st in ("id_test", "ood"):
        print(f"  {st:8s} ellipsoid " + "  ".join(f"{k} {out[st][k]['coverage']:.3f}" for k in meth)
              + " | paper " + "  ".join(f"{k} {out['paper_methods_ood'][('all' if st == 'ood' else 'id_test')][k]:.3f}" for k in ("headline", "B1", "B2")), flush=True)
    for t, v in out["ood_per_type"].items():
        p = out["paper_methods_ood"]["per_type"].get(t, {})
        print(f"  ood {t:24s} n {v['atoms']:5d}  " + "  ".join(f"{k} {v[k]['coverage']:.2f}" for k in meth)
              + " | " + "  ".join(f"{k} {p.get(k, float('nan')):.2f}" for k in ("headline", "B1", "B2")), flush=True)
    json.dump(out, open(RESULTS / f"pops_ellipse_ood_{m}{sfx}.json", "w"), indent=1, default=float)
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
