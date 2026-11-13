"""CR3 and the POPS ellipsoid for step 3's candidates (plan section 5 D), so final.py can score them beside B1.

For a setting (final.py: "in" = all of T, "ood" = T without cracks and screw dislocations) and a candidate (hand, Ex5, Ex10):
the problem is the o4d16 benchmark command on that training set with the candidate's weights file, at the candidate's MAP
(sigma_c, shared sigma) from results/final_<setting>.json; the posterior from its QR statistics (as final.py, so the mean is
the B1 mean).
  cr3        the whole-configuration PRESS jackknife (fit.jackknife.press_scores, exact) and shape_factor: CR3 = D^-1 R
             (blr_cr3.py, weight_cr3.py)
  ellipsoid  the leverage-cut POPS ellipsoid (pops_ellipse_ood.py's recipe: every weighted E, F, V training row, targets the
             weighted residuals about the mean, columns rescaled by Lambda^-1/2 at the MAP, rows at or above the 50th
             percentile of leverage, popsregression `posterior="ellipsoid"`, whiten_ridge 1/N, 5000 L-BFGS iterations per
             continuation stage as o4d20's held-out run); Sigma_e = D B D / (P + 2)
Nothing is chosen on the calibration or held-out files.

    PYTHONPATH=<popsregression checkout>:<this worktree> python -m bench.si_qoi_uq.analysis.final_shapes ood Ex10 cr3|ellipsoid
        -> RUNS/final/<setting>/<shape>_<candidate>.npz
"""
import json, sys, time

import numpy as np

from .final import MODEL, OOD, TAG, out_dir_of
from .weight_sweep import DATA, base_cmd, candidates, frames, resolve

MAX_ITER = 5000


def main(setting, name, shape):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax import cli
    from ace_jax.fit.ard import ARDPosterior, ard_gamma, body_order_columns, E0_GROUP
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.objective import linear_prior_diag, posterior_from_qr
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.stats import linear_qr_statistics
    from ..harness import RUNS
    from ..run import RESULTS
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.0f} s] {setting} {name} {shape}:", *a, flush=True)
    d = out_dir_of(RUNS, setting)
    d.mkdir(parents=True, exist_ok=True)
    train = f"{DATA}/train.xyz" if setting == "in" else f"{OOD}/train.xyz"
    types = sorted(set(frames(train)[1]))
    w, _ = candidates(types, "all")[name]
    hand = json.load(open(f"{DATA}/weights_acejax.json"))
    wf = d / f"weights_{name}.json"
    json.dump({"default": dict(resolve(hand, "__none__")), **w}, open(wf, "w"), indent=1)
    argv, th_path = base_cmd(MODEL)
    for k, v in (("--train", train), ("--weights", wf), ("--out", d / f"fit_{name}")):
        argv[argv.index(k) + 1] = str(v)
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=None, log=lambda *_: None)
    prob = build_problem(cfg, data).prob
    th = Hypers(**{**json.load(open(th_path)), **json.load(open(RESULTS / f"final_{setting}{TAG}.json"))["candidates"][name]["theta"]})
    qs = linear_qr_statistics(prob.model, prob.cfg, data.ds_train)
    mu, L = posterior_from_qr(th, qs, prob)
    mu, L = np.asarray(mu), np.asarray(L)
    del qs
    b1 = np.load(d / f"posterior_{name}.npz")                 # final.py's merged-QR mean: the same problem, exactly
    dev = float(np.max(np.abs(mu - b1["mu"])) / np.max(np.abs(b1["mu"])))
    log(f"QR statistics; mean against final.py's {dev:.1e} (relative)")
    gamma = np.asarray(ard_gamma(prob))
    dinv = 1.0 / gamma
    sig = np.exp([float(getattr(th, f"log_sigma_{q}")) for q in "EFV"])
    info = {"setting": setting, "candidate": name, "train": train, "theta": {k: float(x) for k, x in th._asdict().items() if "sigma" in k},
            "mean_rel_dev_vs_final": dev}
    if shape == "cr3":
        from ace_jax.fit.jackknife import press_scores, shape_factor
        body_col = body_order_columns(data.meta, prob.cfg)
        groups = tuple(int(g) for g in np.unique(body_col[body_col != E0_GROUP]))
        post = ARDPosterior(mu, dinv[:, None] * L, dinv, 1.0, np.full(len(groups), float(-2 * th.log_sigma_c)), groups, body_col, {})
        rc, K = row_clusters(data.ds_train, data.train, np.inf)
        G, lev = press_scores(post, prob, data.ds_train, rc, K, sig, mode="exact")
        R = np.asarray(shape_factor(post, G, 1.0))
        info.update(K=int(K), rank=int(R.shape[1]), lev_max=float(np.max(lev)))
        np.savez(d / f"cr3_{name}.npz", mu=mu, F=dinv[:, None] * R, lev=lev, info=json.dumps(info))
        log(f"PRESS K {K}, rank R {R.shape[1]}, max leverage {float(np.max(lev)):.4f}")
        return
    from popsregression import POPSRegression
    from .diff_rows import rows_of
    from .pops_ellipse import factor_of
    from .pops_ellipse_ood import LEV, leverage
    lam_p = np.asarray(linear_prior_diag(th, prob), np.float64)
    tr = rows_of(prob, data.ds_train, data.train, None)
    X = np.concatenate([tr["E"] * (tr["wE"] / sig[0])[:, None], tr["F"] * (tr["wF"] / sig[1])[:, None], tr["V"] * (tr["wV"] / sig[2])[:, None]])
    y = np.concatenate([tr["yE"] * tr["wE"] / sig[0], tr["yF"] * tr["wF"] / sig[1], tr["yV"] * tr["wV"] / sig[2]])
    del tr
    live = np.any(X != 0, axis=1)
    X, y = X[live], y[live] - X[live] @ mu
    log(f"design {X.shape}; weighted residual RMS {np.sqrt(np.mean(y ** 2)):.3f}")
    dp = 1.0 / np.sqrt(lam_p)
    X *= dp[None, :]
    h = leverage(X)
    keep = h >= np.percentile(h, LEV)
    X, y = X[keep], y[keep]
    log(f"leverage cut {LEV:g} %: {int(keep.sum())} rows")
    est = POPSRegression(posterior="ellipsoid", fit_intercept=False, random_state=0,
                         posterior_options={"whiten_ridge": 1.0 / X.shape[0], "max_iter": MAX_ITER})
    est.fit(X, y)
    eng = est.ellipsoid_
    P = int(eng._ball_dim)
    Se = np.diag(dp) @ (np.asarray(eng.ellipsoid_B_) / (P + 2.0)) @ np.diag(dp)
    info.update(P=P, N=int(X.shape[0]), rank=int(eng.U_.shape[1]), coverage_fraction=float(eng.coverage_fraction_),
                objective=float(eng.objective_), n_iter=int(eng.n_iter_), leverage_pct=LEV, max_iter=MAX_ITER,
                delta_rel=float(np.linalg.norm(dp * np.asarray(est.coef_)) / np.linalg.norm(mu)),
                stages=getattr(eng, "stage_info_", None), fit_s=time.time() - t0)
    np.savez(d / f"ellipsoid_{name}.npz", mu=mu, F=factor_of(Se), Sigma_e=Se, info=json.dumps(info, default=str))
    log(f"ellipsoid fitted: {json.dumps(info, default=str)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
