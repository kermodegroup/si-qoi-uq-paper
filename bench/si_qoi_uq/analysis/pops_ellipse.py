"""The POPS ellipsoid (Swinburne et al. 2026) on the headline fit's rows (next-steps plan §2): per-atom force coverage and QoIs.

popsregression branch `ellipse-ev` (BSD-3; a checkout on PYTHONPATH, imported as `popsregression`, which does not clash with
the installed 0.4.0 module `POPSRegression`): `POPSRegression(posterior="ellipsoid")` fits a uniform ellipsoid
B = B0 + U U^T (whitened coordinates; B0 the package's own POPS hypercube covariance from a pre-fit, U of rank 32) by
minimising the projected-ball NLL with a log-barrier for the POPS covering condition, package defaults throughout.
Rows: every training E, F and V row of the cr3 fit, weighted as the fit weights them (structural weight / sigma_q at the
MAP hyperparameters), targets the weighted residuals about theta_hat, so the package's centre is a correction delta to
the headline mean (reported; the evaluation uses theta_hat, and the size of delta's effect is printed).  Read as a
Gaussian: Sigma = B / (P + 2), the covariance of the uniform P-ball's pushforward (plan §2.2).  Compared, on the same
footing (Gaussian, centre theta_hat):
  ellipsoid      Sigma_e = B / (P + 2)                       (§2.2: no scale)
  ellipsoid+lam  lam_e^2 Sigma_e, lam_e the test-split per-atom RMS |e| / sd (§2.3: one scale)
  ellipsoid_span (+lam)  Sigma_e projected onto the data span: the eigenvectors of G = X^T X / N above the package's own
                 mode_threshold x the largest.  Outside it nothing constrains U, which keeps its random start times the
                 whitening floor 1 / sqrt(threshold) (o3d12: 1e20-scale variances there), so the bare ellipsoid is not
                 numerically defined in those directions, as the hypercube without a ridge is not
  hypercube0     the package's own hypercube baseline B0 (back in model coordinates), its covariance
  headline       CR3 x lam_F; B2 (the paper's ace-jax POPS box, leverage cut 50 %) from results/force_baselines_<m>.npz
Scores: per-atom 90 % Gaussian-ellipsoid coverage on the test split (force_baselines.py's Mahalanobis rule, same eps), RMS z
of test force components, and the delta-method QoIs (17 with DFT; z, 95 % coverage), each with that factor.

    SIQ_RUNS=<BLR root> PYTHONPATH=<popsregression checkout> python -m bench.si_qoi_uq.analysis.pops_ellipse o3d12|o4d20
        [--whiten-ridge-rel r]   (a sweep; r is not selected on anything: every value is reported)
        [--smooth-prior]   the fit's own prior (linear_prior_diag at the MAP: smoothness prior x sigma_c x ARD groups) as the
                 width prior: columns rescaled X Lambda^-1/2 and whiten_ridge 1/N, so the package's G is (X^T X + Lambda) / N
                 in model coordinates, the BLR posterior precision behind CR3 (no free knob); results mapped back by Lambda^-1/2
        [--param-prior]   (with --smooth-prior) the same prior inside the ellipsoid objective: popsregression branch
                 ellipse-param-prior's param_prior_precision = I in those coordinates (the KL from N(0, Lambda^-1) to
                 N(mu, B / (P + 2)), unit weight), at rank P so that B can be full rank
        [--leverage-pct p]   (with --smooth-prior) fit only the rows whose posterior leverage h_i = x_i^T (X^T X + Lambda)^-1 x_i
                 is at or above the p-th percentile, as the paper's B2 box (the reference package's default, p = 50)
        -> RUNS/fits/<m>/pops_ellipse.npz, results/pops_ellipse_<m>.json
"""
import csv, json, pathlib, shlex, sys, time

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")
Z95 = 1.959963984540054


def factor_of(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V * np.sqrt(np.clip(w, 0, None))[None, :]


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
    from ..diagnostics import SYM_ZERO, TEST_XYZ
    from ..harness import RUNS
    from ..run import RESULTS, fit_dir, lin_for
    from .diff_rows import rows_of
    from .force_baselines import ARD_RUNS
    args = sys.argv[1:]
    wr_rel = None                      # --whiten-ridge-rel r: the package's whiten_ridge set to r x the largest eigenvalue of G
    if "--whiten-ridge-rel" in args:
        i = args.index("--whiten-ridge-rel"); wr_rel = float(args[i + 1]); del args[i:i + 2]
    pac = "--pac-bayes" in args        # the PAC-Bayes layer (package defaults): a Gaussian hyperprior ridge on the ellipsoid's
    if pac:                            # parameters, which also bounds U in directions no training row touches
        args.remove("--pac-bayes")
    sp = "--smooth-prior" in args
    if sp:
        args.remove("--smooth-prior")
    lev = None
    if "--leverage-pct" in args:
        i = args.index("--leverage-pct"); lev = float(args[i + 1]); del args[i:i + 2]
    kl = "--param-prior" in args
    if kl:
        args.remove("--param-prior")
        if not sp or pac:
            raise SystemExit("--param-prior needs --smooth-prior and excludes --pac-bayes")
    if lev is not None and not sp:
        raise SystemExit("--leverage-pct needs --smooth-prior (the leverage is the posterior's)")
    if sp and wr_rel is not None:
        raise SystemExit("--smooth-prior fixes whiten_ridge at 1/N; do not combine it with --whiten-ridge-rel")
    m = args[0]
    sfx = ("" if wr_rel is None else f"_wr{wr_rel:g}") + ("_sp" if sp else "") + ("_kl" if kl else "") + ("" if lev is None else f"_lev{lev:g}") + ("_pac" if pac else "")
    t0 = time.time()
    lin = lin_for(m, "cr3")
    th = np.asarray(lin.theta_hat, np.float64)
    n = len(th)
    fout = fit_dir(m, "cr3").parent / f"pops_ellipse{sfx}.npz"

    def design():
        import os
        cache = os.environ.get("POPS_DESIGN_CACHE")              # node-local scratch (e.g. /tmp on mnf148): the o4d20 design
        if cache and pathlib.Path(cache, f"{m}_X.npy").exists():  # takes ~30 min to build; never on /storage (quota)
            X, y = np.load(pathlib.Path(cache, f"{m}_X.npy"), mmap_mode="r"), np.load(pathlib.Path(cache, f"{m}_y.npy"))
            print(f"{m}: design from {cache}: {X.shape}", flush=True)
            return np.ascontiguousarray(X), y, np.load(pathlib.Path(cache, f"{m}_lam.npy"))
        argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
        argv[argv.index("--out") + 1] = "/tmp/unused"
        a = cli._parse(argv)
        cfg = cli._fit_config(a)
        data = load_fit_data(cfg, train=a.train, test=a.test)
        prob = build_problem(cfg, data).prob
        theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
        lam_p = np.asarray(linear_prior_diag(theta, prob), np.float64)  # prior precision diag at the MAP
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
        print(f"{m}: design {X.shape} ({X.nbytes / 2 ** 30:.1f} GiB), {time.time() - t0:.0f} s; weighted residual RMS "
              f"{np.sqrt(np.mean(y ** 2)):.3f}", flush=True)
        if cache:
            pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
            np.save(pathlib.Path(cache, f"{m}_X.npy"), X); np.save(pathlib.Path(cache, f"{m}_y.npy"), y)
            np.save(pathlib.Path(cache, f"{m}_lam.npy"), lam_p)
        return X, y, lam_p

    if fout.exists():
        z = dict(np.load(fout))
        print(f"{m}: ellipsoid from {fout}", flush=True)
        if "G_evals" not in z:                                     # the data span, for the projected ellipsoid
            X, _, _ = design()
            z["G_evals"], z["G_evecs"] = np.linalg.eigh(X.T @ X / X.shape[0])
            del X
            np.savez(fout, **z)
    else:
        X, y, lam_p = design()
        ge, gv = np.linalg.eigh(X.T @ X / X.shape[0])              # model coordinates (the data span)
        opts = None if wr_rel is None else {"whiten_ridge": wr_rel * float(ge.max())}
        dinv = np.ones(n)
        if sp:                                                    # beta = Lambda^1/2 theta: the prior is the identity there
            dinv = 1.0 / np.sqrt(lam_p)
            X *= dinv[None, :]
            opts = {"whiten_ridge": 1.0 / X.shape[0]}
            if kl:
                opts.update(param_prior_precision=np.ones(n), rank=n)
            if lev is not None:                                   # posterior leverage in the prior coordinates
                # h_i = ||Q_i||^2 from the QR of [X; I]: backward stable at kappa(X). Normal equations fail here (columns
                # scaled by up to 1e4): Cholesky of X^T X + I is indefinite, and its eigendecomposition gave sum h = 1274 > P
                # in two stages, so no (N + P) x P Q is formed (o4d20: 20 GB): R of X, then R2 of [R; I], and
                # h_i = ||R2^-T x_i||^2 by chunked triangular solves (backward stable, as the one-shot QR)
                from scipy.linalg import solve_triangular
                R = np.linalg.qr(X, mode="r")
                R2 = np.linalg.qr(np.vstack([R, np.eye(n)]), mode="r")
                del R
                h = np.empty(X.shape[0])
                for j in range(0, X.shape[0], 32768):
                    h[j:j + 32768] = np.sum(solve_triangular(R2, X[j:j + 32768].T, trans="T") ** 2, axis=0)
                assert h.sum() <= n + 1e-6, h.sum()
                keep_r = h >= np.percentile(h, lev)
                X, y = X[keep_r], y[keep_r]
                opts["whiten_ridge"] = 1.0 / X.shape[0]
                print(f"{m}: leverage cut {lev:g} %: {int(keep_r.sum())} of {keep_r.size} rows kept; h median "
                      f"{np.median(h):.3g}, sum {h.sum():.1f}", flush=True)
            print(f"{m}: smooth prior: Lambda in [{lam_p.min():.3g}, {lam_p.max():.3g}], whiten_ridge 1/N", flush=True)
        if X.size >= 2 ** 31:           # SciPy's LP64 LAPACK refuses > 2^31 elements (o4d20: 2.6e9); sklearn's BayesianRidge
            import types as _t           # pre-fit calls scipy.linalg.svd on X: route that one call to NumPy's ILP64 SVD
            import scipy.linalg as _sl
            import sklearn.linear_model._bayes as _bayes
            ns = {k: getattr(_sl, k) for k in dir(_sl) if not k.startswith("_")}
            ns["svd"] = lambda A, full_matrices=False, **_: np.linalg.svd(A, full_matrices=full_matrices)
            _bayes.linalg = _t.SimpleNamespace(**ns)
        est = POPSRegression(posterior="ellipsoid", fit_intercept=False, random_state=0, posterior_options=opts, pac_bayes=pac)
        est.fit(X, y)
        eng = est.ellipsoid_                                                        # the fitted _EllipsoidPosterior
        P = int(eng._ball_dim)
        delta = dinv * np.asarray(est.coef_, np.float64)
        Dm = np.diag(dinv)
        info = {"P": int(P), "N": int(X.shape[0]), "rank": int(eng.U_.shape[1]), "coverage_fraction": float(eng.coverage_fraction_),
                "objective": float(eng.objective_), "n_iter": int(eng.n_iter_), "fit_s": time.time() - t0,
                "delta_rel": float(np.linalg.norm(delta) / np.linalg.norm(th)), "mode_threshold": float(eng.mode_threshold),
                "whiten_ridge": float(eng.whiten_ridge), "whiten_ridge_rel": wr_rel, "pac_bayes": pac, "smooth_prior": sp, "param_prior": kl, "leverage_pct": lev,
                "param_prior_kl": (float(eng.param_prior_kl_) if kl else None),
                **({"tau2": float(eng.tau2_), "bound": float(eng.bound_), "kl": float(eng.kl_)} if pac else {}), "G_eig_max": float(ge.max())}
        z = dict(Sigma_e=Dm @ (np.asarray(eng.ellipsoid_B_) / (P + 2.0)) @ Dm,    # as POPSRegression's own sigma; beta -> theta
                 B0=Dm @ eng._whiten_W @ np.asarray(eng.baseline_B0_) @ eng._whiten_W @ Dm,    # whitened -> model coordinates
                 delta=delta, info=json.dumps(info), G_evals=ge, G_evecs=gv)
        del X, y
        np.savez(fout, **z)
        print(f"{m}: ellipsoid fitted: {info}", flush=True)
    Se, B0, delta, info = z["Sigma_e"], z["B0"], z["delta"], json.loads(str(z["info"]))
    # the data span: the modes the package keeps in its whitening (eigenvalues of G above mode_threshold x the largest);
    # outside it the data never constrain the ellipsoid, whose U there stays at its random start times 1/sqrt(floor)
    mt = info.get("mode_threshold", 1e-8)
    Vr = z["G_evecs"][:, z["G_evals"] > mt * z["G_evals"].max()]
    Pr = Vr @ Vr.T
    Sp = Pr @ Se @ Pr
    info["span_rank"] = int(Vr.shape[1])
    info["trace_outside_span"] = float(1 - np.trace(Sp) / np.trace(Se))
    print(f"{m}: data span rank {Vr.shape[1]} of {n}; share of the ellipsoid's trace outside it {info['trace_outside_span']:.6f}", flush=True)
    facs = {"ellipsoid": factor_of(Se), "ellipsoid_span": factor_of(Sp), "hypercube0": factor_of(B0), "cr3": np.asarray(lin.factor("cr3"))}
    # test split: per-atom force coverage, force RMS z, and delta's effect on the forces
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
    lamF = float(sc["lam_cr3"]["F"])
    ard = ARDPosterior.load(ARD_RUNS / "fits" / m / "cr3" / "posterior.npz")
    eps = float(ard.eps)
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    E, Ed, C, T, gc = [], [], {k: [] for k in facs}, [], 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda x, i=i: x[i], ds)
        F = np.asarray(rows(bt).F)
        a = np.flatnonzero(np.asarray(bt.node_mask) & (np.asarray(bt.w_F) > 0))
        live = np.flatnonzero(np.asarray(bt.cfg_mask))                # config types, as force_baselines.py
        idx = {c: gc + j for j, c in enumerate(live)}
        T += [cfgs[idx[int(c)]].config_type for c in np.asarray(bt.node_cfg)[a]]
        gc += len(live)
        E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
        Ed.append(np.asarray(bt.y_F)[a] - F[a] @ (th + delta))
        Fa = jnp.asarray(F[a])
        for k, Lf in facs.items():
            M = Fa @ jnp.asarray(Lf)
            C[k].append(np.asarray(M @ jnp.swapaxes(M, 1, 2)))
    E, Ed = np.concatenate(E), np.concatenate(Ed)
    C = {k: np.concatenate(v) for k, v in C.items()}
    sd_cr3 = np.sqrt(np.trace(C["cr3"], axis1=1, axis2=2))
    keep = sd_cr3 > SYM_ZERO * np.median(sd_cr3)
    lam = {k: float(np.sqrt(np.mean(np.sum(E[keep] ** 2, 1) / np.maximum(np.trace(C[k][keep], axis1=1, axis2=2), 1e-300)))) for k in facs}
    q90 = chi(3).ppf(0.9)

    T = np.array(T, dtype=object)[keep]

    def dist(Ca, s):
        Ca = Ca[keep] * s ** 2
        tr_ = np.trace(Ca, axis1=1, axis2=2)
        reg = Ca + eps * (tr_ / 3)[:, None, None] * np.eye(3)
        return np.sqrt(np.einsum("na,nab,nb->n", E[keep], np.linalg.inv(reg), E[keep]))

    def cover(Ca, s):
        return float(np.mean(dist(Ca, s) <= q90))
    meth = {"ellipsoid": ("ellipsoid", 1.0), "ellipsoid+lam": ("ellipsoid", lam["ellipsoid"]),
            "ellipsoid_span": ("ellipsoid_span", 1.0), "ellipsoid_span+lam": ("ellipsoid_span", lam["ellipsoid_span"]),
            "hypercube0": ("hypercube0", 1.0), "headline": ("cr3", lamF)}
    fb = np.load(RESULTS / f"force_baselines_{m}.npz", allow_pickle=True)
    out = {"model": m, "info": info, "lam_per_atom_rms": lam, "lam_F_headline": lamF,
           "test_force_rmse": {"theta_hat": float(np.sqrt(np.mean(E ** 2))), "theta_hat+delta": float(np.sqrt(np.mean(Ed ** 2)))},
           "force_cov90": {k: cover(C[f], s) for k, (f, s) in meth.items()},
           "force_cov90_B2_paper": float(np.mean(fb["d_B2"] <= q90)), "force_cov90_headline_paper": float(np.mean(fb["d_headline"] <= q90))}
    # per configuration type (types with >= 300 test atoms, as the paper's Figure 2 / fb_types), with the paper's B2 and
    # headline from force_baselines_<m>.npz (same atoms: the same symmetry-zero rule)
    assert len(fb["d_B2"]) == len(T) and np.array_equal(fb["config_type"].astype(str), T.astype(str)), (len(fb["d_B2"]), len(T))
    tys = [u for u in np.unique(T.astype(str)) if (T == u).sum() >= 300]
    hits = {k: dist(C[f], s) <= q90 for k, (f, s) in meth.items()}
    hits.update(B2_paper=fb["d_B2"] <= q90, headline_paper=fb["d_headline"] <= q90)
    pt = {k: {u: float(h[T == u].mean()) for u in tys} for k, h in hits.items()}
    out["force_cov90_per_type"] = {"atoms": {u: int((T == u).sum()) for u in tys}, "coverage": pt,
                                   "min": {k: [min(v, key=v.get), min(v.values())] for k, v in pt.items()},
                                   "rms_dev_from_0.9": {k: float(np.sqrt(np.mean((np.array(list(v.values())) - 0.9) ** 2))) for k, v in pt.items()}}
    for k in pt:
        print(f"  per-type {k:20s}: min {out['force_cov90_per_type']['min'][k][1]:.2f} ({out['force_cov90_per_type']['min'][k][0]}), "
              f"RMS dev {out['force_cov90_per_type']['rms_dev_from_0.9'][k]:.3f}", flush=True)
    print(f"{m}: test F RMSE theta_hat {out['test_force_rmse']['theta_hat']:.4f}, +delta {out['test_force_rmse']['theta_hat+delta']:.4f}; "
          f"per-atom RMS |e|/sd (lam = 1) {lam}", flush=True)
    print(f"  per-atom 90 % coverage {out['force_cov90']}; paper B2 {out['force_cov90_B2_paper']:.3f}, headline "
          f"{out['force_cov90_headline_paper']:.3f}", flush=True)
    # QoIs, delta method
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    G = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
    csvr = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "qoi_uq.csv"))
            if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
    from ..propagate import tol_abs
    qo = {}
    scal = {"ellipsoid": ("ellipsoid", 1.0), "ellipsoid+lam": ("ellipsoid", lam["ellipsoid"]),
            "ellipsoid_span": ("ellipsoid_span", 1.0), "ellipsoid_span+lam": ("ellipsoid_span", lam["ellipsoid_span"]), "hypercube0": ("hypercube0", 1.0),
            "headline": ("cr3", lamF)}
    for q, g in G.items():
        if q.startswith("energy:") or q not in csvr or csvr[q]["dft"] in ("", "nan"):
            continue
        e = float(csvr[q]["dft"]) - float(csvr[q]["mean"])
        tol = tol_abs(q, float(csvr[q]["dft"]))
        qo[q] = {}
        for k, (f, s) in scal.items():
            sg = s * float(np.linalg.norm(facs[f].T @ g[:n]))
            qo[q][k] = {"sigma": sg, "sigma_over_tol": sg / tol, "z": e / sg, "covered95": bool(abs(e / sg) <= Z95)}
    out["qoi"] = qo
    out["qoi_summary"] = {k: {"n": len(qo), "k95": sum(v[k]["covered95"] for v in qo.values()),
                              "median_sigma_over_tol": float(np.median([v[k]["sigma_over_tol"] for v in qo.values()]))}
                          for k in scal}
    for k, v in out["qoi_summary"].items():
        print(f"  QoIs {k:14s}: 95 % {v['k95']}/{v['n']}, median sigma/tol {v['median_sigma_over_tol']:.2f}", flush=True)
    for q in ("dumbbell_interstitial_energy", "tetrahedral_interstitial_energy", "vacancy_energy", "surface_100 (J/m2)"):
        if q in qo:
            print(f"    {q:32s} " + "  ".join(f"{k} z {qo[q][k]['z']:+.2f}" for k in scal), flush=True)
    json.dump(out, open(RESULTS / f"pops_ellipse_{m}{sfx}.json", "w"), indent=1, default=float)
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
