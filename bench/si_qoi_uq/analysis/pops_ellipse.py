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
    m = args[0]
    sfx = ("" if wr_rel is None else f"_wr{wr_rel:g}") + ("_pac" if pac else "")
    t0 = time.time()
    lin = lin_for(m, "cr3")
    th = np.asarray(lin.theta_hat, np.float64)
    n = len(th)
    fout = fit_dir(m, "cr3").parent / f"pops_ellipse{sfx}.npz"

    def design():
        argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
        argv[argv.index("--out") + 1] = "/tmp/unused"
        a = cli._parse(argv)
        cfg = cli._fit_config(a)
        data = load_fit_data(cfg, train=a.train, test=a.test)
        prob = build_problem(cfg, data).prob
        theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
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
        return X, y

    if fout.exists():
        z = dict(np.load(fout))
        print(f"{m}: ellipsoid from {fout}", flush=True)
        if "G_evals" not in z:                                     # the data span, for the projected ellipsoid
            X, _ = design()
            z["G_evals"], z["G_evecs"] = np.linalg.eigh(X.T @ X / X.shape[0])
            del X
            np.savez(fout, **z)
    else:
        X, y = design()
        ge, gv = np.linalg.eigh(X.T @ X / X.shape[0])
        opts = None if wr_rel is None else {"whiten_ridge": wr_rel * float(ge.max())}
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
        delta = np.asarray(est.coef_, np.float64)
        info = {"P": int(P), "N": int(X.shape[0]), "rank": int(eng.U_.shape[1]), "coverage_fraction": float(eng.coverage_fraction_),
                "objective": float(eng.objective_), "n_iter": int(eng.n_iter_), "fit_s": time.time() - t0,
                "delta_rel": float(np.linalg.norm(delta) / np.linalg.norm(th)), "mode_threshold": float(eng.mode_threshold),
                "whiten_ridge": float(eng.whiten_ridge), "whiten_ridge_rel": wr_rel, "pac_bayes": pac,
                **({"tau2": float(eng.tau2_), "bound": float(eng.bound_), "kl": float(eng.kl_)} if pac else {}), "G_eig_max": float(ge.max())}
        z = dict(Sigma_e=np.asarray(eng.ellipsoid_B_) / (P + 2.0),                # as POPSRegression's own sigma
                 B0=eng._whiten_W @ np.asarray(eng.baseline_B0_) @ eng._whiten_W,    # whitened -> model coordinates
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
    E, Ed, C = [], [], {k: [] for k in facs}
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda x, i=i: x[i], ds)
        F = np.asarray(rows(bt).F)
        a = np.flatnonzero(np.asarray(bt.node_mask) & (np.asarray(bt.w_F) > 0))
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

    def cover(Ca, s):
        Ca = Ca[keep] * s ** 2
        tr_ = np.trace(Ca, axis1=1, axis2=2)
        reg = Ca + eps * (tr_ / 3)[:, None, None] * np.eye(3)
        d = np.sqrt(np.einsum("na,nab,nb->n", E[keep], np.linalg.inv(reg), E[keep]))
        return float(np.mean(d <= q90))
    meth = {"ellipsoid": ("ellipsoid", 1.0), "ellipsoid+lam": ("ellipsoid", lam["ellipsoid"]),
            "ellipsoid_span": ("ellipsoid_span", 1.0), "ellipsoid_span+lam": ("ellipsoid_span", lam["ellipsoid_span"]),
            "hypercube0": ("hypercube0", 1.0), "headline": ("cr3", lamF)}
    fb = np.load(RESULTS / f"force_baselines_{m}.npz", allow_pickle=True)
    out = {"model": m, "info": info, "lam_per_atom_rms": lam, "lam_F_headline": lamF,
           "test_force_rmse": {"theta_hat": float(np.sqrt(np.mean(E ** 2))), "theta_hat+delta": float(np.sqrt(np.mean(Ed ** 2)))},
           "force_cov90": {k: cover(C[f], s) for k, (f, s) in meth.items()},
           "force_cov90_B2_paper": float(np.mean(fb["d_B2"] <= q90)), "force_cov90_headline_paper": float(np.mean(fb["d_headline"] <= q90))}
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
