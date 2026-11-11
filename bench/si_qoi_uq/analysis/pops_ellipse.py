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


def fixed_point(U, r, B, iters, rhos=(1e-1, 1e-2, 1e-3, 1e-4), delta=1e-3, tol=1e-7, jitter=1e-4, P_ball=None, verbose=True):
    """Reweighting iteration for the POPS ellipsoid objective (popsregression's, no baseline) in coordinates U (rows u_i).

    l_i = 0.5 log v_i - log C_P - k L_rho(q_i), v_i = u_i^T B u_i + delta^2, q_i = 1 - r_i^2 / v_i, k = (P - 1) / 2.
    Stationarity in B: B N B = C with N = mean 0.5 u u^T / v (data side) and C = B [mean k L'_rho(q) r^2 / v^2 u u^T] B
    (k x the weighted second moment of the corrections B u_i r_i / v_i). With N, C held at the current B this is solved by
    the matrix geometric mean B = N^-1/2 (N^1/2 C N^1/2)^1/2 N^-1/2 (on N's range). A step that raises the objective is
    halved along the geodesic B^1/2 (B^-1/2 B' B^-1/2)^a B^1/2. Continuation over rho as the package's rho_schedule.
    The optimum is rank-deficient (on the PSD boundary), and a multiplicative update cannot regrow a direction it has
    collapsed (B' lies in range(B)): each proposal gets jitter x tr(B)/P x I back, so a direction the gradient favours can
    return; the jitter is dropped for a final polish at each rho (accepted only if it lowers the objective).
    """
    from popsregression._projected_ball import log_norm_constant, smooth_log
    P = U.shape[1]                     # B's dimension; P_ball (default P) sets k and C_P (the low-rank inner fit: P_ball > P)
    Pb = P if P_ball is None else P_ball
    k = 0.5 * (Pb - 1)
    r2 = r * r

    def psd_pow(S, *a, cut=1e-14):     # S^a for each a, from one eigendecomposition (P = 5477 at o4d20: one eigh per call)
        w, V = np.linalg.eigh(0.5 * (S + S.T))
        keep = w > cut * max(w.max(), 1e-300)
        out = [(V[:, keep] * w[keep] ** e) @ V[:, keep].T for e in a]
        return out[0] if len(out) == 1 else out

    def stats(B, rho):
        v = np.einsum("ij,ij->i", U @ B, U) + delta * delta
        q = 1.0 - r2 / v
        lq, dlq, _ = smooth_log(q, rho)
        obj = float(np.mean(0.5 * np.log(v) - log_norm_constant(Pb) - k * lq))
        return obj, v, q, dlq

    log = []
    for rho in rhos:
        obj, v, q, dlq = stats(B, rho)
        it = -1
        for it in range(iters):
            Nm = (U * (0.5 / v)[:, None]).T @ U / len(r)
            Mz = (U * (k * dlq * r2 / v ** 2)[:, None]).T @ U / len(r)
            C = B @ Mz @ B
            Nh, Nmh = psd_pow(Nm, 0.5, -0.5)
            Bn = Nmh @ psd_pow(Nh @ C @ Nh, 0.5) @ Nmh
            if jitter and it < iters - 1:
                Bn = Bn + jitter * np.trace(Bn) / P * np.eye(P)
            a, Bhs = 1.0, None
            while True:
                if a < 1.0 and Bhs is None:
                    Bhs = psd_pow(B, 0.5, -0.5)
                Bt = Bn if a == 1.0 else Bhs[0] @ psd_pow(Bhs[1] @ Bn @ Bhs[1], a) @ Bhs[0]
                o2, v2, q2, d2 = stats(Bt, rho)
                if o2 <= obj or a < 1e-3:
                    break
                a *= 0.5
            if o2 > obj:               # no descent along the geodesic: converged at this rho (keep the current B)
                break
            rel = abs(obj - o2) / max(abs(obj), 1e-300)
            B, obj, v, q, dlq = 0.5 * (Bt + Bt.T), o2, v2, q2, d2
            if verbose and (it % 10 == 0 or rel < tol or P > 1000):
                print(f"  fixed point rho {rho:g} it {it} obj {obj:.6f} step {a:g} covered {np.mean(q > 0):.4f} "
                      f"min q {q.min():.3g}", flush=True)
            if rel < tol:
                break
        log.append({"rho": rho, "iters": it + 1, "obj": obj, "covered": float(np.mean(q > 0)), "min_q": float(q.min())})
    return B, log


def inner_lbfgs(W, r, C, P, rho, delta=1e-3, maxiter=2000):
    """The objective over C = L L^T in a small basis (W = U V, m columns), by L-BFGS on L: O(N m^2) per evaluation."""
    from popsregression._projected_ball import log_norm_constant, smooth_log
    from scipy.optimize import minimize
    m = W.shape[1]
    k = 0.5 * (P - 1)
    r2 = r * r
    N = len(r)

    def f(x):
        L = x.reshape(m, m)
        H = W @ L
        v = np.einsum("ij,ij->i", H, H) + delta * delta
        q = 1.0 - r2 / v
        lq, dlq, _ = smooth_log(q, rho)
        val = float(np.mean(0.5 * np.log(v) - log_norm_constant(P) - k * lq))
        a = (0.5 / v - k * dlq * r2 / v ** 2) / N
        return val, (2.0 * W.T @ (a[:, None] * H)).ravel()
    w, Q = np.linalg.eigh(0.5 * (C + C.T))
    L0 = Q * np.sqrt(np.maximum(w, 1e-12 * max(w.max(), 1e-300)))
    res = minimize(f, L0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    L = res.x.reshape(m, m)
    H = W @ L
    v = np.einsum("ij,ij->i", H, H) + delta * delta
    return L @ L.T, {"rho": rho, "iters": int(res.nit), "obj": float(res.fun), "covered": float(np.mean(1 - r2 / v > 0))}


def fixed_point_lowrank(U, r, V, C, rank, inner, outer=60, rhos=(1e-1, 1e-2, 1e-3, 1e-4), delta=1e-3, tol=1e-7, seed=1e-2,
                        rng=None, inner_solver="fixed_point"):
    """Rank-limited fixed point: B = V C V^T, V (P, m) orthonormal, m <= 2 rank; per-step cost O(N P rank), as an L-BFGS evaluation.

    Per rho, outer steps of:
      1. inner: fixed_point on C in w_i = V^T u_i (O(N m^2) per step; k and C_P from the full P);
      2. truncate to the top `rank` eigen-directions of C (the package's rank);
      3. expand: the gradient of the objective in B is G = mean a_i u_i u_i^T (a_i = dl_i/dv_i); a direction d outside V lowers
         the objective when d^T G d < 0, so append the `rank` most negative eigenvectors of G on V's complement (block LOBPCG,
         G never formed: O(N P rank) per block product), each seeded at seed x the mean kept width (a multiplicative update
         cannot grow a zero width); refit C on the expanded basis.
    Stops when the objective after truncation changes by < tol relative, or when G on the complement has no negative
    eigenvalue below -tol x |G|'s scale (KKT for the rank-unconstrained problem).
    """
    from popsregression._projected_ball import smooth_log
    from scipy.sparse.linalg import LinearOperator, lobpcg
    rng = np.random.default_rng(0) if rng is None else rng
    N, P = U.shape
    k = 0.5 * (P - 1)
    r2 = r * r

    def trunc(V, C):
        w, Q = np.linalg.eigh(0.5 * (C + C.T))
        idx = np.argsort(w)[::-1][:rank]
        return V @ Q[:, idx], np.diag(np.maximum(w[idx], 0.0))

    def fit_inner(V, C, rho):
        W = U @ V
        if inner_solver == "lbfgs":
            return inner_lbfgs(W, r, C, P, rho, delta=delta, maxiter=inner)
        C, lg = fixed_point(W, r, C, inner, rhos=(rho,), delta=delta, tol=tol, P_ball=P, verbose=False)
        return C, lg[0]

    log = []
    for rho in rhos:
        V, C = trunc(V, C)
        C, lg = fit_inner(V, C, rho)
        prev = lg["obj"]
        for out in range(outer):
            V, C = trunc(V, C)
            W = U @ V
            v = np.einsum("ij,ij->i", W @ C, W) + delta * delta
            _, dlq, _ = smooth_log(1.0 - r2 / v, rho)
            a = (0.5 / v - k * dlq * r2 / v ** 2) / N

            def Gm(X, a=a, V=V):           # G restricted to V's complement, on a block X (P, b)
                X = X - V @ (V.T @ X)
                Y = U.T @ (a[:, None] * (U @ X))
                return Y - V @ (V.T @ Y)
            if P <= 1000:                  # dense: N P^2 is cheap here
                Gd = Gm(np.eye(P))
                ev, D = np.linalg.eigh(0.5 * (Gd + Gd.T))
                ev, D = ev[:rank], D[:, :rank]
            else:                          # o4d20: block LOBPCG on the operator (V's span is projected out by Gm itself)
                X0 = rng.standard_normal((P, rank))
                X0 -= V @ (V.T @ X0)
                op = LinearOperator((P, P), matvec=lambda x: Gm(x[:, None])[:, 0], matmat=Gm, dtype=float)
                ev, D = lobpcg(op, X0, largest=False, maxiter=40, tol=1e-6)
            scale = float(np.abs(a).sum() * np.mean(np.sum(U[:2048] ** 2, 1)))
            neg = ev < -1e-8 * scale
            if not np.any(neg):
                log.append({"rho": rho, "outer": out, "obj": prev, "stop": "kkt", "min_ev": float(ev.min())})
                break
            D = np.linalg.qr(D[:, neg] - V @ (V.T @ D[:, neg]))[0]
            Vn = np.linalg.qr(np.hstack([V, D]))[0]
            wk = np.diag(C)
            Cn = Vn.T @ (V @ C @ V.T) @ Vn                       # C in the new basis
            Cn = Cn + seed * float(np.mean(wk[wk > 0])) * (np.eye(Vn.shape[1]) - Vn.T @ V @ V.T @ Vn)
            C, lg = fit_inner(Vn, Cn, rho)
            V = Vn
            Vt, Ct = trunc(V, C)
            Ct, lgt = fit_inner(Vt, Ct, rho)
            obj = lgt["obj"]
            print(f"  low-rank rho {rho:g} outer {out}: dim {Vn.shape[1]} obj {lg['obj']:.6f} -> rank {rank} {obj:.6f}; "
                  f"{int(neg.sum())} directions with d^T G d < 0 (min {ev.min():.3g}); covered {lgt['covered']:.4f}", flush=True)
            V, C = Vt, Ct
            rel = abs(prev - obj) / max(abs(prev), 1e-300)
            if obj > prev:                                   # truncation lost more than the expansion gained: keep previous
                log.append({"rho": rho, "outer": out, "obj": prev, "stop": "no gain", "min_ev": float(ev.min())})
                break
            prev = obj
            if rel < tol:
                log.append({"rho": rho, "outer": out, "obj": obj, "stop": "tol", "min_ev": float(ev.min())})
                break
        else:
            log.append({"rho": rho, "outer": outer, "obj": prev, "stop": "max_outer"})
    return V, C, log


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
    mxit = None                        # --max-iter k: L-BFGS iterations per continuation stage (package default 500)
    if "--max-iter" in args:
        i = args.index("--max-iter"); mxit = int(args[i + 1]); del args[i:i + 2]
    rw = "--solver-reweight" in args   # popsregression solver="reweight" (branch ellipse-reweight): fixed-point reweighting
    if rw:                             # of the full-rank B in the package's own coordinates, in place of L-BFGS
        args.remove("--solver-reweight")
    mi = "--init-moments" in args      # popsregression init_widths="moments" (branch ellipse-moment-init): U from the
    if mi:                             # per-point optimum v_i = P r_i^2 (a sandwich), not a 1e-3 random factor
        args.remove("--init-moments")
    cf = None                          # --closed-form q: no nonlinear fit (plan §2d): shape A^-1 M A^-1 from the POPS
    if "--closed-form" in args:        # corrections, M = mean x x^T (r/h)^2; scale c covering q % of rows
        i = args.index("--closed-form"); cf = float(args[i + 1]); del args[i:i + 2]
    fp = None                          # --fixed-point K: after --closed-form, K reweighting steps per rho stage (plan §2d)
    if "--fixed-point" in args:
        i = args.index("--fixed-point"); fp = int(args[i + 1]); del args[i:i + 2]
    fpr, fpi = None, "fixed_point"     # --fp-rank R [--fp-inner lbfgs]: the rank-limited fixed point (fixed_point_lowrank)
    if "--fp-rank" in args:
        i = args.index("--fp-rank"); fpr = int(args[i + 1]); del args[i:i + 2]
    fps = None                         # --fp-start seed: start the fixed point from a random Wishart B (multi-start check)
    if "--fp-start" in args:
        i = args.index("--fp-start"); fps = int(args[i + 1]); del args[i:i + 2]
    if "--fp-inner" in args:
        i = args.index("--fp-inner"); fpi = args[i + 1]; del args[i:i + 2]
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
    sfx = ("" if wr_rel is None else f"_wr{wr_rel:g}") + ("_sp" if sp else "") + ("_kl" if kl else "") + ("" if lev is None else f"_lev{lev:g}") + ("" if mxit is None else f"_it{mxit}") + ("_mi" if mi else "") + ("_rw" if rw else "") + ("_pac" if pac else "") + ("" if cf is None else f"_cf{cf:g}") + ("" if fp is None else f"_fp{fp}") + ("" if fpr is None else f"_r{fpr}" + ("_lb" if fpi == "lbfgs" else "")) + ("" if fps is None else f"_s{fps}")
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
    elif cf is not None:
        # Closed-form POPS ellipsoid (plan §2d), in the prior coordinates beta = Lambda^1/2 theta (prior = I), A = X^T X + I.
        # POPS correction of row i: d_i = A^-1 x_i r_i / h_i, h_i = x_i^T A^-1 x_i (the smallest shift fitting it exactly;
        # CR3/PRESS: A^-1 x_i r_i / (1 - h_i)). Shape S = A^-1 M A^-1, M = mean_i x_i x_i^T (r_i / h_i)^2 (over the rows kept
        # by the leverage cut). Scale: the projected-ball covering condition r_i^2 <= c x_i^T S x_i, c its q-th percentile
        # over the kept rows (q = 100: every row, as POPS). Gaussian reading c S / (P + 2), as the fitted ellipsoid's.
        from scipy.linalg import solve_triangular
        if not sp:
            raise SystemExit("--closed-form needs --smooth-prior")
        X, y, lam_p = design()
        X = np.array(X)                                           # the cached design is a read-only memmap
        ge, gv = np.linalg.eigh(X.T @ X / X.shape[0])
        dinv = 1.0 / np.sqrt(lam_p)
        X *= dinv[None, :]
        R = np.linalg.qr(X, mode="r")
        R2 = np.linalg.qr(np.vstack([R, np.eye(n)]), mode="r")    # R2^T R2 = A
        del R
        # Work in u_i = R2^-T x_i (||u_i||^2 = h_i <= 1): x_i^T A^-1 x_j = u_i^T u_j. Forming A^-1 M A^-1 in beta directly
        # cancels catastrophically here (|x| up to 3e8, diag M up to 1e18: v_i came out negative).
        Uu = np.empty_like(X)
        for j in range(0, X.shape[0], 32768):
            Uu[j:j + 32768] = solve_triangular(R2, X[j:j + 32768].T, trans="T").T
        del X
        h = np.sum(Uu * Uu, axis=1)
        keep_r = np.ones(len(h), bool) if lev is None else h >= np.percentile(h, lev)
        Uk, yk, hk = Uu[keep_r], y[keep_r], h[keep_r]
        del Uu
        g = yk / hk
        Su = (Uk * g[:, None]).T @ (Uk * g[:, None]) / len(yk)
        Su = 0.5 * (Su + Su.T)
        v = np.einsum("ij,ij->i", Uk @ Su, Uk)   # BLAS matmul: the 3-operand einsum ran 2 h on one core at o4d20
        ratio = yk ** 2 / np.maximum(v, np.finfo(float).tiny)
        c = float(np.percentile(ratio, cf))
        fp_log = None
        if fp is not None and fpr is not None:   # rank-limited: start from the moment's top fpr eigen-directions
            w0, Q0 = np.linalg.eigh(Su)
            idx = np.argsort(w0)[::-1][:fpr]
            V, Cr, fp_log = fixed_point_lowrank(Uk, yk, Q0[:, idx], c * np.diag(w0[idx]), fpr, fp, inner_solver=fpi)
            c, Su = 1.0, V @ Cr @ V.T
        elif fp is not None:           # fixed-point reweighting from the stationarity of the fitted objective (full-rank B, u coords)
            B0u = c * Su
            if fps is not None:        # random start: B = c' G G^T, G (P, P) Gaussian, c' covering every kept row
                Gr = np.random.default_rng(fps).standard_normal((n, n))
                B0u = Gr @ Gr.T
                v0 = np.einsum("ij,ij->i", Uk @ B0u, Uk)
                B0u *= float(np.max(yk ** 2 / v0))
            Bu, fp_log = fixed_point(Uk, yk, B0u, fp)
            c, Su = 1.0, Bu
        Ri = solve_triangular(R2, np.eye(n))                      # R2^-1
        S = Ri @ Su @ Ri.T                                        # beta coordinates
        S = 0.5 * (S + S.T)
        Dm = np.diag(dinv)
        info = {"P": n, "N": int(len(yk)), "closed_form_pct": cf, "c": c, "leverage_pct": lev, "smooth_prior": True,
                "h_sum": float(h.sum()), "v_min": float(v.min()), "fixed_point": fp_log, "fp_start": fps, "ratio_pct": {str(q): float(np.percentile(ratio, q)) for q in (50, 90, 99, 100)},
                "fit_s": time.time() - t0}
        z = dict(Sigma_e=Dm @ (c * S / (n + 2.0)) @ Dm, B0=np.zeros((n, n)), delta=np.zeros(n), info=json.dumps(info),
                 G_evals=ge, G_evecs=gv)
        del Uk
        np.savez(fout, **z)
        print(f"{m}: closed-form ellipsoid: {info}", flush=True)
    else:
        X, y, lam_p = design()
        ge, gv = np.linalg.eigh(X.T @ X / X.shape[0])              # model coordinates (the data span)
        opts = None if wr_rel is None else {"whiten_ridge": wr_rel * float(ge.max())}
        dinv = np.ones(n)
        if sp:                                                    # beta = Lambda^1/2 theta: the prior is the identity there
            dinv = 1.0 / np.sqrt(lam_p)
            X = np.array(X)                                       # the cached design is a read-only memmap
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
        if mxit is not None:
            opts = {**(opts or {}), "max_iter": mxit}
        if mi:
            opts = {**(opts or {}), "init_widths": "moments"}
        if rw:
            opts = {**(opts or {}), "solver": "reweight"}
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
                "whiten_ridge": float(eng.whiten_ridge), "whiten_ridge_rel": wr_rel, "pac_bayes": pac, "smooth_prior": sp, "param_prior": kl, "leverage_pct": lev, "max_iter": mxit, "init_moments": mi, "solver_reweight": rw,
                "stages": getattr(eng, "stage_info_", None),
                "param_prior_kl": (float(eng.param_prior_kl_) if kl else None),
                **({"tau2": float(eng.tau2_), "bound": float(eng.bound_), "kl": float(eng.kl_)} if pac else {}), "G_eig_max": float(ge.max())}
        z = dict(Sigma_e=Dm @ (np.asarray(eng.ellipsoid_B_) / (P + 2.0)) @ Dm,    # as POPSRegression's own sigma; beta -> theta
                 B0=Dm @ eng._whiten_W @ np.asarray(eng.baseline_B0_) @ eng._whiten_W @ Dm,    # whitened -> model coordinates
                 delta=delta, info=json.dumps(info), G_evals=ge, G_evecs=gv,
                 U=np.asarray(eng.U_), W=np.asarray(eng._whiten_W), dinv=dinv)   # for a warm start / rescoring
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
    if not np.any(B0):                 # --closed-form has no baseline: its hypercube0 rows repeat the ellipsoid (ignore them)
        B0 = Se
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
