"""A shared-bias term for the parameter covariance, from the extrapolated defect-bulk contrast (UQ fix; theta unchanged).

The relaxed-defect QoIs are missed because every hot defect snapshot carries the same thermal-amplitude trend in
the model's error (RECON, "Why the interstitial formation energies are confidently wrong"); a bias shared by all
clusters is invisible to the CR3 jackknife.  The trend's zero-amplitude intercept is linear in the data, so it is a
synthetic observation of the relaxed-defect formation contrast built from DFT training labels alone:

    d*_t = sum_i a_i d_i,   y*_t = sum_i a_i y_i,   a = first row of (X^T X)^-1 X^T,  X = [1, F_rms^2(defect cell i)]

over the training defect snapshots i of type t, with d_i = b_def - (n_def / n_b) b_bulk against the bulk snapshot paired
by (volume, F_rms^2) (analysis/diff_rows.py), y_i likewise.  Its residual at the unchanged theta_hat, r*_t, is the model's
error in the direction the QoIs probe; its standard error s*_t comes from the regression's scatter.  The minimum-norm
correction in the CR3 metric that would remove it (the POPS pointwise correction, here of the synthetic row) is

    delta_t = L L^T d*_t r*_t / |L^T d*_t|^2 ,

and the augmented covariance is Sigma_aug = lam^2 L L^T + sum_t c_t^2 delta_t delta_t^T,  c_t^2 = 1 + (s*_t / r*_t)^2.
No QoI value or tolerance enters.  Checks: QoI coverage through the delta rows (RUNS/delta/<m>.npz), and the
test-split calibration with Sigma_aug.

    python -m bench.si_qoi_uq.analysis.bias_term o3d12|o4d20      (SIQ_RUNS: the BLR run root)
"""
import csv, json, sys, time

import numpy as np

from .diff_rows import DEFECT, pair


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..diagnostics import test_scores
    from ..harness import RUNS
    from ..linear import LinearACE
    from ..propagate import COVER_KEYS, KEYS, dft, harness_values, tol_abs
    from ..run import RESULTS
    m = sys.argv[1]
    t0 = time.time()
    lin = LinearACE(RUNS / "fits" / m / "cr3" / "model.npz", RUNS / "fits" / m / "cr3" / "posterior.npz")
    th = lin.theta_hat
    L = lin.factor("cr3")
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))["lam_cr3"]
    D = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
    train = load_configs(f"{D}/train.xyz", virial_key="virial")
    sel = [i for i, c in enumerate(train) if c.config_type in DEFECT + ("dia",) and c.energy is not None]
    cs = [train[i] for i in sel]
    ds = build_dataset(cs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    E, yE = np.zeros((len(cs), lin.p)), np.zeros(len(cs))
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        RE = np.asarray(rows(bt).E)
        for c in np.flatnonzero(np.asarray(bt.cfg_mask)):
            E[gc], yE[gc] = RE[c], float(bt.y_E[c])
            gc += 1
    nat = np.array([len(c.numbers) for c in cs])
    types = np.array([c.config_type for c in cs], dtype=object)
    pairs = pair(cs)
    di, bi = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ratio = nat[di] / nat[bi]
    Dd = E[di] - ratio[:, None] * E[bi]
    yd = yE[di] - ratio * yE[bi]
    x = np.array([float(np.mean(cs[i].forces ** 2)) for i in di])
    print(f"{len(pairs)} defect-bulk pairs ({time.time() - t0:.0f} s)", flush=True)
    syn, dstars = {}, {}
    for t in DEFECT:
        mt = types[di] == t
        X = np.stack([np.ones(mt.sum()), x[mt]], 1)
        a = np.linalg.solve(X.T @ X, X.T)[0]                          # intercept functional
        d_star, y_star = a @ Dd[mt], a @ yd[mt]
        res = yd[mt] - Dd[mt] @ th
        k = np.linalg.lstsq(X, res, rcond=None)[0]
        s2 = np.sum((res - X @ k) ** 2) / max(mt.sum() - 2, 1)
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]))
        r_star = float(y_star - d_star @ th)
        c = np.sqrt(1 + (se / r_star) ** 2) if r_star else 1.0
        syn[t] = {"n_snapshots": int(mt.sum()), "r_star_eV": r_star, "se_eV": se, "c": float(c),
                  "cr3_sigma_lamF_eV": float(sc["F"] * np.linalg.norm(L.T @ d_star))}
        dstars[t] = (d_star, r_star, c)
        print(f"{t:12s}: r* = {r_star:+.3f} ± {se:.3f} eV (CR3 x lam_F sigma of the same row "
              f"{syn[t]['cr3_sigma_lamF_eV']:.3f} eV)", flush=True)
    # bias columns per metric: minimum-norm correction of each synthetic row, in the CR3 metric (L L^T) or the
    # posterior metric (A^-1 = Lp Lp^T); the latter weighs directions by what the data and prior leave free
    Lp = lin.factor("posterior")
    Bs = {}
    for metric, F in (("cr3", L), ("post", Lp)):
        cols = []
        for t in DEFECT:
            d_star, r_star, c = dstars[t]
            u = F.T @ d_star
            cols.append(c * (F @ u) * r_star / float(u @ u))
        Bs[metric] = np.stack(cols, 1)
    # lambda re-solved on the test split with the bias part held at its absolute size:
    # rms over rows of kind q of resid^2 / (lam^2 a^2 + b^2) = 1, a = |L^T b_row|, b = |B^T b_row|
    res = test_scores(lin, {"cr3": L, "bias_cr3": Bs["cr3"], "bias_post": Bs["post"]}, log=lambda *_: None)
    lam_aug = {}
    for metric in ("cr3", "post"):
        lam_aug[metric] = {}
        for q in "EFV":
            mk = res["kind"] == q
            r2, a2, b2 = res["resid"][mk] ** 2, res["sd"]["cr3"][mk] ** 2, res["sd"][f"bias_{metric}"][mk] ** 2
            f = lambda lam: np.mean(r2 / (lam ** 2 * a2 + b2)) - 1.0
            lo, hi = 1e-3, 1e3
            if f(lo) < 0:                            # the bias alone already over-covers
                lam_aug[metric][q] = 0.0
                continue
            for _ in range(200):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
            lam_aug[metric][q] = float(np.sqrt(lo * hi))
    by_type = {}
    for t in DEFECT + ("dia",):
        for q in "EF":
            mk = (res["ctype"] == t) & (res["kind"] == q)
            if mk.any():
                by_type[f"cr3/{t}/{q}"] = float(np.sqrt(np.mean((res["resid"][mk] / (sc[q] * res["sd"]["cr3"][mk])) ** 2)))
                for metric in ("cr3", "post"):
                    sd = np.hypot(lam_aug[metric][q] * res["sd"]["cr3"][mk], res["sd"][f"bias_{metric}"][mk])
                    by_type[f"bias_{metric}/{t}/{q}"] = float(np.sqrt(np.mean((res["resid"][mk] / sd) ** 2)))
    # QoI sigmas through the delta rows
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    G = dict(zip([str(k) for k in dz["keys"]], dz["G"]))
    Qhat, ref = harness_values(f"SIQ_{m}_cr3"), dft()
    out = []
    for k in COVER_KEYS:
        if k not in G:
            continue
        i = KEYS.index(k)
        tol = tol_abs(k, ref[i])
        g = G[k]
        a_g = float(np.linalg.norm(L.T @ g))
        for q in ("F", "V"):
            cands = [("cr3", sc[q] * a_g, 0.0)]
            for metric in ("cr3", "post"):
                b_g = float(np.linalg.norm(Bs[metric].T @ g))
                cands.append((f"cr3+bias_{metric}", float(np.hypot(lam_aug[metric][q] * a_g, b_g)), b_g))
            for name, sgm, b_g in cands:
                z = (Qhat[i] - ref[i]) / sgm
                out.append(dict(model=m, cov=name, scale=f"lam_{q}", qoi=k, err_over_tol=(Qhat[i] - ref[i]) / tol,
                                sigma=sgm, sigma_over_tol=sgm / tol, sigma_bias=b_g, z=z, cov68=int(abs(z) <= 1),
                                cov95=int(abs(z) <= 1.96)))
    with open(RESULTS / f"bias_term_{m}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summ = {"model": m, "synthetic_rows": syn, "lam_cr3": sc, "lam_with_bias": lam_aug, "test_rms_z_by_type": by_type}
    for name in ("cr3", "cr3+bias_cr3", "cr3+bias_post"):
        for q in ("lam_F", "lam_V"):
            rr = [r for r in out if r["cov"] == name and r["scale"] == q]
            summ[f"{name}/{q}"] = {"n": len(rr), "k68": sum(r["cov68"] for r in rr), "k95": sum(r["cov95"] for r in rr),
                                   "median_sigma_over_tol": float(np.median([r["sigma_over_tol"] for r in rr])),
                                   "rms_z": float(np.sqrt(np.mean([r["z"] ** 2 for r in rr])))}
    json.dump(summ, open(RESULTS / f"bias_term_{m}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("test_rms_z_by_type", "synthetic_rows")}, indent=1, default=float))
    print(json.dumps(by_type, indent=0, default=lambda v: round(v, 2)))
    for r in out:
        if r["scale"] == "lam_F" and ("interstitial" in r["qoi"] or "vacancy" in r["qoi"]):
            print(f"{r['cov']:14s} {r['qoi'][:30]:30s} err/tol {r['err_over_tol']:+7.2f} sigma/tol {r['sigma_over_tol']:6.2f} "
                  f"(bias {r['sigma_bias'] / (r['sigma'] / r['sigma_over_tol']):5.2f}) z {r['z']:+6.2f}")
    print(f"done, {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
