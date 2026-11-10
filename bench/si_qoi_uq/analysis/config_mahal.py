"""Configuration-level Mahalanobis scores of the test-split force errors (next-steps plan §0, a first test of §4).

Per test configuration c, the force errors e_c (all live components, symmetry-zero atoms dropped as in
force_baselines.py) and each method's covariance Sigma_c = s^2 M_c M_c^T, M_c = F_c L (F_c the (3n, p) force rows,
L the method's parameter factor, s its scale: headline lam_F with CR3, B1 q_b1 with the posterior, B2 the POPS box
at the 50 % leverage cut read as a Gaussian, scale 1).  Same factors and scales as force_baselines.py.

With M_c = U S V^T (thin SVD), lam_k = (s S_k)^2 descending, a_k = u_k^T e, and the trace share t_k = cumsum(lam) / tr:
  band ratios  per band of t (BANDS: the directions that carry the first 50 % of the trace, the next 40 %, 9 %, 0.9 %,
               the rest): w = sum a_k^2 / sum lam_k (variance-weighted: ~1 if the band's size is right) and
               m = mean a_k^2 / lam_k (whitened: ~1 if every direction in it is right), and its share of |e|^2
  r_99, d_99   sum a_k^2 / lam_k over the directions with t_k <= 0.99 and their number: chi^2_{d_99} if the shape is right
  f_out        1 - sum_k a_k^2 / |e|^2 over all directions with S_k > TOL S_0: error with no direction at all
  r_eps        e^T (Sigma + delta I)^-1 e / D, delta = eps tr(Sigma) / D, D = len(e), for eps in EPS
Whitening every direction down to roundoff is meaningless (near-null directions dominate it), hence the trace cut.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.config_mahal o3d12|o4d20 -> results/config_mahal_<m>.npz
"""
import json, sys

import numpy as np

TOL = 1e-6
EPS = (1e-3, 1e-2, 1e-1)
BANDS = (0.5, 0.9, 0.99, 0.999, 1.0)


def scores(e, M, s):
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    lam = (s * S) ** 2
    k = S > TOL * S[0]
    U, lam = U[:, k], lam[k]
    a2 = (U.T @ e) ** 2
    tr = float(lam.sum())
    t = np.cumsum(lam) / tr
    band = np.searchsorted(np.array(BANDS), t - 1e-12)              # band index of each direction
    ee = float(e @ e)
    w, mr, sh = [], [], []
    for b in range(len(BANDS)):
        i = band == b
        w.append(float(a2[i].sum() / lam[i].sum()) if i.any() else np.nan)
        mr.append(float(np.mean(a2[i] / lam[i])) if i.any() else np.nan)
        sh.append(float(a2[i].sum() / ee))
    i99 = t <= 0.99
    i99[0] = True
    D = len(e)
    r_eps = []
    for eps in EPS:                                                # (Sigma + dI)^-1 = U diag(1/(l+d)) U^T + (I - UU^T)/d
        d = eps * tr / D
        r_eps.append(float((np.sum(a2 / (lam + d)) + (ee - a2.sum()) / d) / D))
    return dict(w=w, m=mr, share=sh, r_99=float(np.sum(a2[i99] / lam[i99])), d_99=int(i99.sum()),
                f_out=float(1 - a2.sum() / ee), r_eps=r_eps, tr=tr, rank=int(k.sum()))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..diagnostics import SYM_ZERO, TEST_XYZ
    from ..harness import RUNS
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
    lamF, qb1 = float(sc["lam_cr3"]["F"]), float(sc["q_b1"])
    b = np.load(fit_dir(m, "pops") / "pops_box_lev50.npz")
    Lp2 = np.asarray(b["support"]) * ((np.asarray(b["hi"]) - np.asarray(b["lo"])) / np.sqrt(12))[None, :]
    meth = {"headline": (np.asarray(lin.factor("cr3")), lamF), "B1": (np.asarray(linp.factor("posterior")), qb1),
            "B2": (Lp2, 1.0)}
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    Lc = meth["headline"][0]
    per = []                                                       # (config index, e (n,3), F (n,3,p)) per configuration
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        nm, wF = np.asarray(bt.node_mask), np.asarray(bt.w_F)
        a = np.flatnonzero(nm & (wF > 0))
        E = np.asarray(bt.y_F)[a] - F[a] @ th
        nc = np.asarray(bt.node_cfg)[a]
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        for j, c in enumerate(live):
            s = nc == c
            per.append((gc + j, E[s], F[a][s]))
        gc += len(live)
    sd_all = np.concatenate([np.sqrt(np.sum((Fc[..., :Lc.shape[0]] @ Lc) ** 2, axis=(1, 2))) for _, _, Fc in per])
    cut = SYM_ZERO * np.median(sd_all)                             # the force_baselines.py symmetry-zero rule
    out = {k: [] for k in ("cfg", "config_type", "n_atoms", "D")}
    FIELDS = ("w", "m", "share", "r_99", "d_99", "f_out", "r_eps", "tr", "rank")
    for k in meth:
        for f in FIELDS:
            out[f"{f}_{k}"] = []
    for c, E, Fc in per:
        sd = np.sqrt(np.sum((Fc[..., :Lc.shape[0]] @ Lc) ** 2, axis=(1, 2)))
        keep = sd > cut
        if not keep.any():
            continue
        e = E[keep].reshape(-1)
        Fk = Fc[keep].reshape(-1, Fc.shape[-1])
        out["cfg"].append(c), out["config_type"].append(cfgs[c].config_type)
        out["n_atoms"].append(int(keep.sum())), out["D"].append(len(e))
        for k, (L, s) in meth.items():
            r = scores(e, Fk[:, :L.shape[0]] @ L, s)
            for f in FIELDS:
                out[f"{f}_{k}"].append(r[f])
    out = {k: np.array(v, dtype=object if k == "config_type" else None) for k, v in out.items()}
    out["eps"], out["bands"], out["tol"], out["lam_F"], out["q_b1"] = np.array(EPS), np.array(BANDS), TOL, lamF, qb1
    np.savez(RESULTS / f"config_mahal_{m}.npz", **out)
    report(m, out)


def report(m, out):
    from scipy.stats import chi2
    for k in ("headline", "B1", "B2"):
        r, d = out[f"r_99_{k}"], out[f"d_99_{k}"]
        print(f"{m} {k:8s} n_cfg {len(r)}  r_99/d_99 median {np.median(r / d):.2f}  P(r_99 <= chi2_d(0.9)) "
              f"{np.mean(r <= chi2(d).ppf(0.9)):.2f}  median d_99 {np.median(d):.0f}  median f_out {np.median(out[f'f_out_{k}']):.3f}"
              f"  r_eps/D median {np.round(np.median(out[f'r_eps_{k}'], axis=0), 2).tolist()}", flush=True)
        print(f"   per trace band {out['bands'].tolist()}: weighted ratio (median) "
              f"{np.round(np.nanmedian(out[f'w_{k}'], axis=0), 2).tolist()}  whitened (median) "
              f"{np.round(np.nanmedian(out[f'm_{k}'], axis=0), 2).tolist()}  share of |e|^2 (mean) "
              f"{np.round(np.mean(out[f'share_{k}'], axis=0), 3).tolist()}", flush=True)

if __name__ == "__main__":
    main()
