"""Multivariate (whole-configuration) conformal prediction with the CR3 covariance (next-steps plan §4; Fisher, Herbst,
Kermode and Marzouk, arXiv:2609.31384; Mahalanobis score as Braun et al., arXiv:2507.20941).

Per test configuration c, its force errors e_c (D_c live components; symmetry-zero atoms dropped as force_baselines.py)
and the covariance of a method's parameter factor L, with the white nugget §0 showed is needed (config_mahal.py):
    Sigma_c(lam, nu) = lam^2 M_c M_c^T + nu^2 I,   M_c = F_c L,
    s_c = sqrt(e_c^T Sigma_c^-1 e_c / D_c)            (the d^-1/2 |Sigma^-1/2 e| score),
computed from the thin SVD M_c = U S V^T: e^T Sigma^-1 e = sum_k a_k^2 / (lam^2 S_k^2 + nu^2) + (|e|^2 - sum a_k^2) / nu^2.
Protocol (the scales of the paper are all set on the test split, so this one is too, but split so its coverage can be
tested): for each of N_SPLIT random halves of the test configurations,
  1. shape: (lam, nu) by maximum Gaussian likelihood of the calibration half's e_c (no QoI enters);
  2. scale: q = the ceil((1 - alpha)(n + 1)) / n empirical quantile of s_c over the calibration half;
  3. joint coverage on the other half, s_c <= q, overall and by kind of configuration (alpha = 0.1).
The region scales lam and nu together by q, so its parameter part is lam_mv = q lam: QoI intervals sigma_Q = lam_mv |L^T g|
(delta rows, 17 QoIs with DFT; no discrepancy term) are compared with the headline lam_F, using the whole test split.
Methods: CR3 (headline shape) and B1 (posterior shape).  The propagation-corrected variant of Fisher et al. needs labels of
each QoI type (calibration on DFT QoIs), so it is not run.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.mv_conformal o3d12|o4d20 -> RUNS/diag/<m>_mv_spectra.npz,
        results/mv_conformal_<m>.json   (stage 1, the spectra, is cached)
"""
import csv, json, sys

import numpy as np

ALPHA = 0.1
N_SPLIT = 200
Z95 = 1.959963984540054
KIND = {"bulk": ("dia", "bt", "sh", "hcp", "fcc", "bcc", "bc8", "st12", "hex_diamond", "liq", "amorph"),
        "point defect": ("vacancy", "divacancy", "interstitial"), "surface": ("surface", "111adatom"), "extended": ("crack", "screw", "decohesion", "gamma")}


def kind_of(t):
    for k, pre in KIND.items():
        if any(str(t).startswith(p) for p in pre):
            return k
    return "other"


def spectra(m):
    import jax
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..diagnostics import SYM_ZERO, TEST_XYZ
    from ..run import lin_for
    lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
    facs = {"cr3": np.asarray(lin.factor("cr3")), "post": np.asarray(linp.factor("posterior"))}
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    Lc = facs["cr3"]
    per, gc = [], 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        a = np.flatnonzero(np.asarray(bt.node_mask) & (np.asarray(bt.w_F) > 0))
        E = np.asarray(bt.y_F)[a] - F[a] @ th
        nc = np.asarray(bt.node_cfg)[a]
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        for j, c in enumerate(live):
            s = nc == c
            per.append((gc + j, E[s], F[a][s]))
        gc += len(live)
    sd_all = np.concatenate([np.sqrt(np.sum((Fc @ Lc) ** 2, axis=(1, 2))) for _, _, Fc in per])
    cut = SYM_ZERO * np.median(sd_all)
    out = {"cfg": [], "config_type": [], "D": [], "ee": [], **{f"{k}_{f}": [] for k in facs for f in ("S", "a")}}
    for c, E, Fc in per:
        keep = np.sqrt(np.sum((Fc @ Lc) ** 2, axis=(1, 2))) > cut
        if not keep.any():
            continue
        e = E[keep].reshape(-1)
        Fk = Fc[keep].reshape(-1, Fc.shape[-1])
        out["cfg"].append(c); out["config_type"].append(str(cfgs[c].config_type)); out["D"].append(len(e)); out["ee"].append(float(e @ e))
        for k, L in facs.items():
            U, S, _ = np.linalg.svd(Fk @ L, full_matrices=False)
            r = S > 1e-12 * S[0]
            out[f"{k}_S"].append(S[r]); out[f"{k}_a"].append(U[:, r].T @ e)
    return out


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from scipy.optimize import minimize
    from ..harness import RUNS
    from ..propagate import tol_abs
    from ..run import RESULTS, lin_for
    m = sys.argv[1]
    f = RUNS / "diag" / f"{m}_mv_spectra.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        sp = {k: (list(z[k]) if k.endswith(("_S", "_a")) else z[k]) for k in z.files}
    else:
        sp = spectra(m)
        np.savez(f, **{k: (np.array(v, dtype=object) if k.endswith(("_S", "_a")) or k == "config_type" else np.array(v))
                       for k, v in sp.items()})
    D, ee = np.asarray(sp["D"], float), np.asarray(sp["ee"], float)
    kinds = np.array([kind_of(t) for t in sp["config_type"]])
    nC = len(D)
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
    ref_scale = {"cr3": float(sc["lam_cr3"]["F"]), "post": float(sc["q_b1"])}

    def r_of(k, lam, nu, idx):
        out = np.empty(len(idx))
        for j, c in enumerate(idx):
            S, a = sp[f"{k}_S"][c], sp[f"{k}_a"][c]
            v = lam ** 2 * S ** 2 + nu ** 2
            out[j] = np.sum(a ** 2 / v) + (ee[c] - np.sum(a ** 2)) / nu ** 2
        return out

    def nll(k, x, idx):
        lam, nu = np.exp(x)
        t = 0.0
        for c in idx:
            S, a = sp[f"{k}_S"][c], sp[f"{k}_a"][c]
            v = lam ** 2 * S ** 2 + nu ** 2
            t += np.sum(np.log(v)) + (D[c] - len(S)) * np.log(nu ** 2) + np.sum(a ** 2 / v) + (ee[c] - np.sum(a ** 2)) / nu ** 2
        return 0.5 * t

    def fit_shape(k, idx):
        x0 = np.log([ref_scale[k], 0.1 * np.sqrt(np.sum(ee[idx]) / np.sum(D[idx]))])
        return np.exp(minimize(lambda x: nll(k, x, idx), x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8}).x)

    def qhat(s, a=ALPHA):
        n = len(s)
        return float(np.sort(s)[min(int(np.ceil((1 - a) * (n + 1))) - 1, n - 1)])

    rng = np.random.default_rng(0)
    res = {"model": m, "n_configs": int(nC), "alpha": ALPHA, "n_split": N_SPLIT, "kinds": {k: int((kinds == k).sum()) for k in np.unique(kinds)},
           "methods": {}}
    lin = lin_for(m, "cr3")
    facs = {"cr3": np.asarray(lin.factor("cr3")), "post": np.asarray(lin_for(m, "kappa").factor("posterior"))}
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    G = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
    csvr = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "qoi_uq.csv"))
            if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
    for k in ("cr3", "post"):
        lam_all, nu_all = fit_shape(k, np.arange(nC))
        s_all = np.sqrt(r_of(k, lam_all, nu_all, np.arange(nC)) / D)
        q_all = qhat(s_all)
        cov, by, lam_mv, qs = [], {kk: [] for kk in np.unique(kinds)}, [], []
        for it in range(N_SPLIT):
            p = rng.permutation(nC)
            cal, ev = p[: nC // 2], p[nC // 2:]
            lam, nu = fit_shape(k, cal) if it < 40 else (lam_all, nu_all)   # shape refitted on the first 40 splits (cost)
            q = qhat(np.sqrt(r_of(k, lam, nu, cal) / D[cal]))
            ok = np.sqrt(r_of(k, lam, nu, ev) / D[ev]) <= q
            cov.append(ok.mean()); qs.append(q); lam_mv.append(q * lam)
            for kk in by:
                mk = kinds[ev] == kk
                if mk.any():
                    by[kk].append(ok[mk].mean())
        qo = {}
        L = facs[k]
        for q_, g in G.items():
            if q_.startswith("energy:") or q_ not in csvr or csvr[q_]["dft"] in ("", "nan"):
                continue
            e = float(csvr[q_]["dft"]) - float(csvr[q_]["mean"])
            a_g = float(np.linalg.norm(L.T @ g[:L.shape[0]]))
            sg = q_all * lam_all * a_g
            qo[q_] = {"sigma": sg, "sigma_over_tol": sg / tol_abs(q_, float(csvr[q_]["dft"])), "z": e / sg, "covered95": bool(abs(e / sg) <= Z95),
                      "z_ref_scale": e / (ref_scale[k] * a_g)}
        res["methods"][k] = {
            "ref_scale": ref_scale[k], "lam_ml_all": float(lam_all), "nu_ml_all_eV_per_A": float(nu_all), "q_all": q_all,
            "lam_mv_all": float(q_all * lam_all), "nu_mv_all_eV_per_A": float(q_all * nu_all),
            "nu_over_rms_error": float(nu_all / np.sqrt(np.sum(ee) / np.sum(D))),
            "joint_coverage_eval": {"mean": float(np.mean(cov)), "p05": float(np.quantile(cov, 0.05)), "p95": float(np.quantile(cov, 0.95))},
            "joint_coverage_by_kind": {kk: {"mean": float(np.mean(v)), "n_per_half": float((kinds == kk).sum() / 2)} for kk, v in by.items() if v},
            "lam_mv_split": {"median": float(np.median(lam_mv)), "p05": float(np.quantile(lam_mv, 0.05)), "p95": float(np.quantile(lam_mv, 0.95))},
            "qoi": qo, "qoi_k95": int(sum(v["covered95"] for v in qo.values())), "qoi_k95_ref_scale": int(sum(abs(v["z_ref_scale"]) <= Z95 for v in qo.values())),
            "qoi_n": len(qo), "qoi_median_sigma_over_tol": float(np.median([v["sigma_over_tol"] for v in qo.values()]))}
        r = res["methods"][k]
        print(f"{m} {k}: ML lam {lam_all:.3f}, nu {nu_all:.4f} eV/A ({r['nu_over_rms_error']:.2f} of the RMS error); q {q_all:.3f} -> lam_mv "
              f"{r['lam_mv_all']:.3f} (paper scale {ref_scale[k]:.3f}); split lam_mv {r['lam_mv_split']}", flush=True)
        print(f"  joint 90 % coverage on held-out halves {r['joint_coverage_eval']}; by kind "
              + ", ".join(f"{kk} {v['mean']:.2f} (n {v['n_per_half']:.0f})" for kk, v in r["joint_coverage_by_kind"].items()), flush=True)
        print(f"  QoIs at lam_mv: 95 % {r['qoi_k95']}/{r['qoi_n']} (at the paper scale {r['qoi_k95_ref_scale']}), median sigma/tol "
              f"{r['qoi_median_sigma_over_tol']:.2f}", flush=True)
    json.dump(res, open(RESULTS / f"mv_conformal_{m}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
