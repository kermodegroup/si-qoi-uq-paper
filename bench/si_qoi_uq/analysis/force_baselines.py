"""Per-atom force uncertainty of the propagation's methods on the test split (paper fig. 7, beside the Mondrian sigma of
analysis/force_calibration.py).  BLR fits (SIQ_RUNS: the BLR run root), the same test atoms and symmetry-zero rule.

Per test atom, from the fit's force rows F_a (3, p), sigma = sqrt(tr C_a) with C_a the 3x3 force covariance of:
  headline  lam_F^2 F_a L L^T F_a^T, L the CR3 factor, lam_F the global test-split force scale (diag/<m>_scales.json)
  B1        q_b1^2 F_a L_post L_post^T F_a^T (the posterior scaled by one conformal quantile)
  B2        F_a U diag((hi - lo)^2 / 12) U^T F_a^T, the POPS box at the reference package's default leverage cut
            (50 %; uniform variance on each axis)
  B3        F_a L L^T F_a^T (CR3, lam = 1)
  ellipsoid lam_e^2 F_a Sigma_e F_a^T, the converged leverage-cut POPS ellipsoid of pops_ellipse.py (§2), lam_e its
            test-split per-atom scale
Saved per atom also: cfg (test configuration index, for configuration-level splits) and logdet_<method> (log det of the
regularised covariance at the method's scale: the region's log volume up to a constant), for analysis/crc.py.
and each atom's Mondrian group from the ARD posterior's group constants (geometry only; ARD_RUNS).
Also per method the Mahalanobis score d = sqrt(e^T (C_a + eps tr(C_a)/3 I)^-1 e) (eps = the ARD fit's, 1e-3), so every
method's region can be the Gaussian ellipsoid of its own shape, d <= chi_3^-1(level), as the conformal region of the
per-atom method is (s <= q is that ellipsoid at the conformal scale).

    SIQ_RUNS=<BLR root> ARD_RUNS=<ARD root> python -m bench.si_qoi_uq.analysis.force_baselines o3d12|o4d20
        -> results/force_baselines_<m>.npz
"""
import json, os, pathlib, sys

import numpy as np

ELLIPSE = {"o3d12": "_sp_lev50_it20000", "o4d20": "_sp_lev50_it5000"}   # converged L-BFGS leverage-cut ellipsoids (§2)
EPS_SENS = (1e-2, 1e-1)   # x10, x100 the ARD fit's eps: the ellipsoid of a near-singular shape depends on it
ARD_RUNS = pathlib.Path(os.environ.get("ARD_RUNS", "/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_ard137"))


def main():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from scipy.stats import chi
    from ..diagnostics import SYM_ZERO, TEST_XYZ
    from ..harness import RUNS
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
    lamF, qb1 = float(sc["lam_cr3"]["F"]), float(sc["q_b1"])
    b = np.load(fit_dir(m, "pops") / "pops_box_lev50.npz")          # POPSRegression default (leverage_percentile 50)
    Lp2 = np.asarray(b["support"]) * ((np.asarray(b["hi"]) - np.asarray(b["lo"])) / np.sqrt(12))[None, :]
    facs = {"cr3": jnp.asarray(lin.factor("cr3")), "post": jnp.asarray(linp.factor("posterior")), "pops": jnp.asarray(Lp2)}
    ze = np.load(fit_dir(m, "cr3").parent / f"pops_ellipse{ELLIPSE[m]}.npz")    # the converged leverage-cut ellipsoid:
    Se = np.asarray(ze["Sigma_e"]); Se = 0.5 * (Se + Se.T)                      # its covariance, never an eigh factor
    lam_e = float(json.load(open(RESULTS / f"pops_ellipse_{m}{ELLIPSE[m]}.json"))["lam_per_atom_rms"]["ellipsoid"])
    covs = {"ell": jnp.asarray(Se)}
    ard = ARDPosterior.load(ARD_RUNS / "fits" / m / "cr3" / "posterior.npz")
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    E, G, T, CF = [], [], [], []
    V, C = {k: [] for k in (*facs, *covs)}, {k: [] for k in (*facs, *covs)}
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        nm, wF = np.asarray(bt.node_mask), np.asarray(bt.w_F)
        a = np.flatnonzero(nm & (wF > 0))
        E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
        G.append(np.asarray(ard.groups_of(bt))[a])
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        idx = {c: gc + j for j, c in enumerate(live)}
        T += [cfgs[idx[int(c)]].config_type for c in np.asarray(bt.node_cfg)[a]]
        CF.append(np.array([idx[int(c)] for c in np.asarray(bt.node_cfg)[a]], int))   # test configuration index
        gc += len(live)
        Fa = jnp.asarray(F[a])                                         # (n, 3, p)
        for k, L in facs.items():
            M = Fa[..., :L.shape[0]] @ L                               # POPS has no E0 column (zero on force rows)
            V[k].append(np.asarray(jnp.sum(M * M, axis=(1, 2))))
            C[k].append(np.asarray(M @ jnp.swapaxes(M, 1, 2)))                # (n, 3, 3) force covariance
        for k, S in covs.items():
            Fs = Fa[..., :S.shape[0]]
            Ck = Fs @ S @ jnp.swapaxes(Fs, 1, 2)
            V[k].append(np.asarray(jnp.trace(Ck, axis1=1, axis2=2))); C[k].append(np.asarray(Ck))
    E, G, CF = np.concatenate(E), np.concatenate(G), np.concatenate(CF)
    v = {k: np.concatenate(x) for k, x in V.items()}
    eps = float(ard.eps)

    def maha(Ca, scale, eps=eps, logdet=False):                        # d of the method's covariance scale^2 C_a
        Ca = np.concatenate(Ca)[keep] * scale ** 2
        tr = np.trace(Ca, axis1=1, axis2=2)
        reg = Ca + eps * (tr / 3)[:, None, None] * np.eye(3)
        d = np.sqrt(np.einsum("na,nab,nb->n", E[keep], np.linalg.inv(reg), E[keep]))
        return (d, np.linalg.slogdet(reg)[1]) if logdet else d
    sd = np.sqrt(np.maximum(v["cr3"], 0))
    keep = sd > SYM_ZERO * np.median(sd)                               # symmetry-zero atoms: no shape, dropped
    meth = {"headline": ("cr3", lamF), "B1": ("post", qb1), "B2": ("pops", 1.0), "B3": ("cr3", 1.0), "ellipsoid": ("ell", lam_e)}
    d, ld = {}, {}
    for k, (f, sc) in meth.items():                                    # ld: log det of the region's covariance (its
        d[k], ld[k] = maha(C[f], sc, logdet=True)                      # log volume up to a constant), at that scale
    d |= {f"{k}_eps{e:g}": maha(C[f], sc, e) for k, (f, sc) in meth.items() if k != "ellipsoid" for e in EPS_SENS}
    out = dict(err_norm=np.linalg.norm(E, axis=1)[keep], group=G[keep], config_type=np.array(T, dtype=object)[keep],
               sd_headline=lamF * sd[keep], sd_B1=qb1 * np.sqrt(np.maximum(v["post"], 0))[keep],
               sd_B2=np.sqrt(np.maximum(v["pops"], 0))[keep], sd_B3=sd[keep], lam_F=lamF, q_b1=qb1, eps=eps,
               sd_ellipsoid=lam_e * np.sqrt(np.maximum(v["ell"], 0))[keep], lam_e=lam_e, cfg=CF[keep],
               **{f"d_{k}": x for k, x in d.items()}, **{f"logdet_{k}": x for k, x in ld.items()},
               n_dropped=int((~keep).sum()))
    np.savez(RESULTS / f"force_baselines_{m}.npz", **out)
    for k in ("headline", "B1", "B2", "B3", "ellipsoid"):
        r = out["err_norm"] / out[f"sd_{k}"]
        per = {int(g): round(float(np.sqrt(np.mean(r[out["group"] == g] ** 2))), 2) for g in np.unique(out["group"])}
        cv = "  ".join(f"{np.mean(out[f'd_{k}{x}'] <= chi(3).ppf(0.9)):.3f}" for x in ("", *(f"_eps{e:g}" for e in EPS_SENS)) if f"d_{k}{x}" in out)
        print(f"{m} {k:8s} RMS |e|/sd {np.sqrt(np.mean(r ** 2)):.3f}  90 % ellipsoid coverage (eps x1 x10 x100) {cv}"
              f"  per group {per}", flush=True)


if __name__ == "__main__":
    main()
