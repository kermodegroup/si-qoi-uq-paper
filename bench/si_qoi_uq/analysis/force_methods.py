"""Per-atom forces of the propagation's four methods (headline CR3 x lam_F, B1 posterior x q_b1, B2 POPS box at the 50 %
leverage cut, B3 CR3 at lam = 1), in distribution and out of distribution, for the summary figure.

Per atom a with force rows F_a (3, p) and error e = y_F - F_a theta_hat, each method's 3x3 covariance is
C_a = s^2 F_a L L^T F_a^T (L and s as analysis/force_baselines.py), regularised as there, C_a + eps tr(C_a)/3 I, and scored by
  coverage   d^2 = e^T C_a^-1 e <= chi2_3(0.9)            (the Gaussian ellipsoid of the method's own covariance)
  log score  log N(e; 0, C_a) = -d^2/2 - log|2 pi C_a|/2  (nats; averaged over atoms; the B2 box is read as a Gaussian of
                                                           the box's covariance)
Symmetry-zero atoms (CR3 sd < 1e-3 of the median) are dropped, as everywhere.

  id   the benchmark BLR fits (SIQ_RUNS, as fig. 7) on the test split; lam_F and q_b1 from diag/<m>_scales.json, fitted
       on that test split, so the headline and B1 are in sample there (B2, B3 have no fitted scale).
  ood  the BLR refits without cracks and screw dislocations (OOD_RUNS/fits/<m>: the CR3 factor R of its posterior.npz,
       the posterior factor from ood/posterior_chol.py; POPS box from pops_fit.py run on the same reduced training set, OOD_RUNS/fits/<m>/pops),
       lam_F and q_b1 refitted by the same rules (diagnostics.scales, conformal_q) on the refit's in-distribution test
       split, then scored on the held-out cracks and dislocations (out of sample for every method).

    SIQ_RUNS=<BLR root> OOD_RUNS=<ood_holdout_blr> python -m bench.si_qoi_uq.analysis.force_methods o3d12|o4d20
        -> results/force_methods_<m>.json
"""
import json, os, pathlib, sys

import numpy as np

OOD_RUNS = pathlib.Path(os.environ.get("OOD_RUNS", "/storage/eng/essswb/projects/si-qoi-uq-paper/runs/ood_holdout_blr"))
OOD_DATA = pathlib.Path(os.environ.get("OOD_DATA", "/storage/eng/essswb/projects/si-qoi-uq-paper/data/ood_holdout"))
METHODS = ("headline", "B1", "B2", "B3")
LEVEL = 0.9


def pops_factor(d):
    b = np.load(d / "pops_box_lev50.npz")          # POPSRegression default (leverage_percentile 50)
    return np.asarray(b["support"]) * ((np.asarray(b["hi"]) - np.asarray(b["lo"])) / np.sqrt(12))[None, :]


def atom_scores(lin, facs, scale, xyz, eps):
    """Per force-weighted atom of `xyz`: d^2 and log|2 pi C| per method, config index and type."""
    import jax
    import jax.numpy as jnp
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    cfgs = load_configs(str(xyz), virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    L = {k: jnp.asarray(v) for k, v in facs.items()}
    E, Cc, T, C = [], [], [], {k: [] for k in L}
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        nm, wF = np.asarray(bt.node_mask), np.asarray(bt.w_F)
        a = np.flatnonzero(nm & (wF > 0))
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        idx = {c: gc + j for j, c in enumerate(live)}
        E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
        ci = [idx[int(c)] for c in np.asarray(bt.node_cfg)[a]]
        Cc += ci
        T += [cfgs[c].config_type for c in ci]
        gc += len(live)
        Fa = jnp.asarray(F[a])
        for k, Lk in L.items():
            M = Fa[..., :Lk.shape[0]] @ Lk                             # POPS has no E0 column (zero on force rows)
            C[k].append(np.asarray(M @ jnp.swapaxes(M, 1, 2)))
    E = np.concatenate(E)
    C = {k: np.concatenate(v) for k, v in C.items()}
    sd = np.sqrt(np.maximum(np.trace(C["cr3"], axis1=1, axis2=2), 0))
    keep = sd > 1e-3 * np.median(sd)
    out = {"cfg": np.asarray(Cc)[keep], "type": np.array(T, dtype=object)[keep], "n_dropped": int((~keep).sum())}
    meth = {"headline": ("cr3", scale["lam_F"]), "B1": ("post", scale["q_b1"]), "B2": ("pops", 1.0), "B3": ("cr3", 1.0)}
    for k, (f, s) in meth.items():
        Ca = C[f][keep] * s ** 2
        reg = Ca + eps * (np.trace(Ca, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
        out[f"d2_{k}"] = np.einsum("na,nab,nb->n", E[keep], np.linalg.inv(reg), E[keep])
        out[f"logdet_{k}"] = np.linalg.slogdet(2 * np.pi * reg)[1]
    return out


def summary(sc, mask=None):
    from scipy.stats import chi2
    mask = np.ones(len(sc["cfg"]), bool) if mask is None else mask
    k2 = chi2(3).ppf(LEVEL)
    r = {"atoms": int(mask.sum()), "configs": int(len(np.unique(sc["cfg"][mask])))}
    for k in METHODS:
        d2, ld = sc[f"d2_{k}"][mask], sc[f"logdet_{k}"][mask]
        r[k] = {"coverage": float(np.mean(d2 <= k2)), "log_score": float(np.mean(-0.5 * d2 - 0.5 * ld)),
                "rms_d_over_sqrt3": float(np.sqrt(np.mean(d2) / 3))}
    return r


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.ard import ARDPosterior
    from ..diagnostics import TEST_XYZ, conformal_q, scales, test_scores
    from ..harness import RUNS
    from ..linear import LinearACE
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    res = {"model": m, "level": LEVEL}
    # in distribution: the benchmark fits
    lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
    scale = {"lam_F": float(sc["lam_cr3"]["F"]), "q_b1": float(sc["q_b1"])}
    eps = float(ARDPosterior.load(fit_dir(m, "cr3") / "posterior.npz").eps)
    facs = {"cr3": lin.factor("cr3"), "post": linp.factor("posterior"), "pops": pops_factor(fit_dir(m, "pops"))}
    s = atom_scores(lin, facs, scale, TEST_XYZ, eps)
    res["id"] = {"scale": scale, "eps": eps, "dropped": s["n_dropped"], "all": summary(s)}
    print(m, "id", json.dumps(res["id"]["all"]), flush=True)
    # out of distribution: the refits on the reduced training set
    d = OOD_RUNS / "fits" / m
    if not (d / "pops" / "pops_box_lev50.npz").exists():
        print(m, "no OOD POPS box yet: in-distribution part only", flush=True)
        json.dump(res, open(RESULTS / f"force_methods_{m}.json", "w"), indent=1)
        return
    lo = LinearACE(d / "model.npz", d / "posterior.npz")
    eps_o = float(ARDPosterior.load(d / "posterior.npz").eps)
    import scipy.linalg
    pc = np.load(d / "posterior_chol.npz")                       # ood/posterior_chol.py: A = L L^T at the refit's MAP
    if float(pc["mean_rel_dev"]) > 1e-6:
        raise ValueError(f"{m}: posterior_chol mean differs from the refit's ({float(pc['mean_rel_dev']):.1e})")
    Lpost = scipy.linalg.solve_triangular(pc["L"], np.eye(len(pc["L"])), lower=True).T   # A^-1 = L^-T L^-1
    fo = {"cr3": lo.factor("cr3"), "post": Lpost, "pops": pops_factor(d / "pops")}
    r = test_scores(lo, {"cr3": fo["cr3"], "posterior": fo["post"]}, xyz=str(OOD_DATA / "test.xyz"), log=lambda *_: None)
    scale_o = {"lam_F": float(scales(r, "cr3")[1]["F"]), "q_b1": float(conformal_q(r, "posterior"))}
    s_id = atom_scores(lo, fo, scale_o, OOD_DATA / "test.xyz", eps_o)
    s_o = atom_scores(lo, fo, scale_o, OOD_DATA / "ood.xyz", eps_o)
    res["ood"] = {"scale": scale_o, "eps": eps_o, "dropped": s_o["n_dropped"], "id_test": summary(s_id), "all": summary(s_o),
                  "per_type": {t: summary(s_o, s_o["type"] == t) for t in np.unique(s_o["type"])}}
    print(m, "ood", json.dumps({k: res["ood"][k] for k in ("scale", "id_test", "all")}), flush=True)
    json.dump(res, open(RESULTS / f"force_methods_{m}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
