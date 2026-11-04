"""Out-of-distribution check of the per-atom force calibration (paper 3.2): ARD refits without GAP-18's cracks and screw
dislocations (ood/make_split.py, runs/ood_holdout/fits/<m>), scored on those held-out configurations.

Per atom, as analysis/force_calibration.py: V the unscaled 3x3 CR3 shape, v = tr V, the anisotropic conformal score
s = sqrt(e^T (V + eps v/3 I)^-1 e), symmetry-zero atoms (sqrt v < 1e-3 median) dropped.  Coverage of the 90 % region by
  mondrian   s <= q_g (the refit's served groups and quantiles: training hold-out)
  one_group  s <= q_1, one configuration-weighted quantile of the same hold-out scores
  global     s <= lam chi_3^-1(0.9), lam^2 = mean |e|^2 / v over the in-distribution test atoms of the refit (a global
             scale fitted out of sample for the OOD set): the Gaussian ellipsoid of the same shape, so every method's region
             is the same kind (an isotropic ball |e| <= lam sqrt(v chi2_3/3) is larger along the shape's thin directions)
overall, per held-out type and per Mondrian group; 90 % intervals from a bootstrap over configurations (2000 draws).

    OOD_RUNS=<runs/ood_holdout> python -m bench.si_qoi_uq.ood.score o3d12|o4d20  -> results/ood_mondrian_<m>.json
"""
import json, os, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "defect_uq" / "scoring"))
RUNS = pathlib.Path(os.environ.get("OOD_RUNS", "/storage/eng/essswb/projects/si-qoi-uq-paper/runs/ood_holdout"))
DATA = pathlib.Path(os.environ.get("OOD_DATA", "/storage/eng/essswb/projects/si-qoi-uq-paper/data/ood_holdout"))


def atoms(lin, post, xyz):
    """(e (n,3), V (n,3,3), group, config index, config_type) of the force-weighted atoms of `xyz`."""
    import jax
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    cfgs = load_configs(str(xyz), virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    E, V, G, C = [], [], [], []
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        nm, wF = np.asarray(bt.node_mask), np.asarray(bt.w_F)
        a = np.flatnonzero(nm & (wF > 0))
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        idx = {c: gc + j for j, c in enumerate(live)}
        E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
        V.append(post.atom_shape(F[a]))
        G.append(np.asarray(post.groups_of(bt))[a])
        C.append(np.array([idx[int(c)] for c in np.asarray(bt.node_cfg)[a]]))
        gc += len(live)
    E, V, G, C = map(np.concatenate, (E, V, G, C))
    T = np.array([cfgs[c].config_type for c in C], dtype=object)
    return E, V, G, C, T


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from scipy.stats import chi
    from ace_jax.fit.ard import ARDPosterior
    from novelty_groups import quantiles
    from ..linear import LinearACE
    from ..run import RESULTS
    m = sys.argv[1]
    d = RUNS / "fits" / m
    lin, post = LinearACE(d / "model.npz", d / "posterior.npz"), ARDPosterior.load(d / "posterior.npz")
    tab = post.group_table
    alpha, q_g = float(tab["alpha"]), np.asarray(tab["q"], float)
    cal = post.cal
    q1 = float(quantiles(np.asarray(cal["scores"], float), np.zeros(len(cal["scores"]), int), np.asarray(cal["cfg"]),
                         1, alpha, 1, [np.inf])[0][0])

    def scored(xyz):
        E, V, G, C, T = atoms(lin, post, xyz)
        v = np.trace(V, axis1=1, axis2=2)
        k = np.sqrt(np.maximum(v, 0)) > 1e-3 * np.median(np.sqrt(np.maximum(v, 0)))
        E, V, G, C, T, v = E[k], V[k], G[k], C[k], T[k], v[k]
        reg = V + post.eps * (v / 3)[:, None, None] * np.eye(3)
        s = np.sqrt(np.einsum("na,nab,nb->n", E, np.linalg.inv(reg), E))
        return dict(e=np.linalg.norm(E, axis=1), v=v, s=s, g=G, c=C, t=T, dropped=int((~k).sum()))
    idt = scored(DATA / "test.xyz")
    lam = float(np.sqrt(np.mean(idt["e"] ** 2 / idt["v"])))
    ood = scored(DATA / "ood.xyz")
    k3 = chi(3).ppf(1 - alpha)
    cov = {"mondrian": ood["s"] <= q_g[ood["g"]], "one_group": ood["s"] <= q1,
           "global": ood["s"] <= k3 * lam}
    cov_id = {"mondrian": idt["s"] <= q_g[idt["g"]], "one_group": idt["s"] <= q1,
              "global": idt["s"] <= k3 * lam}
    rng = np.random.default_rng(0)
    cfgs = np.unique(ood["c"])

    def summary(mask):
        out = {"atoms": int(mask.sum()), "configs": int(len(np.unique(ood["c"][mask])))}
        cs = np.unique(ood["c"][mask])
        for k, c in cov.items():
            out[k] = float(c[mask].mean()) if mask.any() else None
            if len(cs) > 1:                                    # bootstrap over configurations, atoms pooled per draw
                per = {cc: (c[mask & (ood["c"] == cc)].sum(), (mask & (ood["c"] == cc)).sum()) for cc in cs}
                hits = np.array([per[cc][0] for cc in cs]); ns = np.array([per[cc][1] for cc in cs])
                draws = rng.integers(0, len(cs), (2000, len(cs)))
                b = hits[draws].sum(1) / ns[draws].sum(1)
                out[f"{k}_ci90"] = [float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]
        return out
    res = {"model": m, "alpha": alpha, "q_one": q1, "q_g": q_g.tolist(), "lam_id_test": lam, "n_ood_configs": int(len(cfgs)),
           "dropped": ood["dropped"],
           "in_distribution": {k: float(c.mean()) for k, c in cov_id.items()},
           "all": summary(np.ones(len(ood["s"]), bool)),
           "per_type": {t: summary(ood["t"] == t) for t in np.unique(ood["t"])},
           "per_group": {int(g): summary(ood["g"] == g) for g in np.unique(ood["g"])},
           "group_share": {int(g): float(np.mean(ood["g"] == g)) for g in np.unique(ood["g"])},
           "rms_ratio": {"mondrian": float(np.sqrt(np.mean((ood["e"] / (np.asarray(tab["lam_rms"], float)[ood["g"]]
                                                                        * np.sqrt(ood["v"]))) ** 2))),
                         "global": float(np.sqrt(np.mean(ood["e"] ** 2 / (lam ** 2 * ood["v"]))))}}
    json.dump(res, open(RESULTS / f"ood_mondrian_{m}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("in_distribution", "all", "per_type", "rms_ratio")}, indent=1))


if __name__ == "__main__":
    main()
