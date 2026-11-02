"""A continuous, descriptor-learned conformal scale for per-atom forces (normalised conformal prediction: Lei et al. 2018;
Papadopoulos et al. 2008), against the geometric Mondrian groups, in distribution and on the held-out cracks and screw
dislocations of ood/make_split.py.

The CR3 shape already widens with descriptor novelty; what the groups correct is the error-to-shape ratio, the
anisotropic score s = sqrt(e^T (V + eps v/3 I)^-1 e).  Here that ratio is learned from the descriptors instead:

  phi(x)   the whitened per-species PCA of the normalised site descriptor (the support reference of the ARD posterior,
           fitted on training atoms only), features [1, phi, phi^2]
  f(x)     ridge regression of log s^2 on those features over half A of the calibration hold-out (configurations split
           at random; atoms weighted 1 / n_c so each configuration counts once; ridge by 5-fold configuration CV on A)
  q        the conformal quantile of u = s / exp(f / 2) over the other half B at the fit's alpha, atom-weighted: the
           Mondrian per-group weights (each configuration 1 per group it populates) pooled over groups give the same
           quantile to 0.01; a configuration-weighted pooled quantile is ~12 % lower and under-covers (one_group_cfgq)
  region   s <= q exp(f(x) / 2): an ellipsoid of the CR3 shape whose size varies continuously with the environment.

The finite-sample guarantee of split conformal holds for B's exchangeable configurations, as for one group.  Splits are
repeated over SEEDS; reported: coverage (mean and range over seeds), configuration-bootstrap 90 % interval for seed 0,
and the median region size q exp(f/2) against the Mondrian q_g, in the same score units.

    OOD_RUNS=... OOD_DATA=... python -m bench.si_qoi_uq.ood.learned_scale o3d12|o4d20  -> results/learned_scale_<m>.json
"""
import argparse, json, os, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "defect_uq" / "scoring"))
RUNS = pathlib.Path(os.environ.get("OOD_RUNS", "/storage/eng/essswb/projects/si-qoi-uq-paper/runs/ood_holdout"))
DATA = pathlib.Path(os.environ.get("OOD_DATA", "/storage/eng/essswb/projects/si-qoi-uq-paper/data/ood_holdout"))
SEEDS = range(5)
RIDGE = 10.0 ** np.arange(-4, 4.1, 0.5)


def scored_atoms(lin, post, calc, xyz):
    """(s, group, X, Z, cfg, type) of the force-weighted atoms of `xyz`, symmetry-zero atoms dropped."""
    import jax
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from _frames import read_atoms
    cfgs = load_configs(str(xyz), virial_key="virial")
    frames = read_atoms(str(xyz))
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    E, V, G, X, Z, C = [], [], [], [], [], []
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        nm, wF, ncfg = np.asarray(bt.node_mask), np.asarray(bt.w_F), np.asarray(bt.node_cfg)
        a = np.flatnonzero(nm & (wF > 0))
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        idx = {c: gc + j for j, c in enumerate(live)}
        E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
        V.append(post.atom_shape(F[a]))
        G.append(np.asarray(post.groups_of(bt))[a])
        for c in np.unique(ncfg[a]):
            k = idx[int(c)]
            nodes = np.flatnonzero(nm & (ncfg == c))
            Xc, Zc = calc.support_descriptors(frames[k])
            if len(Zc) != len(nodes):
                raise ValueError(f"config {k}: {len(Zc)} descriptor rows, {len(nodes)} nodes")
            loc = np.searchsorted(nodes, a[ncfg[a] == c])
            X.append(Xc[loc]); Z.append(Zc[loc]); C.append(np.full(len(loc), k))
        gc += len(live)
    E, V, G, X, Z, C = map(np.concatenate, (E, V, G, X, Z, C))
    v = np.trace(V, axis1=1, axis2=2)
    keep = np.sqrt(np.maximum(v, 0)) > 1e-3 * np.median(np.sqrt(np.maximum(v, 0)))
    E, V, G, X, Z, C, v = E[keep], V[keep], G[keep], X[keep], Z[keep], C[keep], v[keep]
    reg = V + post.eps * (v / 3)[:, None, None] * np.eye(3)
    s = np.sqrt(np.einsum("na,nab,nb->n", E, np.linalg.inv(reg), E))
    T = np.array([cfgs[c].config_type for c in C], dtype=object)
    return dict(s=s, g=G, X=X.astype(float), Z=Z.astype(int), c=C, t=T)


def features(ref, X, Z):
    from ace_jax.fit.support import _proj, support_features
    F = support_features(X, ref.get("features"))
    zs = sorted(ref["pca"])
    out = None
    for z in np.unique(Z):
        m = Z == z
        P = _proj(ref["pca"], int(z), F[m])
        if out is None:
            out = np.zeros((len(Z), P.shape[1]))
        out[m] = P
    if len(zs) > 1:
        raise NotImplementedError("one species (Si)")
    return np.concatenate([np.ones((len(Z), 1)), out, out ** 2], 1)


def cfg_weights(cfg):
    _, inv, cnt = np.unique(cfg, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def ridge(Phi, y, w, lam):
    A = Phi.T @ (w[:, None] * Phi) + lam * np.diag(np.r_[0.0, np.ones(Phi.shape[1] - 1)])   # intercept unpenalised
    return np.linalg.solve(A, Phi.T @ (w * y))


def fit_scale(Phi, y, cfg, rng):
    """Ridge of log s^2, lambda by 5-fold configuration CV (weighted squared error)."""
    w = cfg_weights(cfg)
    u = rng.permutation(np.unique(cfg))
    fold = {c: k % 5 for k, c in enumerate(u)}
    fo = np.array([fold[c] for c in cfg])
    err = []
    for lam in RIDGE:
        e = 0.0
        for k in range(5):
            tr, te = fo != k, fo == k
            b = ridge(Phi[tr], y[tr], w[tr], lam)
            e += np.sum(w[te] * (y[te] - Phi[te] @ b) ** 2)
        err.append(e)
    lam = RIDGE[int(np.argmin(err))]
    return ridge(Phi, y, w, lam), float(lam)


def boot_ci(cov, cfg, rng, n=2000):
    cs = np.unique(cfg)
    hits = np.array([cov[cfg == c].sum() for c in cs]); ns = np.array([(cfg == c).sum() for c in cs])
    d = rng.integers(0, len(cs), (n, len(cs)))
    b = hits[d].sum(1) / ns[d].sum(1)
    return [float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ..analysis.novelty_groups_si import _bounded_compiles
    _bounded_compiles()
    import support_rebuild as sr
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    from novelty_groups import quantiles
    from ..linear import LinearACE
    from ..run import RESULTS
    m = sys.argv[1]
    d = RUNS / "fits" / m
    work = RUNS / "descriptors" / m
    if not (work / "support_normalised.npz").exists():
        sr.build(argparse.Namespace(model=str(d / "model.npz"), posterior=str(d / "posterior.npz"),
                                    ard_json=str(d / "ard.json"), train=str(DATA / "train.xyz"), out=str(work),
                                    features="normalised", pca_atoms=10000))
    post = ARDPosterior.load(d / "posterior.npz")
    lin = LinearACE(d / "model.npz", d / "posterior.npz")
    calc = ACECalculator(str(d / "model.npz"), posterior=str(d / "posterior.npz"))
    ref = sr.load_ref(work / "support_normalised.npz")
    D = np.load(work / "descriptors.npz")
    tab = post.group_table
    alpha, q_g = float(tab["alpha"]), np.asarray(tab["q"], float)
    cal = post.cal
    s_c, cfg_c = np.asarray(cal["scores"], float), np.asarray(cal["cfg"])
    Phi_c = features(ref, D["X_cal"].astype(float), D["Z_cal"].astype(int))
    ok = s_c > 0
    sets = {"id_test": scored_atoms(lin, post, calc, DATA / "test.xyz"), "ood": scored_atoms(lin, post, calc, DATA / "ood.xyz")}
    for S in sets.values():
        S["Phi"] = features(ref, S["X"], S["Z"])
    q1 = float(np.quantile(s_c, 1 - alpha, method="higher"))   # one atom-weighted quantile (the pooled Mondrian weights)
    q1_cfg = float(quantiles(s_c, np.zeros(len(s_c), int), cfg_c, 1, alpha, 1, [np.inf])[0][0])
    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        u = np.unique(cfg_c)
        A = set(rng.permutation(u)[: len(u) // 2].tolist())
        inA = np.array([c in A for c in cfg_c])
        b, lam = fit_scale(Phi_c[inA & ok], np.log(s_c[inA & ok] ** 2), cfg_c[inA & ok], rng)
        uB = s_c[~inA] / np.exp(Phi_c[~inA] @ b / 2)
        q = float(np.quantile(uB, 1 - alpha, method="higher"))  # atom-weighted, as Mondrian's per-group weights pool
        q_cfg = float(quantiles(uB, np.zeros(int((~inA).sum()), int), cfg_c[~inA], 1, alpha, 1, [np.inf])[0][0])
        rec = {"seed": seed, "ridge": lam, "q": q, "q_cfgw": q_cfg}
        for name, S in sets.items():
            rad = q * np.exp(S["Phi"] @ b / 2)
            rec[name] = {"cov": S["s"] <= rad, "radius": rad}
        per_seed.append(rec)
    res = {"model": m, "alpha": alpha, "n_features": int(Phi_c.shape[1]), "ridge": [r["ridge"] for r in per_seed],
           "q_one": q1, "q_one_cfgw": q1_cfg, "q_learned": [r["q"] for r in per_seed], "sets": {}}
    rng = np.random.default_rng(0)
    for name, S in sets.items():
        base = {"mondrian": S["s"] <= q_g[S["g"]], "one_group": S["s"] <= q1, "one_group_cfgq": S["s"] <= q1_cfg}
        cols = {"learned": [r[name]["cov"] for r in per_seed]}

        def cw(c, mask):                                   # configuration-weighted: each configuration counts once
            cs = np.unique(S["c"][mask])
            return float(np.mean([c[mask & (S["c"] == q)].mean() for q in cs]))

        def summ(mask):
            o = {"atoms": int(mask.sum()), "configs": int(len(np.unique(S["c"][mask])))}
            for k, c in base.items():
                o[f"{k}_cfgw"] = cw(c, mask)
                o[k] = float(c[mask].mean())
                o[f"{k}_ci90"] = boot_ci(c[mask], S["c"][mask], rng)
            o["learned_cfgw"] = float(np.mean([cw(c, mask) for c in cols["learned"]]))
            lc = np.array([c[mask].mean() for c in cols["learned"]])
            o["learned"] = float(lc.mean()); o["learned_seed_range"] = [float(lc.min()), float(lc.max())]
            o["learned_ci90"] = boot_ci(cols["learned"][0][mask], S["c"][mask], rng)
            o["median_radius"] = {"mondrian": float(np.median(q_g[S["g"]][mask])), "one_group": q1,
                                  "learned": float(np.mean([np.median(r[name]["radius"][mask]) for r in per_seed]))}
            return o
        out = {"all": summ(np.ones(len(S["s"]), bool)),
               "per_group": {int(g): summ(S["g"] == g) for g in np.unique(S["g"])}}
        if name == "ood":
            out["per_type"] = {t: summ(S["t"] == t) for t in np.unique(S["t"])}
        res["sets"][name] = out
    json.dump(res, open(RESULTS / f"learned_scale_{m}.json", "w"), indent=1)
    np.savez(RUNS / "descriptors" / m / "learned_atoms.npz", **{f"{n}_{k}": S[k] for n, S in sets.items()
                                                               for k in ("s", "g", "c", "t", "Phi")},
             **{f"{n}_radius_seed{r['seed']}": r[n]["radius"] for n in sets for r in per_seed}, q_g=q_g, q_one=q1)
    for name, o in res["sets"].items():
        print(m, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o["all"].items()})
        for t, v in o.get("per_type", {}).items():
            print("   ", t, {k: round(v[k], 3) for k in ("mondrian", "one_group", "learned")})


if __name__ == "__main__":
    main()
