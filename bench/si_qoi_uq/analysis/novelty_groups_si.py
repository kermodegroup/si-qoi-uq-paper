"""Descriptor-novelty Mondrian groups on silicon: the CALM-comparison Step 3 test (bench/defect_uq/results/
2026-10-05_calm_comparison.md) rerun on the Si ARD fits, scored on the test split of the paper's per-atom table.

For each model (an `aj fit --uq ard` posterior: runs/si_qoi_uq_ard137):
1. `support_rebuild.build`: raw descriptors of the posterior's calibration atoms (T_val, cal order) and of a
   10 000-atom training pool, and the normalised support reference (whitened per-species PCA);
2. `novelty_groups.fit_novelty`: a CALM-like grade gamma per atom, fitted on the training pool only (label-free);
3. on the calibration atoms, the configuration-weighted pooled quantile q per gamma bin ("novelty") and per
   (distortion group x gamma bin) ("crossed", falling back to the served group q below effective_n_min configs);
4. on the test atoms (the same atoms, shape, scores and symmetry-zero rule as analysis/force_calibration.py),
   coverage of s <= q with the served groups, the novelty groups and the crossed groups, overall and per gamma bin.

    PYTHONPATH=<pr137 ace-jax src> SIQ_RUNS=runs/si_qoi_uq_ard137 python -m bench.si_qoi_uq.analysis.novelty_groups_si o3d12|o4d20
        -> results/novelty_groups_<m>.json
"""
import argparse, json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "defect_uq" / "scoring"))
TRAIN = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/train.xyz"
LABELS = ("<=1", "1-2", ">2")


def test_atoms(lin, post, calc):
    """(s, g, X, Z, cfg) of the test atoms kept by force_calibration.py, in its order."""
    import jax
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..diagnostics import TEST_XYZ
    from _frames import read_atoms
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    frames = read_atoms(TEST_XYZ)
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    E, V, G, X, Z, C = [], [], [], [], [], []
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        nm, wF = np.asarray(bt.node_mask), np.asarray(bt.w_F)
        a = np.flatnonzero(nm & (wF > 0))
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        idx = {c: gc + j for j, c in enumerate(live)}
        ncfg = np.asarray(bt.node_cfg)
        E.append(np.asarray(bt.y_F)[a] - F[a] @ th)
        V.append(post.atom_shape(F[a]))
        G.append(np.asarray(post.groups_of(bt))[a])
        for c in np.unique(ncfg[a]):
            k = idx[int(c)]
            nodes = np.flatnonzero(nm & (ncfg == c))                  # this configuration's nodes, atom order
            Xc, Zc = calc.support_descriptors(frames[k])
            if len(Zc) != len(nodes):
                raise ValueError(f"test config {k}: {len(Zc)} descriptor rows, {len(nodes)} nodes")
            loc = np.searchsorted(nodes, a[ncfg[a] == c])
            X.append(Xc[loc]); Z.append(Zc[loc]); C.append(np.full(len(loc), k))
        gc += len(live)
    E, V, G = np.concatenate(E), np.concatenate(V), np.concatenate(G)
    X, Z, C = np.concatenate(X), np.concatenate(Z), np.concatenate(C)
    v = np.trace(V, axis1=1, axis2=2)
    keep = np.sqrt(np.maximum(v, 0)) > 1e-3 * np.median(np.sqrt(np.maximum(v, 0)))
    E, V, v = E[keep], V[keep], v[keep]                                   # drop symmetry-zero atoms before inverting
    reg = V + post.eps * (v / 3)[:, None, None] * np.eye(3)
    s = np.sqrt(np.einsum("na,nab,nb->n", E, np.linalg.inv(reg), E))
    return s, G[keep], X[keep].astype(float), Z[keep].astype(int), C[keep]


def _bounded_compiles(every=50):
    """support_descriptors compiles once per configuration size and keeps every executable: thousands of them exhaust
    vm.max_map_count (65530 on the compute nodes; LLVM 'Cannot allocate memory').  Clear JAX's caches every `every` calls."""
    import jax
    from ace_jax.calc.point import ACECalculator
    orig, n = ACECalculator.support_descriptors, [0]

    def wrapped(self, atoms):
        n[0] += 1
        if n[0] % every == 0:
            jax.clear_caches()
        return orig(self, atoms)
    ACECalculator.support_descriptors = wrapped


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    _bounded_compiles()
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.conformal import effective_n_min
    from ace_jax.fit.support import _proj, support_features
    import support_rebuild as sr
    from novelty_groups import binned, fit_novelty, gamma_of, grade, quantiles
    from ..harness import RUNS
    from ..linear import LinearACE
    from ..run import RESULTS
    m = sys.argv[1]
    d = RUNS / "fits" / m / "cr3"
    work = RUNS / "novelty" / m
    if not (work / "support_normalised.npz").exists():
        sr.build(argparse.Namespace(model=str(d / "model.npz"), posterior=str(d / "posterior.npz"),
                                    ard_json=str(d / "ard.json"), train=TRAIN, out=str(work), features="normalised",
                                    pca_atoms=10000))
    post = ARDPosterior.load(d / "posterior.npz")
    lin = LinearACE(d / "model.npz", d / "posterior.npz")
    calc = ACECalculator(str(d / "model.npz"), posterior=str(d / "posterior.npz"))
    tab = post.group_table
    alpha, n_min = float(tab["alpha"]), int(tab["n_min"])
    q_served = np.asarray(tab["q"], float)
    G0, nb = len(q_served), 3
    D = np.load(work / "descriptors.npz")
    ref = sr.load_ref(work / "support_normalised.npz")
    nov, info = {}, {}
    for k in D.files:
        if k.startswith("X_pool_"):
            z = int(k.split("_")[-1])
            F = _proj(ref["pca"], z, support_features(D[k].astype(float), ref["features"]))
            nov[z] = fit_novelty(F)
            info[z] = {"train_atoms": len(F), "D": int(F.shape[1]), "K": len(nov[z]),
                       "train_frac_gamma_gt1": float(np.mean(grade(nov[z], F) > 1))}
    cal = post.cal
    s_c, g_c, cfg_c = np.asarray(cal["scores"], float), np.asarray(cal["groups"], int), np.asarray(cal["cfg"])
    b_c = binned(gamma_of(nov, ref, D["X_cal"].astype(float), D["Z_cal"].astype(int)))
    q_all, _ = quantiles(s_c, np.zeros(len(s_c), int), cfg_c, 1, alpha, 1, [np.inf])
    q_nov, n_nov = quantiles(s_c, b_c, cfg_c, nb, alpha, n_min, np.full(nb, q_all[0]))
    q_x, n_x = quantiles(s_c, g_c * nb + b_c, cfg_c, G0 * nb, alpha, n_min, np.repeat(q_served, nb))
    s, g, X, Z, cfg = test_atoms(lin, post, calc)
    b = binned(gamma_of(nov, ref, X, Z))
    cov = {"served": s <= q_served[g], "novelty": s <= q_nov[b], "crossed": s <= q_x[g * nb + b]}
    res = {"model": m, "species": info, "alpha": alpha, "n_test_atoms": int(len(s)), "n_test_configs": int(len(np.unique(cfg))),
           "cal_frac_per_bin": [float(np.mean(b_c == i)) for i in range(nb)], "q_novelty": q_nov.tolist(),
           "n_cfg_novelty": n_nov.tolist(), "q_served": q_served.tolist(),
           "crossed_with_own_q": int(np.sum(n_x >= effective_n_min(n_min, alpha))), "n_crossed": G0 * nb, "bins": {}}
    for i, lab in [(None, "all")] + list(enumerate(LABELS)):
        mk = np.ones(len(s), bool) if i is None else b == i
        res["bins"][lab] = {"atoms": int(mk.sum()), "share": float(mk.mean()),
                            **{k: float(np.mean(c[mk])) if mk.any() else None for k, c in cov.items()}}
    # per served group, for the crossed scheme's redistribution
    res["per_group"] = {int(k): {kk: float(np.mean(c[g == k])) for kk, c in cov.items()} | {"atoms": int(np.sum(g == k))}
                        for k in np.unique(g)}
    json.dump(res, open(RESULTS / f"novelty_groups_{m}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("species", "cal_frac_per_bin", "q_novelty", "q_served", "bins")}, indent=1))


if __name__ == "__main__":
    main()
