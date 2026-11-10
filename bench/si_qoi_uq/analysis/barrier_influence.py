"""Which training configurations drive the CR3 sigma of the vacancy migration barrier?  (next-steps plan §3)

CR3 is the centred delete-one-configuration jackknife: Sigma = sum_k l_k l_k^T with l_k = D^-1 S^-1 D^-1 (g~_k - g-bar)
(fit.jackknife.shape_factor, mapped to the model's coordinates by dinv as LinearACE.factor), one column per
training configuration.  With g the barrier's delta row (analysis/vacancy_barrier.py), the barrier's B3 variance is
sum_k b_k^2, b_k = g . l_k: b_k is configuration k's (leave-one-out, leverage-corrected) influence on the barrier.
If CR3 sees the barrier's misfit through the right data, the largest |b_k| belong to vacancy cells near the hop
(large hop coordinate s, as in the 2026-11-10 hop check: 0 at the minimum, 0.5 at the saddle).

o4d20: K = 2227 < L, so the stored R is the uncompressed matrix of columns (shape_factor returns Q~ itself); o3d12
(L = 224 < K) is replayed (blr_cr3.py's problem, MAP hyperparameters and press_scores) to get the columns.  The
cluster -> configuration order (row_clusters with ell = inf: the k-th live configuration in batch order) is
replayed for both.  Check: sum_k b_k^2 equals the B3 variance from the stored factor.

Also for any QoI with a delta row in RUNS/delta/<m>.npz (the defect formation energies: do their CR3 sigmas come
from the matching defect cells?), and two non-defect controls.  The per-configuration columns are cached
(RUNS/delta/<m>_cr3_columns.npy) so the o3d12 replay runs once.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.barrier_influence o3d12|o4d20 -> results/barrier_influence.json
"""
QOIS = ("vacancy_barrier", "vacancy_energy", "dumbbell_interstitial_energy", "hexagonal_interstitial_energy",
        "tetrahedral_interstitial_energy", "diamond_c44", "surface_100 (J/m2)")
import json, pathlib, shlex, sys

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")


def hop_s(c):
    """Largest projection, onto the bond to the vacant site, of a vacant-site neighbour's displacement from its own
    site, in units of the bond (one-to-one site assignment, drift removed).  63-atom 2x2x2 cells only."""
    from ase.build import bulk
    from scipy.optimize import linear_sum_assignment
    cell, pos = np.asarray(c.cell, float), np.asarray(c.positions, float)
    f = bulk("Si", "diamond", a=1.0, cubic=True).get_scaled_positions()
    g = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    S = ((f[None] + g[:, None]).reshape(-1, 3) / 2) @ cell
    inv = np.linalg.inv(cell)

    def mic(d):
        x = d @ inv
        return (x - np.round(x)) @ cell
    for _ in range(3):
        dv = mic(pos[:, None] - S[None])
        _, own = linear_sum_assignment(np.linalg.norm(dv, axis=2) ** 2)
        pos = pos - np.mean(dv[np.arange(len(pos)), own], 0)
    v = S[np.setdiff1d(np.arange(64), own)[0]]
    best = 0.0
    for i in range(len(pos)):
        u = mic(v - S[own[i]])
        L = np.linalg.norm(u)
        if L < 2.6:
            best = max(best, float(mic(pos[i] - S[own[i]]) @ u / L / L))
    return best


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve
    from ace_jax import cli
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.pipeline import load_fit_data
    from ..harness import RUNS
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    lin = lin_for(m, "cr3")
    argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
    argv[argv.index("--out") + 1] = "/tmp/unused"
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test)
    rc, K = row_clusters(data.ds_train, data.train, np.inf)
    order = []                                                     # cluster k -> data.train index (batch order)
    gc = 0
    for i in range(data.ds_train.n_batches):
        for _ in np.flatnonzero(np.asarray(data.ds_train.cfg_mask[i])):
            order.append(gc)
            gc += 1
    assert len(order) == K
    n_rows = np.zeros(K, int)                                      # atoms per cluster from the F-row ids
    for r in rc:
        ids = r["F"][r["F"] >= 0]
        np.add.at(n_rows, ids, 1)
    if not np.array_equal(n_rows, [len(data.train[i].numbers) for i in order]):
        raise ValueError("cluster -> configuration order does not match atom counts")
    dinv = np.asarray(lin.post.dinv, np.float64)
    cache = RUNS / "delta" / f"{m}_cr3_columns.npy"
    if cache.exists():
        Lk = np.load(cache)
        src = f"cached columns ({cache.name})"
    elif lin.post.R is not None and np.shape(lin.post.R)[1] == K:
        Lk = dinv[:, None] * np.asarray(lin.post.R, np.float64)     # stored uncompressed: one column per cluster
        src = "stored R (uncompressed)"
    else:
        from ace_jax.fit.ard import ARDPosterior
        from ace_jax.fit.hypers import Hypers
        from ace_jax.fit.jackknife import press_scores
        from ace_jax.fit.pipeline.problem import build_problem
        prob = build_problem(cfg, data).prob
        theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
        post = ARDPosterior.load(fit_dir(m, "cr3") / "posterior.npz")
        z = np.load(fit_dir(m, "cr3") / "posterior.npz")
        n = len(dinv)
        Lc = np.zeros((n, n))
        Lc[np.tril_indices(n)] = z["chol_packed"]
        post = post._replace(chol=jnp.asarray(Lc), R=None)
        sig = np.exp([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
        G, _ = press_scores(post, prob, data.ds_train, rc, K, sig, mode="exact")
        G = jnp.asarray(G, jnp.float64)
        Qt = cho_solve((jnp.asarray(Lc), True), post.prior_root.dual(G - G.mean(1, keepdims=True)))
        Lk = dinv[:, None] * np.asarray(Qt)
        src = "replayed press_scores"
        np.save(cache, Lk)
    cfgs = [data.train[i] for i in order]
    ctype = np.array([str(c.config_type) for c in cfgs])
    nat = np.array([len(c.numbers) for c in cfgs])
    s = np.full(K, np.nan)
    for k, c in enumerate(cfgs):
        if ctype[k] == "vacancy" and nat[k] == 63:
            s[k] = hop_s(c)
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    rows = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
    rows["vacancy_barrier"] = np.load(RUNS / "delta" / f"{m}_cr3_vacancy_barrier_row.npy")
    Lc = np.asarray(lin.factor("cr3"))
    out = {}
    for q in QOIS:
        g = rows[q]
        b = Lk.T @ g[:Lk.shape[0]]
        b2 = b ** 2
        tot = b2.sum()
        top = np.argsort(-b2)
        res = {"source": src, "K": int(K), "ratio_check": float(tot / np.sum((Lc.T @ g) ** 2)),
               "top20": [{"k": int(k), "config_type": ctype[k], "n_atoms": int(nat[k]), "b": float(b[k]),
                          "share": float(b2[k] / tot), "hop_s": None if np.isnan(s[k]) else float(s[k])} for k in top[:20]],
               "n_for_half": int(np.searchsorted(np.cumsum(b2[top]) / tot, 0.5) + 1),
               "n_for_90": int(np.searchsorted(np.cumsum(b2[top]) / tot, 0.9) + 1)}
        share = {}
        for t in np.unique(ctype):
            i = ctype == t
            share[t] = {"n": int(i.sum()), "share": float(b2[i].sum() / tot),
                        "enrichment": float(b2[i].sum() / tot / (i.sum() / K))}   # share / share of configurations
        res["share_by_type"] = dict(sorted(share.items(), key=lambda kv: -kv[1]["share"]))
        if q == "vacancy_barrier":
            v63 = ~np.isnan(s)
            bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
            res["vacancy63_by_s"] = [{"s": f"{lo}-{hi}", "n": int(np.sum(v63 & (s >= lo) & (s < hi))),
                                      "share": float(b2[v63 & (s >= lo) & (s < hi)].sum() / tot)}
                                     for lo, hi in zip(bins[:-1], bins[1:])]
        out[q] = res
        print(f"{m} {q}: check {res['ratio_check']:.3f}  configs for 50 / 90 %: {res['n_for_half']} / {res['n_for_90']}  "
              + "  ".join(f"{t} {v['share']:.2f} (x{v['enrichment']:.0f})" for t, v in list(res["share_by_type"].items())[:5]),
              flush=True)
    f = RESULTS / "barrier_influence.json"
    old = json.load(open(f)) if f.exists() else {}
    json.dump(old | {m: out}, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
