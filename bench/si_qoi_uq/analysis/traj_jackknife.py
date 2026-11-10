"""CR3 with interstitial trajectories as clusters (next-steps plan §3: an intermediate cluster size for the interstitials).

The interstitial errors are correlated within an MD trajectory, and a delete-one-configuration jackknife treats the 103
snapshots as independent.  The extxyz carries no trajectory tag, but snapshots of one fixed-cell run share their cell
exactly: here a cluster is (config_type, cell to 1e-6 A) for the interstitial type (o3d12 / o4d20 training: groups of
19, 20 and 21 plus 43 single variable-cell snapshots), and every other configuration stays its own cluster, as in CR3.

Exact leave-group-out, leverage-corrected as press_scores (the CR3 score, at group level):
    u_G = A^-1 G^T (I - G A^-1 G^T)^-1 r_G = theta_hat - theta_(-G)   (the sign of shape_factor's columns),   G = the group's QR-compressed weighted rows [R_q / sigma_q]_q,
    r_G = [c_q / sigma_q] - G theta_hat,
with A = D S D, S = C C^T the stored posterior factor (ARD coordinates, D = diag(1 / dinv)): W = C^-1 D^-1 G^T, so
G A^-1 G^T = W^T W and A^-1 G^T = D^-1 C^-T W (triangular solves only; A is never formed).
Check: two ordinary configurations computed this way must give u_k - l_k equal for both (l_k the cached, centred CR3
column of RUNS/delta/<m>_cr3_columns.npy, u_k the uncentred shift): the constant is the CR3 centring vector, so the new
columns combine with the cached ones exactly.  Then the columns of the grouped configurations are replaced by their group
columns, all re-centred; lam_F is recalibrated on the test split and every delta QoI's sigma and z are compared with
CR3 (same scale rule).

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.traj_jackknife o3d12|o4d20 -> results/traj_jackknife_<m>.json
"""
import json, pathlib, shlex, sys

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")
GROUP_TYPES = ("interstitial",)
Z95 = 1.959963984540054


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import scipy.linalg as sl
    from ace_jax import cli
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.stats import linear_qr_statistics
    from ..diagnostics import scales, test_scores
    from ..harness import RUNS
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    lin = lin_for(m, "cr3")
    argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
    argv[argv.index("--out") + 1] = "/tmp/unused"
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test)
    prob = build_problem(cfg, data).prob
    theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
    sig = {q: float(np.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
    z = np.load(fit_dir(m, "cr3") / "posterior.npz")
    dinv = np.asarray(z["dinv"], np.float64)
    n = len(dinv)
    C = np.zeros((n, n))
    C[np.tril_indices(n)] = z["chol_packed"]
    th = np.asarray(lin.theta_hat, np.float64)

    def shift(idx):
        """Uncentred leave-group-out shift (n,) for training configurations idx (exact, leverage-corrected)."""
        sub = [data.train[i] for i in idx]
        ds = build_dataset(sub, data.meta, data.E0, cfg.batch, log=lambda *_: None)
        qs = linear_qr_statistics(prob.model, prob.cfg, ds)
        G = np.concatenate([np.asarray(getattr(qs, f"R_{q}")) / sig[q] for q in "EFV"], 0)
        c = np.concatenate([np.asarray(getattr(qs, f"c_{q}")) / sig[q] for q in "EFV"])
        Q, R = np.linalg.qr(G)                                     # compress the stack to <= n rows
        G, c = R, Q.T @ c
        r = c - G @ th
        W = sl.solve_triangular(C, (dinv[:, None] * G.T), lower=True)            # C^-1 D^-1 G^T
        x = np.linalg.solve(np.eye(W.shape[1]) - W.T @ W, r)
        return dinv * sl.solve_triangular(C, W @ x, lower=True, trans="T")        # +D^-1 C^-T W x: the score sign of
                                                                   # shape_factor's columns (A^-1 g~ = theta_hat - theta_(-G))
    cache = RUNS / "delta" / f"{m}_cr3_columns.npy"                # o3d12: replayed (barrier_influence.py)
    Lk = np.load(cache) if cache.exists() else np.asarray(lin.factor("cr3"))   # o4d20: stored R is uncompressed
    K = Lk.shape[1]
    assert K == len(data.train), f"{K} columns for {len(data.train)} training configurations"
    ctype = [str(c.config_type) for c in data.train]
    plain = [k for k in range(K) if ctype[k] not in GROUP_TYPES][:2]
    v = [shift([k]) - Lk[:, k] for k in plain]
    check = float(np.linalg.norm(v[0] - v[1]) / max(np.linalg.norm(v[0]), 1e-300))
    print(f"{m}: centring-vector check (two ordinary configurations) rel. diff {check:.2e}", flush=True)
    groups = {}
    for k in range(K):
        if ctype[k] in GROUP_TYPES:
            key = (ctype[k], tuple(np.round(np.asarray(data.train[k].cell, float).ravel(), 6)))
            groups.setdefault(key, []).append(k)
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    print(f"{m}: {sum(sizes)} configurations of {GROUP_TYPES} in {len(groups)} clusters, sizes {sizes[:6]} ... "
          f"({sum(s == 1 for s in sizes)} singletons)", flush=True)
    U = Lk + v[0][:, None]                                         # uncentred CR3 columns
    grouped = [k for g in groups.values() for k in g]
    keep = [k for k in range(K) if k not in set(grouped)]
    cols = [U[:, keep]] + [shift(g)[:, None] for g in groups.values() if len(g) > 1] \
        + [U[:, g] for g in groups.values() if len(g) == 1]
    Ut = np.concatenate(cols, 1)
    Lt = Ut - Ut.mean(1, keepdims=True)
    print(f"{m}: {Lt.shape[1]} clusters (CR3: {K})", flush=True)
    fac = {"cr3": Lk, "traj": Lt}
    res = test_scores(lin, fac, log=lambda *_: None)
    lam = {k: scales(res, k)[1]["F"] for k in fac}
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    rows = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
    import csv
    csvr = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "qoi_uq.csv"))
            if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
    out = {"model": m, "centring_check": check, "group_sizes": sizes, "n_clusters": int(Lt.shape[1]), "lam_F": lam,
           "qoi": {}}
    print(f"{m}: lam_F CR3 {lam['cr3']:.3f}, trajectory clusters {lam['traj']:.3f}")
    for q, g in rows.items():
        if q.startswith("energy:") or q not in csvr or csvr[q]["dft"] in ("", "nan"):
            continue
        e = float(csvr[q]["dft"]) - float(csvr[q]["mean"])
        s = {k: lam[k] * float(np.linalg.norm(fac[k].T @ g[:n])) for k in fac}
        zz = {k: e / x for k, x in s.items()}
        out["qoi"][q] = {"sigma": s, "z": zz, "covered95": {k: bool(abs(x) <= Z95) for k, x in zz.items()}}
        print(f"  {q:34s} sigma {s['cr3']:.4g} -> {s['traj']:.4g} (x{s['traj'] / s['cr3']:.2f})  z {zz['cr3']:+.2f} -> "
              f"{zz['traj']:+.2f}", flush=True)
    cov = {k: sum(v_["covered95"][k] for v_ in out["qoi"].values()) for k in fac}
    out["covered95"] = {"n": len(out["qoi"]), **cov}
    print(f"  95 % covered (delta sigma): CR3 {cov['cr3']}/{len(out['qoi'])}, trajectory clusters {cov['traj']}/{len(out['qoi'])}")
    json.dump(out, open(RESULTS / f"traj_jackknife_{m}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
