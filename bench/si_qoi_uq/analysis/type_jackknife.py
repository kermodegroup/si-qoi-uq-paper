"""Delete-one-config-type jackknife at the benchmark BLR fit: can a coarser cluster see the shared defect bias?

Errors are correlated within a config type (RECON "Why the interstitial formation energies are confidently wrong"),
which a delete-one-configuration jackknife cannot detect.  Here a cluster is a whole GAP-18 config_type.  Clusters
span many batches, so press_scores (one batch per cluster) does not apply; the deletion is done exactly instead:
per-type QR statistics, R_t from a QR of [R_q / sigma_q]_q, and theta_(-k) from the QR of [diag sqrt(Lambda); R_t, t != k]
(the solve objective.posterior_from_qr makes, without type k).  The shape factor is L = [theta_(-k) - mean_k]_k, so
Sigma = L L^T is the centred delete-one-type jackknife (the CR3 construction at type level; scale left to lambda).

    python -m bench.si_qoi_uq.analysis.type_jackknife o3d12|o4d20     (SIQ_RUNS: the BLR run root)
then the scores, scales and QoI sigmas (delta rows of RUNS/delta/<m>.npz):
    python -m bench.si_qoi_uq.analysis.type_jackknife o3d12 --score
"""
import json, pathlib, shlex, sys, time

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")


def build(m, RUNS):
    import jax
    import jax.numpy as jnp
    from ace_jax import cli
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.objective import linear_prior_diag
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.stats import linear_qr_statistics
    t0 = time.time()
    argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
    argv[argv.index("--out") + 1] = "/tmp/unused"
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test)
    prob = build_problem(cfg, data).prob
    theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
    sig = {q: float(np.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
    types = sorted({c.config_type for c in data.train}, key=str)
    Rt, ct, nt = [], [], []
    for t in types:
        sub = [c for c in data.train if c.config_type == t]
        ds = build_dataset(sub, data.meta, data.E0, cfg.batch, log=lambda *_: None)
        qs = linear_qr_statistics(prob.model, prob.cfg, ds)
        A = jnp.concatenate([getattr(qs, f"R_{q}") / sig[q] for q in "EFV"], 0)
        y = jnp.concatenate([getattr(qs, f"c_{q}") / sig[q] for q in "EFV"])
        Q, R = jnp.linalg.qr(A, mode="reduced")
        Rt.append(R); ct.append(Q.T @ y); nt.append(len(sub))
        print(f"{t}: {len(sub)} configs ({time.time() - t0:.0f} s)", flush=True)
    lin = linear_prior_diag(theta, prob)
    P, p0 = jnp.diag(jnp.sqrt(lin)), jnp.zeros_like(lin)

    def solve(keep):
        A = jnp.concatenate([P] + [Rt[i] for i in keep], 0)
        y = jnp.concatenate([p0] + [ct[i] for i in keep])
        Q, R = jnp.linalg.qr(A, mode="reduced")
        return np.asarray(jax.scipy.linalg.solve_triangular(R, Q.T @ y, lower=False))
    K = len(types)
    th_all = solve(range(K))
    th_del = np.stack([solve([i for i in range(K) if i != k]) for k in range(K)])
    L = (th_del - th_del.mean(0)).T
    ref = np.load(RUNS / "fits" / m / "cr3" / "posterior.npz")["mean"]
    dev = float(np.linalg.norm(th_all - ref) / np.linalg.norm(ref))
    out = RUNS / "type_jk"
    out.mkdir(exist_ok=True)
    np.savez(out / f"{m}.npz", L=L, types=np.array(types, dtype=object), n_configs=np.array(nt), theta_all=th_all,
             theta_del=th_del, rel_dev_theta_all_vs_blr=dev)
    print(f"{m}: {K} types, theta_all vs BLR mean {dev:.2e}, {time.time() - t0:.0f} s")


def score(m, RUNS, results):
    """Test-split scales of the type shape, then QoI sigma by the delta rows, against CR3 (BLR posterior file)."""
    import csv
    from ..diagnostics import scales, test_scores
    from ..linear import LinearACE
    from ..propagate import COVER_KEYS, KEYS, dft, harness_values, tol_abs
    z = np.load(RUNS / "type_jk" / f"{m}.npz", allow_pickle=True)
    lin = LinearACE(RUNS / "fits" / m / "cr3" / "model.npz", RUNS / "fits" / m / "cr3" / "posterior.npz")
    Lt, Lc = z["L"], lin.factor("cr3")
    res = test_scores(lin, {"type": Lt, "cr3": Lc})
    _, lam_t = scales(res, "type")
    _, lam_c = scales(res, "cr3")
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    G = dict(zip([str(k) for k in dz["keys"]], dz["G"]))
    Qhat, ref = harness_values(f"SIQ_{m}_cr3"), dft()
    rows = []
    for k in COVER_KEYS:
        if k not in G:
            continue
        i = KEYS.index(k)
        tol = tol_abs(k, ref[i])
        for name, L, lam in (("type", Lt, lam_t), ("cr3", Lc, lam_c)):
            for sc in ("F", "V"):
                s = lam[sc] * float(np.linalg.norm(L.T @ G[k]))
                zz = (Qhat[i] - ref[i]) / s
                rows.append(dict(model=m, shape=name, scale=f"lam_{sc}", lam=lam[sc], qoi=k, err_over_tol=(Qhat[i] - ref[i]) / tol,
                                 sigma=s, sigma_over_tol=s / tol, z=zz, cov68=int(abs(zz) <= 1), cov95=int(abs(zz) <= 1.96)))
    with open(results / f"type_jackknife_{m}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"model": m, "lam_type": lam_t, "lam_cr3": lam_c, "n_types": int(len(z["types"])),
            "rel_dev_theta_all_vs_blr": float(z["rel_dev_theta_all_vs_blr"])}
    for name in ("type", "cr3"):
        for sc in ("lam_F", "lam_V"):
            rr = [r for r in rows if r["shape"] == name and r["scale"] == sc]
            summ[f"{name}/{sc}"] = {"n": len(rr), "k68": sum(r["cov68"] for r in rr), "k95": sum(r["cov95"] for r in rr),
                                    "median_sigma_over_tol": float(np.median([r["sigma_over_tol"] for r in rr]))}
    json.dump(summ, open(results / f"type_jackknife_{m}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for r in rows:
        if r["scale"] == "lam_F" and ("interstitial" in r["qoi"] or "vacancy" in r["qoi"]):
            print(f"{r['shape']:5s} {r['qoi'][:30]:30s} err/tol {r['err_over_tol']:+7.2f} sigma/tol {r['sigma_over_tol']:6.2f} z {r['z']:+6.2f}")


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ..harness import RUNS
    from ..run import RESULTS
    m = sys.argv[1]
    if "--score" in sys.argv:
        score(m, RUNS, RESULTS)
    else:
        build(m, RUNS)


if __name__ == "__main__":
    main()
