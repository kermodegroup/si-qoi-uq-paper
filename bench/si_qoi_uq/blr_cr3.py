"""The CR3 jackknife shape and the posterior factor at the benchmark's own BLR fit (RECON "BLR posterior").

`aj fit --uq ard` could not serve here: in sequential mode with the benchmark's shared noise, the evidence
sits at the box floor of every body-order precision (o4d20: a_k = a_floor = 4.30, cond(S) ~ 7e15), and the
mean it returns from a Cholesky of S is far less accurate than the benchmark fit (test E 16.4 against 3.5
meV/atom).  The BLR fit is ARD at its start point h0 (a_k = -2 log sigma_c: Lambda = Gamma^2 / sigma_c^2,
ARDEvidence.h0) but 12 nats below that floor, so S is beyond a float64 Cholesky; the BLR solve is a QR.

So: rebuild the benchmark fit's problem with its command line (runs/accuracy_cost/ace/<m>/cmd.txt), take its
MAP hyperparameters (theta_map.json, no refit), and factor A = Phi_w^T Phi_w / sigma^2 + Lambda by QR
(objective.posterior_from_qr, as the fit does: L L^T = A).  Then, in ARD's coordinates (D^-1 = 1 / Gamma,
fit.ard.ard_gamma), chol(S) = D^-1 L, and the existing whole-configuration PRESS jackknife
(fit.jackknife.press_scores, shape_factor) gives R.  Written as an ARDPosterior file holding both R (CR3)
and chol (the posterior: B1), next to a model.npz with the QR mean.

    python -m bench.si_qoi_uq.blr_cr3 o3d12|o4d20      (SIQ_RUNS selects the output root)

A key <m>_Ex<f> (plan section 5 D, step 3's selected weights): the same, with every energy weight of the hand weights x f
(weight_sweep.candidates, written to fits/<key>/weights.json) and that candidate's own MAP sigma_c and shared sigma on all of
T (results/final_in.json, analysis/final.py; the rest of theta as the benchmark's); no benchmark model to compare against.
"""
import json, pathlib, shlex, sys, time

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax import cli
    from ace_jax.fit.ard import ARDPosterior, ard_gamma, body_order_columns, E0_GROUP
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.jackknife import press_scores, shape_factor
    from ace_jax.fit.objective import posterior_from_qr
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.export import linear_arrays_from_mean
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.stats import linear_qr_statistics
    from .harness import RUNS
    from .linear import LinearACE
    key = sys.argv[1]
    m, _, ex = key.partition("_Ex")
    out = RUNS / "fits" / key / "cr3"
    out.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}
    argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]          # drop "aj"
    argv[argv.index("--out") + 1] = str(out)
    th_over = {}
    if ex:                                                    # step 3's selected energy weight
        from .analysis.weight_sweep import DATA, candidates, frames, resolve
        RESULTS = pathlib.Path(__file__).parent / "results"     # the package's (SIQ_RESULTS may redirect run.RESULTS)
        types = sorted(set(frames(f"{DATA}/train.xyz")[1]))
        w, _ = candidates(types, "all")[f"Ex{ex}"]
        hand = json.load(open(f"{DATA}/weights_acejax.json"))
        wf = out.parent / "weights.json"
        json.dump({"default": dict(resolve(hand, "__none__")), **w}, open(wf, "w"), indent=1)
        argv[argv.index("--weights") + 1] = str(wf)
        th_over = json.load(open(RESULTS / "final_in.json"))["candidates"][f"Ex{ex}"]["theta"]
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test)
    b = build_problem(cfg, data)
    prob = b.prob
    theta = Hypers(**{**json.load(open(ACC / m / "theta_map.json")), **th_over})
    times["setup"] = time.time() - t0
    qs = linear_qr_statistics(prob.model, prob.cfg, data.ds_train)
    times["qr_statistics"] = time.time() - t0
    mu, L = posterior_from_qr(theta, qs, prob)
    mu, L = np.asarray(mu), jnp.asarray(L)
    del qs
    # the QR mean against the benchmark model file (same data, flags and hyperparameters; ace-jax 0.2.1 then)
    dev = None
    if not ex:
        ref = LinearACE(ACC / m / "model.npz")
        n = ref.len_readout
        dev = float(np.max(np.abs(mu[:n] - ref.theta_hat[:n])) / np.max(np.abs(ref.theta_hat[:n])))
    np.savez(out / "model.npz", **linear_arrays_from_mean(data.z, data.E0, prob.cfg, mu))
    gamma = ard_gamma(prob)
    dinv = 1.0 / gamma
    body_col = body_order_columns(data.meta, prob.cfg)
    groups = tuple(int(g) for g in np.unique(body_col[body_col != E0_GROUP]))
    keep = {k: data.meta[k] for k in ("n_B", "n_pair", "NZ", "rcut", "elements") if k in data.meta}
    keep.setdefault("NZ", len(keep.get("elements", [])) or 1)
    if np.any(body_col == E0_GROUP):
        keep["e0_cols"] = int(np.sum(body_col == E0_GROUP))
    keep["source"] = f"bench.si_qoi_uq.blr_cr3: BLR posterior at {ACC / m / 'theta_map.json'}" + (f", Ex{ex} weights and MAP" if ex else "")
    chol = jnp.asarray(dinv)[:, None] * L                   # S = D^-1 A D^-1 = (D^-1 L)(D^-1 L)^T
    a_blr = float(-2 * theta.log_sigma_c)
    post = ARDPosterior(mu, chol, dinv, 1.0, np.full(len(groups), a_blr), groups, body_col, keep)
    gc = RUNS.parent / "si_qoi_uq" / "fits" / key / "cr3" / "posterior.npz"     # Mondrian groups: data-only
    if gc.exists():                                                           # constants of the ARD run
        post = post._replace(group_consts=ARDPosterior.load(gc).group_consts)
    sig = np.exp([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
    rc, K = row_clusters(data.ds_train, data.train, np.inf)                 # whole configurations (RECON Q3)
    G, lev = press_scores(post, prob, data.ds_train, rc, K, sig, mode="exact")
    times["press"] = time.time() - t0
    R = shape_factor(post, G, 1.0)
    post = post._replace(R=R)
    post.save(out / "posterior.npz", dtype=np.float64)
    z = dict(np.load(out / "posterior.npz"))
    c = np.asarray(chol)
    z["chol_packed"] = c[np.tril_indices(c.shape[0])]       # save() drops chol when R is set; B1 needs it
    np.savez(out / "posterior.npz", **z)
    kp = RUNS / "fits" / key / "kappa"                        # B1 reads the same file (its chol)
    if not kp.exists():
        kp.symlink_to(out)
    info = {"model": key, "theta_map": str(ACC / m / "theta_map.json"), "theta_override": th_over, "a_blr": a_blr, "sigma": sig.tolist(),
            "K_clusters": int(K), "rank_R": int(np.shape(R)[1]), "lev_max": float(np.max(lev)),
            "n_lev_near1": int(np.sum(lev > 1 - 1e-8)), "mean_rel_dev_vs_benchmark_model": dev,
            "diag_L_min_max": [float(jnp.min(jnp.diag(L))), float(jnp.max(jnp.diag(L)))], "seconds": times}
    json.dump(info, open(out / "blr_cr3.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
