"""The BLR posterior factor of an out-of-distribution refit (B1's shape on the reduced training set).

The refits (`aj fit --uq ard --ard-prior fixed`, OOD_RUNS/fits/<m>) save the CR3 factor R but not the posterior
Cholesky.  As blr_cr3.py does for the benchmark fits: rebuild the refit's problem from its resolved fit.yaml, take its
MAP hyperparameters (theta_map.json, no refit), and factor A = Phi_w^T Phi_w / sigma^2 + Lambda by QR
(objective.posterior_from_qr: L L^T = A).  Writes posterior_chol.npz (L, the QR mean, and its relative deviation from
the refit's own mean) next to the refit; B1's factor is L^-T.

    OOD_RUNS=<ood_holdout_blr> python -m bench.si_qoi_uq.ood.posterior_chol o3d12|o4d20
"""
import json, os, pathlib, sys

import numpy as np

OOD_RUNS = pathlib.Path(os.environ.get("OOD_RUNS", "/storage/eng/essswb/projects/si-qoi-uq-paper/runs/ood_holdout_blr"))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax import cli
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.objective import posterior_from_qr
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.stats import linear_qr_statistics
    from ..linear import LinearACE
    m = sys.argv[1]
    d = OOD_RUNS / "fits" / m
    a = cli._parse(["fit", "--config", str(d / "fit.yaml")])
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test)
    prob = build_problem(cfg, data).prob
    theta = Hypers(**json.load(open(d / "theta_map.json")))
    mu, L = posterior_from_qr(theta, linear_qr_statistics(prob.model, prob.cfg, data.ds_train), prob)
    mu, L = np.asarray(mu), np.asarray(L)
    ref = LinearACE(d / "model.npz", d / "posterior.npz")
    n = ref.len_readout
    dev = float(np.max(np.abs(mu[:n] - ref.theta_hat[:n])) / np.max(np.abs(ref.theta_hat[:n])))
    np.savez(d / "posterior_chol.npz", L=L, mean=mu, mean_rel_dev=dev)
    print(json.dumps({"model": m, "p": int(L.shape[0]), "mean_rel_dev_vs_refit": dev}), flush=True)


if __name__ == "__main__":
    main()
