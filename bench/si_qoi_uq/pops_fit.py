"""B2 (spec §6): the existing ace-jax POPS, hypercube form, as K members for committee propagation.

Runs `aj fit --uq pops` with the same data, basis, noise and seed as fits.sh, in-process, keeping the POPS
ridge path the predictions used (pipeline/predict.py `_pops_setup` returns it; `predict_splits` drops it).
From that path the hypercube box is formed exactly as `PopsRidgePath._posterior` forms it -- axes = the
principal directions of sum delta delta^T = A W A, bounds = the min/max of the projected corrections -- and
members theta_k = c* + support u_k with u_k uniform in the box (fixed seed): the box whose second moment
is the POPS hypercube covariance.  Writes pops_box.npz (c_star, support, lo, hi, ridge) next to the fit.

    python -m bench.si_qoi_uq.pops_fit o3d12|o4d20
"""
import json, pathlib, subprocess, sys

import numpy as np

DATA = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
BASIS = {"o4d20": ["--order", "4", "--max-degree", "20"], "o3d12": ["--order", "3", "--max-degree", "12"]}


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax import cli
    from ace_jax.fit.pipeline import predict as ppred
    from ace_jax.fit.pops import hypercube_support
    from .harness import RUNS
    m = sys.argv[1]
    out = RUNS / "fits" / m / "pops"
    out.mkdir(parents=True, exist_ok=True)
    kept = {}
    orig = ppred._pops_setup

    def keep(*a, **k):
        r = orig(*a, **k)
        kept.update(path=r["path"], ridge=r["ridge"])
        return r
    ppred._pops_setup = keep
    argv = ["fit", "--train", f"{DATA}/train.xyz", "--test", f"{DATA}/test.xyz", *BASIS[m], "--rcut", "5.0",
            "--virial-key", "virial", "--weights", f"{DATA}/weights_acejax.json", "--e0", "lsq", "--rungs", "map",
            "--m-per-species", "0", "--noise", "shared", "--configs-per-batch", "1", "--seed", "0", "--uq", "pops",
            # with the default QR statistics the final POPS path's c* (and every test prediction) came out NaN
            # on o3d12 while the ridge selection (rows recomputed) ran: the Gram statistics avoid it (RECON)
            "--lml-solver", "cholesky",
            "--out", str(out)]
    cli.main(argv)
    path, ridge = kept["path"], kept["ridge"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=pathlib.Path(__file__).parent).stdout.strip()
    # one ridge per member set: the predictions use one per quantity, a parameter committee needs one -- the
    # auto-selected force ridge (o3d12: E = F = V = 1e-14); and "blr" (c* the BLR mean, A the BLR posterior),
    # since 1e-14 relative sits at the eigendecomposition's resolution, where A is roundoff (RECON)
    rF = ridge["F"] if isinstance(ridge, dict) else ridge
    # leverage percentile 50: only the high-leverage half of the rows define members (ace-jax's default 0 keeps
    # every row, and the min/max box is then set by low-leverage rows whose corrections r_i / h_i explode)
    for name, rd, lev in (("pops_box", rF, 0.0), ("pops_box_blr", "blr", 0.0),
                          ("pops_box_lev50", rF, 50.0), ("pops_box_blr_lev50", "blr", 50.0)):
        A, coef, keepm = path._coef(rd, lev)
        W, _ = path.rows.moment_sums(coef)
        support = hypercube_support(A @ W @ A, 1.0e-8)
        lo, hi = path.rows.projection_bounds(coef, keepm, A @ support)
        c_star = np.asarray(path.c_star)                         # the mean the predictions used (POPS_MEAN)
        np.savez(out / f"{name}.npz", c_star=c_star, support=np.asarray(support), lo=np.asarray(lo),
                 hi=np.asarray(hi), ridge=str(rd), leverage_pct=lev, ridge_selected=json.dumps(ridge), git=git)
        print(f"{name}: support {support.shape}, ridge {rd}, leverage pct {lev}")

if __name__ == "__main__":
    main()
