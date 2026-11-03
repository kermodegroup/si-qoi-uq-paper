"""B2 (spec §6): the existing ace-jax POPS, hypercube form, as K members for committee propagation.

Runs `aj fit --uq pops` with the same data, basis, noise and seed as fits.sh, in-process, keeping the POPS
ridge path the predictions used (pipeline/predict.py `_pops_setup` returns it; `predict_splits` drops it).
From that path the hypercube box is formed exactly as `PopsRidgePath._posterior` forms it -- axes = the
principal directions of sum delta delta^T = A W A, bounds = the min/max of the projected corrections -- and
members theta_k = c* + support u_k with u_k uniform in the box (fixed seed): the box whose second moment
is the POPS hypercube covariance.  Writes pops_box.npz (c_star, support, lo, hi, ridge) next to the fit.

    python -m bench.si_qoi_uq.pops_fit o3d12|o4d20 [--levels 75,90,95,99]

--levels: only the boxes at these leverage percentiles (both ridges, names pops_box[_blr]_lev{pct}), each screened on
three rattled diamond cells (a = 5.2, 5.46, 5.8 A; E and F are exact rows) for the members' energy and force spread
-> pops_screen.json next to the fit.  Without it, the four boxes of the original run.
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
    args = sys.argv[1:]
    levels = None
    if "--levels" in args:
        i = args.index("--levels"); levels = [float(x) for x in args[i + 1].split(",")]; del args[i:i + 2]
    m = args[0]
    out = RUNS / "fits" / m / ("pops" if levels is None else "pops_sweep")   # the sweep never touches the run boxes
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
    import ace_jax
    acejax = ace_jax.__version__ + "+" + subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                      cwd=pathlib.Path(ace_jax.__file__).parent).stdout.strip()
    # one ridge per member set: the predictions use one per quantity, a parameter committee needs one -- the
    # auto-selected force ridge (o3d12: E = F = V = 1e-14); and "blr" (c* the BLR mean, A the BLR posterior),
    # since 1e-14 relative sits at the eigendecomposition's resolution, where A is roundoff (RECON)
    rF = ridge["F"] if isinstance(ridge, dict) else ridge
    # leverage percentile 50: only the high-leverage half of the rows define members (ace-jax's default 0 keeps
    # every row, and the min/max box is then set by low-leverage rows whose corrections r_i / h_i explode)
    boxes = (("pops_box", rF, 0.0), ("pops_box_blr", "blr", 0.0), ("pops_box_lev50", rF, 50.0), ("pops_box_blr_lev50", "blr", 50.0))
    if levels is not None:
        boxes = tuple((f"pops_box{b}_lev{lev:g}", rd_, lev) for lev in levels for b, rd_ in (("", rF), ("_blr", "blr")))
    screen = {}
    for name, rd, lev in boxes:
        A, coef, keepm = path._coef(rd, lev)
        W, _ = path.rows.moment_sums(coef)
        support = hypercube_support(A @ W @ A, 1.0e-8)
        lo, hi = path.rows.projection_bounds(coef, keepm, A @ support)
        c_star = np.asarray(path.c_star)                         # the mean the predictions used (POPS_MEAN)
        np.savez(out / f"{name}.npz", c_star=c_star, support=np.asarray(support), lo=np.asarray(lo),
                 hi=np.asarray(hi), ridge=str(rd), leverage_pct=lev, ridge_selected=json.dumps(ridge), git=git,
                 acejax=acejax)
        print(f"{name}: support {support.shape}, ridge {rd}, leverage pct {lev}", flush=True)
        if levels is not None:
            screen[name] = dict(ridge=str(rd), leverage_pct=lev, n_support=int(np.asarray(support).shape[1]),
                                **member_screen(m, c_star, np.asarray(support), np.asarray(lo), np.asarray(hi)))
            print(f"  {screen[name]}", flush=True)
    if levels is not None:
        prev = json.load(open(out / "pops_screen.json")) if (out / "pops_screen.json").exists() else {}
        json.dump({**prev, **screen}, open(out / "pops_screen.json", "w"), indent=1)


def member_screen(m, c_star, support, lo, hi, K=200):
    """Members theta_k = c* + support u_k, u_k uniform in [lo, hi] (the seed and draw of run.py samples),
    against the BLR fit's rows on three rattled diamond cells: per member the worst |dE| per atom and
    max |dF| over the cells, relative to c*."""
    from ase.build import bulk
    from .run import lin_for
    from .linear import observable_rows
    lin = lin_for(m, "cr3")
    from .run import SEEDS
    u = np.random.default_rng(SEEDS["pops"]).uniform(lo, hi, size=(K, len(lo)))
    d = u @ support.T                                              # theta_k - c*
    dE, dF = np.zeros(K), np.zeros(K)
    for a, r in ((5.46, 0.03), (5.2, 0.1), (5.8, 0.1)):
        at = bulk("Si", "diamond", a=a, cubic=True); at.rattle(r, seed=1)
        bE, BF, _ = observable_rows(at, lin)
        bE, BF = bE[:support.shape[0]], BF[:, :support.shape[0]]   # POPS fixes the lsq E0: no E0 columns (run.py pads zeros)
        dE = np.maximum(dE, np.abs(d @ bE) / len(at)); dF = np.maximum(dF, np.abs(d @ BF.T).max(1))
    ok = lambda e, f: int(((dE < e) & (dF < f)).sum())
    return dict(median_dE_eV_atom=float(np.median(dE)), median_dF_eV_A=float(np.median(dF)),
                pass_tight=ok(0.05, 1.0), pass_loose=ok(5.0, 50.0), K=K)

if __name__ == "__main__":
    main()
