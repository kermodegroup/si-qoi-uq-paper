"""Debugging dump for the B2 / POPSRegression discrepancy: the port's loss-weighted design and internals, once.

Runs the fit of pops_fit.py (same arguments) and saves, without any POPS post-processing of its own:
  X (rows w*phi/sigma_q scaled by D^-1 = 1/Gamma: the isotropic-ridge design), y (w*y/sigma_q), Dinv,
  the port's eigenvalues Lam of D^-1 M D^-1, lam_blr, theta (sigma_q, sigma_c), the ridge-selection scores,
  and whether the port's BLR mean c*('blr') is finite.
-> $POPS_DUMP/design_<m>.npz (default /storage/eng/essswb/projects/si-qoi-uq-paper/runs/popsref)

    python -m bench.si_qoi_uq.analysis.pops_dump o3d12
"""
import json, os, pathlib, sys

import numpy as np

OUT = pathlib.Path(os.environ.get("POPS_DUMP", "/storage/eng/essswb/projects/si-qoi-uq-paper/runs/popsref"))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax import cli
    from ace_jax.fit.pipeline import predict as ppred
    from ace_jax.fit.stats import HostRows
    from ..pops_fit import BASIS, DATA
    m = sys.argv[1]
    ridge = sys.argv[sys.argv.index("--ridge") + 1] if "--ridge" in sys.argv else None   # skip the ridge selection
    kept, orig = {}, ppred._pops_setup

    def keep(cfg, d, b, stats, theta, log):
        r = orig(cfg, d, b, stats, theta, log)
        kept.update(path=r["path"], out=r, prob=b.prob, ds=d.ds_train, theta=theta)
        return r
    ppred._pops_setup = keep
    try:
        cli.main(["fit", "--train", f"{DATA}/train.xyz", "--test", f"{DATA}/test.xyz", *BASIS[m], "--rcut", "5.0",
                  "--virial-key", "virial", "--weights", f"{DATA}/weights_acejax.json", "--e0", "lsq", "--rungs", "map",
                  "--m-per-species", "0", "--noise", "shared", "--configs-per-batch", "1", "--seed", "0", "--uq", "pops",
                  "--lml-solver", "cholesky", *(["--pops-ridge", ridge] if ridge else []), "--no-save-model", "--out", str(OUT / f"fit_{m}")])
    except ValueError as e:
        if "path" not in kept:
            raise
        print(f"fit output not written ({e})", flush=True)
    path = kept["path"]
    rows = path.rows if isinstance(path.rows, HostRows) else HostRows(kept["prob"].model, kept["prob"].cfg, kept["ds"], path.qs)
    Dinv = np.asarray(path.Dinv)
    X = np.concatenate(rows.blocks)[: rows.n_rows] * Dinv[None, :]
    y = np.concatenate(rows.targets)[: rows.n_rows]
    th = kept["theta"]
    blr = np.asarray(path.c_star_at("blr"))
    meta = {"model": m, "sigma": path.sigma, "lam_blr": path.lam_blr, "qs": list(path.qs),
            "log_sigma_c": float(th.log_sigma_c), "ridge_selected": kept["out"]["ridge"],
            "ridge_scores": kept["out"].get("ridge_scores"), "blr_mean_finite": bool(np.all(np.isfinite(blr))),
            "rows_kind": type(path.rows).__name__}
    print(json.dumps(meta, indent=1, default=str), flush=True)
    np.savez(OUT / f"design_{m}.npz", X=X, y=y, Dinv=Dinv, Lam=np.asarray(path.Lam), blr_mean=blr,
             meta=json.dumps(meta, default=str))
    print("saved", OUT / f"design_{m}.npz", flush=True)


if __name__ == "__main__":
    main()
