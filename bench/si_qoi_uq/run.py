"""Silicon QoI UQ propagation: every stage, and `--all` (spec "Outputs").

    python -m bench.si_qoi_uq.run --all                 # every stage whose inputs exist
    python -m bench.si_qoi_uq.run --stage diagnostics --models o3d12

Heavy stages (diagnostics, delta, committee) belong on mnf148/lestrade, not in a capped agent session.
Large intermediates live under $SIQ_RUNS (/storage/.../runs/si_qoi_uq); `results/` and `figures/` here
hold only what the report reads.
"""
import argparse, json, pathlib, time

import numpy as np

from .harness import RUNS

HERE = pathlib.Path(__file__).parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
MODELS = ("o4d20", "o3d12")
K = 200
SEEDS = {"cr3": 20261031, "posterior": 20261032, "pops": 20261033}     # committee draws (spec §4b, §6)


def fit_dir(m, v):
    return RUNS / "fits" / m / v


def lin_for(m, variant="cr3"):
    from .linear import LinearACE
    d = fit_dir(m, variant)
    return LinearACE(d / "model.npz", d / "posterior.npz")


def stage_diagnostics(m, log=print):
    """Spec §3 -> RUNS/diag/<m>.npz (per-row scores) and the rows of results/scale_diagnostics.csv."""
    from .diagnostics import conformal_q, scales, test_scores
    lin = lin_for(m, "cr3")
    linp = lin_for(m, "kappa")
    if np.abs(lin.theta_hat - linp.theta_hat).max() > 1e-10:
        raise ValueError(f"{m}: cr3 and kappa fits have different means")
    t = time.time()
    res = test_scores(lin, {"cr3": lin.factor("cr3"), "posterior": linp.factor("posterior")}, log=log)
    out = RUNS / "diag"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{m}.npz", **{k: v for k, v in res.items() if k not in ("sd", "n_dropped")},
             **{f"sd_{k}": v for k, v in res["sd"].items()})
    rows, lam = scales(res, "cr3")
    rows_p, lam_p = scales(res, "posterior")
    q = conformal_q(res, "posterior")
    summary = {"model": m, "lam_cr3": lam, "lam_posterior_rms": lam_p, "q_b1": q,
               "q_b1_level": 0.6827, "n_dropped_rows": res["n_dropped"], "seconds": time.time() - t}
    json.dump(summary, open(out / f"{m}_scales.json", "w"), indent=1)
    for r in rows + rows_p:
        r["model"] = m
    log(f"{m}: lam {lam}; B1 q {q:.3g}")
    return rows + rows_p, summary


def write_scale_csv(models):
    import csv
    rows = []
    for m in models:
        p = RUNS / "diag" / f"{m}_rows.json"
        if p.exists():
            rows += json.load(open(p))
    if not rows:
        return
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "scale_diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "factor", "kind", "breakdown", "label", "n", "rms_z", "q68", "q95"])
        w.writeheader()
        w.writerows(rows)


def stage_samples(m, log=print):
    """Spec §4b/§6: K draws per method, sampled once at lambda = 1 -> RUNS/samples/<m>_<set>.npz.
    cr3 (also B3): theta_hat + L_cr3 xi; posterior (B1): theta_hat + L_post xi; pops (B2): c* + box draws."""
    from .harness import save_samples
    lin = lin_for(m, "cr3")
    out = RUNS / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for name, variant in (("cr3", "cr3"), ("posterior", "kappa")):
        L = lin_for(m, variant).factor("cr3" if name == "cr3" else "posterior")
        xi = np.random.default_rng(SEEDS[name]).standard_normal((K, L.shape[1]))
        th = lin.theta_hat[None] + xi @ L.T
        save_samples(out / f"{m}_{name}.npz", f"{m}_{name}", fit_dir(m, "cr3") / "model.npz", th, lin,
                     theta_mean=lin.theta_hat, seed=SEEDS[name])
        log(f"{m} {name}: {th.shape}")
        if name == "cr3":                       # §4c: the full re-run at lam_F, same xi (needs the diagnostics)
            sc = RUNS / "diag" / f"{m}_scales.json"
            if sc.exists():
                lamF = json.load(open(sc))["lam_cr3"]["F"]
                save_samples(out / f"{m}_cr3lamF.npz", f"{m}_cr3lamF", fit_dir(m, "cr3") / "model.npz",
                             lin.theta_hat[None] + lamF * (xi @ L.T), lin, theta_mean=lin.theta_hat,
                             seed=SEEDS[name], lam=lamF)
                log(f"{m} cr3 at lam_F = {lamF:.4g}")
    box = fit_dir(m, "pops") / "pops_box.npz"
    if box.exists():
        b = np.load(box)
        u = np.random.default_rng(SEEDS["pops"]).uniform(b["lo"], b["hi"], size=(K, len(b["lo"])))
        cs = np.asarray(b["c_star"])
        th = cs[None] + u @ np.asarray(b["support"]).T
        if th.shape[1] != lin.p:                      # POPS c* may omit or include the E0 column like the fit
            raise ValueError(f"POPS width {th.shape[1]} vs {lin.p}")
        save_samples(out / f"{m}_pops.npz", f"{m}_pops", fit_dir(m, "cr3") / "model.npz", th, lin,
                     theta_mean=cs, seed=SEEDS["pops"])
        log(f"{m} pops: {th.shape}")


STAGES = {"diagnostics": stage_diagnostics, "samples": stage_samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", choices=list(STAGES), action="append", default=[])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    stages = list(STAGES) if a.all else a.stage
    for s in stages:
        for m in a.models:
            r = STAGES[s](m)
            if s == "diagnostics":
                json.dump(r[0], open(RUNS / "diag" / f"{m}_rows.json", "w"))
        if s == "diagnostics":
            write_scale_csv(a.models)


if __name__ == "__main__":
    main()
