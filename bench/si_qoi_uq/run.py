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
    facs = {"cr3": lin.factor("cr3"), "posterior": linp.factor("posterior")}
    if (fit_dir(m, "cr3_ell3") / "posterior.npz").exists():        # RECON Q3 sensitivity: default 3 r_cut clusters
        le = lin_for(m, "cr3_ell3")
        if np.abs(le.theta_hat - lin.theta_hat).max() < 1e-10:
            facs["cr3_ell3"] = le.factor("cr3")
    res = test_scores(lin, facs, log=log)
    out = RUNS / "diag"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{m}.npz", **{k: v for k, v in res.items() if k not in ("sd", "n_dropped")},
             **{f"sd_{k}": v for k, v in res["sd"].items()})
    rows, lam = scales(res, "cr3")
    rows_p, lam_p = scales(res, "posterior")
    q = conformal_q(res, "posterior")
    if "cr3_ell3" in facs:
        rows_e, lam_e = scales(res, "cr3_ell3")
        rows_p += rows_e
    summary = {"model": m, "lam_cr3": lam, "lam_posterior_rms": lam_p, "q_b1": q,
               **({"lam_cr3_ell3": lam_e} if "cr3_ell3" in facs else {}),
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
    for tag, fname in (("pops", "pops_box.npz"), ("popsblr", "pops_box_blr.npz"),
                       ("popslev50", "pops_box_lev50.npz"), ("popsblrlev50", "pops_box_blr_lev50.npz")):
        box = fit_dir(m, "pops") / fname
        if not box.exists():
            continue
        b = np.load(box)
        u = np.random.default_rng(SEEDS["pops"]).uniform(b["lo"], b["hi"], size=(K, len(b["lo"])))
        cs = np.asarray(b["c_star"])
        th = cs[None] + u @ np.asarray(b["support"]).T
        if lin.e0_cols and th.shape[1] == lin.p - lin.NZ:   # POPS fixes the lsq E0 (as prefit): no E0 column,
            th = np.concatenate([th, np.zeros((K, lin.NZ))], 1)   # i.e. a zero shift of E0_pre
            cs = np.concatenate([cs, np.zeros(lin.NZ)])
        if th.shape[1] != lin.p:
            raise ValueError(f"POPS width {th.shape[1]} vs {lin.p}")
        save_samples(out / f"{m}_{tag}.npz", f"{m}_{tag}", fit_dir(m, "cr3") / "model.npz", th, lin,
                     theta_mean=cs, seed=SEEDS["pops"], ridge=str(b["ridge"]))
        log(f"{m} {tag}: {th.shape}")


def stage_delta(m, log=print):
    """Spec §4a: implicit-differentiation rows at theta_hat (from the mean run's relaxed frames), and the JAX
    re-implementation of both elastic routes on the CR3 lam = 1 samples (the energy route has no harness
    committee) -> RUNS/delta/<m>.npz, <m>_checks.json."""
    from .propagate import delta_rows, elastic_committee
    lin = lin_for(m, "cr3")
    t = time.time()
    G, checks, extra = delta_rows(lin, f"SIQ_{m}_cr3", log=log)
    s = np.load(RUNS / "samples" / f"{m}_cr3.npz")
    el = elastic_committee(lin, s["theta"], extra["a0"])
    out = RUNS / "delta"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(G)
    np.savez(out / f"{m}.npz", keys=np.array(keys), G=np.stack([G[k] for k in keys]), da0=extra["da0"],
             elastic_committee=el)
    json.dump({"checks": {k: [float(v[0]) if v[0] is not None else None,
                              None if v[1] is None else float(v[1]), v[2]] for k, v in checks.items()},
               "a0": extra["a0"], "seconds": time.time() - t}, open(out / f"{m}_checks.json", "w"), indent=1)
    log(f"{m}: {len(keys)} delta rows in {time.time() - t:.0f} s")


def stage_assemble(models, log=print):
    """Spec §4c-§6 tables, joint tests, figures and run metadata, over the models whose inputs exist."""
    from . import assemble, figures
    rows, dvc, cov, joint, extras = [], [], [], {}, {}
    for m in models:
        if not (RUNS / "delta" / f"{m}.npz").exists() or not (RUNS / "diag" / f"{m}_scales.json").exists():
            log(f"{m}: delta or diagnostics missing, skipped")
            continue
        r, d, ex = assemble.model_tables(m, K, log=log)
        rows += r
        dvc += d
        extras[m] = ex
        joint[m] = assemble.joint(m, ex)
    cov = assemble.coverage(rows)
    RESULTS.mkdir(exist_ok=True)
    assemble.write_csv(RESULTS / "qoi_uq.csv", rows)
    assemble.write_csv(RESULTS / "delta_vs_committee.csv", dvc)
    assemble.write_csv(RESULTS / "coverage.csv", cov)
    assemble.write_csv(RESULTS / "pops_committee_finished.csv", [r for ex in extras.values() for r in ex["pops_finished"]])
    json.dump(joint, open(RESULTS / "joint_tests.json", "w"), indent=1)
    write_scale_csv(models)
    figures.all_figures(RESULTS, FIGURES, RUNS, list(extras), extras)
    write_metadata(list(extras))
    log(f"assembled {len(rows)} rows for {list(extras)}")


def write_metadata(models):
    import platform, subprocess, importlib.metadata as md
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())
    pk = {}
    for p in ("jax", "jaxlib", "numpy", "scipy", "ase", "matscipy", "phonopy"):
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    walls = {}
    for m in models:
        for v in ("cr3", "kappa", "pops"):
            st = RUNS / "fits" / m / v / "timings.json"
            if st.exists():
                walls[f"fit/{m}/{v}"] = json.load(open(st)).get("total")
        for f in (RUNS / "diag" / f"{m}_scales.json", RUNS / "delta" / f"{m}_checks.json"):
            if f.exists():
                walls[f"{f.parent.name}/{m}"] = json.load(open(f)).get("seconds")
        for s in ("cr3", "posterior", "cr3lamF", "pops"):
            tot = 0.0
            for t in (RUNS / "logs").glob(f"committee_{m}_{s}_*.times.json"):
                tot += sum(json.load(open(t)).values())
            if tot:
                walls[f"committee/{m}/{s} (cpu-process s)"] = tot
    meta = {"git": git, "dirty": dirty, "packages": pk, "python": platform.python_version(), "host": platform.node(),
            "seeds": {"fit": 0, **SEEDS}, "K": K, "harness": {"path": "~/silicon-testing-framework (py3 port)",
            "env": {"SI_TESTS_ELASTIC_FMAX": "1e-3"}}, "runs": str(RUNS), "wall_s": walls,
            "hardware": "fits, diagnostics, delta, committees: lestrade (32 cores, 62 GB) and mnf148 (32 cores, 256 GB)"}
    json.dump(meta, open(RESULTS / "run_metadata.json", "w"), indent=1)


STAGES = {"diagnostics": stage_diagnostics, "samples": stage_samples, "delta": stage_delta}
GLOBAL = {"assemble": stage_assemble}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", choices=list(STAGES) + list(GLOBAL) + ["report"], action="append", default=[])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    # --all: the report from existing intermediates (the heavy per-model stages run on the compute nodes first;
    # a stage whose output exists is not redone)
    stages = ["assemble", "report"] if a.all else a.stage
    for s in stages:
        if s in GLOBAL:
            GLOBAL[s](a.models)
            continue
        if s == "report":
            from .report import write_report
            write_report(RESULTS, HERE / "REPORT.md")
            continue
        for m in a.models:
            r = STAGES[s](m)
            if s == "diagnostics":
                json.dump(r[0], open(RUNS / "diag" / f"{m}_rows.json", "w"))
        if s == "diagnostics":
            write_scale_csv(a.models)


if __name__ == "__main__":
    main()
