"""Step 3 of plan section 5 D: the selected weights refitted on all of T, CRC-calibrated and reported on the test split; and the
held-out-type check (principle 6).

Selection (weight_sweep.py, wl_sweep.py, selection.py: T_fit / T_val only) chose o4d16 (wL 1.5) with the hand weights' energy
weight x 10 and the B1 shape; x 5 is the alternative and hand the reference. Nothing here chooses anything: the candidates are
fixed, and the test split is used only to calibrate (CRC) and to report.

Setting "in": each candidate fitted on T (train.xyz, the benchmark's training set) with weight_sweep's machinery (per-type QR
factors, an exact weighted merge, its own MAP of sigma_c and one shared sigma, posterior_from_qr; B1 = the posterior factor).
Scored on the test split (test.xyz) as weight_sweep scores T_val: views E, F, dE_type, dE_form (pairs drawn on the test split
with weight_sweep.pairs, before any candidate is seen); per random configuration-level split, the shape (lam_s, nu) on A1 (1/4),
the CRC threshold on A2 (1/4), and on B (1/2) the risk overall and by kind and the size (log geometric semi-axis). The marginal
guarantee: mean risk on B <= alpha.

Setting "ood": the same candidates fitted on ood_holdout/train.xyz (T without cracks and screw dislocations, ood/make_split.py),
shape and threshold from the same split protocol on ood_holdout/test.xyz (the test split without them), and in every split the
risk and size on the held-out configurations (ood.xyz: 33 + 3 crack and screw configurations), overall and by type. dE_form pairs
of a held-out configuration take a diamond configuration of the calibration file, as on T_val. No guarantee applies off
distribution: this measures how far it fails.

Shapes: B1 for every candidate; CR3 and the POPS ellipsoid beside it where final_shapes.py has written
RUNS/final/<setting>/{cr3,ellipsoid}_<candidate>.npz (named "<candidate>/CR3", "<candidate>/ellipsoid"; the B1 mean, each
shape's own factor M_c = rows_c F, with Sigma_c = lam_s^2 M_c M_c^T + nu^2 I fitted per split as for B1).

    python -m bench.si_qoi_uq.analysis.final in|ood   (a compute node)  -> results/final_<setting>.json
"""
import json, os, sys, time

import numpy as np

from .weight_sweep import ALPHAS, DATA, candidates, eval_rows, map_fit, merge, pairs, problem, type_stats, views_of

MODEL = "o4d16"
NAMES = ("hand", "Ex5", "Ex10")
OOD = os.environ.get("OOD_DATA", "/storage/eng/essswb/projects/si-qoi-uq-paper/data/ood_holdout")
N_SPLIT = int(os.environ.get("FINAL_SPLITS", 100))


def splits(cands, cal, ood, kinds, groups, alphas, rng):
    """cands: name -> (View, owner). cal / ood: boolean masks over configurations (by owner). Per split of the calibration
    elements: shape on A1, t on A2, risk / size on B and on every ood element. Paired: the same splits for every candidate."""
    from .crc_config import risk, size, t_crc
    any_c = next(iter(cands.values()))
    own = any_c[1]
    ci, oi = np.flatnonzero(cal[own]), np.flatnonzero(ood[own])
    kk, gg = np.unique(kinds[own[ci]]), np.unique(groups[own[oi]]) if len(oi) else []
    rec = lambda: {"size": [], "risk": [], "by": {}}
    out = {la: {c: {"test": rec(), "ood": rec()} for c in cands} for la in alphas}
    n = len(ci)
    for _ in range(N_SPLIT):
        p = ci[rng.permutation(n)]
        A1, A2, B = p[: n // 4], p[n // 4: n // 2], p[n // 2:]
        for c, (M, ow) in cands.items():
            assert np.array_equal(ow, own)
            sh = M.fit(A1)
            W2 = M.whiten(sh, A2)[0]
            WB, HB = M.whiten(sh, B)
            WO, HO = M.whiten(sh, oi) if len(oi) else ([], np.empty(0))
            for la in alphas:
                loss, a = la
                t = t_crc(W2, loss, a)
                for tag, W, H, idx, keys, lab in (("test", WB, HB, B, kk, kinds), ("ood", WO, HO, oi, gg, groups)):
                    if not len(idx):
                        continue
                    r = out[la][c][tag]
                    r["size"].append(size(H, M.D[idx], t)); r["risk"].append(risk(W, t, loss))
                    for k in keys:
                        sel = [i for i, j in enumerate(idx) if lab[own[j]] == k]
                        r["by"].setdefault(str(k), []).append(risk([W[i] for i in sel], t, loss) if sel else np.nan)
    summ = lambda x: {"mean": float(np.mean(x)), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}
    rnd = lambda x: [None if not np.isfinite(y) else round(float(y), 5) for y in x]
    res = {}
    for la in alphas:
        res[f"{la[0]} {la[1]:g}"] = {c: {tag: {"n": int(len(ci) - len(ci) // 2) if tag == "test" else int(len(oi)),
                                               "size": summ(r["size"]), "risk": summ(r["risk"]),
                                               "risk_by": {k: float(np.nanmean(x)) for k, x in r["by"].items() if np.isfinite(x).any()},
                                               "split": {"size": rnd(r["size"]), "risk": rnd(r["risk"])}}
                                         for tag, r in v.items() if r["size"]} for c, v in out[la].items()}
    return res


def main(setting="in"):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.objective import posterior_from_qr
    from scipy.linalg import solve_triangular
    from .mv_conformal import kind_of
    from ..harness import RUNS
    from ..run import RESULTS
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.0f} s]", *a, flush=True)
    out_dir = RUNS / "final" / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    if setting == "in":
        train, test, extra = f"{DATA}/train.xyz", f"{DATA}/test.xyz", None
    else:
        train, test, extra = f"{OOD}/train.xyz", f"{OOD}/test.xyz", f"{OOD}/ood.xyz"
    cfg, data, prob, theta0 = problem(MODEL, train, test, out_dir)
    st = type_stats(cfg, data, prob, log)
    V, vtypes = eval_rows(prob, data.ds_test), [str(c.config_type) for c in data.test]
    n_cal = len(V)
    if extra:                                                 # the held-out configurations, same model and E0
        from ace_jax.fit.data import build_dataset, load_configs
        oc = load_configs(extra, virial_key="virial")             # the benchmark command's --virial-key
        ods = build_dataset(oc, data.meta, data.E0, cfg.batch, pack=cfg.pack_mode, log=lambda *_: None)
        V += eval_rows(prob, ods)
        vtypes += [str(c.config_type) for c in oc]
    assert len(V) == len(vtypes), (len(V), len(vtypes))
    kinds = np.array([kind_of(t) for t in vtypes])
    groups = np.array(vtypes)
    cal = np.arange(len(V)) < n_cal
    log(f"{setting}: fit {len(data.train)} configurations, calibration {n_cal}, held out {len(V) - n_cal}")
    prs = pairs(vtypes, [v["n"] for v in V], kinds)
    # pairs live on one side: a calibration pair has both ends there; a held-out one has a held-out first end (dE_form: a
    # calibration diamond second end; dE_type: two held-out configurations of one type)
    prs = {k: [(a, b, r) for a, b, r in pl if cal[a] == cal[b] or (not cal[a] and k == "dE_form")] for k, pl in prs.items()}
    types = sorted({str(c.config_type) for c in data.train})
    res = {"model": MODEL, "setting": setting, "train": train, "calibration": test, "held_out": extra, "n_fit": len(data.train),
           "n_cal": n_cal, "n_held": len(V) - n_cal, "pairs": {k: len(v) for k, v in prs.items()}, "candidates": {}, "views": {}}
    views = {}
    for name in NAMES:
        w, mode = candidates(types, "all")[name]
        qs = merge(st, w)
        th, lml, ok, nit = map_fit(qs, prob, theta0, mode)
        mu, L = posterior_from_qr(th, qs, prob)
        mu, L = np.asarray(mu), np.asarray(L)
        np.savez(out_dir / f"posterior_{name}.npz", mu=mu, L=L)
        vw, _, _ = views_of(V, mu, lambda P, L=L: solve_triangular(L, P, lower=True), prs)
        for vn, x in vw.items():
            views.setdefault(vn, {})[name] = x
        for shape, label in (("cr3", "CR3"), ("ellipsoid", "ellipsoid")):   # final_shapes.py, where it ran: same mean
            f = out_dir / f"{shape}_{name}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            dev = float(np.max(np.abs(z["mu"] - mu)) / np.max(np.abs(mu)))
            sw, _, _ = views_of(V, mu, lambda P, F=z["F"]: F.T @ P, prs)
            for vn, x in sw.items():
                views[vn][f"{name}/{label}"] = x
            res["candidates"][f"{name}/{label}"] = {"shape": json.loads(str(z["info"])), "mean_rel_dev": dev}
            log(f"{name}/{label}: factor {z['F'].shape}, its mean against B1's {dev:.1e}")
        err = {}
        for tag, msk in (("test", cal), ("ood", ~cal)):
            if msk.any():
                eE = [(V[j]["yE"] - V[j]["PE"] @ mu) / V[j]["n"] for j in np.flatnonzero(msk)]
                eF = np.concatenate([V[j]["yF"] - V[j]["PF"] @ mu for j in np.flatnonzero(msk)])
                err[tag] = {"rmse_E_meV_atom": 1e3 * float(np.sqrt(np.mean(np.square(eE)))), "rmse_F": float(np.sqrt(np.mean(eF ** 2)))}
        res["candidates"][name] = {"mode": mode, "theta": {k: float(x) for k, x in th._asdict().items() if "sigma" in k},
                                   "lml": lml, "map_ok": ok, "map_nit": nit, **err}
        log(f"{name:5s} sigma_c {np.exp(th.log_sigma_c):.3g} sigma {np.exp(th.log_sigma_E):.3g}  " +
            "  ".join(f"{t}: RMSE E {e['rmse_E_meV_atom']:.2f} meV/atom F {e['rmse_F']:.4f}" for t, e in err.items()))
    rng = np.random.default_rng(0)
    for vn, cands in views.items():
        las = [("joint", a) for a in ALPHAS] + ([("frac", a) for a in ALPHAS] if vn == "F" else [])
        res["views"][vn] = splits(cands, cal, ~cal, kinds, groups, las, rng)
        for la, r in res["views"][vn].items():
            print(f"\n{vn} view, {la}: size (log semi-axis), risk (alpha {la.split()[1]}), worst kind / held-out type")
            for c, x in r.items():
                for tag, y in x.items():
                    wk = max(y["risk_by"], key=y["risk_by"].get)
                    print(f"  {c:5s} {tag:4s} n {y['n']:4d} size {y['size']['mean']:+.3f} ({y['size']['p05']:+.3f}..{y['size']['p95']:+.3f})"
                          f" risk {y['risk']['mean']:.3f} ({y['risk']['p05']:.3f}..{y['risk']['p95']:.3f}) worst {wk} {y['risk_by'][wk]:.2f}")
    res["seconds"] = time.time() - t0
    json.dump(res, open(RESULTS / f"final_{setting}.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main(*sys.argv[1:2])
