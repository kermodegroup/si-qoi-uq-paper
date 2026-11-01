"""Defect-bulk difference rows, and per-type target accuracies by IRLS, for the linear fit (mean only; explicit rows).

The relaxed-interstitial QoIs are missed because the model's defect energy error drifts with thermal amplitude
across the hot training snapshots (RECON, "Why the interstitial formation energies are confidently wrong"), and
per-atom energy weighting makes 0.4-1.6 eV in a 217-atom cell cost only 2-7 meV/atom.  A difference row
d = b_def - (n_def / n_b) b_bulk with target E_def - (n_def / n_b) E_bulk is a linear combination of existing rows:
it adds no configuration, it reweights the fit towards the defect-bulk contrast.

Variants, all at the benchmark BLR prior (sigma_c) and shared noise, solved by QR of [diag sqrt(Lambda); W Phi]:
  base          the original rows (must reproduce the BLR mean)
  augment_k     + difference rows, weight k x the defect cell's own energy-row weight
  replace_k     the defect cells' energy rows dropped (forces, virials kept), difference rows at k x
  irls          weights per (config type, quantity) and for the difference rows, updated w <- w (rmse / target)
                until every group meets its target; targets: the base fit's own per-type training RMSE
                ("as accurate as now"), and TARGET_DIFF eV per defect cell for the difference rows
Pairing: each training defect snapshot with the bulk 'dia' snapshot nearest in (volume per atom, F_rms^2).

    python -m bench.si_qoi_uq.analysis.diff_rows o3d12        (SIQ_RUNS: the BLR run root; o3d12 only: rows in memory)
writes RUNS/diff_rows/<m>.npz (theta per variant) and results/diff_rows_<m>.json; the harness QoIs of each variant
come from `harness run <samples> --mean` on RUNS/samples/<m>_diff_<variant>.npz.
"""
import json, pathlib, shlex, sys, time

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")
DEFECT = ("interstitial", "vacancy", "divacancy")
KS = (1, 3, 10, 30)
TARGET_DIFF = 0.05          # eV per defect cell: half the 0.1 eV defect-energy QoI tolerance
IRLS_ITERS = 30


def rows_of(prob, ds, cfgs, lin_e0):
    """Explicit (unweighted) rows and targets of every config in ds, with their structural weights:
    E (C, p), yE, wE; F (Nat*3, p), yF, wF, cfg index per F row; V (nV*6, p), yV, wV, cfg index per V row."""
    import jax
    from ace_jax.fit.rows import chunked_rows_fn
    rows = chunked_rows_fn(prob.model, prob.cfg)
    E, yE, wE, F, yF, wF, cF, V, yV, wV, cV = ([] for _ in range(11))
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        r = rows(bt)
        cm, nm = np.asarray(bt.cfg_mask), np.asarray(bt.node_mask)
        ncfg = np.asarray(bt.node_cfg)
        live = np.flatnonzero(cm)
        idx = {c: gc + j for j, c in enumerate(live)}
        RE, RF, RV = np.asarray(r.E), np.asarray(r.F), np.asarray(r.V)
        for c in live:
            E.append(RE[c]); yE.append(float(bt.y_E[c])); wE.append(float(bt.w_E[c]))
            if float(bt.w_V[c]) > 0:
                V.append(RV[c]); yV.append(np.asarray(bt.y_V[c])); wV.append(np.full(6, float(bt.w_V[c]))); cV.append(np.full(6, idx[c]))
        a = np.flatnonzero(nm & (np.asarray(bt.w_F) > 0))
        F.append(RF[a].reshape(-1, RF.shape[-1])); yF.append(np.asarray(bt.y_F)[a].reshape(-1))
        wF.append(np.repeat(np.asarray(bt.w_F)[a], 3)); cF.append(np.repeat([idx[int(ncfg[j])] for j in a], 3))
        gc += len(live)
    cat = np.concatenate
    return dict(E=np.stack(E), yE=np.array(yE), wE=np.array(wE), F=cat(F), yF=cat(yF), wF=cat(wF), cF=cat(cF).astype(int),
                V=np.stack(V).reshape(-1, RE.shape[-1]) if V else np.zeros((0, RE.shape[-1])), yV=cat(yV) if yV else np.zeros(0),
                wV=cat(wV) if wV else np.zeros(0), cV=cat(cV).astype(int) if cV else np.zeros(0, int))


def pair(cfgs):
    """(defect index, bulk index) pairs: nearest 'dia' cell in (volume / atom, F_rms^2), scaled by their spreads."""
    v = np.array([abs(np.linalg.det(c.cell)) / len(c.numbers) for c in cfgs])
    x = np.array([float(np.mean(c.forces ** 2)) if c.forces is not None else np.nan for c in cfgs])
    t = np.array([c.config_type for c in cfgs], dtype=object)
    bulk = np.flatnonzero(t == "dia")
    out = []
    for i in np.flatnonzero(np.isin(t, DEFECT)):
        d = np.abs(v[bulk] - v[i]) / 0.3 + np.abs(x[bulk] - x[i]) / 0.1
        out.append((i, int(bulk[np.argmin(d)])))
    return out


def solve(blocks, lam):
    """theta minimising sum |w (y - Phi theta)|^2 + theta^T Lambda theta, by QR (blocks: (Phi, y, w) triples)."""
    A = np.concatenate([np.diag(np.sqrt(lam))] + [P * w[:, None] for P, y, w in blocks])
    b = np.concatenate([np.zeros(len(lam))] + [y * w for P, y, w in blocks])
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax import cli
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.objective import linear_prior_diag
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ..harness import RUNS, save_samples
    from ..linear import LinearACE
    from ..run import RESULTS
    m = sys.argv[1]
    t0 = time.time()
    argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
    argv[argv.index("--out") + 1] = "/tmp/unused"
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test or "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/test.xyz")
    prob = build_problem(cfg, data).prob
    theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
    sig = float(np.exp(theta.log_sigma_F))                       # shared noise
    lam = np.asarray(linear_prior_diag(theta, prob))
    tr = rows_of(prob, data.ds_train, data.train, None)
    te = rows_of(prob, data.ds_test, data.test, None)
    print(f"rows: train E {tr['E'].shape} F {tr['F'].shape} V {tr['V'].shape} ({time.time() - t0:.0f} s)", flush=True)
    types = np.array([c.config_type for c in data.train], dtype=object)
    nat = np.array([len(c.numbers) for c in data.train])
    pairs = pair(data.train)
    di, bi = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ratio = nat[di] / nat[bi]
    Dd = tr["E"][di] - ratio[:, None] * tr["E"][bi]
    yd = tr["yE"][di] - ratio * tr["yE"][bi]
    wd0 = tr["wE"][di]
    is_def = np.isin(types, DEFECT)
    keepE = ~is_def

    def blocks(wE_scale=None, wF_scale=None, wV_scale=None, diff_k=0.0, drop_def_E=False):
        sE = np.ones(len(types)) if wE_scale is None else wE_scale
        sF = np.ones(len(types)) if wF_scale is None else wF_scale
        sV = np.ones(len(types)) if wV_scale is None else wV_scale
        mE = keepE if drop_def_E else np.ones(len(types), bool)
        out = [(tr["E"][mE], tr["yE"][mE], tr["wE"][mE] * sE[mE] / sig),
               (tr["F"], tr["yF"], tr["wF"] * sF[tr["cF"]] / sig),
               (tr["V"], tr["yV"], tr["wV"] * sV[tr["cV"]] / sig)]
        if np.any(np.atleast_1d(diff_k) > 0):
            out.append((Dd, yd, wd0 * diff_k / sig))
        return out

    def errors(th, R, cfgs_types, nat_):
        """Per-type RMSE: E per atom (meV), F (eV/A), V per atom (meV); overall too."""
        eE = (R["yE"] - R["E"] @ th) / nat_
        eF = R["yF"] - R["F"] @ th
        eV = (R["yV"] - R["V"] @ th) / nat_[R["cV"]] if len(R["yV"]) else np.zeros(0)
        q = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else None
        out = {"all": {"E": 1e3 * q(eE), "F": q(eF), "V": 1e3 * q(eV) if len(eV) else None}}
        for t in sorted(set(cfgs_types), key=str):
            mt = cfgs_types == t
            out[t] = {"E": 1e3 * q(eE[mt]), "F": q(eF[mt[R["cF"]]]),
                      "V": (1e3 * q(eV[mt[R["cV"]]]) if np.any(mt[R["cV"]]) else None)}
        return out

    def diff_rms(th):
        return float(np.sqrt(np.mean((yd - Dd @ th) ** 2)))

    def trend(th):
        """Zero-thermal-amplitude intercept of the interstitial difference residual (as analysis/defect_trend.py, with
        the paired bulk as the reference): regress yd - Dd th on F_rms^2 of the defect cell."""
        x = np.array([float(np.mean(data.train[i].forces ** 2)) for i in di])
        r = yd - Dd @ th
        out = {}
        for t in DEFECT:
            mt = types[di] == t
            A = np.stack([np.ones(mt.sum()), x[mt]], 1)
            k, *_ = np.linalg.lstsq(A, r[mt], rcond=None)
            out[t] = {"intercept_eV": float(k[0]), "slope": float(k[1]), "mean_eV": float(r[mt].mean())}
        return out

    test_types = np.array([c.config_type for c in data.test], dtype=object)
    nat_te = np.array([len(c.numbers) for c in data.test])
    variants = {}
    th = solve(blocks(), lam)
    ref = LinearACE(RUNS / "fits" / m / "cr3" / "model.npz", RUNS / "fits" / m / "cr3" / "posterior.npz").theta_hat
    base_dev = float(np.linalg.norm(th - ref) / np.linalg.norm(ref))
    variants["base"] = th
    for k in KS:
        variants[f"augment_{k}"] = solve(blocks(diff_k=k), lam)
        variants[f"replace_{k}"] = solve(blocks(diff_k=k, drop_def_E=True), lam)
    # IRLS: per-(type, quantity) weights and the difference-row weight, to targets
    base_err = errors(th, tr, types, nat)
    tgt = {t: base_err[t] for t in base_err if t != "all"}
    utypes = sorted(set(types), key=str)
    sE, sF, sV = (np.ones(len(types)) for _ in range(3))
    kd = 1.0
    hist = []
    for it in range(IRLS_ITERS):
        th_i = solve(blocks(sE, sF, sV, diff_k=kd), lam)
        err = errors(th_i, tr, types, nat)
        dr = diff_rms(th_i)
        worst = max([dr / TARGET_DIFF] + [err[t][q] / tgt[t][q] for t in utypes for q in "EFV"
                                         if tgt[t][q] and err[t][q] is not None])
        hist.append({"iter": it, "diff_rms_eV": dr, "k_diff": kd, "max_ratio": worst})
        for t in utypes:
            mt = types == t
            for q, s in (("E", sE), ("F", sF), ("V", sV)):
                if tgt[t][q] and err[t][q] is not None:
                    s[mt] *= err[t][q] / tgt[t][q]
        kd *= dr / TARGET_DIFF
    variants["irls"] = th_i
    print(f"IRLS: diff rms {hist[0]['diff_rms_eV']:.3f} -> {hist[-1]['diff_rms_eV']:.3f} eV, worst ratio "
          f"{hist[-1]['max_ratio']:.3f}, k_diff {kd:.3g}", flush=True)
    report = {"model": m, "n_pairs": len(pairs), "base_rel_dev_vs_blr": base_dev, "target_diff_eV": TARGET_DIFF,
              "irls_history": hist, "irls_final_k_diff": kd, "variants": {}}
    out = RUNS / "diff_rows"
    out.mkdir(exist_ok=True)
    lin = LinearACE(RUNS / "fits" / m / "cr3" / "model.npz")
    for name, th_v in variants.items():
        report["variants"][name] = {"test": errors(th_v, te, test_types, nat_te)["all"],
                                    "test_by_type": {t: v for t, v in errors(th_v, te, test_types, nat_te).items()
                                                     if t in DEFECT + ("dia",)},
                                    "train_diff_rms_eV": diff_rms(th_v), "train_trend": trend(th_v)}
        save_samples(RUNS / "samples" / f"{m}_diff_{name}.npz", f"{m}_diff_{name}", RUNS / "fits" / m / "cr3" / "model.npz",
                     th_v[None], lin, theta_mean=th_v)
        print(f"{name:12s} test E {report['variants'][name]['test']['E']:6.2f} F {report['variants'][name]['test']['F']:.4f} "
              f"V {report['variants'][name]['test']['V']:6.1f} | diff rms {report['variants'][name]['train_diff_rms_eV']:.3f} eV | "
              f"interstitial intercept {report['variants'][name]['train_trend']['interstitial']['intercept_eV']:+.3f} eV", flush=True)
    np.savez(out / f"{m}.npz", names=np.array(list(variants)), thetas=np.stack(list(variants.values())))
    json.dump(report, open(RESULTS / f"diff_rows_{m}.json", "w"), indent=1, default=float)
    print(f"base vs BLR mean {base_dev:.2e}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
