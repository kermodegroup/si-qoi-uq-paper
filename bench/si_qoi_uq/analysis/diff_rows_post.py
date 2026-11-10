"""Defect-bulk difference rows added to the headline (cr3) fit, both models: do they bring the defect QoIs within tolerance?

next-steps plan §3, after traj_jackknife.py: if the fit itself got the defect-bulk contrast right, the discrepancy term
would not be needed.  The o3d12 prototype (diff_rows.py, 2026-11-01) refitted with explicit rows at the benchmark BLR
prior, with the (volume, F_rms^2)-paired bulk partner, whose n_def / n_b ratio (median 108) amplifies the partner's noise.
Here, for both models:
  - the rows are the regression form of the headline discrepancy estimator (bias_term.py --form regression): for each
    training cell i of defect class t (interstitial, vacancy, divacancy; all cells of the class), d_i = b_i - n_i (B0 + B1 x_i),
    target y_i - n_i (b0 + b1 x_i), with (B, b) the per-atom thermal trend of the bulk 'dia' cells within 0.6 A^3/atom of
    the class's mean volume and x = F_rms^2; at theta_hat its residual is the u_i the discrepancy term extrapolates;
  - they are added to the stored cr3 posterior (MAP hyperparameters, ARD prior, per-quantity noise) as a rank-m update,
    each row weighted k x the cell's own energy-row weight w_E,i / sigma_E (k = 0 is the headline fit):
        A' = A + W^T W,  theta' = theta_hat + A^-1 W^T (I + W A^-1 W^T)^-1 r,   W = diag(k w_E / sigma_E) D,  r = W (t - D theta_hat),
    with A = Ds S Ds, S = C C^T the stored factor (triangular solves only, as traj_jackknife.py);
  - every delta-row QoI (and the vacancy barrier) moves linearly, Q' = Q + g . (theta' - theta_hat); err / tol and z against
    the headline CR3 x lam_F sigma (no discrepancy term) are reported for each k.  Large k changes A and hence the
    jackknife, so z at k > 0 is indicative; the harness check of chosen k (save_samples -> RUNS/samples/<m>_drp_k<k>.npz)
    gives the relaxed QoIs exactly;
  - test-split RMSE (E per atom, F, V per atom) for each k, streamed batch by batch.
k is a sweep, not a choice: nothing here is tuned on a QoI or on the test split.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.diff_rows_post o3d12|o4d20 -> results/diff_rows_post_<m>.json
"""
import csv, json, pathlib, shlex, sys, time

import numpy as np

ACC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace")
DEFECT = ("interstitial", "vacancy", "divacancy")
KS = (0, 1, 3, 10, 30, 100, 300)
Z95 = 1.959963984540054


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.linalg as sl
    from ace_jax import cli
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.rows import chunked_rows_fn
    from ace_jax.fit.stats import linear_qr_statistics
    from ..harness import RUNS, save_samples
    from ..propagate import tol_abs
    from ..run import RESULTS, fit_dir, lin_for
    m = sys.argv[1]
    t0 = time.time()
    lin = lin_for(m, "cr3")
    argv = shlex.split(open(ACC / m / "cmd.txt").read().strip())[1:]
    argv[argv.index("--out") + 1] = "/tmp/unused"
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test or "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/test.xyz")
    prob = build_problem(cfg, data).prob
    theta = Hypers(**json.load(open(ACC / m / "theta_map.json")))
    sig = {q: float(np.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
    z = np.load(fit_dir(m, "cr3") / "posterior.npz")
    dinv = np.asarray(z["dinv"], np.float64)
    n = len(dinv)
    C = np.zeros((n, n))
    C[np.tril_indices(n)] = z["chol_packed"]
    th = np.asarray(lin.theta_hat, np.float64)
    rows = chunked_rows_fn(prob.model, prob.cfg)

    sel = [i for i, c in enumerate(data.train) if str(c.config_type) in DEFECT + ("dia",)]
    cs = [data.train[i] for i in sel]
    ds = build_dataset(cs, data.meta, data.E0, cfg.batch, log=lambda *_: None)
    E, yE, wE = np.zeros((len(cs), n)), np.zeros(len(cs)), np.zeros(len(cs))
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda x, i=i: x[i], ds)
        RE = np.asarray(rows(bt).E)
        for c in np.flatnonzero(np.asarray(bt.cfg_mask)):
            E[gc], yE[gc], wE[gc] = RE[c], float(bt.y_E[c]), float(bt.w_E[c])
            gc += 1
    assert gc == len(cs)
    # check: one cell's QR-compressed energy row is +-(w_E row) (the weights the stored posterior was built with)
    k0 = next(j for j, c in enumerate(cs) if str(c.config_type) == "interstitial")
    qs = linear_qr_statistics(prob.model, prob.cfg, build_dataset([cs[k0]], data.meta, data.E0, cfg.batch, log=lambda *_: None))
    RE0 = np.asarray(qs.R_E)[0]
    wcheck = float(np.linalg.norm(np.abs(RE0) - np.abs(wE[k0] * E[k0])) / np.linalg.norm(RE0))
    print(f"{m}: {len(cs)} defect + dia cells, rows {time.time() - t0:.0f} s; weight check {wcheck:.1e}", flush=True)
    nat = np.array([len(c.numbers) for c in cs])
    types = np.array([str(c.config_type) for c in cs], dtype=object)
    x = np.array([float(np.mean(np.asarray(c.forces) ** 2)) for c in cs])
    vol = np.array([abs(np.linalg.det(np.asarray(c.cell))) / len(c.numbers) for c in cs])

    D, y, w, cls = [], [], [], []
    for t in DEFECT:
        md = types == t
        mb = (types == "dia") & (np.abs(vol - vol[md].mean()) < 0.6)
        Xb = np.stack([np.ones(mb.sum()), x[mb]], 1)
        Kb = np.linalg.solve(Xb.T @ Xb, Xb.T)
        BE, By = Kb @ (E[mb] / nat[mb][:, None]), Kb @ (yE[mb] / nat[mb])
        idx = np.flatnonzero(md)
        D.append(E[idx] - nat[idx][:, None] * (BE[0] + np.outer(x[idx], BE[1])))
        y.append(yE[idx] - nat[idx] * (By[0] + By[1] * x[idx]))
        w.append(wE[idx] / sig["E"])
        cls += [t] * len(idx)
        print(f"  {t}: {len(idx)} cells, {mb.sum()} bulk cells in the volume window", flush=True)
    D, y, w, cls = np.concatenate(D), np.concatenate(y), np.concatenate(w), np.array(cls)
    xd = np.concatenate([x[types == t] for t in DEFECT])
    nd = np.concatenate([nat[types == t] for t in DEFECT])
    U0 = sl.solve_triangular(C, dinv[:, None] * D.T, lower=True)      # C^-1 Ds^-1 D^T  (n, m)

    def intercepts(thv):
        """F -> 0 intercept of the difference residual per class (vacancy: 215-atom cells, the QoI's size)."""
        r = y - D @ thv
        out = {}
        for t in DEFECT:
            mt = (cls == t) & ((nd == 215) if t == "vacancy" else True)
            X = np.stack([np.ones(mt.sum()), xd[mt]], 1)
            k, *_ = np.linalg.lstsq(X, r[mt], rcond=None)
            out[t] = {"intercept_eV": float(k[0]), "rms_eV": float(np.sqrt(np.mean(r[cls == t] ** 2)))}
        return out

    dz = np.load(RUNS / "delta" / f"{m}.npz")
    G = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
    csvr = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "qoi_uq.csv"))
            if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
    lamF = float(json.load(open(RUNS / "diag" / f"{m}_scales.json"))["lam_cr3"]["F"])
    Lk = np.asarray(lin.factor("cr3"))
    qoi = {}
    for q, g in G.items():
        if q.startswith("energy:") or q not in csvr or csvr[q]["dft"] in ("", "nan"):
            continue
        qoi[q] = dict(g=g[:n], mean=float(csvr[q]["mean"]), dft=float(csvr[q]["dft"]))
    vbf = RUNS / "delta" / f"{m}_cr3_vacancy_barrier_row.npy"
    if vbf.exists():
        qoi["vacancy_barrier"] = dict(g=np.load(vbf)[:n], mean=json.load(open(RESULTS / "vacancy_barrier.json"))[m]["cr3"]["Eb"],
                                      dft=0.299)
    for q, v in qoi.items():
        v["sigma"] = lamF * float(np.linalg.norm(Lk.T @ v["g"]))
        v["tol"] = 0.1 if q == "vacancy_barrier" else tol_abs(q, v["dft"])

    def test_rmse(thv):
        se = {q: [0.0, 0] for q in "EFV"}
        for i in range(data.ds_test.n_batches):
            bt = jax.tree.map(lambda x_, i=i: x_[i], data.ds_test)
            r = rows(bt)
            cm, nm = np.asarray(bt.cfg_mask), np.asarray(bt.node_mask)
            na = np.asarray(bt.n_atoms)
            for c in np.flatnonzero(cm):
                if float(bt.w_E[c]) > 0:
                    se["E"][0] += ((float(bt.y_E[c]) - np.asarray(r.E)[c] @ thv) / na[c]) ** 2; se["E"][1] += 1
                if float(bt.w_V[c]) > 0:
                    e = (np.asarray(bt.y_V)[c] - np.asarray(r.V)[c] @ thv) / na[c]
                    se["V"][0] += float(e @ e); se["V"][1] += 6
            af = np.flatnonzero(nm & (np.asarray(bt.w_F) > 0))
            e = np.asarray(bt.y_F)[af] - np.asarray(r.F)[af] @ thv
            se["F"][0] += float(np.sum(e ** 2)); se["F"][1] += e.size
        return {"E_meV_per_atom": 1e3 * np.sqrt(se["E"][0] / se["E"][1]), "F_eV_per_A": np.sqrt(se["F"][0] / se["F"][1]),
                "V_meV_per_atom": 1e3 * np.sqrt(se["V"][0] / se["V"][1])}

    out = {"model": m, "weight_check": wcheck, "n_rows": {t: int((cls == t).sum()) for t in DEFECT}, "lam_F": lamF,
           "sigma_E": sig["E"], "k": {}}
    thetas = {}
    for k in KS:
        if k == 0:
            dth = np.zeros(n)
        else:
            U = k * w[None, :] * U0                                     # C^-1 Ds^-1 W^T
            r = k * w * (y - D @ th)
            dth = dinv * sl.solve_triangular(C, U @ np.linalg.solve(np.eye(len(r)) + U.T @ U, r), lower=True, trans="T")
        thv = th + dth
        thetas[k] = thv
        res = {"dtheta_rel": float(np.linalg.norm(dth) / np.linalg.norm(th)), "train_diff": intercepts(thv),
               "test": test_rmse(thv), "qoi": {}}
        for q, v in qoi.items():
            pred = v["mean"] + float(v["g"] @ dth)
            e = v["dft"] - pred
            res["qoi"][q] = {"pred": pred, "err": e, "err_over_tol": e / v["tol"], "z_headline_sigma": e / v["sigma"],
                             "within_tol": bool(abs(e) <= v["tol"]), "covered95": bool(abs(e / v["sigma"]) <= Z95)}
        qq = [v for kk, v in res["qoi"].items() if kk != "vacancy_barrier"]
        res["n_within_tol"], res["n_covered95"], res["n"] = sum(v["within_tol"] for v in qq), sum(v["covered95"] for v in qq), len(qq)
        out["k"][str(k)] = res
        dq = {kk: res["qoi"][kk]["err_over_tol"] for kk in res["qoi"] if "interstitial" in kk or "vacancy" in kk}
        print(f"k={k:4d} |dth|/|th| {res['dtheta_rel']:.3f}  test E {res['test']['E_meV_per_atom']:.2f} F {res['test']['F_eV_per_A']:.4f} "
              f"V {res['test']['V_meV_per_atom']:.1f}  interstitial intercept {res['train_diff']['interstitial']['intercept_eV']:+.3f}  "
              f"within tol {res['n_within_tol']}/{res['n']}, covered95 {res['n_covered95']}/{res['n']}", flush=True)
        other = [(kk, v) for kk, v in sorted(res["qoi"].items(), key=lambda kv: -abs(kv[1]["err_over_tol"]))
                 if "interstitial" not in kk and "vacancy" not in kk][:3]
        print("        err/tol " + "  ".join(f"{kk.split('_')[0][:5]} {v:+.1f}" for kk, v in dq.items())
              + "  | worst other " + ", ".join(f"{kk} {v['err_over_tol']:+.1f}" for kk, v in other), flush=True)
    for k in KS[1:]:
        save_samples(RUNS / "samples" / f"{m}_drp_k{k}.npz", f"{m}_drp_k{k}", fit_dir(m, "cr3") / "model.npz",
                     thetas[k][None], lin, theta_mean=thetas[k])
    json.dump(out, open(RESULTS / f"diff_rows_post_{m}.json", "w"), indent=1, default=float)
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
