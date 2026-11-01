"""IRLS with automatically set targets: balance each error group against a reference scale (o3d12; explicit rows).

Groups g: (config type, quantity) for E (per atom), F (components), V (per atom), plus the defect-bulk difference rows
(analysis/diff_rows.py).  Weight multipliers m_g act on the structural weights; the prior and shared noise are the
benchmark BLR's.  Each iteration solves the weighted least squares (QR), measures r_g (RMSE, natural units) and sets

    e_g = r_g / s_g,    m_g <- m_g (e_g / gmean(e))^GAMMA,    m <- m / gmean(m),    m_g in [M_MIN, M_MAX]

so the fixed point equalises the normalised (out-of-sample) errors e_g.  Groups of types with fewer than N_MIN training configs keep
m_g = 1 (they would otherwise steer the balance), but are still reported.

r_g is the K_FOLDS cross-validated RMSE (folds by configuration within each type, by pair for the difference rows).
Reference scales s_g:
  regret      s_g = the group's CV floor: its CV r_g when the fit is weighted towards g alone (m_g = FLOOR_BOOST).
              Balanced = every group equally far from the best this model can do for it.  Uses no QoI information.
  tolerance   difference rows 0.1 eV per defect cell (the defect-energy QoI tolerance); energy rows 10 meV/atom (the
              phase-energy QoI tolerance); forces and virials keep their regret floors (no QoI tolerance applies).
              Uses the QoI tolerances (never DFT QoI values), so the weights are shaped by the same thresholds that
              score coverage: a circularity risk for the UQ comparison, reported as such.

    python -m bench.si_qoi_uq.analysis.irls_balance o3d12        (SIQ_RUNS: the BLR run root)
writes results/irls_balance_<m>.json and RUNS/samples/<m>_irls_<scheme>.npz for `harness run ... --mean`.
"""
import json, shlex, sys, time

import numpy as np

from .diff_rows import ACC, DEFECT, pair, rows_of

GAMMA, M_MIN, M_MAX, N_MIN, ITERS, TOL = 0.5, 1e-2, 1e2, 10, 60, 0.05
K_FOLDS, SEED = 5, 20261101
FLOOR_BOOST = 100.0
TOL_DIFF_EV, TOL_E_MEV = 0.1, 10.0


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
    sig = float(np.exp(theta.log_sigma_F))
    lam = np.asarray(linear_prior_diag(theta, prob))
    tr = rows_of(prob, data.ds_train, data.train, None)
    te = rows_of(prob, data.ds_test, data.test, None)
    types = np.array([c.config_type for c in data.train], dtype=object)
    nat = np.array([len(c.numbers) for c in data.train])
    pairs = pair(data.train)
    di, bi = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ratio = nat[di] / nat[bi]
    Dd = tr["E"][di] - ratio[:, None] * tr["E"][bi]
    yd = tr["yE"][di] - ratio * tr["yE"][bi]
    wd0 = tr["wE"][di]
    print(f"rows ready ({time.time() - t0:.0f} s)", flush=True)

    # groups: (type, q) with rows, then "diff"
    utypes = sorted(set(types), key=str)
    count = {t: int(np.sum(types == t)) for t in utypes}
    groups = []
    for t in utypes:
        mt = types == t
        if mt.any():
            groups.append((t, "E"))
        if np.any(mt[tr["cF"]]):
            groups.append((t, "F"))
        if len(tr["cV"]) and np.any(mt[tr["cV"]]):
            groups.append((t, "V"))
    groups.append(("diff", "D"))
    G = len(groups)
    active = np.array([g[0] == "diff" or count[g[0]] >= N_MIN for g in groups])
    gi = {g: i for i, g in enumerate(groups)}

    # Cross-validated groups.  Configurations of each type (and difference-row pairs) are split into K_FOLDS folds.
    # Each (group, fold) is compressed once by QR: fit rows (structural weights / sigma) -> (Rf, cf); error rows in
    # natural units (E, V per atom; F as is) -> (Re, ce, rest = |y|^2 - |c|^2, n).  A fit leaving fold k out is a QR of
    # the stacked small factors, and a group's CV error sums its held-out folds.  Training-set floors are useless here:
    # with p = 224 a boosted group of a few dozen energy rows is interpolated (floor ~0; surface/crack E reached
    # base/floor 140-3000), so floors and the balanced errors are both out of sample.
    rng = np.random.default_rng(SEED)
    fold_cfg = np.empty(len(types), int)
    for t in utypes:
        idx = np.flatnonzero(types == t)
        fold_cfg[idx] = rng.permutation(len(idx)) % K_FOLDS
    fold_pair = rng.permutation(len(yd)) % K_FOLDS

    def compress(P, y):
        if len(y) == 0:
            return np.zeros((0, len(lam))), np.zeros(0), 0.0
        Q, R = np.linalg.qr(P)
        c = Q.T @ y
        return R, c, float(y @ y - c @ c)
    eE_rows, eE_y = tr["E"] / nat[:, None], tr["yE"] / nat
    eV_rows, eV_y = tr["V"] / nat[tr["cV"], None], tr["yV"] / nat[tr["cV"]]
    comp = []                                   # comp[g][k] = (Rf, cf, Re, ce, rest, n, unit)
    for t, q in groups:
        per = []
        for k in range(K_FOLDS):
            if t == "diff":
                sel = fold_pair == k
                w = wd0[sel] / sig
                Rf, cf, _ = compress(Dd[sel] * w[:, None], yd[sel] * w)
                Re, ce, rest = compress(Dd[sel], yd[sel])
                per.append((Rf, cf, Re, ce, rest, int(sel.sum()), 1.0))
                continue
            mk = (types == t) & (fold_cfg == k)
            if q == "E":
                sel = mk; P, y, w = tr["E"][sel], tr["yE"][sel], tr["wE"][sel]; Pe, ye, unit = eE_rows[sel], eE_y[sel], 1e3
            elif q == "F":
                sel = mk[tr["cF"]]; P, y, w = tr["F"][sel], tr["yF"][sel], tr["wF"][sel]; Pe, ye, unit = P, y, 1.0
            else:
                sel = mk[tr["cV"]]; P, y, w = tr["V"][sel], tr["yV"][sel], tr["wV"][sel]; Pe, ye, unit = eV_rows[sel], eV_y[sel], 1e3
            Rf, cf, _ = compress(P * (w / sig)[:, None], y * w / sig)
            Re, ce, rest = compress(Pe, ye)
            per.append((Rf, cf, Re, ce, rest, int(len(y)), unit))
        comp.append(per)
    print(f"groups compressed ({G} x {K_FOLDS} folds, {time.time() - t0:.0f} s)", flush=True)
    sqlam = np.diag(np.sqrt(lam))

    def theta_at(mult, leave=None):
        blocks = [(mm * c[0], mm * c[1]) for mm, per in zip(mult, comp) for k, c in enumerate(per) if k != leave and len(c[1])]
        A = np.concatenate([sqlam] + [x for x, _ in blocks])
        b = np.concatenate([np.zeros(len(lam))] + [y for _, y in blocks])
        Q, R = np.linalg.qr(A)
        return np.linalg.solve(R, Q.T @ b)

    def cv_rmse(mult):
        """Per group: RMSE over its held-out folds, each predicted by the fit without that fold."""
        sse, n = np.zeros(G), np.zeros(G)
        for k in range(K_FOLDS):
            th = theta_at(mult, leave=k)
            for g, per in enumerate(comp):
                _, _, Re, ce, rest, nk, unit = per[k]
                if nk:
                    sse[g] += unit ** 2 * max(rest + float(np.sum((ce - Re @ th) ** 2)), 0.0)
                    n[g] += nk
        return np.sqrt(sse / np.maximum(n, 1))

    def train_rmse(th):
        out = np.zeros(G)
        for g, per in enumerate(comp):
            sse = sum(c[6] ** 2 * max(c[4] + float(np.sum((c[3] - c[2] @ th) ** 2)), 0.0) for c in per if c[5])
            out[g] = np.sqrt(sse / max(sum(c[5] for c in per), 1))
        return out

    ones = np.ones(G)
    th0 = theta_at(ones)
    r0 = cv_rmse(ones)
    floors = np.empty(G)
    for i in range(G):
        if not active[i]:
            floors[i] = r0[i]
            continue
        mm = ones.copy()
        mm[i] = FLOOR_BOOST
        floors[i] = cv_rmse(mm)[i]
    print(f"CV floors done ({int(active.sum())} active groups, {time.time() - t0:.0f} s)", flush=True)
    scales = {"regret": floors.copy(), "tolerance": floors.copy()}
    for i, (t, q) in enumerate(groups):
        if t == "diff":
            scales["tolerance"][i] = TOL_DIFF_EV
        elif q == "E":
            scales["tolerance"][i] = TOL_E_MEV

    def balance(s):
        mult, hist = ones.copy(), []
        for it in range(ITERS):
            r = cv_rmse(mult)
            e = r / s
            ea = e[active]
            spread = float(ea.max() / ea.min())
            hist.append({"iter": it, "spread": spread, "gmean_e": float(np.exp(np.mean(np.log(ea)))),
                         "diff_cv_rmse_eV": float(r[gi[("diff", "D")]])})
            if spread < 1 + TOL:
                break
            gm = np.exp(np.mean(np.log(ea)))
            upd = np.where(active, (e / gm) ** GAMMA, 1.0)
            mult = mult * upd
            mult[active] /= np.exp(np.mean(np.log(mult[active])))
            mult = np.clip(mult, M_MIN, M_MAX)
        th = theta_at(mult)
        return th, mult, cv_rmse(mult), hist

    report = {"model": m, "settings": {"gamma": GAMMA, "m_range": [M_MIN, M_MAX], "n_min": N_MIN, "iters": ITERS,
                                       "tol_spread": TOL, "floor_boost": FLOOR_BOOST, "tol_diff_eV": TOL_DIFF_EV,
                                       "tol_E_meV": TOL_E_MEV, "k_folds": K_FOLDS, "seed": SEED},
              "groups": [f"{t}/{q}" for t, q in groups], "active": active.tolist(), "base_cv_rmse": r0.tolist(), "base_train_rmse": train_rmse(th0).tolist(),
              "cv_floors": floors.tolist(), "schemes": {},
              "base_rel_dev_vs_blr": float(np.linalg.norm(th0 - LinearACE(RUNS / "fits" / m / "cr3" / "model.npz",
                                           RUNS / "fits" / m / "cr3" / "posterior.npz").theta_hat) / np.linalg.norm(th0))}
    lin = LinearACE(RUNS / "fits" / m / "cr3" / "model.npz")
    test_types = np.array([c.config_type for c in data.test], dtype=object)
    nat_te = np.array([len(c.numbers) for c in data.test])

    def test_err(th):
        eE = (te["yE"] - te["E"] @ th) / nat_te
        eF = te["yF"] - te["F"] @ th
        eV = (te["yV"] - te["V"] @ th) / nat_te[te["cV"]]
        q = lambda x: float(np.sqrt(np.mean(x ** 2)))
        out = {"all": {"E": 1e3 * q(eE), "F": q(eF), "V": 1e3 * q(eV)}}
        for t in DEFECT + ("dia",):
            mt = test_types == t
            out[t] = {"E": 1e3 * q(eE[mt]), "F": q(eF[mt[te["cF"]]])}
        return out
    report["base_test"] = test_err(th0)
    for name, s in scales.items():
        th, mult, r, hist = balance(s)
        e = r / s
        report["schemes"][name] = {"multipliers": mult.tolist(), "cv_rmse": r.tolist(), "normalised": e.tolist(),
                                   "history": hist, "converged": hist[-1]["spread"] < 1 + TOL,
                                   "n_capped": int(np.sum(np.isclose(mult, M_MIN) | np.isclose(mult, M_MAX))),
                                   "test": test_err(th)}
        save_samples(RUNS / "samples" / f"{m}_irls_{name}.npz", f"{m}_irls_{name}", RUNS / "fits" / m / "cr3" / "model.npz",
                     th[None], lin, theta_mean=th)
        t_ = report["schemes"][name]["test"]["all"]
        print(f"{name}: {len(hist)} iters, spread {hist[-1]['spread']:.3f}, capped {report['schemes'][name]['n_capped']}, "
              f"diff CV rmse {hist[0]['diff_cv_rmse_eV']:.3f} -> {hist[-1]['diff_cv_rmse_eV']:.3f} eV, diff weight x{mult[gi[('diff', 'D')]]:.3g}; "
              f"test E {t_['E']:.2f} F {t_['F']:.4f} V {t_['V']:.1f}", flush=True)
    json.dump(report, open(RESULTS / f"irls_balance_{m}.json", "w"), indent=1, default=float)
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
