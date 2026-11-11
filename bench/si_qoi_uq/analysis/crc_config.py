"""Conformal risk control with the whole-configuration score (plan §5; Fisher, Herbst, Kermode and Marzouk, arXiv:2609.31384,
Sec. 3.5): the counterpart of crc.py's per-atom regions, to see whether a score that sees cross-atom correlations ranks the
UQ methods differently.

Per test configuration c (D_c live force components, symmetry-zero atoms dropped as force_baselines.py) and method:
    Sigma_c = lam_s^2 C_c + nu^2 I,   C_c = M_c M_c^T (M_c = F_c L: CR3, B1, B2 factors) or F_c Sigma_e F_c^T (POPS
    ellipsoid, from its covariance, never an eigh factor of Sigma_e),
with (lam_s, nu) by maximum Gaussian likelihood (the nugget §0 showed is needed; mv_conformal.py). Whitened residual (symmetric
root, as Fisher et al.'s Delta): w_c = Sigma_c^-1/2 e_c = U (a / sqrt(lam_s^2 c + nu^2)) + (e - U a) / nu, C_c = U diag(c) U^T.
Regions scaled by t, two losses (both non-increasing in t, bounded by B = 1):
  joint     l_c(t) = 1{|w_c| / sqrt(D_c) > t}          the region is the Mahalanobis ellipsoid (CP; §4)
  frac      l_c(t) = #{i : |w_ci| > t} / D_c           the region is a box in whitened coordinates; controls the expected
                                                       fraction of a configuration's whitened components outside +-t
            (the monotone, bounded form of Fisher et al.'s quantile loss)
CRC: t_hat = inf{t : (sum_c l_c(t) + 1) / (n + 1) <= alpha}.  Size of a configuration's region: log t + log det Sigma_c / (2 D_c),
the log geometric semi-axis per force component (eV/A; same units as crc.py's per-atom size), component-weighted over the
evaluation configurations.
Splits (configurations): halves A / B, A split again: shape on A1, t on A2 (so the score is fixed before calibration), size
and risks on B; selection on thirds S / C / E (shape and choice on S with plain quantiles, shape refit and t by CRC on C,
risk on E).

    stage 1 (rows; a compute node for o4d20):  python -m bench.si_qoi_uq.analysis.crc_config cache o3d12|o4d20
        -> RUNS/diag/<m>_crc_config.npz
    stage 2:                                    python -m bench.si_qoi_uq.analysis.crc_config  -> results/crc_config.json
"""
import json, sys

import numpy as np

ALPHAS = (0.1, 0.05, 0.2)
N_SPLIT = 100
METHODS = ("headline", "B1", "B2", "ellipsoid")
ELLIPSE = {"o3d12": "_sp_lev50_it20000", "o4d20": "_sp_lev50_it5000"}


def cache(m):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..diagnostics import SYM_ZERO, TEST_XYZ
    from ..harness import RUNS
    from ..run import fit_dir, lin_for
    lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
    b = np.load(fit_dir(m, "pops") / "pops_box_lev50.npz")
    Lp2 = np.asarray(b["support"]) * ((np.asarray(b["hi"]) - np.asarray(b["lo"])) / np.sqrt(12))[None, :]
    facs = {"headline": np.asarray(lin.factor("cr3")), "B1": np.asarray(linp.factor("posterior")), "B2": Lp2}
    Se = np.asarray(np.load(fit_dir(m, "cr3").parent / f"pops_ellipse{ELLIPSE[m]}.npz")["Sigma_e"])
    Se = 0.5 * (Se + Se.T)
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    Lc = facs["headline"]
    per, gc = [], 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)
        a = np.flatnonzero(np.asarray(bt.node_mask) & (np.asarray(bt.w_F) > 0))
        E = np.asarray(bt.y_F)[a] - F[a] @ th
        nc = np.asarray(bt.node_cfg)[a]
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        for j, c in enumerate(live):
            s = nc == c
            per.append((gc + j, E[s], F[a][s]))
        gc += len(live)
    sd_all = np.concatenate([np.sqrt(np.sum((Fc @ Lc) ** 2, axis=(1, 2))) for _, _, Fc in per])
    cut = SYM_ZERO * np.median(sd_all)                      # as force_baselines.py: symmetry-zero atoms dropped
    out = {"cfg": [], "config_type": [], "e": [], **{f"{k}_{f}": [] for k in METHODS for f in ("c", "U")}}
    for c, E, Fc in per:
        keep = np.sqrt(np.sum((Fc @ Lc) ** 2, axis=(1, 2))) > cut
        if not keep.any():
            continue
        e = E[keep].reshape(-1)
        Fk = Fc[keep].reshape(-1, Fc.shape[-1])
        out["cfg"].append(c); out["config_type"].append(str(cfgs[c].config_type)); out["e"].append(e)
        for k, L in facs.items():
            U, S, _ = np.linalg.svd(Fk[:, :L.shape[0]] @ L, full_matrices=False)
            r = S > 1e-12 * S[0]
            out[f"{k}_c"].append(S[r] ** 2); out[f"{k}_U"].append(U[:, r])
        Fs = Fk[:, :Se.shape[0]]
        w, U = np.linalg.eigh(Fs @ Se @ Fs.T)               # D x D: the configuration's own covariance, not Sigma_e's
        r = w > 1e-12 * w.max()
        out["ellipsoid_c"].append(w[r]); out["ellipsoid_U"].append(U[:, r])
        print(f"{m}: config {c} D {len(e)}", flush=True)
    f = RUNS / "diag" / f"{m}_crc_config.npz"
    np.savez(f, cfg=np.array(out["cfg"]), config_type=np.array(out["config_type"], dtype=object),
             **{k: np.array(v, dtype=object) for k, v in out.items() if k not in ("cfg", "config_type")})
    print(f"{m}: cached {len(out['cfg'])} configurations -> {f}", flush=True)


class Method:
    """One model x method: per configuration e, eigenpairs (c, U) of C_c; Uᵀe and |e|^2 precomputed."""

    def __init__(self, z, k):
        self.e = list(z["e"]); self.c = list(z[f"{k}_c"]); self.U = list(z[f"{k}_U"])
        self.a = [U.T @ e for U, e in zip(self.U, self.e)]
        self.ee = np.array([e @ e for e in self.e]); self.D = np.array([len(e) for e in self.e], float)
        self.saa = np.array([a @ a for a in self.a])

    def nll(self, x, idx):
        lam, nu = np.exp(x)
        t = 0.0
        for j in idx:
            v = lam ** 2 * self.c[j] + nu ** 2
            r = len(self.c[j])
            t += np.sum(np.log(v)) + (self.D[j] - r) * np.log(nu ** 2) + np.sum(self.a[j] ** 2 / v) + (self.ee[j] - self.saa[j]) / nu ** 2
        return 0.5 * t

    def fit(self, idx):
        from scipy.optimize import minimize
        lam0 = np.sqrt(np.sum(self.ee[idx]) / max(np.sum([np.sum(self.c[j]) for j in idx]), 1e-300))
        x0 = np.log([lam0, 0.1 * np.sqrt(np.sum(self.ee[idx]) / np.sum(self.D[idx]))])
        return np.exp(minimize(lambda x: self.nll(x, idx), x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7}).x)

    def whiten(self, shape, idx):
        """Per configuration: w (symmetric-root whitened residual), and log det Sigma_c / (2 D_c)."""
        lam, nu = shape
        W, H = [], np.empty(len(idx))
        for n_, j in enumerate(idx):
            v = lam ** 2 * self.c[j] + nu ** 2
            W.append(self.U[j] @ (self.a[j] / np.sqrt(v)) + (self.e[j] - self.U[j] @ self.a[j]) / nu)
            H[n_] = (np.sum(np.log(v)) + (self.D[j] - len(v)) * np.log(nu ** 2)) / (2 * self.D[j])
        return W, H


def t_crc(W, loss, alpha, crc=True):
    """CRC threshold (B = 1) over the configurations' whitened residuals W; crc=False: the plain empirical version."""
    n = len(W)
    budget = (n + 1) * alpha - 1 if crc else n * alpha       # allowed sum of losses strictly above t
    if budget < 0:
        return np.inf
    if loss == "joint":
        r = np.sort([np.linalg.norm(w) / np.sqrt(len(w)) for w in W])[::-1]
        k = int(np.floor(budget + 1e-9))
        return float(r[k]) if k < n else 0.0
    x = np.concatenate([np.abs(w) for w in W])
    wt = np.concatenate([np.full(len(w), 1.0 / len(w)) for w in W])
    o = np.argsort(-x)
    cs = np.cumsum(wt[o])                                    # loss mass strictly above x[o][i] is cs[i - 1]
    i = int(np.searchsorted(cs, budget + 1e-12, side="right"))   # first index whose inclusion would exceed the budget
    return float(x[o][i]) if i < len(x) else 0.0


def risk(W, t, loss):
    if loss == "joint":
        return float(np.mean([np.linalg.norm(w) / np.sqrt(len(w)) > t for w in W]))
    return float(np.mean([np.mean(np.abs(w) > t) for w in W]))


def size(H, D, t):
    return float(np.sum(D * (H + np.log(t))) / np.sum(D)) if np.isfinite(t) else np.inf


def stage2():
    from ..harness import RUNS
    from ..run import RESULTS
    from .mv_conformal import kind_of
    z = {m: np.load(RUNS / "diag" / f"{m}_crc_config.npz", allow_pickle=True) for m in ("o3d12", "o4d20")}
    assert np.array_equal(z["o3d12"]["cfg"], z["o4d20"]["cfg"])
    kinds = np.array([kind_of(t) for t in z["o3d12"]["config_type"]])
    cands = {(m, k): Method(z[m], k) for m in z for k in METHODS}
    n = len(kinds)
    rng = np.random.default_rng(0)
    out = {"alphas": ALPHAS, "n_split": N_SPLIT, "n_cfg": n, "kinds": {k: int((kinds == k).sum()) for k in np.unique(kinds)},
           "shape_full": {f"{m}/{k}": [float(x) for x in M.fit(np.arange(n))] for (m, k), M in cands.items()}, "loss": {}}
    print("ML shape (lam_s, nu eV/A) on all configurations:", {k: [round(x, 4) for x in v] for k, v in out["shape_full"].items()})
    summ = lambda x: {"mean": float(np.mean(x)), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}
    LA = [(loss, a) for loss in ("joint", "frac") for a in ALPHAS]
    hv = {la: {c: {"size": [], "risk": [], "risk_by_kind": {k: [] for k in np.unique(kinds)}} for c in cands} for la in LA}
    first = {la: {c: 0 for c in cands} for la in LA}
    for sp in range(N_SPLIT):                                 # halves: shape on A1, CRC on A2, size and risk on B
        p = rng.permutation(n)
        A1, A2, Bi = p[: n // 4], p[n // 4: n // 2], p[n // 2:]
        pre = {}
        for c, M in cands.items():
            sh = M.fit(A1)
            pre[c] = (M.whiten(sh, A2)[0], *M.whiten(sh, Bi))
        for la in LA:
            sz = {}
            for c, M in cands.items():
                W2, W, H = pre[c]
                t = t_crc(W2, la[0], la[1])
                sz[c] = size(H, M.D[Bi], t)
                hv[la][c]["size"].append(sz[c]); hv[la][c]["risk"].append(risk(W, t, la[0]))
                for kk in hv[la][c]["risk_by_kind"]:
                    sel = [i for i, j in enumerate(Bi) if kinds[j] == kk]
                    if sel:
                        hv[la][c]["risk_by_kind"][kk].append(risk([W[i] for i in sel], t, la[0]))
            b = min(sz, key=sz.get)
            if np.isfinite(sz[b]):
                first[la][b] += 1
        if sp % 10 == 0:
            print(f"halves split {sp}", flush=True)
    sel_r = {la: [] for la in LA}
    chosen = {la: {} for la in LA}
    for _ in range(N_SPLIT):                                  # selection: S (shape + choice), C (refit + CRC), E (risk)
        p = rng.permutation(n)
        S, Cc, E = p[: n // 3], p[n // 3: 2 * n // 3], p[2 * n // 3:]
        pre = {}
        for c, M in cands.items():
            W, H = M.whiten(M.fit(S), S)
            shc = M.fit(Cc)
            pre[c] = (W, H, M.whiten(shc, Cc)[0], M.whiten(shc, E)[0])
        for la in LA:
            sz = {c: size(pre[c][1], cands[c].D[S], t_crc(pre[c][0], la[0], la[1], crc=False)) for c in cands}
            best = min(sz, key=sz.get)
            chosen[la][best] = chosen[la].get(best, 0) + 1
            t = t_crc(pre[best][2], la[0], la[1])
            sel_r[la].append(risk(pre[best][3], t, la[0]))
    out["loss"] = {loss: {} for loss in ("joint", "frac")}
    for la in LA:
        loss, alpha = la
        r = {"halves": {f"{m}/{k}": {"size": summ(v["size"]), "risk": summ(v["risk"]), "first": first[la][(m, k)] / N_SPLIT,
                                     "risk_by_kind": {kk: float(np.mean(x)) for kk, x in v["risk_by_kind"].items() if x}}
                        for (m, k), v in hv[la].items()},
             "selection": {"chosen": {f"{m}/{k}": c / N_SPLIT for (m, k), c in chosen[la].items()}, "risk": summ(sel_r[la])}}
        out["loss"][loss][f"{alpha:g}"] = r
        print(f"\n{loss} loss, alpha {alpha:g}: size = log geometric semi-axis per component (eV/A); risk; by kind; first")
        for c, v in sorted(r["halves"].items(), key=lambda kv: kv[1]["size"]["mean"]):
            print(f"  {c:20s} size {v['size']['mean']:+.3f} ({v['size']['p05']:+.3f}..{v['size']['p95']:+.3f}) risk "
                  f"{v['risk']['mean']:.3f}  by kind " + " ".join(f"{kk[:6]} {x:.2f}" for kk, x in v["risk_by_kind"].items())
                  + f"  first {v['first']:.2f}")
        print(f"  selection: chosen {r['selection']['chosen']}; risk on E {r['selection']['risk']['mean']:.3f} "
              f"({r['selection']['risk']['p05']:.3f}..{r['selection']['risk']['p95']:.3f})")
    json.dump(out, open(RESULTS / "crc_config.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "cache":
        cache(sys.argv[2])
    else:
        stage2()
