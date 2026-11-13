"""Energy / force / virial weights chosen under CRC (plan §5 D, step 1), at one ladder model (default o4d16).

Selection never touches the test split: the training set T (train.xyz) is split by configuration type into T_fit and T_val
(20 % of each type with >= 2 configurations, seed SPLIT_SEED). Every candidate is fitted on T_fit and scored on T_val; the
winner is refitted on T and calibrated on the test split later (step 3).

Fit. The benchmark fit's command line (runs/accuracy_cost/ace/<m>/cmd.txt) with unit weights builds the problem once. Per
configuration type t and quantity q the weighted rows are compressed once to QR factors (R_tq, c_tq) (ace-jax
linear_qr_statistics on that type's configurations). A candidate's weights w_tq rescale them, w (R, c), and one QR of the
stacked blocks gives its QRStats exactly (yy, n, logw follow). The noise and prior scale are then the candidate's own MAP
(log marginal likelihood by QR, objective.log_marginal_likelihood_qr): one shared sigma, as the benchmark (--noise shared), or
sigma_E, sigma_F, sigma_V ("pq": the evidence's own energy / force / virial balance). Mean and factor: posterior_from_qr.

Score. B1 (the posterior factor; the whole-configuration front of the ladder, §5 C) on T_val: per configuration c the residual
and C_c = M_c M_c^T, M_c = rows_c L^-T, for two views, each as crc_config.py (Sigma_c = lam_s^2 C_c + nu^2 I by ML, symmetric-
root whitening, CRC threshold, size = log geometric semi-axis):
  F   the configuration's force components (eV/A; symmetry-zero atoms dropped as force_baselines.py)
  E   the configuration's energy per atom (eV/atom; one component, so joint = frac)
  dE_type, dE_form   energy differences E_a - (n_a / n_b) E_b (eV; `pairs`): the QoI proxy that stays within the rules
      (labelled training configurations only; plan §5 D principle 1)
Halves of T_val, as crc_config.py: shape on A1, t by CRC on A2, size and risk (overall and by kind) on B; the same splits for
every candidate (paired); "first" counts how often a candidate has the smallest size.

    python -m bench.si_qoi_uq.analysis.weight_sweep [o4d16 | o4d<d>_wL<w> [base|eladder|all|cr3|wl]]   (a compute node)  -> results/weight_sweep_<m>[_<set>].json
"""
import json, os, pathlib, shlex, sys, time

import numpy as np

ACC = "/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace"
DATA = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
SPLIT_SEED, VAL_FRAC = 20261112, 0.2
ALPHAS = (0.1, 0.05, 0.2)
N_SPLIT = int(os.environ.get("WSWEEP_SPLITS", 100))


def frames(path):
    """The extxyz file as text blocks, one per configuration, and each block's config_type."""
    lines = open(path).read().splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        out.append("".join(lines[i: i + n + 2]))
        i += n + 2
    ct = [b.split("\n", 2)[1].split("config_type=")[1].split()[0] if "config_type=" in b.split("\n", 2)[1] else "" for b in out]
    return out, ct


def split(out_dir):
    """T_fit / T_val by configuration type; written as xyz (text blocks of train.xyz, byte for byte)."""
    blocks, ct = frames(f"{DATA}/train.xyz")
    rng = np.random.default_rng(SPLIT_SEED)
    val = []
    for t in sorted(set(ct)):
        idx = rng.permutation([i for i, c in enumerate(ct) if c == t])
        if len(idx) >= 2:
            val.extend(idx[: min(max(1, int(round(VAL_FRAC * len(idx)))), len(idx) - 1)].tolist())
    val = sorted(val)
    fit = sorted(set(range(len(blocks))) - set(val))
    for name, idx in (("T_fit", fit), ("T_val", val)):
        with open(out_dir / f"{name}.xyz", "w") as f:
            f.write("".join(blocks[i] for i in idx))
    return fit, val, ct


def base_cmd(m):
    """(argv, theta start) for a model key: a ladder fit (o4d16: its cmd.txt and theta_map.json), or o<order>d<degree>_wL<w>
    (the wL sweep, plan section 5 D step 2): the o4d16 command line with --max-degree and --wL replaced, theta started at o4d16's
    MAP (sigma_c and sigma are refitted per candidate; the rest are not hyperparameters of the BLR fit)."""
    import re
    g = re.fullmatch(r"o(\d)d(\d+)(?:_wL([\d.]+))?", m)
    ref = m if g.group(3) is None else "o4d16"
    argv = shlex.split(open(f"{ACC}/{ref}/cmd.txt").read().strip())[1:]
    if g.group(3) is not None:
        argv[argv.index("--order") + 1], argv[argv.index("--max-degree") + 1] = g.group(1), g.group(2)
        argv += ["--wL", g.group(3)]
    return argv, f"{ACC}/{ref}/theta_map.json"


def resolve(weights, t):
    """A weights file's {E, F, V} for config_type t (named keys case-insensitive, else default), as load_configs."""
    for k, v in weights.items():
        if k != "default" and k.lower() == t.lower():
            return v
    return weights["default"]


E_LADDER = (1 / 3, 1, 2, 3, 5, 10, 30, 100)


def candidates(types, cset="base"):
    """name -> (per-type {E, F, V} weights, noise mode). cset "base": step 1's set; "eladder": the hand weights with every
    energy weight scaled by E_LADDER, shared noise; "all": both on the same splits (per-split sizes for selection.py); "cr3": the
    finalists, each scored with B1 and with CR3 (RUNS/wsweep/<m>/cr3_<name>.npz from weight_cr3.py)."""
    hand = json.load(open(f"{DATA}/weights_acejax.json"))
    hw = {t: dict(resolve(hand, t)) for t in types}
    sc = lambda f, q: {t: {**w, q: w[q] * f} for t, w in hw.items()}
    lad = {f"Ex{f:g}" if f >= 1 else f"E/{1 / f:g}": (sc(f, "E"), "shared") for f in E_LADDER}
    if cset == "eladder":
        return lad
    base = {"hand": (hw, "shared"), "hand_pq": (hw, "pq"),
            "E/3": (sc(1 / 3, "E"), "shared"), "Ex3": (sc(3, "E"), "shared"), "Ex10": (sc(10, "E"), "shared"),
            "noV": (sc(0, "V"), "shared"),
            "flat": ({t: dict(hand["default"]) for t in types}, "shared")}
    if cset == "all":                                         # both, on the same splits (Ex1 is hand)
        return {**base, **{k: v for k, v in lad.items() if k not in base and k != "Ex1"}}
    if cset in ("cr3", "wl"):                                 # the finalists (cr3: B1 and CR3, weight_cr3.py first)
        return {"hand": base["hand"], "Ex5": lad["Ex5"], "Ex10": base["Ex10"]}
    return base


def merge(st, w):
    """QRStats of the weighted union: per quantity the stacked w_tq [R_tq | c_tq], one QR."""
    import jax.numpy as jnp
    from ace_jax.fit.stats import QRStats
    f = {}
    for q in "EFV":
        blocks, yy, n, lw = [], 0.0, 0.0, 0.0
        for t, s in st.items():
            wt = w[t][q]
            if wt <= 0 or s[f"n_{q}"] == 0:
                continue
            blocks.append(wt * np.hstack([s[f"R_{q}"], s[f"c_{q}"][:, None]]))
            yy += wt ** 2 * s[f"yy_{q}"]; n += s[f"n_{q}"]; lw += s[f"logw_{q}"] + s[f"n_{q}"] * 2 * np.log(wt)
        L = next(iter(st.values()))["R_E"].shape[0]
        Rt = np.linalg.qr(np.vstack(blocks), mode="r") if blocks else np.zeros((L + 1, L + 1))
        f[q] = (jnp.asarray(Rt[:L, :L]), jnp.asarray(Rt[:L, L]), yy, n, lw)
    return QRStats(*(f[q][0] for q in "EFV"), *(f[q][1] for q in "EFV"), *(jnp.asarray(f[q][k]) for k in (2, 3, 4) for q in "EFV"))


def map_fit(qs, prob, theta0, mode):
    """MAP of (sigma_c, noise) by the QR log marginal likelihood; other hyperparameters fixed at the benchmark's."""
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    from ace_jax.fit.objective import log_marginal_likelihood_qr
    names = ["log_sigma_c", "log_sigma_E"] + (["log_sigma_F", "log_sigma_V"] if mode == "pq" else [])

    def theta_of(x):
        d = dict(theta0._asdict()); d.update(dict(zip(names, x)))
        if mode == "shared":
            d["log_sigma_F"] = d["log_sigma_V"] = d["log_sigma_E"]
        return type(theta0)(**d)
    vg = jax.jit(jax.value_and_grad(lambda x, qs: -log_marginal_likelihood_qr(theta_of(x), qs, prob)))   # qs an argument,
    x0 = np.array([theta0.log_sigma_c] + [theta0.log_sigma_F] * (len(names) - 1))                       # not a constant
    r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), qs)), x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": 500, "gtol": 1e-6})
    return theta_of(r.x), float(-r.fun), bool(r.success) or str(r.message), int(r.nit)


def pairs(vtypes, n_at, kinds, seed=SPLIT_SEED + 1):
    """Energy-difference proxies on T_val, fixed before any candidate is seen; (a, b, r): dE = E_a - r E_b, r = n_a / n_b.
    "dE_type": disjoint random pairs within each configuration type (relative energies of one kind of structure).
    "dE_form": each defect, surface or extended-defect configuration against a diamond configuration (a formation energy
    against the bulk, as the QoIs; the diamond configurations are reused in turn, each at most ceil(n_def / n_dia) times)."""
    rng = np.random.default_rng(seed)
    out = {"dE_type": [], "dE_form": []}
    for t in sorted(set(vtypes)):
        idx = rng.permutation([i for i, x in enumerate(vtypes) if x == t])
        out["dE_type"] += [(int(a), int(b), n_at[a] / n_at[b]) for a, b in zip(idx[0::2], idx[1::2])]
    dia = rng.permutation([i for i, x in enumerate(vtypes) if x == "dia"])
    dfx = [i for i, k in enumerate(kinds) if k in ("point defect", "surface", "extended")]
    out["dE_form"] = [(int(a), int(dia[j % len(dia)]), n_at[a] / n_at[dia[j % len(dia)]]) for j, a in enumerate(rng.permutation(dfx))]
    return out


class View:
    """crc_config.Method for prebuilt (e, c, U) lists."""

    def __new__(cls, e, c, U):
        from .crc_config import Method
        return Method({"e": e, "v_c": c, "v_U": U}, "v")


def halves(cands, kinds, alphas, rng):
    from .crc_config import risk, size, t_crc
    n = len(kinds)
    kk = np.unique(kinds)
    out = {la: {c: {"size": [], "risk": [], "by_kind": {k: [] for k in kk}} for c in cands} for la in alphas}
    first = {la: {c: 0 for c in cands} for la in alphas}
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        A1, A2, Bi = p[: n // 4], p[n // 4: n // 2], p[n // 2:]
        pre = {}
        for c, M in cands.items():
            sh = M.fit(A1)
            pre[c] = (M.whiten(sh, A2)[0], *M.whiten(sh, Bi), M.D[Bi])
        for la in alphas:
            loss, a = la
            sz = {}
            for c in cands:
                W2, W, H, D = pre[c]
                t = t_crc(W2, loss, a)
                sz[c] = size(H, D, t)
                out[la][c]["size"].append(sz[c]); out[la][c]["risk"].append(risk(W, t, loss))
                for k in kk:
                    sel = [i for i, j in enumerate(Bi) if kinds[j] == k]
                    out[la][c]["by_kind"][k].append(risk([W[i] for i in sel], t, loss) if sel else np.nan)
            first[la][min(sz, key=sz.get)] += 1
    summ = lambda x: {"mean": float(np.mean(x)), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}
    rnd = lambda x: [None if not np.isfinite(y) else round(float(y), 5) for y in x]     # per split (paired across candidates)
    return {f"{la[0]} {la[1]:g}": {c: {"size": summ(v["size"]), "risk": summ(v["risk"]), "first": first[la][c] / N_SPLIT,
                                       "risk_by_kind": {k: float(np.nanmean(x)) for k, x in v["by_kind"].items() if np.isfinite(x).any()},
                                       "split": {"size": rnd(v["size"]), "risk": rnd(v["risk"]),
                                                 "risk_by_kind": {k: rnd(x) for k, x in v["by_kind"].items()}}}
                                   for c, v in out[la].items()} for la in alphas}


def problem(m, train, test, out_dir):
    """The benchmark fit's problem for model key m with unit weights on (train, test) xyz paths: (cfg, data, prob, theta0)."""
    from ace_jax import cli
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    json.dump({"default": {"E": 1.0, "F": 1.0, "V": 1.0}}, open(out_dir / "unit_weights.json", "w"))
    argv, th_path = base_cmd(m)
    for k, v in (("--train", train), ("--weights", out_dir / "unit_weights.json"), ("--out", out_dir / "fit")):
        argv[argv.index(k) + 1] = str(v)
    argv += ["--test", str(test)]
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=a.test, log=lambda *_: None)
    return cfg, data, build_problem(cfg, data).prob, Hypers(**json.load(open(th_path)))


def type_stats(cfg, data, prob, log):
    """Per configuration type, the unit-weight QR factors of the training rows (structural 1/sqrt(n) kept)."""
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.stats import linear_qr_statistics
    st = {}
    for t in sorted({str(c.config_type) for c in data.train}):
        sub = [c for c in data.train if str(c.config_type) == t]
        ds = build_dataset(sub, data.meta, data.E0, cfg.batch, pack=cfg.pack_mode, log=lambda *_: None)
        q = linear_qr_statistics(prob.model, prob.cfg, ds)
        st[t] = {k: np.asarray(getattr(q, k)) for k in q._fields if getattr(q, k) is not None}
        log(f"type {t}: {len(sub)} configurations")
    return st


def eval_rows(prob, ds):
    """Unweighted rows of an evaluation dataset, one dict per configuration in file order (they do not depend on weights)."""
    import jax
    from ace_jax.fit.rows import linear_rows_bounded
    rows = jax.jit(lambda b: linear_rows_bounded(prob.model, prob.cfg, b))
    V = []
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda x, i=i: x[i], ds)
        r = rows(bt)
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        nc, nm = np.asarray(bt.node_cfg), np.asarray(bt.node_mask)
        for c in live:
            s = np.flatnonzero(nm & (nc == c))
            V.append({"cfg": len(V), "n": len(s), "yE": float(np.asarray(bt.y_E)[c]), "PE": np.asarray(r.E)[c],
                      "yF": np.asarray(bt.y_F)[s].reshape(-1), "PF": np.asarray(r.F)[s].reshape(-1, r.F.shape[-1])})
    return V


def views_of(V, mu, mfac, prs):
    """One candidate's views over evaluation rows V: {view: (View, owner)} (owner: the configuration index of each element:
    F drops configurations with no live force component; a pair's owner is its first configuration), RMSE E (eV/atom), F."""
    from ..diagnostics import SYM_ZERO
    eE, cE, UE, eF, cF, UF, sdF, rT, MT = [], [], [], [], [], [], [], [], []
    for v in V:
        ME = mfac(v["PE"])
        rT.append(v["yE"] - v["PE"] @ mu); MT.append(ME)          # total-energy residual and factor, for the pairs
        eE.append(np.array([(v["yE"] - v["PE"] @ mu) / v["n"]])); cE.append(np.array([ME @ ME / v["n"] ** 2])); UE.append(np.ones((1, 1)))
        MF = mfac(v["PF"].T).T
        sdF.append(np.sqrt(np.sum(MF.reshape(v["n"], 3, -1) ** 2, axis=(1, 2))))
        eF.append(v["yF"] - v["PF"] @ mu); UF.append(MF)
    cut = SYM_ZERO * np.median(np.concatenate(sdF))
    fF = []                                               # configurations with a live force component
    for j in range(len(V)):
        keep = np.repeat(sdF[j] > cut, 3)
        if not keep.any():
            continue
        U, S, _ = np.linalg.svd(UF[j][keep], full_matrices=False)
        r = S > 1e-12 * S[0]
        eF[j] = eF[j][keep]; UF[j] = U[:, r]; cF.append(S[r] ** 2); fF.append(j)
    eF, UF = [eF[j] for j in fF], [UF[j] for j in fF]
    out = {"E": (View(eE, cE, UE), np.arange(len(V)))}
    for pk, pl in prs.items():                            # eV per pair: dE = E_a - r E_b, g = M_a - r M_b
        g = [MT[a] - r * MT[b] for a, b, r in pl]
        out[pk] = (View([np.array([rT[a] - r * rT[b]]) for a, b, r in pl], [np.array([x @ x]) for x in g], [np.ones((1, 1))] * len(pl)),
                   np.array([a for a, _, _ in pl], int))
    out["F"] = (View(eF, cF, UF), np.array(fF, int))
    return out, float(np.sqrt(np.mean([e[0] ** 2 for e in eE]))), float(np.sqrt(np.mean(np.concatenate(eF) ** 2)))


def main(m="o4d16", cset="base"):
    import jax
    jax.config.update("jax_enable_x64", True)
    from scipy.linalg import solve_triangular
    from ace_jax.fit.objective import posterior_from_qr
    from .mv_conformal import kind_of
    from ..harness import RUNS
    from ..run import RESULTS
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.0f} s]", *a, flush=True)
    out_dir = RUNS / "wsweep" / m
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_i, val_i, ct = split(out_dir)
    log(f"T_fit {len(fit_i)}, T_val {len(val_i)} configurations")
    cfg, data, prob, theta0 = problem(m, out_dir / "T_fit.xyz", out_dir / "T_val.xyz", out_dir)
    types = sorted({str(c.config_type) for c in data.train})
    import pickle
    tmp = pathlib.Path(f"/tmp/essswb_wsweep_{m}.pkl")       # node-local cache of the per-type factors and T_val rows
    cached = pickle.load(open(tmp, "rb")) if tmp.exists() else None
    st = cached["st"] if cached else type_stats(cfg, data, prob, log)
    V = cached["V"] if cached else eval_rows(prob, data.ds_test)
    gc = len(V)
    assert gc == len(val_i), (gc, len(val_i))
    if not cached:
        pickle.dump({"st": st, "V": V}, open(tmp, "wb"), protocol=5)
    vtypes = [str(c.config_type) for c in data.test]
    kinds = np.array([kind_of(t) for t in vtypes])
    log(f"T_val rows cached: {gc} configurations, {sum(v['n'] for v in V)} atoms")
    res = {"model": m, "split_seed": SPLIT_SEED, "val_frac": VAL_FRAC, "n_fit": len(fit_i), "n_val": len(val_i),
           "val_kinds": {k: int((kinds == k).sum()) for k in np.unique(kinds)}, "candidates": {}}
    views = {"E": {}, "F": {}, "dE_type": {}, "dE_form": {}}
    n_at = [v["n"] for v in V]
    prs = pairs(vtypes, n_at, kinds)
    res["pairs"] = {k: len(v) for k, v in prs.items()}
    variants = []                                             # (name, weights, mode, factor): B1, and CR3 where weight_cr3 ran
    for name, (w, mode) in candidates(types, cset).items():
        variants.append((name, w, mode, "B1"))
        if cset == "cr3":
            variants.append((f"{name}/CR3", w, mode, "CR3"))
    post = {}
    for name, w, mode, fac in variants:
        base = name.split("/")[0]
        if base not in post:
            qs = merge(st, w)
            th, lml, ok, nit = map_fit(qs, prob, theta0, mode)
            mu, L = posterior_from_qr(th, qs, prob)
            post[base] = (th, lml, ok, nit, np.asarray(mu), np.asarray(L))
        th, lml, ok, nit, mu, L = post[base]
        if fac == "B1":
            mfac = lambda P, L=L: solve_triangular(L, P, lower=True)          # (r, ...) columns: P^T -> L^-1 P^T
        else:
            z = np.load(out_dir / f"cr3_{base}.npz")
            dev = float(np.max(np.abs(z["mu"] - mu)) / np.max(np.abs(mu)))
            log(f"{name}: CR3 rank {z['Lcr3'].shape[1]}, max leverage {float(z['lev'].max()):.4f}, mean deviation {dev:.1e}")
            mfac = lambda P, Lf=z["Lcr3"]: Lf.T @ P
        vw, rmse_E, rmse_F = views_of(V, mu, mfac, prs)
        for vn, (vv, _) in vw.items():
            views[vn][name] = vv
        fF = vw["F"][1]
        kinds_F = kinds[fF]
        res["n_val_F"] = len(fF)
        res["candidates"][name] = {"mode": mode, "weights": w, "theta": {k: float(x) for k, x in th._asdict().items() if "sigma" in k},
                                   "lml": lml, "map_ok": ok, "map_nit": nit, "rmse_E_meV_atom": 1e3 * rmse_E, "rmse_F": rmse_F}
        log(f"{name:8s} sigma_c {np.exp(th.log_sigma_c):.3g} sigma E/F/V {np.exp(th.log_sigma_E):.3g}/{np.exp(th.log_sigma_F):.3g}/"
            f"{np.exp(th.log_sigma_V):.3g}  T_val RMSE E {1e3 * rmse_E:.2f} meV/atom, F {rmse_F:.4f} eV/A  (MAP {nit} it, ok {ok})")
    rng = np.random.default_rng(0)
    res["halves"] = {}
    for vname, cands in views.items():
        las = [("joint", a) for a in ALPHAS] + ([("frac", a) for a in ALPHAS] if vname == "F" else [])
        kv = {"E": kinds, "F": kinds_F, **{pk: kinds[[a for a, _, _ in pl]] for pk, pl in prs.items()}}[vname]
        res["halves"][vname] = halves(cands, kv, las, rng)
        for la, r in res["halves"][vname].items():
            print(f"\n{vname} view, {la}: size (log semi-axis), risk, worst kind, first")
            for c, x in sorted(r.items(), key=lambda kv: kv[1]["size"]["mean"]):
                wk = max(x["risk_by_kind"], key=x["risk_by_kind"].get)
                print(f"  {c:8s} size {x['size']['mean']:+.3f} ({x['size']['p05']:+.3f}..{x['size']['p95']:+.3f}) risk "
                      f"{x['risk']['mean']:.3f} worst {wk} {x['risk_by_kind'][wk]:.2f} first {x['first']:.2f}")
    res["seconds"] = time.time() - t0
    json.dump(res, open(RESULTS / (f"weight_sweep_{m}.json" if cset == "base" else f"weight_sweep_{m}_{cset}.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main(*sys.argv[1:3])
