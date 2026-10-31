"""Scale diagnostics on the held-out test split (spec §3).

For every test row b (energy, force component, virial component) the standardised score is
z = (y - y_hat) / |L^T b| for a covariance factor L (CR3: D^-1 R; B1: the posterior's D^-1 chol^-T).
Rows come from the fit's own assembly (`fit.rows.chunked_rows_fn` on a `build_dataset` of the test configs),
so their columns and the virial convention are the fitted ones; E0_pre is subtracted from y_E, as the fit does.

Global scales (RMS z = 1 per observable type): lam_E (energy per atom: the per-atom division cancels in z),
lam_F, lam_V, and lam_pool = the RMS over configurations of each configuration's mean over its present types
of (mean z^2 of that type): E, F and V weighted equally within a configuration, configurations equally.
"""
import numpy as np

from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import chunked_rows_fn

TEST_XYZ = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/test.xyz"


def test_scores(lin, factors, xyz=TEST_XYZ, log=print):
    """{name: per-row score arrays} for each factor in `factors` ({name: L (p, r)}), plus row metadata:
    kind ("E", "F", "V"), config index, config_type, Mondrian group (forces: the atom's group; E and V: the
    configuration's stratum = the most extreme group over its atoms), residual and |L^T b|."""
    import jax
    import jax.numpy as jnp
    cfgs = load_configs(xyz, virial_key="virial")
    meta = {"rcut": lin.rc, "elements": lin.z}
    ds = build_dataset(cfgs, meta, lin.E0_pre, 1, log=log)
    gcfg = GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ, C=int(ds.y_E.shape[1]),
                    e0_cols=lin.e0_cols)
    rows = chunked_rows_fn(lin.model, gcfg)
    th = np.asarray(lin.theta_hat)
    Ls = {k: jnp.asarray(v) for k, v in factors.items()}
    out = {k: [] for k in ("kind", "cfg", "ctype", "group", "resid", "n_atoms")}
    sd = {k: [] for k in factors}
    post = lin.post
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        r = rows(bt)
        cm, nm = np.asarray(bt.cfg_mask), np.asarray(bt.node_mask)
        ncfg = np.asarray(bt.node_cfg)
        grp = np.asarray(post.groups_of(bt)) if post is not None and post.group_consts is not None \
            else np.zeros(len(nm), int)
        live = np.flatnonzero(cm)
        gidx = {c: gc + n for n, c in enumerate(live)}
        stratum = {c: int(grp[(ncfg == c) & nm].max()) if ((ncfg == c) & nm).any() else 0 for c in live}
        P, kind, cfg, resid, nat = [], [], [], [], []
        E, F, V = np.asarray(r.E), np.asarray(r.F), np.asarray(r.V)
        for c in live:
            n = float(bt.n_atoms[c])
            if float(bt.w_E[c]) > 0:
                P.append(E[c]); kind.append("E"); cfg.append(c); resid.append(float(bt.y_E[c]) - E[c] @ th); nat.append(n)
            if float(bt.w_V[c]) > 0:
                for a in range(6):
                    P.append(V[c, a]); kind.append("V"); cfg.append(c); resid.append(float(bt.y_V[c, a]) - V[c, a] @ th)
                    nat.append(n)
        wF = np.asarray(bt.w_F)
        for a in np.flatnonzero(nm & (wF > 0)):
            for x in range(3):
                P.append(F[a, x]); kind.append("F"); cfg.append(int(ncfg[a])); nat.append(float(bt.n_atoms[ncfg[a]]))
                resid.append(float(bt.y_F[a, x]) - F[a, x] @ th)
        P = jnp.asarray(np.stack(P))
        for k, L in Ls.items():
            sd[k].append(np.asarray(jnp.linalg.norm(P @ L, axis=1)))
        node_of = {}                                      # force rows: the atom's own group
        f_groups = [int(grp[a]) for a in np.flatnonzero(nm & (wF > 0)) for _ in range(3)]
        n_cfgrows = len(kind) - len(f_groups)
        out["group"] += [stratum[c] for c in cfg[:n_cfgrows]] + f_groups
        out["kind"] += kind; out["resid"] += resid; out["n_atoms"] += nat
        out["cfg"] += [gidx[c] for c in cfg]
        out["ctype"] += [cfgs[gidx[c]].config_type for c in cfg]
        gc += len(live)
        del node_of
    res = {k: np.asarray(v) for k, v in out.items()}
    res["sd"] = {k: np.concatenate(v) for k, v in sd.items()}
    return res


def scales(res, name):
    """(table rows, lam dict) for factor `name`: per type and overall, the RMS z and the 68 / 95 % quantiles
    of |z|; lam_pool per the module docstring."""
    z = res["resid"] / res["sd"][name]
    kind, cfg = res["kind"], res["cfg"]
    rows, lam = [], {}
    for t in ("E", "F", "V"):
        m = kind == t
        a = np.abs(z[m])
        lam[t] = float(np.sqrt(np.mean(a ** 2)))
        rows.append(dict(factor=name, kind=t, breakdown="all", label="all", n=int(m.sum()), rms_z=lam[t],
                         q68=float(np.quantile(a, 0.68)), q95=float(np.quantile(a, 0.95))))
    per_cfg = []
    for c in np.unique(cfg):
        m = cfg == c
        per_cfg.append(np.mean([np.mean(z[m & (kind == t)] ** 2) for t in ("E", "F", "V") if (m & (kind == t)).any()]))
    lam["pool"] = float(np.sqrt(np.mean(per_cfg)))
    a = np.abs(z)
    rows.append(dict(factor=name, kind="pool", breakdown="all", label="all", n=len(per_cfg), rms_z=lam["pool"],
                     q68=float(np.quantile(a, 0.68)), q95=float(np.quantile(a, 0.95))))
    for br, lab in (("config_type", res["ctype"]), ("mondrian", res["group"])):
        for t in ("E", "F", "V"):
            for g in sorted(set(lab[kind == t].tolist()), key=str):
                m = (kind == t) & (lab == g)
                a = np.abs(z[m])
                rows.append(dict(factor=name, kind=t, breakdown=br, label=str(g), n=int(m.sum()),
                                 rms_z=float(np.sqrt(np.mean(a ** 2))), q68=float(np.quantile(a, 0.68)),
                                 q95=float(np.quantile(a, 0.95))))
    return rows, lam


def conformal_q(res, name, level=0.6827):
    """B1's single conformal scale: the finite-sample (ceil((n+1) level) / n) quantile of |z| pooled over E, F
    and V scores, each configuration's scores weighted equally (weights 1 / n_rows(config)), so a 200-atom
    cell does not outvote a 2-atom one."""
    z = np.abs(res["resid"] / res["sd"][name])
    cfg = res["cfg"]
    _, inv, cnt = np.unique(cfg, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    o = np.argsort(z)
    cw = np.cumsum(w[o]) / w.sum()
    n = len(cnt)
    lev = min(1.0, np.ceil((n + 1) * level) / n)
    return float(z[o][np.searchsorted(cw, lev - 1e-12)])
