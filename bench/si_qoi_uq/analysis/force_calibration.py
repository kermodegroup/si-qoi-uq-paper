"""Per-atom force uncertainty on the test split: served Mondrian forces_std against the observed error (paper fig. 7).

Uses an `aj fit --uq ard` posterior (schema 3: CR3 shape R, Mondrian group constants and per-group lam_rms, q), here the
PR 137 fits (runs/si_qoi_uq_ard137).  Per test atom, from the fit's own force rows:
  V          the unscaled 3x3 shape (ARDPosterior.atom_shape), v = tr V
  forces_std the served per-atom sigma, lam_rms[g] sqrt(v)          (Mondrian, scales fitted on a train hold-out)
  sd_global  lam_F sqrt(v), lam_F the global test-split force scale  (the propagation's scale; in-sample on test)
  s          the anisotropic conformal score sqrt(e^T (V + eps v/3 I)^-1 e), covered when s <= q[g] (nominal 1 - alpha)
Atoms whose sqrt(v) is below 1e-3 of the median (force rows that vanish by site symmetry) are dropped, as in diagnostics.

    python -m bench.si_qoi_uq.analysis.force_calibration o3d12|o4d20   (SIQ_RUNS: an ARD run root; PYTHONPATH: its ace-jax)
"""
import json, sys

import numpy as np


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..diagnostics import TEST_XYZ
    from ..harness import RUNS
    from ..linear import LinearACE
    from ..run import RESULTS
    m = sys.argv[1]
    d = RUNS / "fits" / m / "cr3"
    lin = LinearACE(d / "model.npz", d / "posterior.npz")
    post = ARDPosterior.load(d / "posterior.npz")
    lamF = json.load(open(RUNS / "diag" / f"{m}_scales.json"))["lam_cr3"]["F"]
    tab = post.group_table
    cfgs = load_configs(TEST_XYZ, virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    th = np.asarray(lin.theta_hat)
    E, V, G, T = [], [], [], []
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        F = np.asarray(rows(bt).F)                                     # (Ncap, 3, p)
        nm, wF = np.asarray(bt.node_mask), np.asarray(bt.w_F)
        a = np.flatnonzero(nm & (wF > 0))
        live = np.flatnonzero(np.asarray(bt.cfg_mask))
        idx = {c: gc + j for j, c in enumerate(live)}
        e = np.asarray(bt.y_F)[a] - F[a] @ th
        Va = post.atom_shape(F[a])
        g = np.asarray(post.groups_of(bt))[a]
        E.append(e); V.append(Va); G.append(g)
        T += [cfgs[idx[int(c)]].config_type for c in np.asarray(bt.node_cfg)[a]]
        gc += len(live)
    E, V, G = np.concatenate(E), np.concatenate(V), np.concatenate(G)
    v = np.trace(V, axis1=1, axis2=2)
    keep = np.sqrt(np.maximum(v, 0)) > 1e-3 * np.median(np.sqrt(np.maximum(v, 0)))
    E, V, G, v = E[keep], V[keep], G[keep], v[keep]
    T = np.array(T, dtype=object)[keep]
    lam_g = np.asarray(tab["lam_rms"], float)[G]
    q_g = np.asarray(tab["q"], float)[G]
    reg = V + post.eps * (v / 3)[:, None, None] * np.eye(3)
    s = np.sqrt(np.einsum("na,nab,nb->n", E, np.linalg.inv(reg), E))
    out = dict(err_norm=np.linalg.norm(E, axis=1), forces_std=lam_g * np.sqrt(v), sd_global=lamF * np.sqrt(v), group=G,
               score=s, q=q_g, config_type=T, alpha=float(tab["alpha"]), lam_rms=np.asarray(tab["lam_rms"], float),
               q_table=np.asarray(tab["q"], float), n_cfg=np.asarray(tab["n_cfg"]), n_dropped=int((~keep).sum()), lam_F=lamF)
    np.savez(RESULTS / f"force_calibration_{m}.npz", **out)
    cov = np.mean(s <= q_g)
    rz = lambda sd: float(np.sqrt(np.mean((out["err_norm"] / sd) ** 2)))
    print(f"{m}: {len(s)} atoms ({out['n_dropped']} dropped); conformal coverage {cov:.3f} (nominal {1 - out['alpha']:.2f}); "
          f"RMS |e|/forces_std {rz(out['forces_std']):.3f}, RMS |e|/sd_global {rz(out['sd_global']):.3f}")
    for g in np.unique(G):
        mg = G == g
        print(f"  group {g}: {mg.sum():6d} atoms, lam {tab['lam_rms'][g]:.2f}, coverage {np.mean(s[mg] <= q_g[mg]):.3f}, "
              f"RMS ratio {np.sqrt(np.mean((out['err_norm'][mg] / out['forces_std'][mg]) ** 2)):.3f}")


if __name__ == "__main__":
    main()
