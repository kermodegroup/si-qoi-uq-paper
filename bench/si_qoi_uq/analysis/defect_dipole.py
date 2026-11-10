"""Elastic dipole tensors and relaxation volumes of the relaxed defects, with CR3 x lam_F sigma; and the stress channel of
the training defect cells extrapolated to zero thermal amplitude (next-steps plan §3: a second, stress-based diagnostic of
the interstitial bias the energy discrepancy term measures).

(a) The relaxed QoI defects of the mean model (harness frames: vacancy, tetrahedral, hexagonal, dumbbell; 216-site cell at
a0, cell fixed).  Dipole tensor P = W_def - (n_def / n_b) W_bulk, W = -dE/d(eps) the virial (eV; Voigt xx yy zz yz xz xy),
so P = -V sigma_residual (positive trace: the defect pushes the cell outward), and relaxation volume dV = tr P / (3 B) with B
the model's own bulk modulus.  Rows by implicit differentiation, as delta.py: with u the relaxed coordinates (the hexagonal
interstitial's constrained z excluded) and s the homogeneous scaling of positions and cell with a0,
    dW/dtheta = BV + (dW/du) du*/dtheta + [dW/ds + (dW/du) du*/ds] da0/dtheta / a0,
    du*/dtheta = -K^-1 (dPhi/du)^T,  du*/ds = -K^-1 d(grad_u E)/ds,   K the Hessian in u (min-norm solve: translations).
Check: the bulk's total row is zero, since a0(theta) keeps the bulk at zero stress for every theta.
(b) Training labels: for each defect class, each cell's virial residual (DFT - model) minus n_i x the per-atom thermal trend of
the bulk 'dia' residuals (cells within 0.6 A^3/atom of the class's mean volume, as the energy estimator), regressed on
x = F_rms^2; the intercept is the model's dipole error extrapolated to a cold defect of the class (trace / 3 and the
deviatoric norm).  Virials exist for all 190 vacancy cells but only the 43 variable-cell interstitial snapshots (the
three fixed-cell MD runs and the divacancies have none).  No QoI value or DFT relaxation volume enters.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.defect_dipole -> results/defect_dipole.json
"""
import json

import numpy as np

DEFECT = ("interstitial", "vacancy")
GPA = 160.21766208        # GPa per eV/A^3


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import ase.io
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import chunked_rows_fn
    from ..delta import Diamond
    from ..harness import RUNS
    from ..linear import VOIGT, voigt_strain
    from ..propagate import ROOT, qois
    from ..run import RESULTS, lin_for
    E6 = jnp.asarray(np.stack([voigt_strain(a, b) for a, b in VOIGT]))
    out = {}
    for m in ("o3d12", "o4d20"):
        lin = lin_for(m, "cr3")
        th = np.asarray(lin.theta_hat, np.float64)
        thj = jnp.asarray(th)
        name = f"SIQ_{m}_cr3"
        qh = qois(name, ROOT)
        d = Diamond(lin, qh["diamond_a0"])
        a0 = d.a0(th)
        dln = d.da0(th, a0) / a0                                   # d ln a0 / d theta
        lamF = float(json.load(open(RUNS / "diag" / f"{m}_scales.json"))["lam_cr3"]["F"])
        L = np.asarray(lin.factor("cr3"))
        dz = np.load(RUNS / "delta" / f"{m}.npz")
        G = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
        B = qh["diamond_bulk_modulus"] / GPA                       # eV/A^3
        gB = G["diamond_bulk_modulus"][:len(th)] / GPA
        rd = lambda t: ROOT / "test-runner" / f"model-{name}-test-{t}"
        sd = lambda g: lamF * float(np.linalg.norm(L.T @ g))

        def virial_rows(at, fixed=()):
            """(W (6,), dW/dtheta (6, p)) at a relaxed frame, total derivative (relaxation and a0 terms)."""
            phi = lin.phi_fn(lin.graph(at, 1.0))
            pos0, cell0 = jnp.asarray(at.positions), jnp.asarray(at.cell.array)
            n3 = pos0.size
            free = np.setdiff1d(np.arange(n3), np.asarray(fixed, int))

            def Phi(u, s, eps):
                T = jnp.eye(3) + eps
                return phi(((pos0 + u.reshape(-1, 3)) * (1 + s)) @ T.T, (cell0 * (1 + s)) @ T.T)

            BV = lambda u, s: -jax.vmap(lambda e: jax.jvp(lambda x: Phi(u, s, x), (jnp.zeros((3, 3)),), (e,))[1])(E6)
            u0, z = jnp.zeros(n3), jnp.zeros(())
            gradE = lambda u, s: jax.grad(lambda v: Phi(v, s, jnp.zeros((3, 3))) @ thj)(u)
            BV0 = np.asarray(jax.jit(lambda: BV(u0, z))())
            W = BV0 @ th
            # chunked forward-mode Jacobians: vectorising all 3N tangents at once needs ~0.5 TB at o4d20
            def jac_cols(f, chunk=16):
                g = jax.jit(jax.vmap(lambda v: jax.jvp(f, (u0,), (v,))[1]))
                cols = []
                for i in range(0, n3, chunk):
                    Vb = np.zeros((chunk, n3)); k_ = min(chunk, n3 - i)
                    Vb[np.arange(k_), i + np.arange(k_)] = 1.0
                    cols.append(np.asarray(g(jnp.asarray(Vb)))[:k_])
                return np.concatenate(cols)                                   # (n3, out)
            K = jac_cols(lambda v: gradE(v, z))[np.ix_(free, free)]          # Hessian (symmetric)
            Jx = jac_cols(lambda v: Phi(v, z, jnp.zeros((3, 3)))).T[:, free]  # (p, nf)
            dWdu = np.asarray(jax.jit(jax.jacrev(lambda v: BV(v, z) @ thj))(u0))[:, free]                  # (6, nf)
            dWds = np.asarray(jax.jit(lambda: jax.jvp(lambda s: BV(u0, s) @ thj, (z,), (jnp.ones(()),))[1])())
            dgds = np.asarray(jax.jit(lambda: jax.jvp(lambda s: gradE(u0, s), (z,), (jnp.ones(()),))[1])())[free]
            fmax = float(np.max(np.abs(np.asarray(jax.jit(lambda: gradE(u0, z))())[free])))
            sol = np.linalg.lstsq(K, np.concatenate([Jx.T, dgds[:, None]], 1), rcond=1e-10)[0]
            dudt, duds = -sol[:, :-1], -sol[:, -1]
            gW = BV0 + dWdu @ dudt + np.outer(dWds + dWdu @ duds, dln)
            return W, gW, fmax

        fb = ase.io.read(rd("vacancy-energy") / "siq_relaxed.xyz", ":")
        fi = ase.io.read(rd("interstitial-energy") / "siq_relaxed.xyz", ":")     # unit cell, tetrahedral, hexagonal, dumbbell
        Wb, gWb, _ = virial_rows(fb[0])
        res = {"a0": a0, "B_GPa": qh["diamond_bulk_modulus"], "lam_F": lamF,
               "bulk_check": {"W_per_atom": (Wb / len(fb[0])).tolist(),
                              "sigma_of_bulk_row_eV": max(sd(g) for g in gWb[:3])}, "relaxed": {}}
        print(f"{m}: a0 {a0:.4f}, B {qh['diamond_bulk_modulus']:.1f} GPa; bulk W/atom {np.round(Wb / len(fb[0]), 6)} eV, "
              f"CR3 sigma of the bulk diagonal rows {res['bulk_check']['sigma_of_bulk_row_eV']:.2e} eV (should be ~0)", flush=True)
        hexfix = lambda at: [3 * (len(at) - 1) + 2]
        for k, at, fixed in (("vacancy", fb[1], ()), ("tetrahedral", fi[1], ()), ("hexagonal", fi[2], hexfix(fi[2])),
                             ("dumbbell", fi[3], ())):
            W, gW, fmax = virial_rows(at, fixed)
            r = len(at) / len(fb[0])
            P, gP = W - r * Wb, gW - r * gWb
            tr, gtr = float(P[:3].sum()), gP[:3].sum(0)
            dV, gdV = tr / (3 * B), gtr / (3 * B) - tr / (3 * B) / B * gB
            Om = abs(np.linalg.det(at.cell.array)) / len(at)                    # atomic volume (A^3)
            res["relaxed"][k] = {"n_atoms": len(at), "fmax_free": fmax, "P_voigt_eV": P.tolist(),
                                 "P_sigma_eV": [sd(g) for g in gP], "trP_eV": tr, "trP_sigma_eV": sd(gtr),
                                 "dV_A3": dV, "dV_sigma_A3": sd(gdV), "dV_over_Omega": dV / Om, "dV_over_Omega_sigma": sd(gdV) / Om}
            print(f"  {k:12s} fmax {fmax:.1e}  tr P {tr:+7.2f} +- {sd(gtr):5.2f} eV   dV/Omega {dV / Om:+.3f} +- {sd(gdV) / Om:.3f}"
                  f"   P diag {np.round(P[:3], 2)} off {np.round(P[3:], 2)}", flush=True)

        # (b) stress channel of the hot training defect cells
        cs_all = load_configs("/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/train.xyz", virial_key="virial")
        cs = [c for c in cs_all if str(c.config_type) in DEFECT + ("dia",) and c.virial is not None]
        ds = build_dataset(cs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1, log=lambda *_: None)
        rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                                   C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
        rv = np.zeros((len(cs), 6))
        gc = 0
        for i in range(ds.n_batches):
            bt = jax.tree.map(lambda x, i=i: x[i], ds)
            RV = np.asarray(rows(bt).V)
            for c in np.flatnonzero(np.asarray(bt.cfg_mask)):
                rv[gc] = np.asarray(bt.y_V[c]) - RV[c] @ th
                gc += 1
        nat = np.array([len(c.numbers) for c in cs])
        types = np.array([str(c.config_type) for c in cs], dtype=object)
        x = np.array([float(np.mean(np.asarray(c.forces) ** 2)) for c in cs])
        vol = np.array([abs(np.linalg.det(np.asarray(c.cell))) / len(c.numbers) for c in cs])
        res["training"] = {}
        for t in DEFECT:
            md = types == t
            mb = (types == "dia") & (np.abs(vol - vol[md].mean()) < 0.6)
            Xb = np.stack([np.ones(mb.sum()), x[mb]], 1)
            kb = np.linalg.lstsq(Xb, rv[mb] / nat[mb][:, None], rcond=None)[0]          # (2, 6) per-atom bulk trend
            u = rv[md] - nat[md][:, None] * (kb[0] + np.outer(x[md], kb[1]))            # (n, 6)
            ent = {}
            for sub, ms in (("all", np.ones(md.sum(), bool)),) + ((("215", nat[md] == 215), ("63", nat[md] == 63)) if t == "vacancy" else ()):
                if ms.sum() < 4:
                    continue
                X = np.stack([np.ones(ms.sum()), x[md][ms]], 1)
                XtXi = np.linalg.inv(X.T @ X)
                tr3 = u[ms, :3].sum(1)
                k_ = XtXi @ X.T @ tr3
                s2 = np.sum((tr3 - X @ k_) ** 2) / max(ms.sum() - 2, 1)
                dev = u[ms] - np.concatenate([np.repeat(tr3[:, None] / 3, 3, 1), np.zeros((ms.sum(), 3))], 1)
                kd = XtXi @ X.T @ dev
                ent[sub] = {"n": int(ms.sum()), "trP_resid_intercept_eV": float(k_[0]), "trP_resid_se_eV": float(np.sqrt(s2 * XtXi[0, 0])),
                            "trP_resid_slope": float(k_[1]), "trP_resid_mean_eV": float(tr3.mean()),
                            "x_range": [float(x[md][ms].min()), float(x[md][ms].max())],
                            "dev_intercept_norm_eV": float(np.sqrt(np.sum(kd[0, :3] ** 2) + 2 * np.sum(kd[0, 3:] ** 2))),
                            "dV_resid_intercept_over_Omega": float(k_[0] / (3 * B) / vol[md][ms].mean())}
                print(f"  training {t:12s} {sub:3s} n {ms.sum():3d}: tr P residual (DFT - model) at x -> 0 {k_[0]:+6.2f} +- "
                      f"{np.sqrt(s2 * XtXi[0, 0]):.2f} eV (mean over hot cells {tr3.mean():+.2f}; dV/Omega {ent[sub]['dV_resid_intercept_over_Omega']:+.3f})", flush=True)
            res["training"][t] = ent
        out[m] = res
    json.dump(out, open(RESULTS / "defect_dipole.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
