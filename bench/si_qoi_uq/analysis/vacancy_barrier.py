"""Vacancy migration barrier: climbing-image refinement and delta-method rows (next-steps plan §3).

Input: the mean model's run of the framework's `vacancy-path-ref` test (harness `--mean`, SIQ_<m>_<v>): its converged
plain-NEB band (11 images, 63-atom cell at the model's a0) and the harness's barrier max_i E_i - E_0.

1. Climbing-image NEB from that band (ASE NEB, climb=True, same k = 0.1, FIRE to FMAX_CI), the mean model's
   ACECalculator; then Newton on grad_x E = 0 (JAX, fixed topology; pseudo-inverse for the three translations) from the
   climbing image and from image 0, so both are stationary to roundoff.  Index check: the saddle's Hessian has exactly
   one negative eigenvalue, the minimum none (beyond the three translational zeros).
2. Rows: both structures are stationary in x, so by the envelope theorem
       dEb/dtheta = Phi(x_s) - Phi(x_m) + [(dPhi_s - dPhi_m)/d(ln a) . theta] / a0 * da0/dtheta
   (cell and positions scale with a0 together, as delta.py's defect rows).  Eb from the rows is checked against
   the polished energies.
3. sigma of each method: headline lam_F |L_cr3^T g|, B1 q_b1 |L_post^T g|, B2 |L_pops^T g| (POPS box at the 50 %
   leverage cut, read as a Gaussian; on the POPS mean's own band), B3 |L_cr3^T g|.  Provisional until the lam = 1
   committee of `vacancy-path-ref` runs agrees with the delta sigma (the paper's 10 % rule).

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.vacancy_barrier o3d12 o4d20 -> results/vacancy_barrier.json
"""
import json, sys

import numpy as np

E_DFT = 0.299                  # CASTEP vacancy-path-ref, the GAP-18 paper's run (plan §3)
FMAX_CI = 1e-3
SKIN = 1.5


def _newton_stationary(efn, pos, cell, th, tol=1e-9, it=30):
    import jax
    import jax.numpy as jnp
    f = lambda x: efn(x.reshape(-1, 3), cell, th)
    g, H = jax.jit(jax.grad(f)), jax.jit(jax.hessian(f))
    x = jnp.asarray(pos.reshape(-1))
    for _ in range(it):
        gx = np.asarray(g(x))
        if np.max(np.abs(gx)) < tol:
            break
        x = x - np.linalg.lstsq(np.asarray(H(x)), gx, rcond=1e-10)[0]
    w = np.linalg.eigvalsh(np.asarray(H(x)))
    return np.asarray(x).reshape(-1, 3), float(np.max(np.abs(np.asarray(g(x))))), w


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import ase.io
    from ase.mep import NEB
    from ase.optimize import FIRE
    from ace_jax.calc.point import ACECalculator
    from ..delta import Diamond, scaling_rows
    from ..harness import ROOT, RUNS
    from ..propagate import qois
    from ..run import RESULTS, fit_dir, lin_for
    out = {"E_dft": E_DFT, "fmax_ci": FMAX_CI}
    for m in sys.argv[1:]:
        lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
        sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
        lamF, qb1 = float(sc["lam_cr3"]["F"]), float(sc["q_b1"])
        b = np.load(fit_dir(m, "pops") / "pops_box_lev50.npz")
        Lp = np.asarray(b["support"]) * ((np.asarray(b["hi"]) - np.asarray(b["lo"])) / np.sqrt(12))[None, :]
        res = {}
        for v in ("cr3", "pops"):
            name = f"SIQ_{m}_{v}"
            th = np.asarray(np.load(RUNS / "samples" / f"{m}_{v}.npz")["theta_mean"], np.float64)
            d = ROOT / "test-runner" / f"model-{name}-test-vacancy-path-ref"
            band = ase.io.read(d / f"model-{name}-vacancy-path.neb.xyz", ":")
            Eb_h = json.load(open(str(d) + "-properties.json"))["vacancy_path"]
            Eb_h = max(Eb_h) - Eb_h[0]
            qh = qois(f"SIQ_{m}_cr3", ROOT)
            dia = Diamond(lin, qh["diamond_a0"])
            a0 = dia.a0(th)
            da0 = dia.da0(th, a0)
            # 1. climbing image from the converged band, the model at this theta
            from ..harness import _model_at
            from ace_jax.eval.io import load
            s = dict(np.load(RUNS / "samples" / f"{m}_{v}.npz"))
            base, _, _ = load(str(s["model_npz"]), fold=False)
            calc = ACECalculator(str(s["model_npz"]), skin=0.0)
            calc.model = _model_at(base, s, th)
            imgs = [a.copy() for a in band]
            for a in imgs:
                a.calc = calc
            neb = NEB(imgs, k=0.1, climb=True, allow_shared_calculator=True)
            opt = FIRE(neb, logfile=None)
            conv = bool(opt.run(fmax=FMAX_CI, steps=3000))
            E_img = np.array([a.get_potential_energy() for a in imgs])
            i_s = int(np.argmax(E_img))
            # 2. Newton polish of the saddle and of image 0, then the rows
            rows = {}
            for tag, at in (("s", imgs[i_s]), ("m", imgs[0])):
                at = at.copy()
                efn = lin.energy_fn(at, SKIN)
                x, gmax, w = _newton_stationary(efn, at.positions, np.asarray(at.cell.array), th)
                dmax = float(np.max(np.linalg.norm(x - at.positions, axis=1)))
                at.positions = x
                P, dP = scaling_rows(lin, at, SKIN)
                E = float(P @ th) + lin.e_const(at)
                nneg = int(np.sum(w < -1e-6 * np.max(np.abs(w))))
                nzero = int(np.sum(np.abs(w) <= 1e-6 * np.max(np.abs(w))))
                rows[tag] = dict(P=P, dP=dP, E=E, gmax=gmax, n_neg=nneg, n_zero=nzero, newton_shift=dmax,
                                 w_low=w[:4].tolist())
                ase.io.write(d / f"siq_ci_{tag}.xyz", at)
            Eb = rows["s"]["E"] - rows["m"]["E"]
            G = rows["s"]["P"] - rows["m"]["P"] + float((rows["s"]["dP"] - rows["m"]["dP"]) @ th) / a0 * da0
            r = dict(Eb_harness=float(Eb_h), Eb_ci_band=float(E_img[i_s] - E_img[0]), Eb=float(Eb), i_saddle=i_s,
                     ci_steps=int(opt.nsteps), ci_converged=conv,
                     ci_band=(E_img - E_img[0]).tolist(), a0=float(a0),
                     **{f"{t}_{k}": rows[t][k] for t in rows for k in ("gmax", "n_neg", "n_zero", "newton_shift",
                                                                        "w_low")})
            np.save(RUNS / "delta" / f"{m}_{v}_vacancy_barrier_row.npy", G)
            p = Lp.shape[0]
            if v == "cr3":
                sig = {"headline": lamF * np.linalg.norm(np.asarray(lin.factor("cr3")).T @ G),
                       "B1": qb1 * np.linalg.norm(np.asarray(linp.factor("posterior")).T @ G),
                       "B3": np.linalg.norm(np.asarray(lin.factor("cr3")).T @ G)}
            else:
                sig = {"B2": np.linalg.norm(Lp.T @ G[:p])}
            r["sigma"] = {k: float(x) for k, x in sig.items()}
            r["z"] = {k: float((E_DFT - Eb) / x) for k, x in sig.items()}
            res[v] = r
            print(m, v, json.dumps({k: r[k] for k in ("Eb_harness", "ci_converged", "ci_steps", "Eb_ci_band", "Eb", "i_saddle", "s_n_neg", "m_n_neg",
                                                       "s_gmax", "m_gmax", "s_newton_shift", "sigma", "z")}), flush=True)
        out[m] = res
    json.dump(out, open(RESULTS / "vacancy_barrier.json", "w"), indent=1)


if __name__ == "__main__":
    main()
