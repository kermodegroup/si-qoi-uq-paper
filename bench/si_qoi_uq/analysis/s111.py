"""Si(111) surface QoIs (next-steps plan §1): delta rows, CR3 x lam_F intervals and the probability that the DAS order is right.

Harness tests (run unmodified for the mean model, `harness run <m>_cr3.npz --mean --tests ...`; harness.py restores the
`move_mask` array that ASE >= 3.23 turns into a constraint):
  surface-energy-111-relaxed               gamma_1x1 = (E_slab* - N e_b) / (2A): a 1x1x10 cell, opened by 10 A at the
                                           shuffle plane, perturbed and relaxed (no 2x1 Pandey: the cell is 1x1);
  surface-energy-111-2n+1_x_2n+1-relaxed   gamma_n = (E_n* - N_n e_b) / A - e111, n = 3, 5, 7, 9 (DAS, 8 layers, bottom
                                           layers fixed), e111 the unrelaxed 1x1 shuffle-plane surface (both faces).
Rows (as delta.py's surface_100): the envelope theorem for each relaxed slab (frames recorded by the harness), e_b from
the relaxed 8-atom cell, e111 from the unrelaxed cells rebuilt here with ASE at a0, plus the a0 term
dQ/da0 . da0/dtheta (relaxed slabs: homogeneous scaling of positions and cell, as the test scales them; e111: central
difference in a, the 10 A vacuum is not scaled).  Checks: Q from the rows against the harness's own values.
References: CASTEP (harness) for gamma_1x1 only; for DAS there is no CASTEP value, only DFT_literature (Solares et al.,
Langmuir 2005, shifted to the CASTEP 1x1 value), a different functional and protocol, so DAS is scored on order and
differences, not absolute values.  Joint Gaussian with Sigma = lam_F^2 L L^T: P(gamma_7 < gamma_5, gamma_7 < gamma_9), etc.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.s111 -> results/s111.json
"""
import json

import numpy as np

EV_A2 = 16.021766                 # eV/A^2 -> J/m^2
NS = (3, 5, 7, 9)
TR = "/storage/eng/essswb/silicon-gap/test-results"


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import ase.io
    from ase.lattice.cubic import Diamond as LDiamond
    from scipy.stats import multivariate_normal, norm
    from ..delta import Diamond, scaling_rows
    from ..harness import RUNS
    from ..propagate import ROOT, qois
    from ..run import RESULTS, lin_for
    lit = json.load(open(f"{TR}/model-DFT_literature-test-surface-energy-111-2n+1_x_2n+1-relaxed-properties.json"))
    castep = json.load(open(f"{TR}/model-CASTEP-test-surface-energy-111-relaxed-properties.json"))["surface_energy_111_relaxed"]
    gap = json.load(open(f"{TR}/model-GAP-6-test-surface-energy-111-2n+1_x_2n+1-relaxed-properties.json"))
    key = lambda n: f"surface_energy_111_{n}x{n}-reconstructed_relaxed"
    out = {"reference": {"castep_1x1": castep, "literature_das": {n: lit[key(n)] for n in NS},
                         "gap18_das": {n: gap[key(n)] for n in NS}, "literature_note": lit["comment"]}}
    for m in ("o3d12", "o4d20"):
        lin = lin_for(m, "cr3")
        th = np.asarray(lin.theta_hat, np.float64)
        name = f"SIQ_{m}_cr3"
        qh = qois(name, ROOT)
        d = Diamond(lin, qh["diamond_a0"])
        a0 = d.a0(th)
        da0 = d.da0(th, a0)
        lamF = float(json.load(open(RUNS / "diag" / f"{m}_scales.json"))["lam_cr3"]["F"])
        L = np.asarray(lin.factor("cr3"))
        rd = lambda t: ROOT / "test-runner" / f"model-{name}-test-{t}"
        harness = {**json.load(open(str(rd("surface-energy-111-relaxed")) + "-properties.json")),
                   **json.load(open(str(rd("surface-energy-111-2n+1_x_2n+1-relaxed")) + "-properties.json"))}

        def slab_term(slab, bulk8):
            """(S, G_S): S = (E_slab - N e_b) / A for a relaxed slab, with its row (envelope + a0 term)."""
            Pu, dPu = scaling_rows(lin, bulk8)
            P, dP = scaling_rows(lin, slab)
            A = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
            r = len(slab) / len(bulk8)
            S = (P - r * Pu) @ th / A
            G = (P - r * Pu) / A + (((dP - r * dPu) @ th) / a0 / A - 2 * S / a0) * da0
            return float(S), G

        def e111_rows(a):
            """R(a): e111 = R(a) . theta (the E0 constants cancel), the unrelaxed shuffle-plane surface at lattice a."""
            b = LDiamond(symbol="Si", latticeconstant=a, directions=[[1, -1, 0], [1, 0, -1], [1, 1, 1]])
            b *= (1, 1, 10)
            e = b.copy()
            e.positions[:, 2] += 2.0
            e.wrap()
            e.cell[2, :] += [0.0, 0.0, 10.0]
            Pb, _ = scaling_rows(lin, b)
            Pe, _ = scaling_rows(lin, e)
            A = np.linalg.norm(np.cross(b.cell[0], b.cell[1]))
            return 0.5 * (Pe - Pb) / A

        h = 1e-4 * a0
        R0 = e111_rows(a0)
        e111 = float(R0 @ th)
        G_e111 = R0 + float((e111_rows(a0 + h) - e111_rows(a0 - h)) @ th / (2 * h)) * da0
        Q, G = {}, {}
        fr = ase.io.read(rd("surface-energy-111-relaxed") / "siq_relaxed.xyz", ":")   # calls: bulk cell, relaxed slab
        S, GS = slab_term(fr[1], fr[0])
        Q["1x1"], G["1x1"] = 0.5 * S, 0.5 * GS
        fr = ase.io.read(rd("surface-energy-111-2n+1_x_2n+1-relaxed") / "siq_relaxed.xyz", ":")   # bulk, then n = 3..9
        for n, f in zip(NS, fr[1:]):
            S, GS = slab_term(f, fr[0])
            Q[f"{n}x{n}"], G[f"{n}x{n}"] = S - e111, GS - G_e111
        hk = {"1x1": "surface_energy_111_relaxed", **{f"{n}x{n}": key(n) for n in NS}}
        names = list(Q)
        Gm = np.stack([G[k] for k in names])
        Sig = lamF ** 2 * (Gm @ L) @ (Gm @ L).T
        sd = np.sqrt(np.diag(Sig))
        res = {"a0": a0, "lam_F": lamF, "e111_unrelaxed": e111, "qoi": {}}
        for i, k in enumerate(names):
            ref = castep if k == "1x1" else lit[key(int(k[0]))]
            res["qoi"][k] = {"Q_rows": Q[k], "Q_harness": harness[hk[k]], "sigma": float(sd[i]),
                             "ref": ref, "ref_kind": "CASTEP" if k == "1x1" else "literature (shifted)",
                             "z": float((ref - Q[k]) / sd[i]), "Q_J_m2": Q[k] * EV_A2, "sigma_J_m2": float(sd[i]) * EV_A2}
        # differences and order, DAS only
        das = [f"{n}x{n}" for n in NS]
        ix = {k: names.index(k) for k in das}
        diffs = {}
        for a_, b_ in (("5x5", "3x3"), ("7x7", "5x5"), ("9x9", "7x7")):
            v = np.zeros(len(names)); v[ix[a_]], v[ix[b_]] = 1, -1
            mu, s = float(v @ np.array([Q[k] for k in names])), float(np.sqrt(v @ Sig @ v))
            ref = lit[key(int(a_[0]))] - lit[key(int(b_[0]))]
            diffs[f"{a_}-{b_}"] = {"mean": mu, "sigma": s, "ref_literature": ref, "z": (ref - mu) / s,
                                   "P_negative": float(norm.cdf(-mu / s)), "corr": float(Sig[ix[a_], ix[b_]] / (sd[ix[a_]] * sd[ix[b_]]))}
        # P(7x7 lowest of the four DAS): Monte Carlo on the joint Gaussian
        rng = np.random.default_rng(0)
        Sd = Sig[np.ix_([ix[k] for k in das], [ix[k] for k in das])]
        smp = multivariate_normal(np.array([Q[k] for k in das]), Sd, allow_singular=True).rvs(200000, random_state=rng)
        res["P_7x7_lowest"] = float(np.mean(np.argmin(smp, 1) == das.index("7x7")))
        res["P_order_lit"] = float(np.mean((smp[:, 1] < smp[:, 0]) & (smp[:, 2] < smp[:, 1]) & (smp[:, 2] < smp[:, 3])))
        res["differences"] = diffs
        out[m] = res
        print(f"{m}: a0 {a0:.4f}, lam_F {lamF:.2f}")
        for k, v in res["qoi"].items():
            print(f"  {k:4s} Q {v['Q_rows']:.5f} (harness {v['Q_harness']:.5f}) sigma {v['sigma']:.5f} ref {v['ref']:.5f} z {v['z']:+.2f}  eV/A^2")
        for k, v in diffs.items():
            print(f"  {k}: {1e3 * v['mean']:+.3f} +- {1e3 * v['sigma']:.3f} meV/A^2 (lit {1e3 * v['ref_literature']:+.3f}, z {v['z']:+.2f}; "
                  f"P<0 {v['P_negative']:.3f}; corr {v['corr']:.2f})")
        print(f"  P(7x7 lowest) {res['P_7x7_lowest']:.3f}; P(literature order 3>5>7<9) {res['P_order_lit']:.3f}")
    json.dump(out, open(RESULTS / "s111.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
