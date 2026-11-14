"""Is the vacancy-migration saddle novel in descriptor space?  (next-steps plan §3: would missing transition-state
configurations show up in the Bayesian and CR3 estimators?)

For each model, the climbing-image saddle and the minimum of analysis/vacancy_barrier.py (siq_ci_{s,m}.xyz of the mean
model's run), and the migrating atom (largest displacement between them).  Three label-free measures, each ranked
against the paper's 17 537 test atoms:
  gamma      the CALM-like descriptor-novelty grade of analysis/novelty_groups_si.py (fitted on a 10 000-atom training
             pool in the normalised support space; > 1: outside every training cluster's radius)
  support    ace-jax's support diagnostic of the ARD posterior (forces_support: support_ok, support_q)
  sigma_F    the per-atom force sd of B1 (q_b1 x posterior) and the headline (lam_F x CR3) on the BLR fit, as in
             analysis/force_baselines.py
If the saddle environment is absent from training, gamma, support_q and the B1 sigma of the migrating atom sit in the
test atoms' top tail.  If they are typical, the saddle is covered by training data in descriptor space and the barrier
miss is model-form error on seen environments, which only residual-based estimators (CR3) can see.

    SIQ_RUNS=<BLR root> ARD_RUNS=<ARD root> python -m bench.si_qoi_uq.analysis.barrier_novelty o3d12 o4d20
        -> results/barrier_novelty.json
"""
import json, os, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "defect_uq" / "scoring"))
ARD_RUNS = pathlib.Path(os.environ.get("ARD_RUNS", "/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_ard137"))


def pct(x, ref):
    return float(100 * np.mean(ref <= x))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import ase.io
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.support import _proj, support_features
    import support_rebuild as sr
    from novelty_groups import fit_novelty, gamma_of
    from ..harness import ROOT as HROOT, RUNS
    from ..linear import observable_rows
    from ..run import RESULTS, lin_for
    out = {}
    for m in sys.argv[1:]:
        d = ARD_RUNS / "fits" / m / "cr3"
        work = ARD_RUNS / "novelty" / m
        D = np.load(work / "descriptors.npz")
        ref = sr.load_ref(work / "support_normalised.npz")
        nov = {}
        for k in D.files:
            if k.startswith("X_pool_"):
                z = int(k.split("_")[-1])
                nov[z] = fit_novelty(_proj(ref["pca"], z, support_features(D[k].astype(float), ref["features"])))
        T = np.load(work / "scored_test_test.npz")
        g_test = gamma_of(nov, ref, T["X"].astype(float), T["Z"].astype(int))
        calc = ACECalculator(str(d / "model.npz"), posterior=str(d / "posterior.npz"))
        fb = np.load(RESULTS / f"force_baselines_{m}.npz", allow_pickle=True)
        lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
        sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
        lamF, qb1 = float(sc["lam_cr3"]["F"]), float(sc["q_b1"])
        Lc, Lp = np.asarray(lin.factor("cr3")), np.asarray(linp.factor("posterior"))
        run = HROOT / "test-runner" / f"model-SIQ_{m}_cr3-test-vacancy-path-ref"
        S, M = ase.io.read(run / "siq_ci_s.xyz"), ase.io.read(run / "siq_ci_m.xyz")
        disp = S.positions - M.positions
        disp -= np.round(disp @ np.linalg.inv(S.cell.array)) @ S.cell.array
        hop = int(np.argmax(np.linalg.norm(disp, axis=1)))
        res = {"hop_atom": hop, "hop_disp_A": float(np.linalg.norm(disp[hop]))}
        for tag, at in (("saddle", S), ("minimum", M)):
            at = at.copy()
            at.calc = calc
            X, Z = calc.support_descriptors(at)
            g = gamma_of(nov, ref, np.asarray(X, float), np.asarray(Z, int))
            sup = calc.get_property("forces_support", at)
            _, BF, _ = observable_rows(at, lin)
            BF = np.asarray(BF).reshape(len(at), 3, -1)
            p = Lc.shape[0]
            sd_hl = lamF * np.sqrt(np.sum((BF[..., :p] @ Lc) ** 2, axis=(1, 2)))
            sd_b1 = qb1 * np.sqrt(np.sum((BF[..., :Lp.shape[0]] @ Lp) ** 2, axis=(1, 2)))
            r = {"gamma_hop": float(g[hop]), "gamma_hop_pct_test": pct(g[hop], g_test),
                 "gamma_max": float(g.max()), "gamma_max_pct_test": pct(g.max(), g_test),
                 "n_gamma_gt1": int((g > 1).sum()), "support_ok_all": bool(np.all(sup["support_ok"])),
                 "n_unsupported": int((~np.asarray(sup["support_ok"])).sum()),
                 "support_q_hop": float(np.asarray(sup["support_q"])[hop]),
                 "sdF_B1_hop": float(sd_b1[hop]), "sdF_B1_hop_pct_test": pct(sd_b1[hop], fb["sd_B1"]),
                 "sdF_hl_hop": float(sd_hl[hop]), "sdF_hl_hop_pct_test": pct(sd_hl[hop], fb["sd_headline"]),
                 "sdF_B1_max_pct_test": pct(sd_b1.max(), fb["sd_B1"]),
                 "sdF_hl_max_pct_test": pct(sd_hl.max(), fb["sd_headline"])}
            res[tag] = r
        res["test_frac_gamma_gt1"] = float(np.mean(g_test > 1))
        out[m] = res
        print(m, json.dumps(res, indent=1), flush=True)
    f = RESULTS / "barrier_novelty.json"
    old = json.load(open(f)) if f.exists() else {}
    json.dump(old | out, open(f, "w"), indent=1)                       # one model per call keeps the other


if __name__ == "__main__":
    main()
