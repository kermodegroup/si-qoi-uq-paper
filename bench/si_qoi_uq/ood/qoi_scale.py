"""Does the learned per-atom scale matter for the QoIs?  A per-QoI diagnostic, not a propagated distribution.

The learned scale (ood/learned_scale.py) multiplies each atom's force region by e^{f(phi)/2}; it defines no covariance in
theta, so it cannot be propagated.  As a diagnostic, each QoI's sigma is rescaled by the learned factor of the atoms of its
target structure (the mean model's relaxed frames, harness siq_relaxed.xyz), relative to the factor's mean over the test
atoms (what the global lam_F is calibrated against):

    c_Q = mean_{i in Q} e^{f(phi_i)/2} / mean_{test atoms} e^{f(phi_i)/2}      (mean over the five split seeds)

and c_Q^max with the largest single-atom factor (an upper bound for defects, where the plain mean is diluted by the
bulk-like atoms).  Structures: elastic constants and bulk modulus, diamond; phase energies, the relaxed phase; defects, the
relaxed defect cell; surface, the relaxed slab; gamma_us, the stacking-fault frames; decohesion, diamond (the opened cells
are not stored).  Coverage of the CR3 x lam_F intervals (results/qoi_uq.csv; with the headline discrepancy term,
results/discrepancy.csv) is recounted with sigma_Q c_Q.

    python -m bench.si_qoi_uq.ood.qoi_scale o3d12   (ARD_RUNS, SIQ_RUNS: the BLR run root)  -> results/qoi_scale_<m>.json
"""
import csv, json, os, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "defect_uq" / "scoring"))
ARD = pathlib.Path(os.environ.get("ARD_RUNS", "/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_ard137"))
H = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/harness/test-runner")
D0 = pathlib.Path("/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost")
PHASES = ("beta-Sn", "sh", "hex_diamond", "bc8", "st12", "hcp", "fcc", "bcc")


def structures(m):
    """{QoI key: [atoms ...]} from the mean model's relaxed frames."""
    import ase.io
    rd = lambda t: ase.io.read(H / f"model-SIQ_{m}_cr3-test-{t}" / "siq_relaxed.xyz", ":")
    dia = rd("bulk_diamond")[0]
    out = {k: [dia] for k in ("diamond_bulk_modulus", "diamond_c11", "diamond_c12", "diamond_c44", "decohesion_max_stress")}
    for ph in PHASES:
        out[f"dE_{ph} (meV/at)"] = [rd(f"bulk_{ph}")[0]]
    out["vacancy_energy"] = [rd("vacancy-energy")[1]]
    fr = rd("interstitial-energy")                                   # unit cell, tetrahedral, hexagonal, dumbbell
    for k, f in (("tetrahedral", fr[1]), ("hexagonal", fr[2]), ("dumbbell", fr[3])):
        out[f"{k}_interstitial_energy"] = [f]
    out["surface_100 (J/m2)"] = [rd("surface-energy-100-relaxed")[1]]
    sf = rd("stacking-fault-surface")[1:]
    out["gamma_us glide (J/m2)"] = sf
    out["gamma_us shuffle (J/m2)"] = sf
    return out


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import support_rebuild as sr
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    from ..propagate import COVER_KEYS
    from ..run import RESULTS
    from .learned_scale import SEEDS, features, fit_scale
    m = sys.argv[1]
    d, work = ARD / "fits" / m / "cr3", ARD / "novelty" / m
    post = ARDPosterior.load(d / "posterior.npz")
    calc = ACECalculator(str(d / "model.npz"), posterior=str(d / "posterior.npz"))
    ref = sr.load_ref(work / "support_normalised.npz")
    D = np.load(work / "descriptors.npz")
    s_c, cfg_c = np.asarray(post.cal["scores"], float), np.asarray(post.cal["cfg"])
    Phi_c = features(ref, D["X_cal"].astype(float), D["Z_cal"].astype(int))
    ok = s_c > 0
    betas = []
    for seed in SEEDS:                                   # the same splits and fits as learned_scale.py
        rng = np.random.default_rng(seed)
        u = np.unique(cfg_c)
        A = set(rng.permutation(u)[: len(u) // 2].tolist())
        inA = np.array([c in A for c in cfg_c])
        b = fit_scale(Phi_c[inA & ok], np.log(s_c[inA & ok] ** 2), cfg_c[inA & ok], rng)[0]
        betas.append((b, float((Phi_c @ b).min()), float((Phi_c @ b).max())))     # clipped as in learned_scale.py
    fac = lambda Phi: np.mean([np.exp(np.clip(Phi @ b, lo, hi) / 2) for b, lo, hi in betas], axis=0)
    # test-atom reference (all atoms of the test configurations)
    import ase.io
    test = ase.io.read(D0 / "test.xyz", ":")
    ft = np.concatenate([fac(features(ref, *[np.asarray(a) for a in calc.support_descriptors(at)])) for at in test])
    f_ref = float(ft.mean())
    res = {"model": m, "f_ref_test_mean": f_ref, "qoi": {}}
    for k, ats in structures(m).items():
        fq = np.concatenate([fac(features(ref, *[np.asarray(a) for a in calc.support_descriptors(at)])) for at in ats])
        res["qoi"][k] = {"c_mean": float(fq.mean() / f_ref), "c_max": float(fq.max() / f_ref), "atoms": int(len(fq))}
    # coverage recount, CR3 x lam_F (headline sigma), without and with the discrepancy term
    rows = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "qoi_uq.csv"))
            if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
    drows = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "discrepancy.csv"))
             if r["model"] == m and r["method"] == "CR3-headline+disc" and r["scale"] == "lam_F"}
    for tag, src in (("", rows), ("+disc", drows)):
        for kind in ("c_mean", "c_max"):
            k95 = k68 = n = 0
            for k in COVER_KEYS:
                r = src.get(k)
                if r is None or k not in res["qoi"]:
                    continue
                err = float(r["mean"]) - float(r["dft"])
                if tag:                                    # method sigma rescaled, discrepancy part unchanged
                    sm, sd_ = float(r["sigma_method"]), float(r["sigma_disc"])
                    s = float(np.hypot(sm * res["qoi"][k][kind], sd_))
                else:
                    s = float(r["sigma"]) * res["qoi"][k][kind]
                z = abs(err) / s
                n += 1; k95 += z <= 1.959964; k68 += z <= 1.0
            base = [abs(float(src[k]["z"])) for k in COVER_KEYS if k in src and k in res["qoi"]]
            res[f"coverage{tag}/{kind}"] = {"n": n, "k95": int(k95), "k68": int(k68),
                                             "k95_unscaled": int(sum(z <= 1.959964 for z in base)),
                                             "k68_unscaled": int(sum(z <= 1.0 for z in base))}
    json.dump(res, open(RESULTS / f"qoi_scale_{m}.json", "w"), indent=1)
    for k, v in res["qoi"].items():
        print(f"{k:34s} c_mean {v['c_mean']:.3f}  c_max {v['c_max']:.3f}  ({v['atoms']} atoms)")
    for k, v in res.items():
        if k.startswith("coverage"):
            print(k, v)


if __name__ == "__main__":
    main()
