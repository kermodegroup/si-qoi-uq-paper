"""Observable-space discrepancy term for the defect formation-energy QoIs (decision 2026-11-01, RECON).

A bias shared by every hot defect snapshot is invisible to the CR3 jackknife, and a rank-one parameter correction
built from the snapshots does not carry it to the relaxed-defect QoIs (analysis/bias_term.py).  Its size is measured
from DFT training labels alone: r*_t, the zero-thermal-amplitude intercept of the paired defect-bulk residual, with
standard error se_t.  For the QoIs of defect class t (interstitial: dumbbell, hexagonal, tetrahedral; vacancy):

    sigma_Q^2 = sigma_Q,method^2 + r*_t^2 + se_t^2 ,

added to the headline sigma of a method (results/qoi_uq.csv); every other QoI is unchanged.  This leaves the pure
parameter-space framework for these QoIs and is labelled as such.  No QoI value or tolerance enters r*.

Estimators of r*: "paired" (analysis/bias_term.py: each defect snapshot against its nearest bulk snapshot, the form that
is also a linear synthetic row) and "regression" (analysis/defect_trend.py: the bulk reference is a per-atom thermal
trend fitted over all bulk cells near the defect volume).  "headline" (the default) takes regression for the interstitial
class (data-only evidence of analysis/estimator_check.py: the paired form amplifies a 2-atom partner's residual ~108x;
smaller SE; better cold hold-out extrapolation) and, for the vacancy, the 215-atom (the QoI's cell size) zero-temperature
intercept averaged over the size models of analysis/vacancy_evidence.py (decision 2026-11-02: the evidence disfavours a
shared intercept across the 63- and 215-atom cells).  "vaciv" (the earlier headline: inverse-variance combination of the
per-size intercepts, which assumes a shared intercept), "paired" and "regression" (pooled sizes) are sensitivities.

    python -m bench.si_qoi_uq.analysis.discrepancy [--estimator paired|regression|vaciv] o3d12 [o4d20]
        -> results/discrepancy[_paired|_regression].csv, .json
"""
import csv, json, sys

import numpy as np

from ..propagate import COVER_KEYS, clopper
from ..run import RESULTS

CLASS = {"vacancy_energy": "vacancy", "dumbbell_interstitial_energy": "interstitial",
         "hexagonal_interstitial_energy": "interstitial", "tetrahedral_interstitial_energy": "interstitial"}
METHODS = (("CR3-headline", "lam_F"), ("CR3-headline", "lam_V"), ("CR3-headline", "lam_pool"), ("B1-headline", "q_b1"))
Z68, Z95 = 1.0, 1.959963984540054


def main():
    args = sys.argv[1:]
    est = "headline"
    if "--estimator" in args:
        i = args.index("--estimator"); est = args[i + 1]; del args[i:i + 2]
    models = args or ["o3d12"]
    suffix = "" if est == "headline" else f"_{est}"
    rows = list(csv.DictReader(open(RESULTS / "qoi_uq.csv")))
    out, summ = [], {}
    for m in models:
        paired = json.load(open(RESULTS / f"bias_term_{m}.json"))["synthetic_rows"]
        tr_ = json.load(open(RESULTS / "defect_trend.json"))[m]
        iv = json.load(open(RESULTS / "vacancy_size_split.json"))[m]["inverse_variance"]
        vac_ps = {"r_star_eV": iv[0], "se_eV": iv[1], "cr3_sigma_lamF_eV": None}
        ve = json.load(open(RESULTS / "vacancy_evidence.json"))[m]["s_a=1"]["bma"]
        vac_bma = {"r_star_eV": ve["r215_mean_eV"], "se_eV": ve["r215_sd_eV"], "cr3_sigma_lamF_eV": None}
        regr = {t: {"r_star_eV": v["defect_intercept_eV"], "se_eV": v["defect_intercept_se"], "cr3_sigma_lamF_eV": None}
                for t, v in tr_.items()}
        syn = {"paired": paired, "regression": regr,
               "headline": {"interstitial": regr["interstitial"], "vacancy": vac_bma},
               "vaciv": {"interstitial": regr["interstitial"], "vacancy": vac_ps}}[est]
        summ[m] = {"r_star": {t: {"r_star_eV": v["r_star_eV"], "se_eV": v["se_eV"],
                                  "cr3_sigma_lamF_eV": v["cr3_sigma_lamF_eV"]} for t, v in syn.items()}}
        for meth, sc in METHODS:
            sel = {r["qoi"]: r for r in rows if r["model"] == m and r["method"] == meth and r["scale"] == sc}
            k68 = k95 = n = 0
            for k in COVER_KEYS:
                r = sel.get(k)
                if r is None or r.get("z") in ("", None):
                    continue
                mean, dft, sig = float(r["mean"]), float(r["dft"]), float(r["sigma"])
                t = CLASS.get(k)
                s_disc = float(np.hypot(syn[t]["r_star_eV"], syn[t]["se_eV"])) if t else 0.0
                s = float(np.hypot(sig, s_disc))
                z = (mean - dft) / s
                c68, c95 = int(abs(z) <= Z68), int(abs(z) <= Z95)
                n += 1; k68 += c68; k95 += c95
                out.append(dict(model=m, estimator=est, method=f"{meth}+disc", scale=sc, qoi=k, mean=mean, dft=dft, sigma_method=sig,
                                sigma_disc=s_disc, sigma=s, z=z, z_without=float(r["z"]), cov68=c68, cov95=c95,
                                tol=float(r["tol"])))
            summ[m][f"{meth}+disc/{sc}"] = {"n": n, "k68": k68, "ci68": clopper(k68, n), "k95": k95, "ci95": clopper(k95, n),
                                            "median_sigma_over_tol": float(np.median([o["sigma"] / o["tol"] for o in out
                                                                                      if o["model"] == m and o["method"] == f"{meth}+disc" and o["scale"] == sc]))}
    with open(RESULTS / f"discrepancy{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summ["estimator"] = est
    json.dump(summ, open(RESULTS / f"discrepancy{suffix}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for o in out:
        if o["qoi"] in CLASS and o["scale"] in ("lam_F", "q_b1"):
            print(f"{o['model']} {o['method']:20s} {o['qoi'][:30]:30s} sigma {o['sigma_method']:.3f} -> {o['sigma']:.3f} eV, "
                  f"z {o['z_without']:+.2f} -> {o['z']:+.2f}")


if __name__ == "__main__":
    main()
