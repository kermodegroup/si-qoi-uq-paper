"""Observable-space discrepancy term for the defect formation-energy QoIs (decision 2026-11-01, RECON).

A bias shared by every hot defect snapshot is invisible to the CR3 jackknife, and a rank-one parameter correction
built from the snapshots does not carry it to the relaxed-defect QoIs (analysis/bias_term.py).  Its size is measured
from DFT training labels alone: r*_t, the zero-thermal-amplitude intercept of the paired defect-bulk residual, with
standard error se_t.  For the QoIs of defect class t (interstitial: dumbbell, hexagonal, tetrahedral; vacancy):

    sigma_Q^2 = sigma_Q,method^2 + r*_t^2 + se_t^2 ,

added to the headline sigma of a method (results/qoi_uq.csv); every other QoI is unchanged.  This leaves the pure
parameter-space framework for these QoIs and is labelled as such.  No QoI value or tolerance enters r*.

    python -m bench.si_qoi_uq.analysis.discrepancy o3d12 [o4d20]  -> results/discrepancy.csv, results/discrepancy.json
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
    models = sys.argv[1:] or ["o3d12"]
    rows = list(csv.DictReader(open(RESULTS / "qoi_uq.csv")))
    out, summ = [], {}
    for m in models:
        syn = json.load(open(RESULTS / f"bias_term_{m}.json"))["synthetic_rows"]
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
                out.append(dict(model=m, method=f"{meth}+disc", scale=sc, qoi=k, mean=mean, dft=dft, sigma_method=sig,
                                sigma_disc=s_disc, sigma=s, z=z, z_without=float(r["z"]), cov68=c68, cov95=c95,
                                tol=float(r["tol"])))
            summ[m][f"{meth}+disc/{sc}"] = {"n": n, "k68": k68, "ci68": clopper(k68, n), "k95": k95, "ci95": clopper(k95, n),
                                            "median_sigma_over_tol": float(np.median([o["sigma"] / o["tol"] for o in out
                                                                                      if o["model"] == m and o["method"] == f"{meth}+disc" and o["scale"] == sc]))}
    with open(RESULTS / "discrepancy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    json.dump(summ, open(RESULTS / "discrepancy.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for o in out:
        if o["qoi"] in CLASS and o["scale"] in ("lam_F", "q_b1"):
            print(f"{o['model']} {o['method']:20s} {o['qoi'][:30]:30s} sigma {o['sigma_method']:.3f} -> {o['sigma']:.3f} eV, "
                  f"z {o['z_without']:+.2f} -> {o['z']:+.2f}")


if __name__ == "__main__":
    main()
