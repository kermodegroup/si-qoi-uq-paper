"""Joint evidence of the DFT QoI values under each method's predictive distribution (replaces the chi^2 joint test).

For each model, method and covariance source (delta method or committee; results/joint_tests.json, which stores the
error vector d = (DFT - prediction) / tol and the covariance S / (tol tol^T) in tolerance units), the log predictive
density log p(d | method) = log N(d; 0, S + eps I).  The scale of every method is fixed beforehand (held-out forces, or
held-out scores for B1), so the methods have no free parameters left: the difference of two methods' log p on the same
QoIs is a log Bayes factor.  Tolerance units make the comparison independent of the QoIs' own units, and the ridge eps
(a floor on sigma of 1e-4..1e-2 tolerances) is the same for every method.

"+disc" adds the headline discrepancy term of the defect QoIs (results/discrepancy.json: r*^2 + se^2 per defect class,
from the training labels only) to the diagonal, as analysis/discrepancy.py does for the marginal intervals.

    python -m bench.si_qoi_uq.analysis.evidence       -> results/evidence.json, results/evidence.csv
"""
import csv, json

import numpy as np

from ..propagate import EVIDENCE_EPS, TOL, log_predictive
from ..run import RESULTS
from .discrepancy import CLASS

EPS = 1e-6                                     # the headline ridge; EVIDENCE_EPS are the sensitivities
METHODS = ("CR3/lam_F", "CR3/lam_V", "CR3/lam_E", "CR3/lam_pool", "B1/q_b1", "B2lev5/lam_1", "B2lev50/lam_1", "B3/lam_1")
REF = "CR3/lam_F"


def main():
    joint = json.load(open(RESULTS / "joint_tests.json"))
    disc = json.load(open(RESULTS / "discrepancy.json")) if (RESULTS / "discrepancy.json").exists() else {}
    out, rows = {}, []
    for m, per in joint.items():
        rs = disc.get(m, {}).get("r_star", {})
        out[m] = {}
        for name in METHODS:
            for kind, jt in per.get(name, {}).items():
                if "S_tol" not in jt:
                    continue
                d, S, q = np.array(jt["d_tol"]), np.array(jt["S_tol"]), jt["qois"]
                variants = {"": S}
                if rs:
                    add = np.zeros(len(q))
                    for i, k in enumerate(q):
                        t = CLASS.get(k)
                        if t and t in rs:
                            tol, kind_ = TOL[k]                            # point defects: absolute, in eV
                            assert kind_ != "rel", k
                            add[i] = (rs[t]["r_star_eV"] ** 2 + rs[t]["se_eV"] ** 2) / tol ** 2
                    variants["+disc"] = S + np.diag(add)
                for tag, SS in variants.items():
                    lp = log_predictive(d, SS)
                    out[m][f"{name}{tag}/{kind}"] = {"qois": q, **lp}
                    rows.append(dict(model=m, method=name + tag, kind=kind, n=len(q),
                                     **{f"log_p_eps{e:g}": lp[f"eps_{e:g}"]["log_p"] for e in EVIDENCE_EPS},
                                     mahalanobis=lp[f"eps_{EPS:g}"]["mahalanobis"], logdet=lp[f"eps_{EPS:g}"]["logdet"]))
        # log Bayes factors against the headline on the same QoIs and covariance source
        for r in rows:
            if r["model"] != m:
                continue
            ref = next((x for x in rows if x["model"] == m and x["method"] == REF and x["kind"] == r["kind"]), None)
            if ref and ref["n"] == r["n"]:
                r["log_bf_vs_headline"] = r[f"log_p_eps{EPS:g}"] - ref[f"log_p_eps{EPS:g}"]
    json.dump(out, open(RESULTS / "evidence.json", "w"), indent=1)
    with open(RESULTS / "evidence.csv", "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']} {r['method']:22s} {r['kind']:9s} n={r['n']:2d} log p {r[f'log_p_eps{EPS:g}']:9.2f}  "
              f"d2 {r['mahalanobis']:8.1f}  logdet {r['logdet']:8.1f}  logBF {r.get('log_bf_vs_headline', float('nan')):+8.2f}")



if __name__ == "__main__":
    main()
