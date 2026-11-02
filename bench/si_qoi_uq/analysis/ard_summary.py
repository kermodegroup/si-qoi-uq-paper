"""ARD (QR solver, ace-jax PR 137) against the benchmark BLR fits, for the paper's ARD section -> results/ard_summary.json.

Per model: the ARD hyperparameters and endpoint diagnostics (ard.json of the cr3 fit), test-split RMSE of both fits from
the same code (the residuals the scale diagnostics store, so the normalisations match), the test-split scales, and the
propagated coverage of the acceptance run (runs/si_qoi_uq_ard137/report) beside the BLR-based headline (results/).

    python -m bench.si_qoi_uq.analysis.ard_summary
"""
import csv, json, pathlib

import numpy as np

RUNS = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs")
ARD, BLR = RUNS / "si_qoi_uq_ard137", RUNS / "si_qoi_uq_blr"


def rmse(diag):
    d = np.load(diag, allow_pickle=True)
    k, r, n = d["kind"], d["resid"], d["n_atoms"]
    q = lambda x: float(np.sqrt(np.mean(x ** 2)))
    return {"E_meV_atom": 1e3 * q(r[k == "E"] / n[k == "E"]), "F_eV_A": q(r[k == "F"]), "V_meV_atom": 1e3 * q(r[k == "V"] / n[k == "V"])}


def coverage(path, m):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == m and (r["method"], r["scale"]) in (("CR3-headline", "lam_F"), ("CR3-headline", "lam_V"), ("B1-headline", "q_b1")):
            out[f"{r['method']}/{r['scale']}"] = {k: float(r[k]) for k in ("n", "k68", "k95", "median_sigma_over_tol", "rms_z")}
    return out


def main():
    from ..run import RESULTS
    res = {}
    for m in ("o3d12", "o4d20"):
        a = json.load(open(ARD / "fits" / m / "cr3" / "ard.json"))
        opt = a["optimiser"]
        res[m] = {"h_names": a["h_names"], "h": a["h"], "a_floor": a["a_floor"], "solver": a.get("solver"), "cond_max": a.get("cond_max"),
                  "degenerate": a.get("ard_degenerate"), "at_bound": opt.get("at_bound"), "cond_S": opt.get("cond_S"),
                  "a_blr": float(-2 * json.load(open(RUNS / "accuracy_cost" / "ace" / m / "theta_map.json"))["log_sigma_c"]),
                  "test_rmse": {"ard": rmse(ARD / "diag" / f"{m}.npz"), "blr": rmse(BLR / "diag" / f"{m}.npz")},
                  "lam_cr3": {"ard": json.load(open(ARD / "diag" / f"{m}_scales.json"))["lam_cr3"],
                              "blr": json.load(open(BLR / "diag" / f"{m}_scales.json"))["lam_cr3"]},
                  "coverage": {"ard": coverage(ARD / "report" / "results" / "coverage.csv", m),
                               "blr": coverage(RESULTS / "coverage.csv", m)}}
    json.dump(res, open(RESULTS / "ard_summary.json", "w"), indent=1, default=float)
    for m, d in res.items():
        print(m, [round(x, 2) for x in d["h"]], "floor", round(d["a_floor"], 2), "deg", d["degenerate"], d["test_rmse"],
              {k: v["CR3-headline/lam_F"]["k95"] for k, v in d["coverage"].items()})


if __name__ == "__main__":
    main()
