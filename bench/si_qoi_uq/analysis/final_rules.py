"""The step 3 decision rules of plan section 5 D ("decision rules, fixed before any result", 2026-11-13), applied as written.

Rule 1 (held out): results/final_ood.json, alpha 0.1 joint loss, views F and E, per candidate (hand, Ex10): the paired per-split
difference of held-out risk d = risk(B1) - risk(CR3) over the 100 splits, mean and tau = sd / sqrt 2. CR3 preferred off
distribution if mean d > tau in both views at x 10; reversed if mean d < -tau in either. Reported beside it (not deciding):
dE_type, dE_form, the F view's frac loss, sizes (paired size difference, same form) and per held-out type risk.
Rule 2 (informational): the same with the ellipsoid in place of B1, where final_shapes.py's ellipsoid ran.
Rule 3 (DFT QoIs): weights_qoi/results/qoi_joint_score.json and coverage.csv, o4d16_Ex10 (o4d16 the reference): CR3 (lam_F)
against B1 (q_b1), delta covariances with the shared discrepancy term (" +disc"), tau 0.05. CR3 preferred if k95(CR3) >= k95(B1)
and d2/n(CR3) in the chi^2_17 / 17 band while B1's is outside, or both inside with LS/n(CR3) <= LS/n(B1) - 0.05.
Rule 4: the outcome from rules 1 and 3.

    python -m bench.si_qoi_uq.analysis.final_rules   -> results/final_rules.json
"""
import json, pathlib, sys

import numpy as np

from ..run import RESULTS

QOI = pathlib.Path(__file__).parent.parent / "weights_qoi" / "results"
LA = "joint 0.1"


def paired(a, b):
    """mean and tau of a - b over the splits (the selection rule's form)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"mean": float(np.mean(d)), "tau": float(np.std(d, ddof=1) / np.sqrt(2)), "n": int(len(d))}


def verdict(p):
    return "CR3" if p["mean"] > p["tau"] else ("reversed" if p["mean"] < -p["tau"] else "unresolved")


def held_out(res):
    out = {}
    for name in ("hand", "Ex5", "Ex10"):
        for other in ("B1", "ellipsoid"):
            o = name if other == "B1" else f"{name}/ellipsoid"
            c = f"{name}/CR3"
            row = {}
            for vn, v in res["views"].items():
                for la, r in v.items():
                    if c not in r or o not in r or "ood" not in r[c]:
                        continue
                    a, b = r[o]["ood"], r[c]["ood"]
                    row[f"{vn} {la}"] = {"risk": paired(a["split"]["risk"], b["split"]["risk"]),
                                         "size": paired(a["split"]["size"], b["split"]["size"]),
                                         f"risk_{other}": a["risk"]["mean"], "risk_CR3": b["risk"]["mean"],
                                         "test_risk": {other: r[o]["test"]["risk"]["mean"], "CR3": r[c]["test"]["risk"]["mean"]},
                                         "test_size": paired(r[o]["test"]["split"]["size"], r[c]["test"]["split"]["size"]),
                                         "by_type": {other: a["risk_by"], "CR3": b["risk_by"]}}
                    row[f"{vn} {la}"]["verdict"] = verdict(row[f"{vn} {la}"]["risk"])
            if row:
                out[f"{name}: {other} - CR3"] = row
    return out


def qoi(model):
    js = json.load(open(QOI / "qoi_joint_score.json"))
    from scipy.stats import chi2
    n = len(js["models"][model]["qois"])                    # the rule wrote 17 (the paper's set); the band follows the actual n
    band = [float(chi2.ppf(0.025, n) / n), float(chi2.ppf(0.975, n) / n)]
    res = js["models"][model]["results"]
    pick = {"CR3": "CR3 (delta) +disc", "B1": "B1 (delta) +disc"}
    s = {k: res[v]["tau0.05"] for k, v in pick.items()}
    inside = {k: band[0] <= v["d2_over_n"] <= band[1] for k, v in s.items()}
    cov_ok = s["CR3"]["k95"] >= s["B1"]["k95"]
    cr3 = cov_ok and ((inside["CR3"] and not inside["B1"]) or
                      (inside["CR3"] and inside["B1"] and s["CR3"]["ls_over_n"] <= s["B1"]["ls_over_n"] - 0.05))
    return {"band": band, "band_17": js["band_d2_over_n"], "n": n, "qois": js["models"][model]["qois"], "scores": s, "inside_band": inside, "coverage_ok": cov_ok,
            "verdict": "CR3" if cr3 else "B1 matches CR3",
            "without_disc": {k: res[v.replace(" +disc", "")]["tau0.05"] for k, v in pick.items()},
            "tau_sens": {k: {t: res[v][t]["d2_over_n"] for t in ("tau0.01", "tau0.2")} for k, v in pick.items()}}


def main(model=None):
    tag = f"_{model}" if model else ""                     # the paper's models (hand only): rule 1 form, no rule 3
    ood = json.load(open(RESULTS / f"final_ood{tag}.json"))
    out = {"model": model or "o4d16", "rule1_2": held_out(ood)}
    r1 = out["rule1_2"].get("hand: B1 - CR3" if model else "Ex10: B1 - CR3", {})
    v1 = [r1[k]["verdict"] for k in (f"F {LA}", f"E {LA}") if k in r1]
    out["rule1"] = ("CR3" if v1 and all(v == "CR3" for v in v1) else
                    "reversed" if "reversed" in v1 else "unresolved") if len(v1) == 2 else None
    if not model and (QOI / "qoi_joint_score.json").exists():
        out["rule3"] = {m: qoi(m) for m in ("o4d16_Ex10", "o4d16")}
    r3 = out.get("rule3", {}).get("o4d16_Ex10", {}).get("verdict")
    n = (out["rule1"] == "CR3") + (r3 == "CR3")
    out["rule4"] = None if out["rule1"] is None or r3 is None else \
        ["neither: B1 as reliable and sharper after CRC", "one: headline narrowed", "both: headline stands"][n]
    for k, row in out["rule1_2"].items():
        print(f"\n{k} (held-out risk difference, mean (tau); size difference; verdict)")
        for v, x in row.items():
            print(f"  {v:14s} risk {x['risk']['mean']:+.3f} ({x['risk']['tau']:.3f})  size {x['size']['mean']:+.3f} ({x['size']['tau']:.3f})"
                  f"  test size {x['test_size']['mean']:+.3f}  -> {x['verdict']}")
    for m, r in out.get("rule3", {}).items():
        print(f"\n{m} QoIs: " + "  ".join(f"{k}: k95 {s['k95']}/{r['n']} d2/n {s['d2_over_n']:.2f} LS/n {s['ls_over_n']:.2f} sharpJ {s['sharp_j']:.3f}"
                                          for k, s in r["scores"].items()) + f"  -> {r['verdict']}")
    print(f"\nrule 1 ({'hand' if model else 'Ex10'}): {out['rule1']}; rule 3 (Ex10): {r3}; rule 4: {out['rule4']}")
    json.dump(out, open(RESULTS / f"final_rules{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:2])
