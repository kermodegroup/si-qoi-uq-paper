"""Calibrated size against evaluation cost along the linear ACE ladder (plan section 5 C, ladder).

Inputs: results/crc_ladder.json (analysis/crc_config.py ladder; size = log geometric semi-axis per force component,
eV/A, mean over the 100 random test halves at the CRC threshold) and results/ace_pareto.json (A100 float64 cost of the
ace-jax CUDA kernel, µs per atom-step; test force RMSE). For each loss and alpha: every (model, method) row, the
non-dominated set in (cost, size), the gap size - log RMSE_F (the price of calibration over the point error) and the
worst per-kind risk (the failure mode the marginal guarantee hides).

    python -m bench.si_qoi_uq.analysis.crc_pareto     -> results/crc_pareto.json
"""
import json, math

from ..run import RESULTS


def front(rows):
    """Rows not dominated in (cost, size): no other row is at most as costly and strictly smaller, or cheaper and no larger."""
    keep = []
    for r in rows:
        if not any((o["cost_gpu"] <= r["cost_gpu"] and o["size"] < r["size"]) or (o["cost_gpu"] < r["cost_gpu"] and o["size"] <= r["size"])
                   for o in rows):
            keep.append(r["name"])
    return keep


def main():
    lad = json.load(open(RESULTS / "crc_ladder.json"))
    acc = {p["size"]: p for p in json.load(open(RESULTS / "ace_pareto.json"))}
    out = {"cost": "A100 float64 ace-jax CUDA kernel, µs/atom-step (results/ace_pareto.json)", "loss": {}}
    for loss, by_a in lad["loss"].items():
        out["loss"][loss] = {}
        for a, res in by_a.items():
            rows = []
            for name, h in res["halves"].items():
                m, meth = name.split("/")
                p = acc[m]
                rk = h["risk_by_kind"]
                worst = max(rk, key=rk.get)
                rows.append({"name": name, "model": m, "method": meth, "n_params": p["n_params"], "cost_gpu": p["cost_gpu"],
                             "rmse_F": p["rmse_F"], "size": h["size"]["mean"], "size_p05": h["size"]["p05"],
                             "size_p95": h["size"]["p95"], "gap": h["size"]["mean"] - math.log(p["rmse_F"]),
                             "risk": h["risk"]["mean"], "worst_kind": worst, "worst_risk": rk[worst], "first": h["first"]})
            rows.sort(key=lambda r: (r["cost_gpu"], r["size"]))
            fr = front(rows)
            out["loss"][loss][a] = {"rows": rows, "front": fr, "selection": res.get("selection")}
            print(f"\n{loss} loss, alpha {a}: front {fr}")
            print(f"  {'model/method':18s} {'cost':>7s} {'size':>7s} {'gap':>6s} {'risk':>5s}  worst kind")
            for r in rows:
                star = "*" if r["name"] in fr else " "
                print(f" {star}{r['name']:18s} {r['cost_gpu']:7.3f} {r['size']:7.3f} {r['gap']:6.2f} {r['risk']:5.3f}  "
                      f"{r['worst_kind']} {r['worst_risk']:.2f}")
    json.dump(out, open(RESULTS / "crc_pareto.json", "w"), indent=1)


if __name__ == "__main__":
    main()
