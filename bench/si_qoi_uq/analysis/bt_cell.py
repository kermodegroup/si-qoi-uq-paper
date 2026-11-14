"""o3d12: how much of the CR3 QoI width comes from one training cell?  (next-steps plan §3, barrier_influence.py)

One 128-atom beta-tin cell (training configuration K_CELL) carries 87-89 % of the o3d12 CR3 variance of the vacancy
formation energy and the barrier, and most of the tetrahedral and hexagonal interstitials'.  Here its column is
removed from the per-configuration factor (RUNS/delta/o3d12_cr3_columns.npy) and the rest re-centred (the centred
columns sum to zero, so each gains l_k / (K - 1)); lam_F is recomputed on the test split with the downdated factor
(diagnostics.test_scores / scales, the paper's definition), and every delta QoI's sigma = lam_F |L^T g| is compared
against results/qoi_uq.csv (z, 95 % coverage) with and without the cell.  Delta sigma only: QoIs whose headline
sigma came from the committee are marked.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.bt_cell [o3d12] [K_CELL] -> results/bt_cell_<m>.json
"""
import csv, json, sys

import numpy as np

Z95 = 1.959963984540054


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ..diagnostics import scales, test_scores
    from ..harness import RUNS
    from ..run import RESULTS, lin_for
    m = sys.argv[1] if len(sys.argv) > 1 else "o3d12"
    kc = int(sys.argv[2]) if len(sys.argv) > 2 else 996
    lin = lin_for(m, "cr3")
    Lk = np.load(RUNS / "delta" / f"{m}_cr3_columns.npy")
    K = Lk.shape[1]
    L_drop = np.delete(Lk, kc, axis=1) + Lk[:, [kc]] / (K - 1)
    fac = {"full": Lk, "drop": L_drop}
    res = test_scores(lin, fac, log=lambda *_: None)
    lam = {k: scales(res, k)[1]["F"] for k in fac}
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    rows = dict(zip((str(k) for k in dz["keys"]), np.asarray(dz["G"], np.float64)))
    rows["vacancy_barrier"] = np.load(RUNS / "delta" / f"{m}_cr3_vacancy_barrier_row.npy")
    csvr = {r["qoi"]: r for r in csv.DictReader(open(RESULTS / "qoi_uq.csv"))
            if r["model"] == m and r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
    vb = json.load(open(RESULTS / "vacancy_barrier.json"))[m]["cr3"]
    out = {"model": m, "cell": kc, "lam_F": lam, "lam_F_stored": float(json.load(open(RUNS / "diag" / f"{m}_scales.json"))
                                                                    ["lam_cr3"]["F"]), "qoi": {}}
    print(f"{m} cell {kc}: lam_F full {lam['full']:.3f} (stored {out['lam_F_stored']:.3f}), without the cell {lam['drop']:.3f}")
    for q, g in rows.items():
        if q.startswith("energy:"):
            continue
        sig = {k: lam[k] * float(np.linalg.norm(fac[k].T @ g[:Lk.shape[0]])) for k in fac}
        if q == "vacancy_barrier":
            mean, dft, src = vb["Eb"], 0.299, "delta"
        elif q in csvr:
            r = csvr[q]
            mean, dft = float(r["mean"]), float(r["dft"]) if r["dft"] not in ("", "nan") else np.nan
            src = "delta" if abs(float(r["sigma"]) / sig["full"] - 1) < 0.15 else f"committee (csv sigma {float(r['sigma']):.3g})"
        else:
            continue
        e = dft - mean
        z = {k: e / s for k, s in sig.items()}
        out["qoi"][q] = {"sigma": sig, "z": z, "covered95": {k: bool(abs(zz) <= Z95) for k, zz in z.items()},
                         "sigma_ratio_drop_full": sig["drop"] / sig["full"], "csv_sigma_source": src}
        print(f"  {q:34s} sigma {sig['full']:.4g} -> {sig['drop']:.4g} (x{sig['drop'] / sig['full']:.2f})  z {z['full']:+.2f} -> "
              f"{z['drop']:+.2f}  {'' if src == 'delta' else src}", flush=True)
    cov = {k: sum(v["covered95"][k] for v in out["qoi"].values() if np.isfinite(v["z"][k])) for k in fac}
    n = sum(np.isfinite(v["z"]["full"]) for v in out["qoi"].values())
    out["covered95"] = {"n": int(n), **{k: int(c) for k, c in cov.items()}}
    print(f"  95 % covered (delta sigma, these QoIs): full {cov['full']}/{n}, without the cell {cov['drop']}/{n}")
    json.dump(out, open(RESULTS / f"bt_cell_{m}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
