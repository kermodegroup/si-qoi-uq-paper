"""Why is the vacancy formation energy missed, with no thermal trend to explain it?  DFT training labels and model QoIs only.

1. Cell size: the regression-estimator intercept (analysis/defect_trend.py form) for the 63- and 215-atom training
   vacancy cells separately (train + test cells, residuals from analysis/defect_resid.py).
2. Over-relaxation: the coldest training vacancy cells are near a DFT minimum (small DFT forces).  Each model relaxes
   them (positions, cell fixed); a model energy gain far beyond the DFT bound 1/2 sum |F_DFT| |dx| means the model's
   surface has a deeper vacancy minimum than DFT's, which lowers the relaxed formation energy.
3. Geometry: the four atoms around the vacancy in each model's own relaxed harness structure (mean-model run): their six
   mutual distances (Td: all equal; D2d: two short pairs).

    python -m bench.si_qoi_uq.analysis.vacancy_check      (SIQ_RUNS: the BLR run root)
"""
import json, pathlib

import numpy as np

H = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/defect_check")
D = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
N_COLD = 6


def ols(x, y):
    X = np.stack([np.ones_like(x), x], 1)
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = np.sum((y - X @ k) ** 2) / max(len(y) - 2, 1)
    return k, float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]))


def vacancy_neighbours(at, a0):
    """The four atoms nearest the vacancy site of a relaxed 215-atom cell: the site is the lattice point of the 216-atom
    reference with no atom near it."""
    from ase.build import bulk
    ref = bulk("Si", "diamond", a=a0, cubic=True) * (3, 3, 3)
    d = at.get_all_distances
    pos = at.positions
    cell = at.cell.array
    best, site = 1e9, None
    for p in ref.positions:
        dv = pos - p
        dv -= np.round(dv @ np.linalg.inv(cell)) @ cell
        mn = np.min(np.linalg.norm(dv, axis=1))
        if mn > 1.0 and (site is None or mn > best):
            best, site = mn, p
    dv = pos - site
    dv -= np.round(dv @ np.linalg.inv(cell)) @ cell
    nb = np.argsort(np.linalg.norm(dv, axis=1))[:4]
    del d
    pd = sorted(at.get_distance(int(i), int(j), mic=True) for k, i in enumerate(nb) for j in nb[k + 1:])
    return pd, sorted(np.linalg.norm(dv[nb], axis=1).tolist())


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import ase
    from ase.optimize import LBFGS
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.data import load_configs
    from ..harness import ROOT, RUNS
    from ..run import RESULTS
    TYPES = ("interstitial", "vacancy", "divacancy", "dia")
    allc = []
    for split in ("train", "test"):
        allc += [(split, c) for c in load_configs(f"{D}/{split}.xyz", virial_key="virial") if c.config_type in TYPES and c.energy is not None]
    n = np.array([len(c.numbers) for _, c in allc]); x = np.array([float(np.mean(c.forces ** 2)) for _, c in allc])
    v = np.array([abs(np.linalg.det(c.cell)) / len(c.numbers) for _, c in allc]); t = np.array([c.config_type for _, c in allc], dtype=object)
    out = {"size_split": {}, "relax_cold": {}, "geometry": {}}
    for m in ("o3d12", "o4d20"):
        r = np.array([o["resid"] for o in json.load(open(H / f"{m}.json"))])
        assert len(r) == len(allc)
        out["size_split"][m] = {}
        for size in (63, 215):
            mask = (t == "vacancy") & (n == size)
            v0 = v[mask].mean()
            bulk = (t == "dia") & (np.abs(v - v0) < 0.6)
            kb, _ = ols(x[bulk], r[bulk] / n[bulk])
            y = r[mask] - n[mask] * (kb[0] + kb[1] * x[mask])
            k, se = ols(x[mask], y)
            out["size_split"][m][size] = {"n_cells": int(mask.sum()), "intercept_eV": float(k[0]), "se_eV": se,
                                          "slope": float(k[1]), "mean_eV": float(y.mean()), "x_min": float(x[mask].min())}
    # 2. relax the coldest training vacancy cells with each model
    cold = {}
    for size in (63, 215):
        idx = [i for i in np.argsort(x) if t[i] == "vacancy" and n[i] == size and allc[i][0] == "train"][:N_COLD]
        cold[size] = idx
    for m in ("o3d12", "o4d20"):
        calc = ACECalculator(str(RUNS / "fits" / m / "cr3" / "model.npz"), skin=0.0)
        out["relax_cold"][m] = []
        for size, idx in cold.items():
            for i in idx:
                c = allc[i][1]
                at = ase.Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
                at.calc = calc
                e0 = at.get_potential_energy()
                f_model0 = np.abs(at.get_forces()).max()
                p0 = at.positions.copy()
                LBFGS(at, logfile=None).run(fmax=1e-3, steps=2000)
                dx = at.positions - p0
                bound = 0.5 * float(np.sum(np.abs(c.forces) * np.abs(dx)))
                out["relax_cold"][m].append(dict(size=size, x=float(x[i]), dft_fmax=float(np.abs(c.forces).max()),
                                                 model_fmax0=float(f_model0), model_gain_eV=float(e0 - at.get_potential_energy()),
                                                 dft_gain_bound_eV=bound, max_disp_A=float(np.linalg.norm(dx, axis=1).max())))
        g = [o["model_gain_eV"] for o in out["relax_cold"][m] if o["size"] == 215]
        print(f"{m}: 215-atom cold vacancy cells, model relaxation gain {np.round(g, 3)} eV "
              f"(DFT bound {np.round([o['dft_gain_bound_eV'] for o in out['relax_cold'][m] if o['size'] == 215], 3)})", flush=True)
    # 3. geometry of each model's relaxed harness vacancy
    for m in ("o3d12", "o4d20"):
        fr = __import__("ase.io", fromlist=["read"]).read(ROOT / "test-runner" / f"model-SIQ_{m}_cr3-test-vacancy-energy" / "siq_relaxed.xyz", ":")
        a0 = fr[0].cell.lengths()[0]
        pd, rad = vacancy_neighbours(fr[1], a0)
        out["geometry"][m] = {"a0": float(a0), "neighbour_pair_distances_A": pd, "distances_from_site_A": rad}
        print(f"{m}: vacancy neighbour pair distances {np.round(pd, 3)} A; from site {np.round(rad, 3)} A")
    for m, d in out["size_split"].items():
        print(m, {s: f"{v_['intercept_eV']:+.3f} ± {v_['se_eV']:.3f} (n {v_['n_cells']}, x_min {v_['x_min']:.3f})" for s, v_ in d.items()})
    json.dump(out, open(RESULTS / "vacancy_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
