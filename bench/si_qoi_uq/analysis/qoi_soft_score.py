"""Soft QoI scores for an uncertainty method: geometric means over the QoIs, as the ace-jax accuracy benchmark scores a
model (bench/si_qoi/score.py: |error| / tolerance, each clipped to [GM_FLOOR, GM_CAP] = [0.05, 10], tolerances from
the same TOLERANCES), in place of the hard count of QoIs inside +-1.96 sigma, which jumps when one borderline QoI
crosses.

Per method, over the QoIs in a results file (error e, sigma s, tolerance t; z = e / s):
  gm_err    gm of |e| / t                       the benchmark's accuracy score (same for every method on one model)
  gm_sigma  gm of s / t                         sharpness: the typical width in tolerance units
  gm_absz   gm of |z| (clipped to [1e-3, 1e3])  calibration: for z ~ N(0, 1), exp(E log|z|) = exp(-(gamma + ln 2) / 2)
                                                = 0.530, with Var(log|z|) = pi^2 / 8, so for n QoIs a calibrated method
                                                gives 0.530 exp(+-1.96 sqrt(pi^2 / 8 / n)) (17 QoIs: 0.31-0.90); above
                                                the band is overconfident, below it too wide
  gm_crps   gm of CRPS / t                      a proper score (accuracy, sharpness and calibration together); for a
                                                Gaussian, CRPS = s [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)], which
                                                tends to |e| as s -> 0, so it is on the benchmark's scale
  k95       QoIs with |z| <= 1.96               the hard count, for reference
The QoIs are taken as independent in the band for gm_absz (they are not: shared parameters correlate them, §4).

    python -m bench.si_qoi_uq.analysis.qoi_soft_score [results/pops_ellipse_<...>.json ...]
        -> results/qoi_soft_score.json
"""
import json, sys

import numpy as np
from scipy.stats import norm

from ..run import RESULTS

GM_FLOOR, GM_CAP = 0.05, 10.0          # as bench/si_qoi/score.py
Z_REF = float(np.exp(-(np.euler_gamma + np.log(2.0)) / 2.0))
DEFAULT = ["pops_ellipse_o3d12_sp_lev50_it20000", "pops_ellipse_o3d12_sp_lev50_cf100", "pops_ellipse_o3d12_sp_lev50_cf100_fp2000",
           "pops_ellipse_o4d20_sp_lev50", "pops_ellipse_o4d20_sp_lev50_it5000",
           "pops_ellipse_o4d20_sp_lev50_cf100_fp1000"]


def gm(v, lo=GM_FLOOR, hi=GM_CAP):
    return float(np.exp(np.mean(np.log(np.clip(v, lo, hi)))))


def crps(z, s):
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def scores(qoi):
    """{method: soft scores} from a results file's "qoi" block ({QoI: {method: {sigma, sigma_over_tol, z}}})."""
    meths = list(next(iter(qoi.values())).keys())
    out = {}
    for m in meths:
        st = np.array([v[m]["sigma_over_tol"] for v in qoi.values()])
        z = np.array([v[m]["z"] for v in qoi.values()])
        n = len(z)
        half = 1.96 * np.sqrt(np.pi ** 2 / 8 / n)
        out[m] = {"n": n, "gm_err": gm(np.abs(z) * st), "gm_sigma": gm(st), "gm_absz": gm(np.abs(z), 1e-3, 1e3),
                  "gm_absz_band": [Z_REF * np.exp(-half), Z_REF * np.exp(half)], "gm_crps": gm(crps(z, st)),
                  "k95": int(np.sum(np.abs(z) <= 1.96))}
    return out


def main():
    names = [a.removesuffix(".json").rsplit("/", 1)[-1] for a in sys.argv[1:]] or DEFAULT
    res = {}
    for nm in names:
        d = json.load(open(RESULTS / f"{nm}.json"))
        res[nm] = scores(d["qoi"])
    band = next(iter(next(iter(res.values())).values()))["gm_absz_band"]
    print(f"gm|z| reference {Z_REF:.3f}, calibrated 95 % band for 17 QoIs {band[0]:.2f}-{band[1]:.2f}; "
          f"scores are geometric means of x / tolerance clipped to [{GM_FLOOR}, {GM_CAP}] (ace-jax score.py)")
    print(f"{'results':42s} {'method':20s} {'k95':>4s} {'gm err':>7s} {'gm sig':>7s} {'gm|z|':>6s} {'gm CRPS':>8s}")
    for nm, r in res.items():
        for m, v in r.items():
            flag = "" if band[0] <= v["gm_absz"] <= band[1] else (" over" if v["gm_absz"] > band[1] else " wide")
            print(f"{nm.removeprefix('pops_ellipse_'):42s} {m:20s} {v['k95']:2d}/{v['n']} {v['gm_err']:7.3f} {v['gm_sigma']:7.3f} "
                  f"{v['gm_absz']:6.2f} {v['gm_crps']:8.3f}{flag}")
    json.dump({"z_ref": Z_REF, "gm_floor": GM_FLOOR, "gm_cap": GM_CAP, "results": res},
              open(RESULTS / "qoi_soft_score.json", "w"), indent=1)


if __name__ == "__main__":
    main()
