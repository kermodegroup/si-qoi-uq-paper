"""Evidence on the test split for shared against per-observable scales of the CR3 covariance (replaces the fixed
[0.8, 1.25] band on lam_V / lam_F and lam_E / lam_F).

Units are test configurations, as in CR3 itself: the force components of one configuration are correlated, so they
are not independent observations.  For configuration c and observable o in {E, F, V}, m_co is the mean over the
block's components of z^2, z = residual / CR3 marginal sd (RUNS/diag/<m>.npz, symmetry-zero rows already dropped).
The model is m_co = lam_o^2 X, X ~ chi^2_nu / nu, i.e. m_co ~ Gamma(nu_o / 2, rate nu_o / (2 lam_o^2)): mean lam_o^2,
as the RMS-z definition of lam, with an effective dof nu_o per observable that absorbs the within-block correlation and
the tails (free in every model, log-uniform on [NU_LO, NU_HI], integrated out).

A joint Gaussian likelihood of each block under lam_o^2 times the full CR3 block covariance was tried first and fails
for forces: the joint force scale is set by the ridge on the near-singular block (o3d12: 17347 / 1749 / 185 for a
relative ridge 1e-8 / 1e-6 / 1e-4, against a marginal 5.8), i.e. by the CR3 off-diagonal shape, not by the scale.

Models: the five partitions of {E, F, V} into groups sharing one lam, from one scale for all ("shared") to one per
observable ("E|F|V"); each lam log-uniform on [LAM_LO, LAM_HI]; equal prior odds.  Grids in log lam and log nu.

    SIQ_RUNS=<BLR run root> python -m bench.si_qoi_uq.analysis.scale_evidence [o3d12 o4d20]  -> results/scale_evidence.json
"""
import json, sys

import numpy as np
from scipy.special import gammaln

LAM_LO, LAM_HI, NU_LO, NU_HI = 0.1, 1000.0, 0.05, 1000.0
LL = np.linspace(np.log(LAM_LO), np.log(LAM_HI), 1601)
LN = np.linspace(np.log(NU_LO), np.log(NU_HI), 401)
PARTS = {"shared": ("EFV",), "E|FV": ("E", "FV"), "F|EV": ("F", "EV"), "V|EF": ("V", "EF"), "E|F|V": ("E", "F", "V")}


def block_ms(m):
    """{o: array of m_co over configurations}."""
    from ..harness import RUNS
    d = np.load(RUNS / "diag" / f"{m}.npz", allow_pickle=True)
    z2 = (d["resid"] / d["sd_cr3"]) ** 2
    out = {}
    for o in "EFV":
        k = d["kind"] == o
        cfg = d["cfg"][k]
        u, inv = np.unique(cfg, return_inverse=True)
        out[o] = np.bincount(inv, z2[k]) / np.bincount(inv)
    return out


def log_like_lam(ms):
    """log p(m_o | lam) on the LL grid, nu integrated out (log-uniform prior)."""
    nu = np.exp(LN)[:, None]                                  # (nu, 1)
    lam2 = np.exp(2 * LL)[None, :]                            # (1, lam)
    n, sl, sm = len(ms), np.sum(np.log(ms)), np.sum(ms)
    a, b = nu / 2, nu / (2 * lam2)                            # Gamma shape, rate
    ll = n * (a * np.log(b) - gammaln(a)) + (a - 1) * sl - b * sm
    w = np.log(LN[1] - LN[0]) - np.log(LN[-1] - LN[0])
    return np.logaddexp.reduce(ll, axis=0) + w


def run(ms):
    L = {o: log_like_lam(v) for o, v in ms.items()}
    w = np.log(LL[1] - LL[0]) - np.log(LL[-1] - LL[0])
    lz, post = {}, {}
    for name, groups in PARTS.items():
        tot = 0.0
        for g in groups:
            f = sum(L[o] for o in g)
            z = np.logaddexp.reduce(f) + w
            tot += float(z)
            p = np.exp(f + w - z)
            lam = np.exp(LL)
            mu = float(p @ lam); sd = float(np.sqrt(p @ lam ** 2 - mu ** 2))
            c = np.cumsum(p)
            post.setdefault(name, {})[g] = {"mean": mu, "sd": sd, "ci95": [float(lam[np.searchsorted(c, 0.025)]),
                                                                           float(lam[np.searchsorted(c, 0.975)])]}
        lz[name] = tot
    zz = np.array(list(lz.values()))
    pr = np.exp(zz - np.logaddexp.reduce(zz))
    best = max(lz, key=lz.get)
    return {"n_configs": {o: int(len(v)) for o, v in ms.items()},
            "rms_z_per_config": {o: float(np.sqrt(np.mean(v))) for o, v in ms.items()},
            "log_evidence": lz, "log_bf_vs_best": {k: v - lz[best] for k, v in lz.items()},
            "log_bf_vs_shared": {k: v - lz["shared"] for k, v in lz.items()},
            "posterior_prob": dict(zip(lz, map(float, pr))), "best": best, "lam_posterior": post}


def main():
    from ..run import RESULTS
    models = sys.argv[1:] or ["o3d12", "o4d20"]
    out = {}
    for m in models:
        out[m] = r = run(block_ms(m))
        print(m, "n", r["n_configs"], "rms", {k: round(v, 2) for k, v in r["rms_z_per_config"].items()})
        print("   logBF vs best", {k: round(v, 1) for k, v in r["log_bf_vs_best"].items()}, "best", r["best"])
        print("   lam (best)", {g: f"{v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" for g, v in r["lam_posterior"][r["best"]].items()})
    json.dump(out, open(RESULTS / "scale_evidence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
