"""Propagation to QoIs (spec §4) and validation against DFT (spec §5).

QoIs and tolerances are bench/si_qoi/score.py's (21 after its exclusion of compressed hcp); the phonon RMS has
no DFT value (it is a distance to DFT), so it is left out of coverage, z and the joint test (RECON Q2) and
reported against its observed value separately.  Predictions: the mean model's harness value Q_hat (CR3, B1,
B3 share theta_hat; B2 uses POPS's own mean c*).

sigma per QoI:
  delta      lam |L^T g|, g the implicit-differentiation row (delta.py), for the smooth subset (RECON Q1)
  committee  lam x sd over K harness runs drawn at lam = 1 (deviations from Q_hat rescaled: RECON Q4)
Intervals are Gaussian, Q_hat +- {1, 1.96} sigma; the committee's 2.5/97.5 % percentiles (rescaled) are
reported too.  Headline sigma (method "headline"): delta where the delta and the lam = 1 committee agree to
10 %, else the committee (§4c).
"""
import sys

import numpy as np

from .harness import REPO, ROOT, qois

sys.path.insert(0, str(REPO / "bench" / "si_qoi"))
from score import TOLERANCES  # noqa: E402

PHONON = "phonon_f rms vs DFT (THz)"
KEYS = [k for k, _, _ in TOLERANCES]
COVER_KEYS = [k for k in KEYS if k != PHONON]
TOL = {k: (t, kind) for k, t, kind in TOLERANCES}
ELASTIC = ("diamond_c11", "diamond_c12", "diamond_c44")
DELTA_TOL = 0.10


def tol_abs(k, ref):
    t, kind = TOL[k]
    return t * abs(ref) if kind == "rel" else t


def harness_values(name):
    q = qois(name, ROOT)
    return np.array([q.get(k, np.nan) if q.get(k) is not None else np.nan for k in KEYS], float)


def committee(tag, K):
    """(K, n_keys) harness QoIs of samples 0..K-1 (NaN where a test failed or is missing)."""
    from .harness import sample_name
    return np.stack([harness_values(sample_name(tag, k)) for k in range(K)])


def dft():
    return harness_values("CASTEP")


def delta_rows(lin, mean_name, log=print):
    """{key: g (p,)} for the smooth subset, plus {key: (Q_rows, Q_harness, note)} checks and the energy-route
    elastic rows ("energy:<key>").  mean_name: the harness name of theta_hat's run (its relaxed frames)."""
    from .delta import Diamond, decohesion_rows, energy_qoi_rows
    th = lin.theta_hat
    qh = qois(mean_name, ROOT)
    d = Diamond(lin, qh["diamond_a0"])
    a0 = d.a0(th)
    da0 = d.da0(th, a0)
    G, checks = {}, {"diamond_a0": (a0, qh["diamond_a0"], "")}
    Cs, Gs = d.elastic_stress(th, rows=True)
    Ce, Ge = d.elastic_energy(th, rows=True)
    for i, k in enumerate(ELASTIC):
        G[k] = Gs[i]
        G[f"energy:{k}"] = Ge[i]
        checks[k] = (Cs[i], qh[k], "stress route (harness's)")
        checks[f"energy:{k}"] = (Ce[i], qh[k], "energy route")
    G["diamond_bulk_modulus"] = (Gs[0] + 2 * Gs[1]) / 3
    G["energy:diamond_bulk_modulus"] = (Ge[0] + 2 * Ge[1]) / 3
    checks["diamond_bulk_modulus"] = ((Cs[0] + 2 * Cs[1]) / 3, qh["diamond_bulk_modulus"], "")
    rd = lambda t: ROOT / "test-runner" / f"model-{mean_name}-test-{t}"
    for k, (Q, g, note, _) in energy_qoi_rows(lin, rd, a0, da0, th).items():
        checks[k] = (Q, qh.get(k), note)
        if g is not None:
            G[k] = g
    Q, g, i = decohesion_rows(lin, rd("surface-decohesion-100-unrelaxed"), a0, da0, th)
    G["decohesion_max_stress"] = g
    checks["decohesion_max_stress"] = (Q, qh["decohesion_max_stress"], f"max at opening index {i}")
    return G, checks, {"a0": a0, "da0": da0}


def elastic_committee(lin, thetas, a_ref):
    """(K, 2, 3): (stress route, energy route) (C11, C12, C44) of the JAX re-implementation per sample."""
    from .delta import Diamond
    d = Diamond(lin, a_ref)
    return np.stack([[d.elastic_stress(t), d.elastic_energy(t)] for t in thetas])


def clopper(k, n, a=0.05):
    from scipy.stats import binomtest
    if n == 0:
        return (np.nan, np.nan)
    ci = binomtest(int(k), int(n)).proportion_ci(1 - a, "exact")
    return (float(ci.low), float(ci.high))


def joint_test(d, S, eps_rel=(1e-4, 1e-3, 1e-2), rank_tol=1e-10):
    """d^T (S + eps I)^-1 d with eps = eps_rel x mean diag S, against chi^2 with df = numerical rank of S;
    also on the correlation-scaled problem (units differ between QoIs, so the absolute ridge favours the
    large-unit QoIs: the scaled form is the unit-free check)."""
    from scipy.stats import chi2
    out = {}
    for label, (dd, SS) in {"absolute": (d, S),
                            "scaled": (d / np.sqrt(np.diag(S)), S / np.sqrt(np.outer(np.diag(S), np.diag(S))))}.items():
        ev = np.linalg.eigvalsh(SS)
        df = int((ev > rank_tol * ev.max()).sum())
        res = {"df": df, "n": len(dd)}
        for e in eps_rel:
            eps = e * float(np.mean(np.diag(SS)))
            m = float(dd @ np.linalg.solve(SS + eps * np.eye(len(dd)), dd))
            res[f"eps_{e:g}"] = {"eps": eps, "mahalanobis": m, "p_value": float(chi2.sf(m, df)),
                                 "chi2_95": float(chi2.ppf(0.95, df))}
        out[label] = res
    return out
