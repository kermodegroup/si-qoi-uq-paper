"""Per-QoI table, delta-vs-committee check, coverage, joint tests and sharpness (spec §4c, §5, §6).

Inputs (all under $SIQ_RUNS): diag/<m>_scales.json, delta/<m>.npz, the harness outputs of SIQ_<m>_cr3 (mean),
SIQ_<m>_{cr3,posterior,cr3lamF,pops}_<k> and SIQ_<m>_pops (POPS mean).  Outputs: results/qoi_uq.csv,
results/delta_vs_committee.csv, results/coverage.csv, results/joint_tests.json.
"""
import csv, json

import numpy as np

from .harness import RUNS
from .propagate import COVER_KEYS, KEYS, PHONON, DELTA_TOL, clopper, committee, dft, harness_values, joint_test, tol_abs

Z68, Z95 = 1.0, 1.959963984540054
POPS_SETS = (("pops", "B2"), ("popsblr", "B2blr"), ("popslev50", "B2lev50"), ("popsblrlev50", "B2blrlev50"))


def _lin(m, v):
    from .linear import LinearACE
    d = RUNS / "fits" / m / v
    return LinearACE(d / "model.npz", d / "posterior.npz")


def _committee(m, set_, K):
    p = RUNS / "samples" / f"{m}_{set_}.npz"
    if not p.exists():
        return None
    C = committee(f"{m}_{set_}", K)
    return None if np.all(np.isnan(C)) else C


K_MIN = 50                       # spec §4b: a committee needs >= 50 finished members for a QoI


def csd(c):
    """Committee SD of one QoI, None when fewer than K_MIN members finished (failed relaxations are NaN)."""
    c = c[~np.isnan(c)]
    return float(np.std(c, ddof=1)) if len(c) >= K_MIN else None


def robust_sd(c):
    """Half the 16-84 % range: the committee's spread without its tail of unphysical members."""
    c = c[~np.isnan(c)]
    return float(np.diff(np.percentile(c, [15.865525393145708, 84.13447460685429]))[0] / 2) if len(c) > 2 else None


def model_tables(m, K=200, log=print):
    sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
    lam = {f"lam_{k}": v for k, v in sc["lam_cr3"].items()}
    qb1 = sc["q_b1"]
    Qhat = harness_values(f"SIQ_{m}_cr3")
    ref = dft()
    dz = np.load(RUNS / "delta" / f"{m}.npz")
    G = dict(zip([str(k) for k in dz["keys"]], dz["G"]))
    Lc, Lp = _lin(m, "cr3").factor("cr3"), _lin(m, "kappa").factor("posterior")
    facs = [("cr3", Lc), ("posterior", Lp)]
    if "lam_cr3_ell3" in sc and (RUNS / "fits" / m / "cr3_ell3" / "posterior.npz").exists():
        facs.append(("cr3_ell3", _lin(m, "cr3_ell3").factor("cr3")))
    sd_delta = {f: {k: float(np.linalg.norm(L.T @ g)) for k, g in G.items()} for f, L in facs}
    C = {s: _committee(m, s, K) for s in ("cr3", "posterior", "cr3lamF") + tuple(p for p, _ in POPS_SETS)}
    Qpops = harness_values(f"SIQ_{m}_pops") if any(C[p] is not None for p, _ in POPS_SETS) else None
    n_ok = {s: (None if c is None else (~np.isnan(c)).sum(0)) for s, c in C.items()}
    rows, dvc, pops_fail = [], [], []

    def add(method, scale, lv, i, k, mean, sig, comm=None, center=None):
        r0 = ref[i]
        tol = tol_abs(k, r0) if np.isfinite(r0) else np.nan
        row = dict(model=m, method=method, scale=scale, lam=lv, qoi=k, mean=mean, sigma=sig,
                   lo68=mean - Z68 * sig, hi68=mean + Z68 * sig, lo95=mean - Z95 * sig, hi95=mean + Z95 * sig,
                   dft=r0, tol=tol)
        if comm is not None:
            c = comm[~np.isnan(comm)]
            dev = center + lv * (c - center)                    # rescaled committee (RECON Q4)
            row.update(sample_mean=float(dev.mean()), p2_5=float(np.percentile(dev, 2.5)),
                       p97_5=float(np.percentile(dev, 97.5)), K=int(len(c)))
        if k != PHONON and np.isfinite(r0) and sig > 0:
            row.update(z=(mean - r0) / sig, cov68=int(abs(mean - r0) <= Z68 * sig),
                       cov95=int(abs(mean - r0) <= Z95 * sig))
            if "p2_5" in row:
                row["cov95_pct"] = int(row["p2_5"] <= r0 <= row["p97_5"])
        rows.append(row)

    for i, k in enumerate(KEYS):
        cc = None if C["cr3"] is None else C["cr3"][:, i]
        s_comm1 = None if cc is None else csd(cc)
        s_del1 = sd_delta["cr3"].get(k)
        flagged = None
        if s_del1 is not None and s_comm1:
            flagged = abs(s_del1 / s_comm1 - 1) > DELTA_TOL
        lamF_full = lamF_rob = None
        if C["cr3lamF"] is not None:
            lamF_full = csd(C["cr3lamF"][:, i])
            lamF_rob = robust_sd(C["cr3lamF"][:, i])
        rob1 = None if cc is None else robust_sd(cc)
        n_bad = None
        if C["cr3lamF"] is not None and lamF_rob:          # members beyond 10 robust sigma of the lam_F set
            c = C["cr3lamF"][:, i]
            n_bad = int(np.sum(np.abs(c[~np.isnan(c)] - np.nanmedian(c)) > 10 * lamF_rob))
        dvc.append(dict(model=m, qoi=k, sigma_delta_lam1=s_del1, sigma_committee_lam1=s_comm1,
                        ratio=None if (s_del1 is None or not s_comm1) else s_del1 / s_comm1, flagged=flagged,
                        sigma_committee_lamF_full=lamF_full, lam_F=lam["lam_F"],
                        ratio_lamF_full_vs_rescaled=None if (lamF_full is None or not s_comm1)
                        else lamF_full / (lam["lam_F"] * s_comm1),
                        sigma_committee_robust_lam1=rob1,
                        ratio_delta_vs_robust=None if (s_del1 is None or not rob1) else s_del1 / rob1,
                        sigma_committee_robust_lamF_full=lamF_rob,
                        ratio_lamF_full_vs_rescaled_robust=None if (lamF_rob is None or not rob1)
                        else lamF_rob / (lam["lam_F"] * rob1),
                        n_lamF_beyond_10_robust_sigma=n_bad,
                        K_ok_lamF=None if n_ok["cr3lamF"] is None else int(n_ok["cr3lamF"][i]),
                        K_ok=None if n_ok["cr3"] is None else int(n_ok["cr3"][i])))
        scales = dict(lam, lam_1=1.0)
        for sname, lv in scales.items():
            method_tag = "B3" if sname == "lam_1" else "CR3"
            if s_del1 is not None:
                add(f"{method_tag}-delta", sname, lv, i, k, Qhat[i], lv * s_del1)
            if s_comm1 is not None:
                add(f"{method_tag}-committee", sname, lv, i, k, Qhat[i], lv * s_comm1, cc, Qhat[i])
            hs = s_comm1 if (flagged or s_del1 is None) else s_del1
            if hs is not None:
                add(f"{method_tag}-headline", sname, lv, i, k, Qhat[i], lv * hs)
        sp_del = sd_delta["posterior"].get(k)
        cp = None if C["posterior"] is None else C["posterior"][:, i]
        sp_comm = None if cp is None else csd(cp)
        if sp_del is not None:
            add("B1-delta", "q_b1", qb1, i, k, Qhat[i], qb1 * sp_del)
        if sp_comm is not None:
            add("B1-committee", "q_b1", qb1, i, k, Qhat[i], qb1 * sp_comm, cp, Qhat[i])
        fl1 = sp_del is not None and sp_comm and abs(sp_del / sp_comm - 1) > DELTA_TOL
        hs = sp_comm if (fl1 or sp_del is None) else sp_del
        if hs is not None:
            add("B1-headline", "q_b1", qb1, i, k, Qhat[i], qb1 * hs)
        for cs_, meth in ((p, f"{b}-committee") for p, b in POPS_SETS):
            if C[cs_] is not None:                            # both member sets share POPS's mean c*
                cpo = C[cs_][:, i]
                sg = csd(cpo)
                if sg is not None:
                    add(meth, "lam_1", 1.0, i, k, Qpops[i], sg, cpo, Qpops[i])
                pops_fail.append(dict(model=m, set=cs_, qoi=k, K_finished=int((~np.isnan(cpo)).sum())))
        if "cr3_ell3" in sd_delta and k in sd_delta["cr3_ell3"]:      # sensitivity: default clusters, delta only
            for sname, lv in sc["lam_cr3_ell3"].items():
                add("CR3ell3-delta", f"lam_{sname}", lv, i, k, Qhat[i], lv * sd_delta["cr3_ell3"][k])
        ek = f"energy:{k}"
        if ek in sd_delta["cr3"]:                              # elastic energy route (delta; JAX committee below)
            for sname, lv in lam.items():
                add("CR3-delta-energy-route", sname, lv, i, k, Qhat[i], lv * sd_delta["cr3"][ek])
    # elastic: delta vs the JAX re-implementation's committee, both routes (lam = 1)
    el = dz["elastic_committee"]                              # (K, route, 3)
    for r, route in enumerate(("stress", "energy")):
        for j, k in enumerate(("diamond_c11", "diamond_c12", "diamond_c44")):
            gk = k if route == "stress" else f"energy:{k}"
            sdel, scom = sd_delta["cr3"][gk], float(np.std(el[:, r, j], ddof=1))
            dvc.append(dict(model=m, qoi=f"{k} [{route} route, JAX committee]", sigma_delta_lam1=sdel,
                            sigma_committee_lam1=scom, ratio=sdel / scom, flagged=abs(sdel / scom - 1) > DELTA_TOL))
    extra = {"Qhat": Qhat, "ref": ref, "C": C, "G": G, "Lc": Lc, "Lp": Lp, "lam": lam, "qb1": qb1,
             "sd_delta": sd_delta, "elastic_committee": el, "Qpops": Qpops, "pops_finished": pops_fail}
    return rows, dvc, extra


def coverage(rows):
    out = []
    groups = {}
    for r in rows:
        if r["qoi"] in COVER_KEYS and "cov68" in r:
            groups.setdefault((r["model"], r["method"], r["scale"]), []).append(r)
    for (m, meth, sc), rs in groups.items():
        n = len(rs)
        k68, k95 = sum(r["cov68"] for r in rs), sum(r["cov95"] for r in rs)
        sharp = float(np.median([r["sigma"] / r["tol"] for r in rs]))
        d = dict(model=m, method=meth, scale=sc, lam=rs[0]["lam"], n=n, k68=k68, frac68=k68 / n,
                 ci68_lo=clopper(k68, n)[0], ci68_hi=clopper(k68, n)[1], k95=k95, frac95=k95 / n,
                 ci95_lo=clopper(k95, n)[0], ci95_hi=clopper(k95, n)[1], median_sigma_over_tol=sharp,
                 rms_z=float(np.sqrt(np.mean([r["z"] ** 2 for r in rs]))))
        if all("cov95_pct" in r for r in rs):
            d["k95_pct"] = sum(r["cov95_pct"] for r in rs)
        out.append(d)
    return out


def joint(m, extra):
    """Two joint tests per scale: committee covariance over every coverage QoI the committee has, and the
    delta covariance over the delta subset (spec §5: delta for linear/relaxed QoIs, committee otherwise;
    reported separately rather than spliced, so each S is a genuine covariance)."""
    ref, Qhat, C, G = extra["ref"], extra["Qhat"], extra["C"], extra["G"]
    idx = [KEYS.index(k) for k in COVER_KEYS]
    res = {}
    cases = [(f"CR3/{s}", lv, "cr3", extra["Lc"]) for s, lv in extra["lam"].items()]
    cases += [("B3/lam_1", 1.0, "cr3", extra["Lc"]), ("B1/q_b1", extra["qb1"], "posterior", extra["Lp"])]
    for name, lv, cset, L in cases:
        r = {}
        if C[cset] is not None:
            ok = [i for i in idx if np.isfinite(ref[i]) and (~np.isnan(C[cset][:, i])).sum() >= K_MIN]
            X = C[cset][:, ok]
            X = X[~np.isnan(X).any(1)]
            S = lv ** 2 * np.cov(X, rowvar=False)
            r["committee"] = dict(qois=[KEYS[i] for i in ok], K_rows=len(X), **joint_test(Qhat[ok] - ref[ok], S))
        dk = [k for k in COVER_KEYS if k in G and np.isfinite(ref[KEYS.index(k)])]
        GL = np.stack([L.T @ G[k] for k in dk])
        S = lv ** 2 * GL @ GL.T
        ii = [KEYS.index(k) for k in dk]
        r["delta"] = dict(qois=dk, **joint_test(Qhat[ii] - ref[ii], S))
        res[name] = r
    for cs_, name in ((p, f"{b}/lam_1") for p, b in POPS_SETS):
        if C[cs_] is None:
            continue
        ok = [i for i in idx if np.isfinite(ref[i]) and (~np.isnan(C[cs_][:, i])).sum() >= K_MIN]
        X = C[cs_][:, ok]
        X = X[~np.isnan(X).any(1)]
        if ok and len(X) > len(ok):
            res[name] = {"committee": dict(qois=[KEYS[i] for i in ok], K_rows=len(X),
                                           **joint_test(extra["Qpops"][ok] - ref[ok], np.cov(X, rowvar=False)))}
    return res


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
