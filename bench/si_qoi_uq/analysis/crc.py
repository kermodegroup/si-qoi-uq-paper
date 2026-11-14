"""Conformal risk control (CRC) for the per-atom force regions, with the configuration as the exchangeable unit (plan §5).

Per-atom regions {e : s(e) <= q}, s the anisotropic score of a method's 3x3 covariance (s = sqrt(e^T (C + eps tr C/3 I)^-1 e)
at unit scale), are nested in q, so the loss of a configuration c,
    L_c(q) = #{atoms i in c : s_i > q} / nbar          (nbar = mean atoms per calibration configuration)
is non-increasing in q and bounded by B = n_max / nbar (Mondrian: per group, the group's atoms per configuration). CRC (Angelopoulos et al., "Conformal risk control") takes
    q_hat = inf{q : (sum_c L_c(q) + B) / (n + 1) <= alpha}       over the n calibration configurations,
which gives E[L_test(q_hat)] <= alpha for an exchangeable test configuration: the expected number of miscovered atoms per
configuration is at most alpha nbar, i.e. the atom-weighted miscoverage the paper targets (Section 2.5), now with a
finite-sample guarantee. The paper's atom-weighted quantile over pooled atoms has none (atoms of one configuration are not
exchangeable); the configuration-weighted pooled quantile (each configuration one in total, test point of one configuration's
weight; ace-jax fit/conformal.py) has one, for the configuration-weighted target, and under-covers atoms.
n_max: the largest configuration in the calibration and test splits (atom counts only, no labels).

A. The paper's scale (ARD posterior, PR 137 fits): T_val calibration scores with their configurations (posterior.cal), test
   scores of the same shape (results/force_calibration_<m>.npz). q from the three rules, one group and Mondrian; atom
   coverage on the test split.
Groups (Mondrian, per-group risk): the ARD posterior's geometry groups merged to three shell-distortion bands, groups
{0, 1}, {2, 3}, {4..7} (the top band has 31-41 test configurations, too few for a per-group CRC on half the split).
B. A selection score: on the test split (results/force_baselines_<m>.npz: per-atom scores, log det and configuration of the
   headline (CR3), B1, B2 and the POPS ellipsoid at o3d12 and o4d20), random configuration-level splits. Every candidate is
   CRC-calibrated, so none can look better by being overconfident; its size is the mean log radius of the calibrated
   regions, r = (log det(C) / 2 + 3 log q_hat) / 3 (log of the geometric-mean semi-axis, eV/A), smaller is better.
   (1) halves: q_hat on one, size and realised risk on the other (200 splits);
   (2) selection: thirds S / C / E; choose the smallest size on S (calibrated on S by 2-fold), calibrate the choice on C,
   report its risk on E; and the naive protocol (choose and calibrate on S u C) for contrast.

    SIQ_RUNS=<BLR root> ARD_RUNS=<ARD root> python -m bench.si_qoi_uq.analysis.crc   -> results/crc.json
"""
import json, os, pathlib

import numpy as np

ALPHAS = (0.1, 0.05, 0.2)
N_SPLIT = 200
ARD_RUNS = pathlib.Path(os.environ.get("ARD_RUNS", "/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_ard137"))
METHODS = {"headline": "headline", "B1": "B1", "B2": "B2", "ellipsoid": "ellipsoid"}   # npz suffix


def crc_q(s, cfg, alpha, nbar, B):
    """CRC threshold over the configurations in cfg: the smallest q (a score, or inf) with (sum_c L_c(q) + B)/(n+1) <= alpha.
    sum_c L_c(q) = #{s_i > q} / nbar, so the condition is #{s_i > q} <= ((n + 1) alpha - B) nbar."""
    n = len(np.unique(cfg))
    k = int(np.floor(((n + 1) * alpha - B) * nbar + 1e-9))      # atoms allowed above q
    if k < 0:
        return np.inf
    o = np.sort(s)[::-1]
    return float(o[k]) if k < len(o) else 0.0                   # q = the (k+1)-th largest: exactly k strictly above


def atom_q(s, alpha):
    """The paper's atom-weighted split-conformal quantile over pooled atoms."""
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[k - 1]) if k <= n else np.inf


def cfgw_q(s, cfg, alpha):
    """Configuration-weighted pooled quantile (each configuration weighs one; test point one configuration's weight)."""
    _, inv, cnt = np.unique(cfg, return_inverse=True, return_counts=True)
    w, n = 1.0 / cnt[inv], len(cnt)
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / (n + 1)
    k = int(np.searchsorted(cw, 1 - alpha - 1e-15))
    return float(s[o][k]) if k < len(s) else np.inf


def part_a(m, alpha_list):
    from ace_jax.fit.ard import ARDPosterior
    from ..run import RESULTS
    post = ARDPosterior.load(ARD_RUNS / "fits" / m / "cr3" / "posterior.npz")
    cal = post.cal
    s, g, c = (np.asarray(cal[k]) for k in ("scores", "groups", "cfg"))
    s = s.astype(float)
    z = np.load(RESULTS / f"force_calibration_{m}.npz", allow_pickle=True)
    st, gt = z["score"], z["group"]
    zb = np.load(RESULTS / f"force_baselines_{m}.npz", allow_pickle=True)   # same test atoms; their configurations
    assert len(zb["cfg"]) == len(st)
    ct = zb["cfg"]
    _, ncal = np.unique(c, return_counts=True)
    _, ntest = np.unique(ct, return_counts=True)
    nbar, nmax = float(ncal.mean()), int(max(ncal.max(), ntest.max()))
    G = int(max(g.max(), gt.max())) + 1
    # per group: the most atoms of that group in one configuration (calibration or test; counts only), its loss bound
    nmax_g = [int(max(np.bincount(c[g == k]).max() if np.any(g == k) else 0,
                      np.bincount(ct[gt == k]).max() if np.any(gt == k) else 0)) for k in range(G)]
    out = {"n_cal_cfg": int(len(ncal)), "n_cal_atoms": int(len(s)), "nbar": nbar, "n_max": nmax, "B": nmax / nbar, "n_max_group": nmax_g,
           "n_test_atoms": int(len(st)), "n_test_cfg": int(len(ntest)), "alpha": {}}
    for a in alpha_list:
        r = {}
        def crc(x, cc, nm):                    # nbar over the configurations that have atoms here; B = nm / nbar
            nb = float(np.unique(cc, return_counts=True)[1].mean())
            return crc_q(x, cc, a, nb, nm / nb)
        for rule in ("atom", "cfgw", "crc"):
            qa = {"atom": lambda x, cc, nm: atom_q(x, a), "cfgw": lambda x, cc, nm: cfgw_q(x, cc, a), "crc": crc}[rule]
            q1 = qa(s, c, nmax)
            qg = np.array([qa(s[g == k], c[g == k], nmax_g[k]) if np.any(g == k) else np.inf for k in range(G)])
            cov1 = float(np.mean(st <= q1))
            covg = float(np.mean(st <= qg[gt]))
            # per test configuration miscovered fraction (the configuration-weighted view) and the CRC loss itself
            miss = st > qg[gt]
            fr = np.bincount(ct, weights=miss, minlength=ct.max() + 1)[np.unique(ct)] / ntest
            r[rule] = {"q_one": q1, "q_groups": qg.tolist(), "cov_one": cov1, "cov_mondrian": covg,
                       "cov_mondrian_cfg_mean": float(1 - fr.mean()),
                       "cov_mondrian_per_group": [float(np.mean(st[gt == k] <= qg[k])) for k in range(G)]}
        out["alpha"][f"{a:g}"] = r
    return out


def load_b():
    from ..run import RESULTS
    data = {}
    for m in ("o3d12", "o4d20"):
        z = np.load(RESULTS / f"force_baselines_{m}.npz", allow_pickle=True)
        lam = {"headline": float(z["lam_F"]), "B1": float(z["q_b1"]), "B2": 1.0, "ellipsoid": float(z["lam_e"])}
        for k, sfx in METHODS.items():
            # unit scale: score s0 = lam d, log det0 = logdet - 6 log lam (the eps term scales with C), sd0 = sd / lam
            data[(m, k)] = {"s": lam[k] * z[f"d_{sfx}"], "ld": z[f"logdet_{sfx}"] - 6 * np.log(lam[k]),
                            "sd": z[f"sd_{sfx}"] / lam[k], "g": np.minimum(np.asarray(z["group"], int) // 2, 2)}
        cfg = z["cfg"]
        data["type"] = np.asarray(z["config_type"]).astype(str)
        if "cfg" in data:
            assert np.array_equal(data["cfg"], cfg), "models disagree on the test atoms"
        data["cfg"] = cfg
    return data


def part_b(data, alpha, rng):
    """Configuration-level splits of the test split: CRC calibration (marginal, and Mondrian over each candidate's ARD
    groups), sizes (log volume radius and log trace radius) and realised risks, overall and per group."""
    cfg, ctype = data.pop("cfg"), data.pop("type")
    cands = list(data)
    uc, inv, cnt = np.unique(cfg, return_inverse=True, return_counts=True)
    nmax, n = int(cnt.max()), len(uc)
    G = {k: int(data[k]["g"].max()) + 1 for k in cands}
    # per candidate and group: the most atoms of the group in one test configuration (the loss bound, counts only)
    nmax_g = {k: [int(np.bincount(inv[data[k]["g"] == j]).max()) if np.any(data[k]["g"] == j) else 0
                  for j in range(G[k])] for k in cands}

    def q_marg(k, am, crc=True):
        x = data[k]["s"][am]
        if not crc:
            return atom_q(x, alpha)
        nb = float(np.unique(inv[am], return_counts=True)[1].mean())
        return crc_q(x, inv[am], alpha, nb, nmax / nb)

    def q_mond(k, am, crc=True):                             # per-atom q from its group's threshold
        g, x = data[k]["g"], data[k]["s"]
        qg = np.full(G[k], np.inf)
        for j in range(G[k]):
            mj = am & (g == j)
            if not mj.any():
                continue
            if crc:
                nb = float(np.unique(inv[mj], return_counts=True)[1].mean())
                qg[j] = crc_q(x[mj], inv[mj], alpha, nb, nmax_g[k][j] / nb)
            else:
                qg[j] = atom_q(x[mj], alpha)
        return qg[g]

    def size(k, bm, q, kind="vol"):                          # q: scalar or per atom (all atoms); inf -> inf
        qq = np.broadcast_to(q, data[k]["s"].shape)[bm]
        if not np.all(np.isfinite(qq)):
            return np.inf
        if kind == "vol":
            return float(np.mean(data[k]["ld"][bm] / 2 + 3 * np.log(qq)) / 3)
        return float(np.mean(np.log(qq * data[k]["sd"][bm])))

    def risks(k, bm, q):                                     # atom-weighted miscoverage, overall and per group
        miss = data[k]["s"] > q
        g = data[k]["g"]
        per = [float(np.mean(miss[bm & (g == j)])) for j in range(G[k]) if np.any(bm & (g == j))]
        return float(np.mean(miss[bm])), float(np.max(per)), float(np.sqrt(np.mean((np.array(per) - alpha) ** 2)))

    def type_risk(k, bm, q):                                 # per configuration type (types with >= 100 evaluation atoms)
        miss = data[k]["s"] > q
        per = [float(np.mean(miss[bm & (ctype == t)])) for t in np.unique(ctype[bm]) if np.sum(bm & (ctype == t)) >= 100]
        return float(np.max(per)), float(np.sqrt(np.mean((np.array(per) - alpha) ** 2)))

    keys = ("size_vol", "size_tr", "risk", "risk_worst_group", "risk_group_rms", "size_vol_mondrian", "risk_mondrian",
            "risk_worst_group_mondrian", "risk_worst_type", "risk_type_rms")
    halves = {k: {x: [] for x in keys} for k in cands}
    first = {k: {"marginal": 0, "mondrian": 0} for k in cands}
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        A = np.zeros(n, bool); A[p[: n // 2]] = True
        am, bm = A[inv], ~A[inv]
        sv, svm = {}, {}
        for k in cands:
            q, qm = q_marg(k, am), q_mond(k, am)
            r, rw, rr = risks(k, bm, q)
            rm, rmw, _ = risks(k, bm, qm)
            sv[k], svm[k] = size(k, bm, q), size(k, bm, qm)
            for x, v in zip(keys, (sv[k], size(k, bm, q, "tr"), r, rw, rr, svm[k], rm, rmw, *type_risk(k, bm, q))):
                halves[k][x].append(v)
        for lab_, d in (("marginal", sv), ("mondrian", svm)):
            b = min(d, key=d.get)
            if np.isfinite(d[b]):
                first[b][lab_] += 1
    sel = {mode: {"risk": [], "risk_worst_group": [], "chosen": {k: 0 for k in cands}} for mode in ("marginal", "mondrian")}
    naive = []
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        S = np.zeros(n, bool); S[p[: n // 3]] = True
        Cm = np.zeros(n, bool); Cm[p[n // 3: 2 * n // 3]] = True
        Em = ~(S | Cm)[inv]
        pS = p[: n // 3]                                     # 2-fold inside S with plain quantiles (selection needs no
        f1 = np.zeros(n, bool); f1[pS[: len(pS) // 2]] = True   # guarantee; CRC on ~40 configurations is infinite)
        f2 = S & ~f1
        for mode, qf in (("marginal", q_marg), ("mondrian", q_mond)):
            sz = {k: 0.0 for k in cands}
            for fa, fb in ((f1, f2), (f2, f1)):
                for k in cands:
                    sz[k] += size(k, fb[inv], qf(k, fa[inv], crc=False)) / 2
            best = min(sz, key=sz.get)
            if not np.isfinite(sz[best]):                   # every candidate infinite: no choice
                sel[mode]["chosen"]["none"] = sel[mode]["chosen"].get("none", 0) + 1
                continue
            sel[mode]["chosen"][best] += 1
            r, rw, _ = risks(best, Em, qf(best, Cm[inv]))     # the choice CRC-calibrated on C, risk on E
            sel[mode]["risk"].append(r); sel[mode]["risk_worst_group"].append(rw)
        SC = (S | Cm)[inv]                                   # naive: choose and calibrate (CRC) on S u C
        szn = {k: size(k, SC, q_marg(k, SC)) for k in cands}
        bn = min(szn, key=szn.get)
        naive.append(risks(bn, Em, q_marg(bn, SC))[0])

    def summ(x):
        x = np.asarray(x, float)
        f = np.isfinite(x)
        if not f.any():
            return {"mean": None, "p05": None, "p95": None, "frac_inf": 1.0}
        return {"mean": float(np.mean(x[f])), "p05": float(np.quantile(x[f], 0.05)), "p95": float(np.quantile(x[f], 0.95)),
                "frac_inf": float(1 - f.mean())}
    data["cfg"], data["type"] = cfg, ctype
    lab = lambda k: f"{k[0]}/{k[1]}"
    return {"n_cfg": n, "n_max": nmax,
            "halves": {lab(k): {**{x: summ(v[x]) for x in keys}, "first_marginal": first[k]["marginal"] / N_SPLIT,
                                "first_mondrian": first[k]["mondrian"] / N_SPLIT} for k, v in halves.items()},
            "selection": {mode: {"chosen": {(k if k == "none" else lab(k)): c / N_SPLIT for k, c in v["chosen"].items() if c},
                                 "risk": summ(v["risk"]), "risk_worst_group": summ(v["risk_worst_group"])}
                          for mode, v in sel.items()} | {"naive_marginal_risk": summ(naive)}}


def main():
    from ..run import RESULTS
    out = {"alphas": ALPHAS, "n_split": N_SPLIT, "A": {}, "B": {}}
    for m in ("o3d12", "o4d20"):
        out["A"][m] = r = part_a(m, ALPHAS)
        print(f"\nA. {m}: T_val {r['n_cal_cfg']} configurations ({r['n_cal_atoms']} atoms, nbar {r['nbar']:.1f}, n_max {r['n_max']},"
              f" B {r['B']:.2f}); test {r['n_test_cfg']} configurations, {r['n_test_atoms']} atoms")
        for a, v in r["alpha"].items():
            print(f"  alpha {a}: " + " | ".join(f"{k} q {x['q_one']:.2f} cov {x['cov_one']:.3f} / Mondrian {x['cov_mondrian']:.3f}"
                                               f" (cfg-mean {x['cov_mondrian_cfg_mean']:.3f})" for k, x in v.items()))
    data = load_b()
    rng = np.random.default_rng(0)
    for a in ALPHAS:
        out["B"][f"{a:g}"] = r = part_b(data, a, rng)
        print(f"\nB. alpha {a:g}: {r['n_cfg']} test configurations; halves. size = mean log radius (vol: geometric semi-axis; "
              f"tr: sqrt trace), eV/A; risk overall / worst group / group RMS dev; Mondrian: per-group CRC")
        for k, v in sorted(r["halves"].items(), key=lambda kv: kv[1]["size_vol"]["mean"]):
            f = lambda d: "inf" if d["mean"] is None else f"{d['mean']:+.3f}" + (f"(inf {d['frac_inf']:.2f})" if d["frac_inf"] else "")
            print(f"  {k:20s} vol {f(v['size_vol'])} tr {f(v['size_tr'])} risk {v['risk']['mean']:.3f} / "
                  f"{v['risk_worst_group']['mean']:.3f} / {v['risk_group_rms']['mean']:.3f} | Mondrian vol {f(v['size_vol_mondrian'])} "
                  f"risk {v['risk_mondrian']['mean']:.3f} / {v['risk_worst_group_mondrian']['mean']:.3f} | first {v['first_marginal']:.2f} "
                  f"/ {v['first_mondrian']:.2f} | type worst {v['risk_worst_type']['mean']:.3f} RMS {v['risk_type_rms']['mean']:.3f}")
        for mode in ("marginal", "mondrian"):
            sm = r["selection"][mode]
            print(f"  select by {mode} size: chosen {sm['chosen']}; risk on E {sm['risk']['mean']:.3f} "
                  f"({sm['risk']['p05']:.3f}..{sm['risk']['p95']:.3f}), worst group {sm['risk_worst_group']['mean']:.3f}")
        print(f"  naive (choose and calibrate on S u C): risk {r['selection']['naive_marginal_risk']['mean']:.3f}")
    json.dump(out, open(RESULTS / "crc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
