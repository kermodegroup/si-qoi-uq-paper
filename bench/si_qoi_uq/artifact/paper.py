"""Paper skeleton (artifact) for the silicon QoI UQ study: theory from docs/dev/specs/tex/force-uq-math-pipeline.tex
and the ace-jax fitting docs, the propagation methodology of bench/si_qoi_uq, and results read from results/.
Not part of the pipeline.

    python bench/si_qoi_uq/artifact/paper.py   -> bench/si_qoi_uq/artifact/paper/index.html (+ figures)
"""
import csv, json, pathlib, shutil

HERE = pathlib.Path(__file__).parent
RES, FIG = HERE.parent / "results", HERE.parent / "figures"
OUT = HERE / "paper"
OUT.mkdir(exist_ok=True)


def rd(name):
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    p = RES / name
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(p))] if p.exists() else []


def f(x, d=2):
    return "–" if x in ("", None) or isinstance(x, str) else f"{x:.{d}f}"


cov, sd, dvc, mw = rd("coverage.csv"), rd("scale_diagnostics.csv"), rd("delta_vs_committee.csv"), rd("member_widths.csv")
joint = json.load(open(RES / "joint_tests.json"))
phys = json.load(open(RES / "lamF_physical_filter.json")) if (RES / "lamF_physical_filter.json").exists() else {}
models = sorted({r["model"] for r in cov}, key=lambda m: ("o3d12", "o4d20").index(m))
get = lambda m, meth, sc: next((r for r in cov if r["model"] == m and r["method"] == meth and r["scale"] == sc), None)
lam = {m: {r["kind"]: r["rms_z"] for r in sd if r["model"] == m and r["factor"] == "cr3" and r["breakdown"] == "all"}
       for m in models}

# the four methods of Figures 1 and 1b (figures.PANELS), with the same labels; variants in the sensitivity table
TAB = [("CR3-headline", "lam_F", "CR3 × λ<sub>F</sub> (headline)"), ("B1-headline", "q_b1", "B1: posterior × q̂"),
       ("B2lev50-committee", "lam_1", "B2: POPS (leverage cut 50 %, the package default)"), ("B3-headline", "lam_1", "B3: CR3, λ = 1")]
TAB_SENS = [("CR3-headline", "lam_V", "CR3 × λ<sub>V</sub>"), ("CR3-headline", "lam_E", "CR3 × λ<sub>E</sub>"),
            ("CR3-headline", "lam_pool", "CR3 × λ<sub>pool</sub>"),
            ("CR3-committee-lamF-physical", "lam_F", "CR3 at λ<sub>F</sub>, full re-run, physical members"),
            ("B2lev5-committee", "lam_1", "POPS, leverage cut 5 %")]
_cov_row = lambda m, lab, r: (
    f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r['n'])}</td><td class=n>{int(r['k68'])} [{f(r['ci68_lo'])}, {f(r['ci68_hi'])}]</td>"
    f"<td class=n>{int(r['k95'])} [{f(r['ci95_lo'])}, {f(r['ci95_hi'])}]</td><td class=n>{f(r['rms_z'])}</td>"
    f"<td class=n>{f(r['median_sigma_over_tol'])}</td><td class=n>{f(r.get('crps_tol'))}</td></tr>")
sens_rows = "".join(_cov_row(m, lab, r) for m in models for meth, sc, lab in TAB_SENS for r in [get(m, meth, sc)] if r)
cov_rows = "".join(
    f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r['n'])}</td><td class=n>{int(r['k68'])} [{f(r['ci68_lo'])}, {f(r['ci68_hi'])}]</td>"
    f"<td class=n>{int(r['k95'])} [{f(r['ci95_lo'])}, {f(r['ci95_hi'])}]</td><td class=n>{f(r['rms_z'])}</td>"
    f"<td class=n>{f(r['median_sigma_over_tol'])}</td><td class=n>{f(r.get('crps_tol'))}</td></tr>"
    for m in models for meth, sc, lab in TAB for r in [get(m, meth, sc)] if r)
lam_rows = "".join(f"<tr><td>{m}</td>" + "".join(f"<td class=n>{f(lam[m].get(k))}</td>" for k in ("E", "F", "V", "pool"))
                   + f"<td class=n>{f(lam[m]['V'] / lam[m]['F'])}</td><td class=n>{f(lam[m]['E'] / lam[m]['F'])}</td></tr>"
                   for m in models)
sev = json.load(open(RES / "scale_evidence.json")) if (RES / "scale_evidence.json").exists() else {}
SEV_LAB = (("shared", "one scale"), ("E|FV", "E | F = V"), ("F|EV", "F | E = V"), ("V|EF", "V | E = F"), ("E|F|V", "E | F | V"))
sev_rows = "".join(f"<tr><td>{lab}</td>" + "".join(f"<td class=n>{sev[m]['log_bf_vs_best'][k]:+.1f}</td>" for m in ("o3d12", "o4d20") if m in sev)
                   + "</tr>" for k, lab in SEV_LAB) if sev else ""
def sev_lam(m):
    p = sev[m]["lam_posterior"][sev[m]["best"]]
    return ", ".join(f"λ<sub>{'=' .join(g)}</sub> {v['mean']:.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]" for g, v in p.items())
sev_bf = lambda m: f"{-sev[m]['log_bf_vs_best']['shared']:.0f}"
sev_sep = lambda m: f"{-sev[m]['log_bf_vs_best']['E|F|V']:.1f}"
zlist = lambda m: ", ".join(f"{float(r['z']):+.1f}" for r in _qrows if r["model"] == m and r["method"] == "CR3-headline"
                           and r["scale"] == "lam_F" and "interstitial" in r["qoi"])
ev = {(r["model"], r["method"], r["kind"]): r for r in rd("evidence.csv")}
def pcov(m, meth="CR3-headline", sc="lam_F", nominal=0.95):
    from scipy.stats import beta
    r = get(m, meth, sc)
    k, n = int(float(r["k95"])), int(float(r["n"]))
    return f"{beta(k + 0.5, n - k + 0.5).sf(nominal):.2f}"
def ci_txt(k, n):
    from scipy.stats import beta
    post = beta(k + 0.5, n - k + 0.5)
    return f"[{post.ppf(0.025):.2f}, {post.ppf(0.975):.2f}], with P(coverage ≥ 0.95) = {post.sf(0.95):.2f}"
EV_EPS = "1e-06"
import numpy as np
def sci(x, d=0):
    """x as a × 10^b in HTML (d decimals on a)."""
    e = int(np.floor(np.log10(abs(x))))
    a = x / 10 ** e
    return (f"10<sup>{e}</sup>" if round(a, d) == 1 else f"{a:.{d}f} × 10<sup>{e}</sup>").replace("-", "−")


def lbf(m, meth, kind="committee"):
    r = ev.get((m, meth, kind))
    return float(r["log_bf_vs_headline"]) if r and r.get("log_bf_vs_headline") not in ("", None) else float("nan")
lbf_txt = lambda m, meth, kind="committee": f"{lbf(m, meth, kind):+.1f}"
lbf2_txt = lambda m, a, b, kind="committee": f"{lbf(m, a, kind) - lbf(m, b, kind):+.1f}"   # log BF of method a against b
_ood = {m: json.load(open(RES / f"ood_mondrian_{m}.json")) for m in ("o3d12", "o4d20") if (RES / f"ood_mondrian_{m}.json").exists()}
_lso = {m: json.load(open(RES / f"learned_scale_ood_{m}.json")) for m in ("o3d12", "o4d20") if (RES / f"learned_scale_ood_{m}.json").exists()}
_lsm = {m: json.load(open(RES / f"learned_scale_{m}.json")) for m in ("o3d12", "o4d20") if (RES / f"learned_scale_{m}.json").exists()}
_evn = json.load(open(RES / "evenness.json")) if (RES / "evenness.json").exists() else {}
OOD_COLS = (("all", "all held out"), ("crack_110_1-10", "crack (110)"), ("crack_111_1-10", "crack (111)"), ("screw_disloc", "screw dislocation"))
OOD_METH = (("one_group", "one hold-out quantile"), ("learned", "learned scale"), ("global", "global scale (ID test)"))
def _ood_get(m, k, c="all", ci=False):
    """Coverage (or its 90 % configuration-bootstrap interval) on the held-out set: learned_scale_ood_<m>.json for the
    conformal methods, ood_mondrian_<m>.json for the global scale."""
    if k == "global":
        r = _ood.get(m)
        r = None if r is None else (r["all"] if c == "all" else r["per_type"][c])
    else:
        r = _lso.get(m)
        r = None if r is None else (r["sets"]["ood"]["all"] if c == "all" else r["sets"]["ood"]["per_type"][c])
    if r is None:
        return None
    return r.get(f"{k}_ci90") if ci else r[k]
def _ood_id(m, k):
    if k == "global":
        return _ood[m]["in_distribution"]["global"] if m in _ood else None
    return _lso[m]["sets"]["id_test"]["all"][k] if m in _lso else None
def _ood_cell(m, k, c):
    v, ci = _ood_get(m, k, c), _ood_get(m, k, c, ci=True)
    return "–" if v is None else f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
ood_rows = "".join(f"<tr><td>{m}</td><td>{lab}</td><td class=n>{f(_ood_id(m, k))}</td>" + "".join(
    f"<td class=n>{_ood_cell(m, k, c)}</td>" for c, _ in OOD_COLS) + "</tr>" for m in ("o3d12", "o4d20") if m in _lso for k, lab in OOD_METH)
ood_n = lambda m: f"{_ood[m]['all']['atoms']} atoms in {_ood[m]['all']['configs']} configurations" if m in _ood else "–"
ood_v = lambda m, k, c="all": "–" if _ood_get(m, k, c) is None else f"{_ood_get(m, k, c):.2f}"
_atr = json.loads((RES / "ard_transfer.json").read_text()) if (RES / "ard_transfer.json").exists() else {}
atr = lambda m, k, fmt=".2f": format(_atr[m][k], fmt).replace("-", "−") if m in _atr else "–"   # hold-out -> full transfer (analysis/ard_transfer.py)
atr_pct = lambda m: f"{100 * (_atr[m]['test_q_over_q_one'] - 1):.1f} %" if m in _atr else "–"
lm = lambda m, k: "–" if m not in _lsm else f"{_lsm[m]['sets']['test']['all'][k]:.2f}"
lm_seeds = lambda m: "–" if m not in _lsm else "–".join(f"{x:.2f}" for x in _lsm[m]["sets"]["test"]["all"]["learned_seed_range"])
lm_rad = lambda m, k: "–" if m not in _lsm else f"{_lsm[m]['sets']['test']['all']['median_radius'][k]:.1f}"
lm_rms = lambda m: "–" if m not in _lsm else f"{_lsm[m]['sets']['test']['all']['rms_e_over_sigma_learned']:.2f}"
lm_med = lambda m: "–" if m not in _lsm else f"{_lsm[m]['sets']['test']['all'].get('median_e_over_sigma_learned', float('nan')):.2f}"
evdev = lambda tag, st, k: "–" if tag not in _evn else f"{_evn[tag][st][k]['rms_dev']:.3f}"
def _type_rows():
    rows = ""
    for m in ("o3d12", "o4d20"):
        if m not in _evn or m not in _lsm:
            continue
        e_ = _evn[m]["test"]
        pt = _lsm[m]["sets"]["test"]["per_type"]
        for t in sorted(e_["one_group"]["per_type"], key=lambda t: -pt[t]["atoms"]):
            rows += (f"<tr><td>{m}</td><td>{t}</td><td class=n>{pt[t]['atoms']}</td><td class=n>{pt[t]['configs']}</td>"
                     f"<td class=n>{e_['one_group']['per_type'][t]:.2f}</td><td class=n>{e_['learned']['per_type'][t]:.2f}</td></tr>")
        rows += (f"<tr><td>{m}</td><td>RMS deviation from {e_['nominal']:.2f}</td><td></td><td></td>"
                 f"<td class=n>{e_['one_group']['rms_dev']:.3f}</td><td class=n>{e_['learned']['rms_dev']:.3f}</td></tr>")
    return rows
type_rows = _type_rows()
def fb_types(m, k, level=0.9, min_atoms=300):
    """Lowest and highest per-type coverage (types with >= min_atoms test atoms, as Figure 2) of method k, and the
    lowest types: '0.57 (surface_001) to 1.00'."""
    import numpy as _n
    from scipy.stats import chi
    p = RES / f"force_baselines_{m}.npz"
    if not p.exists():
        return "–"
    b = _n.load(p, allow_pickle=True)
    cov = b[f"d_{k}"] <= chi(3).ppf(level)
    t = b["config_type"]
    per = {u: float(cov[t == u].mean()) for u in _n.unique(t) if (t == u).sum() >= min_atoms}
    lo = min(per, key=per.get)
    return f"{per[lo]:.2f} ({lo}) to {max(per.values()):.2f}"


def fb_iso(m, k, level=0.9):
    """Coverage of the isotropic ball |e| <= sigma sqrt(chi2_3(level) / 3), sigma = sqrt(tr C), method k (for contrast)."""
    import numpy as _n
    from scipy.stats import chi2
    b = _n.load(RES / f"force_baselines_{m}.npz")
    return f"{float(_n.mean(b['err_norm'] <= _n.sqrt(chi2.ppf(level, 3) / 3) * b[f'sd_{k}'])):.2f}"


def fb_eps_max(level=0.9):
    """Largest change in ellipsoid coverage of any method and model when eps is raised tenfold (d_<k>_eps0.01)."""
    import numpy as _n
    from scipy.stats import chi
    k3, dm = chi(3).ppf(level), 0.0
    for m in ("o3d12", "o4d20"):
        b = _n.load(RES / f"force_baselines_{m}.npz")
        for k in ("headline", "B1", "B2", "B3"):
            dm = max(dm, abs(float(_n.mean(b[f"d_{k}_eps0.01"] <= k3) - _n.mean(b[f"d_{k}"] <= k3))))
    return f"{dm:.3f}"


def fb(m, k, level=0.9):
    """Coverage of the Gaussian 90 % per-atom ellipsoid of method k's own force covariance (results/force_baselines_<m>.npz)."""
    import numpy as _n
    from scipy.stats import chi
    p = RES / f"force_baselines_{m}.npz"
    if not p.exists():
        return "–"
    b = _n.load(p)
    return f"{float(_n.mean(b[f'd_{k}'] <= chi(3).ppf(level))):.2f}"
dcrps = lambda m, meth, sc: f(disc.get(m, {}).get(f"{meth}+disc/{sc}", {}).get("crps_tol"))
crps = lambda m, meth, sc: f(get(m, meth, sc).get("crps_tol")) if get(m, meth, sc) else "–"
EV_ROWS = (("CR3/lam_F", "CR3 × λ<sub>F</sub> (headline)"), ("B1/q_b1", "B1: posterior × q̂"),
           ("B2lev50/lam_1", "B2: POPS (leverage cut 50 %)"), ("B3/lam_1", "B3: CR3, λ = 1"),
           ("CR3/lam_V", "CR3 × λ<sub>V</sub> (sensitivity)"), ("CR3/lam_F+disc", "CR3 × λ<sub>F</sub> + discrepancy term"),
           ("B1/q_b1+disc", "B1 + discrepancy term"))
jrows = ""
for m in models:
    for meth, lab in EV_ROWS:
        for kind in ("committee", "delta"):
            r = ev.get((m, meth, kind))
            if not r:
                continue
            jrows += (f"<tr><td>{m}</td><td>{lab}</td><td>{kind}</td><td class=n>{int(float(r['n']))}</td>"
                      f"<td class=n>{float(r['log_p_eps' + EV_EPS]):.1f}</td><td class=n>{lbf(m, meth, kind):+.1f}</td>"
                      f"<td class=n>{float(r['mahalanobis']):.1f}</td><td class=n>{float(r['logdet']):.1f}</td></tr>")
nflag = {m: sum(1 for r in dvc if r["model"] == m and r.get("flagged") in ("True", True) and "JAX" not in r["qoi"]) for m in models}
ncmp = {m: sum(1 for r in dvc if r["model"] == m and r.get("flagged") in ("True", "False", True, False) and "JAX" not in r["qoi"]) for m in models}
head = {m: get(m, "CR3-headline", "lam_F") for m in models}
pw = {(r["model"], r["set"]): r for r in mw}

S = []  # sections


def sec(id_, title, body, status="", level=2):
    chip = ""                  # drafting status chips are not shown in the manuscript
    head = f'<h{level}>{title} {chip}</h{level}>' if level else ""
    S.append(f'<section id="{id_}">{head}{body}</section>')


b3 = {m: get(m, "B3-headline", "lam_1") for m in models}
b1 = {m: get(m, "B1-headline", "q_b1") for m in models}
lam_lo = min(min(v for k, v in lam[m].items() if k != "pool") for m in models)
lam_hi = max(max(v for k, v in lam[m].items() if k != "pool") for m in models)
phr = {(r["model"], r["qoi"]): r for r in rd("lamF_physical.csv")}
_ratios = lambda m, ph: [float(r["ratio_physical_vs_rescaled"]) for (mm, k), r in phr.items() if mm == m and k.startswith("dE_") == ph
                         and r["ratio_physical_vs_rescaled"] not in ("", None)]
nonphase = {m: _ratios(m, False) for m in models}
phase = {m: _ratios(m, True) for m in models}
_dvc = [r for r in dvc if r.get("flagged") in ("True", "False", True, False) and "JAX" not in r["qoi"]]
dvc_rng = {m: (min(float(r["ratio"]) for r in _dvc if r["model"] == m), max(float(r["ratio"]) for r in _dvc if r["model"] == m)) for m in models}
dvc_flag = {m: [r["qoi"] for r in _dvc if r["model"] == m and r.get("flagged") in ("True", True)] for m in models}
disc_head = "; ".join(f"{m}: {v[k]['k95']}/{v[k]['n']}" for m, v in
                      (json.load(open(RES / "discrepancy.json")).items() if (RES / "discrepancy.json").exists() else [])
                      for k in ["CR3-headline+disc/lam_F"] if k in v) or "[pending]"
headline = "; ".join(f"{m}: {int(h['k95'])}/{int(h['n'])} at 95 % and {int(h['k68'])}/{int(h['n'])} at 68 %"
                     for m, h in head.items() if h)

disc_l = json.load(open(RES / "discrepancy.json")) if (RES / "discrepancy.json").exists() else {}
disc_txt = "; ".join(f"{m}: {disc_l[m]['CR3-headline+disc/lam_F']['k95']}/{disc_l[m]['CR3-headline+disc/lam_F']['n']}"
                     for m in ("o3d12", "o4d20") if m in disc_l) or "[pending]"

_cv = {r["model"]: float(r["median_sigma_over_tol"]) for r in rd("coverage.csv") if r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
tight_ratio = _cv["o3d12"] / _cv["o4d20"] if {"o3d12", "o4d20"} <= set(_cv) else float("nan")

sec("abstract", "Abstract", f"""
<p>Machine-learned interatomic potentials are misspecified, so Bayesian posterior variances understate their errors. For linear
atomic cluster expansion (ACE) potentials we derive calibrated uncertainties from one misspecification-robust parameter
covariance, the delete-one-configuration jackknife. Scaled by one conformal quantile from a held-out part of the training
set, it gives per-atom force uncertainties close to their target coverage on held-out silicon data, including cracks left out
of the fit. With one scale fitted to held-out forces,
the same parameter distribution is propagated to 21 properties of silicon, from elastic constants to defect and surface energies,
by implicit differentiation through each relaxation and by committees, and is tested against DFT without tuning on any property.
For a smaller and a larger model the intervals cover most properties, and the larger model's are about {tight_ratio:.0f} times
tighter. The Bayesian posterior scaled the same way has the wrong shape: it covers fewer properties and has far lower joint
evidence. POPS, once its hypercube is numerically defined, performs comparably on the properties, but its per-atom force
regions are not calibrated to a target ({fb("o3d12", "B2")} and {fb("o4d20", "B2")} at a nominal 0.90). The exceptions are relaxed point defects, whose training cells are all hot snapshots: a bias that every configuration
shares is invisible to a jackknife, but it can be measured from the training labels, and as a discrepancy term it restores
coverage.</p>""", "draft")

sec("intro", "1  Introduction", """
<p>Machine-learned interatomic potentials are used for forces, in molecular dynamics and in active learning where force errors
must be controlled during a simulation, and for the derived properties that a potential is built to predict. Both uses need an
uncertainty. The training labels are deterministic DFT energies, forces and stresses, so the error of a potential is approximation
error: the model is misspecified. A Bayesian posterior is then far too narrow. Under misspecification it concentrates at the
best-fitting parameters at a rate set by the amount of data, not by the size of the model error, and it collapses to a point as
N<sub>data</sub> → ∞ while the prediction error stays finite [Kleijn and van der Vaart 2012; Müller 2013; Swinburne and Perez 2024].</p>
<p>The most widely used alternatives for forces are committees and ensembles of models [Musil et al. 2019; Imbalzano et al. 2021;
Kellner and Ceriotti 2024] and approximate posteriors on the last layer of a network [Bigi et al. 2024]. They need several fits, or a
calibration of their own. Conformal prediction gives intervals with guaranteed coverage under exchangeability [Vovk, Gammerman and
Shafer 2005; Tibshirani et al. 2019; Barber et al. 2021], and Best et al. [2024] combined it with ensembles of ACE potentials for
silicon. A single conformal scale covers on average, however, while force errors vary strongly between local environments and are
correlated within a configuration.</p>
<p>In this paper we construct calibrated per-atom force uncertainties for linear atomic cluster expansion (ACE) potentials [Drautz
2019] from two parts. The shape is a cluster-robust ("sandwich") covariance of the coefficients [Huber 1967; White 1982], with each
training configuration as one cluster of correlated rows. We use the variant proposed by Bell and McCaffrey [2002] and named CR3 by
Pustejovsky and Tipton [2018] (CV3 in MacKinnon, Nielsen and Webb [2023]), the cluster form of the heteroskedasticity-consistent HC3
estimator [MacKinnon and White 1985]. CR3 is the delete-one-cluster jackknife: it measures how far the fitted coefficients move when
each configuration is left out. It is computed exactly, without refitting, from the PRESS (predicted residual error sum of squares)
residuals. These are the residuals each row would have if its cluster were left out of the fit, and they follow from the ordinary
residuals by a leverage correction: for one row, division by one minus its leverage; for a cluster, the matrix analogue [Allen 1974;
Cook and Weisberg 1982]. The per-atom 3×3 block of the covariance gives anisotropic force regions.</p>
<p>The scale comes from split conformal prediction [Vovk, Gammerman and Shafer 2005], which turns the scores of held-out
calibration points into a quantile that covers a new point with a guaranteed probability. One quantile serves every atom: the
jackknife shape already carries the dependence of the error on the environment. A scale learned from the site descriptor
(normalised conformal prediction [Papadopoulos et al. 2008; Lei et al. 2018; Ho, Ortner and Wang 2026]) makes the coverage more even across environments
represented in the training data, but transfers less well to environments left out of it, so we report it as a refinement,
not as the method. The scores come from a stratified hold-out of the training set, and a fitted transfer exponent carries them to
the fit on all the data. The open-source ace-jax package [ace-jax] provides the shape, the hold-out scores and the transfer;
Sections 2.5 and 3.2 describe and test the scale.</p>
<p>Per-atom intervals cannot be combined into an interval on a derived property, a quantity of interest (QoI), because a QoI depends
on every atom and on relaxed geometries. A QoI's uncertainty has two parts [Simmons, Hudson, Swinburne and Kermode, in prep.]. The
aleatoric part is the sampling error of the simulation that computes the QoI, assessed with statistical convergence tests
[Grossfield and Zuckerman 2009; Lelièvre, Rousset and Stoltz 2010, Sec. 2.3]. The epistemic part comes from the uncertainty in the
fit. It is most widely estimated with a committee of \\(n\\) potentials, an idea long established in statistics [Hoeting et al.
1999]. If member \\(i\\) predicts \\(D_i\\sim\\mathcal N(\\mu_i,\\sigma_i^2)\\), independently, the predictive variance
\\(\\tfrac1n\\sum_i\\sigma_i^2+\\tfrac1{n-1}\\sum_i(\\mu_i-\\bar\\mu)^2\\) is the mean aleatoric variance plus the committee's spread of
means, the epistemic part. The QoIs here come from relaxations and static calculations, so their aleatoric part is zero and their
uncertainty is purely epistemic. For a linear model the consistent object is then a distribution over the coefficients θ: one draw
gives energies, forces and stresses together, and every QoI follows from it.</p>
<p>We propagate the same CR3 distribution, with one scale fitted to held-out forces, to 21 QoIs of silicon, for potentials fitted
to the GAP-18 database [Bartók et al. 2018]. The propagation uses implicit differentiation through every relaxation, including that
of the lattice constant, and is cross-checked by committees run through the unmodified benchmark protocol. We test the intervals
against DFT, one QoI at a time and jointly, and compare them with two baselines: the posterior with one conformal scale, and
pointwise optimal parameter sets (POPS) [Swinburne and Perez 2024].</p>
<p>Misspecification-aware parameter uncertainty has been propagated to material properties before. Perez et al. [2025] propagate
POPS hypercube ensembles of a quadratic SNAP potential for tungsten to phase, elastic, defect and surface properties, re-relaxing
each member, and use the implicit derivative of a relaxed minimum with respect to the parameters [Maliyov, Grigorev and Swinburne
2025] to predict volumes and a vacancy-migration barrier; the DFT values lie inside the ensemble's envelope without a calibration
step. We differ in three ways: the shape is the CR3 covariance with one scale calibrated on held-out data, so its size does not
depend on which training rows define a box; the intervals are tested against DFT one QoI at a time and jointly, by coverage, CRPS
and evidence; and the committees run through an unmodified benchmark protocol, beside implicit differentiation of every
relaxation.</p>
<p>We also show where calibrated parameter-space uncertainty fails, and why. A bias that all training configurations of one kind
share is invisible to any delete-one jackknife; in the GAP-18 database [Bartók et al. 2018] every defect cell is a hot molecular-dynamics snapshot. We
diagnose this bias, measure it from the training labels alone, and add it to the defect QoIs as a discrepancy term, which
leaves the fit unchanged.</p>
<p>Section 2 sets out the methodology: the linear ACE fit and ARD (2.1–2.3), the CR3 shape (2.4), the per-atom force calibration
(2.5), the baselines (2.6) and the propagation to QoIs (2.7). Section 3 gives the validation statistics and the results: the per-atom forces first, then the global scale that the
propagation uses, then the QoIs. Section 4 discusses them and Section 5 concludes.</p>""", "draft")

B2_NOTE = ("<b>[Draft: the B2 (POPS) committees are being rerun with the rebuilt boxes of Section 3.7; every B2 number "
           "in this version comes from the earlier boxes and will change.]</b>")
QOI_ROWS = (  # (key, name, unit, tolerance text): the benchmark's QoIs and tolerances (ace-jax bench/si_qoi/score.py)
    *[(k, n, "GPa", "10 %") for k, n in (("diamond_c11", "C<sub>11</sub>"), ("diamond_c12", "C<sub>12</sub>"),
                                         ("diamond_c44", "C<sub>44</sub>"), ("diamond_bulk_modulus", "bulk modulus B"))],
    *[(f"dE_{p} (meV/at)", f"E({lab}) − E(diamond)", "meV/atom", "10 meV/atom") for p, lab in
      (("hex_diamond", "hexagonal diamond"), ("bc8", "bc8"), ("st12", "st12"), ("beta-Sn", "β-Sn"), ("sh", "simple hexagonal"),
       ("hcp", "hcp"), ("bcc", "bcc"), ("fcc", "fcc"))],
    *[(k, n, "eV", "0.1 eV") for k, n in (("vacancy_energy", "vacancy formation"),
                                          ("tetrahedral_interstitial_energy", "tetrahedral interstitial formation"),
                                          ("hexagonal_interstitial_energy", "hexagonal interstitial formation"),
                                          ("dumbbell_interstitial_energy", "dumbbell interstitial formation"))],
    ("surface_100 (J/m2)", "(100) surface energy, relaxed", "J/m<sup>2</sup>", "5 %"),
    ("gamma_us glide (J/m2)", "unstable stacking fault, glide", "J/m<sup>2</sup>", "10 %"),
    ("gamma_us shuffle (J/m2)", "unstable stacking fault, shuffle", "J/m<sup>2</sup>", "10 %"),
    ("decohesion_max_stress", "maximum (100) decohesion stress, unrelaxed", "eV/Å<sup>3</sup>", "10 %"),
    ("phonon_f rms vs DFT (THz)", "phonon frequencies, RMS against DFT", "THz", "0.2 THz"))
_qv = {(r["model"], r["qoi"]): r for r in rd("qoi_uq.csv") if r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
_QDIG = {"eV": 3, "J/m<sup>2</sup>": 3, "eV/Å<sup>3</sup>": 3, "THz": 3}
def _qoi_table():
    rows = ""
    for k, name, unit, tol in QOI_ROWS:
        rs = [_qv.get((m, k)) for m in ("o3d12", "o4d20")]
        if not all(rs):
            continue
        d, phon = _QDIG.get(unit, 1), k.startswith("phonon")
        cells = ""
        for r in rs:
            e = float(r["mean"]) - float(r["dft"])
            err, pct = ("–", "–") if phon else (f"{e:+.{d}f}".replace("-", "−"), f"{100 * e / float(r['dft']):+.1f}".replace("-", "−"))
            cells += f"<td class=n>{float(r['mean']):.{d}f}</td><td class=n>{err}</td><td class=n>{pct}</td>"
        dft = "–" if phon else f"{float(rs[0]['dft']):.{d}f}"
        rows += f"<tr><td>{name}</td><td class=n>{dft}</td><td>{unit}</td><td class=n>{tol}</td>{cells}</tr>"
    return ("<div class=\"wrap\"><table><caption>Table 0. The silicon QoIs: DFT (CASTEP) reference value, unit, benchmark tolerance, "
            "and the value, error (model − DFT) and relative error (error / DFT) of the two ACE models. A tolerance in % is "
            "relative to the DFT value. The 10 meV/atom phase-energy tolerance is most of the small hexagonal-diamond value; "
            "the relative error is the more telling measure there. "
            "The phonon entry is the RMS difference of the phonon frequencies from DFT, so it has no DFT value or error and is not "
            "scored for coverage.</caption><thead><tr><th rowspan=2>QoI</th><th rowspan=2>DFT</th><th rowspan=2>unit</th>"
            "<th rowspan=2>tolerance</th><th colspan=3>ACE o3d12</th><th colspan=3>ACE o4d20</th></tr>"
            "<tr><th>value</th><th>error</th><th>error (%)</th><th>value</th><th>error</th><th>error (%)</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")
qoi_table = _qoi_table()
_par = {p_["size"]: p_ for p_ in json.load(open(RES / "ace_pareto.json"))} if (RES / "ace_pareto.json").exists() else {}
if {"o3d12", "o4d20", "o4d12", "o4d14"} <= set(_par):
    _a, _b = _par["o3d12"], _par["o4d20"]
    pareto_par = f"""<p><b>The two models.</b> We study two members of the linear ACE ladder of the ace-jax silicon accuracy–cost benchmark,
fitted to the GAP-18 database [Bartók et al. 2018] (Figure 0). A model oNdM has correlation order N and total polynomial degree M.
Table 0 lists the silicon QoIs, their DFT (CASTEP) values and the benchmark's tolerances: 10 % of the DFT value for the elastic
constants, the bulk modulus, the unstable stacking-fault energies and the decohesion stress; 5 % for the (100) surface energy;
10 meV/atom for the phase energies; 0.1 eV for the point-defect formation energies; and 0.2 THz for the phonon RMS. The QoI error
of a model is the geometric mean of |error|/tolerance over these QoIs. o3d12 ({_a['n_params']} coefficients) is the cheaper model, with a
test force RMSE of {_a['rmse_F']:.3f} eV/Å and a typical QoI error of one tolerance ({_a['qoi']:.2f}). o4d20 ({_b['n_params']} coefficients)
is the only size of the ladder that meets both of the benchmark's targets, a test force RMSE no larger than GAP-18's 0.0825 eV/Å
on this split ({_b['rmse_F']:.4f} eV/Å) and a QoI error below 0.5 ({_b['qoi']:.2f}); o3d12 is {_b['cost_cpu'] / _a['cost_cpu']:.0f}× cheaper on a
CPU node and {_b['cost_gpu'] / _a['cost_gpu']:.0f}× cheaper on an A100. The pair spans the useful range of the ladder: a cheap model
whose intervals must be wide to be honest, and an accurate one whose intervals should be narrow. The two columns of Figure 0 also rank the sizes differently: o4d12 has a larger force error than o4d14 but a smaller QoI
error ({_par['o4d12']['qoi']:.2f} against {_par['o4d14']['qoi']:.2f}). Force accuracy alone does not predict QoI accuracy, which is why the
uncertainty is propagated to the QoIs themselves.</p>
{qoi_table}
<figure><img src="fig0_ace_pareto.png" alt="Test force RMSE and QoI error against CPU cost for the linear ACE ladder, o3d12 and o4d20 circled"><figcaption>Figure 0.
The linear ACE ladder of the ace-jax silicon accuracy–cost benchmark: test force RMSE (left) and QoI error (right; the geometric
mean over the silicon QoIs of |error|/tolerance against CASTEP) against cost in LAMMPS, float64: on a Sulis CPU node with the
ace-jax C++ kernel (top) and on a Sulis A100 with the ace-jax CUDA kernel (bottom). Dashed: the benchmark's targets, GAP-18's force RMSE on this split and a QoI error of 0.5. Circled: the two models
studied here.</figcaption></figure>"""
else:
    pareto_par = ""

sec("theory-model", "2  Methodology", r"""
<h3>2.1 Linear ACE: model and observations</h3>
<p>Each atom \(i\), at position \(r_i\) with species \(Z_i\) and environment \(x_i\) (its neighbours within the cutoff), has a site descriptor \(\phi(x_i)\in\mathbb{R}^p\): the atomic cluster expansion (ACE) product basis [Drautz 2019], in
species-major blocks, then pair blocks. The basis, its radial functions and the smoothness prior below are those of
ACEpotentials [Witt et al. 2023]. The energy is linear in the coefficients \(\theta\),</p>
\[ E(\{r_i\};\theta)=\sum_i \phi(x_i)\cdot\theta+\sum_i E^0_{Z_i}, \]
<p>and so is every observation of a configuration:</p>
\[ b_E=\sum_i\phi(x_i),\qquad B_{F,i\alpha}=-\frac{\partial b_E}{\partial r_{i\alpha}},\qquad
   B_{V,\alpha\beta}=-\frac{\partial b_E}{\partial\varepsilon_{\alpha\beta}}\Big|_{\varepsilon=0}, \]
<p>with \(E^0_Z\) a per-species reference energy, \(r_{i\alpha}\) the Cartesian components of \(r_i\), and \(\varepsilon\) a symmetric strain
applied to positions and cell. The predictions are \(b_E\cdot\theta\), \(B_F\theta\) and the virial \(W=B_V\theta\) (in eV; the stress is
\(-W/\Omega\), with \(\Omega\) the cell volume). Each row of \(b_E\), \(B_F\) or \(B_V\) is called an observable row \(b\).
Each row has a weight \(w\), which combines a factor \(1/\sqrt N\) on the energy and virial rows of an \(N\)-atom configuration, a weight per quantity
(energy, force, virial) and a weight per configuration type, and all rows share one noise scale \(\sigma\). The weights set the
balance between energies, forces and virials, and \(\sigma\) sets the overall noise level. The whitened rows and targets are
\(\psi=b\,w/\sigma\) and \(\tilde y=y\,w/\sigma\), with \(y\) the DFT label minus the \(E^0\) baseline. Stacked over all configurations, the observable rows \(b\) form the design matrix \(\Phi\), and the
\(\psi\) its whitened form \(\Psi\). With least-squares \(E^0\), one column per species carries the atom count on the
energy rows (a fixed, wide prior).</p>
""" + pareto_par + r"""
<h3>2.2 Prior, posterior and evidence</h3>
<p>The prior is \(\theta\sim\mathcal N(0,\Lambda^{-1})\) with the smoothness prior \(\Lambda=\operatorname{diag}(\Gamma_j^2)/\sigma_c^2\): \(\sigma_c\) is the prior scale of the coefficients
and \(\Gamma_j=\sum_t\big[(l_t/w_l)^4+(n_t/w_n)^4\big]\), summed over the one-particle factors \(t\) of basis function \(j\) with radial
index \(n_t\) and angular degree \(l_t\) (the algebraic prior of ACEpotentials [Witt et al. 2023], \(w_l=2/3\), \(w_n=1\)), so rough basis functions are damped.
The posterior is Gaussian,</p>
\[ A=\sum_{\text{rows}}\psi\psi^{\mathsf T}+\Lambda,\qquad \hat\theta=A^{-1}\sum_{\text{rows}}\psi\tilde y,\qquad
   \operatorname{Cov}(\theta\mid\mathcal D)=A^{-1}, \]
<p>and the hyperparameters \(h=(\sigma_c,\sigma)\) maximise the evidence</p>
\[ \log p(\mathcal D\mid h)=-\frac{\lVert y^w\rVert^2}{2\sigma^2}+\tfrac12 c^{\mathsf T}A^{-1}c-\tfrac12\log|A|
   +\tfrac12\log|\Lambda|-n\log\sigma , \]
<p>up to a constant, with \(\mathcal D\) the training data, \(y^w\) the weighted targets (\(y\,w\)), \(n\) the number of rows,
and \(c=\sum_{\text{rows}}\psi\tilde y=A\hat\theta\).</p>
<p>Near-interpolating fits leave the condition number \(\operatorname{cond}(A)\) beyond \(1/\epsilon_{\text{mach}}\) (\(\epsilon_{\text{mach}}\)
the float64 unit roundoff), so the evidence and the posterior are computed by QR. \(R_w\) is the triangular factor of the weighted
design (\(\Phi^w=Q_wR_w\), rows \(b\,w\)), accumulated by streamed updating QRs, and
\([\operatorname{diag}\sqrt\Lambda;\,R_w/\sigma]=Q_AR_A\) gives \(R_A^{\mathsf T}R_A=A\) at \(\operatorname{cond}(\Phi)\) instead of
\(\operatorname{cond}(\Phi)^2\).</p>
""", "drafted from docs")

ards = json.load(open(RES / "ard_summary.json")) if (RES / "ard_summary.json").exists() else {}
ard_orders = sorted({n_.replace("a_", "").replace("body", "") for d_ in ards.values() for n_ in d_["h_names"]}, key=int)
ard_rows = ""
for m in ("o3d12", "o4d20"):
    if m not in ards:
        continue
    d_ = ards[m]
    ak = {n_.replace("a_", "").replace("body", ""): (h_, ab) for n_, h_, ab in zip(d_["h_names"], d_["h"], d_["at_bound"])}
    a_cells = "".join(f"<td class=n>{ak[o][0]:.1f}{'*' if ak[o][1] else ''}</td>" if o in ak else "<td class=n>–</td>" for o in ard_orders)
    ta, tb = d_["test_rmse"]["ard"], d_["test_rmse"]["blr"]
    ca, cb = d_["coverage"]["ard"].get("CR3-headline/lam_F", {}), d_["coverage"]["blr"].get("CR3-headline/lam_F", {})
    cv = lambda c: f"{int(c.get('k95', 0))} ({int(c.get('k68', 0))})"
    ard_rows += (f"<tr><td>{m}</td>{a_cells}<td class=n>{d_['a_floor']:.1f}</td><td class=n>{d_['a_blr']:.1f}</td>"
                 f"<td class=n>{ta['E_meV_atom']:.2f}</td><td class=n>{tb['E_meV_atom']:.2f}</td>"
                 f"<td class=n>{ta['F_eV_A']:.4f}</td><td class=n>{tb['F_eV_A']:.4f}</td>"
                 f"<td class=n>{ta['V_meV_atom']:.1f}</td><td class=n>{tb['V_meV_atom']:.1f}</td>"
                 f"<td class=n>{d_['lam_cr3']['ard']['F']:.2f}</td><td class=n>{d_['lam_cr3']['blr']['F']:.2f}</td>"
                 f"<td class=n>{cv(ca)}</td><td class=n>{cv(cb)}</td></tr>")
ard_head = ("<tr><th rowspan=2>model</th>" + f"<th colspan={len(ard_orders)}>ARD log-precision a<sub>k</sub> (body order k)</th>"
            "<th rowspan=2>ARD floor</th><th rowspan=2>BLR a</th><th colspan=2>E RMSE (meV/atom)</th><th colspan=2>F RMSE (eV/Å)</th>"
            "<th colspan=2>V RMSE (meV/atom)</th><th colspan=2>test-split λ<sub>F</sub></th><th colspan=2>coverage, 95 % (68 %)</th></tr>"
            "<tr>" + "".join(f"<th>{o}</th>" for o in ard_orders) + "<th>ARD</th><th>BLR</th>" * 5 + "</tr>")
sec("theory-ard", "2.3  Automatic relevance determination", rf"""
<p>ARD replaces the single \(\sigma_c\) by one log-precision per body order, \(\Lambda=\operatorname{{diag}}(\Gamma_j^2e^{{a_{{k(j)}}}})\), with \(k(j)\) the body order of basis function \(j\), fitted by
type-II maximum likelihood in the prior-scaled system \(S=D^{{-1}}AD^{{-1}}\), \(D=\operatorname{{diag}}\Gamma\). The BLR fit is ARD at
\(a_k=-2\log\sigma_c\) for every \(k\). In sequential mode, used here, the noise scale \(\sigma\) stays at the BLR fit's value and only the
\(a_k\) are fitted.</p>
<p><b>QR evidence.</b> As for the BLR fit (Section 2.2), the evidence, its gradient and Hessian, and the posterior are computed
from the QR factor of
\([\operatorname{{diag}}(e^{{a_{{k(j)}}/2}});\,R_wD^{{-1}}/\sigma]=Q_SR_S\) (so \(R_S^{{\mathsf T}}R_S=S\)), whose condition number is
\(\sqrt{{\operatorname{{cond}}(S)}}\). The posterior stores \(R_S^{{\mathsf T}}\) as its Cholesky factor, so the CR3 jackknife and the per-atom force variances are unchanged. The \(a_k\) are bounded below at
\(a_{{\text{{floor}}}}=\log(\lambda_{{\max}}/10^{{22}})\), with \(\lambda_{{\max}}\) the largest eigenvalue of the data part of \(S\)
(\(D^{{-1}}\Psi^{{\mathsf T}}\Psi D^{{-1}}\)) at the starting noise scale, and an endpoint with every \(a_k\) on the floor and the gradient pointing below it
is flagged as degenerate and refused.</p>
<div class="wrap"><table><caption>Table 1. ARD against the benchmark BLR fits. ARD log-precision per body order at the endpoint
(* on the floor), the floor, and the single log-precision of the BLR prior (−2 log σ<sub>c</sub>); test-split RMSE (E and V per atom);
test-split λ<sub>F</sub>; and the propagated CR3 × λ<sub>F</sub> coverage of the 20 scored QoIs.</caption>
<thead>{ard_head}</thead>
<tbody>{ard_rows}</tbody></table></div>
<p>Neither fit is degenerate. For o3d12 only the two-body precision sits on the floor, and for o4d20 every \(a_k\) is
interior. ARD weakens the prior relative to the BLR's single \(\sigma_c\) on most body orders (on all of them for o3d12; for
o4d20 only the 5-body block is held more tightly), yet the means match the BLR fits on the held-out test split, and the test-split
scales and the propagated coverage agree. The QoI results below use the BLR fits, which are the
benchmark models; ARD gives the same coverage (Table 1), and it provides the per-atom force calibration of
Section 3.2.</p>""", "drafted", level=3)

sec("theory-cr3", "2.4  Misspecification-robust shape: the CR3 jackknife", r"""
<p><b>POPS.</b> The closest existing approach for linear potentials is POPS [Swinburne and Perez 2024]. For each training row
\(i\), with residual \(r_i\) and leverage \(h_i=\psi_i^{\mathsf T}A^{-1}\psi_i\), the pointwise optimal parameter set is
\(\hat\theta+\delta_i\), with \(\delta_i=A^{-1}\psi_i\,r_i/h_i\) the smallest change (in the \(A\) metric) that makes the model
reproduce that row exactly. The spread of the \(\delta_i\), as an ensemble or as a hypercube in their principal directions, is
the uncertainty. It does not shrink as data are added, because each \(\delta_i\) stays finite while the model is misspecified.</p>
<p><b>Comparison.</b> CR3 below is built from the same per-row influence, \(A^{-1}\psi_i r_i\), but asks the opposite question:
how far the fit moves when a configuration is removed, not how far it must move to fit a row exactly. For one row the two differ
only in the leverage factor, \(1/(1-h_i)\) for deletion against \(1/h_i\) for an exact fit. CR3 is therefore largest for rows the
fit leans on, and POPS for rows the fit hardly sees, whose corrections are unbounded as \(h_i\to0\) (Section 3.7). CR3 also deletes
whole configurations, so that the correlated energy, force and virial rows of one configuration move together, and it is a
covariance that is calibrated by one scale on held-out data (Section 3.3); POPS has no calibration step. Like any sampling variance,
the CR3 covariance shrinks as data are added, which is why the scale is needed: CR3 supplies the shape, and the held-out data the
size.</p>
<figure><img src="fig9_cr3_schematic.png" alt="Schematic of the CR3 construction on the regression demo's Lennard-Jones example: a degree-5 polynomial fit with the CR3 and posterior bands; the leave-one-point-out changes of the fit in the leading principal curves of the CR3 covariance; and a six-step flow of the computation"><figcaption>Figure 9.
The CR3 construction on the Lennard-Jones example of the interactive regression demo [kermodegroup demos; scene
lj-cr3], computed exactly. (a) A degree-5 polynomial, fitted by Bayesian ridge regression with evidence-optimised
hyperparameters to 197 nearly noiseless samples (noise standard deviation 0.011, as for DFT labels) of the smooth part of a
Lennard-Jones curve, cannot follow the steep repulsive side and the well: the error is approximation error, not noise. The
posterior band \(\sigma^2A^{-1}\) (orange) is barely visible. The CR3 band (blue), scaled by λ from 50 held-out calibration
points, is wide where the fit is wrong: on the steep side near x = 0 and at the right-hand end. (b) Each dot is the change of
the fitted curve when one training point is left out, \(-A^{-1}\tilde g_k\), in the two leading principal curves of the CR3
covariance of the prediction; the closed form reproduces the refit to the precision the basis allows. The points on the steep
side and in the well (x &lt; 1) move the fit most. The CR3 covariance is the sum of these shifts' outer products over the 197
training points, so its ellipse is about √197 times the spread of single dots; the posterior ellipse is smaller. (c) The
computation for the real models, with the reading of each quantity: one fit, then per configuration its leverage, its
leave-out residuals and its pull, and finally the spread of the pulls.</figcaption></figure>
<p><b>CR3.</b> For a cluster \(k=1,\dots,n_{\text{cl}}\) of rows (here: one whole configuration with its energy, force and virial rows) with
whitened rows \(\Psi_k\) (the cluster's rows \(\psi^{\mathsf T}\) stacked), whitened targets \(\tilde y_k\) and residuals \(\rho_k=\tilde y_k-\Psi_k\hat\theta\), the cluster hat block and the leverage-corrected (PRESS)
score are</p>
\[ H_{kk}=\Psi_kA^{-1}\Psi_k^{\mathsf T}=W_k^{\mathsf T}W_k,\quad W_k=R_S^{-\mathsf T}D^{-1}\Psi_k^{\mathsf T},\qquad
   \tilde g_k=\Psi_k^{\mathsf T}(I-H_{kk})^{-1}\rho_k . \]
<p>Each object has a direct reading.</p>
<ul>
<li>\(H_{kk}\) is the cluster's block of the hat matrix: it maps the cluster's targets to its own fitted values, so it measures how
far the fit is pulled towards that configuration. Because \(\Lambda\succ0\), its eigenvalues lie in \([0,1)\). An eigenvalue near 0
means the configuration is predicted by the rest of the data; one near 1 means the fit nearly interpolates it along that direction,
and its residual there underestimates the error it would have if it were left out.</li>
<li>\(W_k\) is the cluster's rows expressed in whitened coefficient coordinates, scaled by the prior (\(D\)) and by the Cholesky
factor of \(S\) (Section 2.3). In these coordinates the posterior is isotropic, so \(H_{kk}=W_k^{\mathsf T}W_k\) is a plain Gram
matrix; computing \(W_k\) costs one triangular solve per cluster.</li>
<li>\(\tilde g_k\) is the cluster's contribution to the gradient of the least-squares objective, \(\Psi_k^{\mathsf T}\rho_k\),
evaluated with its leave-cluster-out (PRESS) residuals \((I-H_{kk})^{-1}\rho_k\) instead of its ordinary residuals. It is the pull of
configuration \(k\) on the coefficients as an unseen configuration would exert it.</li>
</ul>

<p>With the hyperparameters fixed, deleting cluster \(k\) moves the mean by exactly \(\hat\theta-\hat\theta_{(-k)}=A^{-1}\tilde g_k\)
(the cluster DFBETA [Cook and Weisberg 1982, Sec. 3.6]). The spread of these shifts over clusters is the CR3 covariance. The
centred meat and the
delete-one-cluster (CR3) covariance are</p>
\[ \tilde M=\sum_k(\tilde g_k-\bar g)(\tilde g_k-\bar g)^{\mathsf T},\qquad
   \Sigma_{\text{CR3}}=A^{-1}\tilde MA^{-1}=D^{-1}\tilde Q\tilde Q^{\mathsf T}D^{-1},\quad
   \tilde Q=S^{-1}D^{-1}[\tilde g_1-\bar g,\dots]. \]
<p>with \(\bar g=n_{\text{cl}}^{-1}\sum_k\tilde g_k\). Centring removes the prior's gradient (\(\sum_k\Psi_k^{\mathsf T}\rho_k=\Lambda\hat\theta\) at the MAP).
\(\Sigma_{\text{CR3}}\) is a Huber–White sandwich; with one residual variance \(s^2\) for every row it reduces to \(s^2A^{-1}\), the
homoscedastic special case. It is stored as a factor
\(R\) with \(RR^{\mathsf T}=\tilde Q\tilde Q^{\mathsf T}\) (rank ≤ number of clusters), so that</p>
\[ \Sigma=LL^{\mathsf T},\qquad L=D^{-1}R\in\mathbb R^{p\times r}, \]
<p>with \(r\) the rank. For any observable row \(b\), \(\sigma(b)=\lambda\,\lVert L^{\mathsf T}b\rVert\), where \(\lambda\) is a scale
calibrated on held-out data (Section 3.3). The CR3 covariance measures how much a
prediction depends on which configurations were in the fit: large where the fit leans on configurations it reproduces
poorly. It is a parameter variance used as a proxy for approximation error, which is why it needs a scale.</p>
<h3>2.5 Calibrated per-atom forces: the CR3 shape and one conformal quantile</h3>
<p>This section describes the per-atom force uncertainty introduced in this work. Every predicted force gets
a calibrated uncertainty: a shape times one scale. The shape is the per-atom 3×3 block of the CR3
covariance, \(V(x)_{\alpha\beta}=u_\alpha^{\mathsf T}RR^{\mathsf T}u_\beta\) with \(u_\alpha=D^{-1}B_{F,i\alpha}^{\mathsf T}\) built
from the atom's three force rows (\(\alpha\in\{x,y,z\}\)), and \(v=\operatorname{tr}V\). The scale is one conformal quantile
of scores on held-out training configurations.</p>
<p><b>Hold-out.</b> Training configurations are split 80/20 into \(T_{\text{fit}}\) and \(T_{\text{val}}\), stratified so that
rare, strongly distorted or off-coordinated environments appear in both parts (the ace-jax default). The posterior
\(P_{\text{fit}}\) on \(T_{\text{fit}}\), with its own CR3 shape \(V_{\text{fit}}\), predicts \(T_{\text{val}}\), so both the force
error \(e_i=F_i^{\text{DFT}}-F_i^{\text{fit}}\in\mathbb R^3\) and the shape are out of sample. The anisotropic score is</p>
\[ s_i=\sqrt{e_i^{\mathsf T}\big(V_{\text{fit}}(x_i)+\epsilon_i I\big)^{-1}e_i},\qquad \epsilon_i=10^{-3}\,\tfrac13\operatorname{tr}V_{\text{fit}}(x_i), \]
<p>and \(s/\lambda\sim\chi_3\) (the chi distribution with three degrees of freedom) under \(e\sim\mathcal N(0,\lambda^2V)\).</p>
<p><b>Transfer to the full training set.</b> The final posterior is fitted on all of \(T\), about 25 % more data than
\(P_{\text{fit}}\). A second hold-out fit \(P_{\text{fit2}}\) on a subset of \(T_{\text{fit}}\) scores the same
\(T_{\text{val}}\) atoms. With \(N\), \(N_{\text{fit}}\), \(N_{\text{fit2}}\) the numbers of
configurations behind the final, first and second fits, and \(\lambda_1,\lambda_2\) the RMS scores of \(P_{\text{fit}}\) and \(P_{\text{fit2}}\),
\(\beta=\operatorname{clip}\big(\log(\lambda_1/\lambda_2)/\log(N_{\text{fit}}/N_{\text{fit2}}),0,\tfrac12\big)\) and every score is
multiplied by \(f_{\text{tr}}=(N/N_{\text{fit}})^\beta\). \(\beta=0\) is the variance-dominated limit and \(\tfrac12\) the bias-dominated one.</p>
<p><b>The scale.</b> The conformal quantile \(q\) is the \(\lceil(n+1)(1-\alpha)\rceil/n\) empirical quantile of the \(n\) scores
\(s_i\) of the \(T_{\text{val}}\) atoms, at level \(1-\alpha\) (default \(\alpha=0.1\)). It is atom-weighted: a quantile that counts
each configuration once in total is dominated by the many small, well-predicted cells and under-covers (Section 3.2).</p>
<p><b>Per-atom outputs.</b> The conformal region for a force error \(\Delta F\) is
\(\{\Delta F:\Delta F^{\mathsf T}(V+\epsilon I)^{-1}\Delta F\le q^2\}\), the CR3 ellipsoid scaled to the quantile. Under
exchangeability of the calibration and target configurations it covers a new atom with probability \(\approx1-\alpha\). The
Gaussian reading is the force standard deviation \(\sigma_{F,i}=(q/\chi_3^{-1}(1-\alpha))\sqrt v\), with \(\chi_3^{-1}\) the
\(\chi_3\) quantile function, and covariance \(\sigma_{F,i}^2V/v\).</p>
<p><b>A learned scale, as a refinement.</b> The CR3 shape is a quadratic form in the atom's force rows, so it already grows with
the distance of the environment from the training descriptors. In that it is related to the D-optimality extrapolation grade
used to select training data for linear potentials [Podryabinkin and Shapeev 2017; Lysogorskiy et al. 2023], a leverage of the
atom's descriptor against the training set; the CR3 shape weights that leverage by the residuals the fit leaves, configuration by
configuration, and the quantile calibrates it to a coverage. A scale that depends on the environment should therefore model the
ratio of the error to the shape, the score \(s\), not novelty itself. We test one learned from the site descriptor (normalised
conformal prediction [Papadopoulos et al. 2008; Lei et al. 2018]). Let \(\phi_i\) be atom \(i\)'s whitened principal components
of the normalised site descriptor, from a PCA of training atoms only. On one half of the \(T_{\text{val}}\) configurations a ridge
regression \(f(\phi)=\beta_0+\beta^{\mathsf T}\phi+\gamma^{\mathsf T}\phi^{2}\) of \(\log s_i^2\) is fitted (each configuration
weighted once, the ridge by five-fold cross-validation over configurations) and clipped to the range it takes on the
calibration atoms. On the other half the normalised scores \(s_i\,e^{-f(\phi_i)/2}\) give one quantile \(q_f\), and the region is
\(s\le q_f\,e^{f(\phi)/2}\). The split is repeated over five seeds. Ho, Ortner and Wang [2026] learn an
environment-dependent scale in the same spirit: a neural network of MACE descriptors that multiplies a last-layer uncertainty,
fitted with a weighted absolute-error loss rather than as a conformal quantile.</p>
<p><b>Why the propagation does not use this scale.</b> The conformal quantile tempers force <em>outputs</em> atom by
atom. It does not define a covariance in θ, and a QoI depends on every atom and on relaxed geometries. Propagation therefore uses one
global λ (Section 3.3). Section 3.2 tests the per-atom force uncertainties directly. Energy and virial uncertainties use the unscaled posterior
variance.</p>
<h3>2.6 Baselines</h3>
<p>We compare the CR3 propagation with three baselines. Each changes one ingredient, the shape of the covariance or the way its
size is set, so that the comparison shows which ingredient matters.</p>
<p>The first baseline, B1, is the approach of Best et al. [2024]. It keeps the Bayesian posterior as the shape, \(\Sigma=A^{-1}\),
and corrects its size with one conformal quantile \(\hat q\). The quantile is taken from the absolute standardised scores
\(|z|\) of the held-out test split (Section 3.3, here computed with the posterior shape), pooled over energies, forces and virials,
with every configuration given equal weight. B1 therefore uses the same held-out data, and the same kind of single scale, as our
method. It differs only in the shape, so it tests whether the CR3 covariance describes the errors better than the posterior.</p>
<p>The second baseline, B2, is POPS [Swinburne and Perez 2024], as implemented in ace-jax [ace-jax] by a port of the reference
package POPSRegression [Swinburne, POPSRegression]. For each training row \(i\) the correction is
\(\delta_i=A_\mu^{-1}\psi^w_i\,r_i/h_i\), with leverage \(h_i=\psi_i^{w\mathsf T}A_\mu^{-1}\psi^w_i\). Here \(\psi^w_i=b_i w_i\) carries the
row weights but no noise scale, \(r_i\) is the row's residual at the ridge solution, and \(A_\mu=M+\mu\operatorname{diag}(\Gamma_j^2)\),
with \(M=\sum_i\psi^w_i\psi_i^{w\mathsf T}\) and the ridge parameter \(\mu\) chosen on a path. Members are drawn uniformly in the
hypercube spanned by the principal directions of the \(\delta_i\), with bounds from their extreme projections, as in Swinburne and
Perez [2024]. Following the reference package, the box is built only from the rows at or above the median leverage
(<code>leverage_percentile = 50</code>, its default; every row makes the box unphysical here, Section 3.7). The ridge is selected
per quantity by the CRPS on a hold-out of the training set, over relative ridges from 10<sup>−2</sup> down to 10<sup>−10</sup> of
the largest eigenvalue: below that the box is set by roundoff (Section 3.7). We use ace-jax [ace-jax] version 0.5.2, which includes the
numerical corrections that this study prompted. B2 has no calibration step, so it tests a misspecification-aware shape without a
fitted scale.</p>
<p>The third baseline, B3, is the CR3 covariance with \(\lambda=1\). It isolates the effect of the scale: the difference between B3
and the headline is entirely the calibration of Section 3.3.</p>
<p>Fisher et al. [2026] take the conformal step further. They calibrate joint conformal sets for energies, forces and virials
(Bonferroni-corrected boxes, Mahalanobis ellipsoids, and losses under conformal risk control) and propagate them to derived
quantities such as elastic constants and vacancy formation energies, where modelling the correlations captures error
cancellation. B1 and our CR3 method use one scalar scale instead. We do not use multivariate conformal prediction here; combining
it with the CR3 shape is left to future work.</p>""", "drafted from docs", level=3)

sec("method-calib", "3.3  One global scale for propagation", r"""
<p>The per-atom scale of Section 3.2 calibrates force outputs atom by atom, but it does not define a covariance in θ
(Section 2.5). Propagation needs a single scale \(\lambda\) for the CR3 distribution \(\theta\sim\mathcal N(\hat\theta,\lambda^2LL^{\mathsf T})\).
We fit it on the same 248 held-out test configurations as Section 3.2, with the same CR3 shape, but on the BLR fits that the QoI
benchmark uses and over all rows of a type at once. Every row \(b\) gives a standardised score
\(z=(y-\hat y)/\lVert L^{\mathsf T}b\rVert\), with \(\hat y=b\cdot\hat\theta\) (energies per atom, force components, virial components). The
scale for observable type \(q\in\{E,F,V\}\) is \(\lambda_q=\operatorname{RMS}(z_q)\) over that type's rows; \(\lambda_{\text{pool}}\) weights E, F
and V equally within a configuration and configurations equally. As in Section 3.2, rows that vanish by site symmetry
(\(\lVert L^{\mathsf T}b\rVert\lt 10^{-3}\) of the median of their kind) are dropped. No QoI, and no DFT QoI value, enters any scale.
The headline uses \(\lambda_F\): forces are most of the training rows, and they are the quantity calibrated per atom in Section 3.2.
In Figure 2, \(\lambda_F\) gives the headline line.</p>
<p>A single scale is a compromise. Figure 5 breaks the test-split scale down by configuration type: it ranges from about 1 to 15
across types and observables, largest for the virials of the bt cells, the vacancy and interstitial cells and, for o3d12, the
hexagonal-diamond forces, and smallest for the energies of the crystal phases. The ratios of the global scales (Table 2) show the same at the level of observables.</p>
<figure><img src="fig5_scale_diagnostics.png" alt="Test-split scale per observable and configuration type, both models"><figcaption>Figure 5.
Test-split scale λ = RMS z with the CR3 shape, per observable (energy, force, virial) and GAP-18 configuration type, for both models
("all": the global λ<sub>E</sub>, λ<sub>F</sub>, λ<sub>V</sub> of Table 2). Missing bars: types without that observable.</figcaption></figure>
<div class="wrap"><table><caption>Table 2. Test-split scales (CR3 shape).</caption>
<thead><tr><th>model</th><th>λ<sub>E</sub></th><th>λ<sub>F</sub></th><th>λ<sub>V</sub></th><th>λ<sub>pool</sub></th><th>λ<sub>V</sub>/λ<sub>F</sub></th><th>λ<sub>E</sub>/λ<sub>F</sub></th></tr></thead>
<tbody>""" + lam_rows + r"""</tbody></table></div>
<p>Whether the observables need different scales is a model comparison. The test configurations are the independent
units, as in CR3, because the force components of one configuration are correlated. For configuration c and observable o, let
\(m_{co}\) be the mean of \(z^2\) over the block's components (1 energy, 3N forces, 6 virial components). We model
\(m_{co}=\lambda_o^2X\) with \(X\sim\chi^2_{\nu_o}/\nu_o\), a Gamma distribution with mean \(\lambda_o^2\), as the RMS-z
definition of λ requires. The effective number of degrees of freedom \(\nu_o\) absorbs the correlation within a block and the
heavy tails. It is free for each observable in every model, log-uniform on [0.05, 1000], and integrated out. The models are
the five ways to let E, F and V share scales, each scale log-uniform on [0.1, 1000]. A joint Gaussian likelihood of each block
under λ² times the full CR3 block covariance was also tried and is not usable: for forces it is set by the regularisation of a
nearly singular block, that is by the off-diagonal CR3 shape, not by the scale.</p>
<div class="wrap"><table><caption>Table 2b. Test-split evidence for shared against per-observable scales: log Bayes factor of
each way of sharing scales against the best (configurations as units, equal prior odds).</caption>
<thead><tr><th>scales</th><th>o3d12</th><th>o4d20</th></tr></thead><tbody>""" + sev_rows + r"""</tbody></table></div>
<p>""" + (f"""In both models the evidence favours one scale for energies and forces and a separate, larger one for virials:
log Bayes factors of {sev_bf("o3d12")} (o3d12) and {sev_bf("o4d20")} (o4d20) against one scale for all, and
{sev_sep("o3d12")} and {sev_sep("o4d20")} against three separate scales. The posterior scales are {sev_lam("o3d12")} for
o3d12 and {sev_lam("o4d20")} for o4d20 (mean and 95 % credible interval). With each configuration weighted equally,
λ<sub>E</sub> and λ<sub>F</sub> agree; the smaller λ<sub>E</sub> of Table 2 comes from the component-weighted RMS, in which
the largest cells dominate λ<sub>F</sub>.""" if sev else "") + r""" The degree of misspecification therefore depends on the
observable, and with λ<sub>F</sub> virials are under-covered on the test split. The QoIs most like virials are the elastic
constants, which are derivatives of the stress, and the phase energies, which come from energy–volume curves, so their
intervals at λ<sub>F</sub> may be too narrow; Section 3.4 reports the coverage with λ<sub>V</sub> as a sensitivity.</p>""", "drafted", level=3)

sec("method-prop", "2.7  Propagation to QoIs", r"""
<p>The distribution propagated is \(\theta\sim\mathcal N(\hat\theta,\lambda^2LL^{\mathsf T})\); one sample gives energy, forces and
stress consistently. Per-atom force ellipsoids are never combined.</p>
<h4>2.7.1 The QoIs</h4>
<p>The 21 QoIs are those of the silicon test suite of Bartók et al. [2018], with their DFT (CASTEP) reference values, run with
the tolerances of the ace-jax silicon benchmark (a Python 3 port of that test framework):
C<sub>11</sub>, C<sub>12</sub>, C<sub>44</sub>, B; eight phase energies relative to diamond (compressed hcp excluded as extrapolation);
vacancy and three interstitial formation energies (216-atom cells); the relaxed (100) surface energy; the maximum unrelaxed
decohesion stress; the glide and shuffle unstable stacking-fault energies; the phonon RMS against DFT. Every protocol starts
from the model's own relaxed diamond cell, so no QoI is linear in θ. The phonon RMS is a distance to DFT and has no DFT
value: it is left out of coverage (n = 20).</p>
<h4>2.7.2 Delta method by implicit differentiation</h4>
<p>Implicit differentiation of a relaxed minimum with respect to the parameters of an interatomic potential is the method of
Maliyov, Grigorev and Swinburne [2025]; here it is applied to every relaxation of the benchmark, the lattice constant included.
Write the energy of a structure with coordinates \(x\) (positions, and cell where relaxed) as \(E=b_E(x)\cdot\theta+E^0(x)\), with
\(b_E(x)\) its energy row (Section 2.1) and \(E^0(x)\) the reference energies. A QoI \(Q(\theta)\) is computed at a relaxed geometry
\(x^*(\theta)\) solving \(\nabla_xE=0\), so</p>
\[ \frac{dx^*}{d\theta}=-\mathcal H^{-1}\frac{\partial\nabla_xE}{\partial\theta}=-\mathcal H^{-1}\Big(\frac{\partial b_E}{\partial x}\Big)^{\mathsf T},\qquad
   \frac{dQ}{d\theta}=\frac{\partial Q}{\partial\theta}+\frac{\partial Q}{\partial x}\frac{dx^*}{d\theta}, \]
<p>with \(\mathcal H=\nabla_x^2E\) the Hessian at \(x^*(\hat\theta)\). Then \(\sigma_Q=\lambda\lVert L^{\mathsf T}G\rVert\), with \(G=dQ/d\theta\)
the QoI's row.</p>
<ul>
<li><b>Lattice constant.</b> The diamond lattice constant \(a_0\) makes the diamond-cell energy \(E_{\text{dia}}(a)\) stationary:
\(da_0/d\theta=-(\partial b_{E,\text{dia}}/\partial a)/E_{aa}\), with \(E_{aa}=d^2E_{\text{dia}}/da^2\) at \(a_0\).</li>
<li><b>Energies at a minimum</b> (phase, defect, surface energies). By the envelope theorem the relaxed energy \(E^*\) has \(dE^*/d\theta=b_E(x^*)\) for every
relaxed coordinate, plus the response of the cell the protocol builds from \(a_0\):
\(dE/da_0\cdot da_0/d\theta\), \(dE/da_0=\theta\cdot\partial b_E/\partial\ln a\,/a_0=-\operatorname{tr}W/a_0\) (\(W\) the virial of the
relaxed structure). The relaxed
geometries are the harness's own, recorded for the mean model. Example, vacancy:
\(G=b_{\text{vac}}+(\theta\cdot\partial_{\ln a}b_{\text{vac}}/a_0)\,da_0/d\theta-\tfrac{N-1}{N}b_{\text{bulk}}\), with \(b_{\text{vac}}\) and
\(b_{\text{bulk}}\) the energy rows of the relaxed vacancy cell and the \(N\)-atom bulk cell (the bulk cell has zero virial at \(a_0\), so it
has no \(a_0\) term).</li>
<li><b>Elastic constants, relaxed-ion.</b> At each applied strain the internal coordinates \(u\) (atomic positions in the strained
cell) are relaxed, and the stress \(\tau=-W/\Omega\) is linear in θ at fixed geometry, \(\tau=B_\tau\theta\) with \(B_\tau=-B_V/\Omega\).
Then \(d\tau/d\theta=B_\tau+(\partial\tau/\partial u)\,du^*/d\theta+(\partial\tau/\partial a)\,da_0/d\theta\) with
\(du^*/d\theta=-\mathcal H_{uu}^{-1}[(\partial b_E/\partial u)^{\mathsf T}+\partial_a\nabla_uE\otimes da_0/d\theta]\), \(\mathcal H_{uu}\) the
Hessian in \(u\). The elastic constants are linear fits of \(\tau\) against strain, so their rows are the same fits of these. Stress route: the harness's
matscipy protocol (Voigt strain pattern e<sub>1</sub>+e<sub>4</sub>, 5 steps of ±1 %, linear fits). Energy route: curvature at the same steps
of patterns e<sub>1</sub>, e<sub>4</sub>, e<sub>1</sub>+e<sub>2</sub>.</li>
<li><b>Decohesion.</b> Rigid openings of the relaxed bulk: linear in θ at fixed \(a_0\) plus the \(a_0\) term; the cubic-spline
derivative is linear in the energies.</li>
<li><b>Discrete steps</b> (E(V)-grid argmin, maximum over a grid) are held at their index; γ<sub>us</sub> and phonons are
committee-only.</li>
</ul>
<p>Rows are checked against finite differences in θ (to ~10<sup>−5</sup>) and the values they imply reproduce the harness to
~10<sup>−10</sup> (energy QoIs) and 0.04 GPa (elastic constants).</p>
<h4>2.7.3 Committees</h4>
<p>K = 200 draws \(\theta_k=\hat\theta+L\xi_k\), \(\xi_k\sim\mathcal N(0,I_r)\) (fixed seeds), are run through the unmodified benchmark harness. Per the protocol,
the draws are made once at λ = 1 and QoI deviations are rescaled, \(Q_k(\lambda)=\hat Q+\lambda(Q_k-\hat Q)\), with \(Q_k\) member \(k\)'s QoI and
\(\hat Q\) the mean model's; a full
re-run at λ<sub>F</sub> tests that linearisation. In that re-run a member is rejected, by a rule fixed beforehand that uses no
DFT value, if a test fails, if any relaxed interatomic distance falls below 2.0 Å (collapse), or if its diamond cell is
Born-unstable. The headline σ is the delta-method σ where it agrees with the λ = 1 committee to 10 %, else the committee's.</p>""", "drafted", level=3)

import csv as _csv
_INT = ("dumbbell_interstitial_energy", "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy")
def _rank1():
    """Regression-form synthetic row as a rank-one parameter correction (analysis/bias_term.py --form regression): range of
    projected bias / r* over the interstitial QoIs (CR3 and posterior metrics, lam_F), and 95 % coverage with it."""
    out = {}
    for m in ("o3d12", "o4d20"):
        p = RES / f"bias_term_{m}_regression.json"
        if not p.exists():
            return None
        j = json.load(open(p)); rs = j["synthetic_rows"]["interstitial"]["r_star_eV"]
        rows = [r for r in _csv.DictReader(open(RES / f"bias_term_{m}_regression.csv")) if r["scale"] == "lam_F"
                and r["cov"] != "cr3" and r["qoi"] in _INT]
        fr = [float(r["sigma_bias"]) / rs for r in rows]
        out[m] = {"lo": min(fr), "hi": max(fr), "rs": rs, "sig": j["synthetic_rows"]["interstitial"]["cr3_sigma_lamF_eV"],
                  "k95": sorted(j[f"cr3+bias_{c}/lam_F"]["k95"] for c in ("cr3", "post")), "k0": j["cr3/lam_F"]["k95"],
                  "n": j["cr3/lam_F"]["n"]}
    return out
rk1 = _rank1()
rk1_txt = ("; ".join(f"{m}: {v['lo']:.1f}–{v['hi']:.1f} of r* = {v['rs']:.2f} eV" for m, v in rk1.items()) if rk1 else "(pending)")
rk1_cov = (", ".join(f"{m} {v['k0']} → {v['k95'][0]}–{v['k95'][1]} of {v['n']}" for m, v in rk1.items()) if rk1 else "(pending)")
rk1_sig = (" and ".join(f"{v['sig']:.2f} eV against r* = {v['rs']:.2f} eV ({m})" for m, v in rk1.items()) if rk1 else "(pending)")

sec("method-disc", "2.7.4 Measuring the bias: a discrepancy term", r"""
<p>A bias shared by every training configuration of a kind is invisible to any delete-one jackknife: deleting one
configuration does not move the fit along it. The defect formation energies expose such a bias. Every training interstitial,
vacancy and divacancy cell is a hot MD snapshot, while the QoIs are relaxed defects.</p>
<p><b>Measuring the bias from training labels.</b> For each defect class
\(t\in\{\text{interstitial},\text{vacancy},\text{divacancy}\}\), the bulk reference is a per-atom thermal trend
\(e_{\text{b}}(\eta)=\beta_0+\beta_1\eta\), fitted by least squares to the residuals per atom of the diamond bulk cells within
0.6 Å<sup>3</sup>/atom of the defect cells' mean volume, with \(\eta=F_{\text{rms}}^2\) the mean squared DFT force component of
a cell (proportional to the thermal mean-square displacement for a harmonic solid). Each training defect cell \(i\), with
\(n_i\) atoms, energy row \(b_i\) and energy label \(y_i\), then gives a bulk-subtracted residual
\(u_i=y_i-b_i\cdot\hat\theta-n_i\,e_{\text{b}}(\eta_i)\). Regressing \(u_i\) on \(\eta_i\) and extrapolating to \(\eta\to0\)
estimates the model's error for a relaxed defect. The intercept is linear in the labels and in \(\hat\theta\), so it is a
synthetic observation:</p>
\[ r^*_t=\sum_i a_i u_i=y^*_t-d^*_t\cdot\hat\theta,\qquad a=e_1^{\mathsf T}(X^{\mathsf T}X)^{-1}X^{\mathsf T},\quad X=[1,\,\eta_i], \]
<p>with \(X\) the regression design (rows \([1,\eta_i]\)), \(e_1=(1,0)^{\mathsf T}\) selecting the intercept, \(d^*_t\) and
\(y^*_t\) the same combination of the defect rows and labels with the bulk trend's rows and labels subtracted, and standard
error \(s^*_t\) from the regression scatter. No QoI value or tolerance enters. An alternative reference, each defect cell's
nearest bulk cell in volume and thermal amplitude, was rejected on training data alone: most partners are 2-atom cells, so a
partner's residual enters about 108 times, and that form has the larger standard error and extrapolates worse to held-out cold
snapshots.</p>
<p><b>Training-cell size.</b> The training vacancy cells come in two sizes, 63 and 215 atoms, so the vacancy trend need not be
shared. We compare four linear-Gaussian models of the vacancy residuals \(u_i\) by their evidence: one intercept and slope for
both sizes; per-size intercepts with a shared slope; a shared intercept with per-size slopes; and per-size intercepts and
slopes. The coefficients have Gaussian priors (intercepts N(0, s<sub>a</sub><sup>2</sup>) with s<sub>a</sub> = 1 eV, slopes
scaled so that a slope moves \(u\) by about s<sub>a</sub> across the data) and are integrated analytically. Each size has its
own noise level, log-uniform on [10<sup>−3</sup>, 10] eV and integrated on a grid. The QoI vacancy sits in a 216-site cell, so
its bias is the 215-atom intercept, averaged over the four models with their posterior probabilities (equal prior odds). This
uses training labels only, as for the interstitials.</p>
<p><b>Why not in parameter space.</b> The interstitial synthetic row is far outside the jackknife's reach: its CR3 × λ<sub>F</sub> σ is
""" + rk1_sig + r""". The minimum-norm correction that removes \(r^*_t\), \(\delta_t=\Sigma d^*_t r^*_t/(d^{*\mathsf T}_t\Sigma d^*_t)\), with \(\Sigma\) the
propagated covariance (the POPS construction applied to the synthetic row, in the CR3 or posterior metric), added as
\(\Sigma\mathrel{+}=\delta_t\delta_t^{\mathsf T}\), carries part of the bias to the relaxed defects, but unevenly: the projected σ on the three
interstitial QoIs is """ + rk1_txt + r""" (both metrics). The hot-snapshot contrast and the relaxed defect's formation row point
in different directions in θ, so the transfer depends on the QoI. Coverage at 95 % with λ<sub>F</sub>, over the """ + str(next(iter(rk1.values()))["n"] if rk1 else "?") + r""" QoIs the delta method gives, becomes """ + rk1_cov + r""",
short of the observable-space term below.</p>
<p><b>Discrepancy term.</b> For the QoIs of defect class \(t\), the variance becomes
\(\sigma_Q^2=\sigma_{Q,\text{method}}^2+r^{*2}_t+s^{*2}_t\), with \(\sigma_{Q,\text{method}}\) the σ of the UQ method without the term (interstitial class for the three interstitials; vacancy class for the
vacancy). Every other QoI is unchanged. This leaves the pure parameter-space propagation for these QoIs and is reported as a
separate, labelled method. The term widens σ but leaves the prediction at the potential's own value, on purpose: the uncertainty
describes the potential that is run. Subtracting \(r^*_t\) from the prediction would correct the potential, a different object
and one that no longer has a parameter vector; Table 8 gives the errors of that corrected prediction for reference.</p>
<p>Nothing in this analysis is specific to the CR3 propagation. The bias r*<sub>t</sub> and its standard error come from the
training labels and the fitted mean alone, so the same term can be added to the σ<sub>Q</sub> of any method that shares that mean,
the baselines included.</p>""", "", level=4)

sec("method-valid", "3  Results", r"""
<h3>3.1 Validation statistics</h3>
<p>Section 2 ends with a σ for every prediction we test: a per-atom force covariance and conformal region (Section 2.5), a
scaled CR3 covariance in θ (Section 2.4), and from it a σ<sub>Q</sub> for each QoI by the delta method or by committees, with or
without the discrepancy term (Sections 2.7 and 2.7.4). The same few statistics judge all of them against held-out data or DFT. None
of them is used to fit any scale.</p>
<p>For the per-atom forces (Section 3.2), the test is the coverage of the conformal regions on the 248 test configurations,
overall and per configuration type, together with the RMS of the observed force error over the predicted σ<sub>F,i</sub>, which is
1 for a calibrated Gaussian reading. For the QoIs (Sections 3.4–3.8), each prediction gives a standardised error
\(z_Q=(\hat Q-Q_{\text{DFT}})/\sigma_Q\), with \(\hat Q\) the mean model's value. We count how many of the 20 scored QoIs fall inside
the Gaussian 68 % and 95 % intervals. Each count k of n gives a posterior for the coverage probability, Beta(k + ½, n − k + ½) from
the Jeffreys prior, and we quote its 95 % credible interval and the posterior probability that the coverage reaches its nominal value.
With only 20 QoIs these are broad: 17 of 20 at 95 % gives a credible interval of """ + ci_txt(17, 20) + r""".</p>
<p>Counts treat the QoIs as independent, but they share the same coefficients, so their errors are correlated. Section 3.5 therefore
scores the whole vector of DFT values at once, by the evidence each method's predictive distribution gives it. With
\(\Delta\) the vector of \((Q_{\text{DFT}}-\hat Q)/\text{tol}_Q\) over the scored QoIs and \(C_Q\) its predicted covariance in the same
tolerance units (\(\lambda^2G\,LL^{\mathsf T}G^{\mathsf T}\) for the delta method, with \(G\) the QoI rows stacked, or the committee
covariance),</p>
\[ \log p(\Delta\mid\text{method})=-\tfrac12\,\Delta^{\mathsf T}(C_Q+\epsilon I)^{-1}\Delta
   -\tfrac12\log\lvert 2\pi(C_Q+\epsilon I)\rvert , \]
<p>with a small ridge \(\epsilon=10^{-6}\) (a floor of 10<sup>−3</sup> tolerances on σ; 10<sup>−8</sup> and 10<sup>−4</sup> as
sensitivities). Every method's scale is fixed beforehand, on held-out forces or held-out scores, so no method has a free parameter left,
and the difference of two methods' log evidence on the same QoIs is a log Bayes factor. The first term rewards predictions whose
errors, and their correlations, match the covariance; the second penalises intervals that are wider than they need to be. The evidence
therefore weighs calibration and sharpness together, and a method cannot gain by inflating its intervals.</p>
<p>Coverage alone rewards intervals that are too wide. We therefore also report sharpness, the median of \(\sigma_Q/\text{tol}_Q\)
with \(\text{tol}_Q\) the benchmark's tolerance for QoI \(Q\): a value below 1 means the interval is narrower than the accuracy the
benchmark asks for. A useful method has near-nominal coverage at the smallest sharpness. The continuous ranked probability
score (CRPS) combines the two in one number per QoI [Gneiting and Raftery 2007]. For a Gaussian prediction with mean
\(\hat Q\) and standard deviation \(\sigma_Q\) it is
\(\mathrm{CRPS}=\sigma_Q\,[z(2\Phi(z)-1)+2\varphi(z)-1/\sqrt\pi]\), with \(z\) the standardised error and \(\Phi\),
\(\varphi\) the standard normal distribution and density functions. It is a proper score: it is smallest in expectation for the
true predictive distribution, it grows for intervals that are too narrow and for intervals that are too wide, and it tends
to the absolute error as σ<sub>Q</sub> → 0. We report its mean over the scored QoIs in tolerance units, CRPS/tol (lower is
better); committee methods are scored with the Gaussian of their spread, as for coverage.</p>""", "drafted")

def _phase_names(m, lo):
    """'the sh, β-Sn and hcp energies is 86, 43 and 24 times' for the phases whose physical/rescaled ratio is >= lo."""
    it = sorted(((float(r["ratio_physical_vs_rescaled"]), k) for (mm, k), r in phr.items() if mm == m and k.startswith("dE_")
                 and r["ratio_physical_vs_rescaled"] not in ("", None) and float(r["ratio_physical_vs_rescaled"]) >= lo), reverse=True)
    names = [_qn(k).replace(" phase energy", "").replace("the ", "") for _, k in it]
    nums = [f"{v:.0f}" for v, _ in it]
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    return f"the σ of the {join(names)} energies is {join(nums)} times"


pops = pw.get(("o3d12", "pops"))
pops_sd = sci(pops["sd_E_meV_per_atom"]) if pops else "?"
b3txt = " and ".join(f"{int(float(b3[m]['k95']))}/{int(float(b3[m]['n']))} ({m})" for m in models if b3[m])
b1txt = " and ".join(f"{int(float(b1[m]['k95']))}/{int(float(b1[m]['n']))} ({m})" for m in models if b1[m])
pops_w = "; ".join(f"{m}: 5 % cut {pw[(m, 'popslev5')]['sd_E_meV_per_atom']:.3g}, 50 % cut {pw[(m, 'popslev50')]['sd_E_meV_per_atom']:.3g}, "
                   f"CR3 {pw[(m, 'cr3')]['sd_E_meV_per_atom']:.2f} / {pw[(m, 'cr3lamF')]['sd_E_meV_per_atom']:.2f}"
                   for m in ("o3d12", "o4d20") if all((m, k) in pw for k in ("popslev5", "popslev50", "cr3")))
_pc = {(r["model"], r["method"]): r for r in rd("coverage.csv") if r["method"] in ("B2lev5-committee", "B2lev50-committee")}
pops_cov = "; ".join(f"{m}: {int(float(_pc[(m, f'B2lev{c}-committee')]['k95']))}/{int(float(_pc[(m, f'B2lev{c}-committee')]['n']))} at 95 % "
                     f"with the {c} % cut (RMS z {float(_pc[(m, f'B2lev{c}-committee')]['rms_z']):.1f}, median σ/tol "
                     f"{float(_pc[(m, f'B2lev{c}-committee')]['median_sigma_over_tol']):.2f})"
                     for m in ("o3d12", "o4d20") for c in (5, 50) if (m, f"B2lev{c}-committee") in _pc)
_pr = {m: json.load(open(RES / f"pops_reference_{m}.json")) for m in ("o3d12", "o4d20") if (RES / f"pops_reference_{m}.json").exists()}
def _pr_txt():
    """The reference package on the port's own rows (analysis/pops_reference.py): its defaults and every row.  The
    box-by-box parity at B2's settings is not yet reconciled (results/pops_reference_<m>.json "parity"), so it is
    not reported as agreement."""
    if not _pr:
        return "(Reference-package comparison pending.)"
    v0 = next(iter(_pr.values()))
    rid = " and ".join(f"{v['ridge_rel_ref']:.0e} ({m})" for m, v in _pr.items())
    dft = "; ".join(f"{m}: {v['reference_default']['sd_E_meV_atom']:.3g} meV/atom, with {v['reference_default']['pass_tight']} of {v['K']} "
                    f"members within 0.05 eV/atom and 1 eV/Å of the mean on the rattled cells" for m, v in _pr.items())
    allr = "; ".join(f"{m}: {v['reference_every_row']['sd_E_meV_atom']:.3g} meV/atom" for m, v in _pr.items())
    return (f"We also fitted the reference package POPSRegression {v0['popsregression']} to the port's own loss-weighted rows. "
            f"Its ridge, chosen by evidence, is {rid} of the largest eigenvalue, so its fit is in effect unregularised. With "
            f"its default 50 % cut the members spread the bulk energy by {dft}. With every row they spread it by {allr}: the "
            "unphysical box of the every-row default is a property of POPS on this design, not of the port. [Draft note: a "
            "box-by-box comparison of the port with the package at B2's settings does not yet agree and is being checked.]")
pr_txt = _pr_txt()
_pc_old = json.load(open(RES / "pops_compare_o3d12_oct31box.json")) if (RES / "pops_compare_o3d12_oct31box.json").exists() else None
_pcs = {m: json.load(open(RES / f"pops_compare_{m}.json")) for m in ("o3d12", "o4d20") if (RES / f"pops_compare_{m}.json").exists()}
_box = {m: np.load(f"/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/fits/{m}/pops/pops_box_lev50.npz") for m in ("o3d12", "o4d20")}
if _pc_old:
    _o = _pc_old
    pc_cond = sci(_o["cond"]["cond_X"])
    _dz = np.load("/storage/eng/essswb/projects/si-qoi-uq-paper/runs/popsref/design_o3d12.npz")   # analysis/pops_dump.py
    pc_lamblr = json.loads(str(_dz["meta"]))["lam_blr"] / float(_dz["Lam"].max())
    pc_old = (f"{_o['b2']['sd_E_meV_atom']:.0f} (the box an earlier version of this study used), {_o['eig']['sd_E_meV_atom']:.1f} (the port's "
              f"eigendecomposition route), {_o['svd']['sd_E_meV_atom']:.1f} (an SVD of the rows) and {_o['package_beta']['sd_E_meV_atom']:.1f} "
              f"meV/atom (the reference package's hypercube step)")
    pc_sweep = ", ".join(f"{r['sd_E_svd']:.3g} meV/atom at {sci(r['ridge_rel'])}" for r in _o["sweep"] if r["ridge_rel"] in (1e-10, 1e-6, 1e-2))
else:
    pc_cond = pc_old = pc_sweep = "(pending)"; pc_lamblr = float("nan")
pc_old_short = (f"bulk-energy spreads from {_o['eig']['sd_E_meV_atom']:.1f} to {_o['b2']['sd_E_meV_atom']:.0f} meV/atom at a relative "
                f"ridge of 10<sup>−14</sup>") if _pc_old else "(pending)"
# the rebuilt B2 boxes against four evaluations from the same rows, mean, ridge and cut (analysis/pops_compare.py)
pc_parity = " and ".join(f"{sci(max(v[k]['cov_rel_diff_vs_b2'] for k in ('eig', 'svd', 'package_theta')))} ({m})" for m, v in _pcs.items()) \
    if len(_pcs) == 2 else "(pending)"
pc_axes = (" and ".join(f"{v['package_theta']['n_modes']} against {v['package_beta']['n_modes']} modes, "
                        f"{v['package_theta']['sd_E_meV_atom']:.2f} against {v['package_beta']['sd_E_meV_atom']:.2f} meV/atom ({m})"
                        for m, v in _pcs.items()) + " in θ against Γθ, at the selected ridges") if len(_pcs) == 2 else "(pending)"
pc_sel = "; ".join(f"{m}: " + ", ".join(f"{q} {sci(v)}" for q, v in json.loads(str(b["ridge_selected"])).items()) for m, b in _box.items())
pc_refridge = " and ".join(f"{sci(v['ridge_rel_ref'])} ({m})" for m, v in _pr.items()) if _pr else "(pending)"
dtrend = json.load(open(RES / "defect_trend.json")) if (RES / "defect_trend.json").exists() else {}
disc = json.load(open(RES / "discrepancy.json")) if (RES / "discrepancy.json").exists() else {}
disc_rows = rd("discrepancy.csv")
DEF_Q = (("vacancy_energy", "vacancy"), ("dumbbell_interstitial_energy", "dumbbell int."),
         ("hexagonal_interstitial_energy", "hexagonal int."), ("tetrahedral_interstitial_energy", "tetrahedral int."))
qrow = lambda m, k, meth="CR3-headline", sc="lam_F": next((r for r in rd("qoi_uq.csv") if r["model"] == m and r["qoi"] == k
                                                          and r["method"] == meth and r["scale"] == sc), None)
_qrows = rd("qoi_uq.csv")
b1disc = " and ".join(f"{d_['k95']}/{d_['n']} ({m})" for m in models
                      for d_ in [disc.get(m, {}).get("B1-headline+disc/q_b1")] if d_) if disc else "?"
_QN = {"diamond_bulk_modulus": "the bulk modulus", "vacancy_energy": "the vacancy", "dumbbell_interstitial_energy": "the dumbbell", "hexagonal_interstitial_energy": "the hexagonal",
       "tetrahedral_interstitial_energy": "the tetrahedral interstitial", "surface_100 (J/m2)": "the (100) surface energy",
       "gamma_us glide (J/m2)": "γ<sub>us</sub> glide", "gamma_us shuffle (J/m2)": "γ<sub>us</sub> shuffle",
       "decohesion_max_stress": "the decohesion stress"}
_qn = lambda k: _QN.get(k) or (k.replace("dE_", "the ").replace(" (meV/at)", " phase energy").replace("beta-Sn", "β-Sn") if k.startswith("dE_") else k)
def misses(m, meth="CR3-headline", sc="lam_F"):
    return [_qn(r["qoi"])
            for r in _qrows if r["model"] == m and r["method"] == meth and r["scale"] == sc and r["z"] not in ("", None)
            and abs(float(r["z"])) > 1.96]
def miss_txt(m):
    ms = misses(m)
    ints = ["the dumbbell", "the hexagonal", "the tetrahedral interstitial"]
    if all(x in ms for x in ints):
        ms = ["the three interstitial formation energies"] + [x for x in ms if x not in ints]
    return ms[0] if len(ms) == 1 else ", ".join(ms[:-1]) + " and " + ms[-1]
k95 = lambda m, meth, sc: (lambda r: f"{int(float(r['k95']))}/{int(float(r['n']))}" if r else "?")(get(m, meth, sc))
sot = lambda m, meth, sc: (lambda r: f"{float(r['median_sigma_over_tol']):.2f}" if r else "?")(get(m, meth, sc))
tight = (head["o3d12"]["median_sigma_over_tol"] / head["o4d20"]["median_sigma_over_tol"]) if all(head.get(m) for m in ("o3d12", "o4d20")) else float("nan")
sec("results", "3.4  Coverage and sharpness", f"""
<div class="wrap"><table><caption>Table 3. Coverage of the DFT values (count [95 % credible interval of the coverage probability]) and sharpness, for the four methods of Figures 1 and 1b.</caption>
<thead><tr><th>model</th><th>method</th><th>n</th><th>68 %</th><th>95 %</th><th>RMS z</th><th>median σ/tol</th><th>CRPS/tol</th></tr></thead>
<tbody>{cov_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 3b. Sensitivity of the coverage to the scale and to the committee variant: the CR3 shape
with the energy, virial or pooled scale (Section 3.3), the full re-run at λ<sub>F</sub> keeping physical members only (Section 3.6),
and POPS with the 50 % leverage cut (Section 3.7).</caption>
<thead><tr><th>model</th><th>method</th><th>n</th><th>68 %</th><th>95 %</th><th>RMS z</th><th>median σ/tol</th><th>CRPS/tol</th></tr></thead>
<tbody>{sens_rows}</tbody></table></div>
<p>With the headline CR3 × λ<sub>F</sub>, {k95("o3d12", "CR3-headline", "lam_F")} QoIs of o3d12 and {k95("o4d20", "CR3-headline", "lam_F")}
of o4d20 lie inside their 95 % intervals (Table 3, Figure 1 top row, Figure 1b). Both counts are below the nominal 19, and the coverage posteriors (Section 3.1) give
P(coverage ≥ 0.95) = {pcov("o3d12")} for o3d12 and {pcov("o4d20")} for o4d20: the headline under-covers at the 95 % level. The misses are not spread across the QoIs. For o3d12 they are
{miss_txt("o3d12")}; for o4d20 they are {miss_txt("o4d20")}. The interstitial formation energies, missed by both
models, are the subject of Section 3.8: they share a bias that a delete-one jackknife cannot see, not a scale that is too small.
Elsewhere the 95 % intervals reach DFT, and the 68 % counts ({int(float(get("o3d12", "CR3-headline", "lam_F")["k68"]))} and
{int(float(get("o4d20", "CR3-headline", "lam_F")["k68"]))} of 20, against a nominal 13.6) are close to nominal.</p>
<p>The intervals follow the accuracy of the model. The median σ/tol is {sot("o3d12", "CR3-headline", "lam_F")} for o3d12 and
{sot("o4d20", "CR3-headline", "lam_F")} for o4d20, {tight:.1f}× narrower: o3d12's intervals are wider than the benchmark tolerance, as they
should be for a model whose typical QoI error is about one tolerance (Figure 0), and o4d20's are mostly inside it. So one scale fitted
to held-out forces gives similar coverage for both models while the intervals shrink with model quality. The CRPS, which weighs
calibration and sharpness in one score, ranks the headline first among the four methods for both models:
{crps("o3d12", "CR3-headline", "lam_F")} for o3d12 and {crps("o4d20", "CR3-headline", "lam_F")} for o4d20, against
{crps("o3d12", "B1-headline", "q_b1")} / {crps("o4d20", "B1-headline", "q_b1")} for B1,
{crps("o3d12", "B2lev50-committee", "lam_1")} / {crps("o4d20", "B2lev50-committee", "lam_1")} for B2 and
{crps("o3d12", "B3-headline", "lam_1")} / {crps("o4d20", "B3-headline", "lam_1")} for B3 (o3d12 / o4d20; Table 3). The margins are small
because the mean is dominated by the QoIs with the largest errors, which no width can fix: for o3d12 the three interstitials
are 12–17 tolerances off, so its CRPS/tol is near 3 for every method.</p>
<p>Of the baselines (Section 3.7), B1 and B3 cover fewer QoIs. POPS (B2) covers {k95("o3d12", "B2lev50-committee", "lam_1")} (o3d12)
and {k95("o4d20", "B2lev50-committee", "lam_1")} (o4d20), at widths close to the headline's (median σ/tol
{sot("o3d12", "B2lev50-committee", "lam_1")} and {sot("o4d20", "B2lev50-committee", "lam_1")}). The sensitivity table (Table 3b) shows that the choice of
scale matters at the margin. The virial scale λ<sub>V</sub>, larger than λ<sub>F</sub> by about a third, covers
{k95("o3d12", "CR3-headline", "lam_V")} (o3d12) and {k95("o4d20", "CR3-headline", "lam_V")} (o4d20) at the price of wider intervals
(median σ/tol {sot("o3d12", "CR3-headline", "lam_V")} for o3d12 and {sot("o4d20", "CR3-headline", "lam_V")} for o4d20); λ<sub>E</sub> and λ<sub>pool</sub> give about the
same as λ<sub>F</sub>. The full committee re-run at λ<sub>F</sub>, keeping only physical members, covers
{k95("o3d12", "CR3-committee-lamF-physical", "lam_F")} for o3d12 and {k95("o4d20", "CR3-committee-lamF-physical", "lam_F")} for o4d20,
against {k95("o3d12", "CR3-headline", "lam_F")} and {k95("o4d20", "CR3-headline", "lam_F")} for the headline. The differences come mainly from the
phase energies, which are strongly nonlinear in θ at λ<sub>F</sub> (Section 3.6).</p>
<figure><img src="fig6_model_compare.png" alt="Per QoI, error over tolerance with 95 percent intervals for o3d12 and o4d20, DFT as crosses at zero; one row each for the headline and the three baselines"><figcaption>Figure 1. Model prediction minus DFT, in units of the benchmark tolerance, with the 95 % interval, for o3d12
(circles) and o4d20 (squares); DFT is the cross at 0 and the shaded band is ± 1 tolerance. The axis is linear within ± 1 and
logarithmic beyond, and is shared by all rows. Top row: the headline, CR3 × λ<sub>F</sub>; the larger model's intervals are
{tight:.1f}× tighter in median σ/tol while its coverage stays comparable. Below it the baselines of Section 2.6: B1, the posterior
scaled by one conformal quantile; B2, POPS with the 50 % leverage cut (the reference package's default); B3, CR3 without calibration (λ = 1). Each row's legend gives
its 95 % coverage and median σ/tol.</figcaption></figure>
<figure><img src="fig2_coverage.png" alt="Coverage per method with binomial intervals"><figcaption>Figure 1b. Fraction of the 20 scored QoIs inside the 68 % and 95 % intervals for the four methods of Figure 1
(headline, B1, B2 with the 50 % leverage cut, B3), with 95 % credible intervals from the Jeffreys-prior Beta posterior; dashed lines are the nominal levels.</figcaption></figure>
<h3>3.5 Joint evidence</h3>
<div class="wrap"><table><caption>Table 4. Log evidence of the 20 scored DFT values (17 with the delta method) under each method's
predictive distribution, in tolerance units (ε = 10<sup>−6</sup>), and the log Bayes factor against the headline with the same
covariance source. d² and log det are the two parts of the evidence: the misfit of the errors, and the width of the covariance.</caption>
<thead><tr><th>model</th><th>method</th><th>covariance from</th><th>n</th><th>log p</th><th>log BF vs headline</th><th>d²</th><th>log det</th></tr></thead>
<tbody>{jrows}</tbody></table></div>
<p>The QoIs share one coefficient vector, so their errors are correlated, and Figure 3 shows how. In the CR3 committee for o4d20 the
correlations form blocks: the elastic constants with each other and with the β-Sn and sh phase energies, which are both compressed
diamond-like structures; and the four point defects with each other (0.3–0.8), most strongly the three interstitials. Most other pairs
are only weakly correlated.</p>
<p>Table 4 compares the methods by their evidence. The headline is far ahead of B1 and B3 for both models: by log Bayes
factors of {lbf_txt("o3d12", "B1/q_b1")} (B1) and {lbf_txt("o3d12", "B3/lam_1")} (B3) for o3d12, and {lbf_txt("o4d20", "B1/q_b1")} and
{lbf_txt("o4d20", "B3/lam_1")} for o4d20, with the committee covariance. On the scale of Kass and Raftery [1995] a log Bayes factor
beyond about 3 is strong evidence and beyond 5 very strong. B3 loses through the misfit term, with intervals far too narrow for
its errors. B1 loses less, because its overall size is calibrated, but its shape puts the width in the wrong places. POPS (B2) is
the close competitor. Without the discrepancy term the evidence prefers it for o3d12 ({lbf_txt("o3d12", "B2lev50/lam_1")}) and the
headline for o4d20 ({lbf_txt("o4d20", "B2lev50/lam_1")}). The o3d12 preference does not survive the discrepancy term of
Section 3.8: with the term added to both, the headline is preferred for both models,
strongly for o3d12 ({lbf2_txt("o3d12", "CR3/lam_F+disc", "B2lev50/lam_1+disc")}) and weakly for o4d20
({lbf2_txt("o4d20", "CR3/lam_F+disc", "B2lev50/lam_1+disc")}). One at a time the headline covers more QoIs for o3d12
({k95("o3d12", "CR3-headline", "lam_F")} against {k95("o3d12", "B2lev50-committee", "lam_1")}) and POPS for o4d20
({k95("o4d20", "B2lev50-committee", "lam_1")} against {k95("o4d20", "CR3-headline", "lam_F")}): per-QoI coverage and the joint
evidence need not agree. The correlations can also favour a method that
misses QoIs one at a time: o4d20's headline misses five QoIs, but three of the five are the interstitials, which its covariance
expects to move together, so jointly they cost less than three independent misses.</p>
<p>The virial scale is better than the force scale for o3d12 ({lbf_txt("o3d12", "CR3/lam_V")}), whose QoI errors are larger
relative to its force errors, and about as good for o4d20 ({lbf_txt("o4d20", "CR3/lam_V")}). The discrepancy term of Section 3.8 is
strongly supported for o3d12 ({lbf_txt("o3d12", "CR3/lam_F+disc")} with the committee covariance,
{lbf_txt("o3d12", "CR3/lam_F+disc", "delta")} with the delta method), and roughly neutral for o4d20
({lbf_txt("o4d20", "CR3/lam_F+disc")} and {lbf_txt("o4d20", "CR3/lam_F+disc", "delta")}): o4d20's defect bias is small, so the term's
extra width costs about as much as the misses it removes. The delta-method and committee covariances agree, and the ranking of the
methods is the same for ε from 10<sup>−8</sup> to 10<sup>−4</sup>.</p>
<figure><img src="fig3_qoi_correlation.png" alt="QoI correlation matrix of the o4d20 CR3 committee"><figcaption>Figure 3. Correlation
between the QoIs in the CR3 committee for o4d20 (K = 200 members).</figcaption></figure>
<h3>3.6 Delta method, committees and nonlinearity</h3>
<p>The two routes to σ<sub>Q</sub> agree where both apply. At λ = 1, the delta-method σ is within 10 % of the committee's for all but
{nflag["o3d12"]} of the {ncmp["o3d12"]} QoIs of o3d12 and {nflag["o4d20"]} of the {ncmp["o4d20"]} of o4d20; the ratio ranges from
{dvc_rng["o3d12"][0]:.2f} to {dvc_rng["o3d12"][1]:.2f} for o3d12 and from {dvc_rng["o4d20"][0]:.2f} to {dvc_rng["o4d20"][1]:.2f} for o4d20.
The flagged QoIs ({", ".join(_qn(k) for k in dvc_flag["o3d12"])} for o3d12; {", ".join(_qn(k) for k in dvc_flag["o4d20"])} for o4d20)
are taken from the committee, as Section 2.7.3 prescribes. The implicit differentiation through each relaxation, including that of the
lattice constant, is therefore accurate at the size of the CR3 distribution itself. The elastic constants are a stricter check,
because they can be computed by two routes, from the stress or from the curvature of the energy. Figure 4 shows that the two give
nearly the same covariance ellipses for both models, and that the DFT values lie well outside the 1σ ellipses for o3d12, whose
C<sub>11</sub> is about 15 % too low, but inside them for o4d20.</p>
<p>At λ<sub>F</sub> the distribution is several times wider, and the linear rescaling of the λ = 1 committee,
Q<sub>k</sub>(λ) = Q̂ + λ(Q<sub>k</sub> − Q̂), can fail. The full re-run at λ<sub>F</sub> tests it. Of its 200 members,
{phys["o3d12"]["accepted"]} (o3d12) and {phys["o4d20"]["accepted"]} (o4d20) pass the physicality rule fixed beforehand; the rest
collapse ({phys["o3d12"]["rejected_collapse"]} and {phys["o4d20"]["rejected_collapse"]}), and none fails a test. Outside the phase
energies, the σ of the physical members is {min(nonphase["o3d12"]):.2f}–{max(nonphase["o3d12"]):.2f}× the rescaled σ for o3d12 and
{min(nonphase["o4d20"]):.2f}–{max(nonphase["o4d20"]):.2f}× for o4d20, so the rescaling holds for the elastic constants, defects and
surfaces. The phase energies are different: for o3d12 {_phase_names("o3d12", 10)} the rescaled σ, and for o4d20
{_phase_names("o4d20", 10)} it. These are the phases furthest from diamond, whose structures relax into other phases when θ moves far
enough, so the phase-energy σ at λ<sub>F</sub> is a property of that relaxation path rather than of a Gaussian in θ. The headline
keeps the rescaled σ, and Table 3b reports the full re-run as a sensitivity.</p>
<figure><img src="fig4_elastic_ellipses.png" alt="Elastic constant covariance ellipses from the stress and energy routes, both models"><figcaption>Figure 4.
Elastic constants: the 1σ covariance ellipses of (C<sub>11</sub>, C<sub>12</sub>) and (C<sub>11</sub>, C<sub>44</sub>) at λ<sub>F</sub> from
the delta method, by the stress route (solid) and the energy route (dashed), for o3d12 (top) and o4d20 (bottom); the dot is the
mean model's value and the star DFT.</figcaption></figure>
<h3>3.7 Baselines</h3>
<p>B1 and B3 fall short in different ways (Figure 1, Figure 1b and Table 3); B2 does not, once its box is numerically defined. B3, the CR3 shape without
calibration, covers only {b3txt} of the QoIs at 95 %: the shape alone is several times too narrow, as the test-split scales of
Section 3.3 already show. B1, the posterior with one conformal scale, covers {b1txt}. It has a calibrated overall size but the wrong
shape, so it is too wide for some QoIs and too narrow for others, and its joint evidence is far below the headline's (Table 4).</p>
<p>B2, POPS, needed more care. With the ace-jax port's default, which builds the hypercube from every training row, the members are
unphysical: the bounds are set by a handful of rows with near-zero leverage, whose corrections are unbounded, and the members spread
the bulk energy of o3d12 by {pops_sd} meV/atom. Rejecting unphysical members cannot rescue a box of that size.</p>
<p>The POPS box is also numerically fragile on these designs. The o3d12 design has a condition number of
{pc_cond} and its evidence fit puts σ<sub>c</sub> on its upper bound, so the ridge that gives the BLR mean is
{sci(pc_lamblr)} of the largest eigenvalue. Below a relative ridge of about 10<sup>−11</sup> the box is set by roundoff. At the
10<sup>−14</sup> that an earlier ridge selection chose, four equivalent evaluations from the same rows, mean and cut give
bulk-energy spreads of {pc_old}. The POPS step of the reference package POPSRegression [Swinburne, POPSRegression] agrees with the
stable SVD evaluation there, so the port and the package differ only in these numerics, not in the method. We therefore select
the ridge from 10<sup>−10</sup> upwards ({pc_sel}). For o3d12 the selection takes the smallest ridge it is offered, so the
floor sets the box; for o4d20 it chooses inside the grid. At these ridges the box is reproduced: the stored B2 box, the
eigendecomposition route, the SVD and the reference package's hypercube step on the same corrections agree in covariance to
{pc_parity}, relative. The package takes its axes in the coordinates it is given, here the prior-scaled Γθ; in those its box
differs ({pc_axes}). Within the numerically defined range the box widens with the ridge: for
o3d12 its bulk-energy spread is {pc_sweep}. The reference package's own default ridge, chosen by evidence, is
{pc_refridge} of the largest eigenvalue, in the range where the box is not defined, so we do not use it.</p>
<p>We ran committees at
the 50 % cut (B2) and at 5 % (Table 3b). Their bulk-energy spreads, against the CR3 members at λ = 1 and at
λ<sub>F</sub>, are {pops_w} meV/atom. B2 covers {pops_cov}. With a numerically defined box, POPS gives QoI intervals of about the headline's width and
coverage without a calibration step. What sets its width is the ridge and the leverage cut instead: over the numerically defined
ridges the o3d12 box's bulk-energy spread changes by a factor of about 60 (above).</p>""", "drafted", level=3)

trend_rows = ""
for m in ("o3d12", "o4d20"):
    tr_ = dtrend.get(m, {}).get("interstitial")
    errs = [qrow(m, k) for k in ("dumbbell_interstitial_energy", "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy")]
    if tr_ and all(errs):
        qerrs = " / ".join(f"{e['dft'] - e['mean']:+.2f}" for e in errs)
        trend_rows += (f"<tr><td>{m}</td><td class=n>{tr_['defect_intercept_eV']:+.2f} ± {tr_['defect_intercept_se']:.2f}</td>"
                       f"<td class=n>{qerrs}</td>"
                       f"<td class=n>{tr_['defect_mean_eV']:+.2f}</td></tr>")
dcov_rows = ""
for m in ("o3d12", "o4d20"):
    for meth, sc, lab in (("CR3-headline", "lam_F", "CR3 × λ<sub>F</sub>"), ("CR3-headline", "lam_V", "CR3 × λ<sub>V</sub>"),
                          ("B1-headline", "q_b1", "B1: posterior × q̂")):
        r0, r1 = get(m, meth, sc), disc.get(m, {}).get(f"{meth}+disc/{sc}")
        if r0 and r1:
            dcov_rows += (f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r0['k95'])}/{int(r0['n'])}</td>"
                          f"<td class=n>{r1['k95']}/{r1['n']} [{r1['ci95'][0]:.2f}, {r1['ci95'][1]:.2f}]</td>"
                          f"<td class=n>{int(r0['k68'])} → {r1['k68']}</td><td class=n>{f(r1['median_sigma_over_tol'])}</td>"
                          f"<td class=n>{f(r0.get('crps_tol'))} → {f(r1.get('crps_tol'))}</td></tr>")
_rs = lambda m, q: disc.get(m, {}).get("r_star", {}).get("vacancy" if q == "vacancy_energy" else "interstitial", {}).get("r_star_eV", float("nan"))
dz_rows = "".join(
    f"<tr><td>{r['model']}</td><td>{dict(DEF_Q)[r['qoi']]}</td><td class=n>{r['dft'] - r['mean']:+.3f}</td>"
    f"<td class=n>{r['dft'] - r['mean'] - _rs(r['model'], r['qoi']):+.3f}</td>"
    f"<td class=n>{r['sigma_method']:.3f}</td><td class=n>{r['sigma']:.3f}</td><td class=n>{r['z_without']:+.2f} → {r['z']:+.2f}</td></tr>"
    for r in disc_rows if r["qoi"] in dict(DEF_Q) and r["method"] == "CR3-headline+disc" and r["scale"] == "lam_F")
models_disc = ", ".join(sorted(k for k in disc if k != "estimator")) or "none yet"
vck = json.load(open(RES / "vacancy_check.json")) if (RES / "vacancy_check.json").exists() else {}
EST_LAB = {"interstitial": "bulk-trend regression", "divacancy": "bulk-trend regression", "vacancy": "215-atom intercept, size-model average"}
r7a_rows = "".join(f"<tr><td>{m}</td><td>{t}</td><td>{EST_LAB.get(t, '')}</td><td class=n>{v['r_star_eV']:+.3f} ± {v['se_eV']:.3f}</td></tr>"
                   for m in ("o3d12", "o4d20") for t, v in disc.get(m, {}).get("r_star", {}).items())
vev = json.load(open(RES / "vacancy_evidence.json")) if (RES / "vacancy_evidence.json").exists() else {}
VMOD = (("pool", "one intercept, one slope"), ("int", "per-size intercepts, one slope"),
        ("slope", "one intercept, per-size slopes"), ("sep", "per-size intercepts and slopes"))
VLAB = {"o3d12": "o3d12", "o4d20": "o4d20", "model_vs_model (o4d20 - o3d12)": "model vs model"}
vac_rows = ""
for key, mlab in VMOD:
    cells = ""
    for m in VLAB:
        h = vev.get(m, {}).get("s_a=1", {}).get(key)
        cells += (f"<td class=n>{h['log_bf_vs_best']:+.1f}</td><td class=n>{h['r215_mean_eV']:+.2f} ± {h['r215_sd_eV']:.2f}</td>"
                  if h else "<td>–</td><td>–</td>")
    vac_rows += f"<tr><td>{mlab}</td>{cells}</tr>"
if vev:
    vac_rows += "<tr><td>average over models</td>" + "".join(
        f"<td></td><td class=n>{vev[m]['s_a=1']['bma']['r215_mean_eV']:+.2f} ± {vev[m]['s_a=1']['bma']['r215_sd_eV']:.2f}</td>" for m in VLAB) + "</tr>"
def vsens(m):
    """log BF of the shared-intercept model against the best, and the model average, at the other prior scales."""
    return "; ".join(f"s<sub>a</sub> = {sa:g} eV: {vev[m][f's_a={sa:g}']['slope']['log_bf_vs_best']:+.1f}, average "
                     f"{vev[m][f's_a={sa:g}']['bma']['r215_mean_eV']:+.2f} eV" for sa in (0.3, 3.0))
cold_txt, geo_txt = "", ""
if vck:
    cc = {m: next(o for o in vck["relax_cold"][m] if o["size"] == 63 and o["x"] < 0.05) for m in ("o3d12", "o4d20")}
    cold_txt = ", ".join(f"{m}: relaxation gain {cc[m]['model_gain_eV']:.2f} eV against a DFT bound of {cc[m]['dft_gain_bound_eV']:.2f} eV" for m in cc)
    geo_txt = ", ".join(f"{m} {min(g['neighbour_pair_distances_A']):.2f}–{max(g['neighbour_pair_distances_A']):.2f} Å" for m, g in vck["geometry"].items())
sec("results-defects", "3.8  Defect formation energies: diagnosis and the discrepancy term", f"""
<p>For both models the relaxed interstitial formation energies are the QoIs that are confidently wrong: all three are too
low, and outside the 95 % interval. This holds for every method of Section 3.7, not only the headline, so the diagnosis below
is about the fit, not the UQ method; the discrepancy term it leads to (below) applies to any of them. Table 7 shows it for
the posterior baseline B1 as well as for CR3: with the term B1 covers {b1disc}. They are low for every ACE and PACE model in the benchmark too, and the tetrahedral one
for GAP-18.</p>
<div class="wrap"><table><caption>Table 5. The training residual trend predicts the QoI error. Interstitial cells,
residual (DFT − model) minus a temperature-matched bulk reference, extrapolated to F<sub>rms</sub> → 0 (eV), against the QoI
errors (DFT − model) of the dumbbell / hexagonal / tetrahedral interstitials, and the mean over the hot training cells.</caption>
<thead><tr><th>model</th><th>extrapolated to F → 0</th><th>QoI errors</th><th>mean over hot cells</th></tr></thead>
<tbody>{trend_rows}</tbody></table></div>
<p>Over the hot cells the model's interstitial error averages near zero, so the CR3 intervals are calibrated there (test-split
z RMS 0.85–1.13 for those cells). The error grows towards the cold limit, and the relaxed defect lies beyond the data along
a bias that all interstitial clusters share. With per-atom energy weighting, 0.4–1.6 eV in a 217-atom cell costs only
2–7 meV/atom in the fit.</p>""", "drafted", level=3)

sec("results-defects2", "", f"""
<div class="wrap"><table><caption>Table 7a. The r* ± se used in the headline discrepancy term (DFT training labels only).</caption>
<thead><tr><th>model</th><th>class</th><th>estimator</th><th>r* (eV)</th></tr></thead><tbody>{r7a_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 7. Coverage of the 20 scored QoIs with the discrepancy term on the four defect QoIs
(models with r*: {models_disc}).</caption>
<thead><tr><th>model</th><th>method</th><th>95 % without</th><th>95 % with [credible interval]</th><th>68 %</th><th>median σ/tol</th><th>CRPS/tol without → with</th></tr></thead>
<tbody>{dcov_rows}</tbody></table></div>
<p>The term lowers the CRPS of every method (Table 7), most for o3d12, whose interstitial errors dominate its score. With the
term, B1 and CR3 × λ<sub>F</sub> score almost the same ({dcrps("o3d12", "B1-headline", "q_b1")} against
{dcrps("o3d12", "CR3-headline", "lam_F")} for o3d12, {dcrps("o4d20", "B1-headline", "q_b1")} against
{dcrps("o4d20", "CR3-headline", "lam_F")} for o4d20, B1 marginally lower). Once the defects carry the term, the remaining
difference between the two shapes is small on this score, while CR3 still covers more QoIs.</p>
<p>The interstitial estimator agrees well with the observed interstitial errors. That agreement was seen before the
nearest-bulk reference was rejected on training data (above), so the choice was not made blind, although the
reasons for it use no DFT QoI value.</p>
<div class="wrap"><table><caption>Table 8. Defect QoIs, CR3 × λ<sub>F</sub>: error, error of the prediction corrected by r* (for
reference; not used), σ without and with the discrepancy term (eV), and z.</caption>
<thead><tr><th>model</th><th>QoI</th><th>DFT − model</th><th>DFT − (model + r*)</th><th>σ</th><th>σ with term</th><th>z</th></tr></thead>
<tbody>{dz_rows}</tbody></table></div>""", "", level=0)

sec("results-vacancy", "The vacancy", f"""
<p>The vacancy shows no thermal trend when its 63- and 215-atom training cells are pooled. Table 8b compares the size models of the
method above by their evidence, from training labels only.</p>
<div class="wrap"><table><caption>Table 8b. Vacancy size models: log Bayes factor against the best model, and the posterior
zero-thermal-amplitude bias of a 215-atom cell (eV), for each model and averaged over the models (s<sub>a</sub> = 1 eV). The
last pair of columns applies the same analysis to the energy difference of the two models on the training cells (o4d20 − o3d12);
the true difference of the models' own relaxed vacancy QoIs is 0.77 eV (no DFT value involved).</caption>
<thead><tr><th>model of the size dependence</th><th>o3d12 log BF</th><th>bias</th><th>o4d20 log BF</th><th>bias</th><th>model vs model log BF</th><th>bias</th></tr></thead>
<tbody>{vac_rows}</tbody></table></div>
<p>The data reject one trend for both sizes outright, and disfavour a shared intercept: the zero-temperature bias depends on the
cell size. The per-size models win, and the vacancy QoI, in a 216-site cell, takes the 215-atom intercept averaged over the
models: {vev["o3d12"]["s_a=1"]["bma"]["r215_mean_eV"]:+.2f} ± {vev["o3d12"]["s_a=1"]["bma"]["r215_sd_eV"]:.2f} eV (o3d12) and
{vev["o4d20"]["s_a=1"]["bma"]["r215_mean_eV"]:+.2f} ± {vev["o4d20"]["s_a=1"]["bma"]["r215_sd_eV"]:.2f} eV (o4d20). The 215-atom cells are all hot, so
this extrapolation is weakly determined and depends on the prior. For the shared-intercept model, the log Bayes factor and the model average at the other prior
scales are, for o3d12, {vsens("o3d12")}; for o4d20, {vsens("o4d20")}.</p>
<p>One check disagrees. Applied to the difference between the two models, where the truth is known without DFT, the
shared-intercept model predicts {vev["model_vs_model (o4d20 - o3d12)"]["s_a=1"]["slope"]["r215_mean_eV"]:.2f} of the true 0.77 eV and the model average only
{vev["model_vs_model (o4d20 - o3d12)"]["s_a=1"]["bma"]["r215_mean_eV"]:.2f}. Both miss by several standard errors, so neither
form captures how the relaxed vacancy differs from the hot cells. The one near-relaxed training cell (63 atoms) gives
direct evidence without extrapolation: its bulk-subtracted residual is +0.61 eV (o3d12) and 0.00 eV (o4d20), and relaxing it
({cold_txt}) suggests that o3d12 over-relaxes. Both models relax the harness vacancy to a symmetric structure (vacancy neighbour
distances {geo_txt}); the harness gives no DFT structure to compare.</p>
<p>The vacancy term is small in both models, and the CR3 σ alone already covers the vacancy at 95 % (Table 8).</p>
<figure><img src="fig8_model_compare_disc.png" alt="Per QoI, error over tolerance with 95 percent intervals for both models with the defect discrepancy term; intervals without the term as thin black bars"><figcaption>Figure 8.
As Figure 1, with the headline discrepancy term on the four defect QoIs (CR3 × λ<sub>F</sub>): the bulk-trend regression for the
interstitials and the size-model average for the vacancy. Thin black bars: the same QoIs' 95 % intervals without the term.
Only the four defect QoIs change.</figcaption></figure>""", "", level=4)

sec("results-forces", "3.2  Calibrated per-atom forces", f"""
<p>Figure 2 and Table 2p test the per-atom force uncertainty of Section 2.5 on the 248 test configurations, with the ARD fits;
atoms whose force rows vanish by site symmetry are dropped by the rule of Section 3.3. At a nominal 90 %, the CR3 shape with one
hold-out quantile covers {lm("o3d12", "one_group")} of the test atoms for o3d12 and {lm("o4d20", "one_group")} for o4d20. o4d20 is at nominal; o3d12 is below
it, and the cause is the transfer from the hold-out fit to the full fit. The calibration scores come from the hold-out posterior,
fitted on {atr("o3d12", "N_fit", "d")} of the {atr("o3d12", "N", "d")} training configurations, and are carried to the full posterior by
(N/N<sub>fit</sub>)<sup>β</sup>, with β estimated from a second, smaller hold-out fit ({atr("o3d12", "N_fit2", "d")} configurations) and
clipped to [0, 1/2] (Section 2.5). For o4d20 that estimate gave β = {atr("o4d20", "beta")}, a factor {atr("o4d20", "factor", ".3f")}, and the
test scores' 90 % quantile matches the transferred one to {atr("o4d20", "test_q_over_q_one", ".3f")}. For o3d12 the scale of the
smaller fit came out larger ({atr("o3d12", "lam2")} against {atr("o3d12", "lam1")}), so β<sub>raw</sub> = {atr("o3d12", "beta_raw")} was
clipped to 0 and nothing was transferred; the full fit's test scores are {atr_pct("o3d12")} larger at the 90 % quantile, which is
the shortfall. We report this as found: the transfer rule was fixed before any test coverage was seen, and changing it now would
be tuning on the test split. Where the transfer works, the shape carries most of the dependence of the error on the
environment: one quantile reaches the nominal level overall with no environment-dependent scale.</p>
<p>The coverage is not even across environments, though. The point-defect cells are under-covered and the liquid and surface
cells over-covered in both models (Figure 2, bottom); the amorphous cells are under-covered for o3d12 and over-covered for o4d20.
The learned scale of Section 2.5 corrects much of this in distribution: it covers {lm("o3d12", "learned")} and {lm("o4d20", "learned")}
overall (ranges over the five hold-out splits {lm_seeds("o3d12")} and {lm_seeds("o4d20")}), with similar median region sizes
({lm_rad("o3d12", "learned")} against {lm_rad("o3d12", "one_group")} for o3d12, {lm_rad("o4d20", "learned")} against
{lm_rad("o4d20", "one_group")} for o4d20, in units of the score), and the RMS deviation of the per-type coverage from 0.90 falls
from {evdev("o3d12", "test", "one_group")} to {evdev("o3d12", "test", "learned")} for o3d12 and from {evdev("o4d20", "test", "one_group")} to
{evdev("o4d20", "test", "learned")} for o4d20 (Table 2p; for o3d12 the amorphous cells are the exception). Out of distribution it is
less robust, as the next test shows.</p>
<p>How the quantile weights the calibration atoms matters more than either scale. A quantile that counts each calibration
configuration once in total is about 12 % smaller and covers only {lm("o3d12", "one_group_cfgq")} and
{lm("o4d20", "one_group_cfgq")}, because the many small, well-predicted cells then dominate it; the quantiles here are
atom-weighted. The CR3
shape of the BLR fit with the propagation's global λ<sub>F</sub> covers {fb("o3d12", "headline")} and {fb("o4d20", "headline")}, below the
one quantile for both models and with the same pattern across configuration types, even though λ<sub>F</sub> is fitted on these test
configurations (in sample). λ<sub>F</sub> is an RMS over force components, not a quantile of the 3-D score, and the BLR
shape is strongly anisotropic, so errors along its thin directions leave the ellipsoid. All regions in Figure 2 are
ellipsoids of the method's own 3×3 covariance; an isotropic ball |ΔF| ≤ σ(χ<sup>2</sup><sub>3</sub>(0.9)/3)<sup>1/2</sup> of the same
σ = (tr C)<sup>1/2</sup> is larger along those directions and covers more ({fb_iso("o3d12", "headline")} and {fb_iso("o4d20", "headline")} for
λ<sub>F</sub>); compared that way, λ<sub>F</sub> would look better calibrated than the one quantile, but only because its region is larger. Increasing the regulariser ε of the ellipsoid tenfold changes no coverage by more than {fb_eps_max()}.</p>
<figure><img src="fig7_force_calibration.png" alt="Per-atom force uncertainty against observed error for the learned scale, one hold-out quantile and the baselines, and test coverage per configuration type, for each model"><figcaption>Figure 2.
Top: binned on predicted σ (20 equal-count bins), RMS observed |ΔF| against RMS σ, for the CR3 shape with one hold-out
quantile (the method) and with the learned scale (ARD fit; σ<sub>F,i</sub> from the 90 % quantile), and for the per-atom σ of the
propagation's methods on the BLR fit: CR3 × λ<sub>F</sub> (the headline), B1 (posterior × q̂), B2 (POPS, 50 % leverage cut; the
box's uniform variance) and B3 (CR3, λ = 1); the per-atom density behind. Points below the diagonal are conservative.
Bottom: test coverage of the 90 % region, overall and per GAP-18 configuration type with at least 300 test atoms. Every
method's region is the same kind: the ellipsoid of its own 3×3 force covariance, e<sup>T</sup>C<sup>−1</sup>e ≤ χ<sup>2</sup><sub>3</sub>(0.9)
(for the one quantile and the learned scale this is their conformal region). The headline's λ<sub>F</sub> is fitted on the test split itself, so its per-atom coverage here is in sample.</figcaption></figure>
<div class="wrap"><table><caption>Table 2p. Test coverage of the 90 % per-atom force region per GAP-18 configuration type (types
with at least 300 test atoms): one hold-out quantile against the learned scale (mean over five hold-out splits), and the RMS
deviation of the per-type coverage from nominal.</caption>
<thead><tr><th>model</th><th>configuration type</th><th>atoms</th><th>configs</th><th>one quantile</th><th>learned scale</th></tr></thead>
<tbody>{type_rows}</tbody></table></div>
<p>An out-of-distribution test asks whether this holds for environments absent from the fit. We refitted both ARD models
with GAP-18's cracks ((111) and (110) cleavage, 1-10 front) and screw dislocations removed from training, from the
calibration hold-out and from the test file, and scored the {ood_n("o3d12")} that were removed (Table 2c). The design follows
the big-cell check of an alloy model in the ace-jax development tests (unpublished). The global scale is fitted on the
in-distribution test split, so it is out of sample here, like the conformal ones. Over all held-out atoms one quantile covers
{ood_v("o3d12", "one_group")} (o3d12) and {ood_v("o4d20", "one_group")} (o4d20), and the global scale {ood_v("o3d12", "global")} and
{ood_v("o4d20", "global")}. The cracks are covered ((111) {ood_v("o3d12", "one_group", "crack_111_1-10")} and (110)
{ood_v("o3d12", "one_group", "crack_110_1-10")} for o3d12; {ood_v("o4d20", "one_group", "crack_111_1-10")} and
{ood_v("o4d20", "one_group", "crack_110_1-10")} for o4d20), and the screw dislocations come close for o4d20
({ood_v("o4d20", "one_group", "screw_disloc")}); for o3d12 they are under-covered ({ood_v("o3d12", "one_group", "screw_disloc")}).</p>
<p>The learned scale transfers less well. It covers {ood_v("o3d12", "learned")} (o3d12) and {ood_v("o4d20", "learned")} (o4d20) of
the held-out atoms, and for o4d20 it under-covers the (111) crack ({ood_v("o4d20", "learned", "crack_111_1-10")}) and the screw
dislocations ({ood_v("o4d20", "learned", "screw_disloc")}) that one quantile covers. It narrows the regions of environments whose
training counterparts the model reproduces well, mostly near-bulk ones; atoms at dislocation cores and on crack flanks resemble
those in descriptor space but carry larger errors, so the learned scale narrows exactly where it should not. One quantile cannot
do that, which is why it is the method here: the learned scale is the better choice only when the target environments are
represented in the training data.</p>
<p>Ho, Ortner and Wang [2026] reach a more favourable verdict on their learned scale out of distribution, on compositions and
exchange-correlation functionals outside the calibration set. Their test is the correlation of the predicted uncertainty with the
error; ours is coverage. These are different properties [Shanks et al. 2026]. The two results are not in conflict, since a scale can rank errors better and still be too narrow where the model
extrapolates, but they show that a learned scale should be checked for coverage on environments left out of its calibration.</p>
<div class="wrap"><table><caption>Table 2c. Out-of-distribution per-atom force coverage at nominal 90 %: ARD refits without
cracks and screw dislocations, scored on them (90 % intervals from a bootstrap over configurations). "ID test": the same
refit's coverage on the in-distribution test split.</caption>
<thead><tr><th>model</th><th>method</th><th>ID test</th><th>all held out</th><th>crack (110)</th><th>crack (111)</th><th>screw dislocation</th></tr></thead>
<tbody>{ood_rows}</tbody></table></div>
<p>Figure 2 also shows the per-atom forces of the baselines, to which the QoI comparison of Section 3.4 applies. B3 is
about λ<sub>F</sub> times too narrow everywhere. B1, the posterior shape with one scale, under-covers
({fb("o3d12", "B1")} for o3d12, {fb("o4d20", "B1")} for o4d20), most in the amorphous, point-defect and surface cells. B2
over-covers for o3d12 ({fb("o3d12", "B2")}; per type {fb_types("o3d12", "B2")}). For o4d20 it under-covers overall
({fb("o4d20", "B2")}) and unevenly: per type {fb_types("o4d20", "B2")}, low on the liquid and surface cells.</p>""", "drafted", level=3)

dk95 = lambda m: (lambda d_: f"{d_['k95']}/{d_['n']}" if d_ else "?")(disc.get(m, {}).get("CR3-headline+disc/lam_F"))
sec("discussion", "4  Discussion", f"""
<p>A single parameter covariance, the CR3 jackknife scaled once to held-out forces, gives usable uncertainties for derived
properties that the fit never saw. For both models most QoIs fall inside their 95 % intervals ({k95("o3d12", "CR3-headline", "lam_F")}
for o3d12, {k95("o4d20", "CR3-headline", "lam_F")} for o4d20), and the intervals scale with the model: the median σ/tol is
{sot("o3d12", "CR3-headline", "lam_F")} for o3d12 and {sot("o4d20", "CR3-headline", "lam_F")} for the accurate o4d20, whose
intervals are mostly inside the benchmark tolerance. The scale is fitted once, on forces of held-out configurations, and never on a QoI.
The same CR3 shape serves per-atom forces, with one conformal quantile (Section 3.2), and relaxed QoIs, with one global
scale. The second use is what a distribution over θ buys: one draw gives energies, forces, stresses and every QoI together,
consistently.</p>
<p>The baselines show which ingredient matters. The CR3 shape without a scale (B3) is far too narrow, so calibration is essential,
as expected for a parameter variance that shrinks with the amount of data. The Bayesian posterior with a calibrated scale (B1)
has the right overall size but the wrong shape: it under-covers the QoIs and its joint evidence is far lower, because the posterior describes
noise in the labels, not the approximation error of a misspecified model. Shanks et al. [2026] find the same split between
accuracy and uncertainty for argon: MACE committees and seed ensembles predict accurately but cover far fewer errors than their
intervals claim, and calibration, sharpness and the ranking of errors need not improve together. POPS (B2) targets that error directly and, with a numerically defined box,
performs comparably on these QoIs: similar coverage and widths, and a joint evidence that prefers the CR3 propagation only once
both carry the discrepancy term (strongly for o3d12, weakly for o4d20). The CR3 approach is therefore not more accurate than POPS.
Implicit differentiation is no distinction either: Perez et al. [2025] combine it with POPS, and either distribution can be
propagated by the delta method or by committees. What CR3 adds is a parameter distribution that is calibrated and numerically
well posed.</p>
<p>Three properties of the distribution make the difference. First, its size is one scale, fitted to held-out errors and
interpretable as such. The size of the POPS box is set instead by the ridge and the leverage cut: across the numerically defined
ridges the o3d12 box's bulk-energy spread changes by a factor of about 60, the held-out selection put the o3d12 ridge at the floor,
and the every-row default is unphysical. Second, a covariance is a sum over configurations, and it is formed here through its
factor L, never as an ill-conditioned quadratic form. A hypercube is bounded by the extreme projections of single rows: on the same
rows, mean and cut, equivalent evaluations of one box gave {pc_old_short}, and the box exists only above a ridge floor. Third, a
covariance transforms consistently under a linear change of parameters, and a box does not: with the axes taken in θ or in the
prior-scaled coefficients Γθ, the same corrections give different boxes ({pc_axes}). For forces the calibration is explicit. One conformal
quantile on the CR3 shape covers {lm("o3d12", "one_group")} and {lm("o4d20", "one_group")} of the test atoms against a nominal 0.90 (the
o3d12 shortfall is the hold-out transfer of Section 3.2), and holds out of distribution on the cracks; the global force scale on the same
kind of region covers less ({fb("o3d12", "headline")} and {fb("o4d20", "headline")}), even fitted in sample. The POPS box covers
{fb("o3d12", "B2")} (o3d12) and {fb("o4d20", "B2")} (o4d20), the latter below nominal and unevenly (per type {fb_types("o4d20", "B2")}), and nothing in its construction aims at a target coverage. Two costs of CR3 remain. It is a sampling variance, so it needs held-out data and a
scale; POPS needs neither in principle, although here its ridge was chosen on a hold-out of the training set. And like POPS it cannot see a bias that every configuration of a kind shares (Section 3.8).</p>
<p>The scale is not unique. The virial scale is about a third larger than the force scale, and using it covers
{k95("o3d12", "CR3-headline", "lam_V")} QoIs for o3d12 and {k95("o4d20", "CR3-headline", "lam_V")} for o4d20 with wider intervals; its joint evidence is higher than the force scale's for o3d12 and about equal for o4d20. Misspecification depends on the observable, and one could choose the scale by the QoI's physics, the virial scale
for stress-like QoIs such as elastic constants, or report both. A single scale is also a compromise across configuration types
(Figure 5), which a per-type scale in θ could relax. A learned per-atom scale cannot be used for this directly: it rescales
force outputs atom by atom and defines no covariance in θ; a θ-space analogue would need a shape that depends on the
environment.</p>
<p>Linearisation is adequate at the size of the distribution, but not everywhere at the calibrated size. The delta method agrees
with committees run through the unmodified benchmark for almost every QoI at λ = 1, and implicit differentiation through each
relaxation, including that of the lattice constant, makes it cheap: one set of rows per QoI. At λ<sub>F</sub> the phase energies of
the structures furthest from diamond become strongly nonlinear, because their relaxations change character; for those, committees
are the safer route, and their spread is then a property of the relaxation path rather than of a Gaussian.</p>
<p>The clearest failure is the relaxed point defects, missed by every method. Its cause is in the data: every defect cell in
GAP-18 is a hot molecular-dynamics snapshot, the model's defect–bulk error drifts with thermal amplitude, and the relaxed defect
lies beyond the data along a direction that every snapshot shares. A delete-one jackknife cannot see such a bias, because deleting
one snapshot does not move the fit along it, and a rank-one parameter correction built from the snapshots carries it to the relaxed
geometry only partly and unevenly. The bias can, however, be measured from the training labels alone, by extrapolating the defect–bulk
residual to zero thermal amplitude, and added as a discrepancy term; this restores coverage to {dk95("o3d12")} (o3d12) and
{dk95("o4d20")} (o4d20) without using any DFT QoI value. For the vacancy, Bayesian comparison of models for the training cells'
size dependence shows that the bias depends on the cell size; the 215-atom extrapolation it selects is weakly determined,
a reminder that such extrapolations depend on how the training data were made. Relaxed or low-temperature defect
configurations in the training set would remove the need for the term, and the same diagnostic, a residual trend that extrapolates
away from zero, can flag where such data are missing.</p>
<p>The study has limitations. The calibration assumes that the test split and the QoI structures are exchangeable with the
training data, which the defects show is not always true. The hyperparameters are held fixed in the jackknife, so their own
uncertainty is not propagated. The locality of the model, which cannot represent interactions beyond its cutoff, appears here only
as part of the effective error. And twenty QoIs give coarse coverage estimates: 17 of 20 leaves a 95 % credible interval from
0.65 to 0.96 for the coverage probability, so the evidence of Section 3.5, which uses all the QoIs jointly, is the sharper comparison.</p>""", "")

sec("conclusions", "5  Conclusions", """
<p>For linear ACE potentials, the delete-one-configuration (CR3) jackknife covariance, scaled once, gives calibrated uncertainty at
two levels. Per atom, with one conformal quantile from a held-out part of the training set, the force regions come close to
their target coverage on held-out silicon data and on cracks left out of the fit; a scale learned from the site descriptor evens
the coverage across environments seen in training but transfers less well to those left out. For derived properties, the same covariance with one scale fitted to held-out forces,
propagated by implicit differentiation through each relaxation, covers most of 20 silicon QoIs against DFT, with intervals that
shrink with the accuracy of the model. It outperforms the Bayesian posterior and the uncalibrated shape, and on these QoIs it performs
comparably with POPS once the POPS box is numerically defined. Its advantage over POPS is not accuracy but calibration and numerical
robustness: one interpretable scale, fitted to held-out errors, sets the width at both levels; the covariance needs no ridge or leverage
cut beyond the fit's own prior, and does not depend on the parameterisation; and the per-atom regions are calibrated to a target
coverage. Where it fails, for
relaxed point defects, the cause is a bias shared by all training snapshots of a kind, which no delete-one method can see; it can be
measured from the training labels and added as a discrepancy term, which restores coverage. The methods are implemented in ace-jax.</p>""", "")

sec("methods-detail", "6  Computational details", """
<p>The training and test data are the GAP-18 silicon database [Bartók et al. 2018], split once at random into 2227 training and
248 test configurations (seed 0; the split's content hash is c433c98d49bb). Both models are linear ACE potentials fitted with
ace-jax [ace-jax] at a 5 Å cutoff, with least-squares reference energies, the smoothness prior of Section 2.2, one shared noise scale
and the structural weights of the ace-jax silicon accuracy–cost benchmark. Every calculation is in double precision. The QoIs are
computed by a Python 3 port of the silicon test framework of Bartók et al. [2018], run unmodified apart from a relaxed-ion force
tolerance of 10<sup>−3</sup> eV/Å for the elastic constants. Committees use 200 members drawn with fixed seeds. The fits, committees and analyses ran on
workstations of the University of Warwick's Scientific Computing Research Technology Platform, and the cost timings of Figure 0 on
the Sulis Tier 2 HPC platform.</p>
<p><b>What was fixed in advance.</b> The specification of the study, committed before any QoI was computed (repository below),
fixed the 21 QoIs and their tolerances, the four candidate scales λ<sub>F</sub>, λ<sub>E</sub>, λ<sub>V</sub> and
λ<sub>pool</sub>, the three baselines, the committee size and the 10 % rule that picks the delta method or the committee. λ<sub>F</sub>
was chosen as the headline before any QoI coverage was seen; the specification lists the four scales but does not record that
choice. Decided later, after QoI results had been seen: the POPS leverage cut (the specification's ace-jax default gave unphysical
members, Section 3.7; B2 now follows the reference package's default) and its ridge floor (set by where the box is numerically
defined, Section 3.7), the member-rejection rule of the λ<sub>F</sub> re-run (fixed
before that re-run), the discrepancy term and its estimators (Section 2.7.4), the per-atom force method (Section 2.5, chosen after
comparing it with a learned scale on the test split and on the out-of-distribution set), and Bayesian model comparison in place of
the specification's fixed [0.8, 1.25] band on λ<sub>V</sub>/λ<sub>F</sub>. None of these uses a DFT QoI value as an input, but
they were not made blind.</p>
<p><b>Code and data availability.</b> The uncertainty methods are part of ace-jax version 0.5.2 (MIT licence,
https://github.com/ACEsuit/ace-jax). The analysis code, the results behind every table and figure, the generator of this paper and
the original specification of the study are at https://github.com/kermodegroup/si-qoi-uq-paper [archived on Zenodo, DOI to be
added]. The GAP-18 database is available from its original publication [Bartók et al. 2018]. The Python 3 port of the silicon test
framework used here is the branch py3 of https://github.com/libAtoms/silicon-testing-framework, at commit f23b1df.</p>""", "")

sec("ack", "Acknowledgements", """
<p>The author thanks Thomas Swinburne and George Simmons for useful discussions.</p>
<p>Calculations were performed on the Sulis Tier 2 HPC platform hosted by the Scientific Computing Research
Technology Platform at the University of Warwick; Sulis is funded by EPSRC Grant EP/T022108/1 and the HPC Midlands+ consortium. We also used the Scientific Computing Research Technology Platform's desktop compute nodes.</p>
<p><b>Use of AI tools.</b> Claude Code (Anthropic) was used to write and run the analysis software in this work (the propagation,
calibration and validation code in the ace-jax repository and the scripts that produce every table and figure), and to help draft
and revise the text. All numbers in the paper are generated by that code from the stored results. Claude Code usage for software
development was supported by credits from Anthropic's AI for Science program. The author designed the study, reviewed the code,
results and text, and takes full responsibility for the correctness of the resulting implementation and for the content of this
paper.</p>""", "draft")

sec("refs", "References", """<p class="note">Checked against publisher or repository records, or the full text, on 2026-11-02.</p>
<ol class="refs">
<li>D. M. Allen, The relationship between variable selection and data augmentation and a method for prediction, Technometrics 16, 125–127 (1974). PRESS was introduced in Allen's 1971 technical report; this is its usual published citation.</li>
<li>A. P. Bartók, J. Kermode, N. Bernstein, G. Csányi, Machine learning a general-purpose interatomic potential for silicon, Phys. Rev. X 8, 041048 (2018).</li>
<li>R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani, Predictive inference with the jackknife+, Ann. Stat. 49, 486–507 (2021).</li>
<li>R. M. Bell, D. F. McCaffrey, Bias reduction in standard errors for linear regression with multi-stage samples, Survey Methodology 28(2), 169–181 (2002).</li>
<li>I. R. Best, T. J. Sullivan, J. R. Kermode, Uncertainty quantification in atomistic simulations of silicon using interatomic potentials, J. Chem. Phys. 161 (2024), doi:10.1063/5.0214590.</li>
<li>F. Bigi, S. Chong, M. Ceriotti, F. Grasselli, A prediction rigidity formalism for low-cost uncertainties in trained neural networks, Mach. Learn.: Sci. Technol. 5, 045018 (2024).</li>
<li>R. D. Cook, S. Weisberg, Residuals and Influence in Regression, Chapman and Hall (1982): predicted residuals and PRESS, Sec. 2.2; deletion of a subset of cases, \\(\\hat\\beta-\\hat\\beta_{(I)}=(X^{\\mathsf T}X)^{-1}X_I^{\\mathsf T}(I-V_I)^{-1}e_I\\), Sec. 3.6, eq. 3.6.3.</li>
<li>ace-jax: atomic cluster expansion potentials in JAX, https://github.com/ACEsuit/ace-jax.</li>
<li>W. C. Witt, C. van der Oord, E. Gelžinytė, T. Järvinen, A. Ross, J. P. Darby, C. H. Ho, W. J. Baldwin, M. Sachs, J. Kermode, N. Bernstein, G. Csányi, C. Ortner, ACEpotentials.jl: a Julia implementation of the atomic cluster expansion, J. Chem. Phys. 159, 164101 (2023).</li>
<li>R. Drautz, Atomic cluster expansion for accurate and transferable interatomic potentials, Phys. Rev. B 99, 014104 (2019).</li>
<li>kermodegroup demos: interactive regression demo, https://github.com/kermodegroup/demos (apps/regression-demo.py; scene lj-cr3).</li>
<li>T. Gneiting, A. E. Raftery, Strictly proper scoring rules, prediction, and estimation, J. Am. Stat. Assoc. 102, 359–378 (2007).</li>
<li>A. Grossfield, D. M. Zuckerman, Quantifying uncertainty and sampling quality in biomolecular simulations, in Annual Reports in Computational Chemistry, Vol. 5, Elsevier (2009).</li>
<li>K. Fisher, M. Herbst, J. Kermode, Y. Marzouk, Multivariate conformal uncertainty propagation in multitask atomistic simulation: successes and pitfalls, arXiv:2609.31384 (2026).</li>
<li>C. H. Ho, C. Ortner, Y. Wang, Flexible uncertainty calibration for machine-learned interatomic potentials, npj Comput. Mater. 12, 225 (2026).</li>
<li>R. E. Kass, A. E. Raftery, Bayes factors, J. Am. Stat. Assoc. 90, 773–795 (1995).</li>
<li>J. A. Hoeting, D. Madigan, A. E. Raftery, C. T. Volinsky, Bayesian model averaging: a tutorial, Stat. Sci. 14, 382–417 (1999).</li>
<li>P. J. Huber, The behavior of maximum likelihood estimates under nonstandard conditions, Proc. Fifth Berkeley Symp. Math. Stat. Probab. 1, 221–233 (1967).</li>
<li>G. Imbalzano, Y. Zhuang, V. Kapil, K. Rossi, E. A. Engel, F. Grasselli, M. Ceriotti, Uncertainty estimation for molecular dynamics and sampling, J. Chem. Phys. 154, 074102 (2021).</li>
<li>M. Kellner, M. Ceriotti, Uncertainty quantification by direct propagation of shallow ensembles, Mach. Learn.: Sci. Technol. 5, 035006 (2024).</li>
<li>B. J. K. Kleijn, A. W. van der Vaart, The Bernstein–von Mises theorem under misspecification, Electron. J. Stat. 6, 354–381 (2012).</li>
<li>T. Lelièvre, M. Rousset, G. Stoltz, Free Energy Computations: A Mathematical Perspective, Imperial College Press (2010).</li>
<li>J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman, Distribution-free predictive inference for regression, J. Am. Stat. Assoc. 113, 1094–1111 (2018).</li>
<li>Y. Lysogorskiy, A. Bochkarev, M. Mrovec, R. Drautz, Active learning strategies for atomic cluster expansion models, Phys. Rev. Mater. 7, 043801 (2023).</li>
<li>I. Maliyov, P. Grigorev, T. D. Swinburne, Exploring parameter dependence of atomic minima with implicit differentiation, npj Comput. Mater. 11, 22 (2025).</li>
<li>J. G. MacKinnon, M. Ø. Nielsen, M. D. Webb, Cluster-robust inference: a guide to empirical practice, J. Econometrics 232, 272–299 (2023).</li>
<li>J. G. MacKinnon, H. White, Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties, J. Econometrics 29, 305–325 (1985).</li>
<li>U. K. Müller, Risk of Bayesian inference in misspecified models, and the sandwich covariance matrix, Econometrica 81, 1805–1849 (2013).</li>
<li>F. Musil, M. J. Willatt, M. A. Langovoy, M. Ceriotti, Fast and accurate uncertainty estimation in chemical machine learning, J. Chem. Theory Comput. 15, 906–915 (2019).</li>
<li>D. Perez, A. P. A. Subramanyam, I. Maliyov, T. D. Swinburne, Uncertainty quantification for misspecified machine learned interatomic potentials, arXiv:2502.07104 (2025).</li>
<li>E. V. Podryabinkin, A. V. Shapeev, Active learning of linearly parametrized interatomic potentials, Comput. Mater. Sci. 140, 171–180 (2017).</li>
<li>H. Papadopoulos, V. Vovk, A. Gammerman, Normalized nonconformity measures for regression conformal prediction, Proc. IASTED Int. Conf. Artificial Intelligence and Applications, 64–69 (2008).</li>
<li>J. E. Pustejovsky, E. Tipton, Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models, J. Bus. Econ. Stat. 36, 672–683 (2018).</li>
<li>B. L. Shanks, G. Simmons, A. P. Bartók, J. R. Kermode, Uncertainty quantification design principles for machine learning interatomic potentials: lessons learned from hierarchical Bayesian inference, arXiv:2206.08744v2 (2026).</li>
<li>G. Simmons, T. Hudson, T. D. Swinburne, J. R. Kermode, review of uncertainty quantification for machine-learned interatomic potentials, in preparation.</li>
<li>T. D. Swinburne, D. Perez, Parameter uncertainties for imperfect surrogate models in the low-noise regime, Mach. Learn.: Sci. Technol. 6, 015008 (2024).</li>
<li>T. D. Swinburne, POPSRegression: Bayesian regression for low-noise data with misspecification uncertainty, Python package, https://pypi.org/project/popsregression/.</li>
<li>R. J. Tibshirani, R. F. Barber, E. J. Candès, A. Ramdas, Conformal prediction under covariate shift, Adv. Neural Inf. Process. Syst. 32, 2530–2540 (2019).</li>
<li>V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer (2005).</li>
<li>H. White, Maximum likelihood estimation of misspecified models, Econometrica 50, 1–25 (1982).</li>
</ol>""")

# order: the per-atom force results open the Results section (they test the first method)
_ids = [x.split(chr(34))[1] for x in S]
if "results-forces" in _ids and "method-valid" in _ids:
    _f = S.pop(_ids.index("results-forces"))
    S.insert([x.split(chr(34))[1] for x in S].index("method-valid") + 1, _f)
_ids = [x.split(chr(34))[1] for x in S]
if "method-calib" in _ids and "results-forces" in _ids:
    _c = S.pop(_ids.index("method-calib"))
    S.insert([x.split(chr(34))[1] for x in S].index("results-forces") + 1, _c)
_ids = [x.split(chr(34))[1] for x in S]
if "method-disc" in _ids and "results-defects" in _ids:
    _d = S.pop(_ids.index("method-disc"))
    S.insert([x.split(chr(34))[1] for x in S].index("results-defects") + 1, _d)
toc = "".join(f'<li><a href="#{s.split(chr(34))[1]}">{s.split("<h2>")[1].split("<")[0].strip()}</a></li>' for s in S if "<h2>" in s)
page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jackknife-conformal uncertainty for linear ACE potentials</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }}, svg: {{ fontCache: 'global' }} }};</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"></script>
<style>
:root {{
  /* manuscript draft: one serif reading column with a sticky contents rail; status chips mark how finished each section is */
  --bg: #fbfbf9; --paper: #ffffff; --ink: #1b1d1f; --muted: #5d6268; --rule: #e1e2dd; --accent: #2b5d7d;
  --draft: #3c7a52; --outline: #9a6a10; --partial: #8a4fa0;
  --serif: "Source Serif 4", Georgia, serif; --sans: "Source Sans 3", system-ui, sans-serif; color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #121416; --paper: #191c1f; --ink: #e5e7e9; --muted: #a0a6ad; --rule: #2c3136; --accent: #79aed0; --draft: #73c08f; --outline: #e0b04f; --partial: #c592d6; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #121416; --paper: #191c1f; --ink: #e5e7e9; --muted: #a0a6ad; --rule: #2c3136; --accent: #79aed0; --draft: #73c08f; --outline: #e0b04f; --partial: #c592d6; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.62 var(--serif); }}
.layout {{ display: grid; grid-template-columns: 230px minmax(0, 760px); gap: 40px; justify-content: center; padding: 32px 20px 80px; }}
nav {{ position: sticky; top: 24px; align-self: start; font: 14px/1.5 var(--sans); }}
nav ol {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }} nav a {{ color: var(--muted); text-decoration: none; }} nav a:hover, nav a:focus-visible {{ color: var(--accent); }}
@media (max-width: 900px) {{ .layout {{ grid-template-columns: minmax(0, 1fr); }} nav {{ position: static; }} }}
main {{ display: grid; gap: 34px; }}
header {{ display: grid; gap: 10px; border-bottom: 1px solid var(--rule); padding-bottom: 20px; }}
h1 {{ font-size: 32px; line-height: 1.2; margin: 0; font-weight: 600; text-wrap: balance; }}
h2 {{ font-size: 23px; margin: 0 0 10px; font-weight: 600; text-wrap: balance; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }}
h3 {{ font: 600 16px/1.4 var(--sans); margin: 18px 0 6px; letter-spacing: .01em; }}
p, li {{ max-width: 70ch; }} p {{ margin: 0 0 10px; }} ul, ol {{ margin: 0 0 10px; padding-left: 22px; }} li {{ margin-bottom: 5px; }}
.byline, .note, figcaption, caption {{ font: 14px/1.5 var(--sans); color: var(--muted); }}
.result {{ border-left: 3px solid var(--accent); padding-left: 12px; }}
.status {{ font: 600 11px/1 var(--sans); text-transform: uppercase; letter-spacing: .06em; padding: 4px 8px; border-radius: 999px; border: 1px solid currentColor; }}
.author {{ font-size: 19px; margin: 6px 0 0; }} .affil {{ font-size: 15px; color: var(--muted); margin: 2px 0 10px; }}
.status.draft, .status[class*="drafted"] {{ color: var(--draft); }} .status.outline {{ color: var(--outline); }} .status.partial {{ color: var(--partial); }}
.wrap {{ overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; margin: 8px 0 14px; }}
table {{ border-collapse: collapse; width: 100%; font: 14px/1.4 var(--sans); }} caption {{ text-align: left; padding: 8px 10px 0; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); }} th {{ color: var(--muted); font-weight: 600; }}
td.n {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }} tr:last-child td {{ border-bottom: 0; }}
figure {{ margin: 12px 0; background: #fcfcfb; border: 1px solid var(--rule); border-radius: 6px; padding: 10px; display: grid; gap: 6px; }}
figure img {{ max-width: 100%; height: auto; }} figure figcaption {{ color: #52514e; }}
code {{ font-size: 14px; }} .refs li {{ font-size: 15px; }}
mjx-container {{ overflow-x: auto; overflow-y: hidden; max-width: 100%; }}
</style></head>
<body><div class="layout">
<nav aria-label="Contents"><ol>{toc}</ol></nav>
<main>
<header>
<h1>Jackknife-conformal uncertainty for linear ACE potentials: calibrated forces and silicon properties compared with Bayesian and POPS baselines</h1>
<p class="author">James R. Kermode</p>
<p class="affil">Warwick Centre for Predictive Modelling, School of Engineering, University of Warwick, Coventry CV4 7AL, United Kingdom</p>
<p class="affil"><a href="paper.pdf">PDF (J. Chem. Phys. REVTeX build of this version)</a></p>

</header>
{''.join(S)}
</main></div></body></html>
"""
# number tables and figures in order of appearance (sections are reordered above; labels in the source are stable ids)
import re as _re
_num = {}
for _kind, _pat in (("Table", r"<caption>Table ([0-9]+[a-z]?)\."), ("Figure", r"<figcaption>Figure ([0-9]+[a-z]?)\.")):
    for _n, _lab in enumerate(dict.fromkeys(_re.findall(_pat, page)), 1):
        _num[(_kind, _lab)] = str(_n)
_ref = lambda kind, lab: _num.get((kind, lab), lab)
page = _re.sub(r"\b(Table|Figure)(s?) ([0-9]+[a-z]?(?:(?:, | and |–)[0-9]+[a-z]?)*)\b",
               lambda m: m.group(1) + m.group(2) + " " + _re.sub(r"[0-9]+[a-z]?", lambda n: _ref(m.group(1), n.group(0)), m.group(3)), page)
_bad = [m.group(0) for m in _re.finditer(r"\b(?:Tables?|Figures?) [0-9]+[a-z]\b", page)]
assert not _bad, f"unnumbered table/figure references: {_bad}"
(OUT / "index.html").write_text(page)
for p in FIG.glob("*.png"):
    shutil.copy(p, OUT / p.name)
print("ok", OUT)
