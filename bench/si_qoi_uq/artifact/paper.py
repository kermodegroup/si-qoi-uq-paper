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
models = sorted({r["model"] for r in cov}, key=lambda m: ("o4d20", "o3d12").index(m))
get = lambda m, meth, sc: next((r for r in cov if r["model"] == m and r["method"] == meth and r["scale"] == sc), None)
lam = {m: {r["kind"]: r["rms_z"] for r in sd if r["model"] == m and r["factor"] == "cr3" and r["breakdown"] == "all"}
       for m in models}

# the four methods of Figures 1 and 1b (figures.PANELS), with the same labels; variants in the sensitivity table
TAB = [("CR3-headline", "lam_F", "CR3 × λ<sub>F</sub> (headline)"), ("B1-headline", "q_b1", "B1: posterior × q̂"),
       ("B2lev5-committee", "lam_1", "B2: POPS (leverage cut 5 %)"), ("B3-headline", "lam_1", "B3: CR3, λ = 1")]
TAB_SENS = [("CR3-headline", "lam_V", "CR3 × λ<sub>V</sub>"), ("CR3-headline", "lam_E", "CR3 × λ<sub>E</sub>"),
            ("CR3-headline", "lam_pool", "CR3 × λ<sub>pool</sub>"),
            ("CR3-committee-lamF-physical", "lam_F", "CR3 at λ<sub>F</sub>, full re-run, physical members"),
            ("B2lev50-committee", "lam_1", "POPS, leverage cut 50 %")]
_cov_row = lambda m, lab, r: (
    f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r['n'])}</td><td class=n>{int(r['k68'])} [{f(r['ci68_lo'])}, {f(r['ci68_hi'])}]</td>"
    f"<td class=n>{int(r['k95'])} [{f(r['ci95_lo'])}, {f(r['ci95_hi'])}]</td><td class=n>{f(r['rms_z'])}</td>"
    f"<td class=n>{f(r['median_sigma_over_tol'])}</td></tr>")
sens_rows = "".join(_cov_row(m, lab, r) for m in models for meth, sc, lab in TAB_SENS for r in [get(m, meth, sc)] if r)
cov_rows = "".join(
    f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r['n'])}</td><td class=n>{int(r['k68'])} [{f(r['ci68_lo'])}, {f(r['ci68_hi'])}]</td>"
    f"<td class=n>{int(r['k95'])} [{f(r['ci95_lo'])}, {f(r['ci95_hi'])}]</td><td class=n>{f(r['rms_z'])}</td>"
    f"<td class=n>{f(r['median_sigma_over_tol'])}</td></tr>"
    for m in models for meth, sc, lab in TAB for r in [get(m, meth, sc)] if r)
lam_rows = "".join(f"<tr><td>{m}</td>" + "".join(f"<td class=n>{f(lam[m].get(k))}</td>" for k in ("E", "F", "V", "pool"))
                   + f"<td class=n>{f(lam[m]['V'] / lam[m]['F'])}</td><td class=n>{f(lam[m]['E'] / lam[m]['F'])}</td></tr>"
                   for m in models)
zlist = lambda m: ", ".join(f"{float(r['z']):+.1f}" for r in _qrows if r["model"] == m and r["method"] == "CR3-headline"
                           and r["scale"] == "lam_F" and "interstitial" in r["qoi"])
jp = lambda m, name, kind: (lambda e: pfmt(e["p_value"]) if e else "?")(joint.get(m, {}).get(name, {}).get(kind, {}).get("scaled", {}).get("eps_0.001"))
jrows = ""
pfmt = lambda p_: f"{p_:.2g}" if p_ > 0 else "&lt; 1e-300"
for m in models:
    for name, jlab in (("CR3/lam_F", "CR3 × λ<sub>F</sub> (headline)"), ("B1/q_b1", "B1: posterior × q̂"),
                       ("B2lev5/lam_1", "B2: POPS (leverage cut 5 %)"), ("B3/lam_1", "B3: CR3, λ = 1"),
                       ("CR3/lam_V", "CR3 × λ<sub>V</sub> (sensitivity)")):
        for kind, jt in joint.get(m, {}).get(name, {}).items():
            e = jt["scaled"]["eps_0.001"]
            jrows += (f"<tr><td>{m}</td><td>{jlab}</td><td>{kind}</td><td class=n>{jt['scaled']['df']}</td>"
                      f"<td class=n>{f(e['mahalanobis'], 1)}</td><td class=n>{f(e['chi2_95'], 1)}</td><td class=n>{pfmt(e['p_value'])}</td></tr>")
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

import numpy as _np0
fc_cov = {}
for _m in ("o3d12", "o4d20"):
    _p = RES / f"force_calibration_{_m}.npz"
    if _p.exists():
        _z = _np0.load(_p, allow_pickle=True)
        _g = [float(_np0.mean(_z["score"][_z["group"] == g] <= _z["q"][_z["group"] == g])) for g in _np0.unique(_z["group"])]
        fc_cov[_m] = (float(_np0.mean(_z["score"] <= _z["q"])), min(_g), max(_g), 1 - float(_z["alpha"]), len(_z["score"]))
fc_txt = "; ".join(f"{m}: {c[0]:.3f} (groups {c[1]:.2f}–{c[2]:.2f})" for m, c in fc_cov.items()) or "[pending]"
fc_nom = f"{next(iter(fc_cov.values()))[3]:.2f}" if fc_cov else "0.90"
disc_l = json.load(open(RES / "discrepancy.json")) if (RES / "discrepancy.json").exists() else {}
disc_txt = "; ".join(f"{m}: {disc_l[m]['CR3-headline+disc/lam_F']['k95']}/{disc_l[m]['CR3-headline+disc/lam_F']['n']}"
                     for m in ("o3d12", "o4d20") if m in disc_l) or "[pending]"

_cv = {r["model"]: float(r["median_sigma_over_tol"]) for r in rd("coverage.csv") if r["method"] == "CR3-headline" and r["scale"] == "lam_F"}
tight_ratio = _cv["o3d12"] / _cv["o4d20"] if {"o3d12", "o4d20"} <= set(_cv) else float("nan")

sec("abstract", "Abstract", f"""
<p>Machine-learned interatomic potentials are misspecified, so Bayesian posterior variances understate their errors. For linear
atomic cluster expansion (ACE) potentials we derive calibrated uncertainties from one misspecification-robust parameter
covariance, the delete-one-configuration jackknife. Scaled per group of local environments by conformal quantiles, it gives
per-atom force uncertainties that reach their target coverage on held-out silicon data. With one scale fitted to held-out forces,
the same parameter distribution is propagated to 21 properties of silicon, from elastic constants to defect and surface energies,
by implicit differentiation through each relaxation and by committees, and is tested against DFT without tuning on any property.
For a smaller and a larger model the intervals cover most properties, and the larger model's are about {tight_ratio:.0f} times
tighter. The exceptions are relaxed point defects, whose training cells are all hot snapshots: a bias that every configuration
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
<p>The scale comes from Mondrian conformal prediction [Vovk et al. 2003; Vovk, Gammerman and Shafer 2005]. Conformal prediction
turns the scores of held-out calibration points into a quantile that covers a new point with a guaranteed probability. The Mondrian
variant splits the points into categories and calibrates each category on its own, so that coverage holds within each category and
not only on average; the name refers to the rectangles of Piet Mondrian's paintings. Our categories are groups of local
environments, defined by bands of shell distortion and by coordination. Because the atoms of one configuration are correlated, the
quantiles weight configurations, not atoms [Dunn, Wasserman and Ramdas 2023]. They come from a stratified hold-out of the training
set, and a fitted transfer exponent carries them to the fit on all the data. The open-source ace-jax package [ace-jax] provides
these uncertainties for every predicted force: a standard deviation, a 3×3 covariance and a conformal region (Sections 2.5 and
3.2).</p>
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
<p>We also show where calibrated parameter-space uncertainty fails, and why. A bias that all training configurations of one kind
share is invisible to any delete-one jackknife; in the GAP-18 database every defect cell is a hot molecular-dynamics snapshot. We
diagnose this bias, measure it from the training labels alone, and add it to the defect QoIs as a discrepancy term, which
leaves the fit unchanged.</p>
<p>Section 2 sets out the methodology: the linear ACE fit and ARD (2.1–2.3), the CR3 shape (2.4), the per-atom force calibration
(2.5), the baselines (2.6) and the propagation to QoIs (2.7). Section 3 gives the validation statistics and the results: the per-atom forces first, then the global scale that the
propagation uses, then the QoIs. Section 4 discusses them and Section 5 concludes.</p>""", "draft")

_par = {p_["size"]: p_ for p_ in json.load(open(RES / "ace_pareto.json"))} if (RES / "ace_pareto.json").exists() else {}
if {"o3d12", "o4d20", "o4d12", "o4d14"} <= set(_par):
    _a, _b = _par["o3d12"], _par["o4d20"]
    pareto_par = f"""<p><b>The two models.</b> We study two members of the linear ACE ladder of the ace-jax silicon accuracy–cost benchmark,
fitted to the GAP-18 database [Bartók et al. 2018] (Figure 0). A model oNdM has correlation order N and total polynomial degree M.
o4d20 ({_b['n_params']} coefficients) is the only size of the ladder that meets both of the benchmark's targets, a test force RMSE
no larger than GAP-18's 0.0825 eV/Å on this split (o4d20: {_b['rmse_F']:.4f} eV/Å) and a QoI error, the geometric mean of
|error|/tolerance over the silicon QoIs, below 0.5 (o4d20: {_b['qoi']:.2f}). o3d12 ({_a['n_params']} coefficients) is {_b['cost_cpu'] / _a['cost_cpu']:.0f}× cheaper on a CPU node and
{_b['cost_gpu'] / _a['cost_gpu']:.0f}× cheaper on an A100, with
about twice the force error ({_a['rmse_F']:.3f} eV/Å) and a typical QoI error of one tolerance ({_a['qoi']:.2f}). The pair spans
the useful range of the ladder: an accurate model whose intervals should be narrow, and a cheap one whose intervals must be wide to
be honest. The two columns of Figure 0 also rank the sizes differently: o4d12 has a larger force error than o4d14 but a smaller QoI
error ({_par['o4d12']['qoi']:.2f} against {_par['o4d14']['qoi']:.2f}). Force accuracy alone does not predict QoI accuracy, which is why the
uncertainty is propagated to the QoIs themselves.</p>
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
<p>Neither fit is degenerate. For o4d20 every \(a_k\) is interior, and for o3d12 only the two-body precision sits on
the floor. ARD weakens the prior relative to the BLR's single \(\sigma_c\) on most body orders (on all of them for o3d12; for
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
<h3>2.5 Calibrated per-atom forces: the Mondrian conformal scale</h3>
<p>This section describes the per-atom force uncertainty introduced in this work (Contribution A), as implemented in ace-jax.
Every predicted force gets a calibrated uncertainty. It is a shape times a per-group scale. The shape is the
per-atom 3×3 block of the CR3 covariance, \(V(x)_{\alpha\beta}=u_\alpha^{\mathsf T}RR^{\mathsf T}u_\beta\) with
\(u_\alpha=D^{-1}B_{F,i\alpha}^{\mathsf T}\) built from the atom's three force rows (\(\alpha\in\{x,y,z\}\)), and \(v=\operatorname{tr}V\). The scales are fitted
per Mondrian group on held-out training configurations.</p>
<p><b>Groups.</b> Three constants are fixed from the training set at fit time: \(r_1\), the first minimum of the radial
distribution after its first peak; \(z^\star\), the modal coordination; and band edges at the p50, p90 and p99 quantiles of the
shell distortion. For each atom, with \(r_{ij}=|r_j-r_i|\), the coordination is \(z_i=\#\{j:r_{ij}\lt r_1\}\) and the shell distortion is
\(d_i=\operatorname{std}\{r_{ij}:r_{ij}\lt r_1\}/\operatorname{mean}\{r_{ij}:r_{ij}\lt r_1\}\) (not a number if \(z_i\lt 2\)). The group of atom \(i\) is
\(g_i=(\text{band}(d_i),\,[z_i\neq z^\star])\), the distortion band crossed with an off-coordination flag: 8 groups, merged with a neighbouring band when fewer than 20 configurations populate
them. The groups depend on geometry only, never on labels or scores.</p>
<p><b>Hold-out.</b> Training configurations are split 80/20, stratified by their most extreme group, into
\(T_{\text{fit}}\) and \(T_{\text{val}}\). The posterior \(P_{\text{fit}}\) on \(T_{\text{fit}}\), with its own CR3 shape
\(V_{\text{fit}}\), predicts \(T_{\text{val}}\), so both the force error \(e_i=F_i^{\text{DFT}}-F_i^{\text{fit}}\in\mathbb R^3\) and the shape are out of sample. The anisotropic score
(the default) is</p>
\[ s_i=\sqrt{e_i^{\mathsf T}\big(V_{\text{fit}}(x_i)+\epsilon_i I\big)^{-1}e_i},\qquad \epsilon_i=10^{-3}\,\tfrac13\operatorname{tr}V_{\text{fit}}(x_i), \]
<p>and \(s/\lambda\sim\chi_3\) (the chi distribution with three degrees of freedom) under \(e\sim\mathcal N(0,\lambda^2V)\). The isotropic alternative is \(|e|/\sqrt{v/3}\).</p>
<p><b>Transfer to the full training set.</b> The final posterior is fitted on all of \(T\), about 25 % more data than
\(P_{\text{fit}}\). A second hold-out fit \(P_{\text{fit2}}\) on a stratified subset of \(T_{\text{fit}}\) scores the same
\(T_{\text{val}}\) atoms. With \(N\), \(N_{\text{fit}}\), \(N_{\text{fit2}}\) the numbers of
configurations behind the final, first and second fits, and \(\lambda_1,\lambda_2\) the pooled RMS scores of \(P_{\text{fit}}\) and \(P_{\text{fit2}}\),
\(\beta=\operatorname{clip}\big(\log(\lambda_1/\lambda_2)/\log(N_{\text{fit}}/N_{\text{fit2}}),0,\tfrac12\big)\) and every score is
multiplied by \(f_{\text{tr}}=(N/N_{\text{fit}})^\beta\). \(\beta=0\) is the variance-dominated limit and \(\tfrac12\) the bias-dominated one.</p>
<p><b>Scales.</b> Each calibration configuration carries total weight 1 in every group it populates (atom weight \(1/n_{c,g}\), with \(n_{c,g}\) the atoms of configuration \(c\) in group \(g\), and \(n_{\text{cfg},g}\)
the calibration configurations with at least one atom in \(g\)).
The RMS scale and the conformal quantile of group \(g\) are</p>
\[ (\lambda_g^{\text{rms}})^2=\frac{1}{3\,n_{\text{cfg},g}}\sum_{c}\frac{1}{n_{c,g}}\sum_{i\in c\cap g}s_i^2,\qquad
   q_g=\inf\{s:\hat F_g(s)\ge1-\alpha\}, \]
<p>where \(\hat F_g\) is the configuration-weighted score distribution with a test point at \(+\infty\) (the pooled-CDF
construction of Dunn, Wasserman and Ramdas), and \(1-\alpha\) the target coverage. The default is \(\alpha=0.1\).</p>
<p><b>Per-atom outputs.</b> The force standard deviation \(\sigma_{F,i}=\lambda_g^{\text{rms}}\sqrt v\) and covariance
\(C_{F,i}=(\lambda_g^{\text{rms}})^2V\) are the Gaussian reading, with \(g\) the atom's group. For a force error \(\Delta F\), the conformal region is \(\{\Delta F:\Delta F^{\mathsf T}(V+\epsilon I)^{-1}\Delta F\le q_g^2\}\), and its largest semi-axis gives a single size for it. Assuming exchangeability of calibration and target configurations within a group,
the region covers a uniformly chosen atom of the group in a new configuration with probability \(\approx1-\alpha\); the
finite-sample correction is set by the number of configurations, not atoms. The ratio
\(\varrho_g=q_g/(\lambda_g^{\text{rms}}\chi_3^{-1}(1-\alpha))\), with \(\chi_3^{-1}\) the \(\chi_3\) quantile function, measures tails heavier than
Gaussian (\(\varrho_g=1\) for Gaussian errors). A covariate-shift diagnostic
(a density ratio on a PCA of the site descriptor, giving a weighted quantile) flags atoms outside the calibration support,
but does not rescale σ.</p>
<p><b>Why the propagation does not use these scales.</b> λ<sub>g</sub> and q<sub>g</sub> temper force <em>outputs</em> atom by
atom. They do not define a covariance in θ, and a QoI depends on every atom and on relaxed geometries. Propagation therefore uses one
global λ (Section 3.3). Section 3.2 tests these per-atom force uncertainties directly. Energy and virial uncertainties use the unscaled posterior
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
Perez [2024]. The ace-jax default builds the box from every row. Section 3.7 also reports boxes built only from rows above a
leverage percentile, because the low-leverage rows make the default box unphysical. B2 has no calibration step, so it tests a
misspecification-aware shape without a fitted scale.</p>
<p>The third baseline, B3, is the CR3 covariance with \(\lambda=1\). It isolates the effect of the scale: the difference between B3
and the headline is entirely the calibration of Section 3.3.</p>
<p>Fisher et al. [2026] take the conformal step further. They calibrate joint conformal sets for energies, forces and virials
(Bonferroni-corrected boxes, Mahalanobis ellipsoids, and losses under conformal risk control) and propagate them to derived
quantities such as elastic constants and vacancy formation energies, where modelling the correlations captures error
cancellation. B1 and our CR3 method use one scalar scale instead. We do not use multivariate conformal prediction here; combining
it with the CR3 shape is left to future work.</p>""", "drafted from docs", level=3)

sec("method-calib", "3.3  One global scale for propagation", r"""
<p>The per-atom scales \(\lambda_g\) of Section 3.2 calibrate force outputs atom by atom, but they do not define a covariance in θ
(Section 2.5). Propagation needs a single scale \(\lambda\) for the CR3 distribution \(\theta\sim\mathcal N(\hat\theta,\lambda^2LL^{\mathsf T})\).
We fit it on the same 248 held-out test configurations as Section 3.2, with the same CR3 shape, but on the BLR fits that the QoI
benchmark uses and over all rows of a type at once, not per environment group. Every row \(b\) gives a standardised score
\(z=(y-\hat y)/\lVert L^{\mathsf T}b\rVert\), with \(\hat y=b\cdot\hat\theta\) (energies per atom, force components, virial components). The
scale for observable type \(q\in\{E,F,V\}\) is \(\lambda_q=\operatorname{RMS}(z_q)\) over that type's rows; \(\lambda_{\text{pool}}\) weights E, F
and V equally within a configuration and configurations equally. As in Section 3.2, rows that vanish by site symmetry
(\(\lVert L^{\mathsf T}b\rVert\lt 10^{-3}\) of the median of their kind) are dropped. No QoI, and no DFT QoI value, enters any scale.
The headline uses \(\lambda_F\): forces are most of the training rows, and they are the quantity calibrated per atom in Section 3.2.
In the Gaussian reading of Figure 7, \(\lambda_F\) is the global line, against the per-group \(\lambda_g\).</p>
<p>A single scale is a compromise. Figure 5 breaks the test-split scale down by configuration type: it ranges from about 1 to 15
across types and observables, largest for the virials of the bt cells, the vacancy and interstitial cells and, for o3d12, the
hexagonal-diamond forces, and smallest for the energies of the crystal phases. The ratios of the global scales (Table 2) show the same at the level of observables.</p>
<figure><img src="fig5_scale_diagnostics.png" alt="Test-split scale per observable and configuration type, both models"><figcaption>Figure 5.
Test-split scale λ = RMS z with the CR3 shape, per observable (energy, force, virial) and GAP-18 configuration type, for both models
("all": the global λ<sub>E</sub>, λ<sub>F</sub>, λ<sub>V</sub> of Table 2). Missing bars: types without that observable.</figcaption></figure>
<div class="wrap"><table><caption>Table 2. Test-split scales (CR3 shape).</caption>
<thead><tr><th>model</th><th>λ<sub>E</sub></th><th>λ<sub>F</sub></th><th>λ<sub>V</sub></th><th>λ<sub>pool</sub></th><th>λ<sub>V</sub>/λ<sub>F</sub></th><th>λ<sub>E</sub>/λ<sub>F</sub></th></tr></thead>
<tbody>""" + lam_rows + r"""</tbody></table></div>
<p>The three scales differ: λ<sub>V</sub> is about a third larger than λ<sub>F</sub> (λ<sub>V</sub>/λ<sub>F</sub> = 1.33–1.38, outside
[0.8, 1.25]), and λ<sub>E</sub> about a sixth smaller (0.82–0.84, just inside), so the degree of misspecification depends on the
observable. With λ<sub>F</sub>, virials are therefore
under-covered on the test split. The QoIs most like virials are the elastic constants, which are derivatives of the stress, and the
phase energies, which come from energy–volume curves, so their intervals at λ<sub>F</sub> may be too narrow; Section 3.4 reports the
coverage with λ<sub>V</sub> as a sensitivity.</p>""", "drafted", level=3)

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
<p>Write the energy of a structure with coordinates \(x\) (positions, and cell where relaxed) as \(E=b_E(x)\cdot\theta+E^0(x)\), with
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

sec("method-disc", "Measuring the bias: a discrepancy term", r"""
<p>A bias shared by every training configuration of a kind is invisible to any delete-one jackknife: deleting one
configuration does not move the fit along it. The defect formation energies expose such a bias. Every training interstitial,
vacancy and divacancy cell is a hot MD snapshot, while the QoIs are relaxed defects.</p>
<p><b>Measuring the bias from training labels.</b> Each training defect snapshot \(i\) is paired with the bulk snapshot
nearest in volume and thermal amplitude, giving a defect–bulk contrast row \(d_i=b_{\text{def},i}-(n_{\text{def}}/n_{\text{b}})\,b_{\text{bulk},i}\)
and target \(y_i=y_{\text{def},i}-(n_{\text{def}}/n_{\text{b}})\,y_{\text{bulk},i}\), with \(b\) the energy rows, \(y\) the energy labels
and \(n_{\text{def}}\), \(n_{\text{b}}\) the atom counts of the two cells. Regressing the residual \(y_i-d_i\cdot\hat\theta\) on
\(\eta_i=F_{\text{rms},i}^2\), the mean squared DFT force component of the defect cell (proportional to the thermal mean-square
displacement for a harmonic solid), and extrapolating to \(\eta\to0\) estimates the model's error for a relaxed defect. That intercept is
linear in the data, so it is a synthetic observation:</p>
\[ d^*_t=\sum_i a_i d_i,\qquad y^*_t=\sum_i a_i y_i,\qquad a=e_1^{\mathsf T}(X^{\mathsf T}X)^{-1}X^{\mathsf T},\quad X=[1,\,\eta_i], \]
<p>with \(X\) the regression design (rows \([1,\eta_i]\)), \(e_1=(1,0)^{\mathsf T}\) selecting the intercept, residual
\(r^*_t=y^*_t-d^*_t\cdot\hat\theta\) and standard error \(s^*_t\) from the regression scatter, per defect class
\(t\in\{\text{interstitial},\text{vacancy},\text{divacancy}\}\). No QoI value or tolerance enters.</p>
<p><b>Why not in parameter space.</b> The minimum-norm correction that removes \(r^*_t\), \(\delta_t=\Sigma d^*_t r^*_t/(d^{*\mathsf T}_t\Sigma d^*_t)\), with \(\Sigma\) the propagated covariance
(the POPS construction applied to the synthetic row, in the CR3 or posterior metric), added as \(\Sigma\mathrel{+}=\delta_t\delta_t^{\mathsf T}\),
projects only 0.1–0.2 eV of a 1 eV \(r^*\) onto the relaxed-defect QoI rows. The extrapolated hot-snapshot contrast and the
relaxed defect's formation row point in different directions in θ, so coverage did not change.</p>
<p><b>Estimator.</b> The intercept can be formed with each defect snapshot's nearest bulk snapshot as reference (paired; the form
that is also the synthetic row above) or with a bulk thermal trend fitted over all bulk cells near the defect volume (regression).
The headline uses regression for interstitials (data-only evidence, Table 7c) and, for the vacancy, regression per training-cell
size (63 and 215 atoms) combined by inverse variance, because the two sizes disagree and one pooled trend is rejected (see The vacancy, below).</p>
<p><b>Discrepancy term.</b> For the QoIs of defect class \(t\), the variance becomes
\(\sigma_Q^2=\sigma_{Q,\text{method}}^2+r^{*2}_t+s^{*2}_t\), with \(\sigma_{Q,\text{method}}\) the σ of the UQ method without the term (interstitial class for the three interstitials; vacancy class for the
vacancy). Every other QoI is unchanged. This leaves the pure parameter-space propagation for these QoIs and is reported as a
separate, labelled method.</p>
<p>Nothing in this analysis is specific to the CR3 propagation. The bias r*<sub>t</sub> and its standard error come from the
training labels and the fitted mean alone, so the same term can be added to the σ<sub>Q</sub> of any method that shares that mean,
the baselines included.</p>""", "", level=4)

sec("method-valid", "3  Results", r"""
<h3>3.1 Validation statistics</h3>
<p>Section 2 ends with a σ for every prediction we test: a per-atom force covariance and conformal region (Section 2.5), a
scaled CR3 covariance in θ (Section 2.4), and from it a σ<sub>Q</sub> for each QoI by the delta method or by committees, with or
without the discrepancy term (Sections 2.7 and 2.8). The same few statistics judge all of them against held-out data or DFT. None
of them is used to fit any scale.</p>
<p>For the per-atom forces (Section 3.2), the test is the coverage of the conformal regions on the 248 test configurations,
overall and in each Mondrian group, together with the RMS of the observed force error over the predicted σ<sub>F,i</sub>, which is
1 for a calibrated Gaussian reading. For the QoIs (Sections 3.4–3.11), each prediction gives a standardised error
\(z_Q=(\hat Q-Q_{\text{DFT}})/\sigma_Q\), with \(\hat Q\) the mean model's value. We count how many of the 20 scored QoIs fall inside
the Gaussian 68 % and 95 % intervals, and give each count a Clopper–Pearson 95 % interval: with only 20 QoIs, any count from 17 to 20 is consistent with a nominal 95 % (16 or fewer has probability below 2 %
if the coverage is truly 95 %).</p>
<p>Counts treat the QoIs as independent, but they share the same coefficients, so their errors are correlated. The joint test
(Section 3.5) therefore compares the whole error vector \(\Delta\), the \(\hat Q-Q_{\text{DFT}}\) over the scored QoIs, with its
predicted covariance through the Mahalanobis distance \(\Delta^{\mathsf T}(C_Q+\epsilon I)^{-1}\Delta\). Here
\(C_Q=\lambda^2G\,LL^{\mathsf T}G^{\mathsf T}\) for the delta method, with \(G\) the QoI rows stacked, or the committee covariance. The
problem is first scaled to unit diagonal, \(\epsilon\) is \(10^{-3}\) of the mean diagonal (with \(10^{-4}\) and \(10^{-2}\) as
sensitivities), and the distance is compared with \(\chi^2\) at the numerical rank. A method can pass the per-QoI counts and still fail
this test, if its errors are correlated in a way its covariance does not predict.</p>
<p>Coverage alone rewards intervals that are too wide. We therefore also report sharpness, the median of \(\sigma_Q/\text{tol}_Q\)
with \(\text{tol}_Q\) the benchmark's tolerance for QoI \(Q\): a value below 1 means the interval is narrower than the accuracy the
benchmark asks for. A useful method has near-nominal coverage at the smallest sharpness.</p>""", "drafted")

def _phase_names(m, lo):
    """'the sh, β-Sn and hcp energies is 86, 43 and 24 times' for the phases whose physical/rescaled ratio is >= lo."""
    it = sorted(((float(r["ratio_physical_vs_rescaled"]), k) for (mm, k), r in phr.items() if mm == m and k.startswith("dE_")
                 and r["ratio_physical_vs_rescaled"] not in ("", None) and float(r["ratio_physical_vs_rescaled"]) >= lo), reverse=True)
    names = [_qn(k).replace(" phase energy", "").replace("the ", "") for _, k in it]
    nums = [f"{v:.0f}" for v, _ in it]
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    return f"the σ of the {join(names)} energies is {join(nums)} times"


pops = pw.get(("o3d12", "pops"))
pops_sd = f(pops['sd_E_meV_per_atom'] if pops else None, 0)
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
dtrend = json.load(open(RES / "defect_trend.json")) if (RES / "defect_trend.json").exists() else {}
disc = json.load(open(RES / "discrepancy.json")) if (RES / "discrepancy.json").exists() else {}
disc_rows = rd("discrepancy.csv")
disc_reg = json.load(open(RES / "discrepancy_regression.json")) if (RES / "discrepancy_regression.json").exists() else {}
disc_pair = json.load(open(RES / "discrepancy_paired.json")) if (RES / "discrepancy_paired.json").exists() else {}
btm = {m: json.load(open(RES / f"bias_term_{m}.json")) for m in ("o3d12", "o4d20") if (RES / f"bias_term_{m}.json").exists()}
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
<div class="wrap"><table><caption>Table 3. Coverage of the DFT values (count [95 % CI]) and sharpness, for the four methods of Figures 1 and 1b.</caption>
<thead><tr><th>model</th><th>method</th><th>n</th><th>68 %</th><th>95 %</th><th>RMS z</th><th>median σ/tol</th></tr></thead>
<tbody>{cov_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 3b. Sensitivity of the coverage to the scale and to the committee variant: the CR3 shape
with the energy, virial or pooled scale (Section 3.3), the full re-run at λ<sub>F</sub> keeping physical members only (Section 3.6),
and POPS with the 50 % leverage cut (Section 3.7).</caption>
<thead><tr><th>model</th><th>method</th><th>n</th><th>68 %</th><th>95 %</th><th>RMS z</th><th>median σ/tol</th></tr></thead>
<tbody>{sens_rows}</tbody></table></div>
<p>With the headline CR3 × λ<sub>F</sub>, {k95("o3d12", "CR3-headline", "lam_F")} QoIs of o3d12 and {k95("o4d20", "CR3-headline", "lam_F")}
of o4d20 lie inside their 95 % intervals (Table 3, Figure 1 top row, Figure 1b). Both counts are below the nominal 19, and o4d20's is
below the 17 that a true 95 % coverage makes plausible (Section 3.1). The misses are not spread across the QoIs. For o3d12 they are
{miss_txt("o3d12")}; for o4d20 they are {miss_txt("o4d20")}. The interstitial formation energies, missed by both
models, are the subject of Section 3.8: they share a bias that a delete-one jackknife cannot see, not a scale that is too small.
Elsewhere the 95 % intervals reach DFT, and the 68 % counts ({int(float(get("o3d12", "CR3-headline", "lam_F")["k68"]))} and
{int(float(get("o4d20", "CR3-headline", "lam_F")["k68"]))} of 20, against a nominal 13.6) are close to nominal.</p>
<p>The intervals follow the accuracy of the model. The median σ/tol is {sot("o4d20", "CR3-headline", "lam_F")} for o4d20 and
{sot("o3d12", "CR3-headline", "lam_F")} for o3d12, {tight:.1f}× wider: o4d20's intervals are mostly inside the benchmark tolerance, and
o3d12's are wider than it, as they should be for a model whose typical QoI error is about one tolerance (Figure 0). So one scale fitted
to held-out forces gives similar coverage for both models while the intervals shrink with model quality.</p>
<p>The baselines (Section 3.7) cover fewer QoIs at comparable or smaller widths. The sensitivity table (Table 3b) shows that the choice of
scale matters at the margin. The virial scale λ<sub>V</sub>, larger than λ<sub>F</sub> by about a third, covers
{k95("o3d12", "CR3-headline", "lam_V")} and {k95("o4d20", "CR3-headline", "lam_V")} at the price of wider intervals (median σ/tol
{sot("o3d12", "CR3-headline", "lam_V")} and {sot("o4d20", "CR3-headline", "lam_V")}); λ<sub>E</sub> and λ<sub>pool</sub> give about the
same as λ<sub>F</sub>. The full committee re-run at λ<sub>F</sub>, keeping only physical members, covers
{k95("o3d12", "CR3-committee-lamF-physical", "lam_F")} and {k95("o4d20", "CR3-committee-lamF-physical", "lam_F")}, against
{k95("o3d12", "CR3-headline", "lam_F")} and {k95("o4d20", "CR3-headline", "lam_F")} for the headline. The differences come mainly from the
phase energies, which are strongly nonlinear in θ at λ<sub>F</sub> (Section 3.6).</p>
<figure><img src="fig6_model_compare.png" alt="Per QoI, error over tolerance with 95 percent intervals for o4d20 and o3d12, DFT as crosses at zero; one row each for the headline and the three baselines"><figcaption>Figure 1. Model prediction minus DFT, in units of the benchmark tolerance, with the 95 % interval, for o4d20
(circles) and o3d12 (squares); DFT is the cross at 0 and the shaded band is ± 1 tolerance. The axis is linear within ± 1 and
logarithmic beyond, and is shared by all rows. Top row: the headline, CR3 × λ<sub>F</sub>; the larger model's intervals are
{tight:.1f}× tighter in median σ/tol while its coverage stays comparable. Below it the baselines of Section 2.6: B1, the posterior
scaled by one conformal quantile; B2, POPS with the 5 % leverage cut; B3, CR3 without calibration (λ = 1). Each row's legend gives
its 95 % coverage and median σ/tol.</figcaption></figure>
<figure><img src="fig2_coverage.png" alt="Coverage per method with binomial intervals"><figcaption>Figure 1b. Fraction of the 20 scored QoIs inside the 68 % and 95 % intervals for the four methods of Figure 1
(headline, B1, B2 with the 5 % leverage cut, B3), with 95 % Clopper–Pearson intervals; dashed lines are the nominal levels.</figcaption></figure>
<h3>3.5 Joint test</h3>
<div class="wrap"><table><caption>Table 4. Mahalanobis test (correlation-scaled, ε = 10<sup>−3</sup>) for the four methods of Figure 1, and CR3 × λ<sub>V</sub>
as a sensitivity. "Covariance from": the delta method or the committee.</caption>
<thead><tr><th>model</th><th>method</th><th>covariance from</th><th>df</th><th>d²</th><th>χ²<sub>95</sub></th><th>p</th></tr></thead>
<tbody>{jrows}</tbody></table></div>
<p>The QoIs share one coefficient vector, so their errors are correlated, and Figure 3 shows how. In the CR3 committee for o4d20 the
correlations form blocks: the elastic constants with each other and with the β-Sn and sh phase energies, which are both compressed
diamond-like structures; and the four point defects with each other (0.3–0.8), most strongly the three interstitials. Most other pairs
are only weakly correlated.</p>
<p>The joint test (Table 4) asks whether the whole error vector is consistent with this covariance. For o4d20 the headline passes
(p = {jp("o4d20", "CR3/lam_F", "committee")} with the committee covariance, {jp("o4d20", "CR3/lam_F", "delta")} with the delta method),
even though five QoIs miss individually. Three of the five are the interstitials, which the covariance expects to move together, so
jointly they weigh less than three independent misses. For o3d12 the headline fails
(p = {jp("o3d12", "CR3/lam_F", "committee")} and {jp("o3d12", "CR3/lam_F", "delta")}), although it covers more QoIs one at a time. Its
three interstitials miss by z = {zlist("o3d12")}, all in the same direction; we have not separated their share of d² from the rest. The virial scale passes for both
models (p from {jp("o3d12", "CR3/lam_V", "delta")} to {jp("o4d20", "CR3/lam_V", "committee")}). Every baseline fails, by many orders of
magnitude for B1 and B3. So does B2 for o3d12, which covers 15 of 20 QoIs one at a time: per-QoI coverage alone does not show that
a covariance is right. The delta-method and committee covariances agree, and the outcomes do not change for ε from
10<sup>−4</sup> to 10<sup>−2</sup> of the mean diagonal. The discrepancy term of Section 3.8 is not included here.</p>
<figure><img src="fig3_qoi_correlation.png" alt="QoI correlation matrix of the o4d20 CR3 committee"><figcaption>Figure 3. Correlation
between the QoIs in the CR3 committee for o4d20 (K = 200 members).</figcaption></figure>
<h3>3.6 Delta method, committees and nonlinearity</h3>
<p>The two routes to σ<sub>Q</sub> agree where both apply. At λ = 1, the delta-method σ is within 10 % of the committee's for all but
{nflag["o4d20"]} of the {ncmp["o4d20"]} QoIs of o4d20 and {nflag["o3d12"]} of the {ncmp["o3d12"]} of o3d12; the ratio ranges from
{dvc_rng["o4d20"][0]:.2f} to {dvc_rng["o4d20"][1]:.2f} for o4d20 and from {dvc_rng["o3d12"][0]:.2f} to {dvc_rng["o3d12"][1]:.2f} for o3d12.
The flagged QoIs ({", ".join(_qn(k) for k in dvc_flag["o4d20"])} for o4d20; {", ".join(_qn(k) for k in dvc_flag["o3d12"])} for o3d12)
are taken from the committee, as Section 2.7.3 prescribes. The implicit differentiation through each relaxation, including that of the
lattice constant, is therefore accurate at the size of the CR3 distribution itself. The elastic constants are a stricter check,
because they can be computed by two routes, from the stress or from the curvature of the energy. Figure 4 shows that the two give
nearly the same covariance ellipses for both models, and that the DFT values lie inside the 1σ ellipses for o4d20 but well outside
them for o3d12, whose C<sub>11</sub> is about 15 % too low.</p>
<p>At λ<sub>F</sub> the distribution is several times wider, and the linear rescaling of the λ = 1 committee,
Q<sub>k</sub>(λ) = Q̂ + λ(Q<sub>k</sub> − Q̂), can fail. The full re-run at λ<sub>F</sub> tests it. Of its 200 members,
{phys["o4d20"]["accepted"]} (o4d20) and {phys["o3d12"]["accepted"]} (o3d12) pass the physicality rule fixed beforehand; the rest
collapse ({phys["o4d20"]["rejected_collapse"]} and {phys["o3d12"]["rejected_collapse"]}), and none fails a test. Outside the phase
energies, the σ of the physical members is {min(nonphase["o4d20"]):.2f}–{max(nonphase["o4d20"]):.2f}× the rescaled σ for o4d20 and
{min(nonphase["o3d12"]):.2f}–{max(nonphase["o3d12"]):.2f}× for o3d12, so the rescaling holds for the elastic constants, defects and
surfaces. The phase energies are different: for o4d20 {_phase_names("o4d20", 10)} the rescaled σ, and for o3d12
{_phase_names("o3d12", 10)} it. These are the phases furthest from diamond, whose structures relax into other phases when θ moves far
enough, so the phase-energy σ at λ<sub>F</sub> is a property of that relaxation path rather than of a Gaussian in θ. The headline
keeps the rescaled σ, and Table 3b reports the full re-run as a sensitivity.</p>
<figure><img src="fig4_elastic_ellipses.png" alt="Elastic constant covariance ellipses from the stress and energy routes, both models"><figcaption>Figure 4.
Elastic constants: the 1σ covariance ellipses of (C<sub>11</sub>, C<sub>12</sub>) and (C<sub>11</sub>, C<sub>44</sub>) at λ<sub>F</sub> from
the delta method, by the stress route (solid) and the energy route (dashed), for o4d20 (top) and o3d12 (bottom); the dot is the
mean model's value and the star DFT.</figcaption></figure>
<h3>3.7 Baselines</h3>
<p>The three baselines of Section 2.6 each fall short in a different way (Figure 1, Figure 1b and Table 3). B3, the CR3 shape without
calibration, covers only {b3txt} of the QoIs at 95 %: the shape alone is several times too narrow, as the test-split scales of
Section 3.3 already show. B1, the posterior with one conformal scale, covers {b1txt}. It has a calibrated overall size but the wrong
shape, so it is too wide for some QoIs and too narrow for others, and it fails the joint test outright (Table 4).</p>
<p>B2, POPS, needed more care. With ace-jax's default, which builds the hypercube from every training row, the members are
unphysical: the bounds are set by a handful of rows with near-zero leverage, whose corrections are unbounded, and the members spread
the bulk energy of o3d12 by {pops_sd} meV/atom, so that almost every relaxation fails. Rejecting unphysical members cannot rescue
this. Every axis of the full box spans 10<sup>6</sup>–10<sup>7</sup> eV/atom of bulk energy, and no member of either model comes within
5 eV/atom of the mean on rattled diamond cells (0 of 200).</p>
<p>Dropping the lowest-leverage rows from the box fixes it. A sweep of the cut (the lowest 1, 2, 5, 10, 25 or 50 % of rows, members
screened on three rattled diamond cells) shows that dropping 1 % already gives physical members. The width then hardly depends on the
cut for o4d20: the median worst-cell energy change is 0.001–0.003 eV/atom, against 0.017 for CR3 × λ<sub>F</sub>. We ran committees at
the 5 % cut, the closest to POPS's own default, and at 50 % (Table 3b). Their bulk-energy spreads, against the CR3 members at λ = 1 and at
λ<sub>F</sub>, are {pops_w} meV/atom. B2 covers {pops_cov}. POPS has no calibration step, so its width is whatever the choice of support rows
gives: too narrow for o4d20, and of roughly the right size but in the wrong places for o3d12.</p>""", "drafted", level=3)

trend_rows = ""
for m in ("o3d12", "o4d20"):
    tr_ = dtrend.get(m, {}).get("interstitial")
    errs = [qrow(m, k) for k in ("dumbbell_interstitial_energy", "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy")]
    if tr_ and all(errs):
        qerrs = " / ".join(f"{e['dft'] - e['mean']:+.2f}" for e in errs)
        trend_rows += (f"<tr><td>{m}</td><td class=n>{tr_['defect_intercept_eV']:+.2f} ± {tr_['defect_intercept_se']:.2f}</td>"
                       f"<td class=n>{qerrs}</td>"
                       f"<td class=n>{tr_['defect_mean_eV']:+.2f}</td></tr>")
rstar_rows = "".join(
    f"<tr><td>{m}</td><td>{t}</td><td class=n>{v['r_star_eV']:+.3f} ± {v['se_eV']:.3f}</td><td class=n>{v['cr3_sigma_lamF_eV']:.3f}</td>"
    f"<td class=n>{v['n_snapshots']}</td></tr>" for m, b in btm.items() for t, v in b["synthetic_rows"].items())
dcov_rows = ""
for m in ("o4d20", "o3d12"):
    for meth, sc, lab in (("CR3-headline", "lam_F", "CR3 × λ<sub>F</sub>"), ("CR3-headline", "lam_V", "CR3 × λ<sub>V</sub>"),
                          ("B1-headline", "q_b1", "B1: posterior × q̂")):
        r0, r1 = get(m, meth, sc), disc.get(m, {}).get(f"{meth}+disc/{sc}")
        if r0 and r1:
            dcov_rows += (f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r0['k95'])}/{int(r0['n'])}</td>"
                          f"<td class=n>{r1['k95']}/{r1['n']} [{r1['ci95'][0]:.2f}, {r1['ci95'][1]:.2f}]</td>"
                          f"<td class=n>{int(r0['k68'])} → {r1['k68']}</td><td class=n>{f(r1['median_sigma_over_tol'])}</td></tr>")
dz_rows = "".join(
    f"<tr><td>{r['model']}</td><td>{dict(DEF_Q)[r['qoi']]}</td><td class=n>{r['dft'] - r['mean']:+.3f}</td>"
    f"<td class=n>{r['sigma_method']:.3f}</td><td class=n>{r['sigma']:.3f}</td><td class=n>{r['z_without']:+.2f} → {r['z']:+.2f}</td></tr>"
    for r in disc_rows if r["qoi"] in dict(DEF_Q) and r["method"] == "CR3-headline+disc" and r["scale"] == "lam_F")
models_disc = ", ".join(sorted(k for k in disc if k != "estimator")) or "none yet"
ec = json.load(open(RES / "estimator_check.json")) if (RES / "estimator_check.json").exists() else None
ec_rows = ""
if ec:
    for ty in ("interstitial", "divacancy", "vacancy"):
        c = ec["C_noise"][ty]
        cells = "".join(f"<td class=n>{ec['B_cold_holdout'][m][ty]['rmse_paired_eV']:.2f} / {ec['B_cold_holdout'][m][ty]['rmse_regression_eV']:.2f}</td>"
                        for m in ("o3d12", "o4d20"))
        ses = " · ".join(f"{c[m]['se_paired']:.3f} / {c[m]['se_regression']:.3f}" for m in ("o3d12", "o4d20"))
        a_ = ec["A_model_vs_model"].get(ty)
        mvm = (f"{' / '.join(f'{q:.2f}' for q in a_['truth_Q_o4d20_minus_Q_o3d12_eV'])}: {a_['paired_eV'][0]:+.2f} / {a_['regression_eV'][0]:+.2f}"
               if a_ else "–")
        ec_rows += (f"<tr><td>{ty}</td><td class=n>{c['ratio_quantiles'][1]:.0f}</td><td class=n>{c['o4d20']['partner_term_rms_eV']:.2f}</td>"
                    f"<td class=n>{ses}</td>{cells}<td class=n>{mvm}</td></tr>")
vss = json.load(open(RES / "vacancy_size_split.json")) if (RES / "vacancy_size_split.json").exists() else {}
vck = json.load(open(RES / "vacancy_check.json")) if (RES / "vacancy_check.json").exists() else {}
EST_LAB = {"interstitial": "regression (pooled)", "vacancy": "regression per cell size, inverse variance"}
r7a_rows = "".join(f"<tr><td>{m}</td><td>{t}</td><td>{EST_LAB.get(t, '')}</td><td class=n>{v['r_star_eV']:+.3f} ± {v['se_eV']:.3f}</td></tr>"
                   for m in ("o3d12", "o4d20") for t, v in disc.get(m, {}).get("r_star", {}).items())
vac_rows = ""
for m in ("o3d12", "o4d20", "model_vs_model (o4d20 - o3d12)"):
    if m in vss:
        d_ = vss[m]
        lab = "model vs model (o4d20 − o3d12)" if m.startswith("model") else m
        vac_rows += (f"<tr><td>{lab}</td><td class=n>{d_['per_size']['63'][0]:+.3f} ± {d_['per_size']['63'][1]:.3f}</td>"
                     f"<td class=n>{d_['per_size']['215'][0]:+.3f} ± {d_['per_size']['215'][1]:.3f}</td><td class=n>{d_['heterogeneity_z']:.1f}</td>"
                     f"<td class=n>{d_['inverse_variance'][0]:+.3f} ± {d_['inverse_variance'][1]:.3f}</td></tr>")
cold_txt, geo_txt = "", ""
if vck:
    cc = {m: next(o for o in vck["relax_cold"][m] if o["size"] == 63 and o["x"] < 0.05) for m in ("o3d12", "o4d20")}
    cold_txt = ", ".join(f"{m}: relaxation gain {cc[m]['model_gain_eV']:.2f} eV against a DFT bound of {cc[m]['dft_gain_bound_eV']:.2f} eV" for m in cc)
    geo_txt = ", ".join(f"{m} {min(g['neighbour_pair_distances_A']):.2f}–{max(g['neighbour_pair_distances_A']):.2f} Å" for m, g in vck["geometry"].items())
est_rows = ""
for m in ("o4d20", "o3d12"):
    for sc_, lab in (("lam_F", "λ<sub>F</sub>"), ("lam_V", "λ<sub>V</sub>")):
        r0 = get(m, "CR3-headline", sc_)
        h_, a_, b_ = (d.get(m, {}).get(f"CR3-headline+disc/{sc_}") for d in (disc, disc_pair, disc_reg))
        if r0 and h_ and a_ and b_:
            ra, rb = disc_pair[m]["r_star"]["interstitial"]["r_star_eV"], disc_reg[m]["r_star"]["interstitial"]["r_star_eV"]
            est_rows += (f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r0['k95'])} ({int(r0['k68'])})</td>"
                         f"<td class=n>{h_['k95']} ({h_['k68']})</td><td class=n>{a_['k95']} ({a_['k68']})</td>"
                         f"<td class=n>{b_['k95']} ({b_['k68']})</td><td class=n>{ra:+.2f} / {rb:+.2f}</td></tr>")
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
<div class="wrap"><table><caption>Table 6. Synthetic zero-temperature contrast rows (DFT training labels only): residual r*
± standard error at the unchanged fit, against the CR3 × λ<sub>F</sub> σ of the same row.</caption>
<thead><tr><th>model</th><th>defect class</th><th>r* (eV)</th><th>CR3 σ (eV)</th><th>snapshots</th></tr></thead>
<tbody>{rstar_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 7a. The r* ± se used in the headline discrepancy term (DFT training labels only).</caption>
<thead><tr><th>model</th><th>class</th><th>estimator</th><th>r* (eV)</th></tr></thead><tbody>{r7a_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 7. Coverage of the 20 scored QoIs with the discrepancy term on the four defect QoIs
(models with r*: {models_disc}).</caption>
<thead><tr><th>model</th><th>method</th><th>95 % without</th><th>95 % with [CI]</th><th>68 %</th><th>median σ/tol</th></tr></thead>
<tbody>{dcov_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 7b. Sensitivity to the r* estimator, CR3 × λ<sub>F</sub> and λ<sub>V</sub>, 95 % (68 %) coverage of the
20 QoIs. Headline: regression for the interstitial class, per-size regression for the vacancy. Paired: each defect snapshot against its nearest
bulk snapshot (the form that is also a synthetic row). Regression: the bulk reference is a per-atom thermal trend fitted over all
bulk cells near the defect volume (Table 5).</caption>
<thead><tr><th>model</th><th>scale</th><th>no term</th><th>headline</th><th>all paired</th><th>all regression</th><th>interstitial r* paired / regression (eV)</th></tr></thead>
<tbody>{est_rows}</tbody></table></div>
<div class="wrap"><table><caption>Table 7c. Evidence on the two estimators that uses no DFT QoI value. Columns: median
n<sub>def</sub>/n<sub>b</sub> of the paired form, and the RMS of the amplified partner term (o4d20, eV); intercept standard error paired /
regression (o3d12 · o4d20, eV); cold hold-out RMSE paired / regression (fit without the coldest 20 % of defect snapshots, predict
them, DFT training labels; o3d12, o4d20, eV); model against model (o4d20 as truth for o3d12: the true QoI differences from the
models' own harness values: paired / regression prediction, eV).</caption>
<thead><tr><th>class</th><th>n<sub>def</sub>/n<sub>b</sub></th><th>partner term</th><th>SE</th><th>hold-out o3d12</th><th>hold-out o4d20</th><th>model vs model</th></tr></thead>
<tbody>{ec_rows}</tbody></table></div>
<p>For interstitials the data-only evidence favours the regression estimator. The paired form takes most partners
from 2-atom bulk cells, so a partner's residual enters ×108 (0.35 eV RMS, the size of the bias), its standard error is larger, and
it extrapolates worse to held-out cold snapshots (half the RMSE for regression, in both models; likewise for divacancies). The
model-against-model test does not separate them (both inside the true spread), and neither estimator handles the vacancy in any test.
The headline therefore uses regression for interstitials; the vacancy is treated below. The regression estimator also agrees
better with the observed interstitial errors. That was seen before the data-only tests were run and is not part of the
argument, but it means the choice was not made blind; the all-paired result is kept as a sensitivity.</p>
<div class="wrap"><table><caption>Table 8. Defect QoIs, CR3 × λ<sub>F</sub>: error, σ without and with the discrepancy term
(eV), and z.</caption>
<thead><tr><th>model</th><th>QoI</th><th>DFT − model</th><th>σ</th><th>σ with term</th><th>z</th></tr></thead>
<tbody>{dz_rows}</tbody></table></div>""", "", level=0)

sec("results-vacancy", "The vacancy", f"""
<p>The vacancy shows no thermal trend in either pooled estimator, and both fail the model-against-model test. The training
vacancy cells come in two sizes, 63 and 215 atoms, and the sizes disagree.</p>
<div class="wrap"><table><caption>Table 8b. Zero-temperature vacancy bias from the regression form, per training-cell size, the
z-score of their difference, and their inverse-variance combination (eV). The last row applies the same estimator to the
energy difference between the two models on the training cells; the true difference of the models' own relaxed vacancy QoIs is
0.77 eV (no DFT value involved).</caption>
<thead><tr><th></th><th>63-atom cells</th><th>215-atom cells</th><th>z (difference)</th><th>inverse variance</th></tr></thead>
<tbody>{vac_rows}</tbody></table></div>
<p>The sizes differ at z ≈ 4 in both models, so a single pooled trend, as both earlier estimators assumed, is rejected by the
data. The 63-atom cells reach colder thermal amplitudes, and their intercept does not depend on the one near-relaxed cell. The
per-size combination recovers 0.65 eV of the 0.77 eV model-to-model difference, where the pooled forms give about 0.2. That near-relaxed
63-atom cell gives direct evidence with no extrapolation: its bulk-subtracted residual is +0.61 eV (o3d12) and 0.00 eV (o4d20), and
relaxing it ({cold_txt}) suggests that o3d12 over-relaxes. Both models relax the harness vacancy to a symmetric structure (vacancy
neighbour distances {geo_txt}); the harness gives no DFT structure to compare.</p>
<p>One selection risk remains: the closeness of the 63-atom intercept to the vacancy QoI errors was seen before the heterogeneity and
model-against-model checks. The choice rests on those checks and on the cold cell, which use no DFT QoI value.</p>
<figure><img src="fig8_model_compare_disc.png" alt="Per QoI, error over tolerance with 95 percent intervals for both models with the defect discrepancy term, the vacancy from the per-size estimator; intervals without the term as thin black bars, and the vacancy with the pooled-size estimator as dashed grey bars"><figcaption>Figure 8.
As Figure 1, with the headline discrepancy term on the four defect QoIs (CR3 × λ<sub>F</sub>): the regression estimator for the
interstitials and the per-size inverse-variance estimator for the vacancy. Thin black bars: the same QoIs'
95 % intervals without the term. Dashed grey bars: the vacancy with the pooled-size regression estimator, which hardly widens it.
Only the four defect QoIs change.</figcaption></figure>""", "", level=4)

import numpy as _np
fc_rows = ""
fc_have = []
mg = json.load(open(RES / "mondrian_groups.json")) if (RES / "mondrian_groups.json").exists() else {}
_mg = next(iter(mg.values()), None)
def g_label(g):
    """Group g = 2 band + flag (ace_jax.fit.conformal.assign_groups): distortion band, coordination flag."""
    if _mg is None:
        return ""
    band, flag = divmod(g, 2)
    e = [0.0] + list(_mg["edges"]) + [None]
    lo, hi = e[band], e[band + 1]
    d = f"d &lt; {hi:.3f}" if band == 0 else (f"d ≥ {lo:.3f}" if hi is None else f"{lo:.3f} ≤ d &lt; {hi:.3f}")
    return f"{d}, {'z ≠ ' if flag else 'z = '}{_mg['z_star']}"
for m in ("o4d20", "o3d12"):
    pth = RES / f"force_calibration_{m}.npz"
    if not pth.exists():
        continue
    z = _np.load(pth, allow_pickle=True)
    fc_have.append(m)
    cov_all = float(_np.mean(z["score"] <= z["q"]))
    rr = lambda mk, z=z: float(_np.sqrt(_np.mean((z["err_norm"][mk] / z["forces_std"][mk]) ** 2)))
    allm = _np.ones(len(z["score"]), bool)
    fc_rows += (f"<tr><td>{m}</td><td>all</td><td>all atoms</td><td class=n>–</td><td class=n>–</td><td class=n>–</td>"
                f"<td class=n>{len(z['score'])}</td><td class=n>{cov_all:.3f}</td><td class=n>{rr(allm):.2f}</td></tr>")
    for g in range(len(z["lam_rms"])):
        mk = z["group"] == g
        if not mk.any():
            continue
        fc_rows += (f"<tr><td>{m}</td><td>{g}</td><td>{g_label(g)}</td><td class=n>{z['lam_rms'][g]:.2f}</td><td class=n>{z['q_table'][g]:.1f}</td>"
                    f"<td class=n>{int(z['n_cfg'][g])}</td><td class=n>{int(mk.sum())}</td>"
                    f"<td class=n>{float(_np.mean(z['score'][mk] <= z['q'][mk])):.3f}</td><td class=n>{rr(mk):.2f}</td></tr>")
mg_r1 = f" = {_mg['r1']:.2f} Å" if _mg else ""
mg_z = f" = {_mg['z_star']}" if _mg else ""
sec("results-forces", "3.2  Calibrated per-atom forces (Mondrian conformal)", f"""
<p>Per-atom force uncertainty against the observed error on the 248 test configurations, from the ARD fits
(whose posteriors hold the Mondrian groups and scales; models: {", ".join(fc_have) or "pending"}). Symmetry-zero
atoms are dropped by the rule of Section 3.3.</p>
<p>The eight Mondrian groups (Section 2.5) are numbered g = 2b + f. The band b = 0, …, 3 is the atom's first-shell distortion d,
the relative spread of its neighbour distances within r<sub>1</sub>{mg_r1}, cut at the median, 90th and 99th percentiles of the
training atoms. The flag f is 1 when the atom's coordination differs from the modal z<sup>⋆</sup>{mg_z}. Even groups are therefore
four-fold atoms in increasingly distorted environments, from near-ideal diamond (group 0) to the most distorted 1 % (group 6); odd
groups are the same bands for under- or over-coordinated atoms, such as those at surfaces, defects and in the liquid. Table 11 gives
each group's limits. They depend on geometry only, and are the same for both models.</p>
<p>The groups are deliberately geometric rather than built from the site descriptor itself, for example from clusters or bins in a
low-rank (PCA) projection of it. A conformal group only helps where the ratio of the error to the predicted shape differs from
group to group. The CR3 shape is already a quadratic form in the atom's force rows, so it grows with distance from the training
descriptors: atoms that are novel in descriptor space already have a large <i>v</i>, and grouping by the same novelty mostly
re-sorts atoms the shape has ranked correctly. The shape errs instead along directions the descriptor projection does not single out.
A stretched or partly dissociated atom loses neighbours, so its force rows, and with them <i>v</i>, shrink towards zero while its
error does not, and the leading principal directions of the raw descriptor are likely to be dominated by large-norm, compressed environments.
Coordination and shell distortion separate exactly those cases. We tested descriptor-based groups in development on an
alloy model: a per-species Mahalanobis novelty grade in a whitened PCA of normalised descriptors, alone or crossed with the
distortion bands. The descriptor-novel atoms were already over-covered by the geometric groups (0.97 at a nominal 0.90), and
novelty groups either lowered overall coverage (0.88–0.89) or moved coverage from novel to typical atoms with no net gain
(unpublished ace-jax development report).</p>
<figure><img src="fig7_force_calibration.png" alt="Per-atom force uncertainty against observed error, binned on predicted sigma and on observed error, for each model"><figcaption>Figure 7.
Top: binned on predicted σ (20 equal-count bins), RMS observed |ΔF| against RMS σ, for the Mondrian
σ<sub>F,i</sub> and for the global λ<sub>F</sub> of the propagation, with the per-atom density behind; points below
the diagonal are conservative. Bottom: binned on observed |ΔF|. Binning on the observed error regresses towards the mean, so
the low-error bins read as under-confident and the high-error bins as over-confident even for a calibrated σ.</figcaption></figure>
<div class="wrap"><table><caption>Table 11. Per Mondrian group: the fitted scales (λ<sub>g</sub>, q<sub>g</sub> at α = 0.1),
calibration configurations, test atoms, test coverage of the anisotropic conformal region, and RMS |ΔF|/σ<sub>F,i</sub>
(1 is calibrated in the Gaussian reading).</caption>
<thead><tr><th>model</th><th>group</th><th>environment</th><th>λ<sub>g</sub></th><th>q<sub>g</sub></th><th>cal. configs</th><th>test atoms</th><th>coverage</th><th>RMS |ΔF|/σ</th></tr></thead>
<tbody>{fc_rows}</tbody></table></div>
<p>The conformal regions hold their nominal 90 % on the test split overall and roughly per group. The Gaussian reading
of σ<sub>F,i</sub> is conservative (RMS ratio below 1), because λ<sub>g</sub> is the RMS of heavy-tailed scores. The extreme case is
o4d20's bulk-like group 0 (half the test atoms): λ<sub>0</sub> is 4–6× the other groups' and its σ about 3.6× too wide, while its quantile
region still covers at the nominal rate. The RMS scale is dominated by a few extreme hold-out scores; the conformal quantile is
robust to them. Here λ<sub>F</sub>
was fitted on this same test split, so its line is in-sample for its overall level; λ<sub>g</sub> and q<sub>g</sub> come from the
training hold-out.</p>""", "drafted", level=3)

dk95 = lambda m: (lambda d_: f"{d_['k95']}/{d_['n']}" if d_ else "?")(disc.get(m, {}).get("CR3-headline+disc/lam_F"))
sec("discussion", "4  Discussion", f"""
<p>A single parameter covariance, the CR3 jackknife scaled once to held-out forces, gives usable uncertainties for derived
properties that the fit never saw. For both models most QoIs fall inside their 95 % intervals ({k95("o3d12", "CR3-headline", "lam_F")}
for o3d12, {k95("o4d20", "CR3-headline", "lam_F")} for o4d20), and the intervals scale with the model: the median σ/tol is
{sot("o4d20", "CR3-headline", "lam_F")} for the accurate o4d20, whose intervals are mostly inside the benchmark tolerance, and
{sot("o3d12", "CR3-headline", "lam_F")} for o3d12. The scale is fitted once, on forces of held-out configurations, and never on a QoI.
The same CR3 shape serves per-atom forces, with scales per environment group (Section 3.2), and relaxed QoIs, with one global
scale. The second use is what a distribution over θ buys: one draw gives energies, forces, stresses and every QoI together,
consistently.</p>
<p>The baselines show which ingredient matters. The CR3 shape without a scale (B3) is far too narrow, so calibration is essential,
as expected for a parameter variance that shrinks with the amount of data. The Bayesian posterior with a calibrated scale (B1)
has the right overall size but the wrong shape: it under-covers the QoIs and fails the joint test, because the posterior describes
noise in the labels, not the approximation error of a misspecified model. POPS (B2) targets that error directly, but its width is
set by which training rows define the hypercube, and with no calibration step it is too narrow for the larger model. The CR3 shape,
which measures how far the fit moves when each configuration is left out, is the one that transfers, provided it is scaled.</p>
<p>The scale is not unique. The virial scale is about a third larger than the force scale, and using it covers
{k95("o3d12", "CR3-headline", "lam_V")} and {k95("o4d20", "CR3-headline", "lam_V")} QoIs with wider intervals, and passes the joint test for
both models. Misspecification depends on the observable, and one could choose the scale by the QoI's physics, the virial scale
for stress-like QoIs such as elastic constants, or report both. A single scale is also a compromise across configuration types
(Figure 5), which a per-type or per-group scale in θ could relax. The per-atom Mondrian scales cannot be used for this directly:
they rescale force outputs atom by atom and define no covariance in θ; a θ-space analogue would need a shape that depends on the
environment group.</p>
<p>Linearisation is adequate at the size of the distribution, but not everywhere at the calibrated size. The delta method agrees
with committees run through the unmodified benchmark for almost every QoI at λ = 1, and implicit differentiation through each
relaxation, including that of the lattice constant, makes it cheap: one set of rows per QoI. At λ<sub>F</sub> the phase energies of
the structures furthest from diamond become strongly nonlinear, because their relaxations change character; for those, committees
are the safer route, and their spread is then a property of the relaxation path rather than of a Gaussian.</p>
<p>The clearest failure is the relaxed point defects, missed by every method. Its cause is in the data: every defect cell in
GAP-18 is a hot molecular-dynamics snapshot, the model's defect–bulk error drifts with thermal amplitude, and the relaxed defect
lies beyond the data along a direction that every snapshot shares. A delete-one jackknife cannot see such a bias, because deleting
one snapshot does not move the fit along it, and a rank-one parameter correction built from the snapshots does not carry it to the
relaxed geometry either. The bias can, however, be measured from the training labels alone, by extrapolating the defect–bulk
residual to zero thermal amplitude, and added as a discrepancy term; this restores coverage to {dk95("o3d12")} (o3d12) and
{dk95("o4d20")} (o4d20) without using any DFT QoI value. The vacancy needed the training cells of different sizes to be treated
separately, a reminder that such extrapolations depend on how the training data were made. Relaxed or low-temperature defect
configurations in the training set would remove the need for the term, and the same diagnostic, a residual trend that extrapolates
away from zero, can flag where such data are missing.</p>
<p>The study has limitations. The calibration assumes that the test split and the QoI structures are exchangeable with the
training data, which the defects show is not always true. The hyperparameters are held fixed in the jackknife, so their own
uncertainty is not propagated. The locality of the model, which cannot represent interactions beyond its cutoff, appears here only
as part of the effective error. And twenty QoIs give coarse coverage estimates: counts from 17 to 20 are all consistent with a
nominal 95 %.</p>""", "")

sec("conclusions", "5  Conclusions", """
<p>For linear ACE potentials, the delete-one-configuration (CR3) jackknife covariance, scaled once, gives calibrated uncertainty at
two levels. Per atom, with scales fitted by Mondrian conformal prediction on groups of local environments, the force regions reach
their target coverage on held-out silicon data. For derived properties, the same covariance with one scale fitted to held-out forces,
propagated by implicit differentiation through each relaxation, covers most of 20 silicon QoIs against DFT, with intervals that
shrink with the accuracy of the model, and outperforms the Bayesian posterior, POPS and the uncalibrated shape. Where it fails, for
relaxed point defects, the cause is a bias shared by all training snapshots of a kind, which no delete-one method can see; it can be
measured from the training labels and added as a discrepancy term, which restores coverage. The methods are implemented in ace-jax.</p>""", "")

sec("methods-detail", "6  Computational details", """
<ul>
<li>Data: GAP-18 silicon, 2227 train / 248 test (seed 0, split hash c433c98d49bb). Models: ACE o4d20 (p = 5476) and o3d12 (p = 223),
r<sub>cut</sub> 5 Å, least-squares E<sup>0</sup>, shared noise, smoothness prior, float64.</li>
<li>Harness: py3 port of libAtoms/silicon-testing-framework, run unmodified with relaxed-ion elastic fmax 10<sup>−3</sup> eV/Å.</li>
<li>Code: <code>bench/si_qoi_uq</code> (branch silicon-uq-prop); seeds recorded in <code>results/run_metadata.json</code>.</li>
<li>Compute: lestrade (32 cores, RTX 4000 Ada), mnf148 (32 cores, 256 GB).</li>
</ul>""", "drafted")

sec("ack", "Acknowledgements", """
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
<li>R. Dunn, L. Wasserman, A. Ramdas, Distribution-free prediction sets for two-layer hierarchical models, J. Am. Stat. Assoc. 118, 2491–2502 (2023).</li>
<li>A. Grossfield, D. M. Zuckerman, Quantifying uncertainty and sampling quality in biomolecular simulations, in Annual Reports in Computational Chemistry, Vol. 5, Elsevier (2009).</li>
<li>K. Fisher, M. Herbst, J. Kermode, Y. Marzouk, Multivariate conformal uncertainty propagation in multitask atomistic simulation: successes and pitfalls, arXiv:2609.31384 (2026).</li>
<li>J. A. Hoeting, D. Madigan, A. E. Raftery, C. T. Volinsky, Bayesian model averaging: a tutorial, Stat. Sci. 14, 382–417 (1999).</li>
<li>P. J. Huber, The behavior of maximum likelihood estimates under nonstandard conditions, Proc. Fifth Berkeley Symp. Math. Stat. Probab. 1, 221–233 (1967).</li>
<li>G. Imbalzano, Y. Zhuang, V. Kapil, K. Rossi, E. A. Engel, F. Grasselli, M. Ceriotti, Uncertainty estimation for molecular dynamics and sampling, J. Chem. Phys. 154, 074102 (2021).</li>
<li>M. Kellner, M. Ceriotti, Uncertainty quantification by direct propagation of shallow ensembles, Mach. Learn.: Sci. Technol. 5, 035006 (2024).</li>
<li>B. J. K. Kleijn, A. W. van der Vaart, The Bernstein–von Mises theorem under misspecification, Electron. J. Stat. 6, 354–381 (2012).</li>
<li>T. Lelièvre, M. Rousset, G. Stoltz, Free Energy Computations: A Mathematical Perspective, Imperial College Press (2010).</li>
<li>J. G. MacKinnon, M. Ø. Nielsen, M. D. Webb, Cluster-robust inference: a guide to empirical practice, J. Econometrics 232, 272–299 (2023).</li>
<li>J. G. MacKinnon, H. White, Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties, J. Econometrics 29, 305–325 (1985).</li>
<li>U. K. Müller, Risk of Bayesian inference in misspecified models, and the sandwich covariance matrix, Econometrica 81, 1805–1849 (2013).</li>
<li>F. Musil, M. J. Willatt, M. A. Langovoy, M. Ceriotti, Fast and accurate uncertainty estimation in chemical machine learning, J. Chem. Theory Comput. 15, 906–915 (2019).</li>
<li>J. E. Pustejovsky, E. Tipton, Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models, J. Bus. Econ. Stat. 36, 672–683 (2018).</li>
<li>G. Simmons, T. Hudson, T. D. Swinburne, J. R. Kermode, review of uncertainty quantification for machine-learned interatomic potentials, in preparation.</li>
<li>T. D. Swinburne, D. Perez, Parameter uncertainties for imperfect surrogate models in the low-noise regime, Mach. Learn.: Sci. Technol. 6, 015008 (2024).</li>
<li>T. D. Swinburne, POPSRegression: Bayesian regression for low-noise data with misspecification uncertainty, Python package, https://pypi.org/project/popsregression/.</li>
<li>R. J. Tibshirani, R. F. Barber, E. J. Candès, A. Ramdas, Conformal prediction under covariate shift, Adv. Neural Inf. Process. Syst. 32, 2530–2540 (2019).</li>
<li>V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer (2005).</li>
<li>V. Vovk, D. Lindsay, I. Nouretdinov, A. Gammerman, Mondrian confidence machine, On-line Compression Modelling project (old series), Working Paper 4, Royal Holloway, University of London (2003).</li>
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
<title>Calibrated parameter uncertainty for silicon QoIs</title>
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
<h1>Calibrated parameter-space uncertainty for linear ACE, propagated to silicon properties</h1>
<p class="author">James R. Kermode</p>
<p class="affil">Warwick Centre for Predictive Modelling, School of Engineering, University of Warwick, Coventry CV4 7AL, United Kingdom</p>

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
page = _re.sub(r"\b(Table|Figure) ([0-9]+[a-z]?)\b", lambda m: f"{m.group(1)} {_num.get((m.group(1), m.group(2)), m.group(2))}", page)
(OUT / "index.html").write_text(page)
for p in FIG.glob("*.png"):
    shutil.copy(p, OUT / p.name)
print("ok", OUT)
