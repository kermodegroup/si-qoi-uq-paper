"""Paper skeleton (artifact) for the silicon QoI UQ study: theory from docs/dev/specs/tex/force-uq-math-pipeline.tex
and the ace-jax fitting docs, the propagation methodology of bench/si_qoi_uq, and results read from results/.
Not part of the pipeline.

    python bench/si_qoi_uq/artifact/paper.py   -> bench/si_qoi_uq/artifact/paper/index.html (+ figures)
"""
import csv, html, json, pathlib, shutil

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

TAB = [("CR3-headline", "lam_F", "CR3 × λ<sub>F</sub> (headline)"), ("CR3-headline", "lam_V", "CR3 × λ<sub>V</sub>"),
       ("CR3-headline", "lam_E", "CR3 × λ<sub>E</sub>"), ("CR3-headline", "lam_pool", "CR3 × λ<sub>pool</sub>"),
       ("CR3-committee-lamF-physical", "lam_F", "CR3 at λ<sub>F</sub>, full re-run, physical members"),
       ("B3-headline", "lam_1", "B3: CR3, λ = 1"), ("B1-headline", "q_b1", "B1: posterior × q̂"),
       ("B2lev50-committee", "lam_1", "B2 variant: POPS, leverage pct 50"),
       ("B2lev5-committee", "lam_1", "B2 variant: POPS, leverage pct 5")]
cov_rows = "".join(
    f"<tr><td>{m}</td><td>{lab}</td><td class=n>{int(r['n'])}</td><td class=n>{int(r['k68'])} [{f(r['ci68_lo'])}, {f(r['ci68_hi'])}]</td>"
    f"<td class=n>{int(r['k95'])} [{f(r['ci95_lo'])}, {f(r['ci95_hi'])}]</td><td class=n>{f(r['rms_z'])}</td>"
    f"<td class=n>{f(r['median_sigma_over_tol'])}</td></tr>"
    for m in models for meth, sc, lab in TAB for r in [get(m, meth, sc)] if r)
lam_rows = "".join(f"<tr><td>{m}</td>" + "".join(f"<td class=n>{f(lam[m].get(k))}</td>" for k in ("E", "F", "V", "pool"))
                   + f"<td class=n>{f(lam[m]['V'] / lam[m]['F'])}</td><td class=n>{f(lam[m]['E'] / lam[m]['F'])}</td></tr>"
                   for m in models)
jrows = ""
for m in models:
    for name in ("CR3/lam_F", "CR3/lam_V", "B1/q_b1"):
        for kind, jt in joint.get(m, {}).get(name, {}).items():
            e = jt["scaled"]["eps_0.001"]
            jrows += (f"<tr><td>{m}</td><td>{html.escape(name)}</td><td>{kind}</td><td class=n>{jt['scaled']['df']}</td>"
                      f"<td class=n>{f(e['mahalanobis'], 1)}</td><td class=n>{f(e['chi2_95'], 1)}</td><td class=n>{e['p_value']:.2g}</td></tr>")
nflag = {m: sum(1 for r in dvc if r["model"] == m and r.get("flagged") in ("True", True) and "JAX" not in r["qoi"]) for m in models}
ncmp = {m: sum(1 for r in dvc if r["model"] == m and r.get("flagged") in ("True", "False", True, False) and "JAX" not in r["qoi"]) for m in models}
head = {m: get(m, "CR3-headline", "lam_F") for m in models}
pw = {(r["model"], r["set"]): r for r in mw}

S = []  # sections


def sec(id_, title, body, status=""):
    chip = f'<span class="status {status}">{status}</span>' if status else ""
    S.append(f'<section id="{id_}"><h2>{title} {chip}</h2>{body}</section>')


b3 = {m: get(m, "B3-headline", "lam_1") for m in models}
b1 = {m: get(m, "B1-headline", "q_b1") for m in models}
lam_lo = min(min(v for k, v in lam[m].items() if k != "pool") for m in models)
lam_hi = max(max(v for k, v in lam[m].items() if k != "pool") for m in models)
phr = {r["qoi"]: r for r in rd("lamF_physical.csv")}
nonphase = [r["ratio_physical_vs_rescaled"] for k, r in phr.items() if not k.startswith("dE_") and r["ratio_physical_vs_rescaled"] not in ("", None)]
phase = [r["ratio_physical_vs_rescaled"] for k, r in phr.items() if k.startswith("dE_") and r["ratio_physical_vs_rescaled"] not in ("", None)]
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
<ul class="plan">
<li><b>Motivation.</b> Uncertainty is needed for forces (on-the-fly error control in MD, active learning) and for the derived
properties a potential is built to predict. Machine-learned potentials are misspecified: the labels are deterministic and the
error is approximation error. The Bayesian posterior is then far too narrow: under misspecification it concentrates at the
best-fitting parameters at a rate set by the amount of data, not by the size of the model error, and collapses to a point as
N<sub>data</sub> → ∞ while the prediction error stays finite [Kleijn and van der Vaart 2012; Müller 2013; Swinburne and Perez 2025].</li>
<li><b>Gap 1: per-atom force uncertainty.</b> Committees and ensembles are costly and need their own calibration; posterior
variances are miscalibrated; marginal conformal prediction covers on average but not by environment, and errors are strongly
heterogeneous across local environments (distortion, coordination) and correlated within a configuration.</li>
<li><b>Contribution A, new in this paper:</b> calibrated per-atom forces for linear ACE.
  <ul>
  <li>The shape is a cluster-robust ("sandwich") covariance of the coefficients, with each training configuration as one cluster
  of correlated rows. We use the variant known as CR3 in the econometrics literature: the third of the standard cluster-robust
  estimators CR0–CR3, which extend the heteroskedasticity-consistent HC0–HC3 to clusters [MacKinnon and White 1985; Bell and McCaffrey
  2002; MacKinnon, Nielsen and Webb 2023]. CR3 is the delete-one-cluster jackknife: it measures how far the fitted coefficients move
  when each configuration is left out. It is computed exactly without refitting, from the PRESS (predicted residual error sum of
  squares) residuals: the residuals each row would have if its cluster were left out of the fit, obtained from the ordinary residuals
  by a leverage correction (for a single row, dividing by one minus its leverage; for a cluster, the matrix analogue) [Allen 1974; Cook and Weisberg 1982]. Its per-atom
  3×3 block gives anisotropic force regions.</li>
  <li>The scale is fitted by Mondrian conformal prediction [Vovk et al. 2003; Vovk, Gammerman and Shafer 2005]. Conformal
  prediction turns the scores of held-out calibration points into a quantile that covers a new point with a guaranteed probability,
  under exchangeability. The Mondrian variant splits the points into categories, here groups of local environments
  (shell-distortion bands × coordination), and calibrates each category on its own, so coverage holds within each group and not only on
  average; the name refers to the rectangles of Piet Mondrian's paintings. Because atoms in one configuration are correlated, the
  quantiles weight configurations, not atoms [Dunn, Wasserman and Ramdas 2023]. The quantiles come from a stratified hold-out of the training set, with a fitted transfer exponent carrying
  them to the full fit, and give coverage at the level of configurations.</li>
  <li>It is served by ace-jax as forces_std, forces_cov and a conformal region per atom (Sections 3.1 and 7.7).</li>
  </ul></li>
<li><b>Gap 2: derived properties.</b> A QoI depends on every atom and on relaxed geometries, so per-atom intervals cannot be
combined into a QoI interval. A QoI's uncertainty has two parts [Simmons, Hudson, Swinburne and Kermode, in prep.]. The
aleatoric part is the sampling error of the simulation that computes the QoI, assessed with statistical convergence tests
[Grossfield and Zuckerman 2009; Lelièvre, Rousset and Stoltz 2010, Sec. 2.3]. The epistemic part comes from the uncertainty in
the fit. It is most widely estimated with a committee of \\(n\\) potentials, an idea long established in statistics [Hoeting et
al. 1999]. If member \\(i\\) predicts \\(D_i\\sim\\mathcal N(\\mu_i,\\sigma_i^2)\\), independently, the predictive variance
\\(\\tfrac1n\\sum_i\\sigma_i^2+\\tfrac1{n-1}\\sum_i(\\mu_i-\\bar\\mu)^2\\) is the mean aleatoric variance plus the committee's
spread of means, the epistemic part. The QoIs here come from relaxations and static calculations, so their aleatoric part is zero
and their uncertainty is purely epistemic. For a linear model the consistent object is a distribution over coefficients θ: one draw gives
energies, forces and stresses together, and any QoI follows.</li>
<li><b>Contribution B:</b> propagation of the same CR3 distribution, with one scale fitted to held-out forces, to 21 silicon QoIs.
It uses implicit differentiation through every relaxation, including the lattice constant, cross-checked by committees of the
unmodified benchmark harness, and tests the intervals against DFT with binomial and joint tests and against two baselines: the
posterior with one conformal scale, and POPS.</li>
<li><b>Contribution C:</b> where calibrated parameter-space UQ fails, and why. A bias shared by all training snapshots of a kind
(here, every defect cell is hot) is invisible to any delete-one jackknife. We diagnose it, measure it from training labels, and
compare a discrepancy term, a coarser jackknife and reweighted fits (difference rows, IRLS). Along the way: a QR evidence for ARD,
whose Cholesky form was pinned at its floor on these fits.</li>
<li><b>Related work</b> [to verify]: conformal prediction (Vovk et al. 2005; Tibshirani et al. 2019 covariate shift; Barber et al.
2021 jackknife+; Dunn et al. 2023 hierarchical); cluster-robust and sandwich covariances (Huber 1967; White 1982; Müller 2013;
MacKinnon et al. 2023); UQ for interatomic potentials (Musil et al. 2019 and Imbalzano et al. 2021 committees; Bigi et al. 2024 /
Kellner and Ceriotti 2024 LLPR; Swinburne and Perez POPS; Best et al. 2024 silicon UQ; Bartók et al. 2018 GAP-18).</li>
<li><b>Outline.</b> Section 2: the linear ACE fit and ARD. Section 3: the CR3 shape and the per-atom calibration. Sections 4–6: global
scale, propagation and validation. Section 7: results, defects and ablations. Section 8: discussion.</li>
</ul>""", "outline")

sec("theory-model", "2  Linear ACE and its Bayesian fit", r"""
<h3>2.1 Model and observations</h3>
<p>Each atom \(i\), at position \(r_i\) with species \(Z_i\) and environment \(x_i\) (its neighbours within the cutoff), has a site descriptor \(\phi(x_i)\in\mathbb{R}^p\): the ACE product basis in
species-major blocks, then pair blocks. The energy is linear in the coefficients \(\theta\),</p>
\[ E(\{r_i\};\theta)=\sum_i \phi(x_i)\cdot\theta+\sum_i E^0_{Z_i}, \]
<p>and so is every observation of a configuration:</p>
\[ b_E=\sum_i\phi(x_i),\qquad B_{F,i\alpha}=-\frac{\partial b_E}{\partial r_{i\alpha}},\qquad
   B_{V,\alpha\beta}=-\frac{\partial b_E}{\partial\varepsilon_{\alpha\beta}}\Big|_{\varepsilon=0}, \]
<p>with \(E^0_Z\) a per-species reference energy, \(r_{i\alpha}\) the Cartesian components of \(r_i\), and \(\varepsilon\) a symmetric strain
applied to positions and cell. The predictions are \(b_E\cdot\theta\), \(B_F\theta\) and the virial \(W=B_V\theta\) (in eV; the stress is
\(-W/\Omega\), with \(\Omega\) the cell volume). Each row of \(b_E\), \(B_F\) or \(B_V\) is called an observable row \(b\).
A row has a structural weight \(w\) (per-atom energy normalisation and per-config-type weights) and a noise scale
\(\sigma_q\), \(q\in\{E,F,V\}\); the whitened rows and targets are \(\psi=b\,w/\sigma_q\), \(\tilde y=y\,w/\sigma_q\), with
\(y\) the DFT label minus the \(E^0\) baseline. Stacked over all configurations, the observable rows \(b\) form the design matrix \(\Phi\), and the
\(\psi\) its whitened form \(\Psi\). With least-squares \(E^0\), one column per species carries the atom count on the
energy rows (a fixed, wide prior).</p>
<h3>2.2 Prior, posterior and evidence</h3>
<p>The prior is \(\theta\sim\mathcal N(0,\Lambda^{-1})\) with the smoothness prior \(\Lambda=\operatorname{diag}(\Gamma_j^2)/\sigma_c^2\): \(\sigma_c\) is the prior scale of the coefficients
and \(\Gamma_j=\sum_t\big[(l_t/w_l)^4+(n_t/w_n)^4\big]\), summed over the one-particle factors \(t\) of basis function \(j\) with radial
index \(n_t\) and angular degree \(l_t\) (ACEpotentials' algebraic prior, \(w_l=2/3\), \(w_n=1\)), so rough basis functions are damped.
The posterior is Gaussian,</p>
\[ A=\sum_{\text{rows}}\psi\psi^{\mathsf T}+\Lambda,\qquad \hat\theta=A^{-1}\sum_{\text{rows}}\psi\tilde y,\qquad
   \operatorname{Cov}(\theta\mid\mathcal D)=A^{-1}, \]
<p>and the hyperparameters \(h=(\sigma_c,\sigma_E,\sigma_F,\sigma_V)\) maximise the evidence</p>
\[ \log p(\mathcal D\mid h)=-\tfrac12\sum_q\frac{\lVert y^w_q\rVert^2}{\sigma_q^2}+\tfrac12 c^{\mathsf T}A^{-1}c-\tfrac12\log|A|
   +\tfrac12\log|\Lambda|-\sum_q n_q\log\sigma_q , \]
<p>up to a constant, with \(\mathcal D\) the training data, \(y^w_q\) the structurally weighted targets (\(y\,w\)) of quantity \(q\),
\(n_q\) their number, and \(c=\sum_{\text{rows}}\psi\tilde y=A\hat\theta\).</p>
<p>The fits here share one noise scale (\(\sigma_E=\sigma_F=\sigma_V\); the E:F:V balance is set by the structural weights).
Near-interpolating fits leave the condition number \(\operatorname{cond}(A)\) beyond \(1/\epsilon_{\text{mach}}\) (\(\epsilon_{\text{mach}}\)
the float64 unit roundoff), so the evidence and the posterior are computed by QR. \(R_q\) is the triangular factor of quantity \(q\)'s
structurally weighted rows (\(\Phi^w_q=Q_qR_q\)), accumulated by streamed updating QRs, and
\([\operatorname{diag}\sqrt\Lambda;\,R_q/\sigma_q]_{q}=Q_AR_A\) gives \(R_A^{\mathsf T}R_A=A\) at \(\operatorname{cond}(\Phi)\) instead of
\(\operatorname{cond}(\Phi)^2\).</p>
""", "drafted from docs")

ards = json.load(open(RES / "ard_summary.json")) if (RES / "ard_summary.json").exists() else {}
ard_rows = ""
for m in ("o3d12", "o4d20"):
    if m not in ards:
        continue
    d_ = ards[m]
    names = [n_.replace("a_", "").replace("body", "") for n_ in d_["h_names"]]
    a_txt = ", ".join(f"a<sub>{nm}</sub> {h_:.1f}{' (floor)' if ab else ''}" for nm, h_, ab in zip(names, d_["h"], d_["at_bound"]))
    ta, tb = d_["test_rmse"]["ard"], d_["test_rmse"]["blr"]
    ca, cb = d_["coverage"]["ard"].get("CR3-headline/lam_F", {}), d_["coverage"]["blr"].get("CR3-headline/lam_F", {})
    ard_rows += (f"<tr><td>{m}</td><td>{a_txt}</td><td class=n>{d_['a_floor']:.1f} / {d_['a_blr']:.1f}</td>"
                 f"<td class=n>{ta['E_meV_atom']:.2f} / {tb['E_meV_atom']:.2f}</td><td class=n>{ta['F_eV_A']:.4f} / {tb['F_eV_A']:.4f}</td>"
                 f"<td class=n>{ta['V_meV_atom']:.1f} / {tb['V_meV_atom']:.1f}</td>"
                 f"<td class=n>{d_['lam_cr3']['ard']['F']:.2f} / {d_['lam_cr3']['blr']['F']:.2f}</td>"
                 f"<td class=n>{int(ca.get('k95', 0))} ({int(ca.get('k68', 0))}) / {int(cb.get('k95', 0))} ({int(cb.get('k68', 0))})</td></tr>")
sec("theory-ard", "2.3  Automatic relevance determination", rf"""
<p>ARD replaces the single \(\sigma_c\) by one log-precision per body order, \(\Lambda=\operatorname{{diag}}(\Gamma_j^2e^{{a_{{k(j)}}}})\), with \(k(j)\) the body order of basis function \(j\), fitted by
type-II maximum likelihood in the prior-scaled system \(S=D^{{-1}}AD^{{-1}}\), \(D=\operatorname{{diag}}\Gamma\). The BLR fit is ARD at
\(a_k=-2\log\sigma_c\) for every \(k\). In sequential mode, used here, the noise scales stay at the BLR fit's shared value and only the
\(a_k\) are fitted.</p>
<p><b>QR evidence.</b> Near-interpolating fits leave \(\operatorname{{cond}}(S)\) beyond \(1/\epsilon_{{\text{{mach}}}}\). The evidence, its
gradient and Hessian, and the posterior are therefore computed from the QR factor of
\([\operatorname{{diag}}(e^{{a_{{k(j)}}/2}});\,R_qD^{{-1}}/\sigma_q]_q=Q_SR_S\) (so \(R_S^{{\mathsf T}}R_S=S\)), whose condition number is
\(\sqrt{{\operatorname{{cond}}(S)}}\). The posterior stores \(R_S^{{\mathsf T}}\) as its Cholesky factor, so the CR3 jackknife and the served variances are unchanged. The \(a_k\) are bounded below at
\(a_{{\text{{floor}}}}=\log(\lambda_{{\max}}/10^{{22}})\), with \(\lambda_{{\max}}\) the largest eigenvalue of the data part of \(S\)
(\(D^{{-1}}\Psi^{{\mathsf T}}\Psi D^{{-1}}\)) at the starting noise scales, and an endpoint with every \(a_k\) on the floor and the gradient pointing below it
is flagged as degenerate and refused.</p>
<p>With the earlier Cholesky evidence the floor sat at \(\operatorname{{cond}}(S)=10^{{14}}\), about 12 nats above the evidence optimum on both
silicon models. Every \(a_k\) ended on it and the mean lost accuracy (o4d20 test energy 3.5 → 16.4 meV/atom in the fit's own report;
the floor's prior, not roundoff, was the cause; ace-jax issue #136, fixed in PR #137).</p>
<div class="wrap"><table><caption>Table 1. ARD (QR) against the benchmark BLR fits. ARD log-precisions per body order (endpoint); floor /
BLR prior; test-split RMSE ARD / BLR (E, V per atom); test-split λ<sub>F</sub> ARD / BLR; propagated CR3 × λ<sub>F</sub> coverage, 95 % (68 %),
ARD / BLR, over the 20 scored QoIs.</caption>
<thead><tr><th>model</th><th>a<sub>k</sub></th><th>floor / BLR</th><th>E (meV/atom)</th><th>F (eV/Å)</th><th>V (meV/atom)</th><th>λ<sub>F</sub></th><th>coverage</th></tr></thead>
<tbody>{ard_rows}</tbody></table></div>
<p>With the QR evidence the fits are not degenerate. For o4d20 every \(a_k\) is interior, and for o3d12 only the two-body precision sits on
the new floor. ARD weakens the prior relative to the BLR's single \(\sigma_c\) on most body orders (on all of them for o3d12; for
o4d20 only the 5-body block is held more tightly), yet the means match the BLR fits on the held-out test split, and the test-split
scales and the propagated coverage agree. The QoI results below use the BLR fits, which are the
benchmark models; ARD gives the same coverage (acceptance run in the repository), and it provides the per-atom force calibration of
Section 7.7.</p>""", "drafted")

sec("theory-cr3", "3  Misspecification-robust shape: the CR3 jackknife", r"""
<p>For a cluster \(k=1,\dots,n_{\text{cl}}\) of rows (here: one whole configuration with its energy, force and virial rows) with
whitened rows \(\Psi_k\) (the cluster's rows \(\psi^{\mathsf T}\) stacked), whitened targets \(\tilde y_k\) and residuals \(\rho_k=\tilde y_k-\Psi_k\hat\theta\), the cluster hat block and the leverage-corrected (PRESS)
score are</p>
\[ H_{kk}=\Psi_kA^{-1}\Psi_k^{\mathsf T}=W_k^{\mathsf T}W_k,\quad W_k=R_S^{-\mathsf T}D^{-1}\Psi_k^{\mathsf T},\qquad
   \tilde g_k=\Psi_k^{\mathsf T}(I-H_{kk})^{-1}\rho_k . \]
<p>With the hyperparameters fixed, deleting cluster \(k\) moves the mean by exactly \(\hat\theta-\hat\theta_{(-k)}=A^{-1}\tilde g_k\)
(the cluster DFBETA). Because \(\Lambda\succ0\), every eigenvalue of \(H_{kk}\) lies in \([0,1)\). The centred meat and the
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
calibrated on held-out data (Section 4). The CR3 covariance measures how much a
prediction depends on which configurations were in the fit: large where the fit leans on configurations it reproduces
poorly. It is a parameter variance used as a proxy for approximation error, which is why it needs a scale.</p>
<h3>3.1 Calibrated per-atom forces: the Mondrian conformal scale</h3>
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
<p><b>Transfer to the full training set.</b> The served posterior is fitted on all of \(T\), about 25 % more data than
\(P_{\text{fit}}\). A second hold-out fit \(P_{\text{fit2}}\) on a stratified subset of \(T_{\text{fit}}\) scores the same
\(T_{\text{val}}\) atoms. With \(N\), \(N_{\text{fit}}\), \(N_{\text{fit2}}\) the numbers of
configurations behind the served, first and second fits, and \(\lambda_1,\lambda_2\) the pooled RMS scores of \(P_{\text{fit}}\) and \(P_{\text{fit2}}\),
\(\beta=\operatorname{clip}\big(\log(\lambda_1/\lambda_2)/\log(N_{\text{fit}}/N_{\text{fit2}}),0,\tfrac12\big)\) and every score is
multiplied by \(f_{\text{tr}}=(N/N_{\text{fit}})^\beta\). \(\beta=0\) is the variance-dominated limit and \(\tfrac12\) the bias-dominated one.</p>
<p><b>Scales.</b> Each calibration configuration carries total weight 1 in every group it populates (atom weight \(1/n_{c,g}\), with \(n_{c,g}\) the atoms of configuration \(c\) in group \(g\), and \(n_{\text{cfg},g}\)
the calibration configurations with at least one atom in \(g\)).
The RMS scale and the conformal quantile of group \(g\) are</p>
\[ (\lambda_g^{\text{rms}})^2=\frac{1}{3\,n_{\text{cfg},g}}\sum_{c}\frac{1}{n_{c,g}}\sum_{i\in c\cap g}s_i^2,\qquad
   q_g=\inf\{s:\hat F_g(s)\ge1-\alpha\}, \]
<p>where \(\hat F_g\) is the configuration-weighted score distribution with a test point at \(+\infty\) (the pooled-CDF
construction of Dunn, Wasserman and Ramdas), and \(1-\alpha\) the target coverage. The default is \(\alpha=0.1\).</p>
<p><b>Served quantities.</b> \(\texttt{forces\_std}=\lambda_g^{\text{rms}}\sqrt v\) and \(\texttt{forces\_cov}=(\lambda_g^{\text{rms}})^2V\)
are the Gaussian reading, with \(g\) the atom's group. For a force error \(\Delta F\), the conformal region is \(\{\Delta F:\Delta F^{\mathsf T}(V+\epsilon I)^{-1}\Delta F\le q_g^2\}\), with
\(\texttt{forces\_q}\) its largest semi-axis. Assuming exchangeability of calibration and target configurations within a group,
the region covers a uniformly chosen atom of the group in a new configuration with probability \(\approx1-\alpha\); the
finite-sample correction is set by the number of configurations, not atoms. The ratio
\(\varrho_g=q_g/(\lambda_g^{\text{rms}}\chi_3^{-1}(1-\alpha))\), with \(\chi_3^{-1}\) the \(\chi_3\) quantile function, measures tails heavier than
Gaussian (\(\varrho_g=1\) for Gaussian errors). A covariate-shift diagnostic
(a density ratio on a PCA of the site descriptor, giving a weighted quantile) flags atoms outside the calibration support,
but does not rescale σ.</p>
<p><b>Why the propagation does not use these scales.</b> λ<sub>g</sub> and q<sub>g</sub> temper force <em>outputs</em> atom by
atom. They do not define a covariance in θ, and a QoI depends on every atom and on relaxed geometries. Propagation therefore uses one
global λ (Section 4). The groups also enter this study as a diagnostic breakdown of the test-split scores (Figure 5), and
Section 7.7 tests the served per-atom forces themselves. Energies and virials are served with the unscaled posterior
variance.</p>
<h3>3.2 Baselines</h3>
<p><b>B1 (posterior + conformal):</b> \(\Sigma=A^{-1}\) (the epistemic posterior) scaled by one conformal quantile \(\hat q\) of
the test-split \(|z|\) (the standardised scores of Section 4, here with the posterior shape), pooled over E, F and V with configurations weighted equally.
<b>B2 (POPS):</b> pointwise-optimal parameter sets; for training row \(i\) the minimum-norm correction that fits it exactly,
\(\delta_i=A_\mu^{-1}\psi^w_i\,r_i/h_i\) with leverage \(h_i=\psi_i^{w\mathsf T}A_\mu^{-1}\psi^w_i\). Here \(\psi^w_i=b_i w_i\) carries
structural weights only, \(r_i\) is the row's residual at the ridge solution, and \(A_\mu=M+\mu\operatorname{diag}(\Gamma_j^2)\) with
\(M=\sum_i\psi^w_i\psi_i^{w\mathsf T}\) and the ridge \(\mu\) chosen on a path; members are drawn uniformly in the PCA hypercube of the
\(\delta_i\). <b>B3:</b> CR3 with λ = 1.</p>""", "drafted from docs")

sec("method-calib", "4  Calibrating one global scale", r"""
<p>On the 248 held-out test configurations, every row \(b\) gives a standardised score \(z=(y-\hat y)/\lVert L^{\mathsf T}b\rVert\),
with \(\hat y=b\cdot\hat\theta\) (energies per atom, force components, virial components). The scale for observable type
\(q\in\{E,F,V\}\) is \(\lambda_q=\operatorname{RMS}(z_q)\), over that type's rows; \(\lambda_{\text{pool}}\) weights E, F and V equally within a configuration and
configurations equally. Rows that vanish by site symmetry (\(\lVert L^{\mathsf T}b\rVert\lt 10^{-3}\) of the median of their kind)
are dropped. No QoI, and no DFT QoI value, enters any scale.</p>
<div class="wrap"><table><caption>Table 2. Test-split scales (CR3 shape).</caption>
<thead><tr><th>model</th><th>λ<sub>E</sub></th><th>λ<sub>F</sub></th><th>λ<sub>V</sub></th><th>λ<sub>pool</sub></th><th>λ<sub>V</sub>/λ<sub>F</sub></th><th>λ<sub>E</sub>/λ<sub>F</sub></th></tr></thead>
<tbody>""" + lam_rows + r"""</tbody></table></div>
<p class="note">Both ratios fall outside [0.8, 1.25]: the degree of misspecification depends on the observable.
[Discuss: virials under-covered by λ<sub>F</sub>; consequence for elastic constants and phase energies.]</p>""", "drafted")

sec("method-prop", "5  Propagation to QoIs", r"""
<p>The distribution propagated is \(\theta\sim\mathcal N(\hat\theta,\lambda^2LL^{\mathsf T})\); one sample gives energy, forces and
stress consistently. Per-atom force ellipsoids are never combined.</p>
<h3>5.1 The QoIs</h3>
<p>The 21 QoIs and tolerances of the ace-jax silicon benchmark (py3 port of the silicon testing framework; CASTEP references):
C<sub>11</sub>, C<sub>12</sub>, C<sub>44</sub>, B; eight phase energies relative to diamond (compressed hcp excluded as extrapolation);
vacancy and three interstitial formation energies (216-atom cells); the relaxed (100) surface energy; the maximum unrelaxed
decohesion stress; the glide and shuffle unstable stacking-fault energies; the phonon RMS against DFT. Every protocol starts
from the model's own relaxed diamond cell, so no QoI is linear in θ. The phonon RMS is a distance to DFT and has no DFT
value: it is left out of coverage (n = 20).</p>
<h3>5.2 Delta method by implicit differentiation</h3>
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
<h3>5.3 Committees</h3>
<p>K = 200 draws \(\theta_k=\hat\theta+L\xi_k\), \(\xi_k\sim\mathcal N(0,I_r)\) (fixed seeds), are run through the unmodified benchmark harness. Per the protocol,
the draws are made once at λ = 1 and QoI deviations are rescaled, \(Q_k(\lambda)=\hat Q+\lambda(Q_k-\hat Q)\), with \(Q_k\) member \(k\)'s QoI and
\(\hat Q\) the mean model's; a full
re-run at λ<sub>F</sub> tests that linearisation. In that re-run a member is rejected, by a rule fixed beforehand that uses no
DFT value, if a test fails, if any relaxed interatomic distance falls below 2.0 Å (collapse), or if its diamond cell is
Born-unstable. The headline σ is the delta-method σ where it agrees with the λ = 1 committee to 10 %, else the committee's.</p>""", "drafted")

sec("method-disc", "5.4  Shared extrapolation bias: a discrepancy term for defect QoIs", r"""
<p>A bias shared by every training configuration of a kind is invisible to any delete-one jackknife: deleting one
configuration does not move the fit along it. The defect formation energies expose such a bias. Every training interstitial,
vacancy and divacancy cell is a hot MD snapshot, while the QoIs are relaxed defects.</p>
<p><b>Measuring the bias from training labels.</b> Each training defect snapshot \(i\) is paired with the bulk snapshot
nearest in volume and thermal amplitude, giving a difference row \(d_i=b_{\text{def},i}-(n_{\text{def}}/n_{\text{b}})\,b_{\text{bulk},i}\)
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
<p><b>A coarser jackknife.</b> As an alternative that stays in parameter space, the CR3 construction is repeated with whole GAP-18
config types as clusters (27 types). Clusters then span many batches, so the deletion is done exactly instead of by PRESS scores:
per-type QR statistics, and each leave-one-type-out mean from a QR merge of a prefix factor (prior plus the types before type \(k\)) and a
suffix factor (the types after \(k\)). The shape is the centred spread of those 27 means, scaled by its own test-split λ.</p>
<p><b>Estimator.</b> The intercept can be formed with each defect snapshot's nearest bulk snapshot as reference (paired; the form
that is also the synthetic row above) or with a bulk thermal trend fitted over all bulk cells near the defect volume (regression).
The headline uses regression for interstitials (data-only evidence, Table 7c) and, for the vacancy, regression per training-cell
size (63 and 215 atoms) combined by inverse variance, because the two sizes disagree and one pooled trend is rejected (Section 7.5b).</p>
<p><b>Discrepancy term.</b> For the QoIs of defect class \(t\), the variance becomes
\(\sigma_Q^2=\sigma_{Q,\text{method}}^2+r^{*2}_t+s^{*2}_t\), with \(\sigma_{Q,\text{method}}\) the σ of the UQ method without the term (interstitial class for the three interstitials; vacancy class for the
vacancy). Every other QoI is unchanged. This leaves the pure parameter-space propagation for these QoIs and is reported as a
separate, labelled method.</p>""", "drafted")

sec("method-reweight", "5.5  Changing the fit: defect–bulk difference rows and IRLS with automatic targets", r"""
<p>These change θ̂, so they address accuracy rather than the UQ. They test whether the defect bias is a fitting
compromise, which the weights can move, or a limit of the model. The prior (σ<sub>c</sub>) and shared noise stay at the
benchmark's; each variant is a weighted least-squares solve by QR of \([\operatorname{diag}\sqrt\Lambda;\,\Omega_w\Phi]\), with \(\Omega_w\) the
diagonal matrix of row weights (structural weight / noise scale, times the multipliers below).</p>
<p><b>Difference rows.</b> The rows \(d_i\) of Section 5.4, with target \(y_i\), weighted at \(k\) times the defect cell's own
energy-row weight (\(k\), the difference-row weight factor, is varied). <em>Augment</em> adds them to the design; <em>replace</em> also drops the defect cells' energy rows,
keeping their force and virial rows. A difference row is a linear combination of existing rows: it adds no configuration,
only weight on the defect–bulk contrast.</p>
<p><b>IRLS with automatic targets.</b> Row groups \(\ell\) (not the Mondrian groups) are (config type, quantity) for energies per atom,
force components and virials per atom, plus the difference rows as one group. Each group has a weight multiplier \(m_\ell\), an RMS
error \(\epsilon_\ell\) and a reference scale \(s_\ell\); the multipliers are updated to equalise the normalised errors,</p>
\[ e_\ell=\epsilon_\ell/s_\ell,\qquad m_\ell\leftarrow m_\ell\,(e_\ell/\bar e)^{\nu},\qquad m\leftarrow m/\bar m,\qquad m_\ell\in[10^{-2},10^{2}], \]
<p>with \(\nu=0.5\) and \(\bar e,\bar m\) geometric means over groups. Types with fewer than 10 training configurations keep
\(m_\ell=1\). Both \(\epsilon_\ell\) and the reference scales are 5-fold cross-validated over configurations within each type (by pair
for the difference rows). With training-set errors, a boosted group of a few dozen energy rows is interpolated by 224
parameters, its floor is near zero, and the balance cannot converge. Each (group, fold) is compressed once by QR, so a
cross-validated evaluation costs five small solves. Two choices of \(s_\ell\):</p>
<ul>
<li><b>Regret:</b> the group's cross-validated floor, its \(\epsilon_\ell\) when the fit is weighted towards it alone (×100). Balanced means
every group is equally far from the best this model can do for it. No QoI information enters.</li>
<li><b>Tolerance-guided:</b> 0.1 eV per defect cell for the difference rows and 10 meV/atom for energies (the defect and phase
QoI tolerances); forces and virials keep their regret floors. No DFT QoI value enters, but the weights are shaped by the
thresholds that score the QoIs, so its QoI results are optimistic by construction.</li>
</ul>
<p>IRLS starts from the augmented fit at \(k=1\), since the difference-row group starts at \(m=1\).</p>""", "drafted")

sec("method-valid", "6  Validation statistics", r"""
<ul>
<li>Per QoI \(z_Q=(\hat Q-Q_{\text{DFT}})/\sigma_Q\), with \(\hat Q\) the mean model's value, and coverage of Gaussian 68 % and 95 % intervals; counts with
Clopper–Pearson 95 % intervals (n = 20).</li>
<li>Joint test: \(\Delta^{\mathsf T}(C_Q+\epsilon I)^{-1}\Delta\), with \(\Delta\) the vector of \(\hat Q-Q_{\text{DFT}}\) over the scored
QoIs and \(C_Q=\lambda^2G\,LL^{\mathsf T}G^{\mathsf T}\) (delta method; \(G\) the QoI rows stacked) or the committee
covariance, on the correlation-scaled problem, \(\epsilon=10^{-3}\times\) mean diagonal (sensitivity 10<sup>−4</sup>, 10<sup>−2</sup>),
against \(\chi^2\) at the numerical rank.</li>
<li>Sharpness: median \(\sigma_Q/\text{tol}_Q\), with \(\text{tol}_Q\) the benchmark's tolerance for QoI \(Q\).</li>
</ul>""", "drafted")

phys_txt = ""
for m, fs in phys.items():
    phys_txt += (f"<p>{m}: the λ<sub>F</sub> re-run kept {fs['accepted']}/{fs['K']} members ({fs['rejected_collapse']} collapsed, "
                 f"{fs['rejected_failed']} failed). Outside the phase energies, σ from the physical members is {min(nonphase):.2f}–{max(nonphase):.2f}× the "
                 f"rescaled σ; the phase energies reach {min(phase):.1f}–{max(phase):.1f}× it, a genuine nonlinearity (phases relax into other "
                 "structures at λ<sub>F</sub>).</p>")
pops = pw.get(("o3d12", "pops"))
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
tight = (head["o3d12"]["median_sigma_over_tol"] / head["o4d20"]["median_sigma_over_tol"]) if all(head.get(m) for m in ("o3d12", "o4d20")) else float("nan")
sec("results", "7  Results", f"""
<p class="note">Status: both models complete at the BLR fit, including both POPS committees; the ARD acceptance run (PR #137) is summarised in Table 1. Every number below is read from
<code>bench/si_qoi_uq/results/</code>.</p>
<h3>7.1 Coverage and sharpness</h3>
<div class="wrap"><table><caption>Table 3. Coverage of the DFT values (count [95 % CI]) and sharpness.</caption>
<thead><tr><th>model</th><th>method</th><th>n</th><th>68 %</th><th>95 %</th><th>RMS z</th><th>median σ/tol</th></tr></thead>
<tbody>{cov_rows}</tbody></table></div>
<figure><img src="fig6_model_compare.png" alt="Per QoI, error over tolerance with 95 percent intervals for o4d20 and o3d12, DFT as crosses at zero"><figcaption>Figure 1. Model prediction minus DFT, in units of the benchmark tolerance, with the 95 % interval (CR3 × λ<sub>F</sub>) for o4d20 (circles) and o3d12 (squares); DFT is the cross at 0 and the shaded band is ± 1 tolerance. The axis is linear within ± 1 and logarithmic beyond. The larger model's intervals are {tight:.1f}× tighter in median σ/tol while its coverage stays comparable.</figcaption></figure>
<figure><img src="fig2_coverage.png" alt="Coverage per method with binomial intervals"><figcaption>Figure 1b. Coverage per method.</figcaption></figure>
<figure><img src="fig1_z_strip.png" alt="Standardised error per QoI"><figcaption>Figure 2. z<sub>Q</sub> per QoI (bands ±1, ±2).</figcaption></figure>
<h3>7.2 Joint test</h3>
<div class="wrap"><table><caption>Table 4. Mahalanobis test (correlation-scaled, ε = 10<sup>−3</sup>).</caption>
<thead><tr><th>model</th><th>method</th><th>S from</th><th>df</th><th>d²</th><th>χ²<sub>95</sub></th><th>p</th></tr></thead>
<tbody>{jrows}</tbody></table></div>
<p class="note">[Interpret: per-QoI coverage near nominal while the joint test rejects for λ<sub>F</sub> → errors correlated
beyond S; λ<sub>V</sub> passes.]</p>
<figure><img src="fig3_qoi_correlation.png" alt="QoI correlation matrix"><figcaption>Figure 3. QoI correlations from the CR3 committee.</figcaption></figure>
<h3>7.3 Delta method, committees and nonlinearity</h3>
<p>{"; ".join(f"{m}: {nflag[m]} of {ncmp[m]} QoIs differ by more than 10 % between delta and λ = 1 committee" for m in models)}.</p>
{phys_txt}
<figure><img src="fig4_elastic_ellipses.png" alt="Elastic constant covariance ellipses, two routes"><figcaption>Figure 4. Elastic constants: stress and energy routes.</figcaption></figure>
<h3>7.4 Baselines</h3>
<p>B2 with ace-jax defaults: the hypercube bounds come from low-leverage rows, and members spread the bulk energy (o3d12) by
{f(pops['sd_E_meV_per_atom'] if pops else None, 0)} meV/atom; almost every relaxation fails. Restricting the members to rows
at or above the median leverage gives physical members for both models, but no consistent width. The blow-up comes from a handful
of near-zero-leverage rows. Rejecting unphysical members cannot remove it: every full-box axis spans 10<sup>6</sup>–10<sup>7</sup> eV/atom of
bulk energy, so no member of either model is within 5 eV/atom of the mean on rattled diamond cells (0/200). A sweep of the
leverage cut (dropping the lowest 1, 2, 5, 10, 25 or 50 % of rows; members screened on three rattled diamond cells) shows that
dropping 1 % already gives physical members, and that the o4d20 width hardly depends on the cut (median worst-cell |ΔE|
0.001–0.003 eV/atom, against 0.017 for CR3 × λ<sub>F</sub>). Committees were run at the 5 % cut, which keeps closest to POPS's
default of every row, and at 50 %. Bulk-energy spread per member set (meV/atom), against the CR3 members at λ = 1 and at
λ<sub>F</sub>: {pops_w}. B2 covers {pops_cov}. POPS has no calibration step, so its width follows whatever the support selection gives.</p>
<figure><img src="fig5_scale_diagnostics.png" alt="Scale per observable and config type"><figcaption>Figure 5. λ per observable type and config_type.</figcaption></figure>""", "partial")

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
                          ("B1-headline", "q_b1", "B1")):
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
sec("results-defects", "7.5  Defect formation energies: diagnosis and the discrepancy term", f"""
<p>For both models the relaxed interstitial formation energies are the QoIs that are confidently wrong: all three are too
low, and outside the 95 % interval. They are low for every ACE and PACE model in the benchmark too, and the tetrahedral one
for GAP-18.</p>
<div class="wrap"><table><caption>Table 5. The training residual trend predicts the QoI error. Interstitial cells,
residual (DFT − model) minus a temperature-matched bulk reference, extrapolated to F<sub>rms</sub> → 0 (eV), against the QoI
errors (DFT − model) of the dumbbell / hexagonal / tetrahedral interstitials, and the mean over the hot training cells.</caption>
<thead><tr><th>model</th><th>extrapolated to F → 0</th><th>QoI errors</th><th>mean over hot cells</th></tr></thead>
<tbody>{trend_rows}</tbody></table></div>
<p>Over the hot cells the model's interstitial error averages near zero, so the CR3 intervals are calibrated there (test-split
z RMS 0.85–1.13 for those cells). The error grows towards the cold limit, and the relaxed defect lies beyond the data along
a bias that all interstitial clusters share. With per-atom energy weighting, 0.4–1.6 eV in a 217-atom cell costs only
2–7 meV/atom in the fit.</p>
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
<p class="note">For interstitials the data-only evidence favours the regression estimator. The paired form takes most partners
from 2-atom bulk cells, so a partner's residual enters ×108 (0.35 eV RMS, the size of the bias), its standard error is larger, and
it extrapolates worse to held-out cold snapshots (half the RMSE for regression, in both models; likewise for divacancies). The
model-against-model test does not separate them (both inside the true spread), and neither estimator handles the vacancy in any test.
<b>The headline therefore uses regression for interstitials;</b> the vacancy is treated in Section 7.5b. The regression estimator also agrees
better with the observed interstitial errors. That was seen before the data-only tests were run and is not part of the
argument, but it means the choice was not made blind; the all-paired result is kept as a sensitivity.</p>
<div class="wrap"><table><caption>Table 8. Defect QoIs, CR3 × λ<sub>F</sub>: error, σ without and with the discrepancy term
(eV), and z.</caption>
<thead><tr><th>model</th><th>QoI</th><th>DFT − model</th><th>σ</th><th>σ with term</th><th>z</th></tr></thead>
<tbody>{dz_rows}</tbody></table></div>
<p class="note">A delete-one-config-type jackknife, which stays in parameter space, is compared in Section 7.5c. Changing the fit
instead is reported in Section 7.6.</p>""", "partial")

sec("results-vacancy", "7.5b  The vacancy", f"""
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
<p class="note">Selection risk: the closeness of the 63-atom intercept to the vacancy QoI errors was seen before the heterogeneity and
model-against-model checks. The choice rests on those checks and on the cold cell, which use no DFT QoI value.</p>""", "partial")

tj = {m: json.load(open(RES / f"type_jackknife_{m}.json")) for m in ("o4d20", "o3d12") if (RES / f"type_jackknife_{m}.json").exists()}
tjz = {}
for r in rd("type_jackknife_o4d20.csv") + rd("type_jackknife_o3d12.csv"):
    if r["scale"] == "lam_F":
        tjz[(r["model"], r["shape"], r["qoi"])] = r["z"]
tj_rows = ""
for m, d_ in tj.items():
    for shape, lab in (("cr3", "delete one configuration (CR3)"), ("type", "delete one config type")):
        lam_ = d_["lam_cr3" if shape == "cr3" else "lam_type"]
        zs = " / ".join(f"{tjz.get((m, shape, k), float('nan')):+.1f}" for k in ("vacancy_energy", "dumbbell_interstitial_energy",
                                                                                  "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy"))
        a_, b_ = d_[f"{shape}/lam_F"], d_[f"{shape}/lam_V"]
        tj_rows += (f"<tr><td>{m}</td><td>{lab}</td><td class=n>{lam_['E']:.1f} / {lam_['F']:.1f} / {lam_['V']:.1f}</td>"
                    f"<td class=n>{a_['k95']} ({a_['k68']}) / {a_['n']}</td><td class=n>{b_['k95']} ({b_['k68']}) / {b_['n']}</td>"
                    f"<td class=n>{a_['median_sigma_over_tol']:.2f}</td><td class=n>{zs}</td></tr>")
if tj:
    sec("results-typejk", "7.5c  A coarser jackknife: deleting whole config types", f"""
<p>The defect bias is shared within config types, so a jackknife that deletes a whole type at a time might see it without leaving
parameter space. Delta-method coverage of the 17 QoIs with delta rows, at the type shape's own test-split scales.</p>
<div class="wrap"><table><caption>Table 8c. Delete-one-configuration (CR3) against delete-one-config-type (27 types). λ<sub>E</sub> /
λ<sub>F</sub> / λ<sub>V</sub> from the test split; 95 % (68 %) coverage with λ<sub>F</sub> and λ<sub>V</sub>; median σ/tol with λ<sub>F</sub>;
z (λ<sub>F</sub>) of the vacancy / dumbbell / hexagonal / tetrahedral QoIs.</caption>
<thead><tr><th>model</th><th>shape</th><th>λ (E / F / V)</th><th>λ<sub>F</sub></th><th>λ<sub>V</sub></th><th>σ/tol</th><th>defect z</th></tr></thead>
<tbody>{tj_rows}</tbody></table></div>
<p>The type shape needs far smaller scales (it captures held-out error better than per-configuration deletion). For o4d20 it brings
every defect QoI inside, without any discrepancy term, but at the price of intervals about 3× wider overall: nearly every QoI falls
within ±1σ against a nominal 68 %, so it over-covers. For o3d12 it does not reach the interstitials: there the hot-snapshot bias is
shared across the interstitial, vacancy and divacancy types alike, so deleting one type still leaves it in the fit. It is a conservative,
parameter-space alternative to the discrepancy term, not a calibrated fix.</p>""", "partial")

dr = json.load(open(RES / "diff_rows_o3d12.json")) if (RES / "diff_rows_o3d12.json").exists() else None
ib = json.load(open(RES / "irls_balance_o3d12.json")) if (RES / "irls_balance_o3d12.json").exists() else None
qv = {}
COVER_ORDER = list(dict.fromkeys(r["qoi"] for r in rd("diff_rows_o3d12_qoi.csv")))
for r in rd("diff_rows_o3d12_qoi.csv"):
    qv.setdefault(("diff", r["variant"]), {})[r["qoi"]] = r["err_over_tol"]
for r in rd("irls_balance_o3d12_qoi.csv"):
    qv.setdefault(("irls", r["scheme"]), {})[r["qoi"]] = r["err_over_tol"]
PH = [k for k in COVER_ORDER if k.startswith("dE_")]
INT = ["dumbbell_interstitial_energy", "hexagonal_interstitial_energy", "tetrahedral_interstitial_energy"]


def qsum(e):
    v = [e[k] for k in COVER_ORDER if isinstance(e.get(k), float)]
    worst = lambda ks: max((abs(e[k]) for k in ks if isinstance(e.get(k), float)), default=float("nan"))
    rms = (sum(x * x for x in v) / len(v)) ** 0.5 if len(v) == len(COVER_ORDER) else float("nan")
    return (f"<td class=n>{e.get('vacancy_energy', float('nan')):+.1f}</td><td class=n>{worst(INT):.1f}</td>"
            f"<td class=n>{worst(PH):.1f}</td><td class=n>{rms:.1f}</td><td class=n>{sum(abs(x) <= 1 for x in v)}/{len(COVER_ORDER)}</td>")


dr_rows = ""
if dr:
    for name, lab in (("base", "base (benchmark fit)"), ("augment_1", "augment, k = 1"), ("augment_3", "augment, k = 3"),
                      ("augment_10", "augment, k = 10"), ("augment_30", "augment, k = 30"), ("replace_3", "replace, k = 3"),
                      ("replace_10", "replace, k = 10")):
        v = dr["variants"][name]
        dr_rows += (f"<tr><td>{lab}</td><td class=n>{v['test']['E']:.1f}</td><td class=n>{v['test']['F']:.3f}</td>"
                    f"<td class=n>{v['test']['V']:.0f}</td><td class=n>{v['train_diff_rms_eV']:.2f}</td>"
                    f"<td class=n>{v['train_trend']['interstitial']['intercept_eV']:+.2f}</td>" + qsum(qv.get(("diff", name), {})) + "</tr>")
ib_rows = ""
if ib:
    for name, lab in (("regret", "IRLS, regret (no QoI input)"), ("tolerance", "IRLS, tolerance-guided")):
        v = ib["schemes"][name]
        ib_rows += (f"<tr><td>{lab}</td><td class=n>{v['test']['all']['E']:.1f}</td><td class=n>{v['test']['all']['F']:.3f}</td>"
                    f"<td class=n>{v['test']['all']['V']:.0f}</td><td class=n>{v['history'][-1]['spread']:.2f} ({len(v['history'])} it.)</td>"
                    f"<td class=n>{v['n_capped']}</td>" + qsum(qv.get(("irls", name), {})) + "</tr>")
sec("results-reweight", "7.6  Changing the fit: difference rows and IRLS (o3d12)", f"""
<p>Every number below is a mean-model harness run (no UQ). QoI columns give error/tolerance: the vacancy, the worst of the
three interstitials, the worst of the eight phase energies, the RMS over the 20 scored QoIs, and the number within tolerance.</p>
<div class="wrap"><table><caption>Table 9. Defect–bulk difference rows. Test-split RMSE (E, V in meV/atom; F in eV/Å), training
difference-row RMSE, and the zero-temperature intercept of the interstitial trend (paired-bulk reference, eV).</caption>
<thead><tr><th>fit</th><th>E</th><th>F</th><th>V</th><th>diff RMSE</th><th>F→0 intercept</th><th>vacancy</th><th>worst int.</th><th>worst phase</th><th>RMS</th><th>within tol</th></tr></thead>
<tbody>{dr_rows}</tbody></table></div>
<p>Weighting the defect–bulk contrast moves the relaxed-defect QoIs strongly: at k = 30 the vacancy error falls 8× and the
interstitials 4–6×, so the bias is largely a weighting compromise. On o3d12 general accuracy pays from k = 10, where the phase
energies degrade, and k = 3 is the best compromise. Replacing the defect energy rows gives the same result as augmenting, since
those rows carry little weight next to the difference rows.</p>
<div class="wrap"><table><caption>Table 10. IRLS with automatic, cross-validated targets (starting from the k = 1 augmented
fit). Spread is max/min of the normalised errors at the last iteration; capped counts multipliers at a bound.</caption>
<thead><tr><th>fit</th><th>E</th><th>F</th><th>V</th><th>spread</th><th>capped</th><th>vacancy</th><th>worst int.</th><th>worst phase</th><th>RMS</th><th>within tol</th></tr></thead>
<tbody>{ib_rows}</tbody></table></div>
<p>Regret balancing, with no QoI input, brings every phase energy inside tolerance and gives the most QoIs within tolerance,
but leaves the defects as they are. The tolerance-guided version improves the defects at the cost of the phases and the
surface, and its QoI results are optimistic by construction. Neither converged fully within 60 iterations. Regret balancing
cannot see the defect bias for the same reason the UQ cannot: on held-out hot snapshots the difference rows sit near their
floor (cross-validated RMSE 0.46 eV), and the error that matters is the extrapolation to relaxed geometries. A first IRLS
attempt with a fixed 0.05 eV difference-row target and no caps diverged, because the target is below the floor o3d12 can reach (about 0.14 eV).</p>""", "partial")

import numpy as _np
fc_rows = ""
fc_have = []
for m in ("o4d20", "o3d12"):
    pth = RES / f"force_calibration_{m}.npz"
    if not pth.exists():
        continue
    z = _np.load(pth, allow_pickle=True)
    fc_have.append(m)
    cov_all = float(_np.mean(z["score"] <= z["q"]))
    rr = lambda mk, z=z: float(_np.sqrt(_np.mean((z["err_norm"][mk] / z["forces_std"][mk]) ** 2)))
    allm = _np.ones(len(z["score"]), bool)
    fc_rows += (f"<tr><td>{m}</td><td>all</td><td class=n>–</td><td class=n>–</td><td class=n>–</td>"
                f"<td class=n>{len(z['score'])}</td><td class=n>{cov_all:.3f}</td><td class=n>{rr(allm):.2f}</td></tr>")
    for g in range(len(z["lam_rms"])):
        mk = z["group"] == g
        if not mk.any():
            continue
        fc_rows += (f"<tr><td>{m}</td><td>{g}</td><td class=n>{z['lam_rms'][g]:.2f}</td><td class=n>{z['q_table'][g]:.1f}</td>"
                    f"<td class=n>{int(z['n_cfg'][g])}</td><td class=n>{int(mk.sum())}</td>"
                    f"<td class=n>{float(_np.mean(z['score'][mk] <= z['q'][mk])):.3f}</td><td class=n>{rr(mk):.2f}</td></tr>")
sec("results-forces", "7.7  Calibrated per-atom forces (Mondrian conformal)", f"""
<p>Served per-atom force uncertainty against the observed error on the 248 test configurations, from the ARD fits of
PR #137 (the posterior files that hold the Mondrian groups and scales; models: {", ".join(fc_have) or "pending"}). Symmetry-zero
atoms are dropped, as in the scale diagnostics.</p>
<figure><img src="fig7_force_calibration.png" alt="Per-atom force uncertainty against observed error, binned on predicted sigma and on observed error, for each model"><figcaption>Figure 7.
Top: binned on predicted σ (20 equal-count bins), RMS observed |ΔF| against RMS σ, for the served Mondrian
<code>forces_std</code> and for the global λ<sub>F</sub> of the propagation, with the per-atom density behind; points below
the diagonal are conservative. Bottom: binned on observed |ΔF|. Binning on the observed error regresses towards the mean, so
the low-error bins read as under-confident and the high-error bins as over-confident even for a calibrated σ.</figcaption></figure>
<div class="wrap"><table><caption>Table 11. Per Mondrian group: the fitted scales (λ<sub>g</sub>, q<sub>g</sub> at α = 0.1),
calibration configurations, test atoms, test coverage of the anisotropic conformal region, and RMS |ΔF|/forces_std
(1 is calibrated in the Gaussian reading).</caption>
<thead><tr><th>model</th><th>group</th><th>λ<sub>g</sub></th><th>q<sub>g</sub></th><th>cal. configs</th><th>test atoms</th><th>coverage</th><th>RMS |ΔF|/σ</th></tr></thead>
<tbody>{fc_rows}</tbody></table></div>
<p class="note">The conformal regions hold their nominal 90 % on the test split overall and roughly per group. The Gaussian reading
of <code>forces_std</code> is conservative (RMS ratio below 1), because λ<sub>g</sub> is the RMS of heavy-tailed scores. The extreme case is
o4d20's bulk-like group 0 (half the test atoms): λ<sub>0</sub> is 4–6× the other groups' and its σ about 3.6× too wide, while its quantile
region still covers at the nominal rate. The RMS scale is dominated by a few extreme hold-out scores; the conformal quantile is
robust to them. Here λ<sub>F</sub>
was fitted on this same test split, so its line is in-sample for its overall level; λ<sub>g</sub> and q<sub>g</sub> come from the
training hold-out.</p>""", "partial")

sec("discussion", "8  Discussion", """
<ul class="plan">
<li>Take-away (moved from the abstract): calibrated parameter-space uncertainty transfers from forces to derived properties, except
where the training data share a systematic gap with the target structures, and that gap can be detected from the data.</li>
<li>A force-calibrated parameter covariance transfers to most QoIs at the 95 % level; median σ/tolerance """ + "; ".join(
    f"{m} {head[m]['median_sigma_over_tol']:.2f}" for m in models if head[m]) + """ (the accurate model's intervals are
inside the benchmark tolerance).</li>
<li>Observable-dependent scales: λ<sub>V</sub>/λ<sub>F</sub> > 1; choose λ by the QoI's physics (stress-like QoIs) or report both.</li>
<li>Linearisation: delta ≈ committee at λ = 1; at λ<sub>F</sub> phase energies are strongly nonlinear; robust spreads.</li>
<li>Why per-atom Mondrian scales do not propagate; a θ-space analogue would need group-dependent shapes.</li>
<li>Failure modes found: ARD floor (issue #136, fixed in PR #137), POPS leverage selection.</li>
<li>A delete-one jackknife cannot see a bias shared across all training configurations of a kind. For relaxed defects the
bias is measurable from training labels (zero-temperature extrapolation of the defect–bulk residual), but does not transfer
through a parameter-space correction; an observable-space discrepancy term restores coverage. Relaxed or low-temperature
defect data would remove the need for it.</li>
<li>Limitations: exchangeability of test split and QoI structures; hyperparameters fixed in the jackknife; locality as effective noise.</li>
</ul>""", "outline")

sec("methods-detail", "9  Computational details", """
<ul>
<li>Data: GAP-18 silicon, 2227 train / 248 test (seed 0, split hash c433c98d49bb). Models: ACE o4d20 (p = 5476) and o3d12 (p = 223),
r<sub>cut</sub> 5 Å, least-squares E<sup>0</sup>, shared noise, smoothness prior, float64.</li>
<li>Harness: py3 port of libAtoms/silicon-testing-framework, run unmodified with relaxed-ion elastic fmax 10<sup>−3</sup> eV/Å.</li>
<li>Code: <code>bench/si_qoi_uq</code> (branch silicon-uq-prop); seeds recorded in <code>results/run_metadata.json</code>.</li>
<li>Compute: lestrade (32 cores, RTX 4000 Ada), mnf148 (32 cores, 256 GB).</li>
</ul>""", "drafted")

sec("refs", "References (to verify)", """<ol class="refs">
<li>I. R. Best, T. J. Sullivan, J. R. Kermode, Uncertainty quantification in atomistic simulations of silicon using interatomic potentials, J. Chem. Phys. (2024). [check]</li>
<li>T. D. Swinburne, D. Perez, Parameter uncertainties for imperfect surrogate models in the low-noise regime, arXiv:2402.01810.</li>
<li>B. J. K. Kleijn, A. W. van der Vaart, The Bernstein–von Mises theorem under misspecification, Electron. J. Stat. 6, 354 (2012).</li>
<li>A. P. Bartók, J. Kermode, N. Bernstein, G. Csányi, Machine learning a general-purpose interatomic potential for silicon, Phys. Rev. X 8, 041048 (2018).</li>
<li>J. G. MacKinnon, M. Ø. Nielsen, M. D. Webb, Cluster-robust inference: a guide to empirical practice, J. Econometrics 232, 272 (2023).</li>
<li>J. G. MacKinnon, H. White, Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties, J. Econometrics 29, 305 (1985).</li>
<li>R. M. Bell, D. F. McCaffrey, Bias reduction in standard errors for linear regression with multi-stage samples, Survey Methodology 28, 169 (2002).</li>
<li>V. Vovk, D. Lindsay, I. Nouretdinov, A. Gammerman, Mondrian confidence machine, technical report, Royal Holloway, University of London (2003).</li>
<li>V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer (2005).</li>
<li>R. D. Cook, S. Weisberg, Residuals and Influence in Regression (1982).</li>
<li>G. Simmons, T. Hudson, T. D. Swinburne, J. R. Kermode, review of uncertainty quantification for machine-learned interatomic potentials, in preparation.</li>
<li>J. A. Hoeting, D. Madigan, A. E. Raftery, C. T. Volinsky, Bayesian model averaging: a tutorial, Stat. Sci. 14, 382 (1999).</li>
<li>A. Grossfield, D. M. Zuckerman, Quantifying uncertainty and sampling quality in biomolecular simulations, Annu. Rep. Comput. Chem. 5, 23 (2009).</li>
<li>T. Lelièvre, M. Rousset, G. Stoltz, Free Energy Computations: A Mathematical Perspective, Imperial College Press (2010).</li>
<li>D. M. Allen, The relationship between variable selection and data augmentation and a method for prediction, Technometrics 16, 125 (1974).</li>
<li>P. J. Huber (1967); H. White, Econometrica 50, 1 (1982); U. K. Müller, Econometrica 81, 1805 (2013).</li>
<li>R. Dunn, L. Wasserman, A. Ramdas, Distribution-free prediction sets for two-layer hierarchical models, JASA (2023).</li>
<li>R. Drautz, Atomic cluster expansion for accurate and transferable interatomic potentials, PRB 99, 014104 (2019).</li>
</ol>""")

toc = "".join(f'<li><a href="#{s.split(chr(34))[1]}">{s.split("<h2>")[1].split("<")[0].strip()}</a></li>' for s in S)
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
<p class="byline">Manuscript skeleton · ace-jax UQ preprint and MMM keynote §9 · generated from <code>bench/si_qoi_uq</code></p>
<h1>Calibrated parameter-space uncertainty for linear ACE, propagated to silicon properties</h1>
<p class="byline">[Authors] · Status chips: <span class="status draft">drafted</span> text complete for review; <span class="status outline">outline</span> argument only; <span class="status partial">partial</span> results still arriving.</p>
</header>
{''.join(S)}
</main></div></body></html>
"""
(OUT / "index.html").write_text(page)
for p in FIG.glob("*.png"):
    shutil.copy(p, OUT / p.name)
print("ok", OUT)
