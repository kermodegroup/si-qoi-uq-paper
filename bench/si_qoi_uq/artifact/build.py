"""Interim results page (artifact) from bench/si_qoi_uq/results/ only.  Not part of the pipeline.

    python bench/si_qoi_uq/artifact/build.py   -> bench/si_qoi_uq/artifact/index.html (+ figure copies)
"""
import csv, html, json, pathlib, shutil

HERE = pathlib.Path(__file__).parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
M = "o3d12"


def rd(name):
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(RES / name))]


def f(x, d=2):
    if x in ("", None):
        return "–"
    if isinstance(x, str):
        return html.escape(x)
    return f"{x:.{d}f}" if abs(x) < 1e4 else f"{x:.2e}"


def g(x, d=3):
    return "–" if x in ("", None) or isinstance(x, str) else f"{x:.{d}g}"


cov, sd, dvc, q = rd("coverage.csv"), rd("scale_diagnostics.csv"), rd("delta_vs_committee.csv"), rd("qoi_uq.csv")
mw = rd("member_widths.csv")
pf = rd("pops_committee_finished.csv")
joint = json.load(open(RES / "joint_tests.json"))
meta = json.load(open(RES / "run_metadata.json"))
get = lambda meth, sc: next((r for r in cov if r["model"] == M and r["method"] == meth and r["scale"] == sc), None)
lam = {r["kind"]: r for r in sd if r["model"] == M and r["factor"] == "cr3" and r["breakdown"] == "all"}

head = get("CR3-headline", "lam_F")
b3, b1, b2 = get("B3-headline", "lam_1"), get("B1-headline", "q_b1"), get("B2lev50-committee", "lam_1")

methods = [("CR3-headline", "lam_F", "CR3 × λ_F", "headline"), ("CR3-headline", "lam_V", "CR3 × λ_V", ""),
           ("CR3-headline", "lam_E", "CR3 × λ_E", ""), ("CR3-headline", "lam_pool", "CR3 × λ_pool", ""),
           ("CR3-delta", "lam_F", "CR3 × λ_F, delta method only", ""),
           ("CR3-committee", "lam_F", "CR3 × λ_F, committee only (rescaled)", ""),
           ("CR3-committee-lamF-full", "lam_F", "CR3 × λ_F, full re-run, all members", ""),
           ("CR3-committee-lamF-physical", "lam_F", "CR3 × λ_F, full re-run, physical members", ""),
           ("B3-headline", "lam_1", "B3: CR3, λ = 1", "baseline"), ("B1-headline", "q_b1", "B1: posterior × q̂", "baseline"),
           ("B2lev50-committee", "lam_1", "B2 variant: POPS, leverage pct 50", "variant"),
           ("CR3ell3-delta", "lam_F", "CR3, 3 r_cut clusters, delta", "sensitivity")]


def bar(frac, lo, hi, nominal):
    w = 160
    return (f'<svg class="cov" viewBox="0 0 {w} 14" role="img" aria-label="{frac:.2f} covered, CI {lo:.2f} to {hi:.2f}">'
            f'<rect x="0" y="5" width="{w}" height="4" rx="2" class="track"/>'
            f'<rect x="{lo * w:.1f}" y="3" width="{(hi - lo) * w:.1f}" height="8" rx="3" class="ci"/>'
            f'<line x1="{nominal * w:.1f}" x2="{nominal * w:.1f}" y1="0" y2="14" class="nom"/>'
            f'<circle cx="{frac * w:.1f}" cy="7" r="4" class="pt"/></svg>')


cov_rows = []
for meth, sc, lab, tag in methods:
    r = get(meth, sc)
    if not r:
        continue
    chip = f'<span class="chip {tag}">{tag}</span>' if tag else ""
    cov_rows.append(
        f"<tr><td>{lab} {chip}</td><td class=n>{int(r['n'])}</td>"
        f"<td class=n>{int(r['k68'])}</td><td>{bar(r['frac68'], r['ci68_lo'], r['ci68_hi'], 0.68)}</td>"
        f"<td class=n>{int(r['k95'])}</td><td>{bar(r['frac95'], r['ci95_lo'], r['ci95_hi'], 0.95)}</td>"
        f"<td class=n>{f(r['rms_z'])}</td><td class=n>{f(r['median_sigma_over_tol'])}</td></tr>")

SHORT = {"diamond_bulk_modulus": "B (GPa)", "diamond_c11": "C11 (GPa)", "diamond_c12": "C12 (GPa)",
         "diamond_c44": "C44 (GPa)", "vacancy_energy": "vacancy (eV)", "dumbbell_interstitial_energy": "dumbbell int. (eV)",
         "hexagonal_interstitial_energy": "hexagonal int. (eV)", "tetrahedral_interstitial_energy": "tetrahedral int. (eV)",
         "surface_100 (J/m2)": "(100) surface (J/m²)", "gamma_us glide (J/m2)": "γ_us glide (J/m²)",
         "gamma_us shuffle (J/m2)": "γ_us shuffle (J/m²)", "decohesion_max_stress": "decohesion σ_max (eV/Å³)",
         "phonon_f rms vs DFT (THz)": "phonon RMS vs DFT (THz)"}
short = lambda k: SHORT.get(k) or k.replace("dE_", "ΔE ").replace("(meV/at)", "(meV/atom)")
qrows = []
for r in [r for r in q if r["model"] == M and r["method"] == "CR3-headline" and r["scale"] == "lam_F"]:
    z = r.get("z")
    if z in ("", None):
        st, cls = "not scored", "na"
    else:
        a = abs(z)
        st, cls = ("within 1σ", "ok") if a <= 1 else (("within 2σ", "warn") if a <= 1.96 else ("outside 2σ", "bad"))
    d = next((x for x in dvc if x["model"] == M and x["qoi"] == r["qoi"]), {})
    meth = "committee" if (d.get("sigma_delta_lam1") in ("", None) or d.get("flagged") in ("True", True)) else "delta"
    qrows.append(f"<tr><td>{short(r['qoi'])}</td><td class=n>{g(r['mean'], 4)}</td><td class=n>{g(r['dft'], 4)}</td>"
                 f"<td class=n>{g(r['sigma'])}</td><td class=n>{g(r['tol'])}</td><td class=n>{f(z) if z not in ('', None) else '–'}</td>"
                 f"<td><span class='pill {cls}'>{st}</span></td><td class=dim>{meth}</td></tr>")

WARN10 = '<span class="pill warn">&gt; 10 %</span>'
dv_rows = []
for r in dvc:
    if r["model"] != M or r.get("sigma_delta_lam1") in ("", None) or "JAX committee" in r["qoi"]:
        continue
    fl = r["flagged"] in ("True", True)
    dv_rows.append(f"<tr class='{'flag' if fl else ''}'><td>{short(r['qoi'])}</td><td class=n>{g(r['sigma_delta_lam1'])}</td>"
                   f"<td class=n>{g(r['sigma_committee_lam1'])}</td><td class=n>{f(r['ratio'])}</td>"
                   f"<td class=n>{g(r.get('sigma_committee_robust_lam1'))}</td><td class=n>{f(r.get('ratio_delta_vs_robust'))}</td>"
                   f"<td>{WARN10 if fl else ''}</td></tr>")

j_rows = []
for name, r in joint.get(M, {}).items():
    for kind, jt in r.items():
        s = jt["scaled"]
        e = s["eps_0.001"]
        ok = e["mahalanobis"] <= e["chi2_95"]
        j_rows.append(f"<tr><td>{html.escape(name)}</td><td>{kind}</td><td class=n>{s['n']}</td><td class=n>{s['df']}</td>"
                      f"<td class=n>{f(e['mahalanobis'], 1)}</td><td class=n>{f(e['chi2_95'], 1)}</td>"
                      f"<td class=n>{e['p_value']:.2g}</td><td><span class='pill {'ok' if ok else 'bad'}'>{'consistent' if ok else 'rejected'}</span></td></tr>")

ph_rows, ph_note = [], ""
if (RES / "lamF_physical.csv").exists():
    for r in rd("lamF_physical.csv"):
        if r["model"] != M:
            continue
        rat = r["ratio_physical_vs_rescaled"]
        off = rat not in ("", None) and not (0.8 <= rat <= 1.25)
        ph_rows.append(f"<tr class='{'flag' if off else ''}'><td>{short(r['qoi'])}</td><td class=n>{g(r['sigma_rescaled'])}</td>"
                       f"<td class=n>{g(r['sigma_full_all'])}</td><td class=n>{g(r['sigma_full_physical'])}</td>"
                       f"<td class=n>{f(rat)}</td><td class=n>{g(r['robust_rescaled'])}</td><td class=n>{g(r['robust_full_physical'])}</td>"
                       f"<td class=n>{int(r['K_finished_physical'])}</td></tr>")
    fs = json.load(open(RES / "lamF_physical_filter.json"))[M]
    ph_note = (f"Accepted {fs['accepted']} of {fs['K']} members. Rejected: {fs['rejected_failed']} for a failed test, "
               f"{fs['rejected_collapse']} for a relaxed distance below {fs['d_min_threshold']:g} Å, {fs['rejected_unstable']} for "
               f"unstable diamond (Born); the reasons overlap. The mean model's shortest relaxed distance is {fs['mean_model_d_min']:.2f} Å. "
               "Accepted at 1.8 / 2.0 / 2.2 Å: " + " / ".join(str(v) for v in fs["accepted_at_threshold"].values()) + ".")
el_rows = []
for k in ("diamond_c11", "diamond_c12", "diamond_c44"):
    a = next(r for r in q if r["model"] == M and r["qoi"] == k and r["method"] == "CR3-delta" and r["scale"] == "lam_F")
    b = next(r for r in q if r["model"] == M and r["qoi"] == k and r["method"] == "CR3-delta-energy-route" and r["scale"] == "lam_F")
    el_rows.append(f"<tr><td>{short(k)}</td><td class=n>{f(a['mean'], 1)}</td><td class=n>{f(a['dft'], 1)}</td>"
                   f"<td class=n>{f(a['sigma'])}</td><td class=n>{f(b['sigma'])}</td></tr>")

LAB = {"cr3": "CR3, λ = 1", "cr3lamF": "CR3, λ_F", "posterior": "posterior (B1 shape, unscaled)", "pops": "POPS default (auto ridge 1e-14)",
       "popsblr": "POPS, ridge blr", "popslev50": "POPS, leverage pct 50", "popsblrlev50": "POPS, ridge blr, leverage pct 50"}
mw_rows = []
for r in mw:
    if r["model"] != M:
        continue
    fin = [x for x in pf if x["model"] == M and x["set"] == r["set"] and not x["qoi"].startswith("phonon")]
    run = f"{int(fin[0]['K_attempted'])}/200 run, {sum(x['K_finished'] >= 50 for x in fin)}/20 QoIs usable" if fin else ""
    bad = r["sd_E_meV_per_atom"] > 1e3
    mw_rows.append(f"<tr class='{'flag' if bad else ''}'><td>{LAB.get(r['set'], r['set'])}</td><td class=n>{g(r['sd_E_meV_per_atom'])}</td>"
                   f"<td class=n>{g(r['mean_sd_F_eV_A'])}</td><td class=dim>{run}</td></tr>")

joint_ok = ", ".join(f"{n} ({k})" for n, r in joint.get(M, {}).items() for k, jt in r.items()
                     if jt["scaled"]["eps_0.001"]["mahalanobis"] <= jt["scaled"]["eps_0.001"]["chi2_95"]) or "none"
lv, lf, le, lp = lam["V"]["rms_z"], lam["F"]["rms_z"], lam["E"]["rms_z"], lam["pool"]["rms_z"]

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silicon QoI uncertainty</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400&display=swap" rel="stylesheet">
<style>
:root {{
  /* lab notebook: one reading column, tables as the instrument readout, figures full width */
  --bg: #f7f8f8; --panel: #ffffff; --ink: #14181c; --muted: #5a636b; --rule: #dde2e6;
  --accent: #1f5f8b; --ok: #2f7d4f; --warn: #a86b00; --bad: #b3362f; --track: #e3e7ea;
  --display: "IBM Plex Serif", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #11161a; --panel: #182026; --ink: #e6ebee; --muted: #9aa6ae; --rule: #2a343b;
  --accent: #6fb0de; --ok: #6cc28e; --warn: #e0a640; --bad: #ec7a72; --track: #2a343b; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #11161a; --panel: #182026; --ink: #e6ebee; --muted: #9aa6ae; --rule: #2a343b;
  --accent: #6fb0de; --ok: #6cc28e; --warn: #e0a640; --bad: #ec7a72; --track: #2a343b; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 20px 64px; display: grid; gap: 28px; }}
h1, h2 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }}
h1 {{ font-size: 30px; line-height: 1.2; }} h2 {{ font-size: 20px; }}
p {{ margin: 0; max-width: 72ch; }} .dim {{ color: var(--muted); }} code {{ font-family: var(--mono); font-size: 13px; }}
section {{ display: grid; gap: 12px; }}
.status {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.tag {{ font-size: 12.5px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--rule); background: var(--panel); }}
.tag.done {{ border-color: var(--ok); color: var(--ok); }} .tag.run {{ border-color: var(--warn); color: var(--warn); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.tile {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; display: grid; gap: 4px; }}
.tile b {{ font: 600 26px/1.1 var(--display); font-variant-numeric: tabular-nums; }}
.tile.lead {{ border-left: 4px solid var(--accent); }} .tile.neg {{ border-left: 4px solid var(--bad); }}
.wrap {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: middle; }}
th {{ font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
tr:last-child td {{ border-bottom: 0; }} td.n {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
tr.flag td {{ background: color-mix(in srgb, var(--warn) 9%, transparent); }}
.pill {{ font-size: 12px; padding: 1px 8px; border-radius: 999px; white-space: nowrap; border: 1px solid currentColor; }}
.pill.ok {{ color: var(--ok); }} .pill.warn {{ color: var(--warn); }} .pill.bad {{ color: var(--bad); }} .pill.na {{ color: var(--muted); }}
.chip {{ font-size: 11px; color: var(--muted); border: 1px solid var(--rule); border-radius: 4px; padding: 0 5px; margin-left: 4px; }}
.chip.headline {{ color: var(--accent); border-color: var(--accent); }}
svg.cov {{ width: 160px; height: 14px; display: block; }} svg.cov .track {{ fill: var(--track); }}
svg.cov .ci {{ fill: color-mix(in srgb, var(--accent) 35%, transparent); }} svg.cov .pt {{ fill: var(--accent); }}
svg.cov .nom {{ stroke: var(--ink); stroke-width: 1.2; stroke-dasharray: 2 2; }}
figure {{ margin: 0; background: #fcfcfb; border: 1px solid var(--rule); border-radius: 8px; padding: 10px; display: grid; gap: 6px; }}
figure img {{ max-width: 100%; height: auto; display: block; }} figcaption {{ font-size: 13px; color: #52514e; }}
ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 80ch; }}
.flag-box {{ border-left: 4px solid var(--warn); background: var(--panel); padding: 12px 16px; border-radius: 0 8px 8px 0; display: grid; gap: 6px; }}
</style></head>
<body><main>
<header style="display:grid;gap:10px">
  <p class="dim">ace-jax · linear ACE parameter uncertainty propagated to 21 silicon QoIs · interim, {html.escape(meta['git'][:10])}</p>
  <h1>Do force-calibrated CR3 intervals cover DFT silicon properties?</h1>
  <div class="status"><span class="tag done">o3d12: complete</span><span class="tag run">o4d20: ARD fits running (CPU ×2, GPU racing)</span>
  <span class="tag">K = 200 members per method</span><span class="tag">DFT: CASTEP, silicon testing framework</span></div>
  <p>For o3d12 (223 coefficients), the CR3 jackknife covariance scaled by λ_F covers the DFT value for <b>{int(head['k95'])} of {int(head['n'])}</b> QoIs at 95 %
  and <b>{int(head['k68'])} of {int(head['n'])}</b> at 68 %. λ_F comes from test-split forces only; no QoI or DFT QoI value was used to tune anything. The intervals are wide: the median σ is {f(head['median_sigma_over_tol'], 1)} × the harness tolerance.</p>
</header>

<div class="tiles">
  <div class="tile lead"><span class="dim">CR3 × λ_F, 95 % coverage</span><b>{int(head['k95'])}/{int(head['n'])}</b><span class="dim">CI {f(head['ci95_lo'])}–{f(head['ci95_hi'])}</span></div>
  <div class="tile"><span class="dim">B3, uncalibrated (λ = 1)</span><b>{int(b3['k95'])}/{int(b3['n'])}</b><span class="dim">RMS z = {f(b3['rms_z'], 1)}</span></div>
  <div class="tile"><span class="dim">B1, posterior × one q̂</span><b>{int(b1['k95'])}/{int(b1['n'])}</b><span class="dim">q̂ = {f(b1['lam'], 1)}</span></div>
  <div class="tile neg"><span class="dim">B2, POPS as specified</span><b>failed</b><span class="dim">members unphysical; variant: {int(b2['k95'])}/{int(b2['n'])}</span></div>
</div>

<section><h2>Scales from the test split</h2>
<p>λ is the RMS of z = (y − ŷ)/‖Lᵀb‖ over 248 held-out configurations, with L = D⁻¹R the CR3 factor. The ratios fall outside [0.8, 1.25], so the misspecification depends on the observable:
λ_V/λ_F = {f(lv / lf)} and λ_E/λ_F = {f(le / lf)}.</p>
<div class="wrap"><table><thead><tr><th>observable</th><th>rows</th><th>λ (RMS z)</th><th>68 % |z|</th><th>95 % |z|</th></tr></thead><tbody>
{''.join(f"<tr><td>{n}</td><td class=n>{int(lam[k]['n'])}</td><td class=n>{f(lam[k]['rms_z'])}</td><td class=n>{f(lam[k]['q68'])}</td><td class=n>{f(lam[k]['q95'])}</td></tr>" for k, n in (('E', 'energy per atom'), ('F', 'force component'), ('V', 'virial component'), ('pool', 'pooled, equal weight per configuration')))}
</tbody></table></div>
<p class="dim">The pooled row counts configurations. Force rows that vanish by site symmetry are dropped (σ below 10⁻³ × the median σ). λ_F is 5.5–5.8 for thresholds from 10⁻¹ to 10⁻³.</p>
</section>

<section><h2>Coverage of the DFT values (n = 20)</h2>
<p>Each bar shows the covered fraction (dot) with its 95 % Clopper–Pearson interval; the dashed tick marks the nominal level. The phonon RMS has no DFT value and is not scored.</p>
<div class="wrap"><table><thead><tr><th>method</th><th>n</th><th>k₆₈</th><th>68 % interval</th><th>k₉₅</th><th>95 % interval</th><th>RMS z</th><th>median σ/tol</th></tr></thead>
<tbody>{''.join(cov_rows)}</tbody></table></div>
</section>

<section><h2>Per QoI: CR3 × λ_F</h2>
<div class="wrap"><table><thead><tr><th>QoI</th><th>model Q̂</th><th>DFT</th><th>σ</th><th>tolerance</th><th>z</th><th>DFT inside</th><th>σ from</th></tr></thead>
<tbody>{''.join(qrows)}</tbody></table></div>
<p class="dim">σ comes from the delta method (implicit differentiation through every relaxation, including a₀(θ)) where it agrees with the λ = 1 committee to 10 %, else from the committee.</p>
</section>

<section><h2>Joint test against χ²</h2>
<p>Mahalanobis distance of the 20 errors under the QoI covariance S (correlation-scaled, ridge 10⁻³ × mean diag S), compared with χ² at the numerical rank. Consistent at the 5 % level: {joint_ok}. All other sets are rejected jointly, even where their per-QoI coverage is near nominal.</p>
<div class="wrap"><table><thead><tr><th>method</th><th>S from</th><th>n</th><th>df</th><th>d²</th><th>χ²₉₅</th><th>p</th><th></th></tr></thead>
<tbody>{''.join(j_rows)}</tbody></table></div>
</section>

<section><h2>Delta method against committee (λ = 1)</h2>
<p>Robust σ is half the 16–84 % range of the committee. Where the delta σ matches the robust σ, the disagreement comes from a few members that relax into another structure. It is not curvature in the bulk of the committee.</p>
<div class="wrap"><table><thead><tr><th>QoI</th><th>σ delta</th><th>σ committee</th><th>ratio</th><th>robust σ</th><th>delta / robust</th><th></th></tr></thead>
<tbody>{''.join(dv_rows)}</tbody></table></div>
</section>

{('<section><h2>λ_F committee: rescaled against a full re-run</h2><p>The spec samples once at λ = 1 and rescales the QoI deviations by λ_F. Here the same 200 members, drawn at λ_F, are run through the harness. Members are then rejected by a rule fixed beforehand that uses no DFT value or QoI spread: a failed test, a relaxed interatomic distance below 2.0 Å (collapse), or unstable diamond. ' + html.escape(ph_note) + '</p><div class="wrap"><table><thead><tr><th>QoI</th><th>σ rescaled</th><th>σ full, all</th><th>σ full, physical</th><th>physical / rescaled</th><th>robust, rescaled</th><th>robust, physical</th><th>K</th></tr></thead><tbody>' + ''.join(ph_rows) + '</tbody></table></div><p class="dim">Rows shaded where the physical-member σ differs from the rescaled σ by more than 25 %. Robust σ is half the 16–84 % range.</p></section>') if ph_rows else ''}

<section><h2>Elastic constants by both routes (λ_F)</h2>
<div class="wrap"><table><thead><tr><th>QoI</th><th>Q̂</th><th>DFT</th><th>σ stress route</th><th>σ energy route</th></tr></thead><tbody>{''.join(el_rows)}</tbody></table></div>
</section>

<section><h2>Member widths: why B2 failed</h2>
<p>Spread over the 200 members of the energy per atom and the force components on a rattled 8-atom diamond cell. The default POPS hypercube takes its bounds from low-leverage rows, whose corrections are unbounded; almost every relaxation with those members fails.</p>
<div class="wrap"><table><thead><tr><th>member set</th><th>σ E (meV/atom)</th><th>σ F (eV/Å)</th><th>harness</th></tr></thead><tbody>{''.join(mw_rows)}</tbody></table></div>
</section>

<section><h2>Figures</h2>
<figure><img src="fig1_z_strip.png" alt="Standardised error per QoI for CR3 with lambda_F and lambda_V, B1 and the POPS variant"><figcaption>z = (Q̂ − Q_DFT)/σ per QoI, with bands at ±1 and ±2.</figcaption></figure>
<figure><img src="fig2_coverage.png" alt="Coverage fractions at 68 and 95 percent per method with binomial intervals"><figcaption>Coverage per method, with 95 % Clopper–Pearson intervals.</figcaption></figure>
<figure><img src="fig4_elastic_ellipses.png" alt="One-sigma covariance ellipses of C11 against C12 and C44 for the stress and energy routes, with the DFT point"><figcaption>Elastic constants: 1σ ellipses at λ_F, stress route (solid) and energy route (dashed), with the DFT point.</figcaption></figure>
<figure><img src="fig3_qoi_correlation.png" alt="Correlation matrix of the 20 QoIs from the CR3 committee"><figcaption>QoI correlation from the CR3 committee.</figcaption></figure>
<figure><img src="fig5_scale_diagnostics.png" alt="Scale lambda per observable type and per config_type on a log axis"><figcaption>λ per observable and per GAP-18 config_type (log scale).</figcaption></figure>
</section>

<section><h2>Choices and caveats</h2>
<div class="flag-box"><ul>
<li>No QoI is linear in θ: every harness protocol starts from the model's own relaxed lattice constant, so the delta method is implicit differentiation throughout.</li>
<li>The fits use sequential ARD with shared noise, to keep the benchmark's noise ratios. In joint mode the evidence weighted forces 29× over energies. The ARD precisions ended at their lower bound.</li>
<li>CR3 clusters are whole configurations. The default 3 r_cut blocks give the same scales (λ_F 5.764 against 5.762).</li>
<li>Committees run at λ = 1 and the QoI deviations are rescaled. A full re-run at λ_F produced unphysical members (vacancy energies down to −435 eV), so its SD is unreliable while its robust spread agrees.</li>
<li>Two ace-jax POPS issues: with the default QR statistics the mean is NaN, and the automatic ridge selection picks 1e-14 for E, F and V.</li>
</ul></div>
</section>
<p class="dim">Sources: <code>bench/si_qoi_uq/results/</code> (coverage.csv, qoi_uq.csv, scale_diagnostics.csv, delta_vs_committee.csv, joint_tests.json, member_widths.csv). Spec <code>docs/dev/specs/2026-10-31-silicon-qoi-uq-propagation.md</code>; decisions in <code>bench/si_qoi_uq/RECON.md</code>.</p>
</main></body></html>
"""
(HERE / "index.html").write_text(page)
for p in FIG.glob("*.png"):
    shutil.copy(p, HERE / p.name)
print("ok")
