# Paper revision plan: CRC, the narrowed headline and the next-steps results (2026-11-13)

Branch `revision` (draft PR #1; `main` stays as submitted). Source: `bench/si_qoi_uq/artifact/paper.py` (HTML), converted by
`bench/si_qoi_uq/latex/convert.py`, PDF and arXiv tarball by `latex/arxiv.py`. Results come from `next-steps`
(`docs/dev/plans/2026-11-10-next-steps.md`) and are copied into `bench/si_qoi_uq/results/` here with their scripts, as for the
trajectory-jackknife paragraph (fd440c4).

Decisions (author, 2026-11-13):
- Plan section 5 (CRC, calibrated-size selection, the o4d16 × 10 model, the step 3 decision rules) becomes a new results section.
- Headline narrowed, CR3 kept as the lead: CR3 for QoIs propagated without calibration data and for held-out energies; B1's
  "wrong shape" becomes wrong for propagation; for whole-configuration force regions under CRC, B1 is the sharper choice, at least
  as reliable. The per-atom force results (Section 3.2) are stated as properties of the per-atom conformal scale.
- The held-out CRC comparison is run on the paper's own models, o3d12 and o4d20 at hand weights, before writing (step 0).
- Also in: the vacancy barrier (§3), the ellipsoid (§2e), joint QoI scores, λ_mv (§4) and the DAS order (§1).

## Step 0: held-out CRC on o3d12 and o4d20 (next-steps)

`final.py ood`, `final_shapes.py` (CR3, ellipsoid) and `final_rules.py`, parametrised by model (today o4d16 only), candidate hand
only. Outputs: `results/final_ood_<m>.json`, `results/final_rules_<m>.json`; runs under `RUNS/final/<m>/ood/`. Order per model:
`final.py` (B1 posteriors) -> `final_shapes.py` cr3 and ellipsoid in parallel -> `final.py` again (all shapes, same splits).
Cost: o3d12 (P 224) minutes. o4d20 (P 5477): PRESS as `blr_cr3` o4d20 took; the ellipsoid design after the leverage cut is about
228k x 5477 (10 GB), so a 64 GB node with nothing else on it (mnf144 or dedicated123; not mnf148).

Reading, fixed before the runs (as rule 1 of plan section 5 D, step 3 follow-ups): per model, alpha 0.1 joint loss, views F and
E, d = risk(B1) - risk(CR3) over 100 paired splits, tau = sd / sqrt 2; also the F view's frac loss, ellipsoid - CR3, sizes, per
held-out type. The paper reports the per-model verdicts. The Section 3.2 sentence reads "after whole-configuration CRC, B1's
held-out force risk is no higher than CR3's" only if F is reversed or unresolved at both models; if CR3 is preferred on F at
either model, the paper says so for that model and the narrowing is stated per model. No outcome changes the decisions above.

## Step 1: methods (Section 2)

- 2.8 (new) *Conformal risk control over whole configurations.* Per configuration a region for its whole force (or energy)
  vector with the shape's covariance; loss: joint (any component outside) and frac (fraction of atoms outside); the shape fit
  (lam_s, nu) on A1, the CRC threshold on A2, risk and size (log geometric semi-axis) on B; the marginal guarantee. Views E, F,
  dE_type, dE_form. Cite Angelopoulos et al. (CRC); check the reference against the paper before adding.
- 2.9 (new) *Choosing fixed hyperparameters by calibrated size.* Declared priorities over views, one-SE tolerance on paired
  differences, lam for cost (A100 float64 µs/atom-step), minimax regret; nothing chosen on test or on any QoI.
- 2.6 baselines: the theory sentence of §2e item 4 (POPS correction A⁻¹xᵢrᵢ/hᵢ against the jackknife's A⁻¹xᵢrᵢ/(1 − hᵢ): the
  POPS moment is the CR3 sandwich reweighted by ((1 − h)/h)²; why the leverage cut).

## Step 2: results

- 3.2 forces: one paragraph after the per-atom results: they hold for a per-atom conformal scale; under whole-configuration CRC
  B1 gives smaller in-distribution regions (ladder) and, held out, the step 0 verdicts (o3d12, o4d20) with o4d16 as the third
  size.
- 3.5 joint evidence: d²/n and LS/n beside the hard count (tau 0.05, +disc as a correlated term: one shared interstitial bias);
  the λ_mv sentence (§4: at o4d20 covers 16/17 and is jointly calibrated without the discrepancy term, but scores worse than the
  headline).
- 3.7 baselines: the ellipsoid paragraph (§2e items 1–3, 5–6: needs the leverage cut; most even per-type force coverage with
  almost no scale; non-convex, the lower optimum not better downstream; QoI widths from gᵀΣg; per QoI slightly sharper, jointly
  worse). Table in Appendix B.
- 3.8 defects: the vacancy migration barrier as a subsection: DFT 0.299 eV; o4d20 misses by −0.20 eV (CR3 z 2.31, B1 4.55,
  B2 1.99), o3d12 covered via one β-tin cell; model-form error on seen environments, no tested fix removes it. Reported
  separately: it was added after the QoI results, so it does not enter the coverage counts or the joint tests (the abstract's
  property count is unchanged).
- 3.9 (new) *Calibrated region size: selection and the shapes under CRC.* (a) the ladder: B1 on the whole-configuration front at
  every size; (b) energy weight: × 10 over hand (energy-difference regions smaller, F larger), the evidence route best only for
  forces; (c) wL: default kept at matched A100 cost; (d) o4d16 × 10 refitted on T: risk at alpha on test; held out, × 10 cuts
  energy risk 0.71 -> 0.30; (e) the decision rules, stated as fixed in advance, and their outcome: rule 1 split by view (E for
  CR3, F reversed), rule 2, rule 3 for CR3 on the 18 QoIs (k95 18/18 against 15/18, d²/n 0.66 against 1.88), rule 4 narrowed;
  with step 0's o3d12 / o4d20 verdicts. One table (held-out risk and size by method and view, three sizes) and one figure
  (paired held-out risk differences with tau, per view and size).

## Step 3: abstract, introduction, discussion, conclusions

- Abstract: replace "The Bayesian posterior with one calibrated scale has the wrong shape: ..." with the narrowed form (wrong
  shape for propagating to properties: fewer covered, lower joint evidence; for whole-configuration force regions calibrated by
  CRC it is sharper and at least as reliable); one sentence on calibrated-size selection (energy weight × 10). Keep the per-atom
  force sentence, labelled as per-atom.
- Introduction: contributions list gains CRC and the selection rule.
- Discussion: rewrite the B1 paragraph (two uses, two answers); the force paragraph's "one conformal quantile" claims scoped to
  the per-atom scale; the CRC paragraph; the DAS order (§1: within 0.7 σ, P(7×7 lowest) 0.05–0.15) in one sentence pointing to
  Appendix C; limitations: the decision rules were fixed after the in-distribution CRC results were known.
- Conclusions: same narrowing.

## Step 4: appendices (new; the paper has no supplement)

A: CRC details (split protocol, losses, the dE_form infeasibility at alpha 0.05). B: ellipsoid per-type force table (L-BFGS
fits; evenness varies by ~0.02 between optima). C: Si(111) DAS / Pandey order probabilities. D: the step 3 decision rules
verbatim with the run dates.

## Step 5: build and checks

Every number from a results file (no hand-typed values in paper.py); `python -m bench.si_qoi_uq.artifact.paper`, the LaTeX
convert, `latexmk`, `latex.arxiv` (clean-room page count); reread the changed sections in the PDF; the PR #1 body updated.
Commits per step (`docs(si-qoi-uq): ...`), results and scripts copied in with the paragraph that uses them.

## Answered (author, 2026-11-14)

- Title: kept as is.
- Venue: arXiv v2 only (no journal submission yet), so there is no length limit; the appendices are in the paper.

## Done (2026-11-14)

Steps 0-5 on `revision` (561a9dd .. dd2c172). Deviations from the plan above: the POPS/CR3 leverage-weight sentence went in 2.4
(the comparison paragraph), not 2.6; the o3d12 vacancy barrier is covered with or without the beta-tin cell (only the vacancy
formation energy's coverage rests on it); a v1 error in 2.7.2 was fixed (no delta row for bcc at o3d12 but hcp at o4d20).
