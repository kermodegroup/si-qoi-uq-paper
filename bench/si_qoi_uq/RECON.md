# RECON: silicon QoI UQ propagation (spec §0)

Spec: `docs/dev/specs/2026-10-31-silicon-qoi-uq-propagation.md`. Recon done 2026-10-31 on
`silicon-uq-prop` (base `07abadc`). **Status: stopped before heavy compute.** Several findings
differ materially from the spec; they are listed under "Open questions" at the end.

## 1. Data, harness, references

| item | location | matches spec? |
|---|---|---|
| GAP-18 Si dataset | `/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/{train,test}.xyz`, from `gp_iter6_sparse9k.xml.xyz` (sha256 checked by `bench/accuracy_cost/split.py`) | yes |
| split | 2227 train / 248 test, `split.py` seed 0, hash `c433c98d49bb` (`bench/accuracy_cost/data.md`) | yes |
| QoI harness | **outside the repo**: `~/silicon-testing-framework` (py3 port, branch `py3`, HEAD `7817d21`), run as `test-runner/run-model-test.py <model> <test>`; model = `models/<name>/model.py` exposing `calculator` and `name` | yes, but see Q6 (no licence) |
| DFT refs | `~/silicon-testing-framework/test-results/model-CASTEP_ASE-test-*-properties.json`, read by `bench/si_qoi/compare.py::qois("CASTEP")` | yes |
| QoI list, tolerances | `bench/si_qoi/score.py::TOLERANCES` (rel 10 % elastic, 10 meV/atom phases, 0.10 eV defects, rel 5 % surface, rel 10 % γ_us and decohesion, 0.2 THz phonon rms) | yes |
| clip, exclusion | `score.py`: `GM_FLOOR, GM_CAP = 0.05, 10`; `EXCLUDED = {dE_hcp_short_and_fat}` (extrapolation) | yes |
| fitted base models | `/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace/{o4d20,o3d12}/` (p = 5476 and 223; `--uq blr`, `--noise shared`, `--e0 lsq`, rcut 5.0, weights `weights_acejax.json`) | models exist, but **no `--uq ard` fit**: refit needed (spec §1 expects this) |
| harness model files | `models/ACE_o4d20/model.py`, `models/ACE_o3d12/model.py`: `ACECalculator(<model.npz>, skin=0.0)` in float64. Note: `ACE_o4d20/model.py` has an uncommitted one-line comment deletion in the harness checkout. | |

Harness environment: `run_set.sh` uses `/storage/eng/essswb/venvs/si-tests/bin/python`,
`JAX_PLATFORMS=cpu OMP_NUM_THREADS=4`. **Elastic constants were re-run with
`SI_TESTS_ELASTIC_FMAX=1e-3`** (`tight_elastic.sh`): at the protocol default (ASE's fmax 0.05) the
internal relaxation left relaxed-ion C44 0.4–0.9 tolerance low. Committee runs must use the tight setting,
or optimiser noise enters σ(C44). Wall time: each test takes about 1–15 min per model (o4d20, float64, CPU).
A full 16-test harness run takes about 15–30 CPU-min serially, most of it JIT compilation plus the
216-atom defect relaxations.

## 2. UQ implementation

- `aj fit --uq ard` (`src/ace_jax/fit/ard.py`, `jackknife.py`, `conformal.py`, `clusters.py`). `feat/conformal-uq`
  is not an ancestor of HEAD, but its content is on main (squash-merged). Nothing to pick up from it.
- **Storage** (`posterior.npz`, `ARDPosterior.save/load`, schema 3):
  - θ̂ = `mean` (length L = p + NZ, because the E0 columns are in the coefficients when `--e0 lsq`);
  - `dinv` = D⁻¹ (prior scaling);
  - the CR3 shape factor = `R` (L × r, with Σ_CR3 = D⁻¹ R Rᵀ D⁻¹ in the original coordinates; `Q` is the
    legacy form). In spec notation, the low-rank factor is **L_spec = D⁻¹ R**;
  - `chol` (the posterior Cholesky factor of S = D⁻¹ A D⁻¹) is saved **only for the kappa shape**. The
    sandwich file drops it.
  - Hyperparameters `h` (ARD precisions per body-order group; noise σ_E, σ_F, σ_V), `kappa`, `lam`;
  - `group_table_json` = per-Mondrian-group `lam_rms` (λ_g) and `q` (q_g);
  - `cal_*` = the calibration scores, groups and configurations;
  - `meta_json` = the full report: the hold-out split `split.val_idx`, the strata, the transfer exponent and
    the leverage diagnostics.
- **λ_g and q_g temper forces only** (`"tempered_quantities": ["F"]`). E_var and V_var are served
  untempered. The spec computes its own λ_E, λ_F, λ_V and λ_pool, so this is consistent with the spec.
- **Calibration split:** the ARD λ_g/q_g come from a stratified 20 % hold-out of the **train** split
  (`--ard-val-frac 0.2`, a seed, strata = the most extreme Mondrian group per configuration). The
  248-configuration test split is never used by the fit. The spec's §3 diagnostics use the test split.
- **Mondrian groups:** distortion bands (p50/p90/p99 of std/mean of the first-shell bond lengths) ×
  [coordination ≠ z*], giving 8 groups (`--ard-groups distortion`).
- **CR3 clusters** (`clusters.row_clusters`): every cluster includes energy, force and virial rows. However,
  with the **default `--ard-cluster-size 3.0`** (blocks of 3 r_cut), a configuration's force rows are split
  into spatial blocks, and **its E and V rows form one separate cluster of their own**. They are
  kept together per configuration only with `--ard-cluster-size inf`. See Q3.
- **Epistemic variant:** `--ard-variance kappa` gives the posterior A⁻¹ shape (served as κ² × the posterior
  variance, then the same per-group scales), and the file keeps `chol`. B1 needs the posterior
  covariance without the jackknife, plus one pooled conformal q̂ that we compute. Plan: a second fit per
  model with `--ard-variance kappa`, or recompute A from the same data (same h). The first is simpler.
- **POPS:** `--uq pops` (`fit/pops.py`, `predict.PopsRidgePath`). Hypercube is the default posterior
  form, with the ridge chosen on a 20 % validation split. **No POPS posterior is written to disk.** Only
  the test envelope and the ridge scores are saved. B2 needs the members, so we would draw them in Python
  through `fit.pops` (`pops_posterior(..., form="hypercube")` gives the box covariance; the hypercube
  sampler is `_fit_hypercube_cov`'s support/low/high). This uses existing functions and changes no
  public API.

## 3. The 21 QoIs: class and protocol

None of the 21 is LINEAR as defined in the spec. Every harness test first relaxes the model's own
diamond cell (`relax_atoms_cell`). So every QoI depends on θ through at least the relaxed lattice
constant a₀(θ), and most also through internal or cell relaxations. There are no NEB or MD QoIs among the
21. (`vacancy-path-*`, `liquid` and `qha_diamond` exist in the harness but are not in `TOLERANCES`.)

| # | QoI | class | protocol (harness test) | non-smooth step |
|---|---|---|---|---|
| 1–3 | C11, C12, C44 | RELAXED (a₀ + internal) | `bulk_diamond`: symmetrised cell relax (fmax 1e-3); matscipy `fit_elastic_constants(symmetry="cubic")`, relaxed-ion, ±1 % strain in 5 steps, a stress–strain fit. **Stress route only.** | none |
| 4 | B = (C11+2C12)/3 | RELAXED | derived from 1–3 | none |
| 5–12 | dE_{β-Sn, sh, hex_diamond, bc8, st12, hcp, fcc, bcc} | RELAXED (cell + internal at fixed V) | `bulk_<phase>`: relax, then 12 constant-volume relaxations at V₀(1 + 0.025 i); dE = min over the sampled E(V) points minus the diamond minimum | **argmin over a discrete V grid** |
| 13 | vacancy | RELAXED (a₀ + 215-atom positions) | `vacancy-energy`: 3×3×3 cell, fmax 1e-3, rattled start (0.1 Å) | rattle uses a global RNG (unseeded) |
| 14–16 | dumbbell, hexagonal, tetrahedral interstitials | RELAXED (a₀ + 217-atom positions; hex with one z constraint, rotated cell) | `interstitial-energy`, tol 3e-3 | basin may change with θ |
| 17 | (100) surface | RELAXED (a₀ + positions) | `surface-energy-100-relaxed`: a 1×1×10 slab with +10 Å vacuum; L-BFGS, fmax 1e-3 | |
| 18 | decohesion σ_max | RELAXED (a₀ only) | `surface-decohesion-100-unrelaxed`: 5×1×1 cell, 36 rigid openings 0–3.5 Å; σ = d/dopening of a cubic spline; max abs over the grid | **max over grid** |
| 19–20 | γ_us glide, shuffle | RELAXED-PATH (a₀ + z-only relaxations along a fixed 7-image path) | `stacking-fault-surface` | **max over 7 images** |
| 21 | phonon rms vs DFT | RELAXED (a₀ from `bulk_diamond`) | `phonon_diamond`: phonopy, a 4×4×4 supercell of the 2-atom cell, dx 0.03 Å, 16³ mesh frequencies; QoI = RMS(f_model − f_DFT) over the mesh | **the QoI is a distance to DFT, not a quantity with a DFT value** (Q2) |

The force constants are linear in θ at a fixed geometry. The frequencies are their square roots, at the
θ-dependent a₀.

## 4. Elastic constants: the spec's two routes

The harness computes only the stress route (matscipy, relaxed-ion). An energy route (second
differences of E against strain) has to be written for this benchmark. It must use the same strains
(±1 %, 5 steps) and the same internal relaxation. C11 and C12 have no internal relaxation in diamond by
symmetry. C44 has relaxation (Kleinman ζ).

## 5. Compute estimate (committees)

| set | runs per model | note |
|---|---|---|
| CR3, λ = 1 (also B3) | 200 | the spec rescales by λ afterwards (see Q4) |
| B1 (posterior + pooled q̂) | 200 | |
| B2 (POPS hypercube) | 200 | |

That makes about 1200 full-harness runs for the two models. At 15–30 CPU-min each, it is about 300–600
CPU-h, **if** each run reloads the model. A driver that keeps one process per test and swaps
`calc.model` per sample (the readout swap does not retrace) removes most of the JIT cost. Re-running per
λ (Q4) multiplies the CR3 set by 4.

## Open questions (spec rule: record, continue conservatively; these are material, so stopped)

- **Q1. There are no LINEAR QoIs.** The spec puts the elastic constants under the delta method as
  "Q = Gθ". In the harness they are relaxed-ion constants at the model's own a₀(θ). The delta method
  therefore needs implicit differentiation (including a₀) for **all 21**, and each harness protocol must
  be re-implemented in JAX: discrete E(V) minima, a spline-max decohesion, z-relaxed γ_us paths, phonopy
  frequencies. This is the largest piece of work. Proposed conservative scope: implicit-diff delta
  for the elastic constants, the phases (smooth min of a quadratic fit through the E(V) points,
  with a flag where the argmin moves), the defects, the surface and decohesion; committee only
  for γ_us and phonons. Treat the discrete max/argmin steps as fixed-index (envelope theorem), and
  flag them in §4c.
- **Q2. The phonon QoI has no DFT value.** "phonon rms vs DFT" is ≥ 0, and its target is 0, so z_Q and
  interval coverage are undefined in the spec's sense. Options: (a) replace it with a few signed QoIs
  (for example the Γ optical, X and L frequencies) and count it as one QoI by a joint
  test; (b) exclude it from coverage and report only its predicted σ against its observed RMS.
  The conservative default is (b), and n becomes 20.
- **Q3. CR3 cluster definition.** The default `--ard-cluster-size 3` puts a configuration's E and V
  rows in a cluster separate from its force blocks. The spec expects whole configurations. The
  conservative choice is to follow the spec's expectation (`--ard-cluster-size inf`) for the headline,
  and record the default as a sensitivity run if time allows.
- **Q4. Rescaling committees by λ.** The QoIs are nonlinear in θ. Rescaling θ deviations after the
  fact needs a new harness run per λ. Rescaling QoI deviations is a linearisation. Conservative plan: run
  the committee at λ = 1, rescale QoI deviations for the four λ, and re-run one λ (λ_F) in full as a
  check, reported in `delta_vs_committee.csv`.
- **Q5. Seeds in the harness.** `vacancy-energy` calls `vac.rattle(0.1)` with ASE's fixed default seed
  (42). That is deterministic, so it is acceptable. The relaxations converge to fmax 1e-3, so the
  optimiser noise per sample has to be checked against σ (re-run the mean model twice with different
  rattles).
- **Q6. Licence and placement.** The harness has no licence, so (as `bench/si_qoi/compare.py` already
  notes) none of its code can be copied into ace-jax. The committee driver can run the harness scripts
  in place, and the delta-method re-implementations are written from the protocol descriptions above.
  The spec asks for `benchmarks/si_qoi_uq/`, but this repository keeps every benchmark under `bench/`.
  Proposed: `bench/si_qoi_uq/`, run as `python -m bench.si_qoi_uq.run --all`.
- **Q7. Compute placement.** The ARD fit of o4d20 (p = 5476; the BLR fit took 53 min and 10.5 GB on
  lestrade) plus a PRESS jackknife with exact per-cluster solves would exceed this session's 16 GB cap.
  Run the fits and committees detached on mnf148 or lestrade.

## Decisions (James Kermode, 2026-10-31)

- Q1: delta method (implicit differentiation) for the smooth subset: elastic constants (stress and
  energy routes), phases, defects, (100) surface and decohesion. γ_us and phonons are committee only.
  Discrete max/argmin steps use the fixed-index (envelope) rule and are flagged in §4c.
- Q2: the phonon RMS is excluded from coverage (n = 20). Its predicted spread is reported against the
  observed RMS separately.
- Q3: headline fits use `--ard-cluster-size inf` (whole configurations). The default (3 r_cut) is a
  sensitivity run.
- Q4: committees at λ = 1, with QoI deviations rescaled for each λ. One full re-run at λ_F checks the
  nonlinearity.
- Q6: code under `bench/si_qoi_uq/` (repository convention), run as `python -m bench.si_qoi_uq.run --all`.
- Fits: `--noise shared` (the accuracy-cost fits) is refused by `--uq ard` in joint mode, which
  refits σ_E, σ_F and σ_V by its own evidence. On o3d12 the joint fit chose σ_E/σ_F ≈ 29 (shared: 1), giving
  a force-dominated mean. Test RMSE joint ARD against the benchmark BLR: E 78.7 against 15.9 meV/atom,
  V 191 against 104 meV/atom, F 0.133 against 0.158 eV/Å. All three ARD precisions ended at their lower bound.
  **Decision (2026-10-31): `--ard-mode sequential --noise shared`**, which keeps the benchmark's noise
  ratios, with ARD on the precisions only (`fits.sh`). The joint-mode o3d12 outputs are kept in
  `fits/o3d12/joint_*` for comparison.

## Notes added during the run (2026-10-31)

- **Calibration data.** §3 sets λ_E, λ_F, λ_V and λ_pool from the 248-configuration test split, and §4 uses
  them. The "never tune on the test split" rule is read as forbidding tuning of the *fits* there: the fits
  never see it, and the QoIs (never used for tuning) are the validation. B1's q̂ uses the same test-split
  scores, so both methods are calibrated on the same data.
- **Symmetry-zero rows.** Force components that vanish by site symmetry (1-atom sh cells, surface_110 sites,
  interstitials on symmetric sites) have ‖Lᵀb‖ at roundoff, while the DFT label is not exactly zero. They
  are dropped from the scores if σ < 10⁻³ × the median σ of their kind (o3d12: 328 of 54179 rows).
  λ_F is stable from 10⁻¹ to 10⁻³ (5.53–5.76) and jumps to 47.6 at 10⁻⁴ (`diagnostics.py`).
- **POPS (B2).** With the default QR statistics, the final POPS path gave a NaN mean, so `pops_fit.py` passes
  `--lml-solver cholesky`. The auto ridge selection picked the grid's smallest relative ridge, 1e-14, for
  E, F and V. That is at the resolution of the eigendecomposition A is read from (test CRPS ~1e10). B2 is
  run as specified (auto ridge, force value) and also with `ridge="blr"` (B2blr). Both are reported.
- **Wide committees.** At λ_F the CR3 members include a tail of unphysical models (for example, vacancy
  energies down to −435 eV). The spec's SD-rescaled committee is kept as specified. Robust spreads (half
  the 16–84 % range) are reported beside it in `delta_vs_committee.csv`.
- **Harness E(V) points.** `lattice_tetragonal` skips E(V) points whose relaxation fails. Some wide samples
  then have fewer than 12 points (one has none: NaN). The QoI is the harness's own minimum over the points
  that remain.
- **POPS members (B2).** With the ace-jax default `pops_leverage_pct = 0`, the hypercube's min/max bounds
  are set by low-leverage rows, whose corrections r_i/h_i are unbounded. On o3d12 the members spread the
  bulk-diamond energy by about 7e9 meV/atom (auto ridge) and 5e8 meV/atom (ridge blr). Almost every harness
  relaxation then fails (`results/member_widths.csv`, `pops_committee_finished.csv`). These are reported
  as the spec's B2: failed. A labelled variant uses leverage percentile 50, keeping only rows at or above
  the median leverage (popsregression uses a leverage percentile; recalled, not re-checked): at the auto
  ridge its members spread 73 meV/atom and 0.031 eV/Å, and it is run in full (B2lev50). At ridge blr the
  members are still unphysical (1.6e5 meV/atom), so that set is not run.
- POPS default committees were stopped early, once their members had been shown to fail: `pops` after
  about 125 of 200 members (every relaxing test failed) and `popsblr` after about 14 (single relaxations ran
  for over 10 min). The exact counts are in `results/pops_committee_finished.csv`. That is below the spec's K ≥ 50 floor, which is meant for budget cuts, not for members that
  cannot be evaluated. They are reported as failed, with their attempted counts.
- **λ_F re-run with a physicality filter** (2026-10-31, at the user's request). The same 200 λ_F members were
  re-run with every relaxed structure recorded. They are filtered by a rule fixed before looking at its
  effect (`physical.py`): a failed test, a relaxed interatomic distance below 2.0 Å, or Born-unstable
  diamond. Results are in `results/lamF_physical.csv`, `lamF_physical_filter.json`, and the coverage rows
  `CR3-committee-lamF-{full,physical}`. The 2.0 Å threshold was set from silicon's 2.35 Å bond, not tuned. Note that 2.2 Å would
  reject 125 members, including members as compact as the mean model itself (2.24 Å).

## BLR posterior (decision, 2026-10-31): supersedes the ARD fits for the headline

- The o4d20 `--uq ard` fit (sequential, shared noise; GPU run) is degenerate. Its evidence optimiser never
  left the box floor: every a_k = a_floor = 4.30, the projected gradient is 0, and cond(S) ≈ 7e15 in all three evidence
  fits. Its mean, solved by a Cholesky of S, is far less accurate than the benchmark o4d20. Test RMSE: E 16.4
  against 3.5 meV/atom, F 0.125 against 0.082 eV/Å, V 59.6 against 29.7 meV/atom. o3d12 behaved the same,
  more mildly (E 21.4 against 15.9).
- Cause: the BLR fit is ARD at its start point (a_k = −2 log σ_c, `ARDEvidence.h0`), but 12 nats (o4d20) below
  the ARD floor log(λ_max/1e14). At the BLR prior, cond(S) is beyond a float64 Cholesky, and the BLR fit solves
  by QR instead.
- **Decision: keep the benchmark's own BLR fits** (θ̂ = their posterior mean at `theta_map.json`). Build the
  CR3 shape and the B1 posterior around them (`blr_cr3.py`): the A factor by `posterior_from_qr` (L Lᵀ = A),
  chol(S) = D⁻¹L in ARD's coordinates, then the existing whole-configuration `press_scores` and `shape_factor`.
  The QR mean reproduces the benchmark model's coefficients to 1.7e-6 relative (o3d12).
- Runs live in `runs/si_qoi_uq_blr` (`SIQ_RUNS`). The ARD-based runs stay in `runs/si_qoi_uq`, and their results
  are snapshotted in `runs/si_qoi_uq/ard_snapshot/` as an ablation. POPS needs no change (its mean is the BLR mean), so its fits and
  harness results are linked into the new root after checking that the members are identical.
- Not redone at the BLR fit: the 3 r_cut cluster sensitivity.
- **Pending: rerun with fixed ARD** (ace-jax #136; a fix is being prepared in another session). At n = 20 the o3d12
  difference is within noise. CR3 × λ_F at 95 %: ARD 19/20 [0.75, 1.00] (RMS z 1.03, median σ/tol 2.85) against
  BLR 17/20 [0.62, 0.97] (1.29, 2.52). The BLR intervals are sharper, and its mean is more accurate (o3d12
  E 15.8 against 22.0 meV/atom). When the fix lands, follow these steps:
  1. Refit both models with `fits.sh` (cr3 and kappa) into a fresh `SIQ_RUNS` root.
  2. Link the POPS fits.
  3. Rerun `run.py --stage diagnostics --stage samples`, `committee.sh` (cr3 with `--mean-first`,
     posterior, and cr3lamFrec with `SIQ_RECORD=1`), `--stage delta`, and `--all`.
  Compare against `runs/si_qoi_uq_blr` and `runs/si_qoi_uq/ard_snapshot`.
- Correction (from the PR 137 description): the benchmark o3d12 command has no `--test`, so the o3d12 RMSEs in
  issue #136 and in `runs/si_qoi_uq_blr/ard_floor_check/` (`data.ds_test`) are on the training set. They are
  still like for like. The o4d20 numbers there come from fits with `--test` and are on the test split.
- **PR 137 acceptance run** (2026-10-31): head 54e5274, in a detached checkout `.worktrees/pr137-accept`, run
  with `SIQ_ACEJAX_SRC=<checkout>/src` (`fits.sh`, `committee.sh`) into `runs/si_qoi_uq_ard137`. Same flags
  as the earlier ARD fits; the PR's defaults are QR solver and floor cond 1e22.

## Why the interstitial formation energies are confidently wrong (2026-10-31)

`analysis/defect_resid.py` and `analysis/defect_trend.py` → `results/defect_trend.json`. Each defect cell's energy error
(DFT − model), minus a bulk-diamond reference at the same F_rms², is regressed on F_rms² (thermal ⟨u²⟩). At F → 0
the interstitial error is +1.60 ± 0.06 eV (o3d12) and +0.41 ± 0.03 eV (o4d20). The relaxed-interstitial QoI errors
are +1.18/+1.45/+1.74 and +0.37/+0.41/+0.60 eV. All 103 training interstitial cells are hot MD (median RMS force
0.76 eV/Å), so the relaxed defect is an extrapolation along a bias shared by every interstitial cluster. The CR3
jackknife, the posterior and the committees cannot see it, while on training-like cells the intervals are calibrated (z RMS 0.85–1.13).
Divacancy shows the same trend (no QoI). The vacancy does not: −0.24 / +0.04 eV predicted, +0.11 / +0.88 eV observed.
Candidate UQ remedy, untested: jackknife clusters coarser than a configuration (whole config types).
- **PR 137, o4d20 cr3 fit** (GPU, 1 h 53 min): all four a_k interior (−12.0, −12.4, −8.7, −6.4; floor −14.1); not
  degenerate; cond(S) 1.7e21. Test split: E 3.38 meV/atom, F 0.0817 eV/Å, V 32.4 meV/atom, against the
  benchmark BLR's 3.50, 0.0816 and 32.6 (fit-log table for both). Note: issue #136's o4d20 table compared BLR
  29.7 (metrics.jsonl normalisation) with ARD 59.6 (fit-log table). Like for like, it is 32.6 → 59.6.

## Difference rows and IRLS (prototype, o3d12, 2026-11-01)

`analysis/diff_rows.py` → `results/diff_rows_o3d12.json` and `results/diff_rows_o3d12_qoi.csv`. Each defect training
snapshot is paired with the nearest bulk 'dia' snapshot in (volume, F_rms²), giving a row b_def − (n_def/n_b) b_bulk.
The fit uses the BLR prior and shared noise, solved by QR on explicit rows.
- Augmenting with these rows (weight k × the defect energy-row weight) flattens the interstitial trend:
  +1.04 → +0.46 (k = 3) → +0.15 (k = 10) eV.
- Defect QoI errors (err/tol) fall: vacancy −8.8 → −4.5 / −2.1 / −1.1 and tetrahedral −17.4 → −11.3 / −7.1 / −4.8 at
  k = 3 / 10 / 30.
- On o3d12 general accuracy pays from k = 10 (test E 15.9 → 20.3 meV/atom; bcc phase energy −3.1 → −14.6).
  k = 3 is the best compromise: RMS err/tol over the 20 QoIs is 6.2 → 4.0.
- Replacing the defect energy rows gives the same result as augmenting.
- IRLS with the default targets (base per-type RMSE, plus 0.05 eV per cell for the difference rows) diverged: the
  target is infeasible for o3d12, whose floor is about 0.14 eV. The resulting model fails the harness. Targets must be feasible, or the weights capped.
- Next: o4d20 via per-type QR statistics (explicit rows are too large at p = 5477).
- **IRLS with automatic targets** (`analysis/irls_balance.py` → `results/irls_balance_o3d12{.json,_qoi.csv}`). Groups:
  (config type, quantity) plus the difference rows. Weights are updated to equalise RMSE_g / s_g, with γ = 0.5,
  caps [0.01, 100] and types with fewer than 10 configs fixed. Errors and floors are 5-fold cross-validated by
  configuration: training-set floors are interpolated by 224 parameters for small energy groups (base/floor up to 3000).
  - Regret scales (each group's CV floor, no QoI input): all eight phase energies move inside tolerance; defects
    unchanged; 13/20 QoIs within tolerance; RMS err/tol 6.4.
  - Tolerance scales (0.1 eV difference rows, 10 meV/atom energies; circularity risk): defects much better (vacancy −2.5,
    interstitials −4.0 / −4.5 / −9.0) but phases and the surface worse; 9/20 within tolerance; RMS 2.8.
  - Why: training-distribution errors (even CV) cannot see the relaxed-defect extrapolation. Next candidate: one
    synthetic row per defect type, the F → 0 intercept of the difference residuals (linear in θ, DFT training labels only).
- **Shared-bias term in θ** (`analysis/bias_term.py` → `results/bias_term_o3d12{.json,.csv}`; θ̂ unchanged). There is one
  synthetic row per defect type, the F → 0 intercept functional of the paired difference rows (DFT training labels only).
  Its residual r* is +1.04 ± 0.09 eV for interstitials (CR3 × λ_F σ of the same row 0.69 eV), +1.74 ± 0.14 for divacancy and
  −0.03 ± 0.10 for vacancy. The minimum-norm correction δ of r* (CR3 or posterior metric) was added as Σ += δδᵀ, with λ re-solved
  on the test split with the bias part held fixed. Coverage did not change (14/17 at 95 % with λ_F): δ projects only
  0.1–0.2 eV of the interstitial r* onto the relaxed-defect QoI rows. The hot-snapshot contrast extrapolated to F → 0 is a
  different θ direction from the relaxed defect's formation row, so a rank-one correction from training snapshots
  cannot carry the bias to the QoI. Options: an observable-space discrepancy term σ² += r*² on defect QoIs
  (non-circular, but outside the pure θ framework), or reporting r* as the diagnostic for where intervals fail.
  - Note: IRLS starts from the difference rows at multiplier 1, so the "base_*" fields in `irls_balance_o3d12.json`
    describe the augmented k = 1 fit (test E 16.01 meV/atom, θ 5 % from the BLR mean), not the benchmark fit.
- **Discrepancy term, both models** (`analysis/discrepancy.py`, `results/discrepancy{,_regression}.*`; decision 2026-11-01):
  σ² += r*² + se² on the vacancy and interstitial QoIs. Paired estimator (headline; o4d20 interstitial r* +0.105 ± 0.046 eV,
  vacancy −0.266 ± 0.042): CR3 × λ_F at 95 % goes from 17 to 20/20 for o3d12 and from 15 to 16/20 for o4d20, where hexagonal and
  tetrahedral stay outside. The regression estimator (the bulk thermal trend of `defect_trend.json`) gives 20/20 and 18/20,
  with all defect z within ±1.3. It agrees better with the QoI errors for both models, and so is NOT chosen on that basis
  (that would tune on the DFT QoIs). It is reported as a sensitivity with the selection risk stated.
- **Estimator case without DFT QoIs** (`analysis/estimator_check.py` → `results/estimator_check.json`):
  - C. The paired form pairs most interstitial snapshots with 2-atom bulk cells, so n_def/n_b has median 108. The partner's
    residual × ratio has RMS 0.35 eV, the size of the bias. The interstitial intercept SE is 0.090 / 0.046 eV (paired)
    against 0.062 / 0.032 (regression), for o3d12 / o4d20.
  - B. Fitting on all but the coldest 20 % of defect snapshots (DFT training labels) and predicting the held-out cells,
    regression has the lower RMSE for interstitials (o3d12 0.20 against 0.45 eV, o4d20 0.10 against 0.23) and divacancies
    (0.30 against 0.76, 0.13 against 0.23). Paired is slightly better for vacancies (0.85 against 0.96, 0.37 against 0.42).
  - A. With o4d20 as the truth for o3d12 (harness QoIs of the models, no DFT), interstitial truth is 0.81 / 1.04 / 1.14 eV,
    against paired 0.93 and regression 1.19: inconclusive. Vacancy truth is 0.77 eV and both predict about 0.2.
  - Reading: a data-only case favours regression for interstitials (noise structure, SE, cold extrapolation); neither
    estimator handles the vacancy. A large-partner pairing variant (≥ 54 atoms) would remove the amplification, but it was
    defined after seeing these results.
- **Headline estimator (decision 2026-11-01):** regression for the interstitial class, paired for the vacancy
  (`discrepancy.py --estimator headline`, the default → `results/discrepancy.*`). All-paired and all-regression are kept as
  sensitivities. CR3 × λ_F with the term at 95 % (68 %): o3d12 20/20 (17), o4d20 18/20 (15).

## The vacancy (2026-11-01; `analysis/vacancy_check.py`, `results/vacancy_check.json`, `results/vacancy_size_split.json`)
- The training vacancy cells come in two sizes with different behaviour. Regression-form intercepts are 63-atom +0.838 ± 0.031
  against 215-atom +0.333 ± 0.122 eV (o3d12), and +0.160 ± 0.017 against −0.034 ± 0.045 (o4d20). The heterogeneity z is 4.0 in
  both models, so pooling the sizes under one trend, as both earlier estimators did, is rejected by the data. The 63-atom cells
  reach colder (5th percentile of F_rms² 0.165 against 0.356), and their intercept does not depend on the one cold cell.
- Inverse-variance combination of the per-size intercepts: +0.808 ± 0.030 (o3d12), +0.136 ± 0.016 (o4d20). Model against model
  (no DFT): +0.653 ± 0.026 against a true 0.77 eV, where the pooled estimators gave about 0.2.
- Direct, unextrapolated evidence: the one near-relaxed training vacancy cell (63 atoms, F_rms² 0.007) has bulk-subtracted
  residual +0.61 eV (o3d12) and 0.00 (o4d20). Relaxing it, o3d12 gains 0.34 eV against a DFT bound of 0.24 (o4d20: 0.18 against 0.18).
  The other "cold" cells are hot (max |F| 0.9–2.4 eV/Å), so the over-relaxation test is uninformative there.
- Geometry: both models relax the harness vacancy to a symmetric Td-like structure (six neighbour distances equal: 3.40 Å
  o3d12, 3.47 Å o4d20). There is no DFT structure to compare against.
- Selection risk: the 63-atom intercept's closeness to the vacancy QoI errors (+0.881 / +0.111 eV) was seen before the
  heterogeneity and model-against-model checks were run.
