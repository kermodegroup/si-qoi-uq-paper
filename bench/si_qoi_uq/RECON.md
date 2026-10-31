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
