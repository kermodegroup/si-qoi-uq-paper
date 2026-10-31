# Spec: calibration and propagation of linear ACE force uncertainty to silicon QoIs

For Claude Code, working in the ace-jax repository.
Owner: James Kermode. Target: results for the MMM keynote (section 9) and the ace-jax UQ preprint.

## Goal

Test whether the ace-jax parameter-space uncertainty for linear ACE, calibrated only on
held-out energies, forces and virials (never on the QoIs themselves), gives intervals
that cover DFT reference values for the 21 silicon quantities of interest (QoIs) in the
existing ace-jax Si benchmark.

Compare against two baselines:
- the Bayesian posterior with a single global conformal scale (the Best et al. 2024 protocol);
- POPS.

Core principle: propagate the distribution over parameters, θ ~ N(θ̂, λ²Σ), where Σ = LLᵀ
is the CR3 jackknife covariance. Do NOT try to combine per-atom force ellipsoids into
QoIs. One θ sample gives energy, forces and stress together and consistently.

## 0. Reconnaissance (do first; report findings before heavy compute)

1. Locate:
   - the GAP-18 Si dataset and its existing split (2227 train / 248 test);
   - the 21-QoI Si benchmark harness and its DFT (CASTEP) reference values;
   - the tolerances and the clipping/exclusion rules: per-QoI tolerance, clip to
     [0.05, 10], compressed hcp excluded.
2. Locate the UQ implementation (`aj fit --uq ard`; check the `feat/conformal-uq`
   branch status). Confirm:
   - how θ̂, the low-rank factor L, the per-group scales (λ_g, q_g) and the
     hyperparameters are stored;
   - whether the CR3 clusters include energy and virial rows (expected: yes, kept
     together per configuration);
   - how the "epistemic" (Bayesian posterior) variant is selected;
   - whether a POPS implementation exists and its API.
3. List the 21 QoIs and classify each one:
   - LINEAR: linear in θ at fixed geometry;
   - RELAXED: needs structural or cell relaxation;
   - PATH: NEB or barrier;
   - MD: finite temperature.
   Record the class and the computational protocol of each.
4. Write a short `RECON.md` with all the above. If anything differs materially from this
   spec (missing harness, different split, missing UQ storage), stop and report rather
   than improvising.

## 1. Fits

- Models:
  - o4d20: the model that hits the accuracy target in the Si benchmark;
  - o3d12: a smaller contrast model, more strongly misspecified.
- For each model, fit with `aj fit --uq ard` on the train split (default settings unless
  RECON shows a reason not to). Save:
  - θ̂;
  - L;
  - the hyperparameters (ARD precisions; noise scales for energy, force and virial);
  - λ_g and q_g;
  - the calibration split used.
- Run in float64 (`jax_enable_x64`).
- Fix and record all seeds.

## 2. Observable Jacobians

Implement `observable_rows(atoms, model)`. It returns the derivatives of each observable
with respect to θ:
- b_E: energy, shape (p,);
- B_F: forces, shape (3N, p);
- B_V: virial or stress, shape (6, p), Voigt order; state the sign and per-volume
  convention.

Use `jax.jacfwd` with respect to θ, or the existing design-matrix assembly if it already
does this.

Unit tests:
- linearity: E(θ) == b_E · θ to machine precision, and the same for F and V;
- agreement with finite differences in θ for a random direction;
- stress convention matches the fitted virial rows (same units and weighting).

Predictive standard deviation of any observable row b:
  σ(b) = λ · ‖Lᵀb‖

## 3. Scale diagnostics on the held-out test split

For each model, compute standardised scores separately for:
- energy per atom;
- force components;
- virial components.

  z = (y − ŷ) / ‖Lᵀb‖

Also break these down by GAP-18 config_type and by the Mondrian groups.

For each observable type, report:
- the RMS of z, which gives λ_type for an RMS-z of 1;
- the 68% and 95% empirical quantiles of |z|.

Define four candidate global scales:
- λ_F: from forces;
- λ_V: from virials;
- λ_E: from energies;
- λ_pool: from energy, force and virial scores pooled with equal weight per configuration.

Output `scale_diagnostics.csv` and a short table in the report. Flag it prominently if
λ_V/λ_F or λ_E/λ_F lies outside [0.8, 1.25]. That would mean the misspecification depends
on the observable, which is a finding in itself.

## 4. Propagation to QoIs

For each model, each QoI, and each scale choice (λ_F, λ_V, λ_E, λ_pool), compute the
predicted mean and standard deviation.

### 4a. Delta method (primary method where valid)

LINEAR QoIs:
- Q = Gθ, so Var(Q) = λ²‖LᵀGᵀ‖².
- Elastic constants: compute by BOTH routes and report both.
  - Stress route (preferred): first differences of stress with respect to strain.
  - Energy route: second differences of energy.
  - Use the harness strain magnitudes.
  - Give the full 3×3 covariance of (C11, C12, C44).

RELAXED QoIs:
- First-order implicit differentiation:
    dQ/dθ = ∂Q/∂θ − (∂Q/∂x) K⁻¹ (∂F/∂θ)
  - K is the Hessian with respect to atomic positions (and the cell, where relaxed).
  - Use a dense solve for small cells and CG with Hessian-vector products otherwise.
- Each sample's QoI must be taken at that model's own relaxed geometry and lattice
  constant, matching the DFT protocol. The implicit derivative must therefore include
  the lattice-constant response.
- Use a custom_root or implicit-differentiation rule. Do not differentiate through the
  optimiser's iterations.

PATH and MD QoIs: committee only (4b).

### 4b. Committees (reference for all QoIs)

- Draw K = 200 samples θ_k = θ̂ + λ L ξ_k, with ξ_k ~ N(0, I) and a fixed seed.
  Sample once with λ = 1 and rescale the deviations for each λ choice afterwards.
- Run the full harness protocol for every QoI and every sample.
- If any QoI exceeds the compute budget, reduce K for that QoI only (minimum 50) and
  record it.
- Report the sample mean, the sample standard deviation and the 2.5% and 97.5%
  percentiles.

### 4c. Consistency check

For the LINEAR and RELAXED QoIs, compare σ from the delta method with σ from the
committee. Flag any QoI where they differ by more than 10%; that indicates nonlinearity.
Use the committee value for flagged QoIs in the headline results.

## 5. Validation against DFT

For each model, scale choice and method, compute per QoI:
- z_Q = (Q̂ − Q_DFT) / σ_Q;
- whether the DFT value lies inside the 68% and 95% intervals.

Then:
- **Coverage:** the count and fraction of the 21 QoIs inside each interval (minus any
  excluded by the harness rules). With n ≈ 21, report counts and binomial 95%
  confidence intervals, not just fractions.
- **Joint test:**
  - Build the QoI covariance S = λ² G_all L Lᵀ G_allᵀ (for linear and relaxed QoIs via
    the delta method; otherwise use the committee covariance).
  - Compute the Mahalanobis statistic dᵀ(S + εI)⁻¹d, where d is the vector of
    differences from DFT.
  - Use a ridge ε chosen as 1e-3 × the mean diagonal of S, and report its sensitivity
    to ε.
  - Compare against χ² with an effective number of degrees of freedom (the numerical
    rank).
- **Sharpness:** the median σ_Q / tolerance_Q, using the harness tolerances.

## 6. Baselines (same QoI pipeline, same seeds, same K)

- **B1, Best 2024 protocol:**
  - Bayesian posterior covariance (the epistemic variant), no jackknife.
  - Rescaled by ONE conformal quantile q̂ pooled over energy, force and virial scores on
    a held-out calibration split.
  - Pushed through the same propagation.
- **B2, POPS:**
  - The existing ace-jax POPS with hypercube sampling, K = 200 members.
  - Committee propagation for all QoIs.
- **B3, uncalibrated:** CR3 shape with λ = 1. This shows how much calibration matters.

## 7. Optional (only after 1–6 are complete)

QoI-level calibration by leaving one QoI out at a time:
- for each QoI i, set λ from the z-scores of the other 20 QoIs;
- predict QoI i;
- report the coverage obtained.

Label it clearly as a diagnostic, since the guarantee is weak with 21 points. It must
not be used for the headline numbers.

## Outputs

Put everything under `benchmarks/si_qoi_uq/`, with code in a module and tests under
`tests/`. Do not change public APIs.

- `RECON.md`
- `results/`:
  - per-QoI CSV: model, method, scale, mean, σ, intervals, DFT value, z, covered flags;
  - `scale_diagnostics.csv`;
  - `delta_vs_committee.csv`;
  - `joint_tests.json`;
  - `run_metadata.json`: git hash, package versions, seeds, hardware, wall times.
- `figures/` (PNG and PDF):
  1. Strip plot of z_Q per QoI, coloured by method (CR3 with λ_F, CR3 with λ_V, B1, B2),
     with bands at ±1 and ±2.
  2. Coverage bars (68% and 95%) per method with binomial error bars.
  3. Heatmap of the QoI correlation matrix from S (o4d20, CR3).
  4. Elastic constants: the 3×3 covariance ellipses (C11–C12, C11–C44) with the DFT
     point, comparing the stress and energy routes.
  5. Scale diagnostics: λ_type per observable type and per config_type.
- `REPORT.md`, two pages at most:
  - headline result (one sentence);
  - tables;
  - the figures, embedded;
  - flagged issues (nonlinearity, observable-dependent scales, any failed QoIs);
  - a clear statement of what was and was not calibrated on.

## Rules

- Never tune anything on the test split or on the DFT QoI values, except in the
  clearly labelled optional section 7.
- Use the harness's DFT reference values exactly, not literature values.
- Report negative results plainly (for example, if one λ does not cover the virials,
  or if coverage fails for relaxed defects).
- Every number in `REPORT.md` must trace to a file in `results/`.
- If a step is ambiguous or the code differs from this spec, write the question into
  `RECON.md` and continue with the most conservative choice. Do not silently change
  scope.
- Commit in logical steps, with attribution lines per the repository's conventions.
  Push or open a PR only if asked.

## Acceptance criteria

- [ ] Unit tests for the observable rows pass (linearity, finite differences, stress convention).
- [ ] Scale diagnostics produced for both models.
- [ ] All 21 QoIs propagated with CR3 (both routes for elastic constants), B1, B2 and B3.
- [ ] Delta-method and committee results agree within 10% for linear and relaxed QoIs, or the differences are flagged and explained.
- [ ] Coverage, joint test and sharpness reported with binomial confidence intervals.
- [ ] `REPORT.md` and the figures generated by a single command: `python -m benchmarks.si_qoi_uq.run --all`.
