# Next steps after the MMM Jeju talk (2026-11-10)

Four questions from the talk audience, and a cheap first step that decides how far to take three of them.
The rules of the paper still apply: never tune on the test split or on DFT QoIs, fix every DFT reference before
looking at predictions, report negative results, and every number traces to `bench/si_qoi_uq/results/`.

Order: §0 (existing results) → §3 vacancy barrier → §1 (111) surfaces → §2 POPS ellipsoid → §4 multivariate
conformal.

## §0 Cheap checks (existing results, ~1 day)

- **Per-type force coverage on (111) surfaces.** The test split has 5 `surface_111_pandey`, 5 `surface_111`,
  1 `111adatom` and 1 `crack_111` configurations (training: 45, 42, 10 and 9, plus one `surface_111_3x3_das`).
  Per-atom coverage of each method on these types from the existing force results.
- **Configuration-level Mahalanobis scores.** r_c = ΔF_cᵀ (Σ_c + εI)⁻¹ ΔF_c over all force components of each test
  configuration c, for CR3, B1 and POPS (B2), each at its own scale. If a covariance has the right *shape*,
  r_c / d_c is near 1 with a χ²_d-like spread (d_c the number of components, or the rank used); a right total
  size with the wrong shape shows up as a spread much wider than χ². Also per-atom 3×3 scores, and the
  fraction of ‖ΔF‖² outside the covariance's range (CR3 rank 192 at o3d12, 2226 at o4d20). This is the first test of §4.
- **The vacancy migration barrier reference.** See §3: two DFT values disagree and have to be reconciled first.

## §1 Missing spin: can CR3 resolve the Si(111) 2×1 Pandey and DAS reconstructions?

CR3 measures misspecification *relative to the labels*: how far the fit moves when clusters with large,
aligned residuals are left out. Consequences:

- If the GAP-18 labels are not spin-polarised (the training file records no DFT settings: check the GAP-18
  paper), missing spin is reference-method error, invisible to every method that learns from those labels
  (CR3, B1, POPS, conformal). Talk answer: physics absent from the labels needs new reference data, not another UQ
  method.
- If spin changes the labels for some geometries (for example adatom or rest-atom dangling bonds in DAS), a model
  with no spin descriptor misses it systematically, and CR3 should be wider there. The §0 per-type check shows whether
  this is already the case.

Steps:

1. Add QoIs from the harness tests that already exist: `surface-energy-111-relaxed` (2×1 Pandey) and
   `surface-energy-111-2n+1_x_2n+1-relaxed` (3×3, 5×5, 7×7, 9×9 DAS). The delta rows come from the
   envelope theorem, as for the (100) surface in `delta.py`.
2. Fine-detail QoIs where correlated errors matter: the Pandey buckling-isomer energy difference, and the
   7×7 − 5×5 and 5×5 − 3×3 surface energy differences. From the joint covariance, report the
   probability that the order is right, for example P(γ₇ₓ₇ < γ₅ₓ₅). GAP-18 is known to get this order right.
3. Only if 1–2 show large misses that the covariance does not explain: spin-polarised CASTEP on 5–10
   Pandey and DAS configurations, to measure the size of the effect directly (needs DFT time).

**Results, steps 1–2 (2026-11-10; `analysis/s111.py` → `results/s111.json`).**
- Correction to step 1: `surface-energy-111-relaxed` is a perturbed and relaxed **1×1** cell, which cannot form the 2×1
  Pandey chain, so no harness test gives a Pandey energy. The CASTEP reference exists only for the 1×1 value (0.0980 eV/Å²).
  For DAS there is no CASTEP value. The framework's DFT_literature reference (Solares et al. 2005, shifted to the CASTEP 1×1)
  uses a different functional and protocol, so DAS is scored on order and differences.
- Harness: under ASE ≥ 3.23 the unmodified DAS test failed (`KeyError: 'move_mask'`: the extxyz reader turns the column into
  FixAtoms). `harness.py` now restores the array on read and drops it on write.
- Rows: the envelope theorem on the recorded relaxed slabs, plus the a0 term; e111 (unrelaxed) is rebuilt with ASE. Every
  Q from the rows matches the harness value to 1e-5.
- 1×1 (CR3 × λ_F, z against CASTEP): o3d12 0.0826 ± 0.0103 (z +1.49); o4d20 0.0945 ± 0.0043 (z +0.81) eV/Å².
- DAS: every absolute value is within 0.7 σ of the literature for both models. o4d20 is almost exact
  (0.0888 / 0.0862 / 0.0859 / 0.0861 against 0.0880 / 0.0863 / 0.0860 / 0.0868). The mean order is right for o4d20
  (7×7 lowest, as GAP-18 has it) but wrong for o3d12 (energies fall monotonically to 9×9).
- The order is not resolved by either covariance. Differences (meV/Å²; literature in brackets): o3d12 5−3 −6.8 ± 3.7 (−1.7),
  7−5 −1.5 ± 1.9 (−0.3), 9−7 −0.4 ± 1.2 (+0.8); o4d20 −2.7 ± 6.0, −0.2 ± 4.1, +0.2 ± 2.7. The sizes are correlated at
  0.97–1.00, but the difference σ is still larger than the differences. P(7×7 lowest) is 0.15 (o3d12) and 0.05 (o4d20),
  below the 0.25 of a random order: the intervals say "order unknown", and every literature difference is within 1.4 σ.
- Reading for the talk question: no miss that the covariance does not explain, so step 3 (spin-polarised DFT) is not
  triggered. The fine detail (the 7×7 minimum, ~0.3–0.8 meV/Å²) is below what either model's parameter uncertainty resolves.

## §2 Does the POPS ellipsoid fix held-out coverage on forces and relaxed defect energies?

`POPS-UQ/popsregression` branch `ellipse-ev` (107 commits ahead of main): `POPSRegression(posterior="ellipsoid")`,
a uniform ellipsoid fitted by minimising the width penalty of Swinburne et al. 2026, Eq. (8), subject to covering
every training point; `pac_bayes=True` adds the PAC-Bayes layer, which is out of scope here.

Expectation: probably not, at our N/P. The paper's own linear-ACE (Cu) result has the bare ellipsoid equal to the
hypercube at N/P = 20 (signed miscalibration area +0.074 for both); the ellipsoid only helps clearly at small N/P,
and then together with the PAC layer. Our held-out misses (o4d20 cracks 0.40–0.50) look like distribution shift,
not finite data. A better *shape* could still help, so measure it.

1. Read `_ellipse.py` and check that it scales to P = 5476 (hypercube initialisation, low-rank ellipsoid, cost
   O(N r)). Design rows at o4d20 are ~10–15 GB, so run it on mnf148.
2. Bare ellipsoid on the same rows as B2. Read it as a Gaussian, with covariance B/(P+2) for a uniform P-ball.
   Evaluate it as for B2: per-atom test coverage, held-out cracks and relaxed defect energies.
3. The same ellipsoid with one hold-out calibration scale (as B1 × q̂): separates "better shape" from "no
   calibration step". Measure the shape with §0's scores and with the stretch spectrum in B1's metric
   (`analysis/cov_shapes.py`).

Decision: if 3 covers and 2 does not, the lesson for POPS is "calibrate the scale". That supports our
construction rather than competing with it.

**Results (2026-11-10; `analysis/pops_ellipse.py` → `results/pops_ellipse_<m>*.json`).** o3d12 first, then o4d20 below.
popsregression `ellipse-ev` (6d61264), `POPSRegression(posterior="ellipsoid")`, package defaults (rank 32). Inputs: all 474,493
training E/F/V rows of the cr3 fit, weighted as the fit weights them, with targets the residuals about θ̂ (the package's centre
moves θ̂ by 1e-4 relative; test force RMSE 0.1584 → 0.1581). Read as a Gaussian, Σ = B/(P + 2), as the package's own sigma.
Correction to the plan: the objective is the projected-ball NLL with a covering barrier, B = B0 + UUᵀ in whitened coordinates,
not a width penalty subject to coverage.

| variant | per-atom 90 % coverage (bare / one per-atom scale) | QoIs 95 % of 17, median σ/tol (bare / scaled) |
|---|---|---|
| defaults (whiten ridge 1e-8) | 1.00 / 0.17 | 17, 1.5e10 / 17, 9.1 |
| whiten ridge 1e-6 / 1e-4 / 1e-2 × max eig G | 1.00 / 0.13, 0.05, 0.13 | 17, 5e5–4e12 / 13, 12, 14 at 2.1–4.7 |
| PAC-Bayes layer (defaults; τ² = 7.5e21) | 1.00 / 0.19 | 17, 1.7e10 / 16, 11.4 |
| the package's own hypercube baseline B0 | 0.94 | 13, 1.88 |
| paper B2 (ace-jax POPS box, leverage cut 50 %) / headline CR3 × λ_F | 0.947 / 0.791 | 13 / 14, 2.37 |

Reading (o3d12):
- The bare ellipsoid is not numerically defined on this design (cond G ~ 1e16). Its variance along the directions the
  data barely constrain is ~1e20 (58 % of its trace in the weakest 10 % of precision directions). Rows of relaxed
  structures project onto them, so QoI σ is 1e5–1e12 × tol. The objective's 0.5 log v_i term penalises width only where
  rows have weight, so U grows freely elsewhere.
- Projecting onto the data span (the package's own mode_threshold: rank 90 of 224) does not help: the weak modes inside it
  carry the same blow-up. No whitening ridge from 1e-8 to 1e-2 × max eig G defines it. The PAC-Bayes hyperprior scales with the
  warm start (τ² = 7.5e21), so it does not regularise either.
- With one per-atom scale (plan step 3) the shape is wrong for forces: 0.05–0.19 per-atom coverage at a nominal 0.90. Every
  force direction is tiny next to the blown-up ones that set the scale.
- The package's own hypercube baseline reproduces the paper's B2 (0.94 against 0.947; QoIs 13/17).
- **o4d20** (defaults only; 474,493 × 5,477 rows, 19.4 GiB, ~190 GB peak; sklearn's BayesianRidge SVD routed to NumPy's
  ILP64 LAPACK because SciPy's LP64 refuses 2.6e9 elements; fit 62 min, 66 iterations, every training row covered). It fails the
  same way. The bare QoI σ is 1.5e11 × tol (median); the data span at the package's threshold is rank 39 of 5,477
  (max eig G 2.7e18), with 75 % of the trace outside it. With one per-atom scale, per-atom coverage is 0.42 (span: 0.38) at a
  nominal 0.90. The QoIs then cover 14/17 (span 15/17) at median σ/tol 0.71 (0.65), against the headline's 12/17 at 0.53. But that
  scale makes the force regions wrong, so it is not a calibrated method. The package's own hypercube baseline covers 0.58 per atom
  (paper B2, the ace-jax box with its held-out ridge: 0.84) and 10/17 QoIs at σ/tol 0.23.
- Decision (plan), both models: neither 2 nor 3 covers. The lesson is not "calibrate the scale": at this conditioning the ellipsoid needs
  a prior or ridge on its widths in parameter space, as the hypercube needed a ridge. That is a POPS-package issue to report
  upstream (Swinburne), not a competitor to CR3.
- The smoothness prior as that width prior (o3d12, `--smooth-prior`; `results/pops_ellipse_o3d12_sp[_pac].json`): columns
  rescaled X Λ^-1/2, Λ the cr3 fit's own prior precision at the MAP (`linear_prior_diag`), whiten_ridge 1/N, so the package
  whitens with (XᵀX + Λ)/N, the BLR posterior precision (no free knob). It does not define the ellipsoid: 99.0 % of its trace
  stays outside the data span, its own scale is 4e-12, QoI σ ~1e11 × tol, and with one scale per-atom coverage is 0.04 (0.13–0.19
  without the prior). The prior only sets the whitening coordinates; the objective has no term in U, so nothing bounds the free
  directions. The PAC-Bayes layer changes nothing (the package: `pac_bayes=True` never changes `coef_`/`U_`; τ² 8e21, KL 7.8e4):
  the result is bit-identical. The prior does change the package's hypercube baseline: per-atom coverage 0.98 (0.94 without;
  paper B2 0.947), QoIs 13/17 at σ/tol 2.46 (headline 14/17 at 2.37). Conclusion unchanged and sharpened: the prior has to enter
  the ellipsoid objective itself (e.g. a −log det or tr(Λ B) term), a package change. o4d20 not run (same mechanism).
- The prior in the objective (o3d12, `--smooth-prior --param-prior`; popsregression local branch `ellipse-param-prior`,
  afcaf04 + ad51a00, not pushed): `param_prior_precision` adds KL(N(μ, B/(P+2)) ‖ N(0, Λ⁻¹)) = ½ (tr(ΛB)/(P+2) − log det B)
  at unit weight; rank P = 224. Toy check: an unconstrained coefficient goes to the prior variance (0.2499 for 1/4), constrained
  ones unchanged (package test added). On o3d12 it does not work: cold start and a warm start at the prior optimum reach the same
  point (`results/pops_ellipse_o3d12_sp_kl[_cold].json`), both at the iteration cap (4 × 500). The prior cuts the trace 300×
  (1.4e23 → 4.4e20) and the median in-span variance 400× (1.2e5 → 2.7e2), but the largest variance *inside the data span* stays
  5e19, so the blow-up is data-driven, not the free directions: the covering condition demands it. 20 rows stay outside
  (coverage 0.99996), the barrier then dominates (data objective 4.9e7 per row against 26 without the prior). Own scale 5e-12;
  one scale: per-atom coverage 0.06, QoIs 15/17 at σ/tol 6.7, tetrahedral z 6.5. Reading: every row's pointwise-optimal
  correction must lie in the ellipsoid, and low-leverage rows need corrections of ~1e10 in θ; the hypercube copes only through
  its leverage cut (B2: 50 %), which the ellipsoid lacks. o4d20 low-rank + diagonal not built (gate not met).

### §2b The leverage-cut ellipsoid

`--smooth-prior --leverage-pct 50` (`analysis/pops_ellipse.py`; `analysis/pops_ellipse_ood.py` for the held-out refit): prior
coordinates as above, and only the rows whose posterior leverage h_i = x̃_iᵀ(X̃ᵀX̃ + I)⁻¹x̃_i is at or above the median, B2's rule
(the reference package's default, fixed before any run). Leverages by a two-stage QR (R of X̃, then R of [R; I]): the normal
equations fail at this column scaling (Cholesky indefinite; an eigendecomposition gave Σh = 1274 > P). o3d12 Σh = 212.3.

- **o3d12, test split** (`results/pops_ellipse_o3d12_sp_lev50.json`): the bare ellipsoid is defined for the first time. Own scale
  λ_e = 1.03 (no cut: 1e-12; CR3 4.75); per-atom cov90 0.86 bare, 0.876 with λ_e (headline 0.79, B2 0.947, its own hypercube with
  the same cut 0.954). QoIs 14/17 at median σ/tol 1.76 (headline 14/17 at 2.37); the misses are dumbbell, tetrahedral and vacancy
  (z 2.05, 2.14, 2.42). Per type (≥ 300 atoms): 0.79 (sh) to 0.94, RMS deviation from 0.90 of 0.051 (0.041 with λ_e); headline
  0.70 (interstitial) to 0.98, 0.109; B2 0.87 to 0.97, 0.048 (B2 over-covers most types).
- **o3d12, held out** (`results/pops_ellipse_ood_o3d12.json`; the refit without cracks and screw dislocations, same recipe, λ_e
  on the refit's in-distribution test split): per-atom cov90 in / out of distribution — ellipsoid 0.856 / 0.758 (+λ_e = 1.04:
  0.876 / 0.781), headline 0.797 / 0.762, B1 0.721 / 0.578, B2 0.945 / 0.887. Per held-out type, ellipsoid+λ_e / headline / B2:
  crack_110 0.84 / 0.89 / 0.92, crack_111 0.74 / 0.82 / 0.86, screw 0.78 / 0.65 / 0.90. The ellipsoid stays self-calibrated but
  loses 0.10 off distribution (headline 0.035, B2 0.06); it is more even across the held-out types than the headline.
- Caveats: (1) 99.9 % of the trace lies outside the data span (G eigenvalues above 1e-8 × max); there U keeps its random start.
  Projected on the span, coverage falls 0.86 → 0.70 (0.84 with its λ = 1.21) in distribution and 0.78 → 0.73 held out: about
  15 points of the in-distribution coverage are not fitted. (2) The 4 × 500-iteration fits are not converged, but close: the o3d12
  rerun at 20,000 per stage (`_it20000`; 15,398 iterations; stages 2–4 report convergence) gives objective 1.171 (1.185),
  λ_e 1.039, cov90 0.859 / 0.879, per-type RMS deviation 0.049 / 0.038 (lowest 0.83, surface_001), QoIs 14/17 at σ/tol 1.64;
  span-projected coverage 0.64. (3) With the
  KL prior (`--param-prior`, rank 224; converged, 1573 iterations, KL 7.7e3) the free directions take the prior width, and the
  ace-jax prior is near-improper in some columns (Λ down to 1e-8): QoI σ ~6e5 × tol, per-atom coverage 1.0, so the principled
  version of (1) is not usable with this prior.
- **o4d20, test split** (`results/pops_ellipse_o4d20_sp_lev50.json`; rank 32, 2000 iterations, not converged; Σh = 3968 of
  5477): own scale λ_e = 1.08 (CR3 5.94). Per-atom cov90 0.827 bare, 0.873 with λ_e (headline 0.878, B2 0.837, its own
  hypercube with the cut 0.456). **Per type it removes B2's weakness:** ellipsoid+λ_e 0.74 (surface_001) to 0.96, RMS deviation
  0.060 (bare 0.69 to 0.93, 0.092); B2 0.51 (surface_001) to 0.99, 0.194, with all four surface types and liq at 0.51–0.75;
  headline 0.74 (vacancy) to 1.00, 0.098, over-covering surfaces, liq, sh and bt at 0.98–1.00. QoIs 15/17 at median σ/tol 0.28
  (headline 12/17 at 0.53): misses dE_sh (z 3.19 at σ/tol 0.08) and tetrahedral (z 2.32); dumbbell, hexagonal and surface_100,
  which the headline misses, are covered. The span diagnostic is not meaningful at o4d20 (G's largest eigenvalue 2.7e18, so the
  1e-8 relative cut keeps 39 of 5477 directions); how much of the coverage comes from unfitted widths is therefore not known here.
- **o4d20, held out** (`results/pops_ellipse_ood_o4d20_it5000.json`; the refit without cracks and screw dislocations, 5000 per
  stage as `_it5000`, dedicated-f02, 13,860 s; 12,190 iterations, objective 1.2501; λ_e = 1.11 on the refit's test split): per-atom
  cov90 in / out of distribution — ellipsoid 0.816 / 0.663, +λ_e 0.877 / 0.739; headline 0.879 / 0.906, B1 0.600 / 0.459, B2
  0.836 / 0.588. Per held-out type, ellipsoid+λ_e / headline / B2: crack_110 0.76 / 0.95 / 0.50, crack_111 0.67 / 0.89 / 0.40, screw
  0.78 / 0.89 / 0.77. The ellipsoid loses 0.14 off distribution (o3d12: 0.10); the headline does not lose any; B2 loses 0.25.
  Reading: the ellipsoid's per-type evenness is an in-distribution property. Off distribution it sits between B2 and the
  headline at both orders, so it does not change the §2 out-of-distribution argument for CR3.

### §2c CR3 against the POPS ellipsoid: what each has in principle (for the paper's discussion)

CR3 is the cluster-robust jackknife (Bell–McCaffrey CR3) sandwich A⁻¹MA⁻¹ of the BLR fit, A = XᵀX + Λ, clusters = configurations,
times one split-conformal scale λ_F. The POPS ellipsoid is a uniform set B fitted to cover every training row's pointwise-optimal
correction (projected-ball NLL with a covering barrier), read as N(μ, B/(P+2)).

Advantages of CR3, in principle:
1. **A coverage guarantee.** The scale is split-conformal: under exchangeability of test and calibration configurations,
   marginal coverage holds in finite samples whatever the misspecification. The ellipsoid's covering condition is a statement
   about training rows; its held-out coverage (0.86 here) is an empirical outcome with no guarantee.
2. **No data selection.** CR3 uses every row. A covering criterion behaves like a max over rows, so one low-leverage row that
   needs a correction of ~1e10 in θ stretches the whole set; the ellipsoid is defined only after discarding half the rows, and
   its result depends on that cut (scale 1e-12 without it, 1.03 with it). A configuration enters CR3 through its share of a sum.
3. **Configuration-level errors.** The CR3 meat is built from per-configuration score sums, so errors correlated within a
   configuration (all atoms share one misspecification) are counted. POPS gives each E, F, V row its own correction and covers
   it alone, so within-configuration correlation never enters the shape. §4 found whole-configuration errors at 1.7–2.8 × the
   per-row variance in the leading directions.
4. **Defined in every direction.** Where the data are silent CR3 falls back to the prior through A, and where the data speak the
   data dominate it. The ellipsoid's width outside the data span is its random start (§2b caveat 1); a KL prior makes it the
   prior width, which with a near-improper prior is unusable.
5. **Propagation semantics.** CR3 is a covariance, so the delta method gives a Gaussian σ for any linear QoI, consistent with how
   it was calibrated. The ellipsoid is a support set: its Gaussian reading uses a uniform-ball moment, and its natural QoI output
   would be a bound (the projection of the support), a different meaning of coverage.
6. **Computation and scale.** CR3 is closed form (one QR plus score sums, O(NP²)). The ellipsoid is a non-convex L-BFGS fit that
   did not converge in our runs, is rank-limited (32 of 224), and cannot be full rank at P = 5477.

Advantages of the ellipsoid:
- **Shape from where the model misfits.** Its shape follows the pointwise-optimal corrections directly, so coverage is more even
  across configuration types (o3d12: RMS deviation 0.05 against 0.11 for the headline; held out, 0.74–0.84 against 0.65–0.89).
  CR3 gets its shape from residual-weighted scores and then one global scale, which leaves per-type unevenness (paper §3.x).
- **Self-calibration.** With the leverage cut its scale is ≈ 1 with no test split (o3d12: 1.03; held-out refit: 1.04), whereas
  CR3 needs λ_F ≈ 4.8 from a calibration set.

Possible paper use: items 1–3 as the reason the conformal scale is kept as the headline; the ellipsoid's per-type evenness as
the motivation for a per-type or Mondrian scale on CR3 (already in the paper), and as an open question for POPS (a leverage
cut or a soft covering condition; prior only where the prior is proper). At o4d20, where the paper's B2 is uneven per type, the
leverage-cut ellipsoid is the most even method (RMS deviation 0.060 against 0.098 headline and 0.194 B2) and covers 15/17 QoIs
at σ/tol 0.28, but is not converged and its unfitted-width share is unknown: needs a converged rerun before any paper claim. Rerun at 5000 iterations per stage (`_sp_lev50_it5000`, mnf148, 15,542 s; stage 1 still at its cap, stages 2–4 converged): objective 1.2524 (was 1.2810); cov90 0.816 bare, 0.875 with λ_e (headline 0.878, B2 0.837); per-type RMS deviation 0.100 bare, 0.056 with λ_e, lowest 0.76 surface_001 (headline 0.098, B2 0.194); QoIs 15/17 at σ/tol 0.30 (headline 12/17 at 0.53). The evenness holds with more iterations.

### §2d A closed-form POPS ellipsoid (future; suggested by T. Swinburne, 2026-11-11)

Swinburne suggests the nonlinear fit can be avoided with PRESS-style updates, as in CR3. One form: the POPS pointwise-optimal
correction of row i is the smallest shift fitting it exactly, δθᵢ = A⁻¹xᵢ rᵢ / hᵢ with hᵢ = xᵢᵀA⁻¹xᵢ; the CR3/PRESS
leave-one-out shift is A⁻¹xᵢ rᵢ / (1 − hᵢ). Their second moment Σ δθᵢδθᵢᵀ = A⁻¹ [Σ xᵢxᵢᵀ rᵢ²/hᵢ²] A⁻¹ is the CR3 sandwich with
weight 1/hᵢ² instead of 1/(1 − hᵢ)²: closed form, O(NP²), one scale left (a 1-d quantile of dᵢ² = δθᵢᵀS⁻¹δθᵢ for covering,
or split-conformal). Per configuration: δθ_c = A⁻¹X_cᵀ(X_c A⁻¹X_cᵀ)⁻¹ r_c (CR3: (I − H_cc)⁻¹). Open: 1/h² lets low-leverage
rows dominate (§2b's covering problem moved from a max into a sum, so the leverage cut is probably still needed); whether it
keeps the fitted ellipsoid's per-type evenness. Cheap to test with the CR3 code (score sums with the new weights).
Not in the popsregression prior PR.

Warm start tried first (2026-11-11, negative): popsregression `init_widths="moments"` (U from the per-point optimum v_i = P r_i², a rank-32 sandwich by block power steps), o3d12 `_sp_lev50_it20000_mi` on lestrade: stage 1 still hits the evaluation limit (14,534 iterations, as 14,553 from the random start) and stage 2 then takes 9,464 (771 before); 24,015 iterations, 3443 s against 15,398 and 1973 s. Objective 1.1689 vs 1.1707 (another local optimum), coverage and QoIs as before (cov90 0.860 / 0.882 with λ_e, per-type RMS 0.049 / 0.036, QoIs 14/17). The cost is not the start: stage 1's conditioning. Popsregression PR #15 (the KL prior) opened against `ellipse-ev`; the warm start is not offered.

Closed form, first prototype (2026-11-11; `pops_ellipse.py --closed-form q`, o3d12 `--smooth-prior`, lestrade; results `pops_ellipse_o3d12_sp[_lev50]_cf<q>.json`). Built in u_i = R2^-T x_i (||u_i||^2 = h_i): S_u = mean g_j² u_j u_jᵀ, g = r/h, covering ratio r_i²/(u_iᵀS_u u_i); forming A⁻¹MA⁻¹ in β directly cancelled (v_i < 0). 12 s against 1973 s for the converged L-BFGS fit.
- Without the leverage cut: unusable, as predicted. c = 2.3e5 from rows with h ~ 1e-26 (the 99th percentile of the ratio is 2e-8); QoI σ/tol 1.9e7.
- With the 50 % cut, covering every row: c = 6117 (ratio median 56, 90th 370, 99th 1063: the max over rows is 100 × the median). Not self-calibrated: λ_e 2.20 (fitted ellipsoid 1.04), bare cov90 0.28. With λ_e: cov90 0.873 (fitted 0.879, headline 0.791), per-type RMS deviation 0.091, lowest 0.74 surface_001 (fitted 0.036, headline 0.109, B2 0.048); QoIs 13/17 at σ/tol 1.53 (fitted 14/17 at 1.69, headline 14/17 at 2.37), missing the dumbbell, tetrahedral and vacancy energies at z 4.3–4.7 (headline 3.4, 2.1, 1.6).
- q = 99 or 90 changes only c (the shape is the same), so with λ_e the results are identical.
Reading: the moment shape costs one QR and is about as good as CR3 overall, a little more even per type, worse on defect energies; it keeps neither the fitted ellipsoid's evenness (0.036) nor its self-calibration. Those come from the covering fit itself, which a moment cannot reproduce. Untried: the per-configuration form; a few reweighting steps from the fitted objective's stationarity condition (each costs one weighted sandwich, O(NP²)), which would move the moment towards the fitted optimum.

Reweighting iteration (2026-11-11; `--fixed-point K`, `fixed_point()` in pops_ellipse.py). Stationarity of the package's objective in B (full rank, u coordinates, no baseline) is B N B = C, N = mean ½uuᵀ/v, C = B[mean k L'_ρ(q) r²/v² uuᵀ]B (k × the weighted second moment of the corrections B u_i r_i/v_i); with N, C held, B = N^-½(N^½CN^½)^½N^-½. Step halved along the geodesic if the objective rises; continuation over ρ as the package. Toy (N 4000, P 20): my objective equals the package's at its optimum (0.6574643) and that optimum is a fixed point; the optimum is rank-deficient (8 of 20), and the multiplicative update cannot regrow a collapsed direction (stuck at 0.6678, rank 6) until a 1e-4 × tr(B)/P isotropic jitter is added to each proposal (then widths match the package's to a median ratio 1.0006, but 3000 steps reach only 0.6630 against 138 L-BFGS iterations).
o3d12 `_sp_lev50_cf100_fp2000` (from the closed form, lestrade): 550 steps at ρ = 0.1, then converged at every later ρ; 268 s including the scoring (the L-BFGS fit: 15,398 iterations, 1973 s). Objective 1.15912, below L-BFGS's 1.1707 (rank 32 plus the POPS baseline; same objective). Self-calibrated: λ_e 1.07. cov90 0.852 bare, 0.887 with λ_e (fitted 0.859 / 0.879); per-type RMS 0.054 / 0.036, lowest 0.85 interstitial (fitted 0.049 / 0.036). QoIs 13/17 at median σ/tol 1.48 bare (fitted 14/17 at 1.64); the dumbbell, tetrahedral and vacancy energies at z 2.6–3.2 (fitted closer; headline 3.4, 2.1, 1.6). (Numbers rescored after the eigh fix below; an earlier "QoI σ unstable, 1.54 against 2.20 for objective changes of ~1e-9" was that bug, not the objective.) Reading: the reweighting replaces L-BFGS at o3d12 (7× faster, lower objective, same force coverage and evenness).

o4d20 `_sp_lev50_cf100_fp1000` (mnf148, 20,119 s fit, ~51 s per full-rank step): objective 1.2089 against L-BFGS it5000's 1.2524 (which hit its cap after 4.3 h); it passed 1.2524 after 21 steps. Downstream it is not better: λ_e 1.24 (L-BFGS 1.11), cov90 0.775 bare / 0.893 with λ_e (0.816 / 0.875), per-type RMS 0.074 with λ_e, lowest 0.73 dia (L-BFGS 0.056, lowest 0.76 surface_001); QoIs 14/17 at median σ/tol 0.34 with λ_e (15/17 at 0.33), gm CRPS/t 0.268 (0.260), joint d²/n 3.46 (1.79). So at o4d20 the lower optimum is a little less even on forces and worse jointly calibrated on QoIs; the objective value does not rank the ellipsoids downstream.

**Scoring bug (found and fixed 2026-11-11, commit 6b0ee45).** `factor_of` took the force and QoI covariances from an eigh factor of Σ_e. The reweighting and closed-form fits leave unseen directions with eigenvalues to ~4e8 (o3d12) and ~7e5 (o4d20) against ~1e2 and ~2e-2 for L-BFGS, and eigh's error (ε × the largest eigenvalue) inflated QoI σ against an extended-precision gᵀΣg by up to 6× (o3d12) and 20× (o4d20; median 1.19); diagonal scaling before eigh still left 11 %. Scoring now uses FΣFᵀ and gᵀΣg directly (CR3 keeps its factor). Every closed-form and fixed-point result was rescored from its saved fit (`rescore_o3d12.sh`, `rescore_o4d20.sh`): forces move by < 0.006 (λ_e, cov90, per-type RMS), QoI median σ/tol moves a lot (o3d12 closed-form start 2.20 → 1.48 bare, o4d20 1.09 → 0.34 with λ_e), and the L-BFGS results reproduce exactly. The joint scores used gᵀΣg already and were unaffected except through λ_e.

Rank-limited fixed point (2026-11-11; `fixed_point_lowrank()`, `--fp-rank R [--fp-inner lbfgs]`). B = V C Vᵀ at the package's
rank: fit C in the subspace (reweighting, or L-BFGS on the m × m factor, O(N m²) per step), truncate to the top R
eigen-directions, expand by the R most negative eigenvectors of the gradient G = mean a_i u_i u_iᵀ on V's complement (block
LOBPCG, G never formed, O(N P R) per block), seeded at 1e-2 × the mean width; stop when the objective stops falling or G has
no negative direction off V. A step costs a few L-BFGS evaluations, against P/R ≈ 170 for a full-rank step at o4d20 (break-even
there ≈ 40–60 full-rank steps against L-BFGS's 10,282 iterations; o3d12 needed 550). Toy, rank 5: started at the package's
optimum it stays there (0.657839); from the moment start it stops at 0.6648 (L-BFGS inner) or 0.6679 (reweighting inner), from
a random start at 0.6682, each a stationary point with no descent direction off V.

### §2e The objective is multimodal; what that means for the paper (2026-11-11)

**Finding.** The POPS ellipsoid objective is non-convex in B: each term ½ log v − k log(1 − r²/v) is concave in v once v ≫ r²
(its second derivative changes sign), and the rank limit adds more. Solvers and starts land on different stationary points:
toy (rank 5) 0.6578 (package L-BFGS) / 0.6648 / 0.6682; o3d12 1.1707 (L-BFGS, random start) / 1.1689 (L-BFGS, moment start) /
1.1591 (full-rank reweighting); o4d20 1.2524 (L-BFGS, capped) / 1.2089 (reweighting). Each is a genuine stationary point
(checked on the toy: in-subspace optimum, no descent direction outside it).

**Optimise or marginalise?** exp(−N × objective) is the projected-ball likelihood of B. The o3d12 gap 1.1707 − 1.1591 is
0.0116 × 237k rows ≈ 2,750 nats (toy: 28 nats); even with N cut to the ~2,000 training configurations (rows within one are
correlated) it is ~20 nats. So in the directions the data determine, the lowest optimum carries essentially all the posterior
mass: there the multimodality is a search problem (multi-start, continuation), not a reason to average. In the directions the
data barely see, the fitted widths are set by the start and the solver: the reweighting fits put eigenvalues to ~1e8 there
(L-BFGS ~1e2) and the span-projected coverage is 0.50. The QoIs mostly avoid those directions (after the scoring fix their
widths agree across converged solvers to ~10 %, below). Marginalising those directions would return the prior, so it needs a
proper prior; with the smoothness prior the KL version gave near-improper widths (§2b). The package's PAC-Bayes layer is a Laplace approximation at one optimum and addresses neither.

**o3d12 multi-start** (2026-11-11; lestrade batch `runs/si_qoi_uq_blr/o3d12_fp_batch.sh`; results
`pops_ellipse_o3d12_sp_lev50_cf100_fp2000{,_r32,_r32_lb,_s1..s5}.json`). Full-rank reweighting from five random Wishart starts
(`--fp-start k`, 2,000 steps per ρ) against the closed-form start, and the rank-32 form (rescored after the eigh fix; median
σ/tol with λ_e):

| start | objective | steps at ρ = 0.1 | λ_e | cov90 +λ_e | per-type RMS +λ_e | QoIs 95 % | median σ/tol +λ_e |
|---|---|---|---|---|---|---|---|
| closed form | 1.15912 | 550 (converged) | 1.065 | 0.887 | 0.036 | 13/17 | 1.58 |
| closed form, rank 32 | 1.16271 | 25 outer | 1.060 | 0.886 | 0.037 | 13/17 | 1.63 |
| random s1 | 1.17201 | 2000 (cap) | 1.064 | 0.890 | 0.040 | 13/17 | 2.13 |
| random s2 | 1.17138 | 2000 (cap) | 1.067 | 0.891 | 0.040 | 13/17 | 1.84 |
| random s3 | 1.17172 | 2000 (cap) | 1.067 | 0.889 | 0.040 | 13/17 | 2.06 |
| random s4 | 1.17104 | 2000 (cap) | 1.069 | 0.891 | 0.037 | 13/17 | 1.75 |
| random s5 | 1.16854 | 2000 (cap) | 1.068 | 0.889 | 0.039 | 13/17 | 2.15 |
| random s1, 20,000 steps | 1.16319 | 8,044 (+ 4,179 at ρ = 1e-4) | 1.062 | 0.889 | 0.037 | 13/17 | 1.99 |
| L-BFGS (`_it20000`) | 1.17069 | 15,398 it | 1.039 | 0.879 | 0.038 | 14/17 | 1.70 |
| package `solver='reweight'` (`_it20000_rw`), moment start | 1.16341 | 2,507 steps, 14,204 s | 1.035 | 0.881 | 0.036 | 14/17 | 1.64 |

Reading: (1) the random starts did not converge: each was still falling ~2e-6 per step at the cap, 0.009–0.013 above the
closed-form start (≈ 2,100–3,000 nats × N). Run on to 20,000 steps per ρ (`_fp20000_s1`, lestrade, 6,432 s), start 1 went
1.1720 → 1.1654 (ρ = 0.1, 8,044 steps, until no step descended) → 1.1632 (ρ = 1e-4, 4,179 more, stopped the same way while still
falling ~6e-7 per step), 0.0041 above the closed-form start (≈ 960 nats), and its QoI median σ/tol came from 2.13 to 1.99 (closed
form 1.58), forces unchanged (λ_e 1.062, cov90 0.889, per-type RMS 0.037). So most of the start spread in the table is
non-convergence, and the random start drifts towards the closed-form optimum; it stops early because the multiplicative update
has spurious fixed points on the PSD boundary (it cannot regrow a collapsed direction; fixed in the package version by a
projected-gradient growth step, popsregression branch `ellipse-reweight`). The closed-form start is what makes the reweighting fast. (2) The force predictions do not depend on the start: λ_e 1.04–1.07, cov90
0.879–0.891, per-type RMS 0.036–0.040 over every run. (3) After the scoring fix the QoI widths of the converged fits agree:
median σ/tol 1.58 (closed form), 1.63 (rank 32), 1.70 (L-BFGS), all at 13–14/17. The unconverged random starts are wider
(1.75–2.15) and narrow as they converge (start 1: 2.13 → 1.99 at 20,000 steps). Before the fix this column read 1.63–3.34 and
looked like solver noise; it was mostly the eigh error. This settles implication 5 below for forces, and for the QoIs it turns
"not reproducible" into "reproducible once converged, to ~10 %". (4) Rank 32 with the reweighting
inner solver reaches 1.1627 in 25 outer steps; with the L-BFGS inner solver (`_r32_lb`) it stopped after one outer step at
1.3185 (per-type RMS 0.176): a stopping-rule fault in `inner_lbfgs`, not a result. (5) o4d20 full-rank reweighting is done
(§2d): lower objective than L-BFGS, same per-QoI widths, slightly less even forces, worse joint QoI calibration. The o4d20
held-out refit is done (§2b).

**Implications for the paper** (to decide after the checks; nothing here goes in the revision yet):
1. *Headline unchanged.* CR3 × split-conformal stays the method. The ellipsoid comparison now gives concrete reasons beyond
   §2c's in-principle list: the ellipsoid needs a data selection (the leverage cut) to exist at all; it is a non-convex fit whose
   optimum depends on solver and start (o3d12 1.159–1.171, o4d20 1.209–1.252), and the lower optimum is not better downstream
   (o4d20: less even forces, joint d²/n 3.46 against 1.79); and its unconverged fits give wider QoI bands. CR3 is a unique
   closed form whose scale has a finite-sample guarantee.
2. *The positive result worth reporting.* With the cut, the ellipsoid's per-type force coverage is the most even of any method
   and needs almost no scale: o3d12 RMS deviation 0.036 with λ_e (headline 0.109, B2 0.048), λ_e 1.04–1.08; o4d20 converged
   0.056 (headline 0.098, B2 0.194). Held out (o3d12): 0.876 in / 0.781 out with λ_e (headline 0.797 / 0.762, B2 0.945 /
   0.887). This supports the paper's per-type (Mondrian) scale on CR3 as the practical route to evenness, with POPS showing that
   a shape learned from where the model misfits gets most of it unscaled. Candidate: one paragraph in the discussion plus an SI
   table (forces only).
3. *A QoI claim is possible, but only a modest one.* The "40 % change between runs" that ruled it out was the eigh scoring
   bug (§2d). Once converged, the ellipsoid's QoI widths agree across solvers to ~10 % at o3d12 (1.58–1.70) and between the
   two o4d20 optima (0.33 / 0.34). Per QoI the ellipsoid is a little sharper than the headline at similar calibration (o4d20
   gm CRPS/t 0.260–0.268 against 0.296; o3d12 0.88–0.92 against 1.01). Jointly it is worse calibrated (o4d20 d²/n 1.79–3.46
   against CR3's 1.30; o3d12 3.7–4.8 against 2.8), so the per-QoI advantage comes with correlations that put the errors in
   directions it calls thin. Candidate SI sentence: diagonal scores favour the ellipsoid slightly, the joint score favours CR3.
4. *A theory sentence.* The POPS pointwise correction A⁻¹xᵢrᵢ/hᵢ and the jackknife shift A⁻¹xᵢrᵢ/(1 − hᵢ) differ only in the
   leverage factor, so the moment of the POPS corrections is the CR3 sandwich reweighted by ((1 − h)/h)². That explains why POPS
   needs a leverage cut (low-h rows dominate) and why its closed-form version is not self-calibrated (§2d prototype: λ_e 2.20),
   and places the two methods in one family for the discussion.
5. *Dependency, settled.* The o3d12 multi-start shows force coverage and evenness are stable across optima (per-type RMS
   0.036–0.040), so items 2 and 4 stand. One qualification from o4d20: its lower reweighting optimum is a little less even
   (0.074 against L-BFGS's 0.056), so item 2 should quote the L-BFGS fits and say evenness varies by ~0.02 between optima.
6. *Method note for any reported ellipsoid width.* Compute σ from gᵀΣg, not from an eigendecomposition of Σ_e: the fitted
   widths in unseen directions reach ~1e8 and swamp an eigh factor (§2d).

**Soft QoI score** (2026-11-11; `analysis/qoi_soft_score.py`, `results/qoi_soft_score.json`). The 95 % count jumps when one borderline QoI crosses, so the scores are geometric means over the 17 QoIs of x / tolerance clipped to [0.05, 10], as the ace-jax accuracy benchmark (bench/si_qoi/score.py, same tolerances): gm |e|/t (the benchmark score), gm σ/t (sharpness), gm |z| (calibration: 0.530 for z ~ N(0,1), 95 % band 0.31–0.90 for 17 independent QoIs) and gm CRPS/t (proper; → |e|/t as σ → 0, so on the benchmark's scale). With λ_e:

| model, run | method | k95 | gm σ/t | gm abs z | gm CRPS/t |
|---|---|---|---|---|---|
| o3d12 (gm err/t 0.924) | headline | 14/17 | 2.03 | 0.48 | 1.009 |
| | fitted ellipsoid (L-BFGS, it20000) | 14/17 | 1.83 | 0.54 | 0.922 |
| | closed form (cf100) | 13/17 | 1.30 | 0.76 | 0.908 |
| | reweighting (fp2000) | 13/17 | 1.69 | 0.58 | 0.884 |
| o4d20 (gm err/t 0.366) | headline | 12/17 | 0.52 | 0.69 | 0.296 |
| | fitted ellipsoid (it5000) | 15/17 | 0.46 | 0.78 | 0.260 |
| | reweighting (fp1000) | 14/17 | 0.42 | 0.86 | 0.268 |

(Rescored after the eigh fix; the o3d12 reweighting row read 2.21 / 0.45 / 1.062 before it.) Reading: every method with its scale is inside the calibration band (bare closed form is not: gm |z| 1.58). The hard counts exaggerate the differences: 12 against 15 of 17 at o4d20 becomes a 12 % better CRPS for the ellipsoid (0.260 against 0.296), with a 12 % narrower typical σ and calibration inside the band for both. At o3d12 the three ellipsoid solvers span gm CRPS 0.88–0.92, all a little below the headline's 1.01; at o4d20 the two optima give 0.260 and 0.268. CRPS is dominated by the accuracy term (gm err/t) on both models; the UQ part moves it by ~10 %. Paper candidate: report gm CRPS/t and gm |z| beside the hard count (the latter with its band), using the benchmark's tolerances and clipping. Not yet computed: B1, B2 (their per-QoI σ are not in the ellipsoid result files), and the multi-start spread of these scores.

**Joint (correlation-aware) QoI scores** (2026-11-11; `analysis/qoi_joint_score.py`, `results/qoi_joint_score.json`). Same 17 QoIs, tolerance units, every method: CR3 (λ_F), B1, B2 (lev50, lev5), B3 from `joint_tests.json` (committee and delta covariances), the ellipsoid as λ_e² G Σ_e Gᵀ. Σ_Q is nearly singular (shared parameters; the paper's joint test moves by 4× with its ridge), so every covariance gets a floor τ²I, τ = 0.05 tol (where score.py clips errors), with τ = 0.01 and 0.2 as checks (d²/n moves < 15 % except B3 and o4d20 CR3, 0.75–1.23). "+disc": the paper's discrepancy term as a covariance, one shared bias per defect class (the three interstitial energies fully correlated, the vacancy alone). Scores: joint sharpness det(Σ)^(1/2n); gm |w| of the whitened errors w = Σ^-½ d (band 0.31–0.90); d²/n = |w|²/n (χ²₁₇/17 band 0.44–1.78); the Gaussian log score LS/n (proper); the energy score ES/√n (multivariate CRPS). Delta forms, τ = 0.05:

| model | method | k95 | gm CRPS | d²/n | gm abs w | LS/n | d²/n +disc | LS/n +disc |
|---|---|---|---|---|---|---|---|---|
| o3d12 | CR3 | 14 | 1.009 | 2.82 | 0.76 | 2.48 | 1.01 | 1.71 |
| | B1 | 13 | 0.944 | 3.69 | 0.67 | 2.99 | 1.05 | 1.82 |
| | B2lev50 (committee) | 13 | 1.089 | 1.51 | 0.53 | 2.20 | 0.74 | 1.92 |
| | ellipsoid L-BFGS | 14 | 0.923 | 3.66 | 1.00 | 2.82 | 1.32 | 1.77 |
| | ellipsoid reweighting | 13 | 0.884 | 4.84 | 0.95 | 3.37 | 1.27 | 1.70 |
| o4d20 | CR3 | 13 | 0.296 | 1.30 | 0.46 | 0.73 | 0.86 | 0.60 |
| | B1 | 11 | 0.276 | 6.36 | 0.42 | 2.84 | 2.38 | 1.02 |
| | B2lev50 (committee) | 14 | 0.303 | 2.45 | 0.63 | 1.29 | 0.89 | 0.63 |
| | ellipsoid L-BFGS | 15 | 0.260 | 1.79 | 1.01 | 0.74 | 1.60 | 0.70 |
| | ellipsoid reweighting | 14 | 0.267 | 3.46 | 0.56 | 1.52 | 3.18 | 1.45 |

(B3, the unscaled CR3, fails everything: d²/n 25–87; a sanity check.) Readings:
1. *The joint view reverses the o4d20 diagonal ranking.* Per QoI the ellipsoid looks best (15/17, CRPS 0.260 against 0.296); jointly CR3 is calibrated (d²/n 1.30, gm |w| 0.46) and the ellipsoid is at the edge (1.79, gm |w| 1.01 above the band), with a worse log score (0.74 against 0.73; with the discrepancy 0.70 against 0.60). The ellipsoid's narrower widths come with correlations that put the errors in directions it calls thin.
2. *The discrepancy term is what makes the joint test pass, and it must be correlated.* At o3d12 no parameter-only method except B2 is jointly calibrated (CR3 2.8, B1 3.7, ellipsoids 3.7–4.8); with the shared interstitial bias CR3 is (1.01), and B1 nearly (1.05). This is an argument for keeping the term and for stating it as a shared bias, not three independent variances.
3. *Proper scores.* With the discrepancy, LS/n ranks CR3 best or tied on both models (o3d12 1.71, tied with the reweighted ellipsoid's 1.70; o4d20 0.60, then B2lev50 0.63, ellipsoid L-BFGS 0.70, B1 1.02, ellipsoid reweighting 1.45). The energy score barely separates methods (o3d12 4.22–4.30, o4d20 1.26–1.29 with the discrepancy): like CRPS it is dominated by the accuracy term, so it is not the discriminating score here.
Paper candidate: report the joint d²/n (with its band) and LS/n beside the hard count and the diagonal CRPS; it is the score that sees the shared-mode errors of §4, it favours the headline, and it argues for the correlated discrepancy term.

Upstream: popsregression PR #15 (KL parameter prior) open against `ellipse-ev`. Reweighting-solver PR: held. The package
version (`solver='reweight'`, branch `ellipse-reweight`: geometric-mean update, geodesic over-relaxation, projected-gradient
growth step off the PSD boundary) is slower than L-BFGS on small problems (1,000–17,000 steps against 18–135 iterations, and
worse optima with the 'zero' baseline), and its o3d12 run (dedicated-g02, `_sp_lev50_it20000_rw`) reached 1.1634 (L-BFGS 1.1707, paper-repo fixed point
1.1591) in 2,507 steps and 14,204 s: 7x L-BFGS's 1973 s and 53x the paper-repo fixed point's 268 s, with the same forces and
QoIs (table above). Both start from a POPS-correction moment, but the package version fits the package objective (with
its POPS baseline, at rank 197) and needed 2,507 steps where the paper-repo version (no baseline) needed 550; why the
package form converges so much more slowly is not yet known. The paper-repo fixed point is fast at both models, but at o4d20 its lower optimum
is not better downstream. If a PR goes, it should offer the solver for large N with P above the data rank, find a lower
objective faster, and say the lower objective is not a better ellipsoid.


## §3 Does CR3 fix the under-coverage of the vacancy migration barrier?

The under-coverage is from I. R. Best, T. J. Sullivan and J. R. Kermode, J. Chem. Phys. 161 (2024), shown on the
talk slide "Limitations of conformal prediction: vacancy migration": a p_max = 20 ACE trained on bulk, vacancies
and divacancies, posterior ("Bayesian shape plus rescale") with a conformal scale, ASE NEB with 11 images. Read
off the slide: DFT barrier ≈ 0.30 eV (3.63 → ≈ 3.93 eV), GAP-18 ≈ 0.21 eV, ACE ≈ 0.12 eV with a band that does
not reach DFT at the saddle.

**Reference (settled 2026-11-10).** The GAP-18 paper's run of the framework's `vacancy-path-ref` test with CASTEP
(`/storage/eng/essswb/silicon-gap/test-results/model-CASTEP_socket-test-vacancy-path-ref-properties.json`):
E_vac 3.627 eV, saddle 3.925 eV, **E_b^DFT = 0.299 eV**. It is the curve on Best et al.'s Fig. 8 (0.30 eV, 3.63 eV),
and the framework's GAP-18 run of the same test (`test-runner/model-GAP18_py3-test-vacancy-path-ref`) gives 0.218 eV,
Best's 0.22. Protocol: 2×2×2 diamond cell minus one atom (63 atoms) at the model's own a0; end points relaxed
(fmax 1e-3); 9 interior images + 2 end points, linear interpolation, 0.05 Å rattle; plain NEB (k = 0.1, MDMin
dt 0.05; fmax 0.05 in the 2018 quippy run, 0.01 in the py3 port); CASTEP PW91, 250 eV, MP spacing 0.03 Å⁻¹,
non-spin-polarised (also the training labels' settings).

The 0.653 eV path (`test-results/model-CASTEP_ASE-test-vacancy-path-properties.json`, from the 2018 import, from a
`vacancy-path` test that is not in the repository) is not the reference. Its end points agree with the reference
to 0.03 eV (3.598 vs 3.627), so no shift by the formation energy explains it; its interior is higher. Likely an
unconverged or unrelaxed path: the framework's `no-relax-neb` results are above the `ref` ones for every model
that has both (DFTB 0.35 vs 0.04, MEAM 0.31 vs 0.16, TersoffScr 1.00 vs 0.40, SW 0.55 vs 0.52). Not used.

**The saddle is the middle image.** The hop is symmetric, and both reference paths peak at image 5 of 0–10. With an
odd number of images the middle image lies on the symmetry plane, where the force along the path vanishes by
symmetry, so the converged middle image is the saddle, a stationary point, with no climbing image needed (check
that the model's path is symmetric and peaks there). This keeps the reference protocol exactly.

**No differentiation through the NEB.** A converged climbing image is a stationary point (∇ₓE = 0), as a minimum
is, so the envelope theorem in `delta.py` carries over unchanged:

    dΔE‡/dθ = Φ(x_saddle) − Φ(x_min) + (dΔE‡/da₀)(da₀/dθ)

with Φ the summed site descriptor. The harness's `vacancy-path-*` tests are plain NEBs with MDMin (no climbing
image), whose barrier is a max over images: a non-smooth step, committee only (as γ_us). So:

1. The `vacancy-path-ref` protocol with the mean model (plain NEB, fmax 0.01 or tighter), at o3d12 and o4d20; check
   the path is symmetric and peaks at the middle image (else climbing image on that image).
2. Delta rows as above; check them against a λ = 1 committee of NEB reruns (the 10 % agreement rule).
3. Coverage and |z| for CR3 × λ_F, B1 and POPS against the fixed DFT barrier.

Cost: a few days of compute. The cleanest extension of the paper, and possibly a short note on its own.

**Mean-model runs (2026-11-10, mnf148).** `python -m bench.si_qoi_uq.harness run $SIQ_RUNS/samples/<m>_{cr3,pops}.npz
--mean --tests vacancy-path-ref` (the framework's test unmodified, py3 port: MDMin to fmax 0.01; converged, final
fmax 0.0068 / 0.0100 eV/Å). Harness venv `/storage/eng/essswb/venvs/si-tests` had gone; rebuilt from the repo's
`uv.lock` (ace-jax 0.5.3, ASE 3.29.0, Python 3.14) plus spglib, phonopy and matscipy, the framework's imports.

| | E_vac (63 at.) | barrier | peak image | path (eV above image 0) |
|---|---|---|---|---|
| DFT (CASTEP) | 3.627 | **0.299** | 5 | 0 .026 .095 .199 .285 .299 .234 .135 .054 .012 0 |
| GAP-18 | 3.626 | 0.218 | 5 | 0 .019 .063 .118 .199 .218 .154 .089 .046 .010 0 |
| o3d12 θ̂ | 2.823 | 0.238 | 4 | 0 .029 .111 .198 .238 .233 .230 .156 .076 .014 0 |
| o4d20 θ̂ | 3.578 | 0.098 | 5 | 0 .007 .025 .063 .096 .098 .081 .039 .016 .004 0 |

The POPS means give the same paths to 10⁻³ eV (their coefficients differ from θ̂ by ≤ 0.004 / 0.1; the 215-atom
vacancy energies already agreed to 4×10⁻⁴ eV). o4d20 misses the barrier by −0.20 eV (as Best et al.'s ACE, 0.12 eV),
o3d12 by −0.06 eV with a vacancy formation energy 0.80 eV low. The o3d12 path is flat-topped and asymmetric (images
4–6 within 8 meV, peak at 4): the middle image is not its saddle, so its delta row needs a climbing-image refinement
(from the converged band), and the protocol's max over images is then a non-smooth step there (committee check).
o4d20 is symmetric with its peak at the middle image, as the reference.

Next: (a) climbing-image refinement from each converged band, recording the saddle and its distance from the
protocol's max; (b) delta rows Φ(x_s) − Φ(x_min) + a₀ term; (c) a λ = 1 committee of `vacancy-path-ref` runs to
validate them; (d) coverage of CR3 × λ_F, B1, POPS against 0.299 eV.

**Climbing image and delta rows (2026-11-10, `analysis/vacancy_barrier.py`, `results/vacancy_barrier.json`;
rows `$SIQ_RUNS/delta/<m>_<v>_vacancy_barrier_row.npy`).** CI-NEB from the converged band (FIRE to fmax 1e-3: 121 /
140 steps), then Newton to |∇E| < 1e-11; saddles index 1, minima index 0. o3d12 saddle at image 4 (off-centre), o4d20
at the middle image. Refined barriers within 0.8 meV of the protocol's max.

| | Eb | Eb − DFT | headline σ / z | B1 σ / z | B2 σ / z | B3 z |
|---|---|---|---|---|---|---|
| o3d12 | 0.239 | −0.060 | 0.48 / 0.13 | 0.145 / 0.42 | 0.23 / 0.26 | 0.73 |
| o4d20 | 0.098 | −0.201 | 0.087 / 2.31 | 0.044 / 4.55 | 0.101 / 1.99 | 14.3 |

Committee check (K = 200 per method and model, `analysis/vacancy_barrier_committee.py`,
`results/vacancy_barrier_committee.json`): delta / committee σ ratios 1.01–1.06, so the delta σ stand (10 % rule).
o3d12's committee peak moves between images 4–6 (163 / 18 / 19) without changing σ. No discrepancy term: the
vacancy term is a formation-energy offset and cancels in E_s − E_m.

**Descriptor novelty of the saddle** (`analysis/barrier_novelty.py`): at o4d20 the migrating atom's novelty grade is
1.02 (88th percentile of test atoms), all saddle atoms are supported, per-atom σ median. The saddle is not missing
from training (hot 63-atom vacancy snapshots reach it), so B1's small σ is consistent and the miss is model-form
error on seen environments.

**Which configurations carry the CR3 σ** (`analysis/barrier_influence.py`, `results/barrier_influence.json`;
b_k = g · l_k per training configuration, Σ b_k² = the B3 variance to 0.1 % at o4d20, ≤ 6 % at o3d12 (replay)).
Share of the variance (enrichment over the type's share of configurations):

| QoI | o4d20 | o3d12 |
|---|---|---|
| barrier | vacancy 0.49 (×6), of which s ≥ 0.4 cells 0.31 (27 cells); divacancy 0.25 | one `bt` cell (#996) 0.89 |
| vacancy energy | vacancy 0.35 (×4), divacancy 0.14 | the same `bt` cell 0.87 |
| dumbbell | interstitial 0.36 (×8) | surface_110 / 111, screw dislocation; interstitial < 0.04 |
| hexagonal | interstitial 0.34 (×7) | `bt` 0.49, surface_111 0.21 |
| tetrahedral | interstitial 0.42 (×9) | `bt` 0.67 |
| controls: C44 / surface_100 | spread / surface_001 0.71 (×61) | crack, dia / surface_110 0.32 |

o4d20: every defect σ comes mostly from its own defect class, the barrier's from near-saddle vacancy cells; the
"right" configurations, yet the interstitials still miss without the discrepancy term (λ_F z −2.1 to −2.7): the
jackknife is centred, so a bias shared by every interstitial snapshot moves no leave-one-out fit and adds no width.
o3d12: one β-tin cell dominates the vacancy, tetrahedral, hexagonal and barrier σ: those widths (e.g. barrier 0.48
eV) are an accident of one high-influence configuration, not information about defects.

**The β-tin cell at o3d12** (`analysis/bt_cell.py`, `results/bt_cell_o3d12.json`): its column removed (rest
re-centred) and λ_F recomputed on the test split. λ_F 5.79 → 8.10 (+40 %): the cell also carries much of the test
force variance, so most QoI σ widen ×1.36–1.40. The vacancy formation energy σ halves (0.54 → 0.27 eV, z 1.63 →
3.23: covered → missed), the barrier's too (0.48 → 0.22, still covered), tetrahedral ×0.81 (z 2.07 → 2.56), dumbbell
×1.39 (z 3.38 → 2.43, still missed). Delta-σ coverage of these 18 QoIs: 15 → 14. o3d12's vacancy coverage without
the discrepancy term rests on this one cell.

**Type-level jackknife** (existing `results/type_jackknife_<m>.*`, 27 clusters = config types; RECON 2026-11-02):
o4d20 covers every defect without the term (interstitial z −0.33 to −0.47) but with σ 0.78–1.47 eV against errors
0.37–0.60 eV: 16/17 at both 68 % and 95 %, median σ/tol 1.77 (CR3 0.53), so it over-covers. o3d12: the interstitials
still miss (z −2.1 to −2.5). Neither route removes the need for the discrepancy term at both sizes.

**Interstitial trajectories as clusters** (`analysis/traj_jackknife.py`, `results/traj_jackknife_<m>.json`). No
trajectory tag in the extxyz; snapshots of one fixed-cell MD run share their cell exactly, so interstitial clusters are
(type, cell): three trajectories of 21, 20 and 19 snapshots plus 43 single variable-cell snapshots; every other
configuration stays its own cluster (2170 clusters against CR3's 2227). Exact leave-group-out with CR3's leverage
correction, A never formed; validated against the stored CR3 columns (centring-vector check 2.7e-6 / 2.1e-9). λ_F
recalibrated (5.79 → 5.71, 6.18 → 5.89). Delta σ, no discrepancy term:

| | o3d12 σ ratio / z (CR3 → traj) | o4d20 σ ratio / z (CR3 → traj) |
|---|---|---|
| dumbbell | ×1.17 / 3.38 → 2.88 | ×1.70 / 2.10 → **1.24** |
| hexagonal | ×1.24 / 2.77 → 2.23 | ×1.36 / 2.34 → **1.73** |
| tetrahedral | ×1.19 / 2.07 → **1.74** | ×1.49 / 3.44 → 2.31 |
| vacancy | ×1.02 / 1.63 → 1.59 | ×1.51 / 1.49 → 0.99 |
| 95 % covered (17 delta QoIs) | 14 → 15 | 12 → 15 |

Other QoIs change little (β-Sn, sh, elastic ×1.1–1.7 wider at o4d20, none lose coverage). The trajectory clusters
close about half of the interstitial gap: 1 of 3 interstitials covered without the term at o3d12, 2 of 3 at o4d20.
Not enough to drop the term. Caveat: the grouping was applied to the interstitials only, after seeing them fail;
the same (type, cell) rule over all types also groups whole surface, sp and sp2 types (fixed boxes, not trajectories:
626 of 2227 configurations in multi-configuration groups), so it does not generalise as written.

**Defect-bulk difference rows in the fit** (`analysis/diff_rows_post.py`, `results/diff_rows_post_<m>.json`; question:
can the fit itself get the defects right, so the discrepancy term is not needed?). The rows are the regression form of the
headline discrepancy estimator, one row per training defect cell, d_i = b_i − n_i (bulk thermal trend at x_i): 103 interstitial,
190 vacancy and 70 divacancy cells. They are added to the stored cr3 posterior (MAP hyperparameters, ARD prior) as an exact rank-363
update, at k × each cell's own energy-row weight. k is a sweep, not a choice. QoIs move linearly through the delta rows.
The harness check on the relaxed defects (`RUNS/samples/<m>_drp_k<k>.npz`) matches to 0.01 eV for o4d20 at k = 3 and 10, and is off
by up to 0.23 eV for o3d12 at k = 10, where |Δθ|/|θ| = 0.28 (harness values below where run).

| o4d20, k | 0 | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|---|
| test F (eV/Å) | 0.0816 | 0.0819 | 0.0833 | 0.0899 | 0.1002 | 0.1097 | 0.1129 |
| test E (meV/atom) | 3.50 | 3.54 | 3.62 | 3.88 | 4.30 | 4.80 | 5.00 |
| interstitial training intercept (eV) | +0.385 | +0.181 | +0.057 | +0.010 | +0.001 | 0.000 | 0.000 |
| err/tol dumbbell / hexagonal / tetrahedral | 3.7 / 4.1 / 6.0 | 2.7 / 3.1 / 4.9 | 0.9 / 2.0 / 3.8 | −0.2 / 1.2 / 3.7 | −0.8 / 0.8 / 4.0 | −0.8 / 0.7 / 4.1 | −0.8 / 0.7 / 4.1 |
| err/tol vacancy | 1.1 | 0.6 | 0.3 | 0.7 | 1.0 | 1.4 | 1.4 |
| within tol / covered 95 % (headline σ), of 17 | 13 / 12 | 14 / 15 | 15 / 15 | 15 / 15 | 15 / 15 | 12 / 11 | 12 / 10 |

o3d12 (test F 0.158 at k = 0): k = 3 leaves the interstitials at 7.9–8.9 × tol (test F +4 %). At k = 10 (test F +17 %, E +22 %), the harness
gives dumbbell / hexagonal / tetrahedral 5.4 / 3.5 / 6.7 × tol and vacancy 1.9. Larger k doubles the test errors (|Δθ| > |θ|).

Reading: no. For o4d20 the rows drive the training interstitial intercept to zero by k ≈ 10. The dumbbell, hexagonal and vacancy
QoIs then move within tolerance, and the test errors grow modestly (F +2 % at k = 3, +10 % at k = 10). But the **tetrahedral interstitial stays at
3.7–4.1 × tol for every k** (0.37–0.41 eV), outside the headline 95 % interval (σ 0.17 eV, z ≈ 2.2). The hot training snapshots
constrain the class's defect-bulk contrast as a whole, not each relaxed geometry; a bias specific to the tetrahedral site
remains. For o3d12 the rows cannot fix the defects without wrecking the fit. Caveats: the CR3 σ at k > 0 would come from the
augmented fit, not computed here, and a refit with these rows would also re-estimate the hyperparameters. So the discrepancy
term (or relaxed-defect training data) stays. A k ≈ 3 fit is a reasonable o4d20 variant if one wants better defects at
2 % force cost, but choosing it would use the QoIs.

**Elastic dipole tensors and relaxation volumes** (`analysis/defect_dipole.py`, `results/defect_dipole.json`; question: is
the dipole tensor, or the relaxation volume, a descriptor that flags the under-covered interstitials?).
(a) Relaxed QoI defects (216-site cell at a0, cell fixed): P = W_def − (n/n_b) W_bulk, ΔV = tr P / 3B with the model's own B.
The rows come from implicit differentiation, with the relaxation and a0 terms. Check: the bulk's total row vanishes (CR3 σ
6e-4 / 2e-4 eV), and o3d12's values are identical with the chunked Jacobians.

| ΔV/Ω (CR3 × λ_F σ) | vacancy | tetrahedral | hexagonal | dumbbell |
|---|---|---|---|---|
| o3d12 | −1.05 ± 0.31 | +0.65 ± 0.59 | +0.62 ± 0.36 | +0.84 ± 0.37 |
| o4d20 | −0.73 ± 0.28 | +0.62 ± 0.22 | +0.65 ± 0.19 | +0.69 ± 0.19 |

(b) Training labels, stress channel. Each cell's virial residual (DFT − model) minus n × the bulk per-atom thermal trend,
regressed on F_rms², intercept of tr P (eV). Interstitials: o3d12 −15 ± 11 (hot-cell mean −35), o4d20 +4 ± 7 (hot mean −21).
Vacancy, 215-atom cells: −1.6 ± 2.3 / −4.7 ± 0.9; 63-atom cells: +6.4 ± 0.6 / +1.4 ± 0.3. Only the 43 variable-cell interstitial
snapshots carry virials.

Reading: no, not as a flag for the interstitial miss.
- The three interstitial sites have the same relaxation volume within σ in both models (0.62–0.84 Ω), and the two models
  agree. The tetrahedral site, whose energy error survives the difference rows, does not stand out.
- In the stress channel the cold-limit interstitial residual is consistent with zero for both models (0.4σ to 1.3σ). The
  energy channel's intercept is +1.6 ± 0.06 / +0.39 ± 0.03 eV. So the bias the discrepancy term measures is an energy
  bias that the stress labels do not show. That fits the per-atom energy-weighting explanation: energies of large cells are
  weakly weighted, stresses are not.
- The hot cells do carry a large stress residual (−21 to −35 eV in tr P: the model's hot interstitials push outward more
  than DFT), but it is thermal and extrapolates away.
- The vacancy size split seen in energies appears in stress too (63 against 215 atoms differ by 6–8 eV in both models).
- Without DFT relaxation volumes for the harness defects, (a) is not scored. The literature values would make ΔV a
  stress-type QoI.

## §4 Multivariate conformal prediction with the CR3 covariance

K. Fisher, M. Herbst, J. Kermode, Y. Marzouk, arXiv:2609.31384 (2026): a GAP (SOAP, polynomial kernel)
covariance; the score s = d^(−1/2) ‖Σ^(−1/2)(Y − μ)‖ over the energy and all forces of a configuration;
a propagation-corrected variant that recalibrates on L·Y; conformal risk control losses. Mahalanobis score as in
Braun et al., arXiv:2507.20941.

It ports directly: for a linear model the covariance of any output vector is G C Gᵀ, which the propagation already
builds. Points to handle:

- Rank: CR3 is rank 192 at o3d12 against 646 outputs for a 215-atom cell, so Σ is singular: Σ + εI, or a
  pseudo-inverse in its range plus a separate score for the out-of-range part. §0 chooses ε.
- The current per-atom method is already a local multivariate conformal set (a 3×3 Mahalanobis ellipsoid per atom,
  `forces_q_mahal`, Mondrian groups). The new option is the whole-configuration score.
- For: a joint finite-sample guarantee per configuration, and a test of the cross-atom correlations, which decide
  error cancellation in formation energies and barriers.
- Against: a few hundred calibration scores (one per configuration), not tens of thousands of atoms, and a
  different guarantee (joint, not per atom). The propagation-corrected form needs labels of each QoI type;
  that is calibration on DFT QoIs, ruled out here, except where training-like configurations exist (their
  [E_bulk, E_vac] pairs).

Steps:

1. §0 scores: choose ε, and see which covariance gives a near-χ² score distribution.
2. One scale from the conformal quantile of whole-configuration scores on the validation split, in place of λ_F.
3. Compare with the headline on the 21 QoIs and on the per-atom force tests.

Decision: if QoI coverage holds and the scores favour CR3 over B1, it strengthens the paper's covariance-shape
argument. If not, it is a one-paragraph negative result.

**Results, steps 1–3 (2026-11-10; `analysis/mv_conformal.py` → `results/mv_conformal_<m>.json`).** Score
s_c = (e_cᵀ Σ_c⁻¹ e_c / D_c)^½ over all live force components of a test configuration, Σ_c = λ² M_c M_cᵀ + ν² I, M_c = F_c L.
(λ, ν) come from maximum Gaussian likelihood on a calibration half of the test split, then one conformal quantile q (α = 0.1);
joint coverage is measured on the other half, over 200 random splits. The region's parameter part, λ_mv = q λ, replaces λ_F for the
QoIs (delta method, no discrepancy term; λ_mv and ν from the whole test split). No QoI enters.

| | o3d12 CR3 | o3d12 B1 | o4d20 CR3 | o4d20 B1 |
|---|---|---|---|---|
| ν (eV/Å per component, in the region) / ν over the RMS error | 0.131 / 0.56 | 0.123 / 0.53 | 0.039 / 0.38 | 0.004 / 0.04 |
| λ_mv (paper scale λ_F or q̂_B1) | 14.2 (5.82) | 89.5 (39.8) | **8.27 (6.17)** | 26.6 (8.87) |
| joint 90 % coverage, held-out halves (mean; 5–95 %) | 0.90 (0.84–0.96) | 0.90 | 0.90 (0.84–0.96) | 0.90 |
| by kind: bulk / point defect / surface / extended (n per half 80 / 20 / 8 / 3) | 0.93 / 1.00 / 0.40 / 0.69 | 0.92 / 1.00 / 0.44 / 0.73 | 0.95 / 0.96 / 0.28 / 0.62 | 0.99 / 0.66 / 1.00 / 0.88 |
| QoIs covered at 95 % (paper scale) of 17 | 17 (14) | 15 (13) | **16 (12)** | 16 (10) |
| median σ/tol (paper scale) | 5.78 (2.37) | 3.85 | **0.72 (0.53)** | 1.08 |
| defect z: dumbbell / hexagonal / tetrahedral / vacancy | 1.4 / 1.1 / 0.8 / 0.7 | 2.3 / 1.8 / 1.8 / 2.0 | 1.6 / 1.8 / 2.6 / 1.1 | 1.7 / 1.8 / 3.0 / 1.0 |

Reading:
- The joint guarantee holds on average by construction, but not by kind. Surfaces are under-covered for CR3 at both sizes
  (0.28–0.40), and point defects for B1 at o4d20 (0.66). With 8 surface configurations per half these are coarse.
- At o4d20 the CR3 shape needs only a small nugget, and its joint scale is 1.34 × λ_F. That covers 16 of 17 QoIs **without
  the discrepancy term**, at median σ/tol 0.72, narrower than the type-level jackknife (1.77) for the same coverage. Only the
  tetrahedral interstitial misses (z 2.6), the QoI the difference rows could not fix either. B1 needs 3 × its scale for the
  same coverage and gives wider QoI intervals (1.08).
- At o3d12 the nugget is half the error, and λ_mv is 2.4 × λ_F. Every QoI is covered, but at σ/tol 5.8: wide, not informative.
- Why the joint scale is larger than λ_F: λ_F is a per-component RMS, set mostly by small bulk cells. The whole-configuration
  likelihood weighs the large cells, where §0 found 1.7–2.8 × the variance in the leading directions. The nugget takes the white
  part, so λ is fitted to the correlated part, which is what propagates to QoIs.
- Decision (plan): QoI coverage holds and the scores favour CR3 over B1 at o4d20 (narrower intervals for the same coverage;
  B1 under-covers the defect cells). This is a candidate replacement for λ_F at o4d20 that removes most of the need for the
  discrepancy term. Caveat: the shape was chosen on the same test split as λ_F, after §0 showed the nugget was needed.

**Is the λ_mv QoI covariance jointly calibrated?** (2026-11-11; `qoi_joint_score.py` "lam_mv" rows: the CR3 and B1 delta
covariances × (λ_mv / paper scale)²; the ν nugget is white per force component and does not reach the QoIs. The CR3 rows reproduce
the per-QoI counts above, 17/17 and 16/17.) τ = 0.05:

| model | method | k95 | gm CRPS/t | d²/n | gm abs w | LS/n |
|---|---|---|---|---|---|---|
| o4d20 | CR3 λ_F | 13 | 0.296 | 1.30 | 0.46 | 0.73 |
| | CR3 λ_F + disc (headline) | 16 | **0.282** | 0.86 | 0.29 | **0.60** |
| | CR3 λ_mv | 16 | 0.323 | 0.74 | 0.34 | 0.71 |
| | CR3 λ_mv + disc | 17 | 0.313 | 0.52 | 0.22 | 0.67 |
| | B1 λ_mv | 16 | 0.382 | 0.76 | 0.16 | 1.04 |
| o3d12 | CR3 λ_F + disc (headline) | 17 | **0.980** | 1.01 | 0.61 | **1.71** |
| | CR3 λ_mv | 17 | 1.565 | 0.47 | 0.31 | 2.15 |
| | B1 λ_mv | 15 | 1.376 | 0.73 | 0.30 | 2.27 |

Reading: at o4d20, CR3 λ_mv without the discrepancy term is jointly calibrated (d²/n 0.74, gm |w| 0.34, both inside their
bands), so the whole-configuration scale is a principled alternative to the discrepancy term there. It is not better: the
headline (λ_F + correlated discrepancy) has the better proper scores (LS/n 0.60 against 0.71, CRPS 0.282 against 0.323), and λ_mv
plus the term over-covers (gm |w| 0.22, below the band). At o3d12, λ_mv is too wide (CRPS 1.57 against 0.98; d²/n and gm |w| at the
lower edge). B1 at its λ_mv is worse than CR3 at both sizes (o4d20 gm |w| 0.16: too wide in the directions that matter). Paper:
λ_mv is one sentence in the discussion (a discrepancy-free route that is jointly calibrated at o4d20 but scores worse than the
headline); the headline is unchanged.

## §5 Conformal risk control: an exact per-atom guarantee and a selection score (2026-11-11)

Suggested after §4, following Fisher et al. (arXiv:2609.31384, Sec. 3.5, 4.3). They control losses of the whole-configuration
whitened residual Δ = (d λ Σ)^-½ (Y − μ) (a truncated linear loss, ‖Δ‖₁, the 90 % quantile of Δ's components) on energies and
forces, 120 calibration configurations; the guarantee is a marginal expectation over configurations; they compare methods by
set volume at fixed coverage but select nothing with it, and run no propagated CRC.

Here (`analysis/crc.py`, `results/crc.json`) the score is the paper's per-atom one (the 3×3 Mahalanobis distance of each atom
under its method's covariance), not Fisher et al.'s whole-configuration Δ; the configuration enters through the loss
L_c(q) = #{i ∈ c: s_i > q} / n̄, non-increasing in q, bounded by B = n_max / n̄ (n_max from atom counts of the calibration and
test splits, no labels; Mondrian: per group). CRC: q̂ = inf{q : (Σ_c L_c(q) + B)/(n + 1) ≤ α}, so E[miscovered atoms of a new
configuration] ≤ α n̄: the atom-weighted miscoverage the paper targets, now exact. The paper's atom-weighted quantile over pooled
atoms has no exact guarantee (atoms of one configuration are not exchangeable); ace-jax's configuration-weighted quantile has
one, for the configuration-weighted target, and under-covers atoms (force_onegroup: q ~12 % lower).

**A. The paper's per-atom scale** (T_val calibration scores of the ARD posterior, 422 / 444 configurations, n̄ 74 / 71, B 2.9 / 3.1;
coverage on the 17,537 test atoms):

| model, α | atom-weighted (paper) q / cov | configuration-weighted q / cov | CRC q / cov | Mondrian cov: atom / cfg-w / CRC |
|---|---|---|---|---|
| o3d12, 0.1 | 17.32 / 0.879 | 15.45 / 0.825 | 17.61 / 0.886 | 0.879 / 0.895 / 0.902 |
| o3d12, 0.05 | 20.33 / 0.934 | 18.96 / 0.913 | 20.92 / 0.940 | 0.935 / 0.946 / 0.959 |
| o4d20, 0.1 | 16.49 / 0.902 | 13.84 / 0.836 | 16.85 / 0.908 | 0.905 / 0.902 / 0.923 |
| o4d20, 0.05 | 20.16 / 0.951 | 17.89 / 0.926 | 20.98 / 0.957 | 0.949 / 0.951 / 0.971 |

Reading: the exact guarantee costs 2–4 % in q (B / (n + 1) ≈ 0.007 at n ≈ 430). The one-group shortfall at o3d12 (0.886) is the
T_val → test transfer the paper already reports, not the rule. Mondrian CRC (8 ARD groups, per-group bound = the group's most atoms
in one configuration) reaches target on test at both sizes; the two smallest groups (74 and 51 T_val configurations) are
conservative. Paper candidate: replace the Section 2.5 quantile by CRC (one paragraph, one equation; "≈ 1 − α" becomes "≥ 1 − α in
expectation over configurations").

**B. A selection score: calibrated region size.** Test split, 239 configurations, 200 random configuration-level halves: every
candidate (model × method) CRC-calibrated on one half, so none can look better by being overconfident; size = mean log radius of
the calibrated regions on the other (log of the geometric-mean semi-axis from log det, eV/Å; the √trace radius ranks the same).
Risks: overall, worst of three shell-distortion bands (the ARD groups merged: the top band has 31–41 test configurations, too few
for per-group CRC on a half) and worst configuration type (≥ 100 atoms). α = 0.1:

| candidate | size (vol) | risk | worst band | worst type / type RMS dev |
|---|---|---|---|---|
| o4d20 ellipsoid | **−2.11** | 0.076 | 0.082 | 0.206 / **0.057** |
| o4d20 B2 | −1.90 | 0.078 | 0.135 | 0.363 / 0.108 |
| o4d20 B1 | −1.86 | 0.077 | 0.087 | 0.220 / 0.080 |
| o4d20 CR3 (one scale) | −1.85 | 0.077 | 0.091 | 0.224 / 0.085 |
| o3d12 ellipsoid | −1.46 | 0.077 | 0.089 | 0.161 / 0.049 |
| o3d12 B1 | −1.26 | 0.076 | 0.092 | 0.164 / 0.060 |
| o3d12 B2 | −1.26 | 0.077 | 0.100 | 0.209 / 0.054 |
| o3d12 CR3 (one scale) | −1.09 | 0.076 | 0.087 | 0.167 / 0.069 |

(Risk 0.076, not 0.1: the B / (n + 1) ≈ 0.025 margin at n ≈ 120.) The ellipsoid is smallest in every split at both sizes and
α = 0.05, 0.1, 0.2, with Mondrian (band) CRC too; o4d20 beats o3d12 for every method by 0.60–0.76 in log radius (1.8–2.1×).
Selection: choosing by size on a third (plain quantiles, 2-fold) and CRC-calibrating the choice on another third gives risk 0.065
on the last (α = 0.1; 0.014 at 0.05, 0.166 at 0.2); choosing and calibrating on the same two thirds gives 0.083, still ≤ α
because the winner is clear. Mondrian CRC on half-splits is infinite at α = 0.05 (too few configurations per band).

Reading:
1. *As a model-selection score it works for models:* o4d20 over o3d12 for every method, by a wide margin, with the guarantee
   intact after selection.
2. *For UQ methods it picks the POPS ellipsoid, not the headline.* At one calibrated scale the ellipsoid's per-atom shape tracks
   the force error best: smallest regions, and also the most even across bands and types (as §2 found). CR3 with one scale has
   the largest regions at o3d12 and ties B1 at o4d20. This is a force-level verdict. The QoI scores disagree: jointly the
   ellipsoid is worse calibrated than CR3 (§2e: o4d20 d²/n 1.79 against 1.30; LS/n with the discrepancy 0.70 against 0.60). A
   per-atom score cannot see cross-atom correlations, which is what QoIs depend on.
3. *So the score should match the use.* Per-atom CRC size selects for per-atom forces; for QoIs the relevant region is
   whole-configuration (Fisher et al.'s Δ, §4's score with its nugget), whose size reflects the correlations. Next: the same
   CRC selection with the whole-configuration score and Fisher et al.'s quantile loss (fraction of whitened components outside
   ±λ), from §4's covariances. If that ranks CR3 above the ellipsoid, the two scores together say: ellipsoid for per-atom
   forces, CR3 for anything correlated.
Caveats: the candidates' scales do not matter (CRC recalibrates), but their shapes were all fitted on training data only; the
selection uses the test split, so for the paper it would need its own held-out split (the test split is the paper's
calibration and evaluation set).

**C. The whole-configuration score** (`analysis/crc_config.py`, `results/crc_config.json`). Per configuration and method
Σ_c = λ_s² C_c + ν² I (C_c from the method's factor, or F_c Σ_e F_cᵀ for the ellipsoid; (λ_s, ν) by Gaussian ML, as §4), Fisher
et al.'s whitened residual w_c = Σ_c^-½ e_c (symmetric root). Two losses, both bounded by 1: *joint* 1{|w_c|/√D_c > t} (the
Mahalanobis region, §4) and *fraction* #{|w_ci| > t}/D_c (a box in whitened coordinates; the monotone form of their quantile
loss). Size: log t + log det Σ_c / (2D_c), the log geometric semi-axis per component (eV/Å, as B). 100 splits: shape on a
quarter, CRC on a quarter, size and risk on the other half; selection on thirds. α = 0.1:

| candidate | joint: size / risk | fraction: size / risk | joint risk by kind: bulk / point defect / surface / extended |
|---|---|---|---|
| o4d20 B1 | **−2.79** / 0.098 | **−2.82** / 0.088 | 0.02 / 0.35 / 0.02 / 0.15 |
| o4d20 CR3 | −2.65 / 0.103 | −2.67 / 0.089 | 0.05 / 0.10 / 0.60 / 0.42 |
| o4d20 B2 | −2.17 / 0.100 | −2.25 / 0.087 | 0.06 / 0.00 / 0.58 / 0.32 |
| o4d20 ellipsoid | −2.13 / 0.102 | −2.20 / 0.086 | 0.07 / 0.00 / 0.61 / 0.27 |
| o3d12 B1 | −1.86 / 0.100 | −1.97 / 0.087 | 0.07 / 0.00 / 0.59 / 0.33 |
| o3d12 CR3 | −1.84 / 0.101 | −1.97 / 0.087 | 0.07 / 0.00 / 0.62 / 0.33 |
| o3d12 ellipsoid | −1.76 / 0.104 | −1.83 / 0.086 | 0.07 / 0.00 / 0.63 / 0.30 |
| o3d12 B2 | −1.70 / 0.100 | −1.82 / 0.087 | 0.07 / 0.00 / 0.60 / 0.33 |

Same order at α = 0.05 and 0.2 (at 0.05 CR3 and B1 tie at o4d20). Selection: o4d20 B1 in 89–100 % of splits, risk on the held-out
third 0.106 (α 0.1), 0.055 (0.05), 0.201 (0.2) for the joint loss; 0.093 / 0.041 / 0.194 for the fraction loss. ML shapes: the
nugget ν at o4d20 is 0.031 eV/Å for CR3 but 0.003 for B1 (B1's own shape carries the white part), 0.1 for B2 and the ellipsoid
at both sizes.

Reading:
1. *The whole-configuration score reverses the per-atom verdict on methods.* With cross-atom correlations in the score, the
   parameter-covariance shapes (B1, CR3) give smaller regions per component than the POPS box and ellipsoid: at o4d20 by
   0.47–0.66 in log (1.6–1.9×), at o3d12 only by 0.08–0.16 (~1.1×), both losses; the per-atom score had the ellipsoid
   smaller than CR3 by 0.27 (o4d20) and 0.37 (o3d12) in log. This agrees
   with the QoI joint scores (§2e: CR3 over the ellipsoid jointly), which also depend on the correlations.
2. *Models: o4d20 over o3d12 for every method*, as with the per-atom score (by 0.7–0.9 in log for B1/CR3).
3. *B1 against CR3 is not settled by size.* B1 is smaller at o4d20 but misses point-defect cells (0.35 of them outside at α 0.1),
   CR3 misses surfaces (0.60) and extended defects (0.42); every method misses surfaces at o3d12. The guarantee is marginal over
   configurations; by kind it fails for all methods, as §4 found. A kind-aware (Mondrian) whole-configuration CRC needs more
   configurations per kind than the test split has (8 surfaces per half).
4. *For the paper:* the score must match the use. Per-atom CRC size for per-atom forces (picks the ellipsoid shape), whole-
   configuration CRC size for correlated predictions (picks the parameter-covariance shapes, the family CR3 belongs to). Next:
   the model ladder (o3d8 … o4d20, the accuracy-cost fits on the same split) with CR3 and B1, to see where size saturates.

**C, ladder (2026-11-12).** `crc_config.py ladder` → `results/crc_ladder.json`; `analysis/crc_pareto.py` →
`results/crc_pareto.json`. CR3 (the headline) and B1 at all seven sizes, B2 and the ellipsoid where fitted (o3d12, o4d20); the
same 100 random test halves. Cost: A100 float64, ace-jax CUDA kernel, µs per atom-step (`results/ace_pareto.json`). Size: log
geometric semi-axis per force component (eV/Å) at α = 0.1; gap = size − log test force RMSE; worst = the kind with the highest
miss rate.

| model | cost | RMSE_F | joint B1 | joint CR3 | worst kind (B1, joint) | per-atom best |
|---|---|---|---|---|---|---|
| o3d8 | 0.042 | 0.244 | −1.49 | −1.44 | surface 0.65 | B1 −0.69 |
| o3d10 | 0.045 | 0.202 | −1.63 | −1.60 | surface 0.63 | B1 −0.94 |
| o3d12 | 0.064 | 0.158 | −1.86 | −1.84 | surface 0.59 | ellipsoid −1.47 |
| o4d12 | 0.080 | 0.144 | −1.95 | −1.90 | surface 0.42 | B1 −1.28 |
| o4d14 | 0.142 | 0.118 | −2.30 | −2.13 | extended 0.50 | B1 −1.50 |
| o4d16 | 0.286 | 0.101 | −2.51 | −2.31 | extended 0.34 | B1 −1.63 |
| o4d20 | 1.233 | 0.082 | −2.79 | −2.65 | point defect 0.35 | ellipsoid −2.11 |

Reading:
1. *No saturation yet.* Joint B1 size falls by 0.42, 0.60, 0.30 and 0.19 per e-fold of cost over the last four steps
   (o3d12 → o4d20); test log RMSE_F by 0.14 per e-fold on the last step. The front is B1 at every size for both whole-
   configuration losses; for the per-atom loss it is B1, with the ellipsoid where it was fitted.
2. *Calibrated size improves faster than the point error from o4d14 on.* The B1 gap (joint) is −0.01 at o3d12 and o4d12, then
   −0.16, −0.21, −0.29: as the model grows, more of the residual is in B1's covariance directions (the ML nugget ν falls from
   0.136 to 0.003 eV/Å). CR3's gap stays near 0 up to o4d16 and is −0.14 at o4d20. The per-atom gaps stay flat (B1 0.58–0.73,
   ellipsoid 0.38–0.39): per atom, calibration costs a constant factor over RMSE at every size.
3. *The failure mode moves with complexity.* The marginal risk is at α throughout, but the worst kind changes from surfaces
   (miss rate 0.59–0.65 to o3d12) to extended defects (o4d14, o4d16) to point defects (o4d20), and its miss rate falls from 0.65
   to 0.35. The small models fail on surfaces for every method; that is model error, not a covariance-shape choice.
4. *Extend the ladder?* Not now. The last step costs 4.3× for 0.28 in log (25 % smaller regions); o4d24 would cost several
   times o4d20 again and CR3 is already at rank K. The D sweeps (weights, wL) go at o4d14 / o4d16, where the front is steepest
   per unit cost.

**D. Choosing the fixed hyperparameters under CRC (plan, 2026-11-11; after the ladder results).** Two choices have been held
fixed so far: the energy / force / virial weights (`weights_acejax.json`, per configuration type: default E 30, F 1, V 1, lower
for liq, amorph and sp; one shared MAP noise σ, `--noise shared`), and wL, the basis construction's angular weight (channel level
n/NZ + wL·l ≤ total degree, ace-jax `basis/spec.py`, default 1.5: a larger wL spends the basis on radial rather than angular
detail, and changes the evaluation cost too).

Principles:
1. *The score must contain the trade-off.* A force-only score always prefers more force weight. For the weights, use a
   whole-configuration score over [E_c, F_c] (Fisher et al.'s 1 + 3n components), or report energy and force sizes as a
   two-axis Pareto front and choose by a stated priority. For the QoIs (energy differences of relaxed structures), a proxy that
   stays within the rules (no calibration on DFT QoIs): Fisher et al.'s propagated score L·Y on energy differences between labelled
   configurations of one kind (e.g. vacancy against bulk cells). For wL: per-atom and whole-configuration size against evaluation
   cost, a Pareto front at matched cost beside the degree ladder.
2. *Select on the training hold-out, calibrate on the test split.* The test split gives ~80 configurations per third; the paper
   already has T_fit / T_val (444 validation configurations). Fit each candidate on T_fit, score it on T_val, refit the winner on
   all of T, then CRC-calibrate and report on the test split. Selection never touches the calibration data, so the guarantee holds.
3. *A per-kind constraint.* Choose the smallest calibrated size among candidates whose worst-kind (Mondrian) risk is within
   α + δ, so the winner cannot buy small regions by failing surfaces or defects (§5 C: every method fails some kind).
4. *Stable, flat optima.* Report how often each setting wins across resamples; take the cheapest setting within the
   split-to-split noise of the best (the conformal form of the one-standard-error rule).
5. *Evidence for continuous noise ratios, CRC to decide and validate.* Evidence-chosen per-quantity σ_E, σ_F, σ_V
   (`runs/accuracy_cost/ace_per_quantity_noise`, o3d8–o4d16; checked 2026-11-12: they are the 2026-10-05 fits made before `--noise shared`, when per-quantity σ was the
   default, moved aside when the shared refits replaced them, so `cmd.txt` and `fit.log` name `ace/<m>`; older code, so refit them
   on the current version for the sweep rather than reuse them)
   are cheap and smooth but trust a Gaussian noise model the discrepancy results show is misspecified. CRC compares them with the
   hand weights and settles discrete choices (wL). If the two agree, the cheap route suffices.
6. *Off distribution.* Repeat the held-out-type refits (crack, screw) for the top one or two candidates.

Cost:
- Weights need no new rows: per-(configuration type, quantity) QR statistics built once (ace-jax's per-type noise path has this
  structure); any weight vector is a rescale-and-stack of those blocks and one P × P QR. CR3 needs one pass over the rows (a
  design cached node-locally, never on /storage). Minutes per candidate at o4d14–o4d16, ~20 min at o4d20.
- wL changes the basis. If the candidate bases are all subsets of one union basis, one superset design and a column slice per
  candidate (the column identity checked for degree, 2026-11-11: every o3d12 column equals an o4d20 column to 1e-14, scale 1,
  on 579 rows of three sh configurations; to be repeated across wL and on mixed configurations). Otherwise a refit per value.
- Sweep at o4d14 or o4d16, not o4d20: fits are cheap there and CR3 is full rank (P < K = 2227 configurations); at o4d20 CR3 has
  reached rank K (2227 of 5476 parameters, max leverage 0.998), which would distort the shape being compared.

Steps:
1. Weights at one mid-ladder model: the hand weights, the evidence per-quantity σ, and a few E/F ratios (e.g. E weight × 1/3,
   1, 3, 10), scored on T_val by the joint [E, F] CRC size under the per-kind constraint.
2. wL ∈ {1.0, 1.25, 1.5, 1.75, 2.0} at matched cost, as a Pareto front against the degree ladder.
3. The winners refitted on all of T, CRC-calibrated and reported on the test split, with the held-out-type check.

**D, step 1 results (o4d16, 2026-11-12).** `analysis/weight_sweep.py` → `results/weight_sweep_o4d16.json` (33 min on lestrade).
T split by configuration type (seed 20261112): T_fit 1781, T_val 446 configurations (426 with live force components); per-type
QR factors once, each candidate an exact rescale-and-merge, its own MAP (sigma_c, sigma). B1 on T_val, whole-configuration CRC,
100 paired halves. Size: log semi-axis, eV/atom (E, one component per configuration) and eV/A per component (F); α = 0.1, joint
loss; worst = the kind with the highest miss rate.

| candidate | sigma_c / sigma (E, F, V) | T_val RMSE E (meV/atom) / F | E size | E worst | F size | F worst | front |
|---|---|---|---|---|---|---|---|
| Ex10 | 1.9e4 / 0.132 | 5.34 / 0.136 | −6.14 | point 0.21, ext 0.24 | −2.42 | point 0.27 | yes |
| Ex3 | 2.8e3 / 0.101 | 5.98 / 0.112 | −5.80 | point 0.24 | −2.50 | ext 0.30 | yes |
| hand | 1.2e3 / 0.089 | 5.11 / 0.101 | −5.41 | ext 0.47 | −2.56 | ext 0.34 | yes |
| flat | 1.3e3 / 0.109 | 4.35 / 0.101 | −5.31 | ext 0.43 | −2.54 | ext 0.33 | no (hand) |
| noV | 8.2e2 / 0.087 | 5.02 / 0.100 | −5.22 | ext 0.47 | −2.56 | ext 0.34 | no (hand) |
| E/3 | 8.4e2 / 0.082 | 7.08 / 0.096 | −4.88 | ext 0.60 | −2.58 | ext 0.36 | yes |
| hand_pq | 4.8e2 / 1.84, 0.077, 0.169 | 18.64 / 0.093 | −4.11 | ext 0.63 | −2.59 | ext 0.36 | yes |

Split-to-split spread (p05–p95) is ~0.5 in E size and ~0.17 in F size, but the comparison is paired: Ex10 is smallest in E in 99
of 100 splits, hand_pq smallest in F in 88. Same order at α = 0.05 and 0.2 and for the frac loss, apart from swaps within the near-tie of hand,
flat and noV (≤ 0.04 in log).

Reading:
1. *The trade-off is lopsided.* Along the front (hand_pq → E/3 → hand → Ex3 → Ex10) E size changes by 2.0 in log, F size by
   0.17. From hand, Ex3 buys regions 1.5× smaller in E for 5 % larger in F; Ex10 2.1× for 15 %.
2. *More energy weight also helps per kind.* The E view's worst-kind miss rate falls from 0.47 (hand; extended defects) to
   0.24 (Ex3, Ex10), and the F view's from 0.34 to 0.27–0.30: the per-kind constraint (principle 3) points the same way.
3. *Calibrated size and RMSE disagree for energies.* Ex3 has a higher T_val energy RMSE than hand (5.98 vs 5.11 meV/atom) but
   E regions 1.5× smaller; flat has the lowest RMSE (4.35) but larger regions than hand. The regions follow how well B1's
   energy covariance matches the residuals, which more energy weight improves (sigma_c rises 15× from hand to Ex10: a weaker
   prior, the energy directions then set by the data). Choosing by RMSE would choose differently.
4. *The evidence route fails for energies.* Per-quantity MAP noise sets sigma_E = 1.84 (energies all but ignored: 18.6
   meV/atom, E regions 3.6× hand's) for the smallest F regions. Principle 5's cheap route disagrees with CRC, so CRC decides.
5. *The per-type downweights and virials do not matter at this size:* flat and noV are within noise of hand in F and slightly
   worse in E.

Open: the choice along the front needs a stated E-against-F priority (the QoIs are energy differences, which argues for Ex3
or above). Next, a finer E ladder (×3, ×5, ×10, ×30) to find where the F cost steepens; the energy-difference proxy (pairs of
labelled configurations of one kind, principle 1) on the same candidates; then CR3 for the finalists.

**D, energy-weight ladder and energy-difference views (o4d16, 2026-11-12).** `weight_sweep.py o4d16 all` →
`results/weight_sweep_o4d16_all.json` (11 min on lestrade; the base set and the ladder E × {1/3, 2, 3, 5, 10, 30, 100} on the same
splits; earlier runs `weight_sweep_o4d16{,_eladder}.json` agree to 3e-6). Two energy-difference views on T_val, which stay
within the rules (labelled training configurations only): dE_type, 216 disjoint pairs of configurations of one type, and
dE_form, 114 defect / surface / extended-defect configurations each against a diamond cell, ΔE = E_a − (n_a / n_b) E_b (eV).
Size: log semi-axis at α = 0.1, joint loss; worst = the kind with the highest miss rate.

| E weight | σ_c / σ | RMSE E (meV/atom) / F | E | F | dE_type | dE_form | worst kind (E view) |
|---|---|---|---|---|---|---|---|
| × 1/3 | 8.4e2 / 0.082 | 7.08 / 0.096 | −4.88 | −2.58 | −1.40 | +0.13 | ext 0.60 |
| × 1 (hand) | 1.2e3 / 0.089 | 5.11 / 0.101 | −5.41 | −2.56 | −1.70 | −0.47 | ext 0.47 |
| × 2 | 1.9e3 / 0.096 | 5.43 / 0.107 | −5.69 | −2.53 | −1.93 | −0.79 | ext 0.29 |
| × 3 | 2.8e3 / 0.101 | 5.98 / 0.112 | −5.80 | −2.50 | −2.05 | −0.89 | point 0.24 |
| × 5 | 6.1e3 / 0.112 | 6.20 / 0.120 | −5.98 | −2.47 | −2.17 | −0.93 | point 0.24 |
| × 10 | 1.9e4 / 0.132 | 5.34 / 0.136 | −6.14 | −2.42 | −2.26 | −1.04 | ext 0.24 |
| × 30 | 1.2e5 / 0.189 | 10.16 / 0.188 | −6.22 | −2.27 | −2.28 | −0.96 | point 0.22 |
| × 100 | 6.9e5 / 0.293 | 17.69 / 0.285 | −6.13 | −2.04 | −2.15 | −0.63 | point 0.25 |

Reading:
1. *The F cost steepens after × 10.* Per step in E weight, F size changes by +0.02 to +0.05 up to × 10, then +0.15 (× 30) and
   +0.23 (× 100). E size improves by 0.73 from × 1 to × 10, then by only 0.08 more.
2. *The energy-difference proxies peak at × 10–30.* dE_form (the formation-energy proxy, the QoIs' nearest relative) is
   smallest at × 10 (−1.04 against −0.47 for hand: regions 1.8× smaller) and gets worse again beyond; dE_type at × 30, within
   0.02 of × 10.
3. *Per-kind risk agrees.* The E view's worst-kind miss rate falls from 0.47 (hand) to 0.22–0.25 at × 3 and above; dE_form's from
   0.15 to 0.11.
4. *Shared MAP noise grows with the energy weight* (σ 0.089 → 0.293 eV/Å; σ_c 15× from × 1 to × 10): heavy energy weight makes
   the shared σ explain energies, so force residuals look over-dispersed and F regions widen. MAP flags L-BFGS-B `ABNORMAL` at
   × 10 and × 30 in the 24-thread run and at × 100 in the 32-thread run, at the same θ to 3 digits: line-search ends at the
   optimum, not failures.

**D, the selection rule (2026-11-12).** `analysis/selection.py` → `results/selection.json`; needs the per-split sizes now saved by
`weight_sweep.py` and `crc_config.py` (`split` keys; the ladder rescored on its original splits). The rule, for candidates fitted
on T_fit and scored on T_val by paired halves:
1. *Declare the use first:* views and priorities π_v (fixed before scoring), α. Aggregate J[c, s] = Σ_v π_v (L[c, v, s] −
   L[ref, v, s]) + λ log cost_c. Log sizes add, so J is the log of a priority-weighted geometric mean of region volumes; λ is the
   price, in nats of size, of a factor e in evaluation cost (0: size only).
2. *Tolerance (one-SE):* b = argmin mean J; N = {c: mean J_c − mean J_b ≤ τ_c}, τ_c = sd_s(J_c − J_b) / √2. The halves are
   overlapping resamples of one T_val, so sd/√S understates the noise; the half-sample sd over √2 is the noise of the difference
   on all of T_val. Over all candidates, not a Pareto set first: on one view that filter leaves only b.
3. *Choose in N:* the cheapest, then the smallest worst-kind risk (max over the declared views), then the simplest (nearest the
   default weights).
4. *No single use:* minimax regret over the declared profiles (smallest worst dJ).
5. *Then:* refit on all of T, CRC-calibrate on the test split, report (step 3). Selection never sees the calibration data.

Weight sweep (o4d16; profiles: forces F; energies E ½, dE_type ¼, dE_form ¼; balanced F ½ and the three energy views ⅙ each;
formation F ½, dE_form ½):

| profile | choice | N (within τ) | win rate of choice | baselines |
|---|---|---|---|---|
| forces | hand_pq | hand_pq only (hand +0.034, τ 0.013) | 0.89 | min RMSE_F: hand_pq |
| energies | × 30 | × 30, × 10 (+0.026, τ 0.059); × 30 by worst kind 0.22 vs 0.33 | 0.66 | min RMSE_E: flat |
| balanced | × 10 | × 10 only (× 5 +0.033, τ 0.028) | 0.73 | |
| formation | × 10 | × 10, × 5, × 3, × 2 (+0.027 … +0.074); × 10 by worst kind | 0.49 | min size dE_form: × 10 |
| minimax regret | × 5 | worst dJ 0.157 (× 10: 0.172, from the forces profile; hand 0.672) | | |

Ladder (B1 / headline / B2 / ellipsoid, o3d8 … o4d20; cost A100 fp64 µs/atom-step). The λ intervals on which each candidate
has the smallest mean J, the lower envelope of mean size against log cost:
- whole-configuration F: o4d20/B1 [0, 0.19), o4d16/B1 [0.19, 0.30), o4d14/B1 [0.30, 0.55), o3d12/B1 [0.55, 0.64), o3d10/B1
  [0.64, 1.93), o3d8/B1 beyond;
- per-atom F: o4d20/ellipsoid [0, 0.22), o3d12/ellipsoid [0.22, 1.50), o3d10/B1 [1.50, 3.34), o3d8/B1 beyond;
- both, ½ each: o4d20/B1 [0, 0.18), o4d16/B1, o4d14/B1 [0.25, 0.36), o3d12/ellipsoid [0.36, 0.93), o3d10/B1, o3d8/B1.
With the tolerance at λ = 0.25 (F) the rule takes o4d14/B1 over the best, o4d16/B1 (+0.038, τ 0.057), and at 0.5 o3d12/B1
over o4d14/B1: the one-SE step moves down the ladder whenever adjacent models are close.

Reading:
1. *The use decides; the rule makes that explicit.* Forces alone favour the evidence route (hand_pq, F regions 3 % smaller than
   hand's, a resolved paired difference) at the price of energy regions 3.6× hand's; any profile with energy weight picks × 5–30.
   The hand weights are never chosen and are 0.67 in regret (regions 2× too large for energies).
2. *Recommendation for the paper's QoIs (energy differences, plus forces for relaxation):* × 10 (formation and balanced),
   with × 5 the choice robust to the profile (minimax regret). Both are in N for the formation profile. Carry × 5 and × 10 as
   the finalists into CR3 and step 3.
3. *RMSE chooses differently.* Min RMSE_E picks flat, never chosen by size under any profile; min RMSE_F picks hand_pq.
4. *λ makes the cost trade a single number.* The ladder's breakpoints are the exchange rates each step offers: o4d16 → o4d20
   buys 0.19 nats of F size per e-fold of cost, o4d14 → o4d16 0.30. A user who values a factor e in speed below 0.19 nats (17 %
   in semi-axis) should take o4d20.
5. *Limits.* Win rates of 0.5–0.9 show how often a half-sized T_val would choose differently; the tolerance absorbs that, the
   argmin alone would not. The worst-kind risk is a tie-breaker, not a constraint: at α = 0.1 every candidate fails some kind
   (0.22–0.69), so principle 3's hard cap at α + δ would leave nothing. Mondrian calibration by kind is what removes that
   failure, and its group sizes on T_val (extended defects: tens of configurations) are the limit. All of this is one model
   (o4d16) and one T_val; the held-out-type refits (principle 6) are the out-of-distribution check still to do.

## §0 results (2026-11-10)

**Per-type per-atom force coverage** (Gaussian 90 % ellipsoid of each method's own 3×3 covariance,
`results/force_baselines_<m>.npz`; headline / B1 / B2):

| type | atoms | o3d12 | o4d20 |
|---|---|---|---|
| surface_111_pandey | 480 | 0.90 / 0.65 / 0.93 | 1.00 / 0.63 / 0.58 |
| surface_111 | 480 | 0.91 / 0.65 / 0.93 | 0.99 / 0.66 / 0.59 |
| 111adatom (1 cfg) | 146 | 0.57 / 0.56 / 0.99 | 0.64 / 0.31 / 0.82 |
| crack_111 (1 cfg) | 192 | 0.84 / 0.52 / 0.84 | 0.86 / 0.42 / 0.45 |
| all | 17537 | 0.79 / 0.72 / 0.95 | 0.88 / 0.60 / 0.84 |

The headline covers the Pandey and other (111) surfaces; the one adatom configuration (the dangling-bond
environment of §1) is where it under-covers at both sizes. One configuration: a lead for §1, not a result.

**Configuration-level Mahalanobis scores** (`analysis/config_mahal.py`, `results/config_mahal_<m>.npz`). Whitening
every direction is meaningless (near-null directions dominate: median r/d 10³–10⁶), so directions are cut at 99 %
of the trace and the error is compared with the variance per trace band.

- o3d12, all three methods: the dominant directions (first 50 % of the trace) carry far less error than predicted
  (median band ratio 0.09 headline, 0.42 B1, 0.08 B2), the tail 3–9× more. An isotropic term (label noise) fits
  this: ε = 0.01 of the mean variance, ≈ 0.016 eV/Å per component, 10 % of the force RMSE, brings the median
  headline r/D to 1.29 (B1 1.83, B2 3.07).
- o4d20: medians over configurations (dominated by small bulk cells) put B1 closest (band ratios 0.7, 1.03,
  1.16, 1.76), the headline too wide (0.25–0.5), B2 over-concentrated (0.09 → 33).
- Atom-weighted by type, the large cells decide it: in point-defect cells the error in the 99 % directions is
  1.7–2.8× (variance) what the headline predicts, 2.4–6.0× for B1; joint 90 % χ² coverage per configuration
  is 0.02–0.15 for the headline and B1 there, against 0.86–0.92 for bulk phases. POPS at o3d12 is the exception
  (0.95 on defects), at o4d20 not (0.05).

Consequences for §4: a whole-configuration Mahalanobis score on CR3 needs a white nugget term, Σ = λ²(M Mᵀ) + ν I
with ν from validation data, and the joint guarantee is far from met in the defect cells, whose errors are less
concentrated across atoms than every covariance here says. The per-atom 3×3 ellipsoid (the current method) avoids
this. §4 then becomes: does Σ + νI with a conformal scale cover the defect cells and the QoIs?

## Summary

| step | cost | expected |
|---|---|---|
| §0 | ~1 day, existing results | done (2026-11-10): adatom is the one (111) under-coverage; whole-configuration scores need a white nugget |
| §3 barrier | a few days of compute, no new maths | done (2026-11-10): DFT 0.299 eV; o4d20 misses by −0.20 eV (headline z 2.31, B1 4.55, B2 1.99), o3d12 covered via one β-tin cell; delta rows validated by committee; the miss is model-form error on seen environments, and no tested fix (trajectory clusters, difference rows, dipole tensors) removes the discrepancy term |
| §1 (111) | ~1 week; DFT only if needed | done (2026-11-10): DAS within 0.7 σ, order unresolved by either covariance (P(7×7 lowest) 0.05–0.15); spin DFT not triggered |
| §2 ellipsoid | ~1 week, mnf148 | done (2026-11-11): needs a leverage cut; then the most even per-type force coverage, self-calibrated, stable across optima; non-convex, and the lower optimum is not better downstream; QoI widths reproducible once converged (an eigh scoring bug had made them look unstable); slightly sharper per QoI than CR3, worse jointly. Paper: discussion paragraph + SI forces table, optional SI QoI-score sentence (§2e) |
| §4 multivariate | ~1–2 weeks | done (2026-11-11): joint guarantee holds on average, not by kind (surfaces 0.28–0.40); at o4d20 λ_mv covers 16/17 QoIs and is jointly calibrated without the discrepancy term, but scores worse than the headline; too wide at o3d12 |
| §5 CRC | ~1 day | in progress (2026-11-11): CRC makes the per-atom guarantee exact for 2–4 % in q; calibrated size selects o4d20 over o3d12 for every method; among methods the per-atom score picks the ellipsoid shape, the whole-configuration score the parameter-covariance shapes (B1, CR3), as the QoI joint scores; ladder (2026-11-12): B1 on the whole-configuration front at every size, no saturation by o4d20, the worst kind moves surface → extended → point defect; D step 1 (o4d16): energy weight × 5–10 over hand (energy-difference regions 1.8× smaller, F 6–15 % larger), the evidence route best only for forces; a selection rule (declared priorities, one-SE tolerance, λ for cost, minimax regret) in `selection.py`; next CR3 for × 5 / × 10, wL, step 3 |

All four questions have answers. What is left is deciding what goes into the paper revision (branch `revision`, draft PR #1):
candidates are §3 (barrier: a new QoI and the clearest miss), §2e items 2–4 (ellipsoid: discussion paragraph, SI forces table,
theory sentence), §2e's joint scores (d²/n and LS/n beside the hard count; the case for a correlated discrepancy term), §4
(one sentence on λ_mv), §1 (DAS order probabilities, SI).
