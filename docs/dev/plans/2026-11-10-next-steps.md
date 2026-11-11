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
  15 points of the in-distribution coverage are not fitted. (2) No rank-32 fit converged (4 × 500 iterations each). (3) With the
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
at σ/tol 0.28, but is not converged and its unfitted-width share is unknown: needs a converged rerun before any paper claim.

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
| §0 | ~1 day, existing results | decides how far §1 and §4 go |
| §3 barrier | a few days of compute, no new maths | strongest result |
| §1 (111) | ~1 week; DFT only if needed | "physics absent from the labels needs data"; probabilities that the order is right |
| §2 ellipsoid | ~1 week, mnf148 | ≈ hypercube at our N/P; with a scale, a measure of shape |
| §4 multivariate | ~1–2 weeks | joint guarantee; a test of correlations |
