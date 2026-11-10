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
