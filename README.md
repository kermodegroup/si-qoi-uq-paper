# Jackknife-conformal uncertainty for linear ACE potentials: calibrated forces and silicon properties compared with Bayesian and POPS baselines

Analysis code, results, figures and paper source for J. R. Kermode, *Jackknife-conformal uncertainty for linear ACE potentials:
calibrated forces and silicon properties compared with Bayesian and POPS baselines* (in preparation).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23986930.svg)](https://doi.org/10.5281/zenodo.23986930)
Release v1.0.0, the version used in the paper, is archived on Zenodo at
[10.5281/zenodo.23986930](https://doi.org/10.5281/zenodo.23986930); all versions:
[10.5281/zenodo.23986929](https://doi.org/10.5281/zenodo.23986929).

The uncertainty methods themselves (the CR3 jackknife shape, Mondrian conformal calibration of per-atom forces, QR ARD, POPS)
are part of [ace-jax](https://github.com/ACEsuit/ace-jax) 0.5.3; this repository holds the study that uses them.

## Layout

- `bench/si_qoi_uq/`: the study. `run.py` runs the pipeline stages (diagnostics, samples, delta, assemble, report);
  `analysis/` holds the follow-up analyses (defect diagnosis and discrepancy term, vacancy, per-atom force calibration,
  descriptor-novelty groups, the accuracy-cost figure); `artifact/paper.py` generates the paper from `results/` only.
  `RECON.md` is the research log: decisions, checks and negative results, in order.
- `bench/si_qoi_uq/results/`, `figures/`: every number and figure in the paper.
- `bench/si_qoi/`: the silicon QoI comparison and scoring (from ace-jax).
- `bench/accuracy_cost/results/metrics.jsonl`, `bench/defect_uq/scoring/`: inputs and helpers taken from ace-jax.
- `docs/dev/specs/`: the original specification of the study.
- `tests/`: unit tests of the observable rows and the delta method.

## Setup

```bash
uv sync
uv run pytest
uv run python bench/si_qoi_uq/artifact/paper.py    # paper from results/ -> bench/si_qoi_uq/artifact/paper/index.html
```

Rerunning the fits, committees and harness needs the GAP-18 silicon database and the Python 3 port of
[libAtoms/silicon-testing-framework](https://github.com/libAtoms/silicon-testing-framework): branch `py3`, commit `f23b1df`,
checked out at `~/silicon-testing-framework`. Its tests run unmodified; `SI_TESTS_ELASTIC_FMAX=1e-3` sets the elastic-constant
relaxation tolerance. Paths to data and run directories are set at the top of the
scripts and by `SIQ_RUNS`.

## Licence

MIT. The ace-jax-derived files keep ace-jax's MIT licence.
