"""The ARD fits' hold-out-to-full transfer of the per-atom force scale, and what the test split says about it.

The calibration scores come from the hold-out posterior P_fit (fitted on N_fit of the N training configurations) and are
carried to the served posterior by t = (N / N_fit)^beta, beta estimated from a second, smaller hold-out fit and clipped to
[0, 1/2] (ace_jax.fit.ard.run_ard_stage, step 3b).  Recorded per model: that record (ard.json "transfer"), and on the test
atoms of learned_scale (the served posterior's scores) the ratio of their 90 % quantile to the calibration quantile q_one,
i.e. the factor a perfect transfer would have applied on top of t.

    ARD_RUNS=<ARD root> python -m bench.si_qoi_uq.analysis.ard_transfer      -> results/ard_transfer.json
"""
import json

import numpy as np

from .force_baselines import ARD_RUNS


def main():
    from ..run import RESULTS
    out = {}
    for m in ("o3d12", "o4d20"):
        d = json.load(open(ARD_RUNS / "fits" / m / "cr3" / "ard.json"))
        tr = next(v for k, v in _walk(d) if k == "transfer")
        z = np.load(RESULTS / f"learned_scale_{m}_atoms.npz", allow_pickle=True)
        ts, q1, alpha = z["test_s"], float(z["q_one"]), float(z["alpha"])
        out[m] = {**{k: tr[k] for k in ("N", "N_fit", "N_fit2", "lam1", "lam2", "beta_raw", "beta", "factor", "method")},
                  "q_one": q1, "test_q_over_q_one": float(np.quantile(ts, 1 - alpha, method="higher") / q1),
                  "test_coverage": float(np.mean(ts <= q1))}
        print(m, out[m], flush=True)
    json.dump(out, open(RESULTS / "ard_transfer.json", "w"), indent=1)


def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
            yield from _walk(v)


if __name__ == "__main__":
    main()
