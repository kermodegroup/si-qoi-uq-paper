"""The per-atom force stage on the BLR posterior (ace-jax --ard-prior fixed) against the BLR fits the QoI propagation uses.

Both are posterior.npz files of the same BLR fit: one from the calibrated `--uq ard` stage with the prior held at the
MAP's own (runs/si_qoi_uq_blrcal, fits.sh with SIQ_ARD_PRIOR=fixed), one from blr_cr3.py (runs/si_qoi_uq_blr).  Recorded
per model: the relative difference of the means and of the CR3 covariances D^-1 R R^T D^-1, and the stage's prior and
transfer (ard.json).  Agreement to roundoff means the per-atom and QoI results share one posterior and one shape.

    python -m bench.si_qoi_uq.analysis.blrcal_parity      -> results/blrcal_parity.json
"""
import json, pathlib

import numpy as np

RUNS = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs")
CAL, BLR = RUNS / "si_qoi_uq_blrcal", RUNS / "si_qoi_uq_blr"


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.ard import ARDPosterior
    from ..run import RESULTS
    out = {}
    for m in ("o3d12", "o4d20"):
        a, b = CAL / "fits" / m / "cr3", BLR / "fits" / m / "cr3"
        if not (a / "posterior.npz").exists():
            continue
        pa, pb = ARDPosterior.load(a / "posterior.npz"), ARDPosterior.load(b / "posterior.npz")
        ma, mb = np.asarray(pa.mean), np.asarray(pb.mean)
        La, Lb = np.asarray(pa.dinv)[:, None] * np.asarray(pa.R), np.asarray(pb.dinv)[:, None] * np.asarray(pb.R)
        Ca, Cb = La @ La.T, Lb @ Lb.T
        rep = json.load(open(a / "ard.json"))
        out[m] = {"mean_rel_diff": float(np.linalg.norm(ma - mb) / np.linalg.norm(mb)),
                  "cr3_cov_rel_diff": float(np.linalg.norm(Ca - Cb) / np.linalg.norm(Cb)),
                  "prior": rep.get("prior"), "h": rep["h"], "transfer": rep["transfer"]}
        print(m, {k: v for k, v in out[m].items() if k != "transfer"}, flush=True)
    json.dump(out, open(RESULTS / "blrcal_parity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
