"""Shapes of the three parameter distributions behind the propagation (talk figure only; not in the paper):
headline lam_F^2 * CR3, B1 q_b1^2 * posterior, B2 the POPS box at the 50 % leverage cut read as a Gaussian of its
covariance (hi - lo)^2 / 12, as in analysis/force_methods.py.  Each at its own calibration, on the readout block
(POPS has no E0 column), parameters sorted by body order.

  corr    the correlation matrices (scale-free: per-parameter standard deviations span ~8 decades, so a raw
          covariance or Cholesky-factor heatmap shows a handful of pixels).  Saved only for p <= MAX_CORR (o3d12):
          at o4d20's p = 5476 a heatmap washes out at slide resolution.
  stretch the generalised eigenvalues of each covariance in B1's metric, eig(L1^-1 C L1^-T) with C1 = L1 L1^T: how far
          each method stretches (> 1) or squashes (< 1) B1's distribution along its principal directions.  Invariant
          under any linear reparameterisation.  Entries beyond the factor's rank are zero (no width at all).

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.cov_shapes o3d12 o4d20 -> results/cov_shapes_<m>.npz
"""
import json, sys

import numpy as np

MAX_CORR = 1000


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.linalg
    from ..run import RESULTS, RUNS, fit_dir, lin_for
    from .force_methods import pops_factor
    for m in sys.argv[1:]:
        lin, linp = lin_for(m, "cr3"), lin_for(m, "kappa")
        sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
        lam, q = float(sc["lam_cr3"]["F"]), float(sc["q_b1"])
        Lb = pops_factor(fit_dir(m, "pops"))
        p = Lb.shape[0]
        bc = np.asarray(lin.post.body_col)[:p]
        o = np.argsort(bc, kind="stable")
        F = {"b1": linp.factor("posterior")[:p][o] * q, "hl": lin.factor("cr3")[:p][o] * lam, "b2": Lb[o]}
        L1 = np.linalg.cholesky(F["b1"] @ F["b1"].T)
        out = {"bc": bc[o], "p": p, "lam_F": lam, "q_b1": q}
        for k in ("hl", "b2"):
            s = np.linalg.svd(scipy.linalg.solve_triangular(L1, F[k], lower=True), compute_uv=False)
            out[f"stretch_{k}"] = s                        # sqrt of the generalised eigenvalues, descending
            out[f"rank_{k}"] = int((s > s[0] * 1e-8).sum())
        for k, L in F.items():
            C = L @ L.T
            d = np.sqrt(np.diag(C))
            out[f"sd_{k}"] = d
            out[f"mean_abs_corr_{k}"] = float((np.abs(C / d[:, None] / d[None, :]).sum() - p) / (p * (p - 1)))
            if p <= MAX_CORR:
                out[f"corr_{k}"] = (C / d[:, None] / d[None, :]).astype(np.float32)
        np.savez_compressed(RESULTS / f"cov_shapes_{m}.npz", **out)
        print(m, {k: v for k, v in out.items() if np.ndim(v) == 0}, flush=True)


if __name__ == "__main__":
    main()
