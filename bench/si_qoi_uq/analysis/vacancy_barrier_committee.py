"""Vacancy migration barrier: the lam = 1 committee check of the delta sigma (next-steps plan §3; the paper's rule).

Per model, the harness's `vacancy-path-ref` for each of the K = 200 committee samples SIQ_<m>_{cr3,posterior}_<k>
(drawn at scale 1, as the paper's committees), barrier = max_i E_i - E_0 (the protocol's value).  Committee sigma:
headline lam_F sd(cr3), B3 sd(cr3), B1 q_b1 sd(posterior); delta sigma from results/vacancy_barrier.json.  The
headline sigma is the delta one where the two agree to DELTA_TOL (propagate.DELTA_TOL, 10 %), else the committee's.
Also: the committee mean minus the mean model's barrier (nonlinearity), the peak image of each member (a peak that
moves between images makes the protocol's max non-smooth), and the 2.5 / 97.5 % percentiles of the rescaled
deviations.  z = (E_dft - Eb_hat) / sigma.

    SIQ_RUNS=<BLR root> python -m bench.si_qoi_uq.analysis.vacancy_barrier_committee -> results/vacancy_barrier_committee.json
"""
import json

import numpy as np

K = 200


def main():
    from ..harness import ROOT, RUNS, sample_name
    from ..propagate import DELTA_TOL
    from ..run import RESULTS
    vb = json.load(open(RESULTS / "vacancy_barrier.json"))
    E_dft = vb["E_dft"]
    out = {"E_dft": E_dft, "delta_tol": DELTA_TOL}
    for m in ("o3d12", "o4d20"):
        sc = json.load(open(RUNS / "diag" / f"{m}_scales.json"))
        scale = {"cr3": float(sc["lam_cr3"]["F"]), "posterior": float(sc["q_b1"])}
        Eb_hat = vb[m]["cr3"]["Eb_harness"]
        res = {"Eb_hat": Eb_hat}
        for t, meths in (("cr3", ("headline", "B3")), ("posterior", ("B1",))):
            eb, peak, missing = [], [], []
            for k in range(K):
                f = ROOT / "test-runner" / f"model-{sample_name(f'{m}_{t}', k)}-test-vacancy-path-ref-properties.json"
                try:
                    v = json.load(open(f))["vacancy_path"]
                except (OSError, KeyError, ValueError):
                    missing.append(k)
                    continue
                eb.append(max(v) - v[0])
                peak.append(int(np.argmax(v)))
            eb = np.array(eb)
            sd = float(np.std(eb, ddof=1))
            dev = eb - Eb_hat
            for meth in meths:
                s = scale[t] if meth != "B3" else 1.0
                sig_d = vb[m]["cr3"]["sigma"][meth]
                sig_c = s * sd
                agree = abs(sig_d / sig_c - 1) <= DELTA_TOL
                sig = sig_d if agree else sig_c
                res[meth] = dict(sigma_delta=sig_d, sigma_committee=sig_c, ratio=sig_d / sig_c, agree=bool(agree),
                                 sigma=sig, z=(E_dft - Eb_hat) / sig, covered95=bool(abs(E_dft - Eb_hat) <= 1.96 * sig),
                                 p2_5=float(Eb_hat + s * np.percentile(dev, 2.5)),
                                 p97_5=float(Eb_hat + s * np.percentile(dev, 97.5)))
            res[f"committee_{t}"] = dict(K=int(len(eb)), missing=missing, mean_minus_hat=float(eb.mean() - Eb_hat),
                                         sd=sd, peak_image_counts={int(i): int(c) for i, c in
                                                                   zip(*np.unique(peak, return_counts=True))},
                                         n_negative=int((eb <= 1e-6).sum()))
        out[m] = res
        for meth in ("headline", "B1", "B3"):
            r = res[meth]
            print(f"{m} {meth:8s} sigma delta {r['sigma_delta']:.4f} committee {r['sigma_committee']:.4f} "
                  f"(ratio {r['ratio']:.2f}, {'agree' if r['agree'] else 'COMMITTEE'})  z {r['z']:.2f}  "
                  f"95 % covered {r['covered95']}  committee 95 % [{r['p2_5']:.3f}, {r['p97_5']:.3f}]", flush=True)
        for t in ("cr3", "posterior"):
            c = res[f"committee_{t}"]
            print(f"   {t}: K {c['K']} missing {len(c['missing'])}  mean - hat {c['mean_minus_hat']:+.4f}  peaks "
                  f"{c['peak_image_counts']}  barriers <= 0: {c['n_negative']}", flush=True)
    json.dump(out, open(RESULTS / "vacancy_barrier_committee.json", "w"), indent=1)


if __name__ == "__main__":
    main()
