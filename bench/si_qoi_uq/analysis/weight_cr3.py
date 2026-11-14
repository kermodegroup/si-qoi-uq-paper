"""The CR3 jackknife shape for weight-sweep candidates (plan section 5 D, finalists), on T_fit.

As blr_cr3.py, but on weight_sweep.py's T_fit with a candidate's weights and MAP noise: the problem is rebuilt from the
benchmark command line with --train T_fit and the candidate's per-type weights written as a weights file; the QR statistics
of those weighted rows give the posterior (the weight sweep's merged factors are the same statistics); the whole-configuration
PRESS jackknife (fit.jackknife.press_scores, exact) and shape_factor give R; CR3 = D^-1 R. Saved for weight_sweep.py's
scoring (it adds "<name>/CR3" beside B1 when the file exists).

    python -m bench.si_qoi_uq.analysis.weight_cr3 o4d16 hand Ex5 Ex10   -> RUNS/wsweep/<m>/cr3_<name>.npz
"""
import json, sys, time

import numpy as np

from .weight_sweep import DATA, base_cmd, candidates, resolve


def one(m, name, res):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax import cli
    from ace_jax.fit.ard import ARDPosterior, ard_gamma, body_order_columns, E0_GROUP
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.jackknife import press_scores, shape_factor
    from ace_jax.fit.objective import posterior_from_qr
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.stats import linear_qr_statistics
    from ..harness import RUNS
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.0f} s] {name}:", *a, flush=True)
    d = RUNS / "wsweep" / m
    types = sorted({str(t) for t in res["candidates"]["hand"]["weights"]})
    w, mode = candidates(types, "all")[name]
    hand = json.load(open(f"{DATA}/weights_acejax.json"))
    wf = d / f"weights_{name.replace('/', '_')}.json"
    json.dump({"default": dict(resolve(hand, "__none__")), **w}, open(wf, "w"), indent=1)
    argv, th_path = base_cmd(m)
    for k, v in (("--train", d / "T_fit.xyz"), ("--weights", wf), ("--out", d / f"fit_{name}")):
        argv[argv.index(k) + 1] = str(v)
    a = cli._parse(argv)
    cfg = cli._fit_config(a)
    data = load_fit_data(cfg, train=a.train, test=None, log=lambda *_: None)
    prob = build_problem(cfg, data).prob
    th = Hypers(**{**json.load(open(th_path)), **res["candidates"][name]["theta"]})
    qs = linear_qr_statistics(prob.model, prob.cfg, data.ds_train)
    log("QR statistics")
    mu, L = posterior_from_qr(th, qs, prob)
    mu, L = np.asarray(mu), np.asarray(L)
    del qs
    gamma = ard_gamma(prob)
    dinv = 1.0 / np.asarray(gamma)
    body_col = body_order_columns(data.meta, prob.cfg)
    groups = tuple(int(g) for g in np.unique(body_col[body_col != E0_GROUP]))
    post = ARDPosterior(mu, dinv[:, None] * L, dinv, 1.0, np.full(len(groups), float(-2 * th.log_sigma_c)), groups, body_col, {})
    sig = np.exp([float(getattr(th, f"log_sigma_{q}")) for q in "EFV"])
    rc, K = row_clusters(data.ds_train, data.train, np.inf)
    G, lev = press_scores(post, prob, data.ds_train, rc, K, sig, mode="exact")
    log(f"PRESS, K {K}, max leverage {float(np.max(lev)):.4f}")
    R = np.asarray(shape_factor(post, G, 1.0))
    np.savez(d / f"cr3_{name.replace('/', '_')}.npz", mu=mu, Lcr3=dinv[:, None] * R, lev=lev, sigma=sig)
    log(f"rank R {R.shape[1]}, saved")


def main(m, *names):
    from ..run import RESULTS
    res = json.load(open(RESULTS / f"weight_sweep_{m}_all.json"))
    for n in names:
        one(m, n, res)


if __name__ == "__main__":
    main(*sys.argv[1:])
