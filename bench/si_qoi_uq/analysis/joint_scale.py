"""Bayesian joint calibration of a QoI covariance: the evidence for one overall scale s on it (revision, Section 3.5; replaces
the chi^2 band on d^2/n).

The errors d of n QoIs, given a method's joint covariance Sigma (tolerance units, with its floor), are modelled as
d ~ N(0, s^2 Sigma). s is log-uniform on [lo, hi] (default [0.1, 10]: a factor of ten either way), as the paper's scale
evidence (Section 3.3, Table 2b) treats its scales but on a narrower range (Section 3.3 uses [0.1, 1000] for the unscaled CR3
shape; every covariance here is already scaled), and integrated out. The log BF cap rises with the prior's log width:
about +2.4 for n = 18 here, + log 2 on [0.01, 100]. Only d^2 = d' Sigma^-1 d and n enter: log det Sigma
and n log 2 pi cancel in the Bayes factor and the posterior. Returned:
  s_med, s_lo, s_hi  posterior median and 95 % equal-tailed credible interval of s
  log_bf             log Bayes factor of "calibrated" (s = 1) against "free scale" (s integrated out); > 0 favours s = 1,
                     read on the Kass-Raftery scale as the paper's other Bayes factors
  log_bf_wide        the same with s log-uniform on [0.01, 100] (prior sensitivity: the Occam factor grows by log 2)
"""
import numpy as np


def _loglik(d2, n, log_s):
    return -0.5 * d2 * np.exp(-2 * log_s) - n * log_s          # log N(d; 0, s^2 Sigma) up to the s-free constant


def _lse(a):
    m = a.max()
    return m + np.log(np.sum(np.exp(a - m)))


def joint_scale(d2, n, lo=0.1, hi=10.0, wide=(0.01, 100.0), m=40001):
    def ev(lo_, hi_):
        g = np.linspace(np.log(lo_), np.log(hi_), m)
        ll = _loglik(d2, n, g)
        return g, ll, _lse(ll) + np.log((g[1] - g[0]) / (g[-1] - g[0]))   # log of the prior average of the likelihood
    g, ll, lz = ev(lo, hi)
    c = np.cumsum(np.exp(ll - ll.max()))
    c /= c[-1]
    q = lambda p: float(np.exp(np.interp(p, c, g)))
    l1 = float(_loglik(d2, n, 0.0))
    return {"s_med": q(0.5), "s_lo": q(0.025), "s_hi": q(0.975), "log_bf": l1 - float(lz),
            "log_bf_wide": l1 - float(ev(*wide)[2]), "prior": [lo, hi], "prior_wide": list(wide)}
