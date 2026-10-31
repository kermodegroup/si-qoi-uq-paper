"""Physicality filter for committee members, fixed before looking at its effect on any QoI.

A member is rejected when any of these holds; none uses a DFT value or a QoI's spread:
  failed      a harness test raised, or returned a non-finite QoI (a relaxation the member cannot complete);
  collapse    some structure the member relaxed has an interatomic distance below D_MIN (silicon bonds are
              2.35 A; no reference phase comes near 2.0 A, and the mean model's shortest relaxed distance is
              reported alongside);
  unstable    its relaxed diamond cell violates the cubic Born conditions C11 - C12 > 0, C11 + 2 C12 > 0, C44 > 0.
The distance threshold's sensitivity (1.8 / 2.2 A) is reported, not chosen from.
"""
import ase.io
import numpy as np
from ase.neighborlist import neighbor_list

from .harness import ROOT, TESTS, sample_name
from .propagate import KEYS, harness_values

D_MIN = 2.0
D_SENS = (1.8, 2.0, 2.2)


def min_distance(run_dirs, r=2.6):
    """Shortest interatomic distance over every recorded relaxed structure (inf when none is closer than r)."""
    d = np.inf
    for rd in run_dirs:
        f = rd / "siq_relaxed.xyz"
        if not f.exists():
            continue
        for at in ase.io.read(f, ":"):
            dd = neighbor_list("d", at, r)
            if len(dd):
                d = min(d, float(dd.min()))
    return d


def member_checks(tag, k):
    name = sample_name(tag, k)
    runner = ROOT / "test-runner"
    done = [(runner / f"model-{name}-test-{t}-properties.json").exists() for t in TESTS]
    Q = harness_values(name)
    failed = (not all(done)) or bool(np.any(~np.isfinite(Q)))
    dmin = min_distance([runner / f"model-{name}-test-{t}" for t in TESTS])
    c11, c12, c44 = (Q[KEYS.index(k_)] for k_ in ("diamond_c11", "diamond_c12", "diamond_c44"))
    unstable = not (c11 - c12 > 0 and c11 + 2 * c12 > 0 and c44 > 0) if np.all(np.isfinite([c11, c12, c44])) else True
    return dict(k=k, failed=failed, n_tests_done=int(sum(done)), d_min=dmin, unstable=bool(unstable), Q=Q)


def filter_committee(tag, K=200, d_min=D_MIN):
    """(accepted mask (K,), per-member records, summary)."""
    recs = [member_checks(tag, k) for k in range(K)]
    acc = lambda r, d: not (r["failed"] or r["unstable"] or r["d_min"] < d)
    mask = np.array([acc(r, d_min) for r in recs])
    summ = {"K": K, "d_min_threshold": d_min, "accepted": int(mask.sum()),
            "rejected_failed": int(sum(r["failed"] for r in recs)),
            "rejected_collapse": int(sum(r["d_min"] < d_min for r in recs)),
            "rejected_unstable": int(sum(r["unstable"] for r in recs)),
            "accepted_at_threshold": {f"{d:g}": int(sum(acc(r, d) for r in recs)) for d in D_SENS}}
    return mask, recs, summ
