"""Model selection by calibrated size (plan section 5 D, the selection rule), applied to the weight sweep and the ladder.

The rule. Inputs: candidates c fitted on T_fit; views v of the use (force components, energy per atom, energy differences),
with priorities pi_v (sum 1) that the user declares before scoring; alpha. On T_val, S random halves, the same for every
candidate (paired): the CRC threshold on one half, the log size L[c, v, s] (log geometric semi-axis, crc_config.size) and the
per-kind risk on the other.
  1. Aggregate. J[c, s] = sum_v pi_v (L[c, v, s] - L[ref, v, s]) (+ lam log cost[c]): log sizes add, so J is the log of a
     priority-weighted geometric mean of region volumes relative to the reference; lam is the price of a factor e in cost, in
     nats of size (0: size only).
  2. Tolerance (one-SE). b = argmin mean J. N = {c : mean J_c - mean J_b <= tau_c}, tau_c = sd_s(J_c - J_b) / sqrt(2): the
     halves are overlapping resamples of one T_val, so sd_s / sqrt(S) understates the noise; the sd of a half-sample
     difference over sqrt(2) is the noise of the difference on all of T_val (n_B = n / 2). Over all candidates, not only the
     Pareto set: a first Pareto filter on a single view leaves only b and the tolerance nothing to choose from.
  3. Choice in N: the cheapest; then the smallest worst-kind risk g_c (max over kinds and views with pi_v > 0, at alpha: the
     failure the marginal guarantee hides); then the simplest (closest to the reference weights).
  4. (Step 3 of the sweep, not here.) Refit the choice on T, calibrate CRC on the test calibration half, report test risk.
With no single declared use: minimax regret, the candidate whose largest dJ over the profiles is smallest. Reported with it: the Pareto set (views' mean L and cost); the win rate (fraction of splits where c has the smallest J: what a
half-sized T_val would choose by argmin, the noise the tolerance guards against); for the ladder, the lam intervals over which
each candidate has the smallest mean J (the lower envelope of mean L against log cost). Baselines on the same candidates: min
RMSE_F, min RMSE_E (T_val), min single-view size.

    python -m bench.si_qoi_uq.analysis.selection   -> results/selection.json
"""
import json, math

import numpy as np

from ..run import RESULTS

ALPHA = "0.1"
PROFILES = {                                                  # declared uses (priorities), fixed before scoring
    "forces": {"F": 1.0},
    "energies": {"E": 0.5, "dE_type": 0.25, "dE_form": 0.25},
    "balanced": {"F": 0.5, "E": 1 / 6, "dE_type": 1 / 6, "dE_form": 1 / 6},
    "formation": {"dE_form": 0.5, "F": 0.5},
}


def arr(x):
    return np.array([np.nan if y is None else y for y in x], float)


def select(C, pi, lam=0.0, ref=None):
    """C: name -> {"L": {view: (S,)}, "g": {view: worst-kind risk}, "cost": float, "simple": float}. Returns the choice and
    the table. Candidates without every view in pi are skipped."""
    names = [c for c in C if all(v in C[c]["L"] for v in pi)]
    ref = ref if ref in names else names[0]
    J = {c: sum(w * (C[c]["L"][v] - C[ref]["L"][v]) for v, w in pi.items()) + lam * math.log(C[c]["cost"]) for c in names}
    mJ = {c: float(np.mean(J[c])) for c in names}
    mean_L = {c: {v: float(np.mean(C[c]["L"][v])) for v in pi} for c in names}

    def dominated(c):
        return any(all(mean_L[o][v] <= mean_L[c][v] for v in pi) and C[o]["cost"] <= C[c]["cost"]
                   and (any(mean_L[o][v] < mean_L[c][v] for v in pi) or C[o]["cost"] < C[c]["cost"]) for o in names if o != c)
    P = [c for c in names if not dominated(c)]
    b = min(names, key=mJ.get)
    tau = {c: float(np.std(J[c] - J[b], ddof=1) / math.sqrt(2)) if c != b else 0.0 for c in names}
    N = [c for c in names if mJ[c] - mJ[b] <= tau[c]]
    Js = np.array([J[c] for c in names])
    win = np.bincount(np.argmin(Js, axis=0), minlength=len(names)) / Js.shape[1]
    g = {c: max(C[c]["g"][v] for v in pi) for c in names}
    pick = min(N, key=lambda c: (C[c]["cost"], g[c], C[c]["simple"]))
    rows = {c: {"J": mJ[c], "dJ": mJ[c] - mJ[b], "tau": tau[c], "pareto": c in P, "near": c in N, "worst_kind_risk": g[c],
                "win": float(w), "L": mean_L[c]} for c, w in zip(names, win)}
    return pick, b, rows


def envelope(C, pi):
    """lam intervals [lo, hi) on which each candidate has the smallest mean J = a_c + lam log cost_c (lower envelope)."""
    names = [c for c in C if all(v in C[c]["L"] for v in pi)]
    a = {c: sum(w * float(np.mean(C[c]["L"][v])) for v, w in pi.items()) for c in names}
    k = {c: math.log(C[c]["cost"]) for c in names}
    out, lam = [], 0.0
    cur = min(names, key=lambda c: (a[c], k[c]))
    while True:
        nxt = [((a[o] - a[cur]) / (k[cur] - k[o]), o) for o in names if k[o] < k[cur]]   # cheaper ones take over at lam_o
        nxt = [(x, o) for x, o in nxt if x > lam]
        if not nxt:
            out.append({"choice": cur, "lam_lo": lam, "lam_hi": None})
            return out
        x, o = min(nxt)
        out.append({"choice": cur, "lam_lo": lam, "lam_hi": x})
        cur, lam = o, x


def worst(rk):
    m = {k: np.nanmean(arr(x)) for k, x in rk.items() if np.isfinite(arr(x)).any()}
    return float(max(m.values()))


def sweep_candidates(path):
    d = json.load(open(path))
    C = {}
    for c, x in d["candidates"].items():
        f = 1.0
        if c.startswith("Ex"):
            f = float(c[2:])
        elif c.startswith("E/"):
            f = 1 / float(c[2:])
        simple = abs(math.log(f)) + (1.0 if c in ("hand_pq", "noV", "flat") else 0.0)
        C[c] = {"L": {}, "g": {}, "cost": 1.0, "simple": simple, "rmse_E": x["rmse_E_meV_atom"], "rmse_F": x["rmse_F"]}
        for v, by in d["halves"].items():
            h = by[f"joint {float(ALPHA):g}"][c]
            C[c]["L"][v] = arr(h["split"]["size"])
            C[c]["g"][v] = worst(h["split"]["risk_by_kind"])
    return d["model"], C


def ladder_candidates():
    lad = json.load(open(RESULTS / "crc_ladder.json"))
    acc = {p["size"]: p for p in json.load(open(RESULTS / "ace_pareto.json"))}
    C = {}
    for view, loss in (("F", "joint"), ("F_atom", "atom")):
        for name, h in lad["loss"][loss][ALPHA]["halves"].items():
            m, meth = name.split("/")
            c =C.setdefault(name, {"L": {}, "g": {}, "cost": acc[m]["cost_gpu"], "simple": 0.0 if meth == "B1" else 1.0,
                                    "rmse_F": acc[m]["rmse_F"]})
            c["L"][view] = arr(h["split"]["size"])
            c["g"][view] = worst(h["split"]["risk_by_kind"])
    return C


def show(title, pick, b, rows):
    print(f"\n{title}: choice {pick} (best mean {b})")
    for c, r in sorted(rows.items(), key=lambda kv: kv[1]["J"]):
        flag = ("*" if c == pick else " ") + ("P" if r["pareto"] else " ") + ("N" if r["near"] else " ")
        tau = f"{r['tau']:.3f}" if r["tau"] is not None else "  -  "
        print(f"  {flag} {c:18s} J {r['J']:+.3f}  dJ {r['dJ']:+.3f}  tau {tau}  win {r['win']:.2f}  worst-kind {r['worst_kind_risk']:.2f}  "
              + " ".join(f"{v} {x:+.2f}" for v, x in r["L"].items()))


def hi(e):
    return "inf" if e["lam_hi"] is None else f"{e['lam_hi']:.3f}"


def main():
    out = {"alpha": float(ALPHA), "profiles": PROFILES, "sweep": {}, "ladder": {}}
    path = RESULTS / "weight_sweep_o4d16_all.json"
    m, C = sweep_candidates(path)
    base = {"min_rmse_F": min(C, key=lambda c: C[c]["rmse_F"]), "min_rmse_E": min(C, key=lambda c: C[c]["rmse_E"])}
    for v in C["hand"]["L"]:
        base[f"min_size_{v}"] = min(C, key=lambda c: float(np.mean(C[c]["L"][v])))
    out["sweep"] = {"model": m, "baselines": base, "profiles": {}}
    print(f"weight sweep {m}, alpha {ALPHA}; baselines {base}")
    for p, pi in PROFILES.items():
        pick, b, rows = select(C, pi, ref="hand")
        show(f"profile {p} {pi}", pick, b, rows)
        out["sweep"]["profiles"][p] = {"choice": pick, "best": b, "rows": rows}
    reg = {c: {p: out["sweep"]["profiles"][p]["rows"][c]["dJ"] for p in PROFILES} for c in C}   # no single declared use
    mm = min(reg, key=lambda c: max(reg[c].values()))
    out["sweep"]["minimax_regret"] = {"choice": mm, "regret": reg}
    print("\nminimax regret over profiles (max dJ):", mm, {c: round(max(r.values()), 3) for c, r in sorted(reg.items(), key=lambda kv: max(kv[1].values()))})
    CL = ladder_candidates()
    out["ladder"] = {"cost": "A100 float64 CUDA kernel, µs/atom-step", "lam": {}, "envelope": {}}
    for pi_name, pi in (("F", {"F": 1.0}), ("F_atom", {"F_atom": 1.0}), ("F+F_atom", {"F": 0.5, "F_atom": 0.5})):
        env = envelope(CL, pi)
        out["ladder"]["envelope"][pi_name] = env
        print(f"\nladder {pi_name}: lower envelope over lam: "
              + ", ".join(f"{e['choice']} [{e['lam_lo']:.3f}, {hi(e)})" for e in env))
        for lam in (0.0, 0.1, 0.25, 0.5, 1.0):
            pick, b, rows = select(CL, pi, lam, ref="o4d16/B1")
            show(f"ladder {pi_name}, lam {lam}", pick, b, rows)
            out["ladder"]["lam"].setdefault(pi_name, {})[f"{lam:g}"] = {"choice": pick, "best": b, "rows": rows}
    json.dump(out, open(RESULTS / "selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
