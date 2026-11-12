"""wL, the basis construction's angular weight, under CRC (plan section 5 D, step 2).

Inputs: results/weight_sweep_o4d<d>_wL<w>_wl.json (weight_sweep.py with a wL model key: the o4d16 command line with that
basis, T_fit / T_val and the 100 halves as every other sweep, so all rows are paired), candidates hand, Ex5, Ex10 with B1.
Cost: the A100 float64 CUDA-kernel cost (µs / atom-step, the ladder's axis: results/ace_pareto.json), timed for these bases
on one Sulis A100 with the ladder's harness (results/timing/sulis-gpu-cudak-wl-*.jsonl, accuracy_cost.timing lammps rows,
each behind its parity gate; o4d14 and o4d16 re-timed in both jobs as anchors, mean of the rows). Also the many-body basis
count of build_spec (no coupling call; o4d16 at wL 1.5: 1408 of its 1461 parameters), the size proxy.

Per weight setting and profile: the selection rule (selection.select) with cost = basis count at lam 0 (size only), the lower
envelope of mean J against log A100 cost over lam (selection.envelope) and the rule at lam 0.25, 0.5, 1; paired differences
against wL 1.5 at matched basis count (the ~720 pair near o4d14, the ~1460 pair near o4d16).

    python -m bench.si_qoi_uq.analysis.wl_sweep   -> results/wl_sweep.json
"""
import glob, json, re

import numpy as np

from ..run import RESULTS
from .selection import PROFILES, arr, envelope, hi, select, show, worst

SMALL, LARGE = 720, 1460


def a100():
    t = {}
    for f in sorted(glob.glob(str(RESULTS / "timing" / "sulis-gpu-cudak-wl-*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("mode") == "lammps" and r.get("status") == "ok" and r.get("dtype") == "float64" and r.get("ffi"):
                o, d, w = re.fullmatch(r"o(\d)d(\d+)(?:_wL([\d.]+))?", r["size"]).groups()
                t.setdefault(f"o{o}d{d}_wL{float(w or 1.5):g}", []).append(1e6 / r["atom_steps_per_s"])
    return {k: float(np.mean(v)) for k, v in t.items()}


def load():
    from ace_jax.basis.spec import build_spec
    B, T = {}, a100()
    for f in sorted(glob.glob(str(RESULTS / "weight_sweep_o4d*_wL*_wl.json"))):
        d = json.load(open(f))
        g = re.fullmatch(r"o(\d)d(\d+)_wL([\d.]+)", d["model"])
        n_mb = len(build_spec(1, int(g.group(1)), int(g.group(2)), wL=float(g.group(3)))[0])
        key = f"o{g.group(1)}d{g.group(2)}_wL{float(g.group(3)):g}"
        B[d["model"]] = {"wL": float(g.group(3)), "degree": int(g.group(2)), "n_mb": n_mb, "a100": T.get(key), "d": d}
    return B


def main():
    B = load()
    out = {"bases": {m: {k: b[k] for k in ("wL", "degree", "n_mb", "a100")} for m, b in B.items()}, "by_weight": {}}
    for m, b in sorted(B.items(), key=lambda kv: (kv[1]["n_mb"])):
        c = b["d"]["candidates"]
        print(f"{m:14s} n_mb {b['n_mb']:5d}  A100 {b['a100']:.4f} us  " + "  ".join(f"{w}: RMSE E {c[w]['rmse_E_meV_atom']:.2f} F {c[w]['rmse_F']:.4f}" for w in c))
    for w in ("hand", "Ex5", "Ex10"):
        C = {}
        for m, b in B.items():
            h = b["d"]["halves"]
            C[m] = {"L": {v: arr(h[v]["joint 0.1"][w]["split"]["size"]) for v in h},
                    "g": {v: worst(h[v]["joint 0.1"][w]["split"]["risk_by_kind"]) for v in h},
                    "cost": float(b["n_mb"]), "simple": abs(b["wL"] - 1.5)}
        res = {"profiles": {}, "matched": {}}
        res_ref = min(C, key=lambda m: abs(B[m]["n_mb"] - LARGE) + 10 * abs(B[m]["wL"] - 1.5))   # o4d16_wL1.5
        for p, pi in PROFILES.items():
            pick, best, rows = select(C, pi, ref=res_ref)
            show(f"weights {w}, profile {p} (cost = many-body basis count)", pick, best, rows)
            res["profiles"][p] = {"choice": pick, "best": best, "rows": rows}
            CA = {m: {**c, "cost": B[m]["a100"]} for m, c in C.items()}
            env = envelope(CA, pi)
            print(f"weights {w}, profile {p}, A100 cost: lower envelope over lam: "
                  + ", ".join(f"{e['choice']} [{e['lam_lo']:.3f}, {hi(e)})" for e in env))
            res["profiles"][p]["a100"] = {"envelope": env, "lam": {}}
            for lam in (0.25, 0.5, 1.0):
                pk, bb, rr = select(CA, pi, lam, ref=res_ref)
                show(f"weights {w}, profile {p}, A100 cost, lam {lam}", pk, bb, rr)
                res["profiles"][p]["a100"]["lam"][f"{lam:g}"] = {"choice": pk, "best": bb, "rows": rr}
        for tag, target in (("small", SMALL), ("large", LARGE)):
            ms = {}
            for m, b in B.items():                          # per wL the basis nearest the target size
                if abs(b["n_mb"] - target) / target < 0.35 and (b["wL"] not in ms or abs(b["n_mb"] - target) < abs(B[ms[b["wL"]]]["n_mb"] - target)):
                    ms[b["wL"]] = m
            if 1.5 not in ms:
                continue
            ref = ms[1.5]
            res["matched"][tag] = {}
            print(f"\nweights {w}, matched size {tag} (~{target}): paired difference against {ref}, mean (tau)")
            for wl, m in sorted(ms.items()):
                row = {v: (float(np.mean(C[m]["L"][v] - C[ref]["L"][v])), float(np.std(C[m]["L"][v] - C[ref]["L"][v], ddof=1) / np.sqrt(2)))
                       for v in C[m]["L"]}
                res["matched"][tag][m] = {"n_mb": B[m]["n_mb"], "diff": row}
                print(f"  wL {wl:<5g} {m:14s} n_mb {B[m]['n_mb']:5d}  " + "  ".join(f"{v} {a:+.3f} ({t:.3f})" for v, (a, t) in row.items()))
        out["by_weight"][w] = res
    json.dump(out, open(RESULTS / "wl_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
