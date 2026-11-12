"""wL, the basis construction's angular weight, under CRC (plan section 5 D, step 2).

Inputs: results/weight_sweep_o4d<d>_wL<w>_wl.json (weight_sweep.py with a wL model key: the o4d16 command line with that
basis, T_fit / T_val and the 100 halves as every other sweep, so all rows are paired), candidates hand, Ex5, Ex10 with B1.
Size proxy: the many-body basis count of build_spec (no coupling call; o4d16 at wL 1.5: 1408 of its 1461 parameters). The
A100 kernel cost (the ladder's axis) still has to be timed for the bases kept.

Per weight setting and profile: the selection rule (selection.select) over the bases with cost = basis count, and paired
differences against wL 1.5 at matched size (the ~720 pair near o4d14, the ~1460 pair near o4d16).

    python -m bench.si_qoi_uq.analysis.wl_sweep   -> results/wl_sweep.json
"""
import glob, json, re

import numpy as np

from ..run import RESULTS
from .selection import PROFILES, arr, select, show, worst

SMALL, LARGE = 720, 1460


def load():
    from ace_jax.basis.spec import build_spec
    B = {}
    for f in sorted(glob.glob(str(RESULTS / "weight_sweep_o4d*_wL*_wl.json"))):
        d = json.load(open(f))
        g = re.fullmatch(r"o(\d)d(\d+)_wL([\d.]+)", d["model"])
        n_mb = len(build_spec(1, int(g.group(1)), int(g.group(2)), wL=float(g.group(3)))[0])
        B[d["model"]] = {"wL": float(g.group(3)), "degree": int(g.group(2)), "n_mb": n_mb, "d": d}
    return B


def main():
    B = load()
    out = {"bases": {m: {k: b[k] for k in ("wL", "degree", "n_mb")} for m, b in B.items()}, "by_weight": {}}
    for m, b in sorted(B.items(), key=lambda kv: (kv[1]["n_mb"])):
        c = b["d"]["candidates"]
        print(f"{m:14s} n_mb {b['n_mb']:5d}  " + "  ".join(f"{w}: RMSE E {c[w]['rmse_E_meV_atom']:.2f} F {c[w]['rmse_F']:.4f}" for w in c))
    for w in ("hand", "Ex5", "Ex10"):
        C = {}
        for m, b in B.items():
            h = b["d"]["halves"]
            C[m] = {"L": {v: arr(h[v]["joint 0.1"][w]["split"]["size"]) for v in h},
                    "g": {v: worst(h[v]["joint 0.1"][w]["split"]["risk_by_kind"]) for v in h},
                    "cost": float(b["n_mb"]), "simple": abs(b["wL"] - 1.5)}
        res = {"profiles": {}, "matched": {}}
        for p, pi in PROFILES.items():
            pick, best, rows = select(C, pi, ref=min(C, key=lambda m: abs(B[m]["n_mb"] - LARGE) + 10 * abs(B[m]["wL"] - 1.5)))
            show(f"weights {w}, profile {p} (cost = many-body basis count)", pick, best, rows)
            res["profiles"][p] = {"choice": pick, "best": best, "rows": rows}
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
