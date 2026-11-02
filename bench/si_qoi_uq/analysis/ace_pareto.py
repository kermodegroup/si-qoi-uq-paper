"""Linear ACE accuracy against cost from the accuracy-cost benchmark, for the paper's first figure: why o3d12 and o4d20.

Per size of the linear ACE ladder (bench/accuracy_cost/results/metrics.jsonl, family "ace"): the test force RMSE, the
QoI error of the benchmark (bench/si_qoi/score.py: geometric mean over the silicon QoIs of |error| / tolerance against
CASTEP), the parameter count, and the LAMMPS cost on a Sulis CPU node and an A100 with the ace-jax kernel
(bench/si_qoi/results/cost_sulis_{cpu,gpu}.json, name ACEffi_<size>; µs per atom-step, float64).

    python -m bench.si_qoi_uq.analysis.ace_pareto      -> results/ace_pareto.json
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "si_qoi"))


def main():
    from compare import qois
    from score import score
    from ..run import RESULTS
    ref = qois("CASTEP")
    cpu = json.load(open(ROOT / "bench/si_qoi/results/cost_sulis_cpu.json"))
    gpu = json.load(open(ROOT / "bench/si_qoi/results/cost_sulis_gpu.json"))
    rows = [json.loads(line) for line in open(ROOT / "bench/accuracy_cost/results/metrics.jsonl")]
    out = []
    for r in rows:
        if r["family"] != "ace":
            continue
        s = r["size"]
        sc = score(f"ACE_{s}", ref)
        out.append({"size": s, "n_params": r["n_params"], "rmse_F": r["rmse_F"], "rmse_E": r["rmse_E"],
                    "qoi": sc["gm_norm"] if sc["n_qoi"] == sc["n_total"] else None, "n_qoi": sc["n_qoi"],
                    "frac_within": sc["frac_within"], "cost_cpu": cpu.get(f"ACEffi_{s}"), "cost_gpu": gpu.get(f"ACEffi_{s}")})
    out.sort(key=lambda p: p["n_params"])
    json.dump(out, open(RESULTS / "ace_pareto.json", "w"), indent=1)
    for p in out:
        print(p)


if __name__ == "__main__":
    main()
