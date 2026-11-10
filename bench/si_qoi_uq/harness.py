"""Committee runs of the silicon QoI harness (spec §4b), one coefficient vector per sample.

The harness is the py3 port of libAtoms/silicon-testing-framework in ~/silicon-testing-framework.  It has no
licence, so nothing of it is copied: its `share/` and `tests/` are symlinked into a separate run root and its
test scripts are executed unmodified.  This runner does what `test-runner/run-model-test.py` does for one
(model, test) pair -- chdir into `model-<name>-test-<test>/`, put share/ and the test dir on sys.path, import
`model`, run `test.py`, dump `properties` to `../model-<name>-test-<test>-properties.json` -- but loops over
samples in one process: `model` is one module object whose `calculator` keeps a single ACECalculator and
swaps `calc.model` per sample (a readout swap, so nothing retraces), and each sample re-runs test.py with
fresh globals.  Output files land where `bench/si_qoi/compare.py::qois` reads them (FW = the run root).

    python -m bench.si_qoi_uq.harness setup
    python -m bench.si_qoi_uq.harness run SAMPLES.npz --ks 0-49 [--tests ...]
"""
import argparse, contextlib, json, os, pathlib, runpy, sys, time, types

import numpy as np

FW = pathlib.Path(os.path.expanduser("~/silicon-testing-framework"))
RUNS = pathlib.Path(os.environ.get("SIQ_RUNS", "/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq"))
ROOT = RUNS / "harness"
REPO = pathlib.Path(__file__).resolve().parents[2]
# the 21 QoIs' tests (bulk_hcp_short_and_fat is excluded from the score, so not run); bulk_diamond first:
# phonon_diamond reads its lattice constant from that sample's bulk_diamond properties
TESTS = ("bulk_diamond", "bulk_beta-Sn", "bulk_sh", "bulk_hex_diamond", "bulk_bc8", "bulk_st12", "bulk_hcp",
         "bulk_fcc", "bulk_bcc", "vacancy-energy", "interstitial-energy", "surface-energy-100-relaxed",
         "surface-decohesion-100-unrelaxed", "stacking-fault-surface", "phonon_diamond")
ENV = {"SI_TESTS_ELASTIC_FMAX": "1e-3"}       # the benchmark's converged relaxed-ion C44 (tight_elastic.sh)


def setup(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    for d in ("share", "tests", "test-results"):
        if not (root / d).exists():
            (root / d).symlink_to(FW / d)
    (root / "test-runner").mkdir(exist_ok=True)
    return root


def sample_name(tag, k):
    return f"SIQ_{tag}" if k is None else f"SIQ_{tag}_{k:03d}"


def save_samples(path, tag, model_npz, theta, lin, **extra):
    """theta (K, p) -> npz the runner reads; theta[0] is not special (the mean model is its own tag)."""
    np.savez(path, tag=tag, model_npz=str(model_npz), theta=np.asarray(theta, np.float64), E0_pre=lin.E0_pre,
             n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ, e0_cols=lin.e0_cols, **extra)


def _model_at(base, s, theta):
    import dataclasses
    import jax.numpy as jnp
    from ace_jax.eval.model import fold_readout
    nB, nP, NZ, L = int(s["n_B"]), int(s["n_pair"]), int(s["NZ"]), int(s["n_B"] + s["n_pair"]) * int(s["NZ"])
    E0 = np.asarray(s["E0_pre"]) + (theta[L:L + NZ] if bool(s["e0_cols"]) else 0.0)
    m = dataclasses.replace(base, WB=jnp.asarray(theta[:NZ * nB].reshape(NZ, nB).T),
                            Wpair=jnp.asarray(theta[NZ * nB:L].reshape(NZ, nP).T), E0=jnp.asarray(E0), folded=False)
    return fold_readout(m)


def _strip_calc_write():
    """ASE's extxyz writer also writes the calculator's results; the tests share one calculator across sizes,
    so (as the py3 port's runner does) write calculator-free copies -- these writes are logs only."""
    import ase.io as aio
    if getattr(aio.write, "_siq", False):
        return
    orig = aio.write

    def write(fileobj, images, *a, **k):
        def strip(at):
            c = at.copy()
            c.calc = None
            if "move_mask" in c.arrays and c.constraints:     # the writer makes move_mask from the constraint itself
                del c.arrays["move_mask"]                     # (_move_mask_read restored the array for the DAS test)
            return c
        if isinstance(images, (list, tuple)):
            images = [strip(x) for x in images]
        elif hasattr(images, "get_positions"):
            images = strip(images)
        return orig(fileobj, images, *a, **k)
    write._siq = True
    aio.write = write


def _move_mask_read():
    """ASE >= 3.23's extxyz reader turns a 'move_mask' column into a FixAtoms constraint and drops the array, which the
    (111) DAS test (surface-energy-111-2n+1_x_2n+1-relaxed) reads as atoms.arrays['move_mask'] (written for ASE 3.x
    before that change).  Restore the array from the constraint (1 = free, 0 = fixed); nothing else changes."""
    import ase.io as aio
    from ase.constraints import FixAtoms
    if getattr(aio.read, "_siq", False):
        return
    orig = aio.read

    def fix(at):
        if hasattr(at, "arrays") and "move_mask" not in at.arrays and any(isinstance(c, FixAtoms) for c in at.constraints):
            mm = np.ones(len(at), int)
            for c in at.constraints:
                if isinstance(c, FixAtoms):
                    mm[c.index] = 0
            at.new_array("move_mask", mm)
        return at

    def read(*a, **k):
        out = orig(*a, **k)
        return [fix(x) for x in out] if isinstance(out, list) else fix(out)
    read._siq = True
    aio.read = read


def _record_relaxations(store):
    """Wrap share/utilities.relax_atoms(_cell) to keep every structure they return (in call order), so the
    delta method can read the mean model's relaxed geometries -- the hexagonal interstitial and the
    constant-volume E(V) relaxations (cg_n) write no trajectory.  Patched before any test imports them."""
    import utilities
    if getattr(utilities.relax_atoms, "_siq", False):
        return
    for fn in ("relax_atoms", "relax_atoms_cell"):
        def wrap(*a, _orig=getattr(utilities, fn), _fn=fn, **k):  # noqa: B008  (bind per loop pass)
            out = _orig(*a, **k)
            if store["on"]:
                c = out.copy()
                e = float(out.get_potential_energy())
                c.calc = None
                c.set_constraint()
                c.info.update(siq_fn=_fn, siq_call=len(store["frames"]), siq_energy=e)
                store["frames"].append(c)
            return out
        wrap._siq = True
        setattr(utilities, fn, wrap)


def run(samples_npz, ks, tests=TESTS, root=ROOT, force=False, mean=False, record=False, log=print):
    """Run `tests` for samples ks of samples_npz (mean=True: the npz's theta_mean, tag without index).
    record: keep every member's relaxed structures (siq_relaxed.xyz), as the mean run always does."""
    os.environ.update(ENV)
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.calc.point import ACECalculator
    from ace_jax.eval.io import load
    s = dict(np.load(samples_npz))
    tag = str(s["tag"])
    calc = ACECalculator(str(s["model_npz"]), skin=0.0)
    base, _, _ = load(str(s["model_npz"]), fold=False)
    model = types.ModuleType("model")
    model.calculator, model.no_checkpoint = calc, True
    sys.modules["model"] = model
    for d in (root / "share",):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    _strip_calc_write()
    _move_mask_read()
    store = {"on": mean or record, "frames": []}
    _record_relaxations(store)
    runner = root / "test-runner"
    times = {}
    for k in ([None] if mean else ks):
        theta = np.asarray(s["theta_mean"] if mean else s["theta"][k], np.float64)
        calc.model = _model_at(base, s, theta)
        model.name = name = sample_name(tag, k)
        for t in tests:
            out = runner / f"model-{name}-test-{t}-properties.json"
            if out.exists() and out.stat().st_size and not force:
                continue
            d = runner / f"model-{name}-test-{t}"
            d.mkdir(exist_ok=True)
            calc.reset()
            store["frames"] = []
            t0 = time.time()
            cwd, path0 = os.getcwd(), list(sys.path)
            sys.path.insert(0, str(root / "tests" / t))
            try:
                os.chdir(d)
                with open(runner / f"model-{name}-test-{t}.txt", "w", buffering=1) as fh, \
                        contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                    print(f"Model {name}, Test {t} (bench.si_qoi_uq.harness)")
                    g = runpy.run_path(str(root / "tests" / t / "test.py"), run_name="test")
                props = g["properties"]
                if store["frames"]:
                    import ase.io
                    ase.io.write(d / "siq_relaxed.xyz", store["frames"], format="extxyz")
                with open(out, "w") as fh:
                    json.dump(props, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
            except Exception as e:                        # a failed (sample, test) is recorded, not fatal
                log(f"FAIL {name} {t}: {type(e).__name__}: {e}")
                import traceback
                with open(runner / f"model-{name}-test-{t}.txt", "a") as fh:
                    traceback.print_exc(file=fh)
            finally:
                os.chdir(cwd)
                sys.path[:] = path0
            times[f"{name}/{t}"] = time.time() - t0
        log(f"{name}: done ({sum(v for kk, v in times.items() if kk.startswith(name + '/')):.0f} s)")
    return times


def qois(name, root=ROOT):
    """The bench/si_qoi QoI dict of a harness model in this run root (CASTEP from the framework's results)."""
    sys.path.insert(0, str(REPO / "bench" / "si_qoi"))
    import compare
    compare.FW = str(root)
    if not getattr(compare.ev_min, "_siq", False):         # a sample whose every E(V) point failed: NaN, not a crash
        ev = compare.ev_min

        def ev_min(d):
            k = [k for k in d if k.endswith("E_vs_V")]
            return ev(d) if k and len(d[k[0]]) else (np.nan, np.nan)
        ev_min._siq = True
        compare.ev_min = ev_min
    q = compare.qois(name)
    ref = compare.qois("CASTEP")
    if "_phonon_f" in q and "_phonon_f" in ref and q["_phonon_f"].shape == ref["_phonon_f"].shape:
        q["phonon_f rms vs DFT (THz)"] = float(np.sqrt(np.mean((q["_phonon_f"] - ref["_phonon_f"]) ** 2)))
    return q


def _ks(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("samples")
    r.add_argument("--ks", default=None, help="sample indices, e.g. 0-49,60 (default: all)")
    r.add_argument("--mean", action="store_true", help="run the npz's theta_mean instead of its samples")
    r.add_argument("--record", action="store_true", help="keep each member's relaxed structures")
    r.add_argument("--tests", nargs="*", default=list(TESTS))
    r.add_argument("--force", action="store_true")
    r.add_argument("--times", default=None, help="write per-(sample, test) wall times here (json)")
    a = ap.parse_args()
    if a.cmd == "setup":
        print(setup())
        return
    setup()
    K = len(np.load(a.samples)["theta"])
    times = run(a.samples, _ks(a.ks) if a.ks else range(K), a.tests, force=a.force, mean=a.mean, record=a.record,
                log=lambda m: print(m, flush=True))
    if a.times:
        json.dump(times, open(a.times, "w"), indent=1)


if __name__ == "__main__":
    main()
