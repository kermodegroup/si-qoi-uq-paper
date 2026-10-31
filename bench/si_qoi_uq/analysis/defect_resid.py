"""Per-cell energy residuals (DFT - model, eV) of the defect and bulk training/test cells for one model at the BLR
fit, with the CR3 sigma (lam_F) and the posterior sigma of each cell energy.   python resid.py o4d20|o3d12"""
import json, sys, pathlib
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import chunked_rows_fn
from bench.si_qoi_uq.linear import LinearACE
m = sys.argv[1]
R = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr")
D = "/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost"
lin = LinearACE(R / "fits" / m / "cr3" / "model.npz", R / "fits" / m / "cr3" / "posterior.npz")
lamF = json.load(open(R / "diag" / f"{m}_scales.json"))["lam_cr3"]["F"]
Lc, Lp = jnp.asarray(lin.factor("cr3")), jnp.asarray(lin.factor("posterior"))
TYPES = ("interstitial", "vacancy", "divacancy", "dia")
out = []
for split in ("train", "test"):
    cs = [c for c in load_configs(f"{D}/{split}.xyz", virial_key="virial") if c.config_type in TYPES and c.energy is not None]
    ds = build_dataset(cs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1)
    rows = chunked_rows_fn(lin.model, GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ,
                                               C=int(ds.y_E.shape[1]), e0_cols=lin.e0_cols))
    gc = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        E = rows(bt).E
        for c in np.flatnonzero(np.asarray(bt.cfg_mask)):
            b = E[c]
            cfg = cs[gc]; gc += 1
            out.append(dict(split=split, type=cfg.config_type, n=len(cfg.numbers),
                            resid=float(bt.y_E[c] - b @ jnp.asarray(lin.theta_hat)),
                            sd_cr3=float(lamF * jnp.linalg.norm(b @ Lc)), sd_post=float(jnp.linalg.norm(b @ Lp))))
json.dump(out, open(pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq_blr/defect_check") / f"{m}.json", "w"))
for split in ("train", "test"):
    dia = [o for o in out if o["split"] == split and o["type"] == "dia"]
    eb = np.mean([o["resid"] / o["n"] for o in dia])                     # mean bulk residual per atom
    print(f"--- {m} {split}: bulk (dia) mean residual {1e3 * eb:+.2f} meV/atom over {len(dia)} cells")
    for t in TYPES[:3]:
        s = [o for o in out if o["split"] == split and o["type"] == t]
        if not s:
            continue
        r = np.array([o["resid"] for o in s]); n = np.array([o["n"] for o in s]); sd = np.array([o["sd_cr3"] for o in s])
        f = r - n * eb                                                     # bulk-subtracted: a formation-energy residual
        print(f"{t:12s} N={sorted(set(n))} cells {len(s):3d}: resid mean {r.mean():+.3f} eV (rms {np.sqrt(np.mean(r**2)):.3f}); "
              f"bulk-subtracted mean {f.mean():+.3f} rms {np.sqrt(np.mean(f**2)):.3f} eV; CR3*lamF sigma median {np.median(sd):.3f} eV; "
              f"z(bulk-sub) mean {np.mean(f / sd):+.2f} rms {np.sqrt(np.mean((f / sd) ** 2)):.2f}")
