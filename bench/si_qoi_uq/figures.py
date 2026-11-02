"""The five report figures (spec "Outputs"), PNG and PDF, from results/ (and the RUNS intermediates for the
correlation and ellipse figures).  Palette: the dataviz reference categorical order (validated: CVD and
normal-vision separation pass; two slots sit below 3:1 on the surface, so every method also has its own
marker and a legend)."""
import csv, json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .assemble import Z95  # noqa: E402
from .propagate import COVER_KEYS, KEYS  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"]
MARK = ["o", "s", "D", "^", "v", "P", "X"]
SHORT = {"diamond_bulk_modulus": "B", "diamond_c11": "C11", "diamond_c12": "C12", "diamond_c44": "C44",
         "vacancy_energy": "vacancy", "dumbbell_interstitial_energy": "dumbbell", "hexagonal_interstitial_energy":
         "hex. int.", "tetrahedral_interstitial_energy": "tet. int.", "surface_100 (J/m2)": "(100) surf.",
         "gamma_us glide (J/m2)": "γus glide", "gamma_us shuffle (J/m2)": "γus shuffle",
         "decohesion_max_stress": "decoh. σmax", "phonon_f rms vs DFT (THz)": "phonon rms"}
STRIP = [("CR3-headline", "lam_F", "CR3, λ_F"), ("CR3-headline", "lam_V", "CR3, λ_V"),
         ("B1-headline", "q_b1", "B1 (posterior + q̂)"), ("B2-committee", "lam_1", "B2 (POPS)"),
         ("B2blr-committee", "lam_1", "B2 (POPS, ridge blr)")]


def short(k):
    return SHORT.get(k) or k.replace("dE_", "").replace(" (meV/at)", "")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _rows(results):
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(results / "qoi_uq.csv"))]


def strip(results, out, models):
    rows = _rows(results)
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 3.4 * len(models)), squeeze=False, facecolor=SURFACE)
    for ax, m in zip(axes[:, 0], models):
        _style(ax)
        x = np.arange(len(COVER_KEYS))
        for b, c in ((2, "#f0efec"), (1, "#e4e3df")):
            ax.axhspan(-b, b, color=c, zorder=0)
        for j, (meth, sc, lab) in enumerate(STRIP):
            z = {r["qoi"]: r.get("z") for r in rows if r["model"] == m and r["method"] == meth and r["scale"] == sc}
            zz = np.array([z.get(k, np.nan) if z.get(k) not in ("", None) else np.nan for k in COVER_KEYS], float)
            if np.all(np.isnan(zz)):
                continue
            ax.scatter(x + (j - 2) * 0.15, np.clip(zz, -8, 8), s=26, marker=MARK[j], color=CAT[j], label=lab,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_xticks(x, [short(k) for k in COVER_KEYS], rotation=60, ha="right")
        ax.set_ylabel("z = (Q̂ − Q_DFT) / σ  (clipped ±8)", color=INK, fontsize=8)
        ax.set_title(f"{m}: standardised QoI errors (bands ±1, ±2)", color=INK, fontsize=10, loc="left", pad=22)
        ax.legend(fontsize=7, frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.tight_layout()
    _save(fig, out, "fig1_z_strip")


def coverage_bars(results, out, models):
    cov = [r for r in csv.DictReader(open(results / "coverage.csv"))]
    sel = [("CR3-headline", "lam_F", "CR3 λ_F"), ("CR3-headline", "lam_V", "CR3 λ_V"),
           ("CR3-headline", "lam_pool", "CR3 λ_pool"), ("B3-headline", "lam_1", "B3 (λ=1)"),
           ("B1-headline", "q_b1", "B1"), ("B2-committee", "lam_1", "B2 POPS"),
           ("B2blr-committee", "lam_1", "B2 POPS blr")]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.6), squeeze=False, facecolor=SURFACE)
    for ax, m in zip(axes[0], models):
        _style(ax)
        n = next((r["n"] for r in cov if r["model"] == m and r["method"] == sel[0][0] and r["scale"] == sel[0][1]), "?")
        for j, (lev, nom) in enumerate((("68", 0.68), ("95", 0.95))):
            for i, (meth, sc, _) in enumerate(sel):
                r = next((r for r in cov if r["model"] == m and r["method"] == meth and r["scale"] == sc), None)
                if r is None:
                    continue
                f, lo, hi = float(r[f"frac{lev}"]), float(r[f"ci{lev}_lo"]), float(r[f"ci{lev}_hi"])
                xi = i + (j - 0.5) * 0.36
                ax.bar(xi, f, width=0.34, color=CAT[j], edgecolor=SURFACE, linewidth=2,
                       label=f"{lev}% interval" if i == 0 else None)
                ax.errorbar(xi, f, yerr=[[f - lo], [hi - f]], color=INK, lw=1, capsize=2)
            ax.axhline(nom, color=CAT[j], lw=1, ls="--")
        ax.set_xticks(range(len(sel)), [s[2] for s in sel], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of QoIs covered (95% Clopper–Pearson)", color=INK, fontsize=8)
        ax.set_title(f"{m}: coverage of DFT, n = {n}", color=INK, fontsize=10, loc="left")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig2_coverage")


def correlation(runs, out, m, extra):
    C = extra["C"]["cr3"]
    if C is None:
        return
    ok = [KEYS.index(k) for k in COVER_KEYS if (~np.isnan(C[:, KEYS.index(k)])).sum() > 10]
    X = C[:, ok]
    X = X[~np.isnan(X).any(1)]
    R = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ef8d87", "#8f1d1d"])
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)
    lab = [short(KEYS[i]) for i in ok]
    ax.set_xticks(range(len(ok)), lab, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(ok)), lab, fontsize=7)
    ax.set_title(f"{m}: QoI correlation (CR3 committee, K = {len(X)})", color=INK, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, label="correlation")
    _save(fig, out, "fig3_qoi_correlation")


def _ellipse(ax, mu, S, color, ls, label):
    w, V = np.linalg.eigh(S)
    t = np.linspace(0, 2 * np.pi, 200)
    pts = mu[:, None] + V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(pts[0], pts[1], color=color, ls=ls, lw=2, label=label)


def elastic(out, models, extras):
    pairs = (("diamond_c11", "diamond_c12"), ("diamond_c11", "diamond_c44"))
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 4 * len(models)), squeeze=False, facecolor=SURFACE)
    for r, m in enumerate(models):
        ex = extras.get(m)
        if ex is None:
            continue
        lam = ex["lam"]["lam_F"]
        for c, (a, b) in enumerate(pairs):
            ax = axes[r, c]
            _style(ax)
            ia, ib = KEYS.index(a), KEYS.index(b)
            mu = ex["Qhat"][[ia, ib]]
            for j, (route, pre) in enumerate((("stress route", ""), ("energy route", "energy:"))):
                GL = np.stack([ex["Lc"].T @ ex["G"][pre + a], ex["Lc"].T @ ex["G"][pre + b]])
                _ellipse(ax, mu, lam ** 2 * GL @ GL.T, CAT[j], "-" if j == 0 else "--", f"{route} (1σ, λ_F)")
            ax.scatter(*ex["ref"][[ia, ib]], marker="*", s=120, color=INK, label="DFT (CASTEP)", zorder=4)
            ax.scatter(*mu, marker="o", s=24, color=MUTED, label="Q̂", zorder=4)
            ax.set_xlabel(f"{short(a)} (GPa)", color=INK, fontsize=8)
            ax.set_ylabel(f"{short(b)} (GPa)", color=INK, fontsize=8)
            ax.set_title(f"{m}: {short(a)}–{short(b)}", color=INK, fontsize=10, loc="left")
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig4_elastic_ellipses")


def scales_fig(results, out, models):
    rows = [r for r in csv.DictReader(open(results / "scale_diagnostics.csv")) if r["factor"] == "cr3"]
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 3.2 * len(models)), squeeze=False, facecolor=SURFACE)
    for ax, m in zip(axes[:, 0], models):
        _style(ax)
        labels = sorted({r["label"] for r in rows if r["model"] == m and r["breakdown"] == "config_type"})
        cats = ["all"] + labels
        for j, t in enumerate(("E", "F", "V")):
            v = []
            for lab in cats:
                br = "all" if lab == "all" else "config_type"
                rr = next((r for r in rows if r["model"] == m and r["kind"] == t and r["breakdown"] == br
                           and r["label"] == lab), None)
                v.append(float(rr["rms_z"]) if rr else np.nan)
            ax.bar(np.arange(len(cats)) + (j - 1) * 0.27, v, width=0.25, color=CAT[j], edgecolor=SURFACE,
                   linewidth=1.5, label={"E": "energy", "F": "force", "V": "virial"}[t])
        ax.set_yscale("log")
        ax.set_xticks(range(len(cats)), cats, rotation=60, ha="right")
        ax.set_ylabel("λ = RMS z (CR3 shape)", color=INK, fontsize=8)
        ax.set_title(f"{m}: test-split scale per observable and config_type", color=INK, fontsize=10, loc="left")
        ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, out, "fig5_scale_diagnostics")


def model_compare(results, out, models, meth="CR3-headline", sc="lam_F"):
    """Per QoI, (Q - Q_DFT) / tolerance for each model with its 95 % interval, DFT as a cross at 0: the bigger
    model's intervals are tighter while coverage stays similar.  Symmetric-log axis, linear within +-1 tolerance."""
    rows = _rows(results)
    cov = {(r["model"], r["method"], r["scale"]): r for r in csv.DictReader(open(results / "coverage.csv"))}
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(COVER_KEYS))
    ax.axhspan(-1, 1, color="#ecebe6", zorder=0, label="± 1 tolerance")
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    order = [m for m in ("o4d20", "o3d12") if m in models]
    for j, m in enumerate(order):
        r = {q["qoi"]: q for q in rows if q["model"] == m and q["method"] == meth and q["scale"] == sc}
        e = np.array([(r[k]["mean"] - r[k]["dft"]) / r[k]["tol"] if k in r else np.nan for k in COVER_KEYS])
        h = np.array([Z95 * r[k]["sigma"] / r[k]["tol"] if k in r else np.nan for k in COVER_KEYS])
        c = cov.get((m, meth, sc), {})
        lab = f"{m}: {c.get('k95', '?')}/{c.get('n', '?')} inside 95 %, median σ/tol {float(c.get('median_sigma_over_tol', 'nan')):.2f}"
        ax.errorbar(x + (j - 0.5) * 0.28, e, yerr=h, fmt=MARK[j], ms=6, color=CAT[j], ecolor=CAT[j], elinewidth=2,
                    capsize=3, mec=SURFACE, mew=0.8, label=lab, zorder=3)
    ax.scatter(x, np.zeros(len(x)), marker="x", s=46, color=INK, linewidths=1.6, label="DFT (CASTEP)", zorder=4)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(x, [short(k) for k in COVER_KEYS], rotation=60, ha="right")
    ax.set_ylabel("(Q − Q_DFT) / tolerance  (symlog)", color=INK, fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    _save(fig, out, "fig6_model_compare")


def force_calibration(results, out, nbins=20):
    """Per-atom force uncertainty against the observed error on the test split (analysis/force_calibration.py), per model:
    top, binned on predicted sigma (equal counts): RMS |error| against RMS sigma, Mondrian forces_std and the global
    lam_F sigma, with the per-atom density behind; bottom, binned on |error|: RMS sigma against RMS |error| (binning on the
    observed error regresses towards the mean: the largest-error bins read as overconfident even when calibrated)."""
    ms = [m for m in ("o3d12", "o4d20") if (results / f"force_calibration_{m}.npz").exists()]
    if not ms:
        return
    fig, axes = plt.subplots(2, len(ms), figsize=(5.2 * len(ms), 9.4), squeeze=False, facecolor=SURFACE)

    def binned(x, y, key):
        o = np.argsort(key)
        chunks = np.array_split(o, nbins)
        return (np.array([np.sqrt(np.mean(x[c] ** 2)) for c in chunks]), np.array([np.sqrt(np.mean(y[c] ** 2)) for c in chunks]))
    for j, m in enumerate(ms):
        z = np.load(results / f"force_calibration_{m}.npz", allow_pickle=True)
        e, sm, sg = z["err_norm"], z["forces_std"], z["sd_global"]
        cov = float(np.mean(z["score"] <= z["q"]))
        lo, hi = 0.8 * min(np.percentile(e[e > 0], 1), sm.min()), 1.2 * max(e.max(), sm.max())
        ax = axes[0, j]
        _style(ax)
        ax.hexbin(sm, np.maximum(e, lo), gridsize=60, xscale="log", yscale="log", bins="log", cmap="Greys", mincnt=1,
                  linewidths=0, alpha=0.45, extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)))
        for sd, lab, mk, col in (((sm, "Mondrian forces_std (λ_g)", "o", CAT[0]),
                                                 (sg, "global λ_F", "s", CAT[1]))):
            bx, by = binned(sd, e, sd)
            ax.plot(bx, by, marker=mk, color=col, lw=2, ms=6, mec=SURFACE, label=lab, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="calibrated")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted per-atom σ (eV/Å), bin RMS", color=INK, fontsize=8)
        ax.set_ylabel("observed |ΔF| (eV/Å), bin RMS", color=INK, fontsize=8)
        ax.set_title(f"{m}: binned on predicted σ\n{len(e)} test atoms; conformal coverage {cov:.3f} "
                     f"(nominal {1 - float(z['alpha']):.2f})", color=INK, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax = axes[1, j]
        _style(ax)
        for sd, lab, mk, col in (((sm, "Mondrian forces_std (λ_g)", "o", CAT[0]),
                                                 (sg, "global λ_F", "s", CAT[1]))):
            bx, by = binned(e, sd, e)
            ax.plot(bx, by, marker=mk, color=col, lw=2, ms=6, mec=SURFACE, label=lab, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="σ = |ΔF|")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("observed |ΔF| (eV/Å), bin RMS", color=INK, fontsize=8)
        ax.set_ylabel("predicted per-atom σ (eV/Å), bin RMS", color=INK, fontsize=8)
        ax.set_title(f"{m}: binned on observed error", color=INK, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig7_force_calibration")


def all_figures(results, out, runs, models, extras):
    strip(results, out, models)
    coverage_bars(results, out, models)
    m3 = "o4d20" if "o4d20" in extras else models[0]
    correlation(runs, out, m3, extras[m3])
    elastic(out, [m for m in models if m in extras], extras)
    scales_fig(results, out, models)
    model_compare(results, out, models)
    force_calibration(results, out)
    json.dump({"correlation_model": m3}, open(out / "figures.json", "w"))
