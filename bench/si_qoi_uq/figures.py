"""The five report figures (spec "Outputs"), PNG and PDF, from results/ (and the RUNS intermediates for the
correlation and ellipse figures).  Palette: the dataviz reference categorical order (validated: CVD and
normal-vision separation pass; two slots sit below 3:1 on the surface, so every method also has its own
marker and a legend)."""
import csv, json, os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .assemble import Z95  # noqa: E402
from .propagate import COVER_KEYS, KEYS  # noqa: E402

SURFACE, INK, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"]
MARK = ["o", "s", "D", "^", "v", "P", "X"]
PANELS = (("CR3-headline", "lam_F", "CR3 × λ_F (headline)"), ("B1-headline", "q_b1", "B1: posterior × conformal q̂"),
          ("B2lev50-committee", "lam_1", "B2: POPS (leverage cut 50 %)"), ("B3-headline", "lam_1", "B3: CR3, λ = 1"))
SHORT = {"diamond_bulk_modulus": "B", "diamond_c11": "C11", "diamond_c12": "C12", "diamond_c44": "C44",
         "vacancy_energy": "vacancy", "dumbbell_interstitial_energy": "dumbbell", "hexagonal_interstitial_energy":
         "hex. int.", "tetrahedral_interstitial_energy": "tet. int.", "surface_100 (J/m2)": "(100) surf.",
         "gamma_us glide (J/m2)": "γus glide", "gamma_us shuffle (J/m2)": "γus shuffle",
         "decohesion_max_stress": "decoh. σmax", "phonon_f rms vs DFT (THz)": "phonon rms"}
STRIP = [("CR3-headline", "lam_F", "CR3, λ_F"), ("CR3-headline", "lam_V", "CR3, λ_V"),
         ("B1-headline", "q_b1", "B1: posterior × q̂"), ("B2-committee", "lam_1", "B2 (POPS)"),
         ("B2blr-committee", "lam_1", "B2 (POPS, ridge blr)")]


# Figure contexts: seaborn's font and line scaling plus the physical width each figure is drawn at, so the text is at
# its intended size wherever the figure is used.  "paper": the text width of a two-column article (7.0 in, fonts
# ~9 pt); "talk": a 16:9 slide (12 in, fonts ~18 pt).  Every font size below is relative ("small", ...), never points.
WIDTH = {"paper": 7.0, "talk": 12.0}
COLUMN = 3.37            # one REVTeX two-column column (in): single-column figures are drawn at it, not shrunk to it
CONTEXT = os.environ.get("SIQ_FIG_CONTEXT", "paper")


def set_context(context="paper"):
    """Apply a seaborn context (paper | talk) and the matching figure width to every figure drawn after it."""
    global CONTEXT
    CONTEXT = context
    sns.set_theme(context=context, style="ticks", rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titlesize": "medium", "axes.labelsize": "small", "legend.fontsize": "x-small",
        "xtick.labelsize": "x-small", "ytick.labelsize": "x-small", "pdf.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})          # has ⁻¹, θ̂, subscript letters


def _size(w, h):
    """(w, h) as drawn originally -> the same aspect at the context's width."""
    W = WIDTH[CONTEXT]
    return (W, W * h / w)


set_context(CONTEXT)


def short(k):
    return SHORT.get(k) or k.replace("dE_", "").replace(" (meV/at)", "")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if CONTEXT == "paper" else 150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _top_legend(fig, ax, ncol):
    """One legend for the whole figure, above every panel (axes legends sat on titles and data at talk sizes):
    the layout is redone with the legend's own measured height reserved at the top."""
    h, lab = ax.get_legend_handles_labels()
    leg = fig.legend(h, lab, loc="upper center", ncol=ncol, frameon=False, fontsize="x-small", bbox_to_anchor=(0.5, 1.0))
    fig.canvas.draw()
    frac = leg.get_window_extent().height / fig.bbox.height
    fig.tight_layout(rect=(0, 0, 1, 1 - frac - 0.01))


def _rows(results):
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(results / "qoi_uq.csv"))]


def strip(results, out, models):
    rows = _rows(results)
    fig, axes = plt.subplots(len(models), 1, figsize=_size(11, 3.4 * len(models)), squeeze=False, facecolor=SURFACE)
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
        ax.set_ylabel("z = (Q̂ − Q_DFT) / σ", color=INK, fontsize="small")
        ax.set_title(f"{m}: standardised QoI errors (bands ±1, ±2; clipped at ±8)", color=INK, fontsize="medium", loc="left")
    _top_legend(fig, axes[0, 0], ncol=5)
    _save(fig, out, "fig1_z_strip")


def coverage_bars(results, out, models):
    cov = [r for r in csv.DictReader(open(results / "coverage.csv"))]
    short_lab = {"CR3-headline": "CR3 × λ_F\n(headline)", "B1-headline": "B1: posterior\n× q̂", "B2lev50-committee": "B2: POPS\n(50 % cut)",
                 "B3-headline": "B3: CR3,\nλ = 1"}
    sel = [(meth, sc, short_lab.get(meth, lab)) for meth, sc, lab in PANELS]        # the same methods as fig6's rows
    fig, axes = plt.subplots(1, len(models), figsize=_size(5.5 * len(models), 4.4), squeeze=False, facecolor=SURFACE)
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
        ax.set_xticks(range(len(sel)), [s[2] for s in sel], fontsize="x-small")
        ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("fraction of QoIs covered", color=INK, fontsize="small")      # bars: 95 % credible intervals
        ax.set_title(f"{m}: coverage of DFT, n = {n}", color=INK, fontsize="medium", loc="left")
        if ax is axes[0, 0]:
            ax.legend(fontsize="x-small", frameon=False, ncol=2, loc="upper left")
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
    col = CONTEXT == "paper"                                   # a single-column figure in the paper
    fig, ax = plt.subplots(figsize=(COLUMN, 0.97 * COLUMN) if col else _size(7.2, 6.2), facecolor=SURFACE)
    tick = "small" if col else "x-small"
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ef8d87", "#8f1d1d"])
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)
    lab = [short(KEYS[i]) for i in ok]
    ax.set_xticks(range(len(ok)), lab, rotation=70, ha="right", rotation_mode="anchor", fontsize=tick)
    ax.set_yticks(range(len(ok)), lab, fontsize=tick)
    ax.tick_params(length=2 if col else 3.5, pad=1.5 if col else 3.5)
    ax.set_title(f"{m}: QoI correlation (CR3, K = {len(X)})" if col else f"{m}: QoI correlation (CR3 committee, K = {len(X)})",
                 color=INK, fontsize="small" if col else "medium", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("correlation", fontsize=tick); cb.ax.tick_params(labelsize=tick)
    _save(fig, out, "fig3_qoi_correlation")


def _ellipse(ax, mu, S, color, ls, label):
    w, V = np.linalg.eigh(S)
    t = np.linspace(0, 2 * np.pi, 200)
    pts = mu[:, None] + V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(pts[0], pts[1], color=color, ls=ls, lw=2, label=label)


def elastic(out, models, extras):
    pairs = (("diamond_c11", "diamond_c12"), ("diamond_c11", "diamond_c44"))
    fig, axes = plt.subplots(len(models), 2, figsize=_size(9, 4 * len(models)), squeeze=False, facecolor=SURFACE)
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
            ax.set_xlabel(f"{short(a)} (GPa)", color=INK, fontsize="small")
            ax.set_ylabel(f"{short(b)} (GPa)", color=INK, fontsize="small")
            ax.set_title(f"{m}: {short(a)}–{short(b)}", color=INK, fontsize="medium", loc="left")
    # common limits on every panel (C11 on x everywhere; C12 and C44 on y): the models' widths compare at a glance
    live = [ax for ax in axes.ravel() if ax.has_data()]
    for get, put in (("get_xlim", "set_xlim"), ("get_ylim", "set_ylim")):
        lo = min(getattr(ax, get)()[0] for ax in live); hi = max(getattr(ax, get)()[1] for ax in live)
        for ax in live:
            getattr(ax, put)(lo, hi)
    _top_legend(fig, axes[0, 0], ncol=4)
    _save(fig, out, "fig4_elastic_ellipses")


def scales_fig(results, out, models):
    rows = [r for r in csv.DictReader(open(results / "scale_diagnostics.csv")) if r["factor"] == "cr3"]
    fig, axes = plt.subplots(len(models), 1, figsize=_size(11, 4.6 * len(models)), squeeze=False, facecolor=SURFACE)
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
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([1, 2, 5, 10, 20]); ax.set_ylim(0.9, 20)
        ax.set_xticks(range(len(cats)), cats, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel("λ = RMS z", color=INK, fontsize="small")
        ax.set_title(f"{m}: test-split scale per observable and configuration type", color=INK, fontsize="medium", loc="left")
    _top_legend(fig, axes[0, 0], ncol=3)
    _save(fig, out, "fig5_scale_diagnostics")


def model_compare(results, out, models, meth="CR3-headline", sc="lam_F", disc=False, panels=PANELS):
    """Per QoI, (Q - Q_DFT) / tolerance for each model with its 95 % interval, DFT as a cross at 0: the bigger
    model's intervals are tighter while coverage stays similar.  Symmetric-log axis, linear within +-1 tolerance.
    One row per method in `panels` (headline first, then the baselines), shared axes -> fig6_model_compare.
    disc=True: the headline alone with the defect discrepancy term (results/discrepancy.csv, headline estimator), the
    intervals without it drawn as thin dark bars on the defect QoIs -> fig8_model_compare_disc."""
    rows = _rows(results)
    cov = {(r["model"], r["method"], r["scale"]): r for r in csv.DictReader(open(results / "coverage.csv"))}
    dsig = {}
    if disc:
        if not (results / "discrepancy.csv").exists():
            return
        for r in csv.DictReader(open(results / "discrepancy.csv")):
            if r["method"] == f"{meth}+disc" and r["scale"] == sc:
                dsig[(r["model"], r["qoi"])] = (float(r["sigma"]), float(r["sigma_disc"]))
        dj = json.load(open(results / "discrepancy.json"))
        for m in models:
            k = dj.get(m, {}).get(f"{meth}+disc/{sc}")
            if k:
                cov[(m, meth, sc)] = {"k95": k["k95"], "n": k["n"], "median_sigma_over_tol": k["median_sigma_over_tol"]}
        panels = ((meth, sc, None),)
    panels = [p_ for p_ in panels if any(q["method"] == p_[0] and q["scale"] == p_[1] for q in rows)]
    fig, axes = plt.subplots(len(panels), 1, figsize=_size(11, 4.6 if len(panels) == 1 else 3.3 * len(panels) + 1.3),
                             sharex=True, sharey=True, squeeze=False, facecolor=SURFACE)
    x = np.arange(len(COVER_KEYS))
    order = [m for m in ("o3d12", "o4d20") if m in models]
    for ip, (meth_, sc_, title) in enumerate(panels):
        ax = axes[ip, 0]
        _style(ax)
        ax.axhspan(-1, 1, color="#ecebe6", zorder=0, label="± 1 tolerance")
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        for j, m in enumerate(order):
            r = {q["qoi"]: q for q in rows if q["model"] == m and q["method"] == meth_ and q["scale"] == sc_}
            e = np.array([(r[k]["mean"] - r[k]["dft"]) / r[k]["tol"] if k in r else np.nan for k in COVER_KEYS])
            h = np.array([Z95 * r[k]["sigma"] / r[k]["tol"] if k in r else np.nan for k in COVER_KEYS])
            if disc:
                h0 = h.copy()
                h = np.array([Z95 * dsig[(m, k)][0] / r[k]["tol"] if (m, k) in dsig and k in r else hh for k, hh in zip(COVER_KEYS, h)])
                sel = np.array([(m, k) in dsig and dsig[(m, k)][1] > 0 for k in COVER_KEYS])
                ax.errorbar((x + (j - 0.5) * 0.28)[sel], e[sel], yerr=h0[sel], fmt="none", ecolor=INK, elinewidth=1.2, capsize=2,
                            zorder=4, label="without the discrepancy term" if j == 0 else None)
            c = cov.get((m, meth_, sc_), {})
            lab = (f"{m}: {int(float(c.get('k95', 'nan')))}/{int(float(c.get('n', 'nan')))}, σ/tol "     # inside 95 %; median
                   f"{float(c.get('median_sigma_over_tol', 'nan')):.2f}") if c else m
            ax.errorbar(x + (j - 0.5) * 0.28, e, yerr=h, fmt=MARK[j], ms=6, color=CAT[j], ecolor=CAT[j], elinewidth=2,
                        capsize=3, mec=SURFACE, mew=0.8, label=lab, zorder=3)
        ax.scatter(x, np.zeros(len(x)), marker="x", s=46, color=INK, linewidths=1.6, label="DFT (CASTEP)", zorder=4)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel("(Q − Q_DFT) / tol", color=INK, fontsize="small")
        if title:
            ax.set_title(title, color=INK, fontsize="medium", loc="left")
        hs, ls_ = ax.get_legend_handles_labels()
        keep = [i for i, l_ in enumerate(ls_) if ip == 0 or l_.startswith(tuple(order))]     # shared entries once, on top
        ax.legend([hs[i] for i in keep], [ls_[i] for i in keep], fontsize="x-small", frameon=False, ncol=2, columnspacing=1.0,
                  handletextpad=0.3,
                  loc="lower left" if ip == 0 else "lower right", bbox_to_anchor=(0, 1.08) if ip == 0 else (1.0, 1.0))
    axes[-1, 0].set_xticks(x, [short(k) for k in COVER_KEYS], rotation=60, ha="right")
    fig.tight_layout()
    _save(fig, out, "fig8_model_compare_disc" if disc else "fig6_model_compare")


def force_calibration(results, out, nbins=20, level=0.9, min_atoms=300):
    """Per-atom force uncertainty against the observed error on the test split, per model.  Top: binned on predicted sigma
    (equal counts), RMS |error| against RMS sigma, for the CR3 shape with one hold-out conformal quantile and with the
    learned continuous scale (ARD fit, ood/learned_scale.py; sigma = radius sqrt(v) / chi_3^-1(level)), and the
    propagation's methods on the BLR fit (analysis/force_baselines.py: CR3 x lam_F, B1, B2, B3), per-atom density behind.
    Bottom: coverage at `level` overall and per GAP-18 configuration type with >= min_atoms test atoms: the conformal
    regions (one quantile, learned), and the Gaussian isotropic region |e| <= sigma sqrt(chi2_3(level) / 3) for the others
    (lam_F is fitted on the test split: in sample)."""
    from scipy.stats import chi, chi2
    ms = [m for m in ("o3d12", "o4d20") if (results / f"learned_scale_{m}_atoms.npz").exists()]
    if not ms:
        return
    kq, k3 = np.sqrt(chi2.ppf(level, 3) / 3), chi(3).ppf(level)
    fig, axes = plt.subplots(2, len(ms), figsize=_size(5.6 * len(ms), 9.6), squeeze=False, facecolor=SURFACE)

    def binned(x, y, key):
        o = np.argsort(key)
        chunks = np.array_split(o, nbins)
        return (np.array([np.sqrt(np.mean(x[c] ** 2)) for c in chunks]), np.array([np.sqrt(np.mean(y[c] ** 2)) for c in chunks]))
    for j, m in enumerate(ms):
        z = np.load(results / f"learned_scale_{m}_atoms.npz", allow_pickle=True)
        e, v, s, t = z["test_e"], z["test_v"], z["test_s"], z["test_t"]
        q1, rl = float(z["q_one"]), z["test_radius_seed0"]
        traces = [("CR3 shape × one hold-out quantile", e, q1 * np.sqrt(v) / k3, t, s <= q1, "o", CAT[0]),
                  ("learned continuous scale", e, rl * np.sqrt(v) / k3, t, s <= rl, "P", CAT[5])]
        pb = results / f"force_baselines_{m}.npz"
        if pb.exists():
            b = np.load(pb, allow_pickle=True)
            for k, lab, mk, col in (("headline", "CR3 × λ_F (in sample on test)", "s", CAT[1]), ("B1", "B1: posterior × q̂", "^", CAT[2]),
                                    ("B2", "B2: POPS (50 % cut)", "D", CAT[3]), ("B3", "B3: CR3, λ = 1", "v", CAT[4])):
                sd = b[f"sd_{k}"]
                traces.append((lab, b["err_norm"], sd, b["config_type"], b["err_norm"] <= kq * sd, mk, col))
        allsd = np.concatenate([tr[2] for tr in traces])
        lo = 0.8 * min(np.percentile(e[e > 0], 1), np.percentile(allsd, 0.5))
        hi = 1.2 * max(e.max(), np.percentile(allsd, 99.9))
        ax = axes[0, j]
        _style(ax)
        sm = traces[0][2]
        ax.hexbin(sm, np.maximum(e, lo), gridsize=60, xscale="log", yscale="log", bins="log", cmap="Greys", mincnt=1,
                  linewidths=0, alpha=0.35, extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)))
        for lab, ee, sd, *_, mk, col in traces:
            bx, by = binned(sd, ee, sd)
            ax.plot(bx, by, marker=mk, color=col, lw=1.8, ms=5, mec=SURFACE, label=lab, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="calibrated")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted per-atom σ (eV/Å), bin RMS", color=INK, fontsize="small")
        ax.set_ylabel("observed |ΔF| (eV/Å), bin RMS", color=INK, fontsize="small")
        ax.set_title(f"{m}: binned on predicted σ", color=INK, fontsize="medium", loc="left")
        ax = axes[1, j]
        _style(ax)
        types, cnt = np.unique(t, return_counts=True)
        types = [tt for tt, c in sorted(zip(types, cnt), key=lambda p: -p[1]) if c >= min_atoms]
        xs = np.arange(len(types) + 1)
        for k_, (lab, _e, _s, tt_, c, mk, col) in enumerate(traces):
            y = [c.mean()] + [c[tt_ == tt].mean() if (tt_ == tt).any() else np.nan for tt in types]
            ax.plot(xs + (k_ - 2.5) * 0.07, y, marker=mk, color=col, lw=1.0, ms=5, mec=SURFACE, label=lab, zorder=3)
        ax.axhline(level, color=INK, lw=1, ls="--", label=f"nominal {level:.0%}")
        ax.axvline(0.5, color=GRID, lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(["all"] + list(types), fontsize="x-small", rotation=50, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"test coverage of the {level:.0%} region", color=INK, fontsize="small")
        ax.set_title(f"{m}: coverage per type", color=INK, fontsize="medium", loc="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    hs, ls_ = axes[0, 0].get_legend_handles_labels()                 # one legend for the figure, below the panels
    fig.legend(hs, ls_, loc="lower center", ncol=4, fontsize="x-small", frameon=False, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out, "fig7_force_calibration")


def ace_pareto(results, out, picked=("o3d12", "o4d20"), f_target=0.0825, q_target=0.5):
    """Linear ACE ladder of the accuracy-cost benchmark (results/ace_pareto.json): test force RMSE and QoI error
    against CPU cost, the models studied here highlighted.  Targets: the benchmark's (force RMSE <= GAP-18's
    0.0825 eV/A on this split, QoI error <= 0.5)."""
    pth = results / "ace_pareto.json"
    if not pth.exists():
        return
    P = json.load(open(pth))
    rows_ = (("cost_cpu", "CPU cost (µs / atom-step)", (0.1, 0.2, 0.5, 1, 2, 5)),      # details in the caption
             ("cost_gpu", "A100 cost (µs / atom-step)", (0.03, 0.05, 0.1, 0.2, 0.5, 1, 2)))
    fig, axes = plt.subplots(len(rows_), 2, figsize=_size(10.5, 4.0 * len(rows_)), squeeze=False, facecolor=SURFACE)
    for ir, (ck, xlab, xt) in enumerate(rows_):
      for ax, key, lab, tgt in ((axes[ir, 0], "rmse_F", "test force RMSE (eV/Å)", f_target),
                                (axes[ir, 1], "qoi", "QoI error (|error| / tol, geom. mean)", q_target)):
        _style(ax)
        pts = [p_ for p_ in P if p_.get(key) is not None and p_.get(ck)]
        x = np.array([p_[ck] for p_ in pts]); y = np.array([p_[key] for p_ in pts])
        ax.plot(x, y, "-", color=CAT[0], lw=2, zorder=2)
        ax.scatter(x, y, s=36, color=CAT[0], edgecolor=SURFACE, linewidth=1.2, zorder=3, label="linear ACE ladder")
        for p_, xi, yi in zip(pts, x, y):
            hot = p_["size"] in picked
            if hot:
                ax.scatter([xi], [yi], s=150, facecolor="none", edgecolor=CAT[1], linewidth=2.2, zorder=4,
                           label="studied here" if p_["size"] == picked[0] else None)
            ax.annotate(p_["size"], (xi, yi), xytext=(-4, -11) if p_["size"] == "o4d12" else (5, 4), textcoords="offset points",
                        fontsize="x-small", ha="right" if p_["size"] == "o4d12" else "left",
                        color=INK, fontweight="bold" if hot else "normal")
        ax.axhline(tgt, color=MUTED, lw=1, ls="--", zorder=1, label="benchmark target")
        ax.set_xscale("log"); ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, NullFormatter
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plain); axis.set_minor_formatter(NullFormatter())
        ax.set_yticks([t for t in ((0.08, 0.1, 0.15, 0.2, 0.25) if key == "rmse_F" else (0.4, 0.5, 0.7, 1, 2, 3))])
        ax.set_xticks(list(xt))
        ax.set_xlabel(xlab, color=INK, fontsize="small")
        ax.set_ylabel(lab, color=INK, fontsize="small")
    axes[0, 0].legend(fontsize="x-small", frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "fig0_ace_pareto")


def _lj_demo(x):
    """The LJ ground truth of the kermodegroup regression demo (apps/regression-demo.py, g(..., 'lj'))."""
    r = np.maximum(x / 5.0 + 1.5, 0.3)
    return np.clip(4.0 * ((1.5 / r) ** 12 - (1.5 / r) ** 6), -1.5, 1.5)


def cr3_schematic(out, seed=0, n=500, noise=0.011, P=6, calib=0.2):
    X_MIN = 0.0                                                      # the smooth LJ part (x < 0: the clipped wall)
    """Schematic of the CR3 construction on the regression demo's LJ scene (kermodegroup/demos, ?scene=lj-cr3; labels
    nearly noiseless, noise sd 0.011, as for DFT training data):
    n points uniform in [-10, 10] plus the two end points, kept in [0, 10], the smooth part of the curve (the demo's
    filter range with "keep inside"; its LJ test domain starts there too; x < 0 is the clipped wall), noise sd `noise`,
    an 80/20 train/calibration split, a degree P - 1 polynomial basis, the evidence-fitted Bayesian ridge, CR3 with one
    point per cluster and lambda the RMS standardised error on the calibration points; the demo's own code path and
    random draws.  (a) data, the full fit, the CR3 band lambda sigma_CR3 and the posterior band; (b) the deletion shifts
    theta - theta_(-k) = A^-1 g_k (asserted against refits) as changes of the predicted curve, projected on the two
    leading principal curves of the CR3 covariance of the prediction (coefficient space is misleading for the
    near-degenerate raw polynomial basis), coloured by the deleted point's x, with both 1-sd ellipses projected likewise;
    (c) the computation as a flow."""
    from matplotlib.patches import Ellipse, FancyBboxPatch
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    np.random.seed(seed)                                             # the demo's get_data and g, in order
    x = np.append(np.random.uniform(-10, 10, size=n), np.linspace(-10, 10, 2))
    x = np.sort(x[(x >= X_MIN) & (x <= 10.0)])                       # the demo's filter [0, 10], "keep inside"
    y = _lj_demo(x) + np.random.normal(scale=noise, size=x.shape)
    Xtr, Xc, ytr, yc = train_test_split(x[:, None], y, test_size=calib, random_state=seed)
    pf = PolynomialFeatures(degree=P - 1, include_bias=True)
    Ptr, Pc = pf.fit_transform(Xtr), pf.transform(Xc)
    br = BayesianRidge(fit_intercept=False).fit(Ptr, ytr)        # alpha, lambda by evidence, as the demo
    al, la = br.alpha_, br.lambda_
    xg = np.linspace(-10, 10, 1000)
    xg = xg[xg >= X_MIN]                                             # the demo's LJ test domain
    Bg = pf.transform(xg[:, None])

    def ridge(Pm, ym):
        """The ridge solution at fixed alpha, lambda by least squares on [sqrt(alpha) Phi; sqrt(lambda) I]: the raw
        degree-9 basis has cond(A) ~ 1e18, so normal equations lose every digit of the coefficients."""
        Aug = np.vstack([np.sqrt(al) * Pm, np.sqrt(la) * np.eye(Pm.shape[1])])
        return np.linalg.lstsq(Aug, np.r_[np.sqrt(al) * ym, np.zeros(Pm.shape[1])], rcond=None)[0]
    th = ridge(Ptr, ytr)
    Ai = np.linalg.inv(al * Ptr.T @ Ptr + la * np.eye(P))
    sa = np.sqrt(al)
    Psi, rho = sa * Ptr, sa * (ytr - Ptr @ th)
    h = np.einsum("ij,jk,ik->i", Psi, Ai, Psi)                       # one point per cluster: H_kk is a scalar
    G = Psi * (rho / (1 - h))[:, None]
    shifts = G @ Ai
    for k in np.linspace(0, len(ytr) - 1, 5).astype(int):           # exact: refit without point k, same alpha, lambda
        m = np.arange(len(ytr)) != k
        d_ = np.abs(Bg @ (th - ridge(Ptr[m], ytr[m])) - Bg @ shifts[k]).max()
        assert d_ < 1e-3 * np.abs(Bg @ shifts[k]).max(), f"deletion shift != refit (point {k}: {d_:.2e})"
    Gc = G - G.mean(0)
    Scr3 = Ai @ (Gc.T @ Gc) @ Ai
    sd_c = lambda B: np.sqrt(np.einsum("ij,jk,ik->i", B, Scr3, B))
    lam = float(np.sqrt(np.mean(((yc - Pc @ th) / sd_c(Pc)) ** 2)))
    fit = Bg @ th
    sd_post = np.sqrt(np.einsum("ij,jk,ik->i", Bg, Ai, Bg))
    fig = plt.figure(figsize=_size(10.0, 10.0), facecolor=SURFACE)
    axa = fig.add_axes([0.08, 0.60, 0.50, 0.36]); axb = fig.add_axes([0.70, 0.60, 0.22, 0.36])
    axc = fig.add_axes([0.0, 0.0, 1.0, 0.50]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    _style(axa); _style(axb)
    axa.fill_between(xg, fit - lam * sd_c(Bg), fit + lam * sd_c(Bg), color=CAT[0], alpha=0.20, lw=0,
                     label=f"CR3 band, λ σ (λ = {lam:.1f} from calibration)")
    axa.fill_between(xg, fit - sd_post, fit + sd_post, color=CAT[1], alpha=0.6, lw=0, label="posterior band, σ²A⁻¹")
    axa.scatter(Xtr[:, 0], ytr, s=5, color=MUTED, alpha=0.6, lw=0, label="training points", zorder=2)
    axa.scatter(Xc[:, 0], yc, s=7, color=CAT[2], alpha=0.8, lw=0, label="calibration points", zorder=2)
    axa.plot(xg, _lj_demo(xg), color=INK, lw=1.2, ls="--", label="truth (LJ)")
    axa.plot(xg, fit, color=INK, lw=2, label="fit to all data, θ̂")
    axa.set_xlim(X_MIN, 10); axa.set_ylim(-1.25, 0.6)
    axa.set_xlabel("x", color=INK, fontsize="small"); axa.set_ylabel("y", color=INK, fontsize="small")
    axa.set_title(f"(a) LJ demo scene: a degree-{P - 1} polynomial fit", color=INK, fontsize="medium", loc="left")
    axa.legend(fontsize="x-small", frameon=False, loc="lower right", ncol=1)   # the empty corner below the curve
    # function space: coordinates along the two leading principal curves of the CR3 covariance of the prediction
    # (RMS over the grid, units of y); coefficient space is the wrong view for a near-degenerate basis
    ng = len(xg)
    Cf = Bg @ Scr3 @ Bg.T / ng
    w, U = np.linalg.eigh(Cf)
    U2 = U[:, ::-1][:, :2]                                          # unit grid vectors
    proj = lambda S: U2.T @ (Bg @ S @ Bg.T / ng) @ U2
    pts = -(Bg @ shifts.T).T @ U2 / np.sqrt(ng)                     # prediction of theta_(-k) - theta, projected
    sc = axb.scatter(pts[:, 0], pts[:, 1], c=Xtr[:, 0], cmap="viridis", s=12, lw=0, zorder=3)
    for S, col, lab, a_ in ((proj(Scr3), CAT[0], "CR3, 1 sd (λ = 1)", 0.22), (proj(Ai), CAT[1], "posterior, 1 sd", 0.6)):
        ww, V = np.linalg.eigh(S)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        axb.add_patch(Ellipse((0, 0), 2 * np.sqrt(ww[1]), 2 * np.sqrt(ww[0]), angle=ang, facecolor=col, alpha=a_, edgecolor=col,
                              lw=1.2, label=lab, zorder=2))
    axb.scatter([0], [0], s=40, color=INK, zorder=4, label="θ̂")
    lim = 1.15 * max(np.abs(pts).max(), np.sqrt(np.linalg.eigvalsh(proj(Scr3)).max()))
    axb.set_xlim(-lim, lim); axb.set_ylim(-lim, lim)
    axb.set_xlabel("1st principal curve of Σ_CR3", color=INK, fontsize="small")
    axb.set_ylabel("second principal curve", color=INK, fontsize="small")
    axb.set_title("(b) Leave-one-out changes", color=INK, fontsize="medium", loc="left")
    axb.legend(fontsize="x-small", frameon=False, loc="lower left")
    axb.text(0.03, 0.97, f"Σ_CR3 sums {len(ytr)} outer\nproducts: its ellipse is\n~√{len(ytr)} × the dot spread",
             transform=axb.transAxes, fontsize="x-small", color=MUTED, va="top")
    cb = fig.colorbar(sc, cax=fig.add_axes([0.935, 0.60, 0.012, 0.36]))
    cb.set_label("x of the deleted point", color=INK, fontsize="small"); cb.ax.tick_params(labelsize="x-small")
    steps = [("1  Fit and residuals", [r"$A=\Psi^T\Psi+\Lambda$", r"$\rho_k=\tilde y_k-\Psi_k\hat\theta$"],
              "one fit to all configurations;\nrows Ψₖ of configuration k\n(energy, forces, virial)\nmove together"),
             ("2  Leverage of k", [r"$H_{kk}=\Psi_kA^{-1}\Psi_k^T$"],
              "how strongly the fit is\npulled towards k: eigenvalues\nin [0, 1); near 1 means k\nis nearly interpolated"),
             ("3  Leave-k-out residuals", [r"$(I-H_{kk})^{-1}\rho_k$"],
              "the residuals k would have\nif it were left out of the fit\n(PRESS): no refit needed"),
             ("4  Pull and deletion shift", [r"$\tilde g_k=\Psi_k^T(I-H_{kk})^{-1}\rho_k$", r"$\hat\theta-\hat\theta_{(-k)}=A^{-1}\tilde g_k$"],
              "the pull of k on θ as an\nunseen configuration would\nexert it; the exact shift of\nthe fit (panel b)"),
             ("5  Spread of the shifts", [r"$\tilde M=\sum_k(\tilde g_k-\bar g)(\tilde g_k-\bar g)^T$", r"$\Sigma_{\rm CR3}=A^{-1}\tilde MA^{-1}=LL^T$"],
              "how much the fit depends\non which configurations were\nin it: the shape, wide where\nthe model is misspecified"),
             ("6  Use, with one scale λ", [r"$\sigma(b)=\lambda\,\|L^Tb\|$"],
              "λ from held-out data; per\natom the 3×3 force block V;\nQoIs by the delta method\nand committees in θ")]
    W, Hh, gx, rows_y = 0.315, 0.40, 0.0125, (0.47, 0.02)          # three boxes per row, two rows
    for i, (title, eqs, read) in enumerate(steps):
        r_, c_ = divmod(i, 3)
        x0, y0 = 0.012 + c_ * (W + gx), rows_y[r_]
        axc.add_patch(FancyBboxPatch((x0, y0), W, Hh, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor="#f1f0ec",
                                     edgecolor=GRID, lw=1.0, transform=axc.transAxes))
        axc.text(x0 + 0.012, y0 + Hh - 0.03, title, fontsize="small", color=INK, fontweight="bold", va="top", transform=axc.transAxes)
        for j_, eq in enumerate(eqs):
            yy = y0 + Hh * (0.60 if len(eqs) == 1 else 0.68 - 0.17 * j_)
            axc.text(x0 + W / 2, yy, eq, fontsize="small", color=INK, ha="center", va="center", transform=axc.transAxes)
        axc.text(x0 + 0.012, y0 + 0.025, read, fontsize="x-small", color=MUTED, va="bottom",
                 transform=axc.transAxes, linespacing=1.3)
    axc.text(0.012, 0.95, "(c) The CR3 computation: one fit, then closed-form leave-one-configuration-out quantities",
             fontsize="medium", color=INK, transform=axc.transAxes)
    _save(fig, out, "fig9_cr3_schematic")
    return {"lambda": lam, "n_train": int(len(ytr)), "n_calib": int(len(yc))}


def all_figures(results, out, runs, models, extras, contexts=("paper", "talk")):
    """Every figure in each context: the paper set in `out` (used by the HTML page and the LaTeX source), the talk set
    in out / "talk"."""
    for ctx in contexts:
        set_context(ctx)
        o = out if ctx == "paper" else out / ctx
        ace_pareto(results, o)
        cr3_schematic(o)
        strip(results, o, models)
        coverage_bars(results, o, models)
        m3 = "o4d20" if "o4d20" in extras else models[0]
        correlation(runs, o, m3, extras[m3])
        elastic(o, [m for m in models if m in extras], extras)
        scales_fig(results, o, models)
        model_compare(results, o, models)
        model_compare(results, o, models, disc=True)
        force_calibration(results, o)
        json.dump({"correlation_model": m3}, open(o / "figures.json", "w"))

    set_context("paper")