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
ALPHA_OTHER = 0.6        # fill opacity of every bar but the best on a proper score (fig10); outlines stay opaque
PANELS = (("CR3-headline", "lam_F", "CR3 × λ$_F$ (headline)"), ("B1-headline", "q_b1", "B1: posterior × conformal q̂"),
          ("B2lev50-committee", "lam_1", "B2: POPS (leverage cut 50 %)"), ("B3-headline", "lam_1", "B3: CR3, λ = 1"))
SHORT = {"diamond_bulk_modulus": "B", "diamond_c11": "C$_{11}$", "diamond_c12": "C$_{12}$", "diamond_c44": "C$_{44}$",
         "vacancy_energy": "vacancy", "dumbbell_interstitial_energy": "dumbbell", "hexagonal_interstitial_energy":
         "hex. int.", "tetrahedral_interstitial_energy": "tet. int.", "surface_100 (J/m2)": "(100) surf.",
         "gamma_us glide (J/m2)": "γ$_\\mathrm{us}$ glide", "gamma_us shuffle (J/m2)": "γ$_\\mathrm{us}$ shuffle",
         "decohesion_max_stress": "decoh. σ$_\\max$", "phonon_f rms vs DFT (THz)": "phonon rms"}
STRIP = [("CR3-headline", "lam_F", "CR3, λ$_F$"), ("CR3-headline", "lam_V", "CR3, λ$_V$"),
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
        ax.set_ylabel("z = (Q̂ − Q$_\\mathrm{DFT}$) / σ", color=INK, fontsize="small")
        ax.set_title(f"{m}: standardised QoI errors (bands ±1, ±2; clipped at ±8)", color=INK, fontsize="medium", loc="left")
    _top_legend(fig, axes[0, 0], ncol=5)
    _save(fig, out, "fig1_z_strip")


def coverage_bars(results, out, models):
    cov = [r for r in csv.DictReader(open(results / "coverage.csv"))]
    short_lab = {"CR3-headline": "CR3 × λ$_F$\n(headline)", "B1-headline": "B1: posterior\n× q̂", "B2lev50-committee": "B2: POPS\n(50 % cut)",
                 "B3-headline": "B3: CR3,\nλ = 1"}
    sel = [(meth, sc, short_lab.get(meth, lab)) for meth, sc, lab in PANELS]        # the same methods as fig6's rows
    fig, axes = plt.subplots(1, len(models), figsize=_size(5.5 * len(models), 4.4), squeeze=False, facecolor=SURFACE)
    for ax, m in zip(axes[0], models):
        _style(ax)
        n = next((r["n"] for r in cov if r["model"] == m and r["method"] == sel[0][0] and r["scale"] == sel[0][1]), "?")
        for j, (lev, nom) in enumerate((("68", 0.68), ("95", 0.95))):
            rs = [next((r for r in cov if r["model"] == m and r["method"] == meth and r["scale"] == sc), None) for meth, sc, _ in sel]
            for i, r in enumerate(rs):
                if r is None:
                    continue
                f, lo, hi = float(r[f"frac{lev}"]), float(r[f"ci{lev}_lo"]), float(r[f"ci{lev}_hi"])
                xi = i + (j - 0.5) * 0.36
                ax.bar(xi, f, width=0.34, facecolor=CAT[j], edgecolor=INK, linewidth=0.8)   # unranked: the intervals overlap
                ax.errorbar(xi, f, yerr=[[f - lo], [hi - f]], color=INK, lw=1, capsize=2)
            ax.bar(np.nan, np.nan, facecolor=CAT[j], edgecolor=INK, linewidth=0.8, label=f"{lev}% interval")
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
                _ellipse(ax, mu, lam ** 2 * GL @ GL.T, CAT[j], "-" if j == 0 else "--", f"{route} (1σ, λ$_F$)")
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


SUMMARY_METH = (("CR3-headline", "lam_F", "CR3/lam_F", "headline", "CR3 × λ$_F$"), ("B1-headline", "q_b1", "B1/q_b1", "B1", "B1: posterior × $\\hat{q}$"),
                ("B2lev50-committee", "lam_1", "B2lev50/lam_1", "B2", "B2: POPS"), ("B3-headline", "lam_1", "B3/lam_1", "B3", "B3: CR3, λ = 1"))


def summary_fig(results, out, models):
    """The four methods of fig6 (headline, B1, B2, B3) on the paper's key metrics, laid out as Shanks et al. (2026) Fig. 1:
    one row of panels per model, one column per metric, methods as horizontal bars sharing the row labels, one hue per
    metric family (QoIs, per-atom forces; neither is a model colour of fig6).  QoIs: CRPS/tol, 95 % coverage, log
    evidence of the DFT values (committee covariance, eps 1e-6, no discrepancy term); forces: 90 % ellipsoid coverage on
    the test split, its lowest value over the configuration types of fig7 (at least 300 test atoms), and on the held-out
    cracks and dislocations, mean Gaussian log score per test atom
    (results/force_methods_<m>.json).  Bars beyond a panel's range are cut at its edge and labelled with the value;
    a missing value is marked n/a -> fig10_summary."""
    cov = {(r["model"], r["method"], r["scale"]): r for r in csv.DictReader(open(results / "coverage.csv"))}
    ev = {(r["model"], r["method"]): float(r["log_p_eps1e-06"]) for r in csv.DictReader(open(results / "evidence.csv"))
          if r["kind"] == "committee"}
    fm = {m: json.load(open(results / f"force_methods_{m}.json")) for m in models if (results / f"force_methods_{m}.json").exists()}
    getf = lambda m, st, k, f: fm.get(m, {}).get(st, {}).get("all", {}).get(k, {}).get(f, np.nan)
    from scipy.stats import chi
    k3 = chi(3).ppf(0.9)

    def worst_type(m, k, min_atoms=300):
        """Lowest test coverage of the 90 % ellipsoid over the configuration types of fig7's lower panels."""
        f = results / f"force_baselines_{m}.npz"
        if not f.exists():
            return np.nan
        b = np.load(f, allow_pickle=True)
        t = b["config_type"]
        ty, c = np.unique(t, return_counts=True)
        return 100 * min(np.mean(b[f"d_{k}"][t == x] <= k3) for x in ty[c >= min_atoms])
    QC, FC = CAT[2], CAT[5]
    cols = (("CRPS / tol\nlower is better", lambda m, c: float(cov[(m, c[0], c[1])]["crps_tol"]), None, None, QC),
            ("coverage / %\nnominal 95", lambda m, c: 100 * float(cov[(m, c[0], c[1])]["k95"]) / float(cov[(m, c[0], c[1])]["n"]), 95, (0, 100), QC),
            ("log evidence\nhigher is better", lambda m, c: ev.get((m, c[2]), np.nan), None, (-75, 0), QC),
            ("coverage / %\ntest, nom. 90", lambda m, c: 100 * getf(m, "id", c[3], "coverage"), 90, (0, 100), FC),
            ("worst type / %\ntest, nom. 90", lambda m, c: worst_type(m, c[3]), 90, (0, 100), FC),
            ("coverage / %\nheld out, nom. 90", lambda m, c: 100 * getf(m, "ood", c[3], "coverage"), 90, (0, 100), FC),
            ("log score per atom\nhigher is better", lambda m, c: getf(m, "id", c[3], "log_score"), None, (0, 5), FC))
    fig, axes = plt.subplots(len(models), len(cols), figsize=_size(11, 2.1 * len(models) + 0.5), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    y = np.arange(len(SUMMARY_METH))
    for i, m in enumerate(models):
        for k, (title, fn, nominal, lim, colour) in enumerate(cols):
            ax = axes[i, k]
            ax.set_facecolor(SURFACE)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
            v = np.array([fn(m, c) for c in SUMMARY_METH], float)
            lo = lim[0] if lim else 0.0
            # proper scores only (CRPS, log evidence, log score): best at full colour, the rest faded.  Coverage panels are
            # not ranked (nearest nominal is not a proper score, and the POPS box has no target level): all faded
            score = np.full(len(v), np.nan) if nominal is not None else (-v if k == 0 else v)
            top = np.nanmax(np.where(np.isfinite(score), score, -np.inf)) if np.isfinite(score).any() else np.nan
            best = set(np.flatnonzero(np.isclose(score, top))) if np.isfinite(top) else set()   # ties all marked
            for yi, vi in enumerate(v):
                ax.barh(yi, max(vi, lo) if np.isfinite(vi) else 0, height=0.78, edgecolor=INK, linewidth=0.8, zorder=3,
                        facecolor=matplotlib.colors.to_rgba(colour, 1.0 if yi in best else ALPHA_OTHER))
            for yi, vi in zip(y, v):
                if not np.isfinite(vi):
                    ax.text(0.02, yi, "n/a", transform=ax.get_yaxis_transform(), va="center", fontsize="xx-small", color=MUTED)
                elif vi < lo:                                           # cut at the panel edge
                    ax.text(lo, yi, f" {vi:.0f}", va="center", ha="left", fontsize="xx-small", zorder=6,
                            color=SURFACE if lo < 0 else INK)               # on the cut bar, or beside an empty one
            ax.axvline(0, color=INK, lw=0.8, zorder=5)
            if nominal is not None:
                ax.axvline(nominal, color=MUTED, lw=1, ls="--", zorder=4)
            if lim:
                ax.set_xlim(*lim)
            ax.set_ylim(len(y) - 0.5, -0.5)                             # headline on top
            ax.set_title(title, color=INK, fontsize="x-small")
            ax.tick_params(axis="both", labelsize="xx-small", colors=MUTED, labelcolor=INK)
        axes[i, 0].set_yticks(y, [c[4] for c in SUMMARY_METH])
        axes[i, 0].annotate(m, (0, 1), xycoords="axes fraction", xytext=(-60, 14), textcoords="offset points",
                            fontsize="small", color=INK, ha="left", va="bottom")
    fig.tight_layout(h_pad=1.6, rect=(0, 0, 1, 0.93))
    for c0, c1, lab in ((0, 2, "QoIs (20 DFT values)"), (3, len(cols) - 1, "Per-atom forces")):     # one header per metric family
        b0, b1 = axes[0, c0].get_position(), axes[0, c1].get_position()
        fig.text((b0.x0 + b1.x1) / 2, 0.985, lab, ha="center", va="top", fontsize="small", color=INK)
        fig.add_artist(plt.Line2D([b0.x0, b1.x1], [0.95, 0.95], color=MUTED, lw=0.8))
    _save(fig, out, "fig10_summary")


def model_compare(results, out, models, meth="CR3-headline", sc="lam_F", disc=False, panels=PANELS, name=None,
                  legend_size="x-small"):
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
        ax.set_ylabel("(Q − Q$_\\mathrm{DFT}$) / tol", color=INK, fontsize="small")
        if title:
            ax.set_title(title, color=INK, fontsize="medium", loc="left")
        hs, ls_ = ax.get_legend_handles_labels()
        keep = [i for i, l_ in enumerate(ls_) if ip == 0 or l_.startswith(tuple(order))]     # shared entries once, on top
        ax.legend([hs[i] for i in keep], [ls_[i] for i in keep], fontsize=legend_size, frameon=False, ncol=2, columnspacing=1.0,
                  handletextpad=0.3,
                  loc="lower left" if ip == 0 else "lower right", bbox_to_anchor=(0, 1.08) if ip == 0 else (1.0, 1.0))
    axes[-1, 0].set_xticks(x, [short(k) for k in COVER_KEYS], rotation=60, ha="right")
    fig.tight_layout()
    _save(fig, out, name or ("fig8_model_compare_disc" if disc else "fig6_model_compare"))


def force_calibration(results, out, nbins=20, level=0.9, min_atoms=300):
    """Per-atom force uncertainty against the observed error on the test split, per model.  Top: binned on predicted sigma
    (equal counts), RMS |error| against RMS sigma, for the CR3 shape with one hold-out conformal quantile and with the
    learned continuous scale (ARD fit, ood/learned_scale.py; sigma = radius sqrt(v) / chi_3^-1(level)), and the
    propagation's methods on the BLR fit (analysis/force_baselines.py: CR3 x lam_F, B1, B2, B3); the grey per-atom density
    behind is the one-quantile method's (traces[0]) only.
    Bottom: coverage at `level` overall and per GAP-18 configuration type with >= min_atoms test atoms, every method's
    region the Gaussian ellipsoid of its own 3x3 force covariance, e^T (C + eps tr C/3 I)^-1 e <= chi_3^-1(level)^2: for the
    one quantile and the learned scale that is their conformal region (s <= q); for the others d_<k> of force_baselines
    (lam_F is fitted on the test split: in sample)."""
    from scipy.stats import chi
    ms = [m for m in ("o3d12", "o4d20") if (results / f"learned_scale_{m}_atoms.npz").exists()]
    if not ms:
        return
    k3 = chi(3).ppf(level)
    fig, axes = plt.subplots(2, len(ms), figsize=_size(5.6 * len(ms), 9.6), squeeze=False, facecolor=SURFACE)

    def binned(x, y, key):
        o = np.argsort(key)
        chunks = np.array_split(o, nbins)
        return (np.array([np.sqrt(np.mean(x[c] ** 2)) for c in chunks]), np.array([np.sqrt(np.mean(y[c] ** 2)) for c in chunks]))
    for j, m in enumerate(ms):
        z = np.load(results / f"learned_scale_{m}_atoms.npz", allow_pickle=True)
        e, v, s, t = z["test_e"], z["test_v"], z["test_s"], z["test_t"]
        q1, rl = float(z["q_one"]), z["test_radius_seed0"]
        # method colours as the talk figures (METH_COL), never the model blue/orange of fig6: both CR3 scales share the
        # headline magenta (solid: one quantile; dash-dot: lam_F), the learned scale is grey
        traces = [("CR3 shape × one hold-out quantile", e, q1 * np.sqrt(v) / k3, t, s <= q1, "o", METH_COL["headline"]),
                  ("learned continuous scale", e, rl * np.sqrt(v) / k3, t, s <= rl, "P", MUTED)]
        pb = results / f"force_baselines_{m}.npz"
        if pb.exists():
            b = np.load(pb, allow_pickle=True)
            for k, lab, mk, col in (("headline", "CR3 × λ$_F$ (in sample on test)", "s", METH_COL["headline"]),
                                    ("B1", "B1: posterior × q̂", "^", METH_COL["B1"]),
                                    ("B2", "B2: POPS (50 % cut)", "D", METH_COL["B2"]), ("B3", "B3: CR3, λ = 1", "v", CAT[4])):
                sd = b[f"sd_{k}"]
                traces.append((lab, b["err_norm"], sd, b["config_type"], b[f"d_{k}"] <= k3, mk, col))
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
            ax.plot(bx, by, marker=mk, color=col, lw=1.8, ms=5, mec=SURFACE, label=lab, zorder=3,
                    ls="-." if lab.startswith("CR3 × λ") else "-")
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
            ax.plot(xs + (k_ - 2.5) * 0.07, y, marker=mk, color=col, lw=1.0, ms=5, mec=SURFACE, label=lab, zorder=3,
                    ls="-." if lab.startswith("CR3 × λ") else "-")
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


def cr3_schematic(out, seed=0, n=500, noise=0.001, P=6, calib=0.2):
    X_MIN = 0.0                                                      # the smooth LJ part (x < 0: the clipped wall)
    """Schematic of the CR3 construction on the regression demo's LJ scene (kermodegroup/demos, ?scene=lj-cr3; labels
    nearly noiseless, noise sd 0.001, as for DFT training data):
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
    axa.fill_between(xg, fit - lam * sd_c(Bg), fit + lam * sd_c(Bg), color=METH_COL["headline"], alpha=0.20, lw=0,
                     label=f"CR3 band, λ σ (λ = {lam:.1f} from calibration)")
    axa.fill_between(xg, fit - sd_post, fit + sd_post, color=METH_COL["B1"], alpha=0.5, lw=0, label="posterior band, σ²A⁻¹")
    axa.scatter(Xtr[:, 0], ytr, s=5, color=MUTED, alpha=0.6, lw=0, label="training points", zorder=2)
    axa.scatter(Xc[:, 0], yc, s=12, color=INK, marker="x", alpha=0.8, lw=0.8, label="calibration points", zorder=2)
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
    for S, col, lab, a_ in ((proj(Scr3), METH_COL["headline"], "CR3, 1 sd (λ = 1)", 0.22), (proj(Ai), METH_COL["B1"], "posterior, 1 sd", 0.5)):
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
        summary_fig(results, o, models)
        model_compare(results, o, models)
        model_compare(results, o, models, disc=True)
        force_calibration(results, o)
        json.dump({"correlation_model": m3}, open(o / "figures.json", "w"))

    set_context("paper")

# Talk method colours: never CAT[0]/CAT[1], which the strip plots (fig6, fig8) use for the two model sizes.  B1 and POPS
# keep their fig7 colours (green, amber).  The headline is a deep magenta outside CAT: every free CAT slot sat within
# OKLab dE 6 of a model or method colour under simulated deutan/protan vision (CAT[5] purple: 5.5 from the model
# blue); #a8277e is >= 14 from all four, >= 8.9 from the prior's grey, contrast 6.4:1 on white.
METH_COL = {"headline": "#a8277e", "B1": CAT[2], "B2": CAT[3]}
COV_METH = (("b1", "B1: posterior × q̂", METH_COL["B1"]), ("b2", "B2: POPS box", METH_COL["B2"]),
            ("hl", "CR3 × λ$_F$ (headline)", METH_COL["headline"]))
COV_PRIOR = ("prior", "prior", MUTED)


def talk_cov_shapes(results, out, corr_model="o3d12", models=("o3d12", "o4d20"), floor=1e-4):
    """Talk only (not in the paper, not called by all_figures): the shapes of the three parameter distributions
    (results/cov_shapes_<m>.npz, analysis/cov_shapes.py).  Top: correlation matrices of corr_model, the prior (shared,
    diagonal) first, parameters sorted
    by body order (a raw covariance shows nothing: the standard deviations span ~8 decades).  Bottom: per model, how
    far the headline and POPS stretch or squash B1 along its principal directions (generalised eigenvalues in B1's
    metric, each method at its own calibration; the prior unscaled), computed within each body-order block so the axis can be labelled
    by body order (whole-space directions mix them); each curve ends at its rank, beyond which the width is zero,
    and values below `floor` are drawn at it (marked ▼)."""
    set_context("talk")
    z = {m: np.load(results / f"cov_shapes_{m}.npz") for m in models}
    fig = plt.figure(figsize=(12, 6.75), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.62)
    top = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.08)
    bot = gs[1].subgridspec(1, len(models), wspace=0.22)
    c = z[corr_model]
    bc = c["bc"]
    edges = np.flatnonzero(np.diff(bc)) + 0.5
    mid = (np.r_[-0.5, edges] + np.r_[edges, len(bc) - 0.5]) / 2
    orders = [str(int(b)) for b in np.unique(bc)]
    for j, (k, lab, col) in enumerate((("prior", "prior (all)", MUTED),) + COV_METH):
        ax = fig.add_subplot(top[j])
        im = ax.imshow(c[f"corr_{k}"], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        for e in edges:
            ax.axhline(e, color=INK, lw=0.7)
            ax.axvline(e, color=INK, lw=0.7)
        ax.set_xticks(mid, orders)
        ax.set_yticks(mid, orders if j == 0 else [])
        ax.tick_params(length=0, labelsize="x-small", labelcolor=MUTED)
        if j == 0:
            ax.set_ylabel("body order", fontsize="x-small", color=MUTED)
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(3)
        ax.set_title(lab, fontsize="small", color=INK, loc="left")
        ax.set_xlabel(f"mean |ρ| = {float(c[f'mean_abs_corr_{k}']):.2f}", fontsize="x-small", color=INK)
    cb = fig.colorbar(im, cax=fig.add_subplot(top[4]), ticks=[-1, 0, 1])
    cb.set_label(f"correlation, {corr_model}", fontsize="x-small", color=INK)
    cb.ax.tick_params(labelsize="x-small", colors=MUTED, labelcolor=INK)
    cb.outline.set_visible(False)
    for i, m in enumerate(models):
        ax = fig.add_subplot(bot[i])
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        zm = z[m]
        orders = [int(b) for b in np.unique(zm["bc"])]
        ax.axhline(1, color=INK, lw=1.2, ls="--", zorder=1)
        for n, b in enumerate(orders):                     # one equal-width segment per body order
            nb = int((zm["bc"] == b).sum())
            if n:
                ax.axvline(n, color=MUTED, lw=0.8)
            ax.text(n + 0.5, 1.02, f"{b}-body", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize="xx-small", color=INK)
            ax.text(n + 0.5, -0.04, f"{nb}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize="xx-small", color=MUTED)
            for k, lab, col in (COV_PRIOR,) + COV_METH[1:]:
                r = int(zm[f"rank_{k}_b{b}"])
                sb = np.asarray(zm[f"stretch_{k}_b{b}"])[:r]
                if not r:
                    continue
                x = n + (np.arange(r) + 0.5) / nb
                below = np.flatnonzero(sb < floor)
                e = below[0] if len(below) else r            # stop at the floor: ▼ marks where the curve leaves it
                ax.plot(x[:e + 1] if e < r else x, np.maximum(sb[:e + 1] if e < r else sb, floor), color=col, lw=2.0,
                        zorder=3, solid_capstyle="butt")
                ax.plot(x[min(e, r - 1)], floor if e < r else sb[-1], "v" if e < r else "o", color=col, ms=6,
                        mec=SURFACE, zorder=4, clip_on=False)
        ax.set_xlim(0, len(orders))
        ax.set_xticks([])
        ax.set_yscale("log")
        ax.set_ylim(floor, 1e7)
        ax.set_title(f"{m}: width relative to B1, per body order", fontsize="x-small", color=INK, loc="left", pad=22)
        if i == 0:
            ax.set_ylabel("width / B1 width", fontsize="x-small", color=INK)
    fig.text(0.5, 0.035, "principal directions, sorted within each body-order block (number of parameters below the axis);"
             " ● rank reached, ▼ below 10$^{-4}$", ha="center", fontsize="xx-small", color=INK)
    h = [plt.Line2D([], [], color=col, lw=2.4, label=lab) for k, lab, col in (COV_PRIOR,) + COV_METH[1:]]
    h.append(plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="B1: posterior × q̂ (= 1)"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize="x-small")
    _save(fig, out / "talk", "talk_cov_shapes")


def talk_cov_qoi(results, out, models=("o3d12", "o4d20")):
    """Talk only (not in the paper, not called by all_figures): the three parameter distributions pushed to the QoIs through the implicit-differentiation rows (results/cov_shapes_<m>.npz, linear as the delta
    method), as correlation matrices in the order and colours of talk_cov_shapes; one figure per model.  No prior
    panel: an uncalibrated prior pushed to the QoIs is not a distribution anyone would propagate."""
    set_context("talk")
    for m in models:
        z = np.load(results / f"cov_shapes_{m}.npz")
        keys, grp = [str(k) for k in z["qoi_keys"]], [str(g) for g in z["qoi_group"]]
        n = len(keys)
        edges = [i - 0.5 for i in range(1, n) if grp[i] != grp[i - 1]]
        fig = plt.figure(figsize=(12, 4.9), facecolor=SURFACE)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.06)
        for j, (k, lab, col) in enumerate(COV_METH):
            ax = fig.add_subplot(gs[j])
            im = ax.imshow(z[f"qoi_corr_{k}"], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            for e in edges:
                ax.axhline(e, color=INK, lw=0.7)
                ax.axvline(e, color=INK, lw=0.7)
            lab_q = [short(q) if "$" in short(q) else short(q).replace("_", " ") for q in keys]   # talk-only tidy
                                                                     # (short() is shared with the paper; keep mathtext)
            ax.set_xticks(range(n), lab_q, rotation=90)
            ax.set_yticks(range(n), lab_q if j == 0 else [])
            ax.tick_params(length=0, labelsize=7, labelcolor=INK)
            for sp in ax.spines.values():
                sp.set_color(col)
                sp.set_linewidth(3)
            ax.set_title(f"{lab}\nmean |ρ| = {float(z[f'qoi_mean_abs_corr_{k}']):.2f}", fontsize="x-small", color=INK,
                         loc="left")
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[3]), ticks=[-1, 0, 1])
        cb.set_label(f"QoI correlation, {m}", fontsize="x-small", color=INK)
        cb.ax.tick_params(labelsize="x-small", colors=MUTED, labelcolor=INK)
        cb.outline.set_visible(False)
        _save(fig, out / "talk", f"talk_cov_qoi_{m}")


def talk_keynote(results, out, m="o4d20"):
    """Talk only (not in the paper, not called by all_figures): keynote-sized versions of two paper figures.
    talk_strip_headline: fig6's headline row alone, counts in large type.  talk_summary_<m>: fig10 cut to one model,
    three methods (B3 dropped: near-zero bars) and the three coverages that carry the comparison (QoIs at 95 %; per-atom
    forces at 90 %, worst configuration type on the test split and held-out cracks and dislocations); same data.  The
    legend notes that POPS has no calibration target."""
    set_context("talk")
    model_compare(results, out / "talk", ["o3d12", "o4d20"], panels=PANELS[:1], name="talk_strip_headline",
                  legend_size="small")
    meth = SUMMARY_METH[:3]
    cols = _keynote_cols(results, m)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    y = np.arange(len(meth))
    for ax, (title, sub, fn, nom, col) in zip(axes, cols):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        for yi, c in enumerate(meth):
            v, lab = fn(c)
            ax.barh(yi, v, height=0.72, facecolor=METH_COL[c[3]], edgecolor=INK, linewidth=0.8, zorder=3)
            ax.text(v - 2, yi, lab or f"{v:.0f} %", va="center", ha="right", fontsize="small", color=SURFACE, zorder=5,
                    fontweight="bold")                                      # inside the bar: clear of the nominal line
        # the target applies only to methods whose scale is set to reach it (CR3 x lam_F, B1); the POPS box has none
        ax.axvline(nom, color=INK, lw=1.4, ls="--", zorder=4)
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(len(y) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize="x-small", colors=MUTED, labelcolor=INK)
        ax.set_title(f"{title}\n", fontsize="medium", color=INK, loc="left")
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize="x-small", color=MUTED, va="bottom")
        ax.set_xlabel("coverage / %", fontsize="x-small", color=INK)
    axes[0].set_yticks(y, [c[4] for c in meth], fontsize="small")
    axes[0].tick_params(axis="y", length=0, labelcolor=INK)
    fig.suptitle(f"{m}", x=0.01, ha="left", fontsize="medium", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.99, 0.01, "dashed: target level (POPS has no calibration target)", ha="right",
             va="bottom", fontsize="x-small", color=MUTED)
    _save(fig, out / "talk", f"talk_summary_{m}")


def _keynote_cols(results, m):
    """The three coverages of the keynote comparison for model m, as (title, subtitle, fn(method) -> (value %, label
    or None), target %, colour): QoIs at 95 % (results/coverage.csv); per-atom forces at 90 %, the worst configuration
    type (>= 300 test atoms, force_baselines_<m>.npz) and the held-out cracks and dislocations (force_methods_<m>.json)."""
    cov = {(r["model"], r["method"], r["scale"]): r for r in csv.DictReader(open(results / "coverage.csv"))}
    fm = json.load(open(results / f"force_methods_{m}.json"))
    from scipy.stats import chi
    k3 = chi(3).ppf(0.9)
    b = np.load(results / f"force_baselines_{m}.npz", allow_pickle=True)
    ty, cnt = np.unique(b["config_type"], return_counts=True)
    worst = lambda k: 100 * min(np.mean(b[f"d_{k}"][b["config_type"] == x] <= k3) for x in ty[cnt >= 300])

    def qcov(c):
        r = cov[(m, c[0], c[1])]
        return 100 * float(r["k95"]) / float(r["n"]), f"{int(float(r['k95']))}/{int(float(r['n']))}"
    return (("properties (QoIs)", "coverage of DFT, target 95 %", qcov, 95, CAT[2]),
            ("forces, worst type", "test split, target 90 %", lambda c: (worst(c[3]), None), 90, CAT[5]),
            ("forces, held out", "cracks + dislocations, target 90 %",
             lambda c: (100 * fm["ood"]["all"][c[3]]["coverage"], None), 90, CAT[5]))


def talk_dumbbell(results, out, models=("o3d12", "o4d20")):
    """Talk only (not in the paper, not called by all_figures): the keynote comparison as one dumbbell chart, the
    three coverages of _keynote_cols, one row per method (METH_COL), each method's two models
    as two dots joined by a line (hollow: smaller model, filled: larger; the arrow points to the larger).  The target
    legend notes that POPS has no calibration target -> talk_dumbbell."""
    set_context("talk")
    meth = SUMMARY_METH[:3]
    mcol = METH_COL
    vals = {m: _keynote_cols(results, m) for m in models}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    y = np.arange(len(meth))
    for j, ax in enumerate(axes):
        title, sub, _, nom, _ = vals[models[0]][j]
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for yi, c in enumerate(meth):
            col = mcol[c[3]]
            (v0, l0), (v1, l1) = (vals[m][j][2](c) for m in models)
            ax.annotate("", xy=(v1, yi), xytext=(v0, yi), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=3.2, mutation_scale=16, shrinkA=7, shrinkB=8))
            ax.plot(v0, yi, "o", ms=12, mfc=SURFACE, mec=col, mew=2.6, zorder=4)
            ax.plot(v1, yi, "o", ms=12, mfc=col, mec=col, mew=2.6, zorder=4)
            for v, l, other in ((v0, l0, v1), (v1, l1, v0)):          # label on the outer side of each dot
                left = v < other or (v == other and l is l0)
                ax.text(v + (-4 if left else 4), yi - 0.3, l or f"{v:.0f}", ha="right" if left else "left", va="center",
                        fontsize="x-small", color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))   # target line passes behind
        ax.axvline(nom, color=INK, lw=1.4, ls="--", zorder=2)
        ax.set_xlim(20, 105)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(len(meth) - 0.5, -0.65)
        ax.tick_params(axis="x", labelsize="x-small", colors=MUTED, labelcolor=INK)
        ax.set_title(f"{title}\n", fontsize="medium", color=INK, loc="left")
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize="x-small", color=MUTED, va="bottom")
        ax.set_xlabel("coverage / %", fontsize="x-small", color=INK)
    axes[0].set_yticks(y, [c[4] for c in meth], fontsize="small")
    for t, c in zip(axes[0].get_yticklabels(), meth):
        t.set_color(mcol[c[3]])
    axes[0].tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], ls="", marker="o", ms=11, mfc=SURFACE, mec=INK, mew=2.4, label=models[0]),
         plt.Line2D([], [], ls="", marker="o", ms=11, mfc=INK, mec=INK, mew=2.4, label=models[1]),
         plt.Line2D([], [], color=INK, lw=1.4, ls="--", label="target level (POPS has no calibration target)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize="x-small", bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out / "talk", "talk_dumbbell")
