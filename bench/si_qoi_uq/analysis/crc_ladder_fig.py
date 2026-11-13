"""CRC region size along the linear ACE ladder, one line per method (plan section 5 C, ladder; revision, new figure).

results/crc_ladder.json (crc_config.py ladder: CR3 headline, B1, B2 and the POPS ellipsoid at every size since 2026-11-13; the
same 100 random test halves; alpha 0.1) and results/ace_pareto.json (A100 float64 µs/atom-step, test force RMSE). Panels:
whole-configuration CRC (joint loss: the Mahalanobis region of a configuration's force vector) and per-atom CRC (crc.py's
loss). y: the geometric semi-axis per force component exp(size), eV/Å, mean over the halves, bars p05-p95; the test force RMSE
as the reference a region cannot beat by much. B3 is not drawn: CRC refits the scale of every shape, so B3 (CR3 unscaled) gives
exactly the headline's regions.

    python -m bench.si_qoi_uq.analysis.crc_ladder_fig   -> figures/fig11_crc_ladder.{png,pdf}
"""
import json

import numpy as np

from .. import figures as F
from ..run import RESULTS

METHODS = (("headline", "CR3 × CRC (headline)", "#a8277e", "o"), ("B1", "B1: posterior", F.CAT[2], "s"),
           ("B2", "B2: POPS box", F.CAT[3], "^"), ("ellipsoid", "POPS ellipsoid", F.CAT[5], "D"))
LOSSES = (("joint", "Whole configuration (joint)"), ("atom", "Per atom"))
ALPHA = "0.1"
PAPER = ("o3d12", "o4d20")


def main():
    from matplotlib.ticker import FuncFormatter, NullFormatter
    plt = F.plt
    lad = json.load(open(RESULTS / "crc_ladder.json"))
    acc = {p["size"]: p for p in json.load(open(RESULTS / "ace_pareto.json"))}
    models = [m for m in acc if any(f"{m}/headline" in lad["loss"][lo][ALPHA]["halves"] for lo, _ in LOSSES)]
    models.sort(key=lambda m: acc[m]["cost_gpu"])
    fig, axes = plt.subplots(1, 2, figsize=F._size(10.5, 4.2), sharey=True, facecolor=F.SURFACE)
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax, (lo, title) in zip(axes, LOSSES):
        F._style(ax)
        H = lad["loss"][lo][ALPHA]["halves"]
        x_all = np.array([acc[m]["cost_gpu"] for m in models])
        ax.plot(x_all, [acc[m]["rmse_F"] for m in models], "--", color=F.MUTED, lw=1, zorder=1, label="test force RMSE")
        for k, lab, col, mk in METHODS:
            ms = [m for m in models if f"{m}/{k}" in H]
            if not ms:
                continue
            x = np.array([acc[m]["cost_gpu"] for m in ms])
            s = [H[f"{m}/{k}"]["size"] for m in ms]
            y = np.exp([v["mean"] for v in s])
            err = np.abs(np.exp([[v["p05"] for v in s], [v["p95"] for v in s]]) - y)
            ax.errorbar(x, y, yerr=err, fmt="none", ecolor=col, elinewidth=1, capsize=0, zorder=2)
            ax.plot(x, y, "-", color=col, lw=2, zorder=3)
            ax.scatter(x, y, s=36, marker=mk, color=col, edgecolor=F.SURFACE, linewidth=1.2, zorder=4, label=lab)
        for i, m in enumerate(models):                           # model names along the top (two rows: the cheap end is
            ax.annotate(m, (acc[m]["cost_gpu"], 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3 + 9 * (i % 2)),  # crowded), the paper's two in bold
                        textcoords="offset points", ha="center", va="bottom", fontsize="xx-small", color=F.INK,
                        fontweight="bold" if m in PAPER else "normal")
        ax.set_xscale("log"); ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plain); axis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1])
        ax.set_xlabel("A100 cost (µs / atom-step)", color=F.INK, fontsize="small")
        ax.set_title(title, color=F.INK, pad=24)
    axes[0].set_ylabel("region semi-axis per force component (eV/Å)", color=F.INK, fontsize="small")
    axes[0].set_yticks([0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(h), fontsize="x-small", frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F._save(fig, RESULTS.parent / "figures", "fig11_crc_ladder")
    print("figures/fig11_crc_ladder.{png,pdf}:", ", ".join(models))


if __name__ == "__main__":
    main()
