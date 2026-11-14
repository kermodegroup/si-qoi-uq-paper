"""Held-out whole-configuration CRC: paired risk differences against CR3 (revision Section 3.9; plan section 5 D, step 3 and the
paper's models, step 0).

results/final_rules.json (o4d16, hand and x 10 energy weight) and final_rules_<m>.json (o3d12, o4d20, hand), from final.py ood:
the BLR refits without cracks and screw dislocations, every shape CRC-calibrated on the in-distribution test split (alpha 0.1),
risk on the held-out configurations, 100 paired splits. Per view (E, F joint, F frac, dE_type) and model, d = risk(other) -
risk(CR3) for B1 and the POPS ellipsoid, mean and tau = sd / sqrt 2 (the decision rules' form). Below zero: the other shape
misses fewer held-out configurations than CR3.

    python -m bench.si_qoi_uq.analysis.crc_heldout_fig   -> figures/fig12_crc_heldout.{png,pdf}
"""
import json

import numpy as np

from .. import figures as F
from ..run import RESULTS

ROWS = (("o3d12", "hand", "o3d12"), ("o4d16", "hand", "o4d16"), ("o4d16", "Ex10", "o4d16, E × 10"), ("o4d20", "hand", "o4d20"))
VIEWS = (("E joint 0.1", "energy"), ("F joint 0.1", "forces (joint)"), ("F frac 0.1", "forces (fraction)"),
         ("dE_type joint 0.1", "energy differences (dE_type)"))
OTHERS = (("B1", "B1: posterior", F.METH_COL["B1"], "s"), ("ellipsoid", "POPS ellipsoid", F.CAT[5], "D"))


def load():
    out = {}
    for m, cand, lab in ROWS:
        r = json.load(open(RESULTS / ("final_rules.json" if m == "o4d16" else f"final_rules_{m}.json")))["rule1_2"]
        for k, _, _, _ in OTHERS:
            row = r.get(f"{cand}: {k} - CR3", {})
            out[(lab, k)] = {v: (row[v]["risk"]["mean"], row[v]["risk"]["tau"]) for v, _ in VIEWS if v in row}
    return out


def main():
    plt = F.plt
    d = load()
    labs = [lab for *_, lab in ROWS]
    fig, axes = plt.subplots(1, len(VIEWS), figsize=F._size(10.5, 3.4), sharey=True, facecolor=F.SURFACE)
    y = np.arange(len(labs))[::-1]
    for ax, (v, title) in zip(axes, VIEWS):
        F._style(ax)
        ax.axvline(0, color=F.INK, lw=1)
        for j, (k, lab, col, mk) in enumerate(OTHERS):
            pts = [(yy + (0.15 - 0.3 * j), *d[(l, k)][v]) for yy, l in zip(y, labs) if v in d.get((l, k), {})]
            if not pts:
                continue
            yy, m_, t_ = map(np.array, zip(*pts))
            ax.errorbar(m_, yy, xerr=t_, fmt="none", ecolor=col, elinewidth=1.4, capsize=0, zorder=2)
            ax.scatter(m_, yy, s=36, marker=mk, color=col, edgecolor=F.SURFACE, linewidth=1.2, zorder=3, label=lab)
        ax.set_title(title, color=F.INK, fontsize="small")
        ax.set_xlim(-0.8, 0.5)
        ax.set_xlabel("risk − risk(CR3)", color=F.INK, fontsize="small")
        ax.grid(axis="x", color=F.GRID, lw=0.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labs, fontsize="small")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(h), fontsize="x-small", frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F._save(fig, RESULTS.parent / "figures", "fig12_crc_heldout")
    print("figures/fig12_crc_heldout.{png,pdf}")


if __name__ == "__main__":
    main()
