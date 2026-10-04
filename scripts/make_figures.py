"""Render Figures 1-3 into figures/ (CPU).

    python scripts/make_figures.py

Fig 1: qualitative contested item. Selection rule: among contested items, maximise
       rater entropy x mean guard rank-confidence (item 64 on DICES-350), subject to a manual
       screen excluding gratuitously offensive content (override with --item).
Fig 2: per-group AUC heatmap.
Fig 3: threshold-constituency maps (group best matched by Cohen's kappa vs. threshold percentile).
"""
import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from align.data import (GROUP_SHORT, GUARD_SHORT, ROOT, contested_items,  # noqa: E402
                        human_structures, load_dices, load_scores)
from align.metrics import base_rate_threshold, group_alignment, rank_confidence  # noqa: E402

FIG = ROOT / "figures"


def fig1(df, scores, guards, pooled, entropy, race, item=None):
    q = 1 - pooled.mean()
    conf = pd.concat([rank_confidence(scores[c], q) for c in guards], axis=1).mean(axis=1)
    contested = contested_items(race)
    ranked = (entropy.loc[contested] * conf.loc[contested]).sort_values(ascending=False)
    pick = ranked.index[0] if item is None else item
    print(f"Fig 1: item {pick} (top candidates: {list(ranked.index[:5])})")

    row = df[df.item_id == pick].iloc[0]
    rates = df[df.item_id == pick].groupby("rater_race").unsafe.mean().rename(GROUP_SHORT)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(11, 2.6), gridspec_kw={"width_ratios": [2.2, 1.6, 1.6]})
    turns = str(row.context).strip().split("\n")[-2:] + [f"Chatbot: {row.response}"]
    a0.text(0, 1, "Conversation (final turns):\n" + "\n".join(textwrap.fill(t, 48) for t in turns)
            + f"\n\nPooled unsafe-rate: {pooled[pick]:.2f}   entropy: {entropy[pick]:.2f} bits",
            va="top", fontsize=7.5, family="monospace")
    a0.axis("off")
    a1.barh(rates.index[::-1], rates.values[::-1], color="0.55")
    a1.axvline(.5, ls="--", c="k", lw=.8); a1.set_xlim(0, 1)
    a1.set_xlabel("share rating unsafe"); a1.set_title("Rater groups", fontsize=9)
    names = [GUARD_SHORT[c] for c in guards]
    vals = [scores.at[pick, c] for c in guards]
    thrs = [base_rate_threshold(scores[c].loc[race.index], pooled.mean()) for c in guards]
    a2.barh(names[::-1], vals[::-1],
            color=["tab:red" if v >= t else "tab:blue" for v, t in zip(vals[::-1], thrs[::-1])])
    for y, t in enumerate(thrs[::-1]):
        a2.plot([t, t], [y - .4, y + .4], c="k", lw=1)
    a2.set_xlabel("unsafe-score (| = threshold)"); a2.set_title("Guard verdicts", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig1_contested_item.pdf"); plt.close()


def fig2(scores, guards, pooled, race):
    A = pd.DataFrame({GUARD_SHORT[c]: group_alignment(scores[c], race, pooled.mean())[0].auc
                      for c in guards}).rename(index=GROUP_SHORT).astype(float)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    im = ax.imshow(A.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(A.columns, fontsize=8)
    ax.set_yticks(range(A.shape[0])); ax.set_yticklabels(A.index, fontsize=8)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A.values[i, j]
            star = " *" if v == A.values[:, j].max() else ""
            ax.text(j, i, f"{v:.3f}{star}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < A.values.mean() else "black")
    plt.colorbar(im, label="AUC vs. group majority")
    ax.set_title("Per-group ranking alignment (* = best-tracked group)", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig2_group_auc_heatmap.pdf"); plt.close()


def fig3(scores, guards, race):
    groups = [GROUP_SHORT[g] for g in race.columns]
    pcts = np.linspace(.50, .99, 50)
    fig, axes = plt.subplots(1, len(guards), figsize=(3 * len(guards), 2.4), sharey=True)
    for ax, col in zip(axes, guards):
        s = scores[col].loc[race.index].dropna()
        best = []
        for pct in pcts:
            v = (s >= np.quantile(s, pct)).astype(int)
            kk = {GROUP_SHORT[g]: cohen_kappa_score(race[g].loc[s.index].astype(int), v)
                  for g in race.columns}
            best.append(groups.index(max(kk, key=kk.get)))
        ax.scatter(pcts, best, s=8, c=[f"C{b}" for b in best])
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=7)
        ax.set_xlabel("threshold percentile", fontsize=8); ax.set_title(GUARD_SHORT[col], fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig3_threshold_constituency.pdf"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", type=int, default=None, help="override the Fig 1 item_id")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    df = load_dices()
    scores, guards = load_scores()
    pooled, entropy, race = human_structures(df)
    fig1(df, scores, guards, pooled, entropy, race, args.item)
    fig2(scores, guards, pooled, race)
    fig3(scores, guards, race)
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
