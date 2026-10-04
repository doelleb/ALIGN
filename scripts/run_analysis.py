"""Compute every number reported in the paper from data/guard_scores_full.csv (CPU, ~3 min).

    python scripts/run_analysis.py

Writes results/results.json and prints Tables 1-5, the Section 3 / Appendix E feasibility values,
and the Appendix C robustness checks.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from align.data import (GROUP_SHORT, GUARD_SHORT, ROOT, contested_items,  # noqa: E402
                        human_structures, load_dices, load_scores)
from align.feasibility import PatternLP, pairwise_disagreement  # noqa: E402
from align.metrics import (auc_difference_in_differences, base_rate_threshold,  # noqa: E402
                           bootstrap_did, bootstrap_gaps, disagreement_sensitivity,
                           group_alignment, max_min_gap)

SEED = 20260717
N_BOOT = 2000
OUT = ROOT / "results" / "results.json"


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * (70 - len(title))}")


def main():
    df = load_dices()
    scores, guards = load_scores()
    pooled, entropy, race = human_structures(df)
    q = pooled.mean()                     # pooled human unsafe rate
    contested = contested_items(race)
    groups = list(race.columns)
    R = {"seed": SEED, "n_boot": N_BOOT}

    # ------------------------------------------------------------------ dataset (Sec. 4.1)
    section("Dataset")
    R["dataset"] = dict(
        n_items=int(len(race)), n_ratings=int(len(df)),
        raters_per_group={GROUP_SHORT[g]: int(n) for g, n in
                          df.groupby("rater_race").rater_id.nunique().items()},
        group_unsafe_rate={GROUP_SHORT[g]: r3(v) for g, v in
                           df.groupby("rater_race").unsafe.mean().items()},
        pooled_unsafe_rate=r3(q), n_contested=int(len(contested)),
        n_scored={GUARD_SHORT[c]: int(scores[c].notna().sum()) for c in guards},
        unparseable_items={GUARD_SHORT[c]: scores.index[scores[c].isna()].tolist() for c in guards})
    print(json.dumps(R["dataset"], indent=1))

    # ------------------------------------------------------------ feasibility (Sec. 3, App. E)
    section("Proposition 1 / achievable-alignment region")
    delta = pairwise_disagreement(race)
    (ga, gb), dmax = max(delta.items(), key=lambda kv: kv[1])
    lp = PatternLP(race)
    egal, util, equal = lp.egalitarian(), lp.utilitarian(), lp.utilitarian_equal()
    R["feasibility"] = dict(
        max_pairwise_delta=r3(dmax), max_pair=[GROUP_SHORT[ga], GROUP_SHORT[gb]],
        prop1_bound=r3(1 - dmax / 2), n_patterns=lp.m,
        egalitarian_min_agreement=r3(egal.min()),
        utilitarian_mean=r3(util.mean()), utilitarian_gap=r3(util.max() - util.min()),
        equal_constrained_mean=r3(equal.mean()),
        cost_of_equality=r3(util.mean() - equal.mean()),
        coverage_frontier={f"{c:.2f}": r3(lp.coverage_frontier(c))
                           for c in (1.0, .95, .90, .85, .80, .78)})
    print(json.dumps(R["feasibility"], indent=1))

    # ------------------------------------------------------- per-group alignment (Tables 1, 2)
    section("Table 1 / Table 2: AUC and kappa vs. racial-group majority")
    R["guards"] = {}
    for col in guards:
        full, thr = group_alignment(scores[col], race, q)
        sub, _ = group_alignment(scores[col], race, q, subset=contested)
        rho, p = disagreement_sensitivity(scores[col], entropy, 1 - q)
        R["guards"][GUARD_SHORT[col]] = dict(
            threshold=float(thr),
            auc={GROUP_SHORT[g]: r3(full.auc[g]) for g in groups},
            kappa={GROUP_SHORT[g]: r3(full.kappa[g]) for g in groups},
            auc_contested={GROUP_SHORT[g]: r3(sub.auc[g]) for g in groups},
            auc_gap_full=r3(max_min_gap(full.auc)), kappa_gap_full=r3(max_min_gap(full.kappa)),
            auc_gap_contested=r3(max_min_gap(sub.auc)),
            kappa_gap_contested=r3(max_min_gap(sub.kappa)),
            best_tracked_group=GROUP_SHORT[full.auc.idxmax()],
            conf_entropy_rho=r3(rho), conf_entropy_p=float(p))
    G = R["guards"]
    names = list(G)
    tab = pd.DataFrame({n: {**{f"AUC {GROUP_SHORT[g]}": G[n]["auc"][GROUP_SHORT[g]] for g in groups},
                            **{f"kappa {GROUP_SHORT[g]}": G[n]["kappa"][GROUP_SHORT[g]] for g in groups},
                            "AUC gap (full)": G[n]["auc_gap_full"],
                            "AUC gap (contested)": G[n]["auc_gap_contested"],
                            "conf-entropy rho": G[n]["conf_entropy_rho"],
                            "conf-entropy p": round(G[n]["conf_entropy_p"], 4)} for n in names})
    print(tab.to_string())

    # Human reference column of Table 2: pooled rater distribution predicting each group majority.
    R["human_pooled_auc"] = {GROUP_SHORT[g]: r3(roc_auc_score(race[g], pooled)) for g in groups}
    print("human pooled AUC:", R["human_pooled_auc"])

    # --------------------------------------------------------------- bootstrap CIs (Table 1, App. C)
    section(f"Bootstrap 95% CIs ({N_BOOT} item-level resamples)")
    rng = np.random.default_rng(SEED)
    for col in guards:
        a, k = bootstrap_gaps(scores[col], race, q, rng, N_BOOT)
        G[GUARD_SHORT[col]].update(auc_gap_full_ci95=[r3(x) for x in a],
                                   kappa_gap_full_ci95=[r3(x) for x in k])
    rng = np.random.default_rng(SEED)
    for col in guards:
        a, k = bootstrap_gaps(scores[col], race.loc[contested], q, rng, N_BOOT)
        G[GUARD_SHORT[col]].update(auc_gap_contested_ci95=[r3(x) for x in a],
                                   kappa_gap_contested_ci95=[r3(x) for x in k])
    for n in names:
        print(f"{n:10s} full {G[n]['auc_gap_full']:.3f} {G[n]['auc_gap_full_ci95']} | "
              f"contested {G[n]['auc_gap_contested']:.3f} {G[n]['auc_gap_contested_ci95']}")

    # ------------------------------------------------------------- pooled vs worst group (Table 3)
    section("Table 3: pooled-majority vs. worst-group AUC")
    pooled_maj = (pooled > .5).astype(int)
    for col in guards:
        s = scores[col].loc[race.index]; ok = s.notna()
        n = GUARD_SHORT[col]
        pa = roc_auc_score(pooled_maj[ok], s[ok])
        worst = min(G[n]["auc"].values())
        G[n].update(pooled_auc=r3(pa), worst_group_auc=r3(worst), overstatement=r3(pa - worst))
        print(f"{n:10s} pooled {pa:.3f} worst {worst:.3f} overstatement {pa - worst:.3f}")

    # ----------------------------------------------------- agreement by disagreement tercile (Table 4)
    section("Table 4: agreement with pooled majority by rater-entropy tercile")
    tercile = pd.qcut(entropy, 3, labels=["low", "medium", "high"])
    for col in guards:
        s = scores[col].loc[race.index]; ok = s.notna()
        v = (s >= base_rate_threshold(s, q)).astype(int)
        agree = (v == pooled_maj)[ok]
        t = agree.groupby(tercile[ok], observed=True).mean()
        G[GUARD_SHORT[col]]["agreement_by_tercile"] = {k: r3(x) for k, x in t.items()}
        print(f"{GUARD_SHORT[col]:10s}", {k: round(x, 2) for k, x in t.items()})

    # ------------------------------------------------------- cross-guard orderings (Table 5)
    section("Table 5: Kendall tau between guards' per-group AUC orderings")
    R["kendall_tau"] = {}
    for a, b in itertools.combinations(names, 2):
        t, p = kendalltau(pd.Series(G[a]["auc"]).rank(), pd.Series(G[b]["auc"]).rank())
        R["kendall_tau"][f"{a} / {b}"] = dict(tau=r3(t), p=r3(p))
        print(f"{a:10s} / {b:10s} tau={t:+.2f} p={p:.2f}")

    # ------------------------------------------------------------------ reversal test (App. C)
    section("Reversal test: (Black - White) AUC contrast, SG-2B minus LG3-1B, contested items")
    inv = {v: k for k, v in GUARD_SHORT.items()}
    blk, wht = "Black/African American", "White"
    s_sg, s_lg = scores[inv["SG-2B"]], scores[inv["LG3-1B"]]
    did = auc_difference_in_differences(s_sg, s_lg, race, blk, wht, contested)
    ci = bootstrap_did(s_sg, s_lg, race, blk, wht, contested, np.random.default_rng(SEED), N_BOOT)
    R["reversal_test"] = dict(guards=["SG-2B", "LG3-1B"], groups=["Black", "White"],
                              subset="contested", did=r3(did), ci95=[r3(x) for x in ci])
    print(f"DiD = {did:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # --------------------------------------------------------- sensitivity coding (App. C)
    section("Sensitivity: 'Unsure' coded as unsafe")
    pooled_s, _, race_s = human_structures(df, coding="unsafe_sens")
    for col in guards:
        full, _ = group_alignment(scores[col], race_s, pooled_s.mean())
        G[GUARD_SHORT[col]]["auc_gap_full_unsure_unsafe"] = r3(max_min_gap(full.auc))
        print(f"{GUARD_SHORT[col]:10s} AUC gap {max_min_gap(full.auc):.3f}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1))
    print("\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
