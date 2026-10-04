"""Audit metrics: per-group ranking/verdict alignment, gaps, bootstrap CIs, disagreement sensitivity."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

MIN_ITEMS = 20   # min scored items with a group majority for that group's AUC/kappa


def base_rate_threshold(s, pooled_rate):
    """Flag the top q of scores, q = pooled human unsafe rate (32.7% on DICES-350)."""
    return float(np.nanquantile(s, 1 - pooled_rate))


def group_alignment(s, maj, pooled_rate, subset=None):
    """AUC and Cohen's kappa of guard scores `s` against each group's majority label.

    Returns a DataFrame indexed by group with columns auc, kappa, n, plus the threshold used.
    The base-rate-matched threshold is computed on the evaluated subset.
    """
    idx = maj.index if subset is None else maj.index.intersection(subset)
    s = s.loc[idx]
    ok = s.notna()
    thr = base_rate_threshold(s, pooled_rate)
    v = (s >= thr).astype(int)
    rows = {}
    for grp in maj.columns:
        gm = maj[grp].loc[idx]
        m = ok & gm.notna()
        if m.sum() < MIN_ITEMS or gm[m].nunique() < 2:
            rows[grp] = (np.nan, np.nan, int(m.sum()))
            continue
        rows[grp] = (roc_auc_score(gm[m], s[m]),
                     cohen_kappa_score(gm[m].astype(int), v[m]),
                     int(m.sum()))
    return pd.DataFrame(rows, index=["auc", "kappa", "n"]).T, thr


def max_min_gap(values):
    values = pd.Series(values).dropna()
    return float(values.max() - values.min()) if len(values) else np.nan


def bootstrap_gaps(s, maj, pooled_rate, rng, n_boot=2000):
    """Item-level bootstrap 95% CIs for the max-min AUC gap and kappa gap across groups.

    Group-majority targets are treated as fixed (rater-sampling uncertainty is not propagated).
    """
    s = s.loc[maj.index]
    items = maj.index[s.notna()].to_numpy()
    auc_gaps, kappa_gaps = [], []
    for _ in range(n_boot):
        bi = rng.choice(items, len(items), replace=True)
        sb = s.loc[bi].to_numpy()
        vb = sb >= np.nanquantile(sb, 1 - pooled_rate)
        aucs, kaps = [], []
        for grp in maj.columns:
            gm = maj[grp].loc[bi].to_numpy()
            m = ~np.isnan(gm)
            if m.sum() < MIN_ITEMS or len(set(gm[m])) < 2:
                continue
            aucs.append(roc_auc_score(gm[m], sb[m]))
            kaps.append(cohen_kappa_score(gm[m].astype(int), vb[m].astype(int)))
        auc_gaps.append(max(aucs) - min(aucs) if len(aucs) >= 2 else np.nan)
        kappa_gaps.append(max(kaps) - min(kaps) if len(kaps) >= 2 else np.nan)
    return (np.nanpercentile(auc_gaps, [2.5, 97.5]).tolist(),
            np.nanpercentile(kappa_gaps, [2.5, 97.5]).tolist())


def rank_confidence(s, q):
    """|F(p) - q| with F the empirical CDF of the guard's scores and q the threshold quantile."""
    s = s.dropna()
    return pd.Series(np.abs(rankdata(s) / len(s) - q), index=s.index)


def disagreement_sensitivity(s, entropy, q):
    """Spearman rho between rank-based guard confidence and rater entropy."""
    conf = rank_confidence(s.loc[entropy.index], q)
    rho, p = spearmanr(conf, entropy.loc[conf.index])
    return float(rho), float(p)


def auc_difference_in_differences(s_a, s_b, maj, grp_1, grp_2, items):
    """(AUC_g1 - AUC_g2) for guard a minus the same contrast for guard b, on `items`."""
    def contrast(s):
        sub = s.loc[items]
        ok = sub.notna()
        return (roc_auc_score(maj[grp_1].loc[items][ok], sub[ok])
                - roc_auc_score(maj[grp_2].loc[items][ok], sub[ok]))
    return contrast(s_a) - contrast(s_b)


def bootstrap_did(s_a, s_b, maj, grp_1, grp_2, items, rng, n_boot=2000):
    items = np.asarray(items)
    out = []
    for _ in range(n_boot):
        bi = pd.Index(rng.choice(items, len(items), replace=True))
        if maj[grp_1].loc[bi].nunique() < 2 or maj[grp_2].loc[bi].nunique() < 2:
            continue
        out.append(auc_difference_in_differences(s_a, s_b, maj, grp_1, grp_2, bi))
    return np.percentile(out, [2.5, 97.5]).tolist()
