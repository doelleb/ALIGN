"""DICES-350 loading and human-side structures (group majorities, entropy, contested items)."""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DICES_URL = ("https://raw.githubusercontent.com/google-research-datasets/"
             "dices-dataset/main/350/diverse_safety_adversarial_dialog_350.csv")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DICES_PATH = DATA_DIR / "diverse_safety_adversarial_dialog_350.csv"
SCORES_PATH = DATA_DIR / "guard_scores_full.csv"

GROUP_AXIS = "rater_race"
MIN_RATERS = 10   # min raters per group per item for that group's majority (never binding on DICES-350)

# Short labels used in tables and figures.
GROUP_SHORT = {
    "Asian/Asian subcontinent": "Asian",
    "Black/African American": "Black",
    "LatinX, Latino, Hispanic or Spanish Origin": "Latine",
    "Multiracial": "Multiracial",
    "White": "White",
}
GUARD_SHORT = {
    "p_llama-guard-3-1b": "LG3-1B",
    "p_shieldgemma-2b": "SG-2B",
    "p_shieldgemma-9b": "SG-9B",
    "p_wildguard": "WildGuard",
}


def load_dices(path=DICES_PATH):
    """Rater-level DICES-350 (43,050 rows). Downloaded and cached on first use."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading DICES-350 -> {path}")
        urllib.request.urlretrieve(DICES_URL, path)
    df = pd.read_csv(path)
    assert df.shape[0] == 43050, df.shape
    # Primary coding: unsafe iff Q_overall == 'Yes'. Sensitivity: 'Unsure' also coded unsafe.
    df["unsafe"] = (df["Q_overall"] == "Yes").astype(int)
    df["unsafe_sens"] = df["Q_overall"].isin(["Yes", "Unsure"]).astype(int)
    return df


def load_items(df):
    """One row per conversation: item_id, context, response."""
    return (df.groupby("item_id")
              .agg(context=("context", "first"), response=("response", "first"))
              .reset_index())


def load_scores(path=SCORES_PATH):
    scores = pd.read_csv(path).set_index("item_id")
    guards = [c for c in scores.columns if c.startswith("p_")]
    return scores, guards


def binary_entropy(rate):
    p = np.clip(rate, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def human_structures(df, coding="unsafe", axis=GROUP_AXIS):
    """Pooled unsafe rate, rater entropy, and per-group majority labels (items x groups)."""
    pooled = df.groupby("item_id")[coding].mean().rename("pooled_rate")
    entropy = binary_entropy(pooled).rename("entropy")
    counts = df.groupby(["item_id", axis])[coding].count().unstack()
    means = df.groupby(["item_id", axis])[coding].mean().unstack()
    maj = (means > .5).astype(float).where(counts >= MIN_RATERS)
    return pooled, entropy, maj


def contested_items(maj):
    """Items on which group majorities directly oppose each other."""
    return maj.index[maj.nunique(axis=1) > 1]
