"""Score DICES-350 with the guard battery (GPU). Writes data/guard_scores_full.csv.

The released CSV already contains the scores used in the paper; rerun this only to regenerate
them. All guards are gated on HuggingFace: accept each model's license, then `export HF_TOKEN=...`.

    python scripts/score_guards.py                      # smoke test, then all guards
    python scripts/score_guards.py --guards wildguard   # a subset
    python scripts/score_guards.py --out my_scores.csv
"""
import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from align.data import SCORES_PATH, load_dices, load_items  # noqa: E402
from align.guards import GUARDS  # noqa: E402

SEED = 20260717


def free(g):
    del g
    gc.collect()
    torch.cuda.empty_cache()


def smoke_test(name, items, n=5):
    """Go/no-go gate: a guard must produce non-constant, parseable scores on a few items."""
    g = GUARDS[name]()
    ps = [g.p_unsafe(r.context, r.response) for r in items.sample(n, random_state=SEED).itertuples()]
    free(g)
    print(f"  smoke {name}:", [None if math.isnan(p) else round(p, 3) for p in ps])
    return pd.Series(ps).nunique(dropna=True) > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guards", nargs="+", default=list(GUARDS), choices=list(GUARDS))
    ap.add_argument("--out", type=Path, default=SCORES_PATH)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    items = load_items(load_dices())
    print(len(items), "items |", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    run = args.guards
    if not args.skip_smoke:
        run = [n for n in run if smoke_test(n, items)]
        print("passed smoke test:", run)

    # Resume: keep any completed, non-degenerate columns from an earlier run.
    results = items.copy()
    if args.out.exists():
        prev = pd.read_csv(args.out)
        for name in list(run):
            col = f"p_{name}"
            if col in prev.columns and prev[col].dropna().nunique() > 10:
                results = results.merge(prev[["item_id", col]], on="item_id", how="left")
                run.remove(name)
                print("reusing saved scores:", name)

    for name in run:
        print(f"scoring {name} ...")
        g = GUARDS[name]()
        ps = []
        for k, r in enumerate(items.itertuples()):
            try:
                ps.append(g.p_unsafe(r.context, r.response))
            except Exception as e:  # counted as unparseable, never silently dropped
                print(f"  item {r.item_id} failed: {e}")
                ps.append(np.nan)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(items)}")
        results[f"p_{name}"] = ps
        results.to_csv(args.out, index=False)   # checkpoint after every guard
        free(g)

    # Diagnostics: near-constant scores mean a broken wrapper, not a finding.
    for col in [c for c in results.columns if c.startswith("p_")]:
        p = results[col].dropna()
        q = np.round(np.quantile(p, [0, .25, .5, .75, 1]), 3) if len(p) else []
        print(f"[{col}] n={len(p)} quantiles={q} nunique={p.nunique()}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
