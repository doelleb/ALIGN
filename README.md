# ALIGN: Auditing Language-Model Intergroup Guard Neutrality

Code and data for the paper *ALIGN: Auditing Language-Model Intergroup Guard Neutrality*.

The audit compares open guard models against **disaggregated human judgments**: each racial
rater group's majority label on DICES-350, not one collapsed label. The repository contains
the guard-scoring harness (prompts, policy text, verdict-token scoring), the guard scores used
in the paper, and the analysis that computes every reported number and figure.

> **Content warning:** DICES-350 contains adversarial conversations with offensive content.

## Repository layout

```
align/
  data.py          DICES-350 loading, group majorities, rater entropy, contested items
  guards.py        guard wrappers: prompts, policies, decode-and-parse unsafe scores
  metrics.py       per-group AUC / kappa, gaps, bootstrap CIs, rank-based confidence
  feasibility.py   Proposition 1 bound and achievable-alignment LP (Sec. 3, App. E)
scripts/
  score_guards.py  GPU: score DICES-350 with the guard battery -> data/guard_scores_full.csv
  run_analysis.py  CPU: all tables and in-text numbers -> results/results.json
  make_figures.py  CPU: Figures 1-3 -> figures/
data/guard_scores_full.csv   unsafe scores p in [0,1] for 4 guards x 350 items
results/results.json         output of run_analysis.py
figures/                     output of make_figures.py
```

## Reproducing the paper (CPU only, a few minutes)

```bash
pip install -r requirements.txt
python scripts/run_analysis.py
python scripts/make_figures.py
```

DICES-350 is downloaded from the
[official repository](https://github.com/google-research-datasets/dices-dataset) on first use and
cached in `data/`. Both analysis scripts start from the released `data/guard_scores_full.csv`,
so no GPU or model access is needed.

| Paper | Source |
|---|---|
| Sec. 3: δ = .131, bound .934, 19 patterns, LP optimum .934 / .939, gap .034, cost of equality .005 | `run_analysis.py` → `feasibility` |
| App. E: coverage–equality frontier | `run_analysis.py` → `feasibility.coverage_frontier` |
| Sec. 4.1: rater counts, group base rates, 77 contested items | `run_analysis.py` → `dataset` |
| Table 1, Table 2: per-group AUC / κ, gaps, CIs, confidence–entropy ρ | `run_analysis.py` → `guards.*` |
| Table 2, Pooled column (human reference AUC) | `run_analysis.py` → `human_pooled_auc` |
| Table 3: pooled vs. worst-group AUC | `guards.*.pooled_auc`, `worst_group_auc` |
| Table 4: agreement by entropy tercile | `guards.*.agreement_by_tercile` |
| Table 5: Kendall's τ between guard orderings | `kendall_tau` |
| App. C: contested CIs, Unsure→unsafe coding, reversal test | `guards.*.auc_gap_contested_ci95`, `auc_gap_full_unsure_unsafe`, `reversal_test` |
| Figures 1, 2, 3 | `make_figures.py` |

### Analysis choices

- **Label coding:** `Q_overall == "Yes"` → unsafe. Sensitivity analysis: `"Unsure"` also → unsafe.
- **Group targets:** per-item majority (> 50% unsafe) of each racial rater group (26 Asian,
  29 Black, 22 Latine, 16 Multiracial, 30 White raters; every rater rated every item).
- **Verdict threshold:** base-rate-matched. The top 32.7% of each guard's scores are flagged,
  matching the pooled human unsafe rate.
- **Disagreement sensitivity:** Spearman ρ between rank-based confidence |F(p) − q| and binary
  rater entropy, where F is the empirical CDF of the guard's scores and q = 0.673.
- **Uncertainty:** item-level bootstrap with 2,000 resamples and seed `20260717`. Group-majority
  targets are treated as fixed, so rater-sampling uncertainty is not propagated.

## Re-scoring the guards (optional, GPU)

```bash
pip install -r requirements-gpu.txt
export HF_TOKEN=...          # accept each model's license on HuggingFace first
python scripts/score_guards.py --out data/guard_scores_rerun.csv
```

| Short name | HuggingFace model |
|---|---|
| LG3-1B | `meta-llama/Llama-Guard-3-1B` |
| SG-2B | `google/shieldgemma-2b` |
| SG-9B | `google/shieldgemma-9b` |
| WildGuard | `allenai/wildguard` |

Each guard sees the final chatbot response together with its conversation context. Llama Guard
gets the full parsed conversation. ShieldGemma and WildGuard get the last user turn plus the
response. ShieldGemma is prompted with a broadened, DICES-aligned policy (`ShieldGemma.POLICY` in
`align/guards.py`). The unsafe score comes from the verdict-token distribution at the generated
verdict step. Unparseable generations are recorded as NaN; on DICES-350 the only one is
WildGuard on item 154.

The paper's scores were produced on Kaggle (2× T4, float16, < 6 GPU-hours total) using each
model's default HuggingFace revision. The script runs a 5-item smoke test per guard before the
full run. It also checkpoints the CSV after each guard, so an interrupted run can resume.

## Data and licenses

- Code: MIT (see `LICENSE`).
- [DICES-350](https://github.com/google-research-datasets/dices-dataset) (Aroyo et al., 2023)
  is released under CC BY 4.0. `data/guard_scores_full.csv` repeats the DICES-350 `context`
  and `response` fields alongside the guard scores.
- Guard models are used under their respective licenses (Llama 3.2 Community License, Gemma
  Terms of Use, Apache 2.0 for WildGuard).

## Citation

```bibtex
@inproceedings{align2026,
  title     = {{ALIGN}: Auditing Language-Model Intergroup Guard Neutrality},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
