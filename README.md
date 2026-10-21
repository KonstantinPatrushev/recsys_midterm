# LLM4Rec midterm: SWEET.TV TV-programme recommender (Kaggle)

Konstantin Patrushev, HSE LLM4Rec, midterm quiz. Competition:
[sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) —
predict each user's 5 most frequently fully-watched TV shows for weeks 31–42 (MAP@5).

## Result

| Submission (file in `submissions/`) | Public | Private |
|---|---|---|
| `sub01_ranker_v1.csv` — LightGBM LambdaRank, habit + schedule features | 0.44705 | 0.43278 |
| `sub02_hybrid_lr31_r250_s5.csv` — hybrid ranker with the four course methods as features | 0.44688 | 0.43848 |
| `det_ens_D.csv` — hybrid, sum/max label ensemble, test history weeks 20–24 + 28–30 | 0.44206 | 0.43082 |
| **`det_ens_AD.csv` — hybrid, sum/max label ensemble, both test histories (designated final)** | 0.44608 | **0.43968** |
| `det_ens_A.csv` — hybrid, sum/max label ensemble, test history weeks 15–19 + 28–30 | 0.44847 | 0.43865 |

The original 2020 winner scored 0.44297 public / 0.42912 private. Scores are recorded in
`evidence/kaggle_submissions.csv`. Files in `submissions/not_submitted/` were built but never uploaded.

## Layout

| Path | What |
|---|---|
| `src/prep.py` | raw CSV → parquet; interval join of viewing sessions with the schedule (`src/intervals.py`) |
| `src/data.py` | ground truth exactly as the organisers describe it (screen time ≥ 0.8, top-5 by count), folds |
| `src/eda.py` | data checks: logging gap in weeks 20–27, timezone alignment, ties at the 5th place |
| `src/baselines.py` | popularity / repeat baselines |
| `src/methods.py` | course methods as stand-alone recommenders: CBF, association rules, MF (implicit ALS), personalized PageRank |
| `src/features.py` | candidate generation + features for the ranker |
| `src/hybrid.py` | hybrid table: method scores as features and extra candidates |
| `src/ranker.py`, `src/experiments*.py`, `src/cv_final.py` | LightGBM ranker, user-grouped CV, paired experiments |
| `src/robustness.py`, `src/final_selection.py`, `src/history_proxy.py`, `src/diagnostics.py`, `src/report_stats.py` | folds F0/F2, label-definition study, test-history proxy, adversarial validation, error analysis, costs |
| `src/predict_test.py`, `src/blend_submission.py` | train on the validation fold, predict weeks 31–42, rank-average blends |
| `src/business.py`, `src/figures.py`, `src/build_report.py` | two-user analysis, figures, report (numbers read from `evidence/`) |
| `tests/` | metric and interval-join checks, submission format check, byte-for-byte rebuilds of submission 1 and of the final file |
| `evidence/` | every number used in the report (JSON) and the Kaggle submission log |

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python kaggle pandas polars pyarrow numpy scipy scikit-learn lightgbm implicit matplotlib
# LightGBM on macOS needs: brew install libomp
.venv/bin/kaggle competitions download sweettv-tv-program-recommender -p data/raw && (cd data/raw && unzip -o *.zip)
.venv/bin/python -I src/prep.py
.venv/bin/python -I tests/test_metrics.py && .venv/bin/python -I tests/test_intervals.py
.venv/bin/python -I src/methods.py          # stand-alone methods on the validation fold
tests/reproduce_final.sh                    # rebuilds submissions/det_ens_AD.csv and compares SHA-256
.venv/bin/python -I tests/check_submission.py submissions/det_ens_AD.csv
```

Competition data is not redistributed here (`data/` is git-ignored).
