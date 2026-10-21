# LLM4Rec midterm: SWEET.TV TV-programme recommender (Kaggle)

Konstantin Patrushev, HSE LLM4Rec, midterm quiz. Competition:
[sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) —
predict each user's 5 most frequently fully-watched TV shows for weeks 31–42 (MAP@5).

## Layout

| Path | What |
|---|---|
| `src/prep.py` | raw CSV → parquet; interval join of viewing sessions with the schedule (`src/intervals.py`) |
| `src/data.py` | ground truth exactly as the organisers describe it (screen time ≥ 0.8, top-5 by count), folds |
| `src/eda.py` | data checks: logging gap in weeks 20–27, timezone alignment, ties at the 5th place |
| `src/baselines.py` | popularity / repeat baselines |
| `src/methods.py` | course methods as stand-alone recommenders: CBF, association rules, MF (implicit ALS), personalized PageRank |
| `src/features.py` | candidate generation + features for the ranker |
| `src/hybrid.py` | hybrid table: method scores as features and extra candidates; ablation |
| `src/ranker.py`, `src/experiments.py` | LightGBM ranker, user-grouped CV, paired experiments |
| `src/predict_test.py` | train on the validation fold, predict weeks 31–42, write `submissions/*.csv` |
| `src/business.py` | two-user business analysis |
| `tests/` | metric and interval-join checks, submission format check |
| `evidence/` | every number used in the report (JSON) and the Kaggle submission log |

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python kaggle pandas polars pyarrow numpy scipy scikit-learn lightgbm implicit
# LightGBM on macOS needs: brew install libomp
.venv/bin/kaggle competitions download sweettv-tv-program-recommender -p data/raw && (cd data/raw && unzip -o *.zip)
.venv/bin/python -I src/prep.py
.venv/bin/python -I tests/test_metrics.py && .venv/bin/python -I tests/test_intervals.py
.venv/bin/python -I src/methods.py          # stand-alone methods on the validation fold
.venv/bin/python -I src/hybrid.py           # hybrid ablation
.venv/bin/python -I src/predict_test.py --name <name>
.venv/bin/python -I tests/check_submission.py submissions/<name>.csv
```

Competition data is not redistributed here (`data/` is git-ignored).
