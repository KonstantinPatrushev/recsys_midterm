"""Candidate statistics of the final (hybrid) table on F1 and on the test -> evidence/candidates_final.json

Run: .venv/bin/python -I src/candidates_final.py
"""
from __future__ import annotations

import json
import sys
import pathlib

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from features import candidate_recall  # noqa: E402
from hybrid import make_table  # noqa: E402
from predict_test import F1_HIST, F1_TARGET, TEST_HIST, TEST_TARGET  # noqa: E402
from ranker import feature_cols  # noqa: E402

views, sched = load_views("sum"), load_schedule()
gt = ground_truth(views, (19, 30))
sub_u = set(submission_users()["u"].to_list())
users = list(gt)
X = make_table(views, sched, F1_HIST, F1_TARGET)
Xe = X.filter(pl.col("u").is_in(pl.Series(users).implode()))
Xt = make_table(views, sched, TEST_HIST, TEST_TARGET, with_labels=False).filter(
    pl.col("u").is_in(pl.Series(list(sub_u)).implode()))
out = {
    "F1_rows_eval_users": Xe.height,
    "F1_candidates_per_user_median": float(Xe.group_by("u").len()["len"].median()),
    "F1_recall_all": round(candidate_recall(Xe, gt, users), 4),
    "F1_recall_sub": round(candidate_recall(Xe, gt, [u for u in users if u in sub_u]), 4),
    "test_rows": Xt.height,
    "test_candidates_per_user_median": float(Xt.group_by("u").len()["len"].median()),
    "n_features": len([c for c in feature_cols(X) if c not in ("cnt",)]),
}
(ROOT / "evidence" / "candidates_final.json").write_text(json.dumps(out, indent=2))
print(out)
