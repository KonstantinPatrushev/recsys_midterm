"""Two Kaggle-style diagnostics -> evidence/diagnostics.json

1. Adversarial validation: can a classifier tell training rows (F1, history 11-18) from
   test rows (history 15-19 + 28-30, target 31-42)?  AUC near 0.5 = no detectable shift.
   Reported for the full feature set and after dropping the most shifted features.
2. Error analysis on F1 out-of-fold predictions of the submitted configuration:
   MAP@5 split by candidate recall (were all 5 true shows among the candidates?) and by
   novelty of the truth (how many true shows were never fully watched in history).

Run: .venv/bin/python -I src/diagnostics.py
"""
from __future__ import annotations

import json
import sys
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import rank_norm, run_cv  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import apk  # noqa: E402
from predict_test import F1_HIST, F1_TARGET, L31, TEST_HIST, TEST_TARGET  # noqa: E402
from ranker import CATS, feature_cols, to_pandas, top5_from_scores  # noqa: E402


def adversarial(Xtr, Xte, cols, seed=0):
    both = pl.concat([Xtr.select(cols).with_columns(pl.lit(0).alias("is_test")),
                      Xte.select(cols).with_columns(pl.lit(1).alias("is_test"))])
    Pb = to_pandas(both, cols)
    for c in CATS:
        if c in Pb:
            Pb[c] = Pb[c].astype(str).astype("category")
    y = both["is_test"].to_numpy()
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 4, len(y))
    pred = np.zeros(len(y))
    imp = np.zeros(len(cols))
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "verbose": -1, "num_threads": 8,
              "seed": seed, "deterministic": True, "force_col_wise": True}
    for f in range(4):
        tr, va = fold != f, fold == f
        m = lgb.train(params, lgb.Dataset(Pb[tr], y[tr], categorical_feature=[c for c in CATS if c in cols]), 100)
        pred[va] = m.predict(Pb[va])
        imp += m.feature_importance("gain")
    return roc_auc_score(y, pred), sorted(zip(cols, imp / 4), key=lambda kv: -kv[1])[:10]


def main():
    views, sched = load_views("sum"), load_schedule()
    gt = ground_truth(views, (19, 30))
    users = list(gt)
    sub_u = set(submission_users()["u"].to_list())
    Xtr = label_top5(make_table(views, sched, F1_HIST, F1_TARGET), gt)
    Xtr = Xtr.filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
    Xte = make_table(views, sched, TEST_HIST, TEST_TARGET, with_labels=False)
    Xte = Xte.filter(pl.col("u").is_in(pl.Series(list(sub_u)).implode()))
    cols = feature_cols(Xtr)
    out = {}
    auc, top = adversarial(Xtr, Xte, cols)
    out["adversarial_all_features"] = {"auc": round(auc, 4), "top_features": [(k, round(v, 1)) for k, v in top]}
    print("adversarial AUC (all features):", round(auc, 4), [k for k, _ in top[:6]], flush=True)
    drop = [k for k, _ in top[:5]]
    cols2 = [c for c in cols if c not in drop]
    auc2, top2 = adversarial(Xtr, Xte, cols2)
    out["adversarial_without_top5_shifted"] = {"dropped": drop, "auc": round(auc2, 4),
                                               "top_features": [(k, round(v, 1)) for k, v in top2]}
    print("adversarial AUC (without top-5):", round(auc2, 4), [k for k, _ in top2[:6]], flush=True)

    # error analysis with the submitted configuration (lambdarank, 31 leaves, 250 rounds, 5 seeds)
    score = np.zeros(Xtr.height)
    for seed in range(5):
        p = dict(L31, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        score += rank_norm(Xtr, run_cv(Xtr, gt, p, rounds=250, cols=cols))
    pred = top5_from_scores(Xtr, score)
    cand = {u: set(s) for u, s in Xtr.group_by("u").agg(pl.col("tv_show_id")).iter_rows()}
    hv = views.filter(pl.col("week").is_between(11, 18) & pl.col("full"))
    seen = {u: set(s) for u, s in hv.group_by("u").agg(pl.col("tv_show_id")).iter_rows()}
    rows = []
    for u in users:
        t = gt[u]
        rows.append({"u": u, "sub": u in sub_u, "ap": apk(t, pred.get(u, []), 5),
                     "in_cand": len(set(t) & cand.get(u, set())), "novel": sum(s not in seen.get(u, set()) for s in t)})
    E = pl.DataFrame(rows)
    by_cand = E.group_by("in_cand").agg(pl.len().alias("users"), pl.col("ap").mean().round(4).alias("map5")).sort("in_cand")
    by_nov = E.group_by(pl.col("novel").clip(0, 3)).agg(pl.len().alias("users"), pl.col("ap").mean().round(4).alias("map5")).sort("novel")
    out["error_by_truth_items_in_candidates"] = by_cand.to_dicts()
    out["error_by_novel_truth_items_(3=3+)"] = by_nov.to_dicts()
    full = E.filter(pl.col("in_cand") == 5)
    out["map5_if_all_5_in_candidates"] = round(float(full["ap"].mean()), 4)
    out["share_users_all_5_in_candidates"] = round(full.height / E.height, 4)
    out["map5_all_check"] = round(float(E["ap"].mean()), 5)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("adversarial")}, indent=1))
    (ROOT / "evidence" / "diagnostics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
