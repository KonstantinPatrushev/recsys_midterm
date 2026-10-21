"""LightGBM ranker over the candidate table: user-grouped K-fold CV on a validation fold.

Run: .venv/bin/python -I src/ranker.py   (from midterm/)
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_views, submission_users  # noqa: E402
from metrics import mapk  # noqa: E402

DROP = {"u", "tv_show_id", "cnt", "rel", "y", "main_c_t", "main_hour_t", "fold"}
CATS = ["tv_show_category", "tv_show_genre_1"]

PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [5],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 8,
    "seed": 0,
    "deterministic": True,
    "force_col_wise": True,
}


def feature_cols(X: pl.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in DROP]


def to_pandas(X: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = X.select(cols).to_pandas()
    for c in CATS:
        if c in df:
            df[c] = df[c].astype("category")
    return df


def user_folds(users: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    uu = np.unique(users)
    perm = rng.permutation(len(uu))
    return {u: int(i % k) for u, i in zip(uu[perm], range(len(uu)))}


def top5_from_scores(X: pl.DataFrame, score: np.ndarray) -> dict:
    s = X.select(["u", "tv_show_id"]).with_columns(pl.Series("score", score))
    top = (s.sort(["u", "score"], descending=[False, True]).group_by("u", maintain_order=True).head(5)
           .group_by("u", maintain_order=True).agg(pl.col("tv_show_id")))
    return {u: list(t) for u, t in top.iter_rows()}


def cv(X: pl.DataFrame, gt: dict, params=PARAMS, label="y", rounds=400, k=5, seed=0, cols=None, verbose=True):
    """OOF predictions for users in gt (train only on gt users)."""
    X = X.filter(pl.col("u").is_in(pl.Series(list(gt.keys())).implode())).sort(["u", "tv_show_id"])
    cols = cols or feature_cols(X)
    fmap = user_folds(X["u"].to_numpy(), k, seed)
    fold = np.array([fmap[u] for u in X["u"].to_numpy()])
    oof = np.zeros(X.height)
    Xp = to_pandas(X, cols)
    y = X[label].to_numpy()
    uarr = X["u"].to_numpy()
    imps = np.zeros(len(cols))
    for f in range(k):
        tr, va = fold != f, fold == f
        gtr = np.unique(uarr[tr], return_counts=True)[1]  # rows are sorted by u
        dtr = lgb.Dataset(Xp[tr], y[tr], group=gtr, categorical_feature=[c for c in CATS if c in cols])
        m = lgb.train(params, dtr, num_boost_round=rounds)
        oof[va] = m.predict(Xp[va])
        imps += m.feature_importance("gain")
    pred = top5_from_scores(X, oof)
    return X, oof, pred, dict(zip(cols, imps / k))


def main():
    views = load_views("sum")
    X = pl.read_parquet(ROOT / "cache" / "X_F1.parquet")
    gt = ground_truth(views, (19, 30))
    sub_u = set(submission_users()["u"].to_list())
    users = list(gt.keys())
    sub_eval = [u for u in users if u in sub_u]
    t0 = time.time()
    res = {}
    for label in ["y", "rel"]:
        Xs, oof, pred, imp = cv(X, gt, label=label)
        res[label] = {"map5_all": mapk(gt, pred, 5, users), "map5_sub": mapk(gt, pred, 5, sub_eval)}
        print(label, res[label], f"{time.time()-t0:.0f}s", flush=True)
    top_imp = sorted(imp.items(), key=lambda kv: -kv[1])[:25]
    for k_, v in top_imp:
        print(f"  {k_:22s} {v:12.1f}")
    (ROOT / "evidence" / "ranker_v1.json").write_text(json.dumps({"scores": res, "importance_top25": top_imp}, indent=2))


if __name__ == "__main__":
    main()
