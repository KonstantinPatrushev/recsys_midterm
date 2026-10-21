"""Paired CV experiments on F1 (same user folds for every config).

Run: .venv/bin/python -I src/experiments.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import mapk  # noqa: E402
from ranker import CATS, PARAMS, feature_cols, to_pandas, top5_from_scores, user_folds  # noqa: E402

K = 5


def run_cv(X, gt, params, label="y", rounds=400, cols=None, weight=None, k=K):
    cols = cols or feature_cols(X)
    fmap = user_folds(X["u"].to_numpy(), k, 0)
    fold = np.array([fmap[u] for u in X["u"].to_numpy()])
    Xp = to_pandas(X, cols)
    y = X[label].to_numpy()
    uarr = X["u"].to_numpy()
    oof = np.zeros(X.height)
    for f in range(k):
        tr, va = fold != f, fold == f
        g = np.unique(uarr[tr], return_counts=True)[1]
        w = None if weight is None else weight[tr]
        ds = lgb.Dataset(Xp[tr], y[tr], group=g if params["objective"] in ("lambdarank", "rank_xendcg") else None,
                         weight=w, categorical_feature=[c for c in CATS if c in cols])
        m = lgb.train(params, ds, num_boost_round=rounds)
        oof[va] = m.predict(Xp[va])
    return oof


def rank_norm(X, s):
    return X.select("u").with_columns(pl.Series("s", s)).with_columns(
        (pl.col("s").rank() / pl.len()).over("u").alias("r"))["r"].to_numpy()


def main():
    t0 = time.time()
    views = load_views("sum")
    sched = load_schedule()
    gt = ground_truth(views, (19, 30))
    sub_u = set(submission_users()["u"].to_list())
    users = list(gt.keys())
    sub_eval = [u for u in users if u in sub_u]
    X = label_top5(make_table(views, sched, list(range(11, 19)), list(range(19, 31))), gt)
    X = X.filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
    X = X.with_columns(pl.col("cnt").clip(0, 30).alias("cnt_c"))
    base_cols = feature_cols(X)
    base_cols = [c for c in base_cols if c != "cnt_c"]
    no_method = [c for c in base_cols if not c.startswith("m_")]
    res, oofs = {}, {}

    def score(name, oof):
        pred = top5_from_scores(X, oof)
        res[name] = {"map5_all": round(mapk(gt, pred, 5, users), 5), "map5_sub": round(mapk(gt, pred, 5, sub_eval), 5)}
        oofs[name] = oof
        print(f"{name:40s} {res[name]}  ({time.time()-t0:.0f}s)", flush=True)

    P = dict(PARAMS)
    score("lambdarank_nomethod", run_cv(X, gt, P, cols=no_method))
    score("lambdarank", run_cv(X, gt, P, cols=base_cols))
    score("lambdarank_lr03_r800", run_cv(X, gt, dict(P, learning_rate=0.03), rounds=800, cols=base_cols))
    score("lambdarank_leaves31", run_cv(X, gt, dict(P, num_leaves=31), cols=base_cols))
    score("lambdarank_leaves127_mdl100", run_cv(X, gt, dict(P, num_leaves=127, min_data_in_leaf=100), cols=base_cols))
    score("lambdarank_trunc10", run_cv(X, gt, dict(P, lambdarank_truncation_level=10), cols=base_cols))
    score("xendcg", run_cv(X, gt, dict(P, objective="rank_xendcg"), cols=base_cols))
    score("binary", run_cv(X, gt, dict(P, objective="binary", metric="auc"), cols=base_cols))
    score("poisson_cnt", run_cv(X, gt, dict(P, objective="poisson", metric="poisson"), label="cnt_c", cols=base_cols))
    w = np.where(np.isin(X["u"].to_numpy(), list(sub_u)), 2.0, 1.0)
    score("lambdarank_subw2", run_cv(X, gt, P, cols=base_cols, weight=w))

    # rank-average ensembles of the variants above
    def ens(names):
        return sum(rank_norm(X, oofs[n]) for n in names) / len(names)

    score("ENS_lr+bin", ens(["lambdarank", "binary"]))
    score("ENS_lr+bin+pois", ens(["lambdarank", "binary", "poisson_cnt"]))
    score("ENS_lr+xendcg+bin+pois", ens(["lambdarank", "xendcg", "binary", "poisson_cnt"]))
    score("ENS_all_rankers", ens(["lambdarank", "lambdarank_lr03_r800", "lambdarank_leaves31",
                                  "lambdarank_leaves127_mdl100", "xendcg"]))
    (ROOT / "evidence" / "experiments_F1.json").write_text(json.dumps(res, indent=2))
    np.save(ROOT / "cache" / "oof_F1.npy", np.stack([oofs[k_] for k_ in res]))
    (ROOT / "cache" / "oof_F1_names.json").write_text(json.dumps(list(res)))


if __name__ == "__main__":
    main()
