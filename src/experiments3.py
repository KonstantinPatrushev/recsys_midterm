"""Third round on F1: number of boosting rounds, and whether to train only on users whose
target window (weeks 19-30) is fully logged (the test window is presumably complete).

Run: .venv/bin/python -I src/experiments3.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, P, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import rank_norm, run_cv  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import mapk  # noqa: E402
from ranker import PARAMS, feature_cols, top5_from_scores  # noqa: E402


def main():
    t0 = time.time()
    views = load_views("sum")
    sched = load_schedule()
    gt = ground_truth(views, (19, 30))
    sub_u = set(submission_users()["u"].to_list())
    users = list(gt.keys())
    sub_eval = [u for u in users if u in sub_u]
    sess = pl.read_parquet(P / "sessions.parquet")
    wk = sess.filter(pl.col("week").is_between(19, 30)).group_by("u").agg(pl.col("week").n_unique().alias("nw"))
    nw = dict(wk.iter_rows())
    clean = [u for u in users if nw.get(u, 0) >= 11]
    clean_sub = [u for u in clean if u in sub_u]
    print(f"eval users {len(users)}, clean (>=11 of 12 target weeks logged) {len(clean)}, clean&sub {len(clean_sub)}")

    X = label_top5(make_table(views, sched, list(range(11, 19)), list(range(19, 31))), gt)
    X = X.filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
    X = X.with_columns(pl.col("cnt").clip(0, 30).alias("cnt_c"))
    cols = [c for c in feature_cols(X) if c != "cnt_c"]
    L31 = dict(PARAMS, num_leaves=31)
    res, oofs = {}, {}

    def score(name, oof, Xs=X):
        pred = top5_from_scores(Xs, oof)
        res[name] = {"map5_all": round(mapk(gt, pred, 5, users), 5), "map5_sub": round(mapk(gt, pred, 5, sub_eval), 5),
                     "map5_clean": round(mapk(gt, pred, 5, clean), 5), "map5_clean_sub": round(mapk(gt, pred, 5, clean_sub), 5)}
        oofs[name] = oof
        print(f"{name:34s} {res[name]}  ({time.time()-t0:.0f}s)", flush=True)

    for r in [150, 200, 250, 300]:
        score(f"lr_l31_r{r}", run_cv(X, gt, L31, rounds=r, cols=cols))
    # train only on fully-logged users: zero weight for the others (same folds, same eval)
    w = np.where(np.isin(X["u"].to_numpy(), clean), 1.0, 0.0)
    score("lr_l31_r250_trainclean", run_cv(X, gt, L31, rounds=250, cols=cols, weight=w))
    w2 = np.where(np.isin(X["u"].to_numpy(), clean), 1.0, 0.3)
    score("lr_l31_r250_downweight_unclean", run_cv(X, gt, L31, rounds=250, cols=cols, weight=w2))
    for s in [1, 2]:
        score(f"lr_l31_r250_seed{s}", run_cv(X, gt, dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s), rounds=250, cols=cols))
    for s in [0, 1, 2]:
        score(f"bin_l31_r250_seed{s}", run_cv(X, gt, dict(L31, objective="binary", metric="auc", seed=s, bagging_seed=s,
                                                          feature_fraction_seed=s), rounds=250, cols=cols))

    def ens(names):
        return sum(rank_norm(X, oofs[n]) for n in names) / len(names)

    score("ENS_lr3_r250", ens(["lr_l31_r250", "lr_l31_r250_seed1", "lr_l31_r250_seed2"]))
    score("ENS_lr3+bin3_r250", ens(["lr_l31_r250", "lr_l31_r250_seed1", "lr_l31_r250_seed2",
                                    "bin_l31_r250_seed0", "bin_l31_r250_seed1", "bin_l31_r250_seed2"]))
    (ROOT / "evidence" / "experiments3_F1.json").write_text(json.dumps(
        {"n_eval": len(users), "n_clean": len(clean), "n_clean_sub": len(clean_sub), "scores": res}, indent=2))


if __name__ == "__main__":
    main()
