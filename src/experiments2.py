"""Second round of paired CV experiments on F1: simpler trees, regularisation, seed noise,
ensembles of simple models.

Run: .venv/bin/python -I src/experiments2.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import rank_norm, run_cv  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import apk, mapk  # noqa: E402
from ranker import PARAMS, feature_cols, top5_from_scores  # noqa: E402


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
    cols = [c for c in feature_cols(X) if c != "cnt_c"]
    res, oofs, aps = {}, {}, {}

    def score(name, oof):
        pred = top5_from_scores(X, oof)
        res[name] = {"map5_all": round(mapk(gt, pred, 5, users), 5), "map5_sub": round(mapk(gt, pred, 5, sub_eval), 5)}
        oofs[name] = oof
        aps[name] = np.array([apk(gt[u], pred.get(u, []), 5) for u in users])
        print(f"{name:40s} {res[name]}  ({time.time()-t0:.0f}s)", flush=True)

    P = dict(PARAMS)
    L31 = dict(P, num_leaves=31)
    for s in [0, 1, 2]:
        score(f"lr_l31_seed{s}", run_cv(X, gt, dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s), cols=cols))
    score("lr_l15", run_cv(X, gt, dict(P, num_leaves=15), cols=cols))
    score("lr_l31_mdl200", run_cv(X, gt, dict(L31, min_data_in_leaf=200), cols=cols))
    score("lr_l31_ff05_l2_10", run_cv(X, gt, dict(L31, feature_fraction=0.5, lambda_l2=10.0), cols=cols))
    score("lr_l31_r250", run_cv(X, gt, L31, rounds=250, cols=cols))
    score("lr_l31_r600", run_cv(X, gt, L31, rounds=600, cols=cols))
    score("bin_l31", run_cv(X, gt, dict(L31, objective="binary", metric="auc"), cols=cols))
    score("pois_l31", run_cv(X, gt, dict(L31, objective="poisson", metric="poisson"), label="cnt_c", cols=cols))
    score("xendcg_l31", run_cv(X, gt, dict(L31, objective="rank_xendcg"), cols=cols))

    def ens(names):
        return sum(rank_norm(X, oofs[n]) for n in names) / len(names)

    score("ENS_l31_lr+bin+pois", ens(["lr_l31_seed0", "bin_l31", "pois_l31"]))
    score("ENS_l31_lr3seeds+bin+pois", ens(["lr_l31_seed0", "lr_l31_seed1", "lr_l31_seed2", "bin_l31", "pois_l31"]))
    score("ENS_l31_lr+bin+pois+xendcg", ens(["lr_l31_seed0", "bin_l31", "pois_l31", "xendcg_l31"]))

    # paired bootstrap over users: P(ensemble > single lambdarank seed0)
    rng = np.random.default_rng(0)
    base = aps["lr_l31_seed0"]
    boot = {}
    for name in ["ENS_l31_lr+bin+pois", "ENS_l31_lr3seeds+bin+pois", "lr_l31_seed1", "lr_l15"]:
        d = aps[name] - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        boot[name] = {"mean_diff": round(float(d.mean()), 5), "ci95": [round(float(np.quantile(bs, 0.025)), 5),
                                                                     round(float(np.quantile(bs, 0.975)), 5)]}
        print("bootstrap vs lr_l31_seed0:", name, boot[name])
    (ROOT / "evidence" / "experiments2_F1.json").write_text(json.dumps({"scores": res, "paired_bootstrap_vs_lr_l31_seed0": boot}, indent=2))


if __name__ == "__main__":
    main()
