"""Training-data augmentation: add fold F3 (history 11-18, clean target 28-30) to the
training rows of the F1 model; evaluate on held-out users of F1 only (user-disjoint).

Run: .venv/bin/python -I src/experiments4.py
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
from experiments import rank_norm  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import mapk  # noqa: E402
from ranker import CATS, PARAMS, feature_cols, to_pandas, top5_from_scores, user_folds  # noqa: E402

L31 = dict(PARAMS, num_leaves=31)


def main():
    t0 = time.time()
    views, sched = load_views("sum"), load_schedule()
    gt1 = ground_truth(views, (19, 30))
    gt3 = ground_truth(views, (28, 30))
    sub_u = set(submission_users()["u"].to_list())
    users = list(gt1)
    sub_eval = [u for u in users if u in sub_u]
    X1 = label_top5(make_table(views, sched, list(range(11, 19)), list(range(19, 31))), gt1)
    X1 = X1.filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
    X3 = label_top5(make_table(views, sched, list(range(11, 19)), list(range(28, 31))), gt3)
    X3 = X3.filter(pl.col("u").is_in(pl.Series(list(gt3)).implode())).sort(["u", "tv_show_id"])
    cols = feature_cols(X1)
    X3 = X3.select(X1.columns)
    allu = np.union1d(X1["u"].to_numpy(), X3["u"].to_numpy())
    fmap = user_folds(allu, 5, 0)
    f1 = np.array([fmap[u] for u in X1["u"].to_numpy()])
    f3 = np.array([fmap[u] for u in X3["u"].to_numpy()])
    both = pl.concat([X1.with_columns(pl.lit(1).alias("fold_id")), X3.with_columns(pl.lit(3).alias("fold_id"))])
    Pb = to_pandas(both, cols)
    for c in CATS:
        Pb[c] = Pb[c].astype(str).astype("category")
    P1, P3 = Pb.iloc[: X1.height], Pb.iloc[X1.height:]
    res = {}
    for variant in ["F1_only", "F1+F3"]:
        oof = np.zeros(X1.height)
        for s in range(3):
            p = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
            o = np.zeros(X1.height)
            for f in range(5):
                tr1, va1 = f1 != f, f1 == f
                parts_X, parts_y, groups = [P1[tr1]], [X1["y"].to_numpy()[tr1]], [np.unique(X1["u"].to_numpy()[tr1], return_counts=True)[1]]
                if variant == "F1+F3":
                    tr3 = f3 != f
                    parts_X.append(P3[tr3])
                    parts_y.append(X3["y"].to_numpy()[tr3])
                    groups.append(np.unique(X3["u"].to_numpy()[tr3], return_counts=True)[1])
                import pandas as pd
                ds = lgb.Dataset(pd.concat(parts_X), np.concatenate(parts_y), group=np.concatenate(groups),
                                 categorical_feature=[c for c in CATS if c in cols])
                m = lgb.train(p, ds, num_boost_round=250)
                o[va1] = m.predict(P1[va1])
            oof += rank_norm(X1, o)
        pred = top5_from_scores(X1, oof)
        res[variant] = {"map5_all": round(mapk(gt1, pred, 5, users), 5), "map5_sub": round(mapk(gt1, pred, 5, sub_eval), 5)}
        print(variant, res[variant], f"{time.time()-t0:.0f}s", flush=True)
    res["n_F3_train_users"] = len(gt3)
    (ROOT / "evidence" / "experiments4_augment_F1.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
