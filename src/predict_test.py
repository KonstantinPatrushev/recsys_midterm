"""Train on the validation fold F1 and write a Kaggle submission for weeks 31-42.

Test features use the 8 most recent *clean* weeks (15-19, 28-30): the same window length
as the training fold F1 (11-18), skipping weeks 20-27 with broken logging.  Users with no
views in those weeks fall back to the F1 window 11-18.

Run: .venv/bin/python -I src/predict_test.py --name sub02 --ensemble ens3 [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from features import build, label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from ranker import CATS, PARAMS, feature_cols, to_pandas, top5_from_scores  # noqa: E402

TEST_HIST = [15, 16, 17, 18, 19, 28, 29, 30]
TEST_TARGET = list(range(31, 43))
F1_HIST, F1_TARGET = list(range(11, 19)), list(range(19, 31))

L31 = dict(PARAMS, num_leaves=31)
ENSEMBLES = {
    # name -> list of (objective params, label, rounds)
    "v1": [(PARAMS, "y", 400)],
    "lr31": [(L31, "y", 400)],
    "ens3": [(L31, "y", 400), (dict(L31, objective="binary", metric="auc"), "y", 400),
             (dict(L31, objective="poisson", metric="poisson"), "cnt_c", 400)],
}


def rank_within_user(X: pl.DataFrame, s: np.ndarray) -> np.ndarray:
    return X.select("u").with_columns(pl.Series("s", s)).with_columns(
        (pl.col("s").rank() / pl.len()).over("u").alias("r"))["r"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ensemble", default="ens3", choices=list(ENSEMBLES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--table", default="hybrid", choices=["hybrid", "base"])
    args = ap.parse_args()
    t0 = time.time()

    views = load_views("sum")
    sched = load_schedule()
    sub = submission_users()
    sub_u = sub["u"].to_list()

    def table(hist, target, with_labels):
        if args.table == "hybrid":
            return make_table(views, sched, hist, target, with_labels=with_labels)
        return build(views, sched, hist, target, with_labels=with_labels)

    gt = ground_truth(views, (19, 30))
    Xtr = label_top5(table(F1_HIST, F1_TARGET, True), gt)
    Xtr = Xtr.filter(pl.col("u").is_in(pl.Series(list(gt.keys())).implode())).sort(["u", "tv_show_id"])
    Xtr = Xtr.with_columns(pl.col("cnt").clip(0, 30).alias("cnt_c"))
    cols = [c for c in feature_cols(Xtr) if c != "cnt_c"]

    Xte = table(TEST_HIST, TEST_TARGET, False).filter(pl.col("u").is_in(pl.Series(sub_u).implode()))
    have = set(Xte["u"].to_list())
    no_hist = [u for u in sub_u if u not in have]
    if no_hist:
        Xfb = table(F1_HIST, TEST_TARGET, False).filter(pl.col("u").is_in(pl.Series(no_hist).implode()))
        Xte = pl.concat([Xte, Xfb.select(Xte.columns)])
    Xte = Xte.sort(["u", "tv_show_id"])
    assert not [c for c in cols if c not in Xte.columns]

    Xp_tr, Xp_te = to_pandas(Xtr, cols), to_pandas(Xte, cols)
    for c in CATS:
        cats = sorted(set(Xp_tr[c].astype(str)) | set(Xp_te[c].astype(str)))
        Xp_tr[c] = Xp_tr[c].astype(str).astype("category").cat.set_categories(cats)
        Xp_te[c] = Xp_te[c].astype(str).astype("category").cat.set_categories(cats)
    group = np.unique(Xtr["u"].to_numpy(), return_counts=True)[1]
    score = np.zeros(Xte.height)
    n_models = 0
    for params, label, rounds in ENSEMBLES[args.ensemble]:
        is_rank = params["objective"] in ("lambdarank", "rank_xendcg")
        seeds = range(args.seeds)
        for seed in seeds:
            p = dict(params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            ds = lgb.Dataset(Xp_tr, Xtr[label].to_numpy(), group=group if is_rank else None,
                             categorical_feature=[c for c in CATS if c in cols])
            m = lgb.train(p, ds, num_boost_round=rounds)
            score += rank_within_user(Xte, m.predict(Xp_te))
            n_models += 1
    score /= n_models
    pred = top5_from_scores(Xte, score)

    pop = (views.filter(pl.col("week").is_in(pl.Series(TEST_HIST).implode()) & pl.col("full"))
           .group_by("tv_show_id").agg(pl.col("u").n_unique().alias("n")).join(
               sched.filter(pl.col("week").is_between(31, 42)).select("tv_show_id").unique(), on="tv_show_id")
           .sort(["n", "tv_show_id"], descending=[True, False])["tv_show_id"].to_list())
    n_filled = 0
    rows = []
    for uid, u in sub.select(["user_id", "u"]).iter_rows():
        items = pred.get(u, [])
        if len(items) < 5:
            n_filled += 1
            items = items + [x for x in pop if x not in items][: 5 - len(items)]
        assert len(set(items)) == 5
        rows.append((uid, " ".join(str(int(x)) for x in items)))
    out = pl.DataFrame(rows, schema={"user_id": pl.UInt64, "tv_show_id": pl.Utf8}, orient="row")
    path = ROOT / "submissions" / f"{args.name}.csv"
    out.write_csv(path)
    # keep the per-candidate scores for later blending / analysis
    Xte.select(["u", "tv_show_id"]).with_columns(pl.Series("score", score)).write_parquet(
        ROOT / "cache" / f"test_scores_{args.name}.parquet")

    info = {
        "name": args.name, "ensemble": args.ensemble, "table": args.table, "models": n_models,
        "train_rows": Xtr.height, "train_users": len(gt), "test_rows": Xte.height,
        "test_users_with_candidates": Xte["u"].n_unique(), "users_filled_from_pop": n_filled,
        "users_with_fallback_history_11_18": len(no_hist), "features": cols, "test_hist_weeks": TEST_HIST,
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "evidence" / f"{args.name}_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
