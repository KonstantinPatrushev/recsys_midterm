"""User-grouped 5-fold CV on F1 for an exact submission configuration (same table, model,
rounds and seeds as src/predict_test.py) -> evidence/cv_<name>.json

Run: .venv/bin/python -I src/cv_final.py --name <name> --ensemble lr31_r250 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import rank_norm, run_cv  # noqa: E402
from features import build, label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import apk, mapk  # noqa: E402
from predict_test import ENSEMBLES  # noqa: E402
from ranker import feature_cols, top5_from_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ensemble", default="lr31_r250")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--table", default="hybrid", choices=["hybrid", "base"])
    ap.add_argument("--variant", default="sum", choices=["sum", "max"])
    ap.add_argument("--cols-file", default=None, help="evidence/<info>.json whose 'features' list restricts the columns")
    ap.add_argument("--agg", default="user", choices=["user", "global"],
                    help="seed averaging by within-user rank (submission 2+) or by global rank (submission 1)")
    args = ap.parse_args()
    views, sched = load_views(args.variant), load_schedule()
    gt = ground_truth(views, (19, 30))
    users = list(gt)
    sub_u = set(submission_users()["u"].to_list())
    sub_eval = [u for u in users if u in sub_u]
    hist, target = list(range(11, 19)), list(range(19, 31))
    X = make_table(views, sched, hist, target) if args.table == "hybrid" else build(views, sched, hist, target)
    X = label_top5(X, gt).filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
    X = X.with_columns(pl.col("cnt").clip(0, 30).alias("cnt_c"))
    cols = [c for c in feature_cols(X) if c != "cnt_c"]
    if args.cols_file:
        cols = json.loads((ROOT / "evidence" / args.cols_file).read_text())["features"]
    score = np.zeros(X.height)
    n = 0
    for params, label, rounds in ENSEMBLES[args.ensemble]:
        for seed in range(args.seeds):
            p = dict(params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            o = run_cv(X, gt, p, label=label, rounds=rounds, cols=cols)
            score += rank_norm(X, o) if args.agg == "user" else pl.Series(o).rank().to_numpy() / len(o)
            n += 1
    pred = top5_from_scores(X, score / n)
    aps = {u: apk(gt[u], pred.get(u, []), 5) for u in users}
    (ROOT / "cache" / f"cv_pred_{args.name}.json").write_text(json.dumps({str(u): [int(x) for x in v] for u, v in pred.items()}))
    res = {"name": args.name, "ensemble": args.ensemble, "seeds": args.seeds, "table": args.table, "variant": args.variant,
           "cols_file": args.cols_file, "agg": args.agg,
           "map5_all": round(mapk(gt, pred, 5, users), 5), "map5_sub": round(mapk(gt, pred, 5, sub_eval), 5),
           "n_all": len(users), "n_sub": len(sub_eval),
           "users_without_candidates": int(sum(1 for u in users if u not in pred))}
    (ROOT / "evidence" / f"cv_{args.name}.json").write_text(json.dumps(res, indent=2))
    pl.DataFrame({"u": list(aps), "ap5": list(aps.values())}).write_parquet(ROOT / "cache" / f"cv_aps_{args.name}.parquet")
    print(res)


if __name__ == "__main__":
    main()
