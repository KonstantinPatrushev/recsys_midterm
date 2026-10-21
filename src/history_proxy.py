"""Which 8 history weeks to use for the test?  Proxy on the clean target weeks 28-30:
model trained on F1 rows of half A of the users, scored on half B (same split as
src/robustness.py), for several 8-week history windows.

Run: .venv/bin/python -I src/history_proxy.py
"""
from __future__ import annotations

import json
import sys
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import rank_norm  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import mapk  # noqa: E402
from predict_test import L31  # noqa: E402
from ranker import CATS, feature_cols, to_pandas, top5_from_scores, user_folds  # noqa: E402

HISTORIES = {
    "clean8_12_19": list(range(12, 20)),
    "contig8_20_27": list(range(20, 28)),
    "recent_skip_worst_17_24": list(range(17, 25)),
    "mixed_12_14_20_24": [12, 13, 14, 20, 21, 22, 23, 24],
}


def main():
    views, sched = load_views("sum"), load_schedule()
    sub_u = set(submission_users()["u"].to_list())
    gt1 = ground_truth(views, (19, 30))
    gt3 = ground_truth(views, (28, 30))
    X1 = label_top5(make_table(views, sched, list(range(11, 19)), list(range(19, 31))), gt1)
    X1 = X1.filter(pl.col("u").is_in(pl.Series(list(gt1)).implode())).sort(["u", "tv_show_id"])
    cols = feature_cols(X1)
    allu = np.array(sorted(set(X1["u"].to_list()) | set(gt3)))
    fmap = user_folds(allu, 2, 0)
    trainA = [u for u in X1["u"].unique().to_list() if fmap[u] == 0]
    evalB = [u for u in gt3 if fmap.get(u, 0) == 1]
    evalB_sub = [u for u in evalB if u in sub_u]
    Xa = X1.filter(pl.col("u").is_in(pl.Series(trainA).implode()))
    g = np.unique(Xa["u"].to_numpy(), return_counts=True)[1]
    out = {"n_evalB": len(evalB), "n_evalB_sub": len(evalB_sub)}
    for name, hw in HISTORIES.items():
        Xb = make_table(views, sched, hw, [28, 29, 30], with_labels=False)
        Xb = Xb.filter(pl.col("u").is_in(pl.Series(evalB).implode())).sort(["u", "tv_show_id"])
        Pa, Pb = to_pandas(Xa, cols), to_pandas(Xb, cols)
        for c in CATS:
            cats = sorted(set(Pa[c].astype(str)) | set(Pb[c].astype(str)))
            Pa[c] = Pa[c].astype(str).astype("category").cat.set_categories(cats)
            Pb[c] = Pb[c].astype(str).astype("category").cat.set_categories(cats)
        sc = np.zeros(Xb.height)
        for s in range(3):
            p = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
            m = lgb.train(p, lgb.Dataset(Pa, Xa["y"].to_numpy(), group=g, categorical_feature=CATS), 250)
            sc += rank_norm(Xb, m.predict(Pb))
        pr = top5_from_scores(Xb, sc)
        out[name] = {"weeks": hw, "map5_evalB": round(mapk(gt3, pr, 5, evalB), 5),
                     "map5_evalB_sub": round(mapk(gt3, pr, 5, evalB_sub), 5)}
        print(name, out[name], flush=True)
    (ROOT / "evidence" / "history_proxy.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
