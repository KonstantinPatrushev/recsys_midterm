"""Robustness checks recommended by the research playbook -> evidence/robustness.json

1. RRF hybrid of the course methods (model-free): score = sum_m 1 / (60 + rank_m), on F1.
2. The ranker vs the best single method on other folds (F0: 11-15 -> 16-19; F2: 11-27 -> 28-30),
   each with its own user-grouped CV, to check that the gain is not an F1 artefact.
3. Tie-break sensitivity of the submitted configuration: OOF predictions scored against
   truth built with id_asc / id_desc / random tie-breaks.
4. Test-history proxy on F2-like target 28-30: model trained on F1, applied to the history
   "8 most recent clean weeks" (12-19) vs "8 contiguous recent weeks" (20-27), user-disjoint.

Run: .venv/bin/python -I src/robustness.py
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
from experiments import rank_norm, run_cv  # noqa: E402
from features import label_top5  # noqa: E402
from hybrid import global_ranks, make_table, method_matrices  # noqa: E402
from methods import repeat_scores, topk_from_matrix  # noqa: E402
from metrics import mapk  # noqa: E402
from predict_test import L31  # noqa: E402
from ranker import CATS, feature_cols, to_pandas, top5_from_scores, user_folds  # noqa: E402

SEEDS = 3


def ranker_oof(X, gt, cols):
    score = np.zeros(X.height)
    for s in range(SEEDS):
        p = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
        score += rank_norm(X, run_cv(X, gt, p, rounds=250, cols=cols))
    return score / SEEDS


def main():
    t0 = time.time()
    views, sched = load_views("sum"), load_schedule()
    sub_u = set(submission_users()["u"].to_list())
    out = {}

    # ---- 1 + 2: per fold: methods, RRF, ranker
    for name, hist, target in [("F1", list(range(11, 19)), list(range(19, 31))),
                               ("F0", list(range(11, 16)), list(range(16, 20))),
                               ("F2", list(range(11, 28)), list(range(28, 31)))]:
        gt = ground_truth(views, (min(target), max(target)))
        users = list(gt)
        sub_eval = [u for u in users if u in sub_u]
        space, mats = method_matrices(views, sched, hist, target)
        mats["repeat"] = repeat_scores(space)
        r = {}
        for m, M in mats.items():
            p = topk_from_matrix(space, M, users)
            r[m] = {"map5_all": round(mapk(gt, p, 5, users), 5), "map5_sub": round(mapk(gt, p, 5, sub_eval), 5)}
        ranks = {m: global_ranks(space, M) for m, M in mats.items()}
        for combo in [("ppr", "ar", "als", "cbf"), ("ppr", "ar", "als", "cbf", "repeat")]:
            S = sum(1.0 / (60.0 + ranks[m]) for m in combo)
            p = topk_from_matrix(space, S, users)
            r["RRF_" + "+".join(combo)] = {"map5_all": round(mapk(gt, p, 5, users), 5),
                                           "map5_sub": round(mapk(gt, p, 5, sub_eval), 5)}
        X = label_top5(make_table(views, sched, hist, target), gt)
        X = X.filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
        cols = feature_cols(X)
        pred = top5_from_scores(X, ranker_oof(X, gt, cols))
        r["hybrid_ranker"] = {"map5_all": round(mapk(gt, pred, 5, users), 5), "map5_sub": round(mapk(gt, pred, 5, sub_eval), 5)}
        out[name] = {"hist": [min(hist), max(hist)], "target": [min(target), max(target)],
                     "n_users": len(users), "n_sub": len(sub_eval), "scores": r}
        print(name, json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)

        if name == "F1":
            # ---- 3: tie-break sensitivity of the ranker's OOF (same predictions, other truths)
            tb = {}
            for tie in ["id_asc", "id_desc", "random1", "random2", "random3"]:
                g = ground_truth(views, (19, 30), tie=tie)
                tb[tie] = {"map5_all": round(mapk(g, pred, 5, list(g)), 5),
                           "map5_sub": round(mapk(g, pred, 5, [u for u in g if u in sub_u]), 5)}
            out["F1_tiebreak_sensitivity_hybrid"] = tb
            print("tie-break", json.dumps(tb), flush=True)
            X1, gt1, cols1 = X, gt, cols

    # ---- 4: test-history proxy, target 28-30, user-disjoint (train F1 on half A, evaluate half B)
    gt3 = ground_truth(views, (28, 30))
    allu = np.array(sorted(set(X1["u"].to_list()) | set(gt3)))
    fmap = user_folds(allu, 2, 0)
    trainA = [u for u in X1["u"].unique().to_list() if fmap[u] == 0]
    evalB = [u for u in gt3 if fmap.get(u, 0) == 1]
    Xa = X1.filter(pl.col("u").is_in(pl.Series(trainA).implode()))
    Pa = to_pandas(Xa, cols1)
    proxy = {}
    for hname, hw in [("clean8_12_19", list(range(12, 20))), ("contig8_20_27", list(range(20, 28)))]:
        Xb = make_table(views, sched, hw, [28, 29, 30], with_labels=False)
        Xb = Xb.filter(pl.col("u").is_in(pl.Series(evalB).implode())).sort(["u", "tv_show_id"])
        Pb = to_pandas(Xb, cols1)
        for c in CATS:
            cats = sorted(set(Pa[c].astype(str)) | set(Pb[c].astype(str)))
            Pa[c] = Pa[c].astype(str).astype("category").cat.set_categories(cats)
            Pb[c] = Pb[c].astype(str).astype("category").cat.set_categories(cats)
        g = np.unique(Xa["u"].to_numpy(), return_counts=True)[1]
        sc = np.zeros(Xb.height)
        for s in range(SEEDS):
            p = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
            m = lgb.train(p, lgb.Dataset(Pa, Xa["y"].to_numpy(), group=g, categorical_feature=CATS), 250)
            sc += rank_norm(Xb, m.predict(Pb))
        pr = top5_from_scores(Xb, sc)
        proxy[hname] = {"map5_evalB": round(mapk(gt3, pr, 5, evalB), 5),
                        "map5_evalB_sub": round(mapk(gt3, pr, 5, [u for u in evalB if u in sub_u]), 5),
                        "n_evalB": len(evalB)}
        print("proxy", hname, proxy[hname], flush=True)
    out["test_history_proxy_target_28_30"] = proxy
    (ROOT / "evidence" / "robustness.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
