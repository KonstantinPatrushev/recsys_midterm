"""Hybrid table: base candidates/features + course-method scores (PPR, AR, ALS, CBF)
as features and as extra candidates; ablation on F1 with user-grouped CV.

Run: .venv/bin/python -I src/hybrid.py
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
from features import build, candidate_recall, label_top5  # noqa: E402
from methods import Space, als_scores, ar_scores, cbf_scores, ppr_scores  # noqa: E402
from metrics import mapk  # noqa: E402
from ranker import cv  # noqa: E402

METHOD_CFG = {"ppr": 0.7, "ar_support": 20, "als": (64, 10.0, 1.0), "cbf": "raw"}


def method_matrices(views, sched, hist, target, users=None):
    space = Space(views, sched, hist, target)
    users = list(space.users) if users is None else users
    mats = {
        "ppr": ppr_scores(space, METHOD_CFG["ppr"], users=users)[0],
        "ar": ar_scores(space, METHOD_CFG["ar_support"], "max")[0],
        "als": als_scores(space, *METHOD_CFG["als"], iters=30),
        "cbf": cbf_scores(space, METHOD_CFG["cbf"]),
    }
    return space, mats


def global_ranks(space, M):
    """Rank (1 = best) of every show among shows airing in the target window, per user."""
    S = np.where(space.airing_mask[None, :], M, -np.inf)
    order = np.argsort(-S, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(S.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, S.shape[1] + 1)[None, :]
    return ranks


def method_candidates(space, mats, users, k=20):
    rows = []
    uset = np.array(users)
    ui = np.searchsorted(space.users, uset)
    for name, M in mats.items():
        S = np.where(space.airing_mask[None, :], M[ui], -np.inf)
        top = np.argpartition(-S, k, axis=1)[:, :k]
        for r, u in enumerate(uset):
            for j in top[r]:
                if np.isfinite(S[r, j]):
                    rows.append((int(u), int(space.shows[j])))
    return pl.DataFrame(rows, schema={"u": pl.Int64, "tv_show_id": pl.Int64}, orient="row").unique()


def add_method_scores(X, space, mats, ranks):
    u = X["u"].to_numpy().astype(np.int64)
    s = X["tv_show_id"].to_numpy()
    ui = np.searchsorted(space.users, u)
    si = np.clip(np.searchsorted(space.shows, s), 0, len(space.shows) - 1)
    valid = (space.shows[si] == s) & (space.users[np.clip(ui, 0, len(space.users) - 1)] == u)
    cols = []
    for name, M in mats.items():
        v = np.zeros(X.height, dtype=np.float32)
        v[valid] = M[ui[valid], si[valid]]
        r = np.full(X.height, len(space.shows) + 1, dtype=np.int32)
        r[valid] = ranks[name][ui[valid], si[valid]]
        cols += [pl.Series(f"m_{name}", v), pl.Series(f"m_{name}_rank", r)]
    return X.with_columns(cols)


def make_table(views, sched, hist, target, with_labels=True, users=None, extra_k=20, method_feats=True):
    t0 = time.time()
    space, mats = method_matrices(views, sched, hist, target)
    ranks = {n: global_ranks(space, M) for n, M in mats.items()}
    cand_users = list(space.users) if users is None else users
    extra = method_candidates(space, mats, cand_users, extra_k) if extra_k else None
    X = build(views, sched, hist, target, with_labels=with_labels, extra_cands=extra)
    if method_feats:
        X = add_method_scores(X, space, mats, ranks)
    print(f"hybrid table {X.height} rows in {time.time()-t0:.1f}s", flush=True)
    return X


def main():
    views = load_views("sum")
    sched = load_schedule()
    gt = ground_truth(views, (19, 30))
    sub_u = set(submission_users()["u"].to_list())
    users = list(gt.keys())
    sub_eval = [u for u in users if u in sub_u]
    hist, target = list(range(11, 19)), list(range(19, 31))
    res = {}
    for name, kw in [("base", dict(extra_k=0, method_feats=False)),
                     ("base+method_feats", dict(extra_k=0, method_feats=True)),
                     ("base+method_feats+method_cands20", dict(extra_k=20, method_feats=True))]:
        X = label_top5(make_table(views, sched, hist, target, **kw), gt)
        rec_all = candidate_recall(X, gt, users)
        rec_sub = candidate_recall(X, gt, sub_eval)
        Xs, oof, pred, imp = cv(X, gt)
        res[name] = {"map5_all": round(mapk(gt, pred, 5, users), 5), "map5_sub": round(mapk(gt, pred, 5, sub_eval), 5),
                     "recall_all": round(rec_all, 4), "recall_sub": round(rec_sub, 4), "rows": X.height,
                     "top_features": sorted(imp.items(), key=lambda kv: -kv[1])[:12]}
        print(name, {k: v for k, v in res[name].items() if k != "top_features"}, flush=True)
        if name.endswith("cands20"):
            X.write_parquet(ROOT / "cache" / "X_F1_hybrid.parquet")
    (ROOT / "evidence" / "hybrid_ablation_F1.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
