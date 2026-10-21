"""The four course methods as stand-alone recommenders (dense user x show score matrices).

All methods see only the history window (full views) plus programme metadata; the
target-window schedule is used for the "airs in the target window" filter and, in CBF,
as the description of where/when a show will be on air.

  CBF  cosine between a user profile and show vectors (category, genres, channel,
       hour-of-day and weekday/weekend of airings, duration, decade)
  AR   pairwise association rules A -> B mined from users' sets of fully watched
       shows (support = #users); score(u,B) = max confidence over A in u's history
  MF   implicit ALS (Hu, Koren, Volinsky 2008) on log-confidence of full-view counts
  PPR  personalized PageRank on the bipartite user-show graph (edge = log1p count)

Run: .venv/bin/python -I src/methods.py   (evaluates every method on F1)
"""
from __future__ import annotations

import json
import os
import sys
import pathlib
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from metrics import mapk  # noqa: E402


class Space:
    """Index spaces and the history count matrix for one fold."""

    def __init__(self, views, sched, hist_weeks, target_weeks):
        hw = pl.Series(sorted(hist_weeks)).implode()
        tw = pl.Series(sorted(target_weeks)).implode()
        self.vh = views.filter(pl.col("week").is_in(hw))
        self.sh = sched.filter(pl.col("week").is_in(hw))
        self.st = sched.filter(pl.col("week").is_in(tw))
        full = self.vh.filter(pl.col("full"))
        cnt = full.group_by(["u", "tv_show_id"]).len("n")
        users = np.sort(views["u"].unique().to_numpy())
        shows = np.sort(pl.concat([cnt["tv_show_id"], self.st["tv_show_id"]]).unique().to_numpy())
        self.users, self.shows = users, shows
        self.uidx = {u: i for i, u in enumerate(users)}
        self.sidx = {s: j for j, s in enumerate(shows)}
        ui = np.searchsorted(users, cnt["u"].to_numpy())
        si = np.searchsorted(shows, cnt["tv_show_id"].to_numpy())
        self.R = sp.csr_matrix((cnt["n"].to_numpy().astype(np.float32), (ui, si)), shape=(len(users), len(shows)))
        airing = np.unique(self.st["tv_show_id"].to_numpy())
        self.airing_mask = np.isin(shows, airing)


def topk_from_matrix(space: Space, S: np.ndarray, users, k=5, airing_only=True) -> dict:
    S = S.copy()
    if airing_only:
        S[:, ~space.airing_mask] = -np.inf
    out = {}
    for u in users:
        i = space.uidx[u]
        row = S[i]
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.argsort(-row[idx], kind="stable")]
        out[u] = [int(space.shows[j]) for j in idx if np.isfinite(row[j])]
    return out


# ------------------------------------------------------------------------------- CBF
def show_vectors(space: Space, sched_part: pl.DataFrame) -> sp.csr_matrix:
    """Row-normalised feature matrix for all shows, built from `sched_part` airings
    (history schedule for the profile, target schedule for scoring)."""
    s = sched_part.with_columns(
        pl.from_epoch("start_ts", time_unit="s").dt.hour().alias("hour"),
        (pl.from_epoch("start_ts", time_unit="s").dt.weekday() >= 6).cast(pl.Int8).alias("weekend"),
    )
    blocks = []
    # share of airings per channel / hour / weekend flag
    for col, prefix in [("c", "ch"), ("hour", "hr"), ("weekend", "we")]:
        t = s.group_by(["tv_show_id", col]).len("n").with_columns(
            (pl.col("n") / pl.col("n").sum().over("tv_show_id")).alias("w"),
            (pl.lit(prefix + "_") + pl.col(col).cast(pl.Utf8)).alias("feat"))
        blocks.append(t.select(["tv_show_id", "feat", "w"]))
    meta = s.group_by("tv_show_id").agg(
        pl.col("tv_show_category").drop_nulls().first(), pl.col("tv_show_genre_1").drop_nulls().first(),
        pl.col("tv_show_genre_2").drop_nulls().first(), pl.col("tv_show_genre_3").drop_nulls().first(),
        pl.col("duration").median().alias("dur"), pl.col("year_of_production").drop_nulls().first())
    m = meta.with_columns(
        ("cat_" + pl.col("tv_show_category").fill_null("NA")).alias("f_cat"),
        ("dur_" + (pl.col("dur") / 1800).floor().clip(0, 8).cast(pl.Int32).cast(pl.Utf8)).alias("f_dur"),
        ("dec_" + pl.col("year_of_production").str.slice(0, 3).fill_null("NA")).alias("f_dec"),
    )
    for col in ["f_cat", "f_dur", "f_dec"]:
        blocks.append(m.select(["tv_show_id", pl.col(col).alias("feat"), pl.lit(1.0).alias("w")]))
    for g in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        blocks.append(m.filter(pl.col(g).is_not_null()).select(
            ["tv_show_id", ("gen_" + pl.col(g)).alias("feat"), pl.lit(1.0).alias("w")]))
    F = pl.concat(blocks).filter(pl.col("tv_show_id").is_in(pl.Series(space.shows).implode()))
    feats = np.sort(F["feat"].unique().to_numpy())
    space.cbf_feats = feats
    rows = np.searchsorted(space.shows, F["tv_show_id"].to_numpy())
    cols = np.searchsorted(feats, F["feat"].to_numpy())
    M = sp.csr_matrix((F["w"].to_numpy().astype(np.float32), (rows, cols)), shape=(len(space.shows), len(feats)))
    return M


def cbf_scores(space: Space, weighting="log") -> np.ndarray:
    Mh = show_vectors(space, space.sh)
    feats_h = space.cbf_feats
    Mt = show_vectors(space, space.st)
    feats_t = space.cbf_feats
    # put both on the union feature space
    allf = np.union1d(feats_h, feats_t)

    def remap(M, f):
        M = M.tocoo()
        return sp.csr_matrix((M.data, (M.row, np.searchsorted(allf, f[M.col]))), shape=(M.shape[0], len(allf)))

    Mh, Mt = remap(Mh, feats_h), remap(Mt, feats_t)
    # idf over shows (history vectors), then L2-normalise rows
    df = np.asarray((Mh > 0).sum(axis=0)).ravel() + 1
    idf = np.log(Mh.shape[0] / df).astype(np.float32)

    def l2(M):
        M = M @ sp.diags(idf)
        n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()) + 1e-9
        return sp.diags(1 / n) @ M

    Mh, Mt = l2(Mh), l2(Mt)
    W = space.R.copy()
    if weighting == "log":
        W.data = np.log1p(W.data)
    prof = W @ Mh                               # users x feats
    n = np.sqrt(np.asarray(prof.multiply(prof).sum(axis=1)).ravel()) + 1e-9
    prof = sp.diags(1 / n) @ prof
    # shows without target airings fall back to their history vector
    no_t = np.asarray(Mt.sum(axis=1)).ravel() == 0
    Mt = Mt.tolil()
    Mt[no_t] = Mh[no_t]
    S = (prof @ Mt.tocsr().T).toarray().astype(np.float32)
    return S


# -------------------------------------------------------------------------------- AR
def ar_scores(space: Space, min_support=10, agg="max"):
    B = (space.R > 0).astype(np.float32).tocsc()      # users x shows, binary baskets
    supp = np.asarray(B.sum(axis=0)).ravel()           # #users per show
    C = (B.T @ B).tocsr()                              # co-occurrence (#users with both)
    C.setdiag(0)
    C.eliminate_zeros()
    C.data[C.data < min_support] = 0
    C.eliminate_zeros()
    # confidence(A -> B) = supp(A,B) / supp(A): divide row A by supp(A)
    conf = sp.diags(1 / np.maximum(supp, 1)) @ C
    U = (space.R > 0).astype(np.float32).tocsr()
    if agg == "max":
        S = np.zeros(space.R.shape, dtype=np.float32)
        conf = conf.tocsr()
        for i in range(U.shape[0]):
            items = U.indices[U.indptr[i]:U.indptr[i + 1]]
            if len(items):
                S[i] = conf[items].max(axis=0).toarray().ravel()
    else:
        W = space.R.copy()
        W.data = np.log1p(W.data)
        S = (W @ conf).toarray().astype(np.float32)
    lift_base = supp / max(len(space.users), 1)
    return S, {"n_rules": int(C.nnz), "min_support": min_support, "agg": agg, "max_item_support_share": float(lift_base.max())}


# -------------------------------------------------------------------------------- MF
def als_scores(space: Space, factors=64, reg=0.05, alpha=10.0, iters=20, seed=0):
    from implicit.cpu.als import AlternatingLeastSquares
    C = space.R.copy().astype(np.float32)
    C.data = alpha * np.log1p(C.data)
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters,
                                    random_state=seed, calculate_training_loss=False)
    model.fit(C, show_progress=False)
    return (model.user_factors @ model.item_factors.T).astype(np.float32)


# ------------------------------------------------------------------------------- PPR
def ppr_scores(space: Space, damping=0.5, iters=50, users=None, batch=1024, tol=1e-6):
    """Random walk with restart on the bipartite graph; returns user x show scores
    (rows for `users`, others 0).  damping = probability of continuing the walk."""
    W = space.R.copy().astype(np.float64)
    W.data = np.log1p(W.data)
    nU, nS = W.shape
    A = sp.bmat([[None, W], [W.T, None]]).tocsr()          # (nU+nS) square, symmetric
    deg = np.asarray(A.sum(axis=1)).ravel()
    Pt = (sp.diags(1 / np.maximum(deg, 1e-12)) @ A).T.tocsr()  # column-stochastic transpose
    dangling = deg == 0
    users = space.users if users is None else users
    rows = np.array([space.uidx[u] for u in users])
    S = np.zeros((nU, nS), dtype=np.float32)
    iters_used = []
    for b in range(0, len(rows), batch):
        r_idx = rows[b:b + batch]
        E = np.zeros((nU + nS, len(r_idx)))
        E[r_idx, np.arange(len(r_idx))] = 1.0
        X = E.copy()
        for it in range(iters):
            leak = X[dangling].sum(axis=0)                    # mass at dangling nodes restarts
            Xn = damping * (Pt @ X) + (1 - damping + damping * leak) * E
            delta = np.abs(Xn - X).sum(axis=0).max()
            X = Xn
            if delta < tol:
                break
        iters_used.append(it + 1)
        S[r_idx] = X[nU:].T.astype(np.float32)
    return S, {"damping": damping, "max_iters_used": int(max(iters_used))}


# ------------------------------------------------------------------ popularity / repeat
def pop_scores(space: Space):
    p = np.asarray((space.R > 0).sum(axis=0)).ravel().astype(np.float32)
    return np.tile(p, (space.R.shape[0], 1))


def repeat_scores(space: Space):
    S = space.R.toarray().astype(np.float32)
    # tie-break unseen items by popularity (tiny epsilon)
    p = np.asarray((space.R > 0).sum(axis=0)).ravel().astype(np.float32)
    return S + 1e-6 * p / max(p.max(), 1)


def evaluate_all(fold_hist, fold_target, out_name):
    views = load_views("sum")
    sched = load_schedule()
    gt = ground_truth(views, (min(fold_target), max(fold_target)))
    sub_u = set(submission_users()["u"].to_list())
    users = list(gt.keys())
    sub_eval = [u for u in users if u in sub_u]
    space = Space(views, sched, fold_hist, fold_target)
    res = {}

    def record(name, S, extra=None, t=None):
        p = topk_from_matrix(space, S, users)
        p_any = topk_from_matrix(space, S, users, airing_only=False)
        res[name] = {"map5_all": round(mapk(gt, p, 5, users), 5), "map5_sub": round(mapk(gt, p, 5, sub_eval), 5),
                     "map5_all_no_airing_filter": round(mapk(gt, p_any, 5, users), 5),
                     "seconds": round(t, 1) if t else None, **(extra or {})}
        print(f"{name:28s} {res[name]}", flush=True)

    t = time.time(); record("POP", pop_scores(space), t=time.time() - t)
    t = time.time(); record("REPEAT", repeat_scores(space), t=time.time() - t)
    for w in ["log", "raw"]:
        t = time.time(); S = cbf_scores(space, w); record(f"CBF_{w}", S, t=time.time() - t)
    for ms in [5, 20]:
        for agg in ["max", "sum"]:
            t = time.time(); S, ex = ar_scores(space, ms, agg); record(f"AR_s{ms}_{agg}", S, ex, time.time() - t)
    for f, reg, alpha in [(32, 0.05, 10.0), (64, 0.05, 10.0), (128, 0.1, 10.0), (64, 0.05, 40.0), (64, 0.5, 10.0)]:
        t = time.time(); S = als_scores(space, f, reg, alpha); record(f"ALS_f{f}_r{reg}_a{alpha}", S, t=time.time() - t)
    for d in [0.3, 0.5, 0.7, 0.85]:
        t = time.time(); S, ex = ppr_scores(space, d, users=users); record(f"PPR_d{d}", S, ex, time.time() - t)
    meta = {"hist": list(fold_hist), "target": list(fold_target), "n_eval_users": len(users),
            "n_eval_sub_users": len(sub_eval), "n_users": len(space.users), "n_shows": len(space.shows),
            "n_airing_shows": int(space.airing_mask.sum())}
    (ROOT / "evidence" / out_name).write_text(json.dumps({"meta": meta, "scores": res}, indent=2))


if __name__ == "__main__":
    evaluate_all(list(range(11, 19)), list(range(19, 31)), "methods_F1.json")
