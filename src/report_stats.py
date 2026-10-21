"""Numbers required by the updated task sheet -> evidence/report_stats.json

1. Matching rule and missing programme ids: how much session time overlaps airings with a
   tv_show_id, airings with tv_show_id = 0, or no schedule entry at all.
2. Every method on ONE candidate pool (the hybrid's candidates) on F1 / F0 / F2.
3. Computational cost: wall-clock seconds to fit and score all users on F1, per method.
4. The final configuration on F1: rank average of the sum-label and max-label rankers
   (5 seeds each) = the validation analogue of the designated final submission; its
   out-of-fold top-5 lists are saved for the two-user analysis.

Run: .venv/bin/python -I src/report_stats.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import P, RAW, ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import run_cv  # noqa: E402
from features import label_top5  # noqa: E402
from final_selection import blend, top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from intervals import overlap_pairs  # noqa: E402
from methods import Space, als_scores, ar_scores, cbf_scores, pop_scores, ppr_scores, repeat_scores  # noqa: E402
from metrics import mapk  # noqa: E402
from predict_test import L31  # noqa: E402
from ranker import feature_cols  # noqa: E402
from prep import SCHEDULE_FILES, to_epoch_seconds  # noqa: E402

out = {}
T = time.perf_counter


# ---------------------------------------------------------------- 1. matching coverage
def matching():
    sess = pl.read_parquet(P / "sessions.parquet")
    chans = pl.read_parquet(P / "channels.parquet")
    frames = []
    for f in SCHEDULE_FILES:
        df = pl.read_csv(RAW / f, infer_schema_length=0).select(
            pl.col("channel_id").cast(pl.Int64), pl.col("start_time").str.strptime(pl.Datetime("us"), "%d.%m.%Y %H:%M:%S"),
            pl.col("duration").cast(pl.Int64), pl.col("tv_show_id").cast(pl.Int64))
        frames.append(df)
    s = pl.concat(frames).with_columns(to_epoch_seconds(pl.col("start_time")).alias("st")).with_columns(
        (pl.col("st") + pl.col("duration")).alias("en")).join(chans, on="channel_id").sort(["c", "st", "en"])
    si, ai, ov = overlap_pairs(sess["c"].to_numpy().astype(np.int64), sess["s_ts"].to_numpy(), sess["e_ts"].to_numpy(),
                               s["c"].to_numpy().astype(np.int64), s["st"].to_numpy(), s["en"].to_numpy())
    zero = s["tv_show_id"].to_numpy()[ai] == 0
    total = int((sess["e_ts"] - sess["s_ts"]).sum())
    # overlapping airings inside a channel can count a second twice; the schedule has very few
    covered_nonzero, covered_zero = int(ov[~zero].sum()), int(ov[zero].sum())
    return {
        "schedule_rows_all": s.height, "schedule_rows_id0": int((s["tv_show_id"] == 0).sum()),
        "share_schedule_rows_id0": round(float((s["tv_show_id"] == 0).mean()), 4),
        "session_seconds_total": total,
        "share_session_time_on_nonzero_airings": round(covered_nonzero / total, 4),
        "share_session_time_on_id0_airings": round(covered_zero / total, 4),
        "share_session_time_without_schedule": round(max(total - covered_nonzero - covered_zero, 0) / total, 4),
        "sessions_on_channels_without_any_nonzero_airing": int((~sess["c"].is_in(
            s.filter(pl.col("tv_show_id") != 0)["c"].unique().implode())).sum()),
        "sessions_total": sess.height,
    }


t0 = T()
out["matching"] = matching()
print("matching", out["matching"], f"{T()-t0:.1f}s", flush=True)

views = {v: load_views(v) for v in ["sum", "max"]}
sched = load_schedule()
sub_u = set(submission_users()["u"].to_list())


def pool_topk(space, M, pool: pl.DataFrame, k=5):
    """top-k of score matrix M restricted to each user's candidate pool"""
    u = pool["u"].to_numpy().astype(np.int64)
    s = pool["tv_show_id"].to_numpy()
    ui = np.searchsorted(space.users, u)
    si = np.clip(np.searchsorted(space.shows, s), 0, len(space.shows) - 1)
    ok = space.shows[si] == s
    sc = np.full(len(u), -np.inf)
    sc[ok] = M[ui[ok], si[ok]]
    t = pool.select(["u", "tv_show_id"]).with_columns(pl.Series("s", sc))
    t = t.sort(["u", "s", "tv_show_id"], descending=[False, True, False]).group_by("u", maintain_order=True).head(k)
    return {u_: list(x) for u_, x in t.group_by("u", maintain_order=True).agg(pl.col("tv_show_id")).iter_rows()}


# ---------------------------------------------------------------- 2 + 3. common pool, cost
FOLDS = {"F1": (list(range(11, 19)), list(range(19, 31))),
         "F0": (list(range(11, 16)), list(range(16, 20))),
         "F2": (list(range(11, 28)), list(range(28, 31)))}
out["common_pool"] = {}
out["cost_seconds_F1"] = {}
for fname, (hist, target) in FOLDS.items():
    gt = ground_truth(views["sum"], (min(target), max(target)))
    users = list(gt)
    sub_eval = [u for u in users if u in sub_u]
    t0 = T()
    space = Space(views["sum"], sched, hist, target)
    t_space = T() - t0
    mats, cost = {}, {}
    for name, fn in [("Popularity (reference)", lambda: pop_scores(space)),
                     ("Repeat own history (reference)", lambda: repeat_scores(space)),
                     ("CBF", lambda: cbf_scores(space, "raw")),
                     ("Association rules", lambda: ar_scores(space, 20, "max")[0]),
                     ("MF (implicit ALS)", lambda: als_scores(space, 64, 10.0, 1.0, iters=30)),
                     ("PPR", lambda: ppr_scores(space, 0.7)[0])]:
        t0 = T()
        mats[name] = fn()
        cost[name] = round(T() - t0 + t_space, 1)
    t0 = T()
    X = label_top5(make_table(views["sum"], sched, hist, target), gt)
    X = X.filter(pl.col("u").is_in(pl.Series(users).implode())).sort(["u", "tv_show_id"])
    t_table = T() - t0
    res = {}
    for name, M in mats.items():
        p = pool_topk(space, M, X.select(["u", "tv_show_id"]))
        res[name] = {"map5_all": round(mapk(gt, p, 5, users), 5), "map5_sub": round(mapk(gt, p, 5, sub_eval), 5)}
    # ranker on the same pool: 5 seeds, out-of-fold
    cols = feature_cols(X)
    t0 = T()
    acc = np.zeros(X.height)
    for s in range(5):
        p_ = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
        o = run_cv(X, gt, p_, rounds=250, cols=cols)
        acc += X.select("u").with_columns(pl.Series("o", o)).select((pl.col("o").rank() / pl.len()).over("u"))["o"].to_numpy()
    t_rank = T() - t0
    hyb = X.select(["u", "tv_show_id"]).with_columns(pl.Series("s", acc / 5))
    p = top5(hyb)
    res["Hybrid LightGBM ranker (sum label)"] = {"map5_all": round(mapk(gt, p, 5, users), 5),
                                                 "map5_sub": round(mapk(gt, p, 5, sub_eval), 5)}
    res["_pool"] = {"rows": X.height, "median_candidates_per_user": float(X.group_by("u").len()["len"].median())}
    out["common_pool"][fname] = res
    if fname == "F1":
        cost["Hybrid: candidate/feature table incl. the four methods"] = round(t_table, 1)
        cost["Hybrid: LightGBM 5 seeds x 5 CV folds (train + predict)"] = round(t_rank, 1)
        out["cost_seconds_F1"] = cost
        # ---- 4. the final configuration: + max-label ranker, rank-averaged
        gtm = ground_truth(views["max"], (19, 30))
        Xm = label_top5(make_table(views["max"], sched, hist, target), gtm)
        Xm = Xm.filter(pl.col("u").is_in(pl.Series(list(gtm)).implode())).sort(["u", "tv_show_id"])
        accm = np.zeros(Xm.height)
        for s in range(5):
            p_ = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
            o = run_cv(Xm, gtm, p_, rounds=250, cols=feature_cols(Xm))
            accm += Xm.select("u").with_columns(pl.Series("o", o)).select((pl.col("o").rank() / pl.len()).over("u"))["o"].to_numpy()
        hm = Xm.select(["u", "tv_show_id"]).with_columns(pl.Series("s", accm / 5))
        fin = top5(blend(hyb, hm))
        out["final_config_F1"] = {
            "description": "rank average of the sum-label and max-label rankers, 5 seeds each (validation analogue of det_ens_AD)",
            "truth_sum_all": round(mapk(gt, fin, 5, users), 5), "truth_sum_sub": round(mapk(gt, fin, 5, sub_eval), 5),
            "truth_max_all": round(mapk(gtm, fin, 5, list(gtm)), 5),
            "truth_max_sub": round(mapk(gtm, fin, 5, [u for u in gtm if u in sub_u]), 5)}
        (ROOT / "cache" / "cv_pred_final_config.json").write_text(json.dumps({str(u): [int(x) for x in v] for u, v in fin.items()}))
        print("final config", out["final_config_F1"], flush=True)
    print(fname, json.dumps(res), flush=True)
(ROOT / "evidence" / "report_stats.json").write_text(json.dumps(out, indent=2))
print("cost", out["cost_seconds_F1"])
