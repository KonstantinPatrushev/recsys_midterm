"""Candidate generation + features for the (user, show) ranker.

A fold is described by the list of history weeks (features) and target weeks (labels /
future schedule).  Only past viewing and the programme schedule are used: the target
window contributes nothing but its schedule (allowed metadata) and, for validation
folds, the labels.

Run as a script to cache candidate tables: .venv/bin/python -I src/features.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users, target_counts  # noqa: E402

CACHE = ROOT / "cache"

N_POP = 30           # popular shows added to every user's pool
N_SLOT = 30          # best slot-habit shows added per user
PARTIAL = 0.2        # screen ratio that counts as "sampled the show"
FULL_THR = 0.8       # organisers' "high screen time"
N_WEEK_FEATS = 8     # per-week full-view counts for the last 8 history weeks


def dow_hour(ts: pl.Expr):
    dt = pl.from_epoch(ts, time_unit="s")
    return dt.dt.weekday().cast(pl.Int8).alias("dow"), dt.dt.hour().cast(pl.Int8).alias("hour")


def build(views: pl.DataFrame, sched: pl.DataFrame, hist_weeks: list[int], target_weeks: list[int],
          users: list[int] | None = None, with_labels: bool = True,
          extra_cands: pl.DataFrame | None = None, extra_feats: pl.DataFrame | None = None) -> pl.DataFrame:
    """extra_cands: (u, tv_show_id) pairs to add to the pool (e.g. top-K of PPR/ALS/AR);
    extra_feats: (u, tv_show_id, ...) method scores joined as features (missing -> 0)."""
    t0 = time.time()
    hist_weeks = sorted(hist_weeks)
    hw = pl.Series(hist_weeks)
    n_hw = len(hist_weeks)
    # recency index: 0 = last history week, 1 = the one before, ... (skips excluded weeks)
    rec_map = pl.DataFrame({"week": hist_weeks[::-1], "rec": list(range(n_hw))}).with_columns(
        pl.col("week").cast(pl.Int32), pl.col("rec").cast(pl.Int32))

    vh = views.filter(pl.col("week").is_in(hw.implode())).join(rec_map, on="week")
    if users is not None:
        vh = vh.filter(pl.col("u").is_in(pl.Series(users).implode()))
    sh = sched.filter(pl.col("week").is_in(hw.implode()))
    st = sched.filter(pl.col("week").is_in(pl.Series(sorted(target_weeks)).implode()))

    # ---------------- show-level: schedule in history and in target window -------------
    air_h = sh.group_by("tv_show_id").agg(
        pl.len().alias("air_h"), pl.col("duration").median().alias("dur_med_h"),
        pl.col("c").n_unique().alias("n_ch_h"))
    air_t = st.group_by("tv_show_id").agg(
        pl.len().alias("air_t"), pl.col("duration").median().alias("dur_med_t"),
        pl.col("duration").sum().alias("air_secs_t"), pl.col("c").n_unique().alias("n_ch_t"),
        pl.col("week").n_unique().alias("weeks_t"),
        pl.col("c").mode().min().alias("main_c_t"),
    )
    # airings in the first/last weeks of the target window (is the show starting or ending?)
    tw = sorted(target_weeks)
    air_t = air_t.join(
        st.filter(pl.col("week").is_in(pl.Series(tw[:2]).implode())).group_by("tv_show_id").len("air_t_first2"),
        on="tv_show_id", how="left").join(
        st.filter(pl.col("week").is_in(pl.Series(tw[-2:]).implode())).group_by("tv_show_id").len("air_t_last2"),
        on="tv_show_id", how="left").fill_null(0)
    meta = (
        sched.sort("airing_id").group_by("tv_show_id", maintain_order=True).agg(
            pl.col("tv_show_category").first(), pl.col("tv_show_genre_1").first(),
            pl.col("year_of_production").first(),
        )
    )

    full = vh.filter(pl.col("full"))
    pop = full.group_by("tv_show_id").agg(
        pl.col("u").n_unique().alias("pop_users"), pl.len().alias("pop_full"),
        pl.col("u").filter(pl.col("rec") < 2).n_unique().alias("pop_users_l2"),
        (pl.col("rec") < 2).sum().alias("pop_full_l2"),
    )
    n_users_h = max(full["u"].n_unique(), 1)

    # ---------------- user-show history ------------------------------------------------
    us = vh.group_by(["u", "tv_show_id"]).agg(
        pl.col("full").sum().cast(pl.Int32).alias("f_cnt"),
        (pl.col("full") & (pl.col("rec") < 1)).sum().cast(pl.Int32).alias("f_l1"),
        (pl.col("full") & (pl.col("rec") < 2)).sum().cast(pl.Int32).alias("f_l2"),
        (pl.col("full") & (pl.col("rec") < 4)).sum().cast(pl.Int32).alias("f_l4"),
        pl.col("week").filter(pl.col("full")).n_unique().cast(pl.Int32).alias("f_weeks"),
        pl.col("rec").filter(pl.col("full")).min().alias("f_rec_min"),
        pl.col("rec").filter(pl.col("full")).max().alias("f_rec_max"),
        (pl.col("scr") >= PARTIAL).sum().cast(pl.Int32).alias("p_cnt"),
        ((pl.col("scr") >= PARTIAL) & (pl.col("rec") < 2)).sum().cast(pl.Int32).alias("p_l2"),
        pl.len().cast(pl.Int32).alias("any_cnt"),
        pl.col("scr").clip(0, 1).mean().alias("scr_mean"),
        pl.col("watched").sum().alias("secs"),
        pl.col("c").n_unique().alias("n_ch_watched"),
        ((pl.col("scr") >= 0.5) & (pl.col("scr") < FULL_THR)).sum().cast(pl.Int32).alias("nm_cnt"),
        *[(pl.col("full") & (pl.col("rec") == k)).sum().cast(pl.Int16).alias(f"f_w{k}") for k in range(N_WEEK_FEATS)],
    )
    # airings of the show in history on the channels where the user watched it -> per-airing rate
    uch = vh.select(["u", "tv_show_id", "c"]).unique()
    air_hc = sh.group_by(["tv_show_id", "c"]).len("air_hc")
    air_tc = st.group_by(["tv_show_id", "c"]).len("air_tc")
    rate = (
        uch.join(air_hc, on=["tv_show_id", "c"], how="left").join(air_tc, on=["tv_show_id", "c"], how="left")
        .fill_null(0).group_by(["u", "tv_show_id"]).agg(
            pl.col("air_hc").sum().alias("air_h_userch"), pl.col("air_tc").sum().alias("air_t_userch"))
    )
    us = us.join(rate, on=["u", "tv_show_id"], how="left")

    # ---------------- user-level ---------------------------------------------------------
    sess_users = vh.group_by("u").agg(
        pl.col("full").sum().alias("u_full"), pl.col("tv_show_id").filter(pl.col("full")).n_unique().alias("u_shows"),
        pl.col("week").n_unique().alias("u_weeks"), pl.col("watched").sum().alias("u_secs"),
        (pl.col("full") & (pl.col("rec") < 2)).sum().alias("u_full_l2"),
    )

    # ---------------- slot habits: P(user fully watches what starts at (c, dow, hour)) ---
    dh = full.with_columns(*dow_hour(pl.col("start_ts")))
    slot_u = dh.group_by(["u", "c", "dow", "hour"]).agg(pl.col("week").n_unique().alias("slot_weeks"))
    slot_u = slot_u.with_columns((pl.col("slot_weeks") / n_hw).alias("p_slot"))
    ch_u = vh.group_by(["u", "c"]).agg(pl.col("watched").sum().alias("ch_secs"))
    ch_u = ch_u.join(sess_users.select(["u", "u_secs"]), on="u").with_columns(
        (pl.col("ch_secs") / pl.col("u_secs")).alias("ch_share")).select(["u", "c", "ch_share"])
    hour_u = dh.group_by(["u", "hour"]).len("h_full").join(
        dh.group_by("u").len("u_f"), on="u").with_columns((pl.col("h_full") / pl.col("u_f")).alias("hour_share")
    ).select(["u", "hour", "hour_share"])

    fut_slots = st.with_columns(*dow_hour(pl.col("start_ts"))).group_by(["tv_show_id", "c", "dow", "hour"]).len("n_fut")

    # show-specific slot habit: user's full-view rate of this show in each (c, dow, hour) slot
    # times the number of future airings of the show in that slot
    air_slot_h = sh.with_columns(*dow_hour(pl.col("start_ts"))).group_by(["tv_show_id", "c", "dow", "hour"]).len("n_air_slot")
    # rows are sorted before the float sum: hash joins return rows in a run-dependent order, and a
    # different summation order changed the last bits of exp_show_slot between runs
    show_slot = (
        dh.group_by(["u", "tv_show_id", "c", "dow", "hour"]).len("f_slot")
        .join(air_slot_h, on=["tv_show_id", "c", "dow", "hour"], how="left")
        .join(fut_slots, on=["tv_show_id", "c", "dow", "hour"], how="left").fill_null(0)
        .with_columns((pl.col("f_slot") / pl.col("n_air_slot").clip(1, None)).alias("r_slot"))
        .sort(["u", "tv_show_id", "c", "dow", "hour"])
        .group_by(["u", "tv_show_id"], maintain_order=True).agg(
            (pl.col("r_slot") * pl.col("n_fut")).sum().alias("exp_show_slot"),
            pl.col("n_fut").sum().alias("fut_in_user_slots"),
            pl.col("r_slot").max().alias("r_slot_max"),
        )
    )
    # structure of the target-window schedule
    st_dh = st.with_columns(*dow_hour(pl.col("start_ts")))
    tw_sorted = sorted(target_weeks)
    t_struct = st_dh.group_by("tv_show_id").agg(
        pl.col("dow").n_unique().alias("n_dows_t"),
        pl.col("hour").is_between(18, 23).mean().alias("prime_share_t"),
        pl.col("week").is_in(pl.Series(tw_sorted[:4]).implode()).sum().alias("air_t_first4"),
        pl.col("week").is_in(pl.Series(tw_sorted[-4:]).implode()).sum().alias("air_t_last4"),
    )
    # how completely do viewers watch the show (history, all users)
    show_comp = vh.group_by("tv_show_id").agg(
        pl.col("full").mean().alias("show_full_rate"), pl.col("scr").clip(0, 1).mean().alias("show_scr_mean"))
    u_conc = ch_u.group_by("u").agg(pl.len().alias("u_n_ch"), pl.col("ch_share").max().alias("u_top_ch_share"))

    # ---------------- candidates ---------------------------------------------------------
    airing_t = air_t.select("tv_show_id")
    cand_hist = us.filter(pl.col("p_cnt") > 0).select(["u", "tv_show_id"]).join(airing_t, on="tv_show_id")
    pop_t = pop.join(airing_t, on="tv_show_id").sort(["pop_users", "tv_show_id"], descending=[True, False])
    top_pop = pop_t.head(N_POP).select("tv_show_id")
    all_users = sess_users.select("u")
    cand_pop = all_users.join(top_pop, how="cross")
    # slot-habit candidates: expected full views from habits, for every show airing in target
    slot_score = (
        slot_u.join(fut_slots, on=["c", "dow", "hour"])
        .group_by(["u", "tv_show_id"]).agg((pl.col("p_slot") * pl.col("n_fut")).sum().alias("exp_slot"),
                                            pl.col("p_slot").max().alias("p_slot_max"))
    )
    cand_slot = (
        slot_score.sort(["u", "exp_slot", "tv_show_id"], descending=[False, True, False]).group_by("u", maintain_order=True)
        .head(N_SLOT).select(["u", "tv_show_id"])
    )
    parts = [cand_hist, cand_pop, cand_slot]
    if extra_cands is not None:
        parts.append(extra_cands.select(["u", "tv_show_id"]).cast({"u": cand_hist.schema["u"], "tv_show_id": pl.Int64})
                     .join(airing_t, on="tv_show_id"))
    cand = pl.concat(parts).unique()

    # ---------------- assemble feature table ---------------------------------------------
    X = (
        cand.join(us, on=["u", "tv_show_id"], how="left")
        .join(slot_score, on=["u", "tv_show_id"], how="left")
        .join(air_h, on="tv_show_id", how="left")
        .join(air_t, on="tv_show_id", how="left")
        .join(pop, on="tv_show_id", how="left")
        .join(sess_users, on="u", how="left")
        .join(meta, on="tv_show_id", how="left")
        .join(ch_u.rename({"c": "main_c_t"}), on=["u", "main_c_t"], how="left")
        .join(show_slot, on=["u", "tv_show_id"], how="left")
        .join(t_struct, on="tv_show_id", how="left")
        .join(show_comp, on="tv_show_id", how="left")
        .join(u_conc, on="u", how="left")
    )
    # hour share at the show's most common future hour
    main_hour = st.with_columns(*dow_hour(pl.col("start_ts"))).group_by("tv_show_id").agg(
        pl.col("hour").mode().min().alias("main_hour_t"))
    X = X.join(main_hour, on="tv_show_id", how="left").join(
        hour_u.rename({"hour": "main_hour_t"}), on=["u", "main_hour_t"], how="left")

    if extra_feats is not None:
        X = X.join(extra_feats.cast({"u": X.schema["u"], "tv_show_id": pl.Int64}), on=["u", "tv_show_id"], how="left")
    num_fill = [c for c, t in X.schema.items() if t.is_numeric() and c not in ("u", "tv_show_id")]
    X = X.with_columns([pl.col(c).fill_null(0) for c in num_fill])
    X = X.with_columns(
        (pl.col("f_cnt") / n_hw).alias("f_rate_wk"),
        (pl.col("f_cnt") / pl.col("air_h_userch").clip(1, None)).alias("f_rate_air"),
        (pl.col("f_cnt") / pl.col("air_h_userch").clip(1, None) * pl.col("air_t_userch")).alias("exp_rate_ch"),
        (pl.col("f_cnt") / pl.col("air_h").clip(1, None) * pl.col("air_t")).alias("exp_rate"),
        (pl.col("f_cnt") / pl.col("u_full").clip(1, None)).alias("f_share"),
        (pl.col("f_l2") / pl.col("u_full_l2").clip(1, None)).alias("f_share_l2"),
        (pl.col("air_t") / pl.col("air_h").clip(1, None) * n_hw / len(target_weeks)).alias("air_ratio"),
        (pl.col("pop_users") / n_users_h).alias("pop_share"),
        (pl.col("u_full") / n_hw).alias("u_full_wk"),
        (pl.col("air_h") == 0).cast(pl.Int8).alias("is_new"),
        pl.lit(n_hw).alias("n_hist_weeks"),
    )
    X = X.sort(["u", "tv_show_id"]).with_columns(
        pl.col("f_cnt").rank("ordinal", descending=True).over("u").alias("f_rank"),
        pl.col("exp_slot").rank("ordinal", descending=True).over("u").alias("slot_rank"),
        pl.col("exp_rate_ch").rank("ordinal", descending=True).over("u").alias("exp_rate_ch_rank"),
        pl.col("exp_show_slot").rank("ordinal", descending=True).over("u").alias("exp_show_slot_rank"),
    )
    for c in ["tv_show_category", "tv_show_genre_1"]:
        X = X.with_columns(pl.col(c).fill_null("NA").cast(pl.Categorical))
    X = X.with_columns(pl.col("year_of_production").str.slice(0, 4).cast(pl.Int32, strict=False).fill_null(0))

    if with_labels:
        cnt = target_counts(views, (min(target_weeks), max(target_weeks)))
        X = X.join(cnt, on=["u", "tv_show_id"], how="left").with_columns(pl.col("cnt").fill_null(0))
    X = X.sort(["u", "tv_show_id"])
    print(f"features: {X.height} rows, {X['u'].n_unique()} users, {time.time()-t0:.1f}s", flush=True)
    return X


def label_top5(X: pl.DataFrame, gt: dict) -> pl.DataFrame:
    rows = [(u, s, 5 - i) for u, items in gt.items() for i, s in enumerate(items)]
    lab = pl.DataFrame(rows, schema={"u": X.schema["u"], "tv_show_id": pl.Int64, "rel": pl.Int32}, orient="row")
    return X.join(lab, on=["u", "tv_show_id"], how="left").with_columns(
        pl.col("rel").fill_null(0), (pl.col("rel").fill_null(0) > 0).cast(pl.Int8).alias("y"))


def candidate_recall(X: pl.DataFrame, gt: dict, users) -> float:
    have = X.select(["u", "tv_show_id"]).group_by("u").agg(pl.col("tv_show_id")).iter_rows()
    cmap = {u: set(s) for u, s in have}
    found = total = 0
    for u in users:
        a = set(gt[u])
        found += len(a & cmap.get(u, set()))
        total += len(a)
    return found / total


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    views = load_views("sum")
    sched = load_schedule()
    sub_u = set(submission_users()["u"].to_list())
    info = {}
    for name, hist, target in [("F1", list(range(11, 19)), list(range(19, 31)))]:
        gt = ground_truth(views, (min(target), max(target)))
        X = build(views, sched, hist, target)
        X = label_top5(X, gt)
        users = list(gt.keys())
        info[name] = {
            "rows": X.height, "users_with_candidates": X["u"].n_unique(),
            "candidates_per_user_median": float(X.group_by("u").len()["len"].median()),
            "recall_all": candidate_recall(X, gt, users),
            "recall_sub": candidate_recall(X, gt, [u for u in users if u in sub_u]),
        }
        X.write_parquet(CACHE / f"X_{name}.parquet")
        print(name, info[name])
    (ROOT / "evidence" / "candidates.json").write_text(json.dumps(info, indent=2))
