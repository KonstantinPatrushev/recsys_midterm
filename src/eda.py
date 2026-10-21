"""Exploratory checks that decide the validation design. Writes evidence/eda.json.

Run: .venv/bin/python -I src/eda.py   (from midterm/)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

sched = pl.read_parquet(P / "schedule.parquet")
sess = pl.read_parquet(P / "sessions.parquet")
views = pl.read_parquet(P / "views.parquet")
users = pl.read_parquet(P / "users.parquet")
sub = pl.read_csv(RAW / "submission.csv", schema_overrides={"user_id": pl.UInt64})
sub = sub.join(users, on="user_id", how="left")

out = {}
out["n_users_logs"] = users.height
out["n_sub_users"] = sub.height
out["n_sub_users_in_logs"] = int(sub["u"].is_not_null().sum())
out["n_channels_logs"] = sess["c"].n_unique()
out["n_channels_schedule_nonzero"] = sched["c"].n_unique()
out["weeks_logs"] = [int(sess["week"].min()), int(sess["week"].max())]
out["weeks_schedule"] = sched.group_by("week").len().sort("week").to_dicts()

# sessions on channels that have (non-zero) schedule
ch_sched = set(sched["c"].unique().to_list())
out["share_sessions_on_scheduled_channels"] = float(sess["c"].is_in(list(ch_sched)).mean())
sd = sess["duration"].to_numpy()
out["session_duration_quantiles_s"] = {str(q): float(np.quantile(sd, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]}
out["session_duration_vs_end_minus_start_mismatch"] = int(((sess["e_ts"] - sess["s_ts"]) != sess["duration"]).sum())

# overlapping sessions of one user (multi-device or duplicates)
s2 = sess.sort(["u", "s_ts"]).with_columns(
    pl.col("e_ts").shift(1).over("u").alias("prev_e"), pl.col("c").shift(1).over("u").alias("prev_c")
)
out["sessions_overlapping_previous_same_user"] = int((s2["s_ts"] < s2["prev_e"]).sum())
out["sessions_contiguous_same_channel_gap_le_5s"] = int(
    ((s2["s_ts"] - s2["prev_e"]).is_between(0, 5) & (s2["c"] == s2["prev_c"])).sum()
)

# programme durations / schedule overlaps inside a channel
sc = sched.sort(["c", "start_ts"]).with_columns(pl.col("end_ts").shift(1).over("c").alias("prev_end"))
out["schedule_overlaps_within_channel"] = int((sc["start_ts"] < sc["prev_end"]).sum())
out["airing_duration_quantiles_s"] = {str(q): float(np.quantile(sched["duration"].to_numpy(), q)) for q in [0.1, 0.5, 0.9, 0.99]}
out["n_shows_total"] = sched["tv_show_id"].n_unique()
w_tr = sched.filter(pl.col("week") <= 30)["tv_show_id"].unique()
w_te = sched.filter(pl.col("week") >= 31)["tv_show_id"].unique()
out["n_shows_train_weeks"] = w_tr.len()
out["n_shows_test_weeks"] = w_te.len()
out["n_shows_test_not_in_train"] = int((~w_te.is_in(w_tr.implode())).sum())

# timezone alignment: how often do session starts fall right after an airing start (any channel) for offsets
air_starts = np.unique(sched["start_ts"].to_numpy())
ss = sess["s_ts"].to_numpy()
align = {}
for off_h in [-3, -2, -1, 0, 1, 2, 3]:
    t = ss + off_h * 3600
    idx = np.searchsorted(air_starts, t, side="right") - 1
    ok = idx >= 0
    gap = t[ok] - air_starts[idx[ok]]
    align[str(off_h)] = float(np.mean(gap < 120))
out["share_session_starts_within_2min_after_some_airing_start_by_offset_h"] = align

# full views per week and the per-user target in a 12-week window (weeks 19-30), "sum" variant
v = views.with_columns((pl.col("screen") >= 0.8).alias("full"), (pl.col("screen_max") >= 0.8).alias("full_max"))
out["full_views_per_week"] = v.filter(pl.col("full")).group_by("week").len().sort("week").to_dicts()


def target(vv: pl.DataFrame, w0: int, w1: int, col: str) -> pl.DataFrame:
    return (
        vv.filter(pl.col("week").is_between(w0, w1) & pl.col(col))
        .group_by(["u", "tv_show_id"]).len("cnt")
    )


for col in ["full", "full_max"]:
    t = target(v, 19, 30, col)
    per_user = t.group_by("u").agg(pl.len().alias("n_shows"), pl.col("cnt").sort(descending=True).alias("cnts"))
    n5 = per_user.filter(pl.col("n_shows") >= 5)
    # ties at the 5th place: users whose 5th and 6th counts are equal
    ties = n5.filter(pl.col("n_shows") >= 6).with_columns(
        (pl.col("cnts").list.get(4) == pl.col("cnts").list.get(5)).alias("tie56")
    )
    out[f"val19_30_{col}"] = {
        "users_with_any": per_user.height,
        "users_with_ge5_shows": n5.height,
        "share_users_with_tie_at_5th": float(ties["tie56"].sum() / max(n5.height, 1)),
        "median_cnt_1st": float(n5["cnts"].list.get(0).median()),
        "median_cnt_5th": float(n5["cnts"].list.get(4).median()),
    }

# submission users: activity in train weeks vs everybody
act = sess.group_by("u").agg(
    pl.col("duration").sum().alias("secs"), pl.col("week").n_unique().alias("n_weeks"), pl.col("week").max().alias("last_week")
)
fullu = v.filter(pl.col("full")).group_by("u").agg(pl.len().alias("n_full"), pl.col("tv_show_id").n_unique().alias("n_full_shows"))
act = act.join(fullu, on="u", how="left").fill_null(0)
subset = act.filter(pl.col("u").is_in(sub["u"].drop_nulls().implode()))
rest = act.filter(~pl.col("u").is_in(sub["u"].drop_nulls().implode()))


def desc(df):
    return {c: {q: float(df[c].quantile(float(q))) for q in ["0.1", "0.5", "0.9"]} for c in ["secs", "n_weeks", "last_week", "n_full", "n_full_shows"]}


out["sub_users_activity"] = desc(subset)
out["other_users_activity"] = desc(rest)
out["sub_users_last_week_30_share"] = float((subset["last_week"] == 30).mean())
out["other_users_last_week_30_share"] = float((rest["last_week"] == 30).mean())
# would the submission users qualify as ">=5 full shows" users in weeks 19-30?
t = target(v, 19, 30, "full").group_by("u").len("n_shows")
q = sub.join(t, on="u", how="left").fill_null(0)
out["sub_users_with_ge5_full_shows_in_weeks_19_30"] = float((q["n_shows"] >= 5).mean())
oth = act.join(t, on="u", how="left").fill_null(0)
out["all_users_with_ge5_full_shows_in_weeks_19_30"] = int((oth["n_shows"] >= 5).sum())

(ROOT / "evidence" / "eda.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
