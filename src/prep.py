"""Raw Kaggle CSV -> parquet tables used by every model.

Outputs (data/processed/):
  schedule.parquet   one row per airing with tv_show_id != 0 (organisers drop 0s before scoring)
  sessions.parquet   viewing sessions with integer user / channel codes
  views.parquet      one row per (user, airing) the user watched: seconds watched (sum over
                     sessions), longest single-session overlap, screen-time ratios
  users.parquet      user code <-> original user_id

Run: .venv/bin/python -I src/prep.py   (from midterm/)
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from intervals import overlap_pairs  # noqa: E402

SCHEDULE_FILES = ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]
LOG_FILE = "dataset11-30.csv"


WEEK11_START = 1583712000  # 2020-03-09 00:00:00 (naive timestamps read as UTC), Monday of ISO week 11


def to_epoch_seconds(col: pl.Expr) -> pl.Expr:
    return col.dt.epoch(time_unit="s")


def week_of(ts: pl.Expr) -> pl.Expr:
    return ((ts - WEEK11_START) // (7 * 86400) + 11).cast(pl.Int32)


def load_schedule() -> tuple[pl.DataFrame, dict]:
    frames = []
    stats = {}
    for f in SCHEDULE_FILES:
        df = pl.read_csv(RAW / f, infer_schema_length=0)  # all strings, cast explicitly below
        df = df.with_columns(
            pl.col("channel_id").cast(pl.Int64),
            pl.col("start_time").str.strptime(pl.Datetime("us"), "%d.%m.%Y %H:%M:%S"),
            pl.col("duration").cast(pl.Int64),
            pl.col("tv_show_id").cast(pl.Int64),
            pl.lit(f).alias("src_file"),
        )
        stats[f] = {"rows": df.height, "zero_show_rows": int((df["tv_show_id"] == 0).sum()), "columns": df.columns}
        frames.append(df)
    sched = pl.concat(frames, how="diagonal_relaxed")
    sched = sched.filter(pl.col("tv_show_id") != 0)
    sched = sched.with_columns(
        to_epoch_seconds(pl.col("start_time")).alias("start_ts"),
    ).with_columns(
        (pl.col("start_ts") + pl.col("duration")).alias("end_ts"),
        week_of(pl.col("start_ts")).alias("week"),
    )
    sched = sched.sort(["channel_id", "start_ts", "end_ts"]).with_row_index("airing_id")
    return sched, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = {}

    sched, sstats = load_schedule()
    report["schedule_files"] = sstats
    report["schedule_rows_nonzero"] = sched.height

    # header is "user_id","vsetv_id","start_time","stop_time","duraton" (sic); user_id exceeds int64
    logs = pl.read_csv(RAW / LOG_FILE, infer_schema_length=0)
    report["log_rows"] = logs.height
    report["log_columns"] = logs.columns
    logs = logs.rename({"stop_time": "end_time", "duraton": "duration"}).with_columns(
        pl.col("user_id").cast(pl.UInt64),
        pl.col("vsetv_id").cast(pl.Int64),
        pl.col("start_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"),
        pl.col("end_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"),
        pl.col("duration").cast(pl.Int64),
    )
    logs = logs.with_columns(
        to_epoch_seconds(pl.col("start_time")).alias("s_ts"),
        to_epoch_seconds(pl.col("end_time")).alias("e_ts"),
    ).with_columns(week_of(pl.col("s_ts")).alias("week"))

    users = logs.select("user_id").unique().sort("user_id").with_row_index("u")
    channels = (
        pl.concat([sched.select(pl.col("channel_id")), logs.select(pl.col("vsetv_id").alias("channel_id"))])
        .unique()
        .sort("channel_id")
        .with_row_index("c")
    )
    logs = logs.join(users, on="user_id").join(channels, left_on="vsetv_id", right_on="channel_id")
    sched = sched.join(channels, on="channel_id").sort(["c", "start_ts", "end_ts"])
    # airing_id must follow the (c, start) order used by the join
    sched = sched.drop("airing_id").with_row_index("airing_id")

    si, ai, ov = overlap_pairs(
        logs["c"].to_numpy().astype(np.int64),
        logs["s_ts"].to_numpy().astype(np.int64),
        logs["e_ts"].to_numpy().astype(np.int64),
        sched["c"].to_numpy().astype(np.int64),
        sched["start_ts"].to_numpy().astype(np.int64),
        sched["end_ts"].to_numpy().astype(np.int64),
    )
    report["session_airing_pairs"] = int(len(si))

    pairs = pl.DataFrame({
        "u": logs["u"].to_numpy()[si],
        "airing_id": ai.astype(np.uint32),
        "overlap": ov.astype(np.int64),
    })
    views = (
        pairs.group_by(["u", "airing_id"])
        .agg(
            pl.col("overlap").sum().alias("watched"),
            pl.col("overlap").max().alias("watched_max"),
            pl.len().alias("n_sessions"),
        )
        .join(sched.select(["airing_id", "tv_show_id", "c", "start_ts", "duration", "week"]), on="airing_id")
        .with_columns(
            (pl.col("watched") / pl.col("duration")).alias("screen"),
            (pl.col("watched_max") / pl.col("duration")).alias("screen_max"),
        )
    )
    report["user_airing_views"] = views.height
    report["full_views_sum_ge_0.8"] = int((views["screen"] >= 0.8).sum())
    report["full_views_max_ge_0.8"] = int((views["screen_max"] >= 0.8).sum())

    sched.write_parquet(OUT / "schedule.parquet")
    logs.select(["u", "c", "s_ts", "e_ts", "duration", "week"]).write_parquet(OUT / "sessions.parquet")
    views.write_parquet(OUT / "views.parquet")
    users.write_parquet(OUT / "users.parquet")
    channels.write_parquet(OUT / "channels.parquet")
    report["seconds"] = round(time.time() - t0, 1)
    (ROOT / "evidence" / "prep.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
