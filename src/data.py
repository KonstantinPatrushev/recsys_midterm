"""Shared loaders: full views, ground truth for a week window, folds.

Ground truth follows the organisers' recipe (Kaggle data page):
  screen = watched seconds / airing duration, keep screen >= 0.8,
  count such airings per (user, tv_show_id), keep users with >= 5 shows,
  take the 5 most frequent.  The tie-break at the 5th place is not documented;
  we break ties by tv_show_id ascending (what a stable sort of a groupby(...).size()
  table would do) and measure the sensitivity separately.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

FULL = 0.8


@dataclass(frozen=True)
class Fold:
    name: str
    hist: tuple[int, int]        # inclusive week range used for features
    target: tuple[int, int]      # inclusive week range used for labels (None for test)


FOLDS = {
    # primary: 12-week horizon like the test; history = 8 clean weeks
    "F1": Fold("F1", (11, 18), (19, 30)),
    # robustness: clean 3-week target right after a history that ends with the logging gap
    "F2": Fold("F2", (11, 27), (28, 30)),
    # robustness: clean 4-week target, short clean history
    "F0": Fold("F0", (11, 15), (16, 19)),
    # test: history 11-30, target 31-42 (labels unknown)
    "TEST": Fold("TEST", (11, 30), (31, 42)),
}


def load_views(variant: str = "sum") -> pl.DataFrame:
    """(u, airing_id, tv_show_id, c, start_ts, duration, week, screen, full)."""
    v = pl.read_parquet(P / "views.parquet")
    col = {"sum": "screen", "max": "screen_max"}[variant]
    return v.with_columns(pl.col(col).alias("scr"), (pl.col(col) >= FULL).alias("full"))


def load_schedule() -> pl.DataFrame:
    return pl.read_parquet(P / "schedule.parquet")


def load_users() -> pl.DataFrame:
    return pl.read_parquet(P / "users.parquet")


def submission_users() -> pl.DataFrame:
    sub = pl.read_csv(RAW / "submission.csv", schema_overrides={"user_id": pl.UInt64})
    return sub.select("user_id").join(load_users(), on="user_id", how="left")


def target_counts(views: pl.DataFrame, weeks: tuple[int, int]) -> pl.DataFrame:
    """Full-view counts per (u, tv_show_id) in the week window."""
    return (
        views.filter(pl.col("week").is_between(*weeks) & pl.col("full"))
        .group_by(["u", "tv_show_id"]).len("cnt")
    )


def ground_truth(views: pl.DataFrame, weeks: tuple[int, int], tie: str = "id_asc") -> dict:
    """{u: [top-5 tv_show_id]} for users with >= 5 distinct full-view shows."""
    cnt = target_counts(views, weeks)
    if tie == "id_asc":
        cnt = cnt.sort(["u", "cnt", "tv_show_id"], descending=[False, True, False])
    elif tie == "id_desc":
        cnt = cnt.sort(["u", "cnt", "tv_show_id"], descending=[False, True, True])
    elif tie.startswith("random"):
        seed = int(tie[6:] or 0)
        cnt = cnt.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).alias("_r")).sort(
            ["u", "cnt", "_r"], descending=[False, True, False])
    else:
        raise ValueError(tie)
    n = cnt.group_by("u").len("n")
    top = cnt.join(n.filter(pl.col("n") >= 5), on="u").group_by("u", maintain_order=True).head(5)
    return {u: shows for u, shows in top.group_by("u", maintain_order=True).agg(pl.col("tv_show_id")).iter_rows()}


def shows_airing(sched: pl.DataFrame, weeks: tuple[int, int]) -> pl.DataFrame:
    """Airing counts per show in a week window (schedule is future metadata, allowed)."""
    return (
        sched.filter(pl.col("week").is_between(*weeks))
        .group_by("tv_show_id").agg(pl.len().alias("n_air"), pl.col("duration").sum().alias("air_secs"))
    )
