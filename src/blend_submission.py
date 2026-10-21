"""Blend saved test score tables (cache/test_scores_<name>.parquet) by within-user rank averaging
and write a submission; report overlap with earlier submissions.

Run: .venv/bin/python -I src/blend_submission.py --out <name> --parts <name1> <name2> ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import pathlib

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, load_schedule, load_views, submission_users  # noqa: E402
from predict_test import TEST_HIST  # noqa: E402


def overlap(a_path, b_path):
    a = {r[0]: r[1].split() for r in list(csv.reader(open(a_path)))[1:]}
    b = {r[0]: r[1].split() for r in list(csv.reader(open(b_path)))[1:]}
    return {"identical_sets": sum(set(a[u]) == set(b[u]) for u in a),
            "mean_overlap_of_5": round(sum(len(set(a[u]) & set(b[u])) for u in a) / len(a), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    args = ap.parse_args()
    tabs = []
    for i, name in enumerate(args.parts):
        t = pl.read_parquet(ROOT / "cache" / f"test_scores_{name}.parquet")
        t = t.with_columns((pl.col("score").rank() / pl.len()).over("u").alias(f"s{i}")).select(["u", "tv_show_id", f"s{i}"])
        tabs.append(t)
    out = tabs[0]
    for t in tabs[1:]:
        out = out.join(t, on=["u", "tv_show_id"], how="full", coalesce=True)
    cols = [f"s{i}" for i in range(len(tabs))]
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in cols]).with_columns(
        (sum(pl.col(c) for c in cols) / len(cols)).alias("s"))
    top = (out.sort(["u", "s", "tv_show_id"], descending=[False, True, False]).group_by("u", maintain_order=True).head(5)
           .group_by("u", maintain_order=True).agg(pl.col("tv_show_id")))
    pred = {u: list(t) for u, t in top.iter_rows()}
    views, sched = load_views("sum"), load_schedule()
    pop = (views.filter(pl.col("week").is_in(pl.Series(TEST_HIST).implode()) & pl.col("full"))
           .group_by("tv_show_id").agg(pl.col("u").n_unique().alias("n")).join(
               sched.filter(pl.col("week").is_between(31, 42)).select("tv_show_id").unique(), on="tv_show_id")
           .sort(["n", "tv_show_id"], descending=[True, False])["tv_show_id"].to_list())
    sub = submission_users()
    rows, n_filled = [], 0
    for uid, u in sub.select(["user_id", "u"]).iter_rows():
        items = pred.get(u, [])
        if len(items) < 5:
            n_filled += 1
            items = items + [x for x in pop if x not in items][: 5 - len(items)]
        assert len(set(items)) == 5
        rows.append((uid, " ".join(str(int(x)) for x in items)))
    path = ROOT / "submissions" / f"{args.out}.csv"
    pl.DataFrame(rows, schema={"user_id": pl.UInt64, "tv_show_id": pl.Utf8}, orient="row").write_csv(path)
    info = {"name": args.out, "parts": args.parts, "users_filled_from_pop": n_filled, "overlap": {}}
    for prev in sorted((ROOT / "submissions").glob("sub0*.csv")):
        if prev.name != path.name:
            info["overlap"][prev.stem] = overlap(path, prev)
    (ROOT / "evidence" / f"{args.out}_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
