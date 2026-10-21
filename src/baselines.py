"""Simple baselines on every validation fold -> evidence/baselines.json.

POP          most fully-watched shows overall in the history window (same list for everyone)
REPEAT       user's own full-view counts in history (fill with POP)
REPEAT_AIR   REPEAT restricted to shows that air in the target window (future schedule)
RATE_AIR     user's full-view rate per airing x number of airings in the target window
DECAY_AIR    exponentially time-decayed counts (half-life h weeks), restricted to airing shows

Run: .venv/bin/python -I src/baselines.py   (from midterm/)
"""
from __future__ import annotations

import json
import sys
import pathlib

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import FOLDS, ROOT, ground_truth, load_schedule, load_views, shows_airing, submission_users  # noqa: E402
from metrics import mapk  # noqa: E402


def topk_lists(scores: pl.DataFrame, k: int = 5, fill: list | None = None) -> dict:
    """scores: (u, tv_show_id, score) -> {u: top-k list}, padded with `fill` items."""
    top = (
        scores.sort(["u", "score", "tv_show_id"], descending=[False, True, False])
        .group_by("u", maintain_order=True).head(k)
        .group_by("u", maintain_order=True).agg(pl.col("tv_show_id"))
    )
    out = {}
    for u, items in top.iter_rows():
        items = list(items)
        if fill and len(items) < k:
            items += [x for x in fill if x not in items][: k - len(items)]
        out[u] = items
    return out


def run_fold(views, sched, fold, sub_u, halflives=(1, 2, 4, 8)):
    gt = ground_truth(views, fold.target)
    users = list(gt.keys())
    sub_eval = [u for u in users if u in sub_u]
    h0, h1 = fold.hist
    hist = views.filter(pl.col("week").is_between(h0, h1) & pl.col("full"))
    air_t = shows_airing(sched, fold.target)
    air_h = shows_airing(sched, fold.hist).rename({"n_air": "n_air_h", "air_secs": "air_secs_h"})
    airing = air_t.select("tv_show_id")

    pop_all = hist.group_by("tv_show_id").len("score").sort("score", descending=True)
    pop_air = pop_all.join(airing, on="tv_show_id").sort(["score", "tv_show_id"], descending=[True, False])
    pop_list = pop_air["tv_show_id"].head(50).to_list()
    pop_list_any = pop_all.sort(["score", "tv_show_id"], descending=[True, False])["tv_show_id"].head(50).to_list()

    res = {}
    preds = {}
    preds["POP"] = {u: pop_list[:5] for u in users}
    preds["POP_ANY"] = {u: pop_list_any[:5] for u in users}

    cnt = hist.group_by(["u", "tv_show_id"]).len("score")
    preds["REPEAT"] = topk_lists(cnt, fill=pop_list_any)
    cnt_air = cnt.join(airing, on="tv_show_id")
    preds["REPEAT_AIR"] = topk_lists(cnt_air, fill=pop_list)

    rate = cnt.join(air_h, on="tv_show_id").join(air_t, on="tv_show_id").with_columns(
        (pl.col("score") / pl.col("n_air_h") * pl.col("n_air")).alias("score")
    )
    preds["RATE_AIR"] = topk_lists(rate.select(["u", "tv_show_id", "score"]), fill=pop_list)

    for h in halflives:
        dec = (
            hist.with_columns((0.5 ** ((h1 - pl.col("week")) / h)).alias("w"))
            .group_by(["u", "tv_show_id"]).agg(pl.col("w").sum().alias("score"))
            .join(airing, on="tv_show_id")
        )
        preds[f"DECAY_AIR_h{h}"] = topk_lists(dec, fill=pop_list)
        dr = dec.join(air_h, on="tv_show_id").join(air_t, on="tv_show_id").with_columns(
            (pl.col("score") / pl.col("n_air_h") * pl.col("n_air")).alias("score"))
        preds[f"DECAY_RATE_AIR_h{h}"] = topk_lists(dr.select(["u", "tv_show_id", "score"]), fill=pop_list)

    for name, p in preds.items():
        res[name] = {
            "map5_all": round(mapk(gt, p, 5, users), 5),
            "map5_sub": round(mapk(gt, p, 5, sub_eval), 5),
        }
    meta = {"n_eval_users": len(users), "n_eval_sub_users": len(sub_eval), "hist": fold.hist, "target": fold.target}
    return meta, res


def main():
    out = {}
    sched = load_schedule()
    sub_u = set(submission_users()["u"].to_list())
    for variant in ["sum", "max"]:
        views = load_views(variant)
        for fname in ["F1", "F2", "F0"]:
            meta, res = run_fold(views, sched, FOLDS[fname], sub_u)
            out[f"{variant}/{fname}"] = {"meta": meta, "scores": res}
            print(variant, fname, meta)
            for k, v in sorted(res.items(), key=lambda kv: -kv[1]["map5_all"]):
                print(f"   {k:22s} all={v['map5_all']:.5f}  sub={v['map5_sub']:.5f}")
    (ROOT / "evidence" / "baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
