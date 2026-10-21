"""Business analysis of two users on the validation fold F1 (history 11-18, truth 19-30),
plus their test-period recommendations.

Selection (no hand-picking): among submission users with ground truth on F1 and typical
activity (history full views between the 25th and 75th percentile),
  * "habitual": every true top-5 show was already fully watched in history (novelty 0)
  * "explorer": at least 2 true top-5 shows were never fully watched in history
one user is drawn from each segment with a fixed seed.

Run: .venv/bin/python -I src/business.py [--test-sub submissions/<file>.csv]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import pathlib

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_users, load_views, submission_users  # noqa: E402
from methods import Space, als_scores, ar_scores, cbf_scores, pop_scores, ppr_scores, repeat_scores, topk_from_matrix  # noqa: E402
from metrics import apk  # noqa: E402

SEED = 42


def show_titles(sched: pl.DataFrame) -> dict:
    t = (sched.group_by(["tv_show_id", "tv_show_title"]).len("n")
         .sort(["tv_show_id", "n", "tv_show_title"], descending=[False, True, False])
         .group_by("tv_show_id", maintain_order=True).first())
    return {s: title for s, title, _ in t.iter_rows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-sub", default=None)
    args = ap.parse_args()
    views = load_views("sum")
    sched = load_schedule()
    titles = show_titles(sched)
    chan = {c: t for c, t in sched.select(["c", "channel_title"]).unique(subset="c").iter_rows()}
    meta = {s: (cat, g) for s, cat, g in sched.group_by("tv_show_id", maintain_order=True).agg(
        pl.col("tv_show_category").first(), pl.col("tv_show_genre_1").first()).iter_rows()}
    hist, target = list(range(11, 19)), list(range(19, 31))
    gt = ground_truth(views, (19, 30))
    sub = submission_users()
    sub_u = set(sub["u"].to_list())
    uid_of = dict(load_users().select(["u", "user_id"]).iter_rows())

    hv = views.filter(pl.col("week").is_between(11, 18) & pl.col("full"))
    hcnt = hv.group_by(["u", "tv_show_id"]).len("n")
    seen = {u: set(s) for u, s in hcnt.group_by("u").agg(pl.col("tv_show_id")).iter_rows()}
    act = {u: n for u, n in hv.group_by("u").len("n").iter_rows()}
    cand = [u for u in gt if u in sub_u and u in act]
    a = np.array([act[u] for u in cand])
    lo, hi = np.quantile(a, 0.25), np.quantile(a, 0.75)
    typical = [u for u in cand if lo <= act[u] <= hi]
    novelty = {u: sum(s not in seen.get(u, set()) for s in gt[u]) for u in typical}
    habitual = sorted(u for u in typical if novelty[u] == 0)
    explorer = sorted(u for u in typical if novelty[u] >= 2)
    rng = random.Random(SEED)
    picks = {"habitual": rng.choice(habitual), "explorer": rng.choice(explorer)}

    # method predictions on F1
    space = Space(views, sched, hist, target)
    users = list(gt)
    preds = {
        "POP": topk_from_matrix(space, pop_scores(space), users),
        "REPEAT": topk_from_matrix(space, repeat_scores(space), users),
        "CBF": topk_from_matrix(space, cbf_scores(space, "raw"), users),
        "AR": topk_from_matrix(space, ar_scores(space, 20, "max")[0], users),
        "MF (ALS)": topk_from_matrix(space, als_scores(space, 64, 10.0, 1.0, iters=30), users),
        "PPR": topk_from_matrix(space, ppr_scores(space, 0.7, users=users)[0], users),
    }
    # hybrid = out-of-fold top-5 of the final configuration on F1 (sum-label + max-label rankers,
    # 5 seeds each; src/report_stats.py) -- the validation analogue of the designated final submission
    cvp = json.loads((ROOT / "cache" / "cv_pred_final_config.json").read_text())
    preds["Hybrid (LGBM)"] = {int(u): v for u, v in cvp.items()}

    test_pred = {}
    if args.test_sub:
        ts = pl.read_csv(args.test_sub, schema_overrides={"user_id": pl.UInt64})
        test_pred = {uid: [int(x) for x in s.split()] for uid, s in ts.iter_rows()}
    seen_all = {u: set(s) for u, s in views.filter(pl.col("week").is_in(pl.Series([15, 16, 17, 18, 19, 28, 29, 30]).implode()) & pl.col("full"))
                .group_by("u").agg(pl.col("tv_show_id")).iter_rows()}

    out = {"selection": {"typical_activity_range_full_views_11_18": [float(lo), float(hi)],
                         "n_typical_sub_users": len(typical), "n_habitual": len(habitual), "n_explorer": len(explorer),
                         "seed": SEED, "picked_u": picks}}

    def desc(s):
        cat, g = meta.get(s, (None, None))
        return {"id": int(s), "title": titles.get(s), "category": cat, "genre": g}

    for seg, u in picks.items():
        uh = hv.filter(pl.col("u") == u)
        top_hist = uh.group_by("tv_show_id").agg(pl.len().alias("n"), pl.col("c").mode().min().alias("c")).sort(
            ["n", "tv_show_id"], descending=[True, False]).head(10)
        ch = views.filter((pl.col("u") == u) & pl.col("week").is_between(11, 18)).group_by("c").agg(
            pl.col("watched").sum().alias("secs")).sort("secs", descending=True).head(5)
        tot = views.filter((pl.col("u") == u) & pl.col("week").is_between(11, 18))["watched"].sum()
        hours = (uh.with_columns(pl.from_epoch("start_ts", time_unit="s").dt.hour().alias("h"))
                 .group_by("h").len("n").sort("n", descending=True).head(3))
        tcnt = views.filter((pl.col("u") == u) & pl.col("week").is_between(19, 30) & pl.col("full")).group_by(
            "tv_show_id").agg(pl.len().alias("n"), pl.col("c").mode().min().alias("c"),
                              pl.col("week").min().alias("first_week"))
        tc = {s: n for s, n, _, _ in tcnt.iter_rows()}
        tch = {s: (chan.get(c), w) for s, _, c, w in tcnt.iter_rows()}
        hist_ch = set(views.filter((pl.col("u") == u) & pl.col("week").is_between(11, 18))["c"].unique().to_list())
        ch_id = {t: c for c, t in chan.items()}
        rec = {
            "user_id": str(uid_of[u]), "u": int(u), "history_full_views_11_18": act[u],
            "novelty_of_truth": novelty[u],
            "top_history_shows": [dict(desc(s), full_views=n, channel=chan.get(c)) for s, n, c in top_hist.iter_rows()],
            "top_channels_by_time": [{"channel": chan.get(c), "share": round(sec / tot, 3)} for c, sec in ch.iter_rows()],
            "top_start_hours_of_full_views": [{"hour": h, "n": n} for h, n in hours.iter_rows()],
            "truth_19_30": [dict(desc(s), full_views=tc.get(s), seen_in_history=s in seen.get(u, set()),
                                 channel=tch[s][0], first_full_view_week=tch[s][1],
                                 channel_watched_in_history=ch_id.get(tch[s][0]) in hist_ch) for s in gt[u]],
            "methods": {},
        }
        for name, p in preds.items():
            items = p.get(u, [])
            rec["methods"][name] = {
                "top5": [dict(desc(s), hit=s in set(gt[u]), seen_in_history=s in seen.get(u, set())) for s in items],
                "ap5": round(apk(gt[u], items, 5), 4),
            }
        uid = uid_of[u]
        if uid in test_pred:
            rec["test_31_42_final_top5"] = [dict(desc(s), familiar=s in seen_all.get(u, set())) for s in test_pred[uid]]
        out[seg] = rec
    # how often does each method recommend something new (not fully watched before) on F1, sub users
    nov = {}
    for name, p in preds.items():
        tot_items = sum(len(p.get(u, [])) for u in cand)
        new_items = sum(sum(s not in seen.get(u, set()) for s in p.get(u, [])) for u in cand)
        new_hits = sum(sum((s not in seen.get(u, set())) and (s in set(gt[u])) for s in p.get(u, [])) for u in cand)
        nov[name] = {"share_new_in_top5": round(new_items / max(tot_items, 1), 4),
                     "precision_of_new_items": round(new_hits / max(new_items, 1), 4)}
    out["novelty_by_method_sub_users_F1"] = nov
    cat_of = {s: c for s, (c, _) in meta.items()}
    t_items = [s for u in gt for s in gt[u]]
    cats = {}
    for s in t_items:
        cats[cat_of.get(s)] = cats.get(cat_of.get(s), 0) + 1
    out["truth_category_share_F1"] = {str(k): round(v / len(t_items), 4) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
    st_ = sched.filter(pl.col("week").is_between(19, 30))
    sc = st_.group_by("tv_show_category").len("n")
    out["airing_category_share_weeks_19_30"] = {str(k): round(v / st_.height, 4) for k, v in sc.sort("n", descending=True).iter_rows()}
    (ROOT / "evidence" / "business_two_users.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["selection"], indent=1))
    print(json.dumps(nov, indent=1))
    for seg in picks:
        r = out[seg]
        print(f"\n===== {seg}: u={r['u']} user_id={r['user_id']} history full views={r['history_full_views_11_18']}")
        print("top history:", [(x['title'][:40] if x['title'] else None, x['full_views']) for x in r['top_history_shows'][:6]])
        print("truth:", [(x['title'][:40] if x['title'] else None, x['full_views'], x['seen_in_history']) for x in r['truth_19_30']])
        for name, m in r["methods"].items():
            print(f"  {name:14s} AP={m['ap5']:.3f}", [((x['title'] or '')[:28], 'HIT' if x['hit'] else '') for x in m['top5']])


if __name__ == "__main__":
    main()
