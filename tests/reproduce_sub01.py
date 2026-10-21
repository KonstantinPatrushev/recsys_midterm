"""Re-create submission 1 from scratch with the current code and compare byte-for-byte.

Submission 1 used the v1 feature set (the column list stored in evidence/sub01_ranker_v1_info.json),
the base candidate table, LightGBM lambdarank with 63 leaves, 5 seeds and global rank averaging.
Later feature additions only add columns, so selecting the v1 columns must give the same file.

Run: .venv/bin/python -I tests/reproduce_sub01.py
"""
import hashlib
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from features import build, label_top5  # noqa: E402
from ranker import CATS, PARAMS, to_pandas, top5_from_scores  # noqa: E402

info = json.loads((ROOT / "evidence" / "sub01_ranker_v1_info.json").read_text())
cols = info["features"]
views, sched = load_views("sum"), load_schedule()
sub = submission_users()
sub_u = sub["u"].to_list()
gt = ground_truth(views, (19, 30))
Xtr = label_top5(build(views, sched, list(range(11, 19)), list(range(19, 31))), gt)
Xtr = Xtr.filter(pl.col("u").is_in(pl.Series(list(gt)).implode())).sort(["u", "tv_show_id"])
Xte = build(views, sched, info["test_hist_weeks"], list(range(31, 43)), with_labels=False)
Xte = Xte.filter(pl.col("u").is_in(pl.Series(sub_u).implode()))
no_hist = [u for u in sub_u if u not in set(Xte["u"].to_list())]
Xfb = build(views, sched, list(range(11, 19)), list(range(31, 43)), with_labels=False)
Xte = pl.concat([Xte, Xfb.filter(pl.col("u").is_in(pl.Series(no_hist).implode())).select(Xte.columns)]).sort(["u", "tv_show_id"])
Xp_tr, Xp_te = to_pandas(Xtr, cols), to_pandas(Xte, cols)
for c in CATS:
    cats = sorted(set(Xp_tr[c].astype(str)) | set(Xp_te[c].astype(str)))
    Xp_tr[c] = Xp_tr[c].astype(str).astype("category").cat.set_categories(cats)
    Xp_te[c] = Xp_te[c].astype(str).astype("category").cat.set_categories(cats)
group = np.unique(Xtr["u"].to_numpy(), return_counts=True)[1]
score = np.zeros(Xte.height)
for seed in range(info["seeds"]):
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    m = lgb.train(p, lgb.Dataset(Xp_tr, Xtr["y"].to_numpy(), group=group,
                                 categorical_feature=[c for c in CATS if c in cols]), num_boost_round=info["rounds"])
    score += pl.Series(m.predict(Xp_te)).rank().to_numpy() / Xte.height
pred = top5_from_scores(Xte, score)
pop = (views.filter(pl.col("week").is_in(pl.Series(info["test_hist_weeks"]).implode()) & pl.col("full"))
       .group_by("tv_show_id").agg(pl.col("u").n_unique().alias("n")).join(
           sched.filter(pl.col("week").is_between(31, 42)).select("tv_show_id").unique(), on="tv_show_id")
       .sort(["n", "tv_show_id"], descending=[True, False])["tv_show_id"].to_list())
rows = []
for uid, u in sub.select(["user_id", "u"]).iter_rows():
    items = pred.get(u, [])
    items = items + [x for x in pop if x not in items][: 5 - len(items)]
    rows.append((uid, " ".join(str(int(x)) for x in items)))
out_path = ROOT / "cache" / "sub01_reproduced.csv"
pl.DataFrame(rows, schema={"user_id": pl.UInt64, "tv_show_id": pl.Utf8}, orient="row").write_csv(out_path)
h_new = hashlib.sha256(out_path.read_bytes()).hexdigest()
h_old = hashlib.sha256((ROOT / "submissions" / "sub01_ranker_v1.csv").read_bytes()).hexdigest()
res = {"sha256_submitted": h_old, "sha256_reproduced": h_new, "identical": h_new == h_old}
(ROOT / "evidence" / "reproduce_sub01.json").write_text(json.dumps(res, indent=2))
print(res)
