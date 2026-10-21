"""Evidence for choosing the remaining submissions -> evidence/final_selection.json

1. Label definition: screen time summed over sessions ("sum") vs longest single session ("max").
   How often do the true top-5 sets differ?  Cross-evaluation: models trained on each label,
   scored against both truths, on F1 / F0 / F2.
2. Candidate final models on every fold (3 seeds each, out-of-fold, user-grouped):
   H_sum (submission 2 config), H_max, ENS(H_sum, H_max), V1 (submission 1 config, sum),
   ENS(H_sum, V1).
3. Noise: standard error of the difference of two models' MAP@5 on a random half of the
   submission users (the size of the public board).
4. Test-history proxy (target 28-30): rank-average of two history windows vs each alone.

Run: .venv/bin/python -I src/final_selection.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT, ground_truth, load_schedule, load_views, submission_users  # noqa: E402
from experiments import run_cv  # noqa: E402
from features import build, label_top5  # noqa: E402
from hybrid import make_table  # noqa: E402
from metrics import apk, mapk  # noqa: E402
from predict_test import L31  # noqa: E402
from ranker import CATS, PARAMS, feature_cols, to_pandas, top5_from_scores, user_folds  # noqa: E402

SEEDS = 3
FOLDS = {"F1": (list(range(11, 19)), list(range(19, 31))),
         "F0": (list(range(11, 16)), list(range(16, 20))),
         "F2": (list(range(11, 28)), list(range(28, 31)))}
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)


def within_user_rank(df: pl.DataFrame, col: str) -> pl.Series:
    return df.select((pl.col(col).rank() / pl.len()).over("u").alias("r"))["r"]


def oof_scores(X, gt, params, rounds, cols):
    """seed-averaged within-user rank of out-of-fold scores -> DataFrame (u, tv_show_id, s)"""
    acc = np.zeros(X.height)
    for s in range(SEEDS):
        p = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s)
        o = run_cv(X, gt, p, rounds=rounds, cols=cols)
        acc += X.select("u").with_columns(pl.Series("o", o)).select(
            (pl.col("o").rank() / pl.len()).over("u"))["o"].to_numpy()
    return X.select(["u", "tv_show_id"]).with_columns(pl.Series("s", acc / SEEDS))


def blend(*tables):
    """rank-average on the union of candidates; a model that did not score a pair gives it 0"""
    out = None
    for i, t in enumerate(tables):
        t = t.rename({"s": f"s{i}"})
        out = t if out is None else out.join(t, on=["u", "tv_show_id"], how="full", coalesce=True)
    cols = [f"s{i}" for i in range(len(tables))]
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in cols])
    return out.with_columns((sum(pl.col(c) for c in cols) / len(cols)).alias("s")).select(["u", "tv_show_id", "s"])


def top5(t):
    return top5_from_scores(t.sort(["u", "tv_show_id"]), t.sort(["u", "tv_show_id"])["s"].to_numpy())


def main():
    sched = load_schedule()
    views = {v: load_views(v) for v in ["sum", "max"]}
    sub_u = set(submission_users()["u"].to_list())
    out = {"label_agreement": {}, "folds": {}}
    aps_F1 = {}
    for fname, (hist, target) in FOLDS.items():
        gts = {v: ground_truth(views[v], (min(target), max(target))) for v in views}
        common = sorted(set(gts["sum"]) & set(gts["max"]))
        same = sum(set(gts["sum"][u]) == set(gts["max"][u]) for u in common)
        overlap = float(np.mean([len(set(gts["sum"][u]) & set(gts["max"][u])) for u in common]))
        out["label_agreement"][fname] = {"users_sum": len(gts["sum"]), "users_max": len(gts["max"]),
                                         "users_both": len(common), "identical_top5_share": round(same / len(common), 4),
                                         "mean_overlap_of_5": round(overlap, 3)}
        log(fname, out["label_agreement"][fname])
        models = {}
        for v in ["sum", "max"]:
            X = label_top5(make_table(views[v], sched, hist, target), gts[v])
            X = X.filter(pl.col("u").is_in(pl.Series(list(gts[v])).implode())).sort(["u", "tv_show_id"])
            models[f"H_{v}"] = oof_scores(X, gts[v], L31, 250, feature_cols(X))
            log(fname, "trained", f"H_{v}")
        Xv1 = label_top5(build(views["sum"], sched, hist, target), gts["sum"])
        Xv1 = Xv1.filter(pl.col("u").is_in(pl.Series(list(gts["sum"])).implode())).sort(["u", "tv_show_id"])
        v1_cols = json.loads((ROOT / "evidence" / "sub01_ranker_v1_info.json").read_text())["features"]
        models["V1"] = oof_scores(Xv1, gts["sum"], PARAMS, 400, v1_cols)
        log(fname, "trained V1")
        models["ENS_Hsum_Hmax"] = blend(models["H_sum"], models["H_max"])
        models["ENS_Hsum_V1"] = blend(models["H_sum"], models["V1"])
        models["ENS_Hsum_Hmax_V1"] = blend(models["H_sum"], models["H_max"], models["V1"])
        res = {}
        for name, t in models.items():
            pred = top5(t)
            r = {}
            for v in ["sum", "max"]:
                g = gts[v]
                us = list(g)
                r[f"truth_{v}_all"] = round(mapk(g, pred, 5, us), 5)
                r[f"truth_{v}_sub"] = round(mapk(g, pred, 5, [u for u in us if u in sub_u]), 5)
            res[name] = r
            if fname == "F1":
                aps_F1[name] = {u: apk(gts["sum"][u], pred.get(u, []), 5) for u in gts["sum"] if u in sub_u}
        out["folds"][fname] = res
        log(fname, json.dumps(res))

    # ---- noise of a public-board difference (630 random submission users), F1, truth=sum
    rng = np.random.default_rng(0)
    users = sorted(aps_F1["H_sum"])
    noise = {}
    for a, b in [("H_sum", "V1"), ("H_sum", "ENS_Hsum_V1"), ("H_sum", "H_max"), ("H_sum", "ENS_Hsum_Hmax")]:
        d = np.array([aps_F1[a][u] - aps_F1[b][u] for u in users])
        halves = [d[rng.choice(len(d), 630, replace=False)].mean() for _ in range(2000)]
        noise[f"{a}-{b}"] = {"mean_diff_all_sub": round(float(d.mean()), 5), "sd_per_user": round(float(d.std()), 4),
                             "se_630": round(float(d.std() / np.sqrt(630)), 5),
                             "half_sample_diff_p05_p95": [round(float(np.quantile(halves, 0.05)), 5),
                                                          round(float(np.quantile(halves, 0.95)), 5)]}
    a = np.array([aps_F1["H_sum"][u] for u in users])
    halves = [a[rng.choice(len(a), 630, replace=False)].mean() for _ in range(2000)]
    noise["H_sum_level_half_p05_p95"] = [round(float(np.quantile(halves, 0.05)), 5), round(float(np.quantile(halves, 0.95)), 5)]
    out["public_noise_F1_sub"] = noise
    log("noise", json.dumps(noise))

    # ---- test-history proxy: ensemble of two windows (target 28-30), same split as history_proxy.py
    v = views["sum"]
    gt1 = ground_truth(v, (19, 30))
    gt3 = ground_truth(v, (28, 30))
    X1 = label_top5(make_table(v, sched, list(range(11, 19)), list(range(19, 31))), gt1)
    X1 = X1.filter(pl.col("u").is_in(pl.Series(list(gt1)).implode())).sort(["u", "tv_show_id"])
    cols = feature_cols(X1)
    allu = np.array(sorted(set(X1["u"].to_list()) | set(gt3)))
    fmap = user_folds(allu, 2, 0)
    trainA = [u for u in X1["u"].unique().to_list() if fmap[u] == 0]
    evalB = [u for u in gt3 if fmap.get(u, 0) == 1]
    evalB_sub = [u for u in evalB if u in sub_u]
    Xa = X1.filter(pl.col("u").is_in(pl.Series(trainA).implode()))
    g = np.unique(Xa["u"].to_numpy(), return_counts=True)[1]
    tabs = {}
    for name, hw in {"old_12_19": list(range(12, 20)), "recent_17_24": list(range(17, 25)),
                     "contig_20_27": list(range(20, 28))}.items():
        Xb = make_table(v, sched, hw, [28, 29, 30], with_labels=False)
        Xb = Xb.filter(pl.col("u").is_in(pl.Series(evalB).implode())).sort(["u", "tv_show_id"])
        Pa, Pb = to_pandas(Xa, cols), to_pandas(Xb, cols)
        for c in CATS:
            cats = sorted(set(Pa[c].astype(str)) | set(Pb[c].astype(str)))
            Pa[c] = Pa[c].astype(str).astype("category").cat.set_categories(cats)
            Pb[c] = Pb[c].astype(str).astype("category").cat.set_categories(cats)
        acc = np.zeros(Xb.height)
        for s in range(SEEDS):
            p = dict(L31, seed=s, bagging_seed=s, feature_fraction_seed=s)
            m = lgb.train(p, lgb.Dataset(Pa, Xa["y"].to_numpy(), group=g, categorical_feature=CATS), 250)
            o = m.predict(Pb)
            acc += Xb.select("u").with_columns(pl.Series("o", o)).select((pl.col("o").rank() / pl.len()).over("u"))["o"].to_numpy()
        tabs[name] = Xb.select(["u", "tv_show_id"]).with_columns(pl.Series("s", acc / SEEDS))
    tabs["ENS_old+recent"] = blend(tabs["old_12_19"], tabs["recent_17_24"])
    tabs["ENS_old+contig"] = blend(tabs["old_12_19"], tabs["contig_20_27"])
    prox = {}
    for name, t in tabs.items():
        pred = top5(t)
        prox[name] = {"map5_evalB": round(mapk(gt3, pred, 5, evalB), 5), "map5_evalB_sub": round(mapk(gt3, pred, 5, evalB_sub), 5)}
    out["history_proxy_ensembles"] = prox
    log("history proxy", json.dumps(prox))
    (ROOT / "evidence" / "final_selection.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
