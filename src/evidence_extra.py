"""Regenerate the evidence files that were first written by one-off session scripts:

  tz_alignment.json                          clock check between logs and schedule
  history_blocks.json                        submission users without a full view per history block
  rebuild_sub02_after_determinism_fix.json   uploaded submission 2 vs its rebuild with the fixed code
  nondeterminism_check.json                  the old feature code (commit 47bffcd) run twice
  kaggle_meta/private_leaderboard_2020.json  private scores of the 2020 teams (from topics_leaderboard.txt)

Run: .venv/bin/python -I src/evidence_extra.py   (from midterm/; needs git and data/processed)
"""
from __future__ import annotations

import ast
import csv
import json
import pathlib
import subprocess
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import P, load_views, submission_users  # noqa: E402

EV = ROOT / "evidence"


def tz_alignment():
    """Share of session starts / ends within 60 s after an airing start / end on the same channel,
    for clock shifts of -3..+3 hours (a peak at 0 means both files use the same clock)."""
    sess = pl.read_parquet(P / "sessions.parquet")
    sc = pl.read_parquet(P / "schedule.parquet").sort(["c", "start_ts"])
    shift = np.int64(1) << np.int64(40)
    ak = sc["c"].to_numpy().astype(np.int64) * shift + sc["start_ts"].to_numpy()
    ae = sc["c"].to_numpy().astype(np.int64) * shift + sc["end_ts"].to_numpy()
    cc = sc["c"].to_numpy()
    s = sess.filter(pl.col("c").is_in(sc["c"].unique().implode()))
    ch = s["c"].to_numpy().astype(np.int64)
    st, en = s["s_ts"].to_numpy(), s["e_ts"].to_numpy()
    res = {}
    for off in range(-3, 4):
        k = ch * shift + st + off * 3600
        i = np.searchsorted(ak, k, side="right") - 1
        ok = (i >= 0) & (cc[np.clip(i, 0, None)] == ch)
        gap = k[ok] - ak[i[ok]]
        k2 = ch * shift + en + off * 3600
        j = np.searchsorted(ae, k2, side="right") - 1
        ok2 = (j >= 0) & (cc[np.clip(j, 0, None)] == ch)
        gap2 = k2[ok2] - ae[j[ok2]]
        res[str(off)] = (round(float(np.mean((gap >= 0) & (gap < 60))), 4), round(float(np.mean((gap2 >= 0) & (gap2 < 60))), 4))
    return res


def history_blocks():
    full = load_views("sum").filter(pl.col("full"))
    sub = submission_users()["u"].to_list()
    res = {}
    for name, weeks in {"weeks_15_19": range(15, 20), "weeks_20_24": range(20, 25), "weeks_28_30": [28, 29, 30]}.items():
        have = set(full.filter(pl.col("week").is_in(pl.Series(list(weeks)).implode()))["u"].unique().to_list())
        res[name] = {"submission_users_without_full_view": sum(u not in have for u in sub)}
    res["n_submission_users"] = len(sub)
    return res


def rebuild_sub02():
    def load(f):
        return {r[0]: r[1] for r in list(csv.reader(open(f)))[1:]}
    import hashlib
    a_path = ROOT / "submissions" / "sub02_hybrid_lr31_r250_s5.csv"
    b_path = ROOT / "submissions" / "not_submitted" / "sub02_rebuilt_after_fix.csv"
    a, b = load(a_path), load(b_path)
    return {"users_with_different_list": sum(a[u] != b[u] for u in a),
            "users_with_different_set": sum(set(a[u].split()) != set(b[u].split()) for u in a),
            "sha256_submitted": hashlib.sha256(a_path.read_bytes()).hexdigest(),
            "sha256_rebuilt": hashlib.sha256(b_path.read_bytes()).hexdigest()}


CHILD = """
import sys, importlib.util
sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('features', 'cache/oldcode/features.py')
old = importlib.util.module_from_spec(spec); sys.modules['features'] = old; spec.loader.exec_module(old)
import hybrid
from data import load_views, load_schedule
X = hybrid.make_table(load_views('sum'), load_schedule(), list(range(11, 19)), list(range(19, 31)))
X.select(['u', 'tv_show_id', 'exp_show_slot', 'exp_show_slot_rank']).write_parquet('cache/oldcode/run_{i}.parquet')
"""


def nondeterminism():
    old_dir = ROOT / "cache" / "oldcode"
    old_dir.mkdir(parents=True, exist_ok=True)
    src = subprocess.run(["git", "show", "47bffcd:src/features.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (old_dir / "features.py").write_text(src)
    for i in (1, 2):
        subprocess.run([sys.executable, "-I", "-c", CHILD.format(i=i)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a = pl.read_parquet(old_dir / "run_1.parquet")
    b = pl.read_parquet(old_dir / "run_2.parquet")
    d = a.join(b, on=["u", "tv_show_id"], suffix="_2")
    diff = (d["exp_show_slot"] - d["exp_show_slot_2"]).abs()
    old_cv = json.loads(subprocess.run(["git", "show", "47bffcd:evidence/cv_sub02_hybrid_lr31_r250_s5.json"], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout)
    new_cv = json.loads((EV / "cv_sub02_hybrid_lr31_r250_s5.json").read_text())
    return {"old_code_commit": "47bffcd", "rows": d.height,
            "exp_show_slot_values_differing_between_two_runs": int((diff > 0).sum()),
            "max_abs_difference": float(diff.max()),
            "exp_show_slot_rank_differing": int((d["exp_show_slot_rank"] != d["exp_show_slot_rank_2"]).sum()),
            "cv_sub02_map5_all_recorded_at_47bffcd": old_cv["map5_all"], "cv_sub02_map5_sub_recorded_at_47bffcd": old_cv["map5_sub"],
            "cv_sub02_map5_all_after_fix": new_cv["map5_all"], "fixed_code_check": "evidence/determinism_check.json"}


def private_leaderboard():
    """topics_leaderboard.txt holds the Kaggle API leaderboard view (private scores after the close)."""
    rows = []
    for line in (EV / "kaggle_meta" / "topics_leaderboard.txt").read_text().splitlines():
        if line.startswith("{'team_name'"):
            d = ast.literal_eval(line.replace("datetime.datetime", "").split(", 'submission_date'")[0] + "}")
            rows.append({"team": d["team_name"], "private_score": float(d["score"])})
    return {"source": "Kaggle API competition_leaderboard_view, evidence/kaggle_meta/topics_leaderboard.txt", "top": rows}


if __name__ == "__main__":
    jobs = {"tz_alignment.json": tz_alignment, "history_blocks.json": history_blocks,
            "rebuild_sub02_after_determinism_fix.json": rebuild_sub02, "nondeterminism_check.json": nondeterminism,
            "kaggle_meta/private_leaderboard_2020.json": private_leaderboard}
    for name, fn in jobs.items():
        res = fn()
        (EV / name).write_text(json.dumps(res, indent=2))
        print(name, json.dumps(res)[:300])
