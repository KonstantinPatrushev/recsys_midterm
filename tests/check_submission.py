"""Format check of a submission file against the sample submission.
Run: .venv/bin/python -I tests/check_submission.py submissions/<file>.csv"""
import csv, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sample = list(csv.reader(open(ROOT / "data/raw/submission.csv")))
sub = list(csv.reader(open(sys.argv[1])))
assert sub[0] == sample[0] == ["user_id", "tv_show_id"], sub[0]
su = [r[0] for r in sample[1:]]
pu = [r[0] for r in sub[1:]]
assert len(pu) == len(su) == 1260, (len(pu), len(su))
assert su == pu, "user order differs from sample"
import polars as pl
sched_ids = set(pl.read_parquet(ROOT / "data/processed/schedule.parquet").filter(pl.col("week").is_between(31, 42))["tv_show_id"].to_list())
bad = 0
for r in sub[1:]:
    items = r[1].split(" ")
    assert len(items) == 5 and len(set(items)) == 5, r
    assert all(x.isdigit() and x != "0" for x in items), r
    bad += sum(int(x) not in sched_ids for x in items)
print({"rows": len(pu), "users_match_sample_order": True, "items_not_airing_in_31_42": bad, "ok": True})
