"""Build the F1 and test feature tables in three separate processes and check they are identical
(the fix for run-dependent float sums in src/features.py) -> evidence/determinism_check.json

Run: .venv/bin/python -I tests/check_determinism.py   (from midterm/)
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "determinism"
OUT.mkdir(parents=True, exist_ok=True)
CHILD = """
import sys; sys.path.insert(0, 'src')
from data import load_views, load_schedule
from hybrid import make_table
from predict_test import TEST_HIST, TEST_TARGET
v, s = load_views('sum'), load_schedule()
make_table(v, s, list(range(11, 19)), list(range(19, 31))).write_parquet('cache/determinism/f1_{i}.parquet')
make_table(v, s, TEST_HIST, TEST_TARGET, with_labels=False).write_parquet('cache/determinism/test_{i}.parquet')
"""
for i in range(3):
    subprocess.run([sys.executable, "-I", "-c", CHILD.format(i=i)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import polars as pl  # noqa: E402

res = {}
for name in ["f1", "test"]:
    t = [pl.read_parquet(OUT / f"{name}_{i}.parquet") for i in range(3)]
    res[name] = {"rows": t[0].height, "columns": t[0].width,
                 "identical_run0_run1": t[0].equals(t[1]), "identical_run0_run2": t[0].equals(t[2])}
res["all_identical"] = all(v["identical_run0_run1"] and v["identical_run0_run2"] for v in res.values())
(ROOT / "evidence" / "determinism_check.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res))
sys.exit(0 if res["all_identical"] else 1)
