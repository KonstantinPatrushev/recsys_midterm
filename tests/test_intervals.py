"""Brute-force check of src/intervals.overlap_pairs on random schedules and sessions,
including overlapping programmes inside a channel and sessions outside any airing.

Run: .venv/bin/python -I tests/test_intervals.py   (from midterm/)
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intervals import overlap_pairs  # noqa: E402

rng = np.random.default_rng(7)
ok = True
n_pairs_total = 0
for trial in range(300):
    n_ch = rng.integers(1, 5)
    air = []
    for c in range(n_ch):
        t = int(rng.integers(0, 50))
        for _ in range(int(rng.integers(0, 12))):
            d = int(rng.integers(1, 30))
            # occasionally overlap the previous programme or leave a gap
            t0 = t - int(rng.integers(0, 10)) if rng.random() < 0.15 else t + int(rng.integers(0, 5))
            air.append((c, t0, t0 + d))
            t = t0 + d
    air.sort()
    air = np.array(air, dtype=np.int64).reshape(-1, 3)
    n_s = int(rng.integers(0, 40))
    ss = rng.integers(0, 400, n_s)
    sess = np.stack([rng.integers(0, n_ch, n_s), ss, ss + rng.integers(1, 80, n_s)], axis=1).astype(np.int64)

    si, ai, ov = overlap_pairs(sess[:, 0], sess[:, 1], sess[:, 2], air[:, 0], air[:, 1], air[:, 2])
    got = sorted(zip(si.tolist(), ai.tolist(), ov.tolist()))
    exp = []
    for i, (c, s, e) in enumerate(sess.tolist()):
        for j, (ac, a_s, a_e) in enumerate(air.tolist()):
            if ac == c:
                o = min(e, a_e) - max(s, a_s)
                if o > 0:
                    exp.append((i, j, o))
    exp.sort()
    if got != exp:
        ok = False
        print("MISMATCH in trial", trial)
        break
    n_pairs_total += len(exp)

out = {"trials": 300, "pairs_checked": n_pairs_total, "all_equal_to_bruteforce": ok}
(ROOT / "evidence").mkdir(exist_ok=True)
(ROOT / "evidence" / "test_intervals.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
sys.exit(0 if ok else 1)
