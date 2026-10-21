"""Checks for src/metrics.py: hand-computed cases + an independent re-implementation.

Run: .venv/bin/python -I tests/test_metrics.py   (from midterm/)
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import apk, mapk  # noqa: E402


def apk_independent(actual, predicted, k=5):
    """Written from the formula, without looking at apk(): precision at each hit rank."""
    actual = list(dict.fromkeys(actual))
    if len(actual) == 0:
        return 0.0
    first_pos = {}
    for pos, item in enumerate(predicted[:k], start=1):
        first_pos.setdefault(item, pos)
    hit_ranks = sorted(pos for item, pos in first_pos.items() if item in set(actual))
    precisions = [(j + 1) / r for j, r in enumerate(hit_ranks)]
    return sum(precisions) / min(len(actual), k)


cases = [
    # (actual, predicted, expected, how it was computed by hand)
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0, "all 5 hits: (1+1+1+1+1)/5"),
    ([1, 2, 3, 4, 5], [9, 1, 8, 2, 7], (1 / 2 + 2 / 4) / 5, "hits at ranks 2,4: (1/2+2/4)/5 = 0.2"),
    ([10, 20], [20, 30, 10, 40, 50], (1 / 1 + 2 / 3) / 2, "hits at 1,3; min(2,5)=2: (1+2/3)/2 = 0.8333"),
    ([7], [1, 2, 3, 4, 7], (1 / 5) / 1, "single item at rank 5: 0.2"),
    ([7], [1, 2, 3, 4, 5, 7], 0.0, "item at rank 6 is cut by K=5"),
    ([1, 2, 3], [1, 1, 1, 2, 3], (1 / 1 + 2 / 4 + 3 / 5) / 3, "duplicates earn nothing: (1+0.5+0.6)/3 = 0.7"),
    ([], [1, 2, 3, 4, 5], 0.0, "empty ground truth"),
    ([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5], 1.0, "|actual|>K -> divide by K"),
]

results = []
ok = True
for actual, predicted, expected, how in cases:
    got = apk(actual, predicted, 5)
    ind = apk_independent(actual, predicted, 5)
    passed = abs(got - expected) < 1e-12 and abs(ind - expected) < 1e-12
    ok &= passed
    results.append({"actual": actual, "predicted": predicted, "expected": expected,
                    "apk": got, "independent": ind, "how": how, "pass": passed})

# Randomised agreement between the two implementations
rng = random.Random(42)
max_diff = 0.0
for _ in range(20000):
    actual = rng.sample(range(30), rng.randint(0, 8))
    predicted = [rng.randrange(30) for _ in range(rng.randint(0, 7))]
    d = abs(apk(actual, predicted, 5) - apk_independent(actual, predicted, 5))
    max_diff = max(max_diff, d)
ok &= max_diff < 1e-12

# mapk averages over ground-truth users; missing predictions count as 0
gt = {"u1": [1, 2], "u2": [3]}
pred = {"u1": [1, 9, 2, 8, 7]}
m = mapk(gt, pred, 5)
exp_m = ((1 + 2 / 3) / 2 + 0.0) / 2
ok &= abs(m - exp_m) < 1e-12

out = {"cases": results, "random_pairs": 20000, "max_abs_diff_vs_independent": max_diff,
       "mapk_missing_user_case": {"got": m, "expected": exp_m}, "all_pass": ok}
(ROOT / "evidence").mkdir(exist_ok=True)
(ROOT / "evidence" / "test_metrics.json").write_text(json.dumps(out, indent=2))
print(json.dumps({"all_pass": ok, "max_abs_diff_vs_independent": max_diff}, indent=2))
sys.exit(0 if ok else 1)
