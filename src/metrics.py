"""MAP@K as used by Kaggle's "MAP@{K}" metric (ml_metrics.apk / mapk semantics).

AP@K(u) = 1 / min(|actual_u|, K) * sum_{k=1..K} P@k(u) * rel_k(u)

A predicted item counts only on its first occurrence (duplicates earn nothing),
and a user with an empty ground-truth list contributes 0.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def apk(actual: Iterable, predicted: Sequence, k: int = 5) -> float:
    actual = set(actual)
    if not actual:
        return 0.0
    predicted = list(predicted)[:k]
    hits = 0
    score = 0.0
    seen = set()
    for i, p in enumerate(predicted):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1.0)
        seen.add(p)
    return score / min(len(actual), k)


def mapk(actual: Mapping, predicted: Mapping, k: int = 5, users: Iterable | None = None) -> float:
    """Mean AP@K over `users` (default: every user that has ground truth).

    Users missing from `predicted` get an empty list, i.e. AP = 0.
    """
    users = list(actual.keys()) if users is None else list(users)
    if not users:
        return 0.0
    return sum(apk(actual.get(u, ()), predicted.get(u, ()), k) for u in users) / len(users)


def recall_at_n(actual: Mapping, candidates: Mapping, users: Iterable | None = None) -> float:
    """Share of ground-truth items that appear anywhere in each user's candidate pool (micro-average)."""
    users = list(actual.keys()) if users is None else list(users)
    found = total = 0
    for u in users:
        a = set(actual.get(u, ()))
        c = set(candidates.get(u, ()))
        found += len(a & c)
        total += len(a)
    return found / total if total else 0.0
