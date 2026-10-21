"""Interval join of viewing sessions with the TV schedule.

For every session (user, channel, [s, e)) find all airings on the same channel
that overlap it and return the overlap in seconds.  Works on plain numpy arrays:
airings are sorted by (channel, start); a session's first candidate airing is the
first one whose running-max end is after s, the last is the last one starting
before e.  Pairs with zero overlap (possible when the schedule itself has
overlapping entries) are dropped.
"""
from __future__ import annotations

import numpy as np

SHIFT = np.int64(1) << np.int64(40)  # > any timestamp in seconds since epoch


def overlap_pairs(sess_ch, sess_s, sess_e, air_ch, air_s, air_e):
    """Return (session_idx, airing_idx, overlap_seconds) for all overlapping pairs.

    All inputs are int64 numpy arrays (channel codes as small non-negative ints,
    times in seconds).  Airings must be sorted by (channel, start).
    """
    air_ch = air_ch.astype(np.int64)
    air_s = air_s.astype(np.int64)
    air_e = air_e.astype(np.int64)
    order_ok = np.all(np.diff(air_ch * SHIFT + air_s) >= 0)
    if not order_ok:
        raise ValueError("airings must be sorted by (channel, start)")

    start_key = air_ch * SHIFT + air_s
    # running max of end inside each channel keeps the key monotone even if the
    # schedule has overlapping programmes
    end_key = air_ch * SHIFT + air_e
    end_key = np.maximum.accumulate(end_key)

    s_key = sess_ch.astype(np.int64) * SHIFT + sess_s.astype(np.int64)
    e_key = sess_ch.astype(np.int64) * SHIFT + sess_e.astype(np.int64)
    first = np.searchsorted(end_key, s_key, side="right")
    last = np.searchsorted(start_key, e_key, side="left")  # exclusive
    counts = np.maximum(last - first, 0)

    total = int(counts.sum())
    sess_idx = np.repeat(np.arange(len(sess_s), dtype=np.int64), counts)
    # airing index = first[session] + offset within the session's range
    offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    air_idx = np.repeat(first, counts) + offsets

    ov = np.minimum(sess_e[sess_idx], air_e[air_idx]) - np.maximum(sess_s[sess_idx], air_s[air_idx])
    same_ch = sess_ch[sess_idx] == air_ch[air_idx]
    keep = (ov > 0) & same_ch
    return sess_idx[keep], air_idx[keep], ov[keep]
