"""Report figures (SVG) from processed data and evidence files.

fig_activity.svg  weekly share of users with any session (logging gap) + fold timeline
fig_methods.svg   MAP@5 of every method on the validation fold F1

Run: .venv/bin/python -I src/figures.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import P, ROOT, submission_users  # noqa: E402

# reference palette (dataviz skill), validated: slot 1 blue, slot 2 orange on the light surface
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_WASH = "#cde2fb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8, "svg.fonttype": "none",
})
OUT = ROOT / "report" / "fig"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_activity():
    sess = pl.read_parquet(P / "sessions.parquet")
    sub = set(submission_users()["u"].to_list())
    n_users = sess["u"].n_unique()
    w = (sess.with_columns(pl.col("u").is_in(pl.Series(list(sub)).implode()).alias("is_sub"))
         .group_by(["week", "is_sub"]).agg(pl.col("u").n_unique().alias("n")).sort("week"))
    sub_share = {wk: n / len(sub) for wk, s, n in w.iter_rows() if s}
    oth_share = {wk: n / (n_users - len(sub)) for wk, s, n in w.iter_rows() if not s}
    weeks = sorted(sub_share)
    days = sess.with_columns(pl.from_epoch("s_ts", time_unit="s").dt.date().alias("d"))["d"].unique().sort()
    import datetime as _dt
    all_days = [days[0] + _dt.timedelta(n) for n in range((days[-1] - days[0]).days + 1)]
    missing = [str(d) for d in all_days if d not in set(days.to_list())]
    (ROOT / "evidence" / "weekly_activity.json").write_text(json.dumps({
        "share_active_submission_users": {str(k): round(v, 4) for k, v in sub_share.items()},
        "share_active_other_users": {str(k): round(v, 4) for k, v in oth_share.items()},
        "days_without_sessions": missing}, indent=1))
    fig, (ax, tl) = plt.subplots(2, 1, figsize=(3.45, 2.75), gridspec_kw={"height_ratios": [2.3, 1.0]}, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axvspan(19.5, 27.5, color="#f0efec", zorder=0)
    ax.text(23.5, 0.27, "logging gap", ha="center", va="bottom", color=INK2, fontsize=7)
    ax.plot(weeks, [sub_share[k] for k in weeks], color=BLUE, lw=1.6, solid_capstyle="round", label=f"submission users (n={len(sub)})")
    ax.plot(weeks, [oth_share[k] for k in weeks], color=ORANGE, lw=1.6, solid_capstyle="round",
            label=f"other users (n={n_users - len(sub)})")
    ax.set_ylim(0.25, 1.02)
    ax.set_ylabel("share active in week")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5, labelcolor=INK2, handlelength=1.4)
    # fold timeline
    tl.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        tl.spines[s].set_visible(False)
    tl.set_yticks([0, 1])
    tl.set_yticklabels(["test", "F1 (valid.)"], color=INK2)
    bars = [(1, 11, 18, BLUE_WASH, "history"), (1, 19, 30, BLUE, "target"),
            (0, 15, 19, BLUE_WASH, None), (0, 28, 30, BLUE_WASH, "history"), (0, 31, 42, BLUE, "target (Kaggle)")]
    for y, a, b, col, lab in bars:
        tl.barh(y, b - a + 1, left=a - 0.5, height=0.55, color=col, edgecolor=SURFACE, linewidth=1.5)
        if lab:
            tl.text((a + b) / 2, y, lab, ha="center", va="center", fontsize=6,
                    color=INK if col == BLUE_WASH else "#ffffff")
    tl.set_xlim(10.4, 42.6)
    tl.set_ylim(-0.6, 1.6)
    tl.set_xlabel("ISO week of 2020")
    tl.set_xticks([11, 15, 19, 23, 27, 31, 35, 39, 42])
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(OUT / "fig_activity.svg")
    plt.close(fig)


def fig_methods(hybrid_key: str, hybrid_file: str):
    m = json.loads((ROOT / "evidence" / "methods_F1.json").read_text())["scores"]
    als = json.loads((ROOT / "evidence" / "als_sweep_F1.json").read_text())
    best_als_key = max(als, key=lambda k: als[k][0])
    hyb = json.loads((ROOT / "evidence" / hybrid_file).read_text())
    hyb = hyb["scores"][hybrid_key] if hybrid_key else hyb
    rows = [
        ("CBF (course)", m["CBF_raw"]["map5_all"]),
        ("Popularity", m["POP"]["map5_all"]),
        ("Assoc. rules (course)", m["AR_s20_max"]["map5_all"]),
        ("MF / ALS, tuned (course)", als[best_als_key][0]),
        ("Repeat own history", m["REPEAT"]["map5_all"]),
        ("PPR (course)", m["PPR_d0.7"]["map5_all"]),
        ("Hybrid LightGBM ranker", hyb["map5_all"]),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ys = range(len(rows))
    ax.barh(list(ys), [v for _, v in rows], height=0.6, color=[BLUE if "Hybrid" not in n else ORANGE for n, _ in rows])
    for y, (_, v) in zip(ys, rows):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _ in rows], color=INK2)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("MAP@5 on F1, all 4,254 evaluation users")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_methods.svg")
    plt.close(fig)
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_activity()
    # the hybrid bar = exact CV of the configuration of submission 2 (5 seeds)
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    f = sys.argv[2] if len(sys.argv) > 2 else "cv_sub02_hybrid_lr31_r250_s5.json"
    print(fig_methods(key, f))
