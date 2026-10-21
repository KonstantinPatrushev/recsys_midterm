"""Render report/report.html from the evidence files (every number comes from evidence/),
then print it to PDF with headless Chrome.

Run: .venv/bin/python -I src/build_report.py [--pdf]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
REP = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF_NAME = "patrushev_konstantin_midterm_report.pdf"


# English names of the shows quoted in Sec. 4.5 (the schedule titles are in Russian)
EN_TITLES = {
    "Губка Боб": "SpongeBob SquarePants", "Волшебные покровители": "The Fairly OddParents",
    "Элвиннн": "ALVINNN!!! and the Chipmunks", "Кунг-фу панда": "Kung Fu Panda: Legends of Awesomeness",
    "Пингвины из": "The Penguins of Madagascar", "Аватар": "Avatar: The Last Airbender",
    "Ниндзяго": "Ninjago: Masters of Spinjitzu", "Бен 10": "Ben 10", "Удивительный мир Гамбола": "The Amazing World of Gumball",
    "Октонавты": "Octonauts", "Свинка Пеппа": "Peppa Pig", "ТСН": "TSN (news, 1+1)", "Факты": "Fakty (news, ICTV)",
    "Чистоnews": "Chysto News", "Маша и Медведь": "Masha and the Bear", "Санджей и Крейг": "Sanjay and Craig",
    "Аладдин": "Aladdin", "Викна-Новости": "Vikna-Novyny (news, STB)", "Сегодня": "Segodnya (news)",
}


def j(name):
    return json.loads((EV / name).read_text())


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def f5(x):
    return f"{x:.5f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def n0(x):
    return f"{int(x):,}"


def esc(s):
    return html.escape(str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    prep, eda, tz = j("prep.json"), j("eda.json"), j("tz_alignment.json")
    meth, als = j("methods_F1.json"), j("als_sweep_F1.json")
    hyb = j("hybrid_ablation_F1.json")
    ex1, ex2, ex3 = j("experiments_F1.json"), j("experiments2_F1.json"), j("experiments3_F1.json")
    aug = j("experiments4_augment_F1.json")
    rob = j("robustness.json")
    diag = j("diagnostics.json")
    biz = j("business_two_users.json")
    cand = j("candidates_final.json")
    tb_v1 = j("tiebreak_sensitivity_F1.json")
    rep1 = j("reproduce_sub01.json")
    tm, ti = j("test_metrics.json"), j("test_intervals.json")
    info2 = j("sub02_hybrid_lr31_r250_s5_info.json")
    cv1, cv2 = j("cv_sub01_ranker_v1.json"), j("cv_sub02_hybrid_lr31_r250_s5.json")
    act = j("weekly_activity.json")
    gap_weeks = [str(w) for w in range(20, 28)]
    gap_min_sub = min(act["share_active_submission_users"][w] for w in gap_weeks)
    gap_min_oth = min(act["share_active_other_users"][w] for w in gap_weeks)
    no_days = act["days_without_sessions"]
    subs = list(csv.DictReader(open(EV / "kaggle_submissions.csv", encoding="utf-8")))
    subs = {r["fileName"].replace(".csv", ""): r for r in subs}
    lb = list(csv.DictReader(open(EV / "kaggle_meta" / "public_leaderboard_2020_final.csv", encoding="utf-8-sig")))
    lb_pub = {r["TeamName"]: float(r["Score"]) for r in lb}

    m = meth["scores"]
    best_als_key = max(als, key=lambda k: als[k][0])
    als_best = als[best_als_key]
    als_default = m["ALS_f64_r0.05_a10.0"]
    F1, F0, F2 = rob["F1"]["scores"], rob["F0"]["scores"], rob["F2"]["scores"]
    proxy = rob.get("test_history_proxy_target_28_30", {})
    tbh = rob["F1_tiebreak_sensitivity_hybrid"]
    rnd_all = [tbh[k]["map5_all"] for k in tbh if k.startswith("random")]
    rnd_sub = [tbh[k]["map5_sub"] for k in tbh if k.startswith("random")]
    hab, exp_ = biz["habitual"], biz["explorer"]
    nov = biz["novelty_by_method_sub_users_F1"]
    sel = biz["selection"]
    s1, s2 = subs["sub01_ranker_v1"], subs["sub02_hybrid_lr31_r250_s5"]
    pub1, prv1 = float(s1["publicScore"]), float(s1["privateScore"])
    pub2, prv2 = float(s2["publicScore"]), float(s2["privateScore"])
    winner_pub = lb_pub["Magic City"]
    tz0 = tz["0"][1]
    tz_other = max(v[1] for k, v in tz.items() if k != "0")
    ties = eda["val19_30_full"]["share_users_with_tie_at_5th"]
    gap_sub = eda["sub_users_activity"]
    by_nov = {d["novel"]: d for d in diag["error_by_novel_truth_items_(3=3+)"]}
    n_eval = rob["F1"]["n_users"]
    n_sub = rob["F1"]["n_sub"]

    def ap_row(name):
        return hab["methods"][name]["ap5"], exp_["methods"][name]["ap5"]

    def hits(rec, name):
        return sum(1 for x in rec["methods"][name]["top5"] if x["hit"])

    def title(x):
        t = (x["title"] or "").replace("М/с ", "").replace("'", "").strip(" .")
        for ru, en in EN_TITLES.items():
            if t.startswith(ru):
                return esc(en)
        return esc(t)

    shot = REP / "fig" / "leaderboard.png"
    shot_html = (f'<img src="fig/leaderboard.png" style="width:100%">' if shot.exists() else
                 '<div style="border:1px dashed #000;padding:18pt;text-align:center">Leaderboard screenshot '
                 '(Kaggle &rarr; Submissions, both scored files visible) &mdash; to be inserted</div>')

    rows_methods = [
        ("Popularity (users who fully watched, airing shows only)", m["POP"], None, None),
        ("CBF &mdash; course method", m["CBF_raw"], F0["cbf"], F2["cbf"]),
        ("Association rules (support &ge; 20, max conf.) &mdash; course", m["AR_s20_max"], F0["ar"], F2["ar"]),
        ("MF: implicit ALS, default (64 f., &alpha;=10, &lambda;=0.05)", als_default, None, None),
        ("MF: implicit ALS, tuned (64 f., &alpha;=1, &lambda;=10) &mdash; course", {"map5_all": als_best[0], "map5_sub": als_best[1]}, F0["als"], F2["als"]),
        ("Repeat own history (reference)", m["REPEAT"], F0["repeat"], F2["repeat"]),
        ("PPR, damping 0.7 &mdash; course", m["PPR_d0.7"], F0["ppr"], F2["ppr"]),
        ("RRF of PPR+AR+ALS+CBF+Repeat (no learning)", F1["RRF_ppr+ar+als+cbf+repeat"], F0["RRF_ppr+ar+als+cbf+repeat"], F2["RRF_ppr+ar+als+cbf+repeat"]),
        ("<b>Hybrid LightGBM ranker</b> (3 seeds)", F1["hybrid_ranker"], F0["hybrid_ranker"], F2["hybrid_ranker"]),
    ]

    def cell(d, key):
        return f3(d[key]) if d else "&ndash;"

    methods_tr = "\n".join(
        f"<tr><td>{n}</td><td class=n>{cell(a,'map5_all')}</td><td class=n>{cell(a,'map5_sub')}</td>"
        f"<td class=n>{cell(b,'map5_all')}</td><td class=n>{cell(c,'map5_all')}</td></tr>"
        for n, a, b, c in rows_methods)

    biz_rows = []
    for name in ["POP", "REPEAT", "CBF", "AR", "MF (ALS)", "PPR", "Hybrid (LGBM)"]:
        a, b = ap_row(name)
        biz_rows.append(f"<tr><td>{esc(name)}</td><td class=n>{hits(hab, name)} / {f3(a)}</td>"
                        f"<td class=n>{hits(exp_, name)} / {f3(b)}</td><td class=n>{pct(nov[name]['share_new_in_top5'])}</td>"
                        f"<td class=n>{pct(nov[name]['precision_of_new_items'])}</td></tr>")
    biz_tr = "\n".join(biz_rows)
    hab_truth = ", ".join(f"{title(x)} ({x['full_views']})" for x in hab["truth_19_30"])
    exp_truth = ", ".join(f"{title(x)} ({x['full_views']}{'' if x['seen_in_history'] else ', new'}"
                          f"{', ' + esc(x['channel']) if not x['channel_watched_in_history'] else ''})" for x in exp_["truth_19_30"])
    hab_test = ", ".join(title(x) for x in hab.get("test_31_42_final_top5", []))
    exp_test = ", ".join(title(x) for x in exp_.get("test_31_42_final_top5", []))
    hab_hist = ", ".join(f"{title(x)} ({x['full_views']})" for x in hab["top_history_shows"][:3])
    exp_hist = ", ".join(f"{title(x)} ({x['full_views']})" for x in exp_["top_history_shows"][:4])
    hab_ch = ", ".join(f"{esc(c['channel'])} {pct(c['share'], 0)}" for c in hab["top_channels_by_time"][:4])
    exp_ch = ", ".join(f"{esc(c['channel'])} {pct(c['share'], 0)}" for c in exp_["top_channels_by_time"][:4])
    hyb_hab = ", ".join(title(x) + (" &#10003;" if x["hit"] else "") for x in hab["methods"]["Hybrid (LGBM)"]["top5"])

    proxy_txt = ""
    if proxy:
        pa, pb = proxy.get("clean8_12_19", {}), proxy.get("contig8_20_27", {})
        if pa and pb:
            proxy_txt = (f" A proxy on the clean target weeks 28&ndash;30 (model trained on half of the F1 users, scored on the other half) "
                         f"gives MAP@5 = {f3(pa['map5_evalB'])} with the 8 most recent clean weeks (12&ndash;19) as history and "
                         f"{f3(pb['map5_evalB'])} with the 8 contiguous weeks 20&ndash;27, which fall inside the gap.")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Midterm Report &mdash; SWEET.TV TV-show recommender</title>
<style>
  a {{ color: #000; text-decoration: none; }}
  @page {{ size: A4; margin: 13mm 14mm 13mm 14mm; }}
  body {{ font-family: "Times New Roman", "Liberation Serif", serif; font-size: 9.9pt; line-height: 1.24; color: #000; margin: 0; }}
  h1 {{ font-size: 14pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }}
  .meta {{ text-align: center; font-size: 9.5pt; margin-bottom: 6pt; }}
  h2 {{ font-size: 10.6pt; margin: 6pt 0 2pt; text-transform: uppercase; letter-spacing: .3pt; break-after: avoid; }}
  h3 {{ font-size: 10pt; margin: 4pt 0 1.5pt; font-style: italic; font-weight: normal; break-after: avoid; }}
  p {{ margin: 0 0 3pt; text-align: justify; }}
  ul, ol {{ margin: 0 0 3pt; padding-left: 15pt; }}
  li {{ margin-bottom: 1pt; text-align: justify; }}
  .abstract {{ font-size: 9.4pt; margin: 0 5mm 3pt; }}
  code {{ font-family: "DejaVu Sans Mono", "Menlo", monospace; font-size: 7.8pt; }}
  table {{ border-collapse: collapse; margin: 3pt auto 2pt; font-size: 8.4pt; }}
  th, td {{ border-top: 1px solid #000; border-bottom: 1px solid #000; padding: 1.1pt 4pt; text-align: left; vertical-align: top; }}
  th {{ font-weight: bold; }}
  td.n, th.n {{ text-align: right; white-space: nowrap; }}
  .cap {{ font-size: 8.4pt; text-align: center; margin: 1.5pt 0 5pt; }}
  .refs p {{ text-align: left; padding-left: 18pt; text-indent: -18pt; font-size: 8.7pt; margin-bottom: 1.8pt; }}
  .avoid {{ break-inside: avoid; }}
  .two {{ display: flex; gap: 10pt; align-items: flex-start; }}
  .two > div {{ flex: 1; }}
  .todo {{ background: #fff3a8; }}
  img.fig {{ width: 100%; }}
</style>
</head>
<body>

<h1>Habits Meet the Broadcast Schedule: a Hybrid Ranker for<br>SWEET.TV TV-Show Recommendation (Kaggle Midterm)</h1>
<div class="meta">
  Student: Konstantin Patrushev &nbsp;&nbsp; Team: Individual<br>
  Email: kapatrushev@edu.hse.ru &nbsp;&nbsp; Date: 2026-10-21<br>
  Assignment: Midterm Quiz &mdash; TV Program Recommendation (Kaggle <code>sweettv-tv-program-recommender</code>)
</div>

<h2>Abstract</h2>
<p class="abstract">SWEET.TV asks for the five TV shows each of 1,260 users will most often watch to at least 80% of an airing
during twelve future weeks, scored by MAP@5 [1]. I rebuilt this target exactly from raw viewing sessions and the programme
schedule, validated on a time split with the same 12-week horizon, and trained a LightGBM LambdaRank ranker whose candidates and
features come from the user's history, the future schedule and four course methods (content-based filtering, association
rules, ALS matrix factorization, personalized PageRank). On validation the hybrid reaches MAP@5 = {f3(cv2['map5_all'])} against
{f3(m['PPR_d0.7']['map5_all'])} for the best course method, PPR; on Kaggle it scores {f5(pub2)} public and {f5(prv2)} private,
and my first ranker {f5(pub1)} public, above the original winner's {f5(winner_pub)}. The lesson: for linear TV, when a show
will air, crossed with when the user watches, carries more signal than collaborative similarity.</p>
<p class="abstract"><i>Index Terms</i> &mdash; recommender systems, linear television, learning to rank, personalized PageRank,
implicit feedback</p>

<h2>1. Introduction</h2>
<p><b>Problem.</b> The competition [1] provides {n0(prep['log_rows'])} channel-viewing sessions of {n0(eda['n_users_logs'])} users
in weeks 11&ndash;30 of 2020 and the programme schedules of weeks 11&ndash;42. The hidden truth is built per user: screen time
of an airing = seconds watched / airing duration; airings with screen time &ge; 0.8 count; the five shows with the most such
airings in weeks 31&ndash;42 are the answer; every scored user has exactly five, and public/private split the users 50/50 [1].
The course adds: at least two of CBF, association rules, MF and PPR, compared on one validation protocol; programme metadata
may be used, future viewing may not; and an analysis of two users.</p>
<p><b>Motivation.</b> For an operator the five shows are the &ldquo;your shows&rdquo; row and the reminder list. Unlike an on-demand
catalogue, a linear show can be watched only when it is on air, so the schedule of weeks 31&ndash;42 is part of the answer.</p>
<p><b>Concrete example.</b> Household u{hab['u']} fully watched <i>SpongeBob</i> {hab['top_history_shows'][0]['full_views']} times in
weeks 11&ndash;18, mostly around 11:00&ndash;15:00 on {esc(hab['top_history_shows'][0]['channel'])}; in weeks 19&ndash;30 it was its
#1 show with {hab['truth_19_30'][0]['full_views']} full views. The hybrid's top-5 for this user holds three of its five true
shows (AP@5 = {f3(hab['methods']['Hybrid (LGBM)']['ap5'])}, Sec. 4.5).</p>
<p><b>Contributions.</b></p>
<ul>
  <li>An exact rebuild of the organisers' target and a validation split whose estimate for the submission users ({f3(cv1['map5_sub'])})
  matched the first public score ({f3(pub1)}) within {f3(abs(pub1 - cv1['map5_sub']))}.</li>
  <li>The four course methods on one protocol and three folds, a hybrid ranker that uses them, data diagnostics (a logging gap,
  tie-breaks), and a two-user business analysis.</li>
</ul>

<h2>2. Related Work</h2>
<p>Implicit-feedback matrix factorization (ALS with confidence weights) was introduced on TV-viewing data [2]. Personalized
PageRank comes from topic-sensitive PageRank [3]; association rules from Agrawal et al. [4]; content-based recommendation is
surveyed by Pazzani and Billsus [5]. For linear TV, Turrin et al. [6] predict what a user watches from the time of day and
the channel, and Watch-It-Next [7] conditions on the programme being watched. The ranker is LambdaMART [8] as implemented in
LightGBM [9]. Kaggle practice used here: validation that mirrors the test split and trusting it over a small public board
[10], and two-stage candidate generation with a GBDT re-ranker, as in the H&amp;M winning solution [11]. The 2nd-place team of
this competition also used candidates plus LightGBM, with LightFM for candidates [12].</p>
<p><b>Artifacts.</b> Code, tests, evidence: <code>github.com/KonstantinPatrushev/recsys_midterm</code> (see Sec. 6).</p>
<p><b>Alternatives considered.</b> (i) A per-airing classifier P(full view) summed over future airings: closest to the target, but
about ten times more rows; the slot-habit features approximate it. (ii) Sequence models (SASRec-type): {n0(eda['n_users_logs'])}
users whose behaviour is driven by the schedule rather than by item order. (iii) Rank fusion of the course methods without
learning: evaluated (Table 1) and clearly worse than the ranker.</p>

<h2>3. Method</h2>
<p><b>From sessions to the target.</b> Each session is joined with the airings that overlap it on the same channel
(interval join, checked against brute force on {ti['pairs_checked']:,} pairs). Seconds watched per (user, airing) are summed over
sessions, giving {n0(prep['user_airing_views'])} user&ndash;airing views, {n0(prep['full_views_sum_ge_0.8'])} of them with screen
time &ge; 0.8 (taking only the longest single session instead gives {n0(prep['full_views_max_ge_0.8'])}). Airings with
<code>tv_show_id = 0</code> are removed first, as the organisers did ({n0(prep['schedule_rows_nonzero'])} airings remain). The
local truth applies the organisers' rule to any week window; ties at the 5th place are broken by ascending id (Sec. 4.3).</p>
<p><b>Data checks.</b> (a) <code>user_id</code> exceeds 2<sup>63</sup>, so ids are read as unsigned 64-bit. (b) Logs and schedule
share a clock: {pct(tz0, 2)} of session ends fall within 60 s after an airing end at zero shift, at most {pct(tz_other, 2)} at
&plusmn;1&ndash;3 h. (c) Weeks 20&ndash;27 lost much of the logging (Fig. 1): the weekly share of active users falls to
{gap_min_sub:.2f} for submission users and {gap_min_oth:.2f} for the others, and {', '.join(no_days)} has no sessions at all. (d) {pct(ties, 0)} of users have a tie between their 5th and 6th show in weeks 19&ndash;30.</p>
<div class="two avoid">
<div><img class="fig" src="fig/fig_activity.svg">
<p class="cap">Fig. 1. Share of users with any session per week; the logging gap; history and target windows of the
validation fold F1 and of the test.</p></div>
<div><img class="fig" src="fig/fig_methods.svg">
<p class="cap">Fig. 2. MAP@5 on F1 ({n0(n_eval)} users). The hybrid bar is the 5-seed
configuration of submission 2.</p></div>
</div>
<p><b>Validation.</b> Fold F1 takes features from weeks 11&ndash;18 and truth from weeks 19&ndash;30 &mdash; the test's 12-week
horizon; it scores the {n0(n_eval)} users with at least five qualifying shows, {n0(n_sub)} of whom are submission users. The
ranker is evaluated with 5-fold cross-validation grouped by user. Folds F0 (11&ndash;15 &rarr; 16&ndash;19) and F2
(11&ndash;27 &rarr; 28&ndash;30) check that conclusions hold in other periods. For the test, features come from the eight most
recent <i>clean</i> weeks (15&ndash;19 and 28&ndash;30): the same length as F1's history, skipping the gap; spring weeks also keep shows
that pause in summer and return in autumn. Eight submission users without views there use weeks 11&ndash;18.</p>
<p><b>Candidates.</b> Shows the user watched to &ge; 20% in history that air in the target window; the 30 most-watched airing
shows; the 30 best shows by slot habit; the top 20 of each course method. Median {cand['F1_candidates_per_user_median']:.0f}
candidates per user; {pct(cand['F1_recall_all'])} of true shows are among them ({pct(cand['F1_recall_sub'])} for submission users).</p>
<p><b>Features ({len(info2['features'])}).</b> User &times; show: full views over the last 1/2/4/8 weeks and per week, recency,
partial views, share of the user's views. Schedule-aware: airings in the target window on the channels where the user watched
the show (the top feature by gain), and the expected number of full views
&Sigma;<sub>future airings</sub> P(user fully watches what starts on that channel, weekday and hour). Show popularity and
completion, user activity, category and genre, and each course method's score and rank.</p>
<p><b>Course methods.</b> <i>CBF</i>: TF-IDF show vectors (category, genres, shares of airings by channel, hour and weekend,
duration, decade); profile = count-weighted sum; cosine. <i>AR</i>: pairwise rules A&rarr;B over users' sets of fully watched
shows, support &ge; 20 users; score = max confidence. <i>MF</i>: implicit ALS [2] on 1+&alpha;&middot;log(1+count), chosen from
{len(als)} settings. <i>PPR</i>: random walk with restart on the user&ndash;show graph, edge weight log(1+count), damping 0.7
chosen from {{0.3, 0.5, 0.7, 0.85}}. All four see only the history window.</p>
<p><b>Ranker and configuration.</b> LightGBM LambdaRank, label = &ldquo;in the user's true top 5&rdquo;, 31 leaves, learning rate
0.05, 250 rounds, 5 seeds averaged by within-user rank; deterministic mode. Python 3.12, polars 2.0.0, NumPy 2.5.4, SciPy
1.18.1, LightGBM 4.7.0, implicit 0.7.3, scikit-learn 1.9.1, Kaggle CLI 2.2.4; Apple M5, 24 GB.</p>
<p><b>Kaggle practices applied</b> [10], [11]: read the data page through the API (it holds the exact target, absent from the
task slides); a validation fold that mirrors the test horizon; an early calibration submission; candidate recall measured
separately from ranking; paired comparisons on fixed folds with seed noise and a bootstrap interval; seed averaging; a
deterministic pipeline; a pre-planned budget of five submissions.</p>

<h2>4. Experiments</h2>
<h3>4.1 The course methods on one protocol</h3>
<div class="avoid">
<table>
<tr><th>Method (top 5 among shows airing in the target window)</th><th class=n>F1 all</th><th class=n>F1 sub.</th><th class=n>F0 all</th><th class=n>F2 all</th></tr>
{methods_tr}
</table>
<p class="cap">Table 1. MAP@5. F1 = weeks 11&ndash;18 &rarr; 19&ndash;30 ({n0(n_eval)} users; &ldquo;sub.&rdquo; = {n0(n_sub)} submission users);
F0 = 11&ndash;15 &rarr; 16&ndash;19 ({n0(rob['F0']['n_users'])} users); F2 = 11&ndash;27 &rarr; 28&ndash;30 ({n0(rob['F2']['n_users'])} users).
Files: <code>methods_F1.json</code>, <code>als_sweep_F1.json</code>, <code>robustness.json</code>.</p>
</div>
<p>The ranking is the same on all three folds: hybrid &gt; PPR &gt; repeat &asymp; ALS &gt; association rules &gt; CBF. PPR is the
best course method: the walk keeps most mass on the user's own shows and adds shows of similar users. ALS with common default
settings scores {f3(als_default['map5_all'])}, below popularity ({f3(m['POP']['map5_all'])}); strong regularization raises it to
{f3(als_best[0])}. CBF alone is weakest: it recommends similar cartoons or series rather than the exact show.</p>

<h3>4.2 What the hybrid gains, and where the gain stops</h3>
<p>On F1 the base ranker (habit and schedule features only) reaches {f5(hyb['base']['map5_all'])}; adding the course-method scores
gives {f5(hyb['base+method_feats']['map5_all'])}, and their top-20 candidates raise recall from {pct(hyb['base']['recall_all'])} to
{pct(hyb['base+method_feats+method_cands20']['recall_all'])} but not MAP@5 ({f5(hyb['base+method_feats+method_cands20']['map5_all'])}).
Simpler trees helped: 31 leaves {f5(ex1['lambdarank_leaves31']['map5_all'])} vs 63 leaves {f5(ex1['lambdarank']['map5_all'])} vs
127 leaves {f5(ex1['lambdarank_leaves127_mdl100']['map5_all'])}; 150&ndash;300 rounds are a plateau
({f5(ex3['scores']['lr_l31_r150']['map5_all'])}&ndash;{f5(ex3['scores']['lr_l31_r300']['map5_all'])}). Three seeds of the same model score {f5(min(ex3['scores'][k]['map5_all'] for k in ['lr_l31_r250','lr_l31_r250_seed1','lr_l31_r250_seed2']))}&ndash;{f5(max(ex3['scores'][k]['map5_all'] for k in ['lr_l31_r250','lr_l31_r250_seed1','lr_l31_r250_seed2']))};
their rank average scores {f5(ex3['scores']['ENS_lr3_r250']['map5_all'])}. A paired bootstrap over users even calls the difference between two
single seeds significant (95% CI [{f4(ex2['paired_bootstrap_vs_lr_l31_seed0']['lr_l31_seed1']['ci95'][0])}, {f4(ex2['paired_bootstrap_vs_lr_l31_seed0']['lr_l31_seed1']['ci95'][1])}]),
so I only accept changes larger than the seed spread.
Binary, XE-NDCG and Poisson objectives, up-weighting submission users, training only on fully logged users
({f5(ex3['scores']['lr_l31_r250_trainclean']['map5_all'])}) and adding a second training window
(&plus;{f4(aug['F1+F3']['map5_all'] - aug['F1_only']['map5_all'])}) did not help beyond seed noise. Error analysis of the final
configuration: when all five true shows are among the candidates ({pct(diag['share_users_all_5_in_candidates'], 0)} of users)
MAP@5 is {f3(diag['map5_if_all_5_in_candidates'])}; users whose truth has no show new to them reach {f3(by_nov[0]['map5'])}, those with three
or more new shows ({n0(by_nov[3]['users'])} users) only {f3(by_nov[3]['map5'])}. The bottleneck is novelty, not ranking among known shows.</p>

<h3>4.3 Kaggle submissions and the tie-break</h3>
<div class="two avoid">
<div>
<table>
<tr><th>Submission</th><th class=n>CV all</th><th class=n>CV sub.</th><th class=n>Public</th><th class=n>Private</th></tr>
<tr><td>1: ranker v1 (63 leaves, no method features)</td><td class=n>{f5(cv1['map5_all'])}</td><td class=n>{f5(cv1['map5_sub'])}</td><td class=n><b>{f5(pub1)}</b></td><td class=n>{f5(prv1)}</td></tr>
<tr><td>2: hybrid, 31 leaves, 250 rounds, 5 seeds</td><td class=n>{f5(cv2['map5_all'])}</td><td class=n>{f5(cv2['map5_sub'])}</td><td class=n>{f5(pub2)}</td><td class=n><b>{f5(prv2)}</b></td></tr>
<tr><td>Original winner, 2020 [1]</td><td class=n>&ndash;</td><td class=n>&ndash;</td><td class=n>{f5(winner_pub)}</td><td class=n>0.42912</td></tr>
</table>
<p class="cap">Table 2. CV on F1 and Kaggle scores (<code>kaggle_submissions.csv</code>, <code>cv_*.json</code>).
Both description fields read &ldquo;Константин Патрушев&rdquo;.</p>
</div>
<div>{shot_html}<p class="cap">Fig. 3. Kaggle submissions page.</p></div>
</div>
<p>Submission 1, sent early as a calibration point, scored {f5(pub1)} public against a CV of {f5(cv1['map5_sub'])} on the
submission users. Submission 2 gained {f4(cv2['map5_sub'] - cv1['map5_sub'])} on CV; the test moved by {f4(pub2 - pub1)} public and
{f4(prv2 - prv1)} private, about {f4(((pub2 + prv2) - (pub1 + prv1)) / 2)} on all 1,260 users: each half holds about 630 users,
so a file's two halves already differ by {f4(pub1 - prv1)}. The undocumented tie-break matters as much: the same out-of-fold
predictions score {f3(tbh['id_asc']['map5_all'])} against truth with ties broken by ascending id, {f3(min(rnd_all))}&ndash;{f3(max(rnd_all))}
with random ties and {f3(tbh['id_desc']['map5_all'])} with descending id. The test scores lie between the ascending-id and
random estimates, so the organisers' rule is not identified; I keep ascending id, the order a stable sort of a
<code>groupby</code> count table produces.</p>

<h3>4.4 Verification</h3>
<ul>
  <li>MAP@5 equals {len(tm['cases'])} hand-computed cases and an independent re-implementation on {n0(tm['random_pairs'])} random pairs
  (max. difference {tm['max_abs_diff_vs_independent']:.1e}); the interval join equals brute force on {ti['trials']} random schedules.</li>
  <li>Submission 1 was rebuilt from scratch with the current code: identical SHA-256 (<code>reproduce_sub01.json</code>).</li>
  <li>By hand, household u{hab['u']} (Sec. 4.5): the hybrid's list hits at ranks 1, 2 and 3, so AP@5 = (1/1 + 2/2 + 3/3)/5 =
  {f3(hab['methods']['Hybrid (LGBM)']['ap5'])}, as the code reports.</li>
  <li>Every submitted file has 1,260 rows in the sample's order and five distinct ids, all airing in weeks 31&ndash;42
  (<code>tests/check_submission.py</code>); no user viewing after week 30 is read anywhere in the code.</li>
</ul>

<h3>4.5 Business analysis: two users</h3>
<p><b>Selection.</b> Not hand-picked: among the {sel['n_typical_sub_users']} submission users with F1 truth and typical activity
(the interquartile range, {sel['typical_activity_range_full_views_11_18'][0]:.0f}&ndash;{sel['typical_activity_range_full_views_11_18'][1]:.0f}
full views in weeks 11&ndash;18), one user was drawn at random (seed {sel['seed']}) from the <i>habitual</i> segment (all five
true shows already watched; {sel['n_habitual']} users) and one from the <i>explorer</i> segment (&ge; 2 true shows never fully
watched; {sel['n_explorer']} users). Both draws turned out to be households with children's channels.</p>
<div class="avoid">
<table>
<tr><th>Method</th><th class=n>u{hab['u']} hits / AP@5</th><th class=n>u{exp_['u']} hits / AP@5</th><th class=n>new shows in top-5*</th><th class=n>their precision*</th></tr>
{biz_tr}
</table>
<p class="cap">Table 3. F1 lists of the two users; * over all {n0(n_sub)} submission users (<code>business_two_users.json</code>).</p>
</div>
<p><b>u{hab['u']}, habitual.</b> History: {hab_hist}; channels {hab_ch}. Truth 19&ndash;30: {hab_truth}. <i>Familiar
preferences</i> dominate: every method except popularity puts SpongeBob first. The hybrid ({hyb_hab}) misses Kung-Fu Panda and
Penguins, watched only a few times before, and fills ranks 4&ndash;5 with news an adult in the same household watches &mdash;
one account mixes several viewers. CBF finds Penguins by similarity ({hits(hab, 'CBF')} hits). Test list (31&ndash;42):
{hab_test}.</p>
<p><b>u{exp_['u']}, explorer.</b> History: {exp_hist}; channels {exp_ch}. Truth 19&ndash;30: {exp_truth}. Three of the four
<i>new discoveries</i> air on Cartoon Network, a channel this household never watched in weeks 11&ndash;18 (first full view in
week 20). No method can propose a channel the user never touched; CBF offers other cartoons, not these. Once the shows enter
the history (weeks 28&ndash;30), the final model ranks them: {exp_test}.</p>
<p><b>Limitations and strategy.</b> Household accounts, channel-level cold start, and the season change from summer to the
school year in weeks 31&ndash;42. The hybrid is the strategy: best MAP@5 on F1, F0 and F2 (Table 1), and its new-show suggestions
are right {pct(nov['Hybrid (LGBM)']['precision_of_new_items'])} of the time against {pct(nov['CBF']['precision_of_new_items'])} for CBF.
Product use: a &ldquo;your shows this week&rdquo; row with airtime reminders; one discovery slot only when a new show scores
high; a children's profile to separate household viewers.</p>

<h2>5. Discussion</h2>
<p><b>Failure case.</b> Two identical runs of candidate generation printed <code>features: 392690 rows</code> and
<code>features: 392686 rows</code>. <i>Root cause:</i> ties &mdash; candidates sorted by slot score without a tie-breaker,
<code>mode().first()</code> on multi-modal channels, and ordinal ranks over a row order that depends on hash joins. <i>Fix:</i>
total orders (ties broken by show id), <code>mode().min()</code>, sorting before ranking, deterministic LightGBM. <i>Verification:</i>
two builds are <code>equal()</code> ({n0(391458)} rows each) and submission 1 rebuilds byte-for-byte. A second bug: users without
history got method &ldquo;candidates&rdquo; from all-zero scores, which hid the fallback history (it covered 2 users instead of 8);
candidates now come only from users with history, and the fallback count is back to 8.</p>
<p><b>What worked.</b> Reading the data page via the API (the exact target); a fold with the test's horizon; the schedule as
features; seed averaging.</p>
<p><b>What surprised me or did not work.</b> ALS with default settings lost to popularity; the course methods add little once
habit and schedule features exist; a tie-break rule can move MAP@5 by {f3(tbh['id_asc']['map5_all'] - tbh['id_desc']['map5_all'])};
a CV gain of {f4(cv2['map5_sub'] - cv1['map5_sub'])} became {f4(((pub2 + prv2) - (pub1 + prv1)) / 2)} on the test.</p>
<p><b>Next improvement.</b> Model each future airing (P(full view) from slot, channel and show history) and sum per show, and add
a &ldquo;recently adopted channel&rdquo; signal for explorers like u{exp_['u']}.</p>

<h2>6. AI Usage Disclosure</h2>
<p><b>AI tools used.</b> Claude Code (VS Code extension) with Claude Opus 5.5, one session; it also ran sub-agents for the
literature and Kaggle-practice search.</p>
<p><b>How AI was used.</b> Code generation (data preparation, methods, ranker, diagnostics, report builder); data analysis and
debugging; search and verification of references through Crossref, OpenAlex and the publishers' pages; Kaggle submissions
through the API; draft of this report, rendered from <code>evidence/</code> by <code>src/build_report.py</code>.</p>
<p><b>What I personally verified.</b> <span class="todo">[to be completed by the student &mdash; keep only checks actually done,
e.g. the hand computation in Sec. 4.4, Tables 1&ndash;2 against <code>evidence/</code>, the leaderboard screenshot.]</span></p>
<p><b>What I trusted without verification.</b> <span class="todo">[to be completed by the student.]</span></p>
<p><b>Session log.</b> <code>patrushev_konstantin_midterm_session.json</code> is the unmodified Claude Code session log.</p>

<h2>References</h2>
<div class="refs">
<p>[1] &ldquo;sweet.tv &ndash; TV program recommender,&rdquo; Kaggle Community Prediction Competition, 2020. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender. [Accessed: 2026-10-21].</p>
<p>[2] Y. Hu, Y. Koren, and C. Volinsky, &ldquo;Collaborative filtering for implicit feedback datasets,&rdquo; in Proc. 8th IEEE Int. Conf. Data Mining (ICDM), Pisa, Italy, 2008, pp. 263&ndash;272, doi: 10.1109/ICDM.2008.22.</p>
<p>[3] T. H. Haveliwala, &ldquo;Topic-sensitive PageRank,&rdquo; in Proc. 11th Int. Conf. World Wide Web (WWW), Honolulu, HI, USA, 2002, pp. 517&ndash;526, doi: 10.1145/511446.511513.</p>
<p>[4] R. Agrawal, T. Imieli&nacute;ski, and A. Swami, &ldquo;Mining association rules between sets of items in large databases,&rdquo; in Proc. ACM SIGMOD Int. Conf. Manage. Data, Washington, DC, USA, 1993, pp. 207&ndash;216, doi: 10.1145/170035.170072.</p>
<p>[5] M. J. Pazzani and D. Billsus, &ldquo;Content-based recommendation systems,&rdquo; in <i>The Adaptive Web</i> (LNCS, vol. 4321), P. Brusilovsky, A. Kobsa, and W. Nejdl, Eds. Berlin, Germany: Springer, 2007, pp. 325&ndash;341, doi: 10.1007/978-3-540-72079-9_10.</p>
<p>[6] R. Turrin, A. Condorelli, P. Cremonesi, and R. Pagano, &ldquo;Time-based TV programs prediction,&rdquo; presented at the 1st Workshop Recommender Syst. Television Online Video (RecSysTV), ACM RecSys 2014, Foster City, CA, USA, Oct. 2014. [Online]. Available: https://recsys.deib.polimi.it/wp-content/uploads/Time-basedTVprogramsprediction.pdf</p>
<p>[7] M. Aharon, E. Hillel, A. Kagian, R. Lempel, H. Makabee, and R. Nissim, &ldquo;Watch-It-Next: A contextual TV recommendation system,&rdquo; in <i>Machine Learning and Knowledge Discovery in Databases</i> (ECML PKDD 2015, Part III, LNCS, vol. 9286). Cham, Switzerland: Springer, 2015, pp. 180&ndash;195, doi: 10.1007/978-3-319-23461-8_12.</p>
<p>[8] C. J. C. Burges, &ldquo;From RankNet to LambdaRank to LambdaMART: An overview,&rdquo; Microsoft Research, Redmond, WA, USA, Tech. Rep. MSR-TR-2010-82, Jun. 2010.</p>
<p>[9] G. Ke et al., &ldquo;LightGBM: A highly efficient gradient boosting decision tree,&rdquo; in Proc. Adv. Neural Inf. Process. Syst. (NIPS), vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html</p>
<p>[10] K. Onodera, T. Viel, G. Titericz, and C. Deotte, &ldquo;The Kaggle Grandmasters Playbook: 7 battle-tested modeling techniques for tabular data,&rdquo; NVIDIA Technical Blog, Sep. 18, 2025. [Online]. Available: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/. [Accessed: 2026-10-21].</p>
<p>[11] senkin13 and 30CrMnSiA, &ldquo;1st place solution,&rdquo; H&amp;M Personalized Fashion Recommendations, Kaggle discussion, 2022. [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070. [Accessed: 2026-10-21].</p>
<p>[12] N. Melentev, &ldquo;Two-layered recommender system methodology: a prize-winning solution,&rdquo; Towards Data Science, Mar. 20, 2021. [Online]. Available: https://web.archive.org/web/20250827164704/https://towardsdatascience.com/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156/. [Accessed: 2026-10-21].</p>
</div>
</body>
</html>
"""
    REP.mkdir(exist_ok=True)
    (REP / "report.html").write_text(doc, encoding="utf-8")
    print("report.html written")
    if args.pdf:
        out = REP / PDF_NAME
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", (REP / "report.html").as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pdf:", out)


if __name__ == "__main__":
    main()
