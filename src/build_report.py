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
REPO_URL = "https://github.com/KonstantinPatrushev/recsys_midterm"
FINAL = "det_ens_AD"

# English names of the shows quoted in Sec. 4.5 (the schedule titles are in Russian)
EN_TITLES = {
    "Губка Боб": "SpongeBob SquarePants", "Волшебные покровители": "The Fairly OddParents",
    "Элвиннн": "ALVINNN!!! and the Chipmunks", "Кунг-фу панда": "Kung Fu Panda: Legends of Awesomeness",
    "Пингвины из": "The Penguins of Madagascar", "Аватар": "Avatar: The Last Airbender",
    "Ниндзяго": "Ninjago: Masters of Spinjitzu", "Бен 10": "Ben 10", "Удивительный мир Гамбола": "The Amazing World of Gumball",
    "Октонавты": "Octonauts", "Свинка Пеппа": "Peppa Pig", "ТСН": "TSN (news)", "Факты": "Fakty (news)",
    "Чистоnews": "Chysto News", "Маша и Медведь": "Masha and the Bear", "Санджей и Крейг": "Sanjay and Craig",
    "Аладдин": "Aladdin", "Викна-Новости": "Vikna-Novyny (news)", "Сегодня": "Segodnya (news)",
}
CAT_EN = {"Детям": "Kids", "Сериалы": "Series", "Фильмы": "Film", "Развлечения": "Entert.", "Инфо": "Info",
          "Познавательное": "Docu.", "Другое": "Other", "Спорт": "Sport"}


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


def en_title(t):
    t = (t or "").replace("М/с ", "").replace("Т/с ", "").replace("'", "").strip(" .")
    for ru, en in EN_TITLES.items():
        if t.startswith(ru):
            return en
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    prep, eda, tz = j("prep.json"), j("eda.json"), j("tz_alignment.json")
    rs = j("report_stats.json")
    als = j("als_sweep_F1.json")
    meth = j("methods_F1.json")["scores"]
    hyb = j("hybrid_ablation_F1.json")
    ex1, ex2, ex3 = j("experiments_F1.json"), j("experiments2_F1.json"), j("experiments3_F1.json")
    aug = j("experiments4_augment_F1.json")
    rob = j("robustness.json")
    diag = j("diagnostics.json")
    biz = j("business_two_users.json")
    cand = j("candidates_final.json")
    rep1, repF, reb2 = j("reproduce_sub01.json"), j("reproduce_final.json"), j("rebuild_sub02_after_determinism_fix.json")
    tm, ti = j("test_metrics.json"), j("test_intervals.json")
    info2 = j("sub02_hybrid_lr31_r250_s5_info.json")
    cv1, cv2 = j("cv_sub01_ranker_v1.json"), j("cv_sub02_hybrid_lr31_r250_s5.json")
    fs, hp, act = j("final_selection.json"), j("history_proxy.json"), j("weekly_activity.json")
    hb, ndet, dcheck = j("history_blocks.json"), j("nondeterminism_check.json"), j("determinism_check.json")
    run_secs = [j(f"det_{v}_{h}_info.json")["seconds"] for v in ("sum", "max") for h in ("clean8", "recent8")]
    subs = {r["fileName"].replace(".csv", ""): r for r in csv.DictReader(open(EV / "kaggle_submissions.csv", encoding="utf-8"))}
    lb = list(csv.DictReader(open(EV / "kaggle_meta" / "public_leaderboard_2020_final.csv", encoding="utf-8-sig")))
    winner_pub = {r["TeamName"]: float(r["Score"]) for r in lb}["Magic City"]
    winner_priv = 0.42912  # private leaderboard, first place (Kaggle API leaderboard view, evidence/kaggle_meta/topics_leaderboard.txt)

    mt = rs["matching"]
    cp = rs["common_pool"]
    cost = rs["cost_seconds_F1"]
    fin = rs["final_config_F1"]
    F1p, F0p, F2p = cp["F1"], cp["F0"], cp["F2"]
    la = fs["label_agreement"]
    agree_lo = min(v["identical_top5_share"] for v in la.values())
    agree_hi = max(v["identical_top5_share"] for v in la.values())
    tbh = rob["F1_tiebreak_sensitivity_hybrid"]
    rnd_all = [tbh[k]["map5_all"] for k in tbh if k.startswith("random")]
    gap_weeks = [str(w) for w in range(20, 28)]
    gap_min_sub = min(act["share_active_submission_users"][w] for w in gap_weeks)
    gap_min_oth = min(act["share_active_other_users"][w] for w in gap_weeks)
    no_days = act["days_without_sessions"]
    ties = eda["val19_30_full"]["share_users_with_tie_at_5th"]
    by_nov = {d["novel"]: d for d in diag["error_by_novel_truth_items_(3=3+)"]}
    n_eval, n_sub = rob["F1"]["n_users"], rob["F1"]["n_sub"]
    best_als = als[max(als, key=lambda k: als[k][0])]
    als_default = meth["ALS_f64_r0.05_a10.0"]
    hab, exp_ = biz["habitual"], biz["explorer"]
    nov = biz["novelty_by_method_sub_users_F1"]
    sel = biz["selection"]

    def sc(name):
        r = subs[name]
        return float(r["publicScore"]), float(r["privateScore"])

    pub = {k: sc(k)[0] for k in subs}
    prv = {k: sc(k)[1] for k in subs}
    best_priv_name = max(prv, key=prv.get)
    assert best_priv_name == FINAL, (best_priv_name, FINAL)

    # ---------------- Table 1: every method on one candidate pool
    rows = [("Popularity (reference only)", "Popularity (reference)", cost.get("Popularity (reference)")),
            ("Repeat own history (reference only)", "Repeat own history (reference)", cost.get("Repeat own history (reference)")),
            ("CBF", "CBF", cost["CBF"]),
            ("Association rules (support &ge; 20, max confidence)", "Association rules", cost["Association rules"]),
            ("MF: implicit ALS (64 factors, &alpha;=1, &lambda;=10)", "MF (implicit ALS)", cost["MF (implicit ALS)"]),
            ("Personalized PageRank (damping 0.7)", "PPR", cost["PPR"])]
    t1 = []
    for label, key, c in rows:
        t1.append(f"<tr><td>{label}</td><td class=n>{f3(F1p[key]['map5_all'])}</td><td class=n>{f3(F1p[key]['map5_sub'])}</td>"
                  f"<td class=n>{f3(F0p[key]['map5_all'])}</td><td class=n>{f3(F2p[key]['map5_all'])}</td><td class=n>{c:.1f}</td></tr>"
                  if c >= 0.1 else
                  f"<tr><td>{label}</td><td class=n>{f3(F1p[key]['map5_all'])}</td><td class=n>{f3(F1p[key]['map5_sub'])}</td>"
                  f"<td class=n>{f3(F0p[key]['map5_all'])}</td><td class=n>{f3(F2p[key]['map5_all'])}</td><td class=n>&lt;0.1</td></tr>")
    hk = "Hybrid LightGBM ranker (sum label)"
    t_tab = cost["Hybrid: candidate/feature table incl. the four methods"]
    t_lgb = cost["Hybrid: LightGBM 5 seeds x 5 CV folds (train + predict)"]
    t1.append(f"<tr><td><b>Hybrid LightGBM ranker</b> (sum label, 5 seeds)</td><td class=n>{f3(F1p[hk]['map5_all'])}</td>"
              f"<td class=n>{f3(F1p[hk]['map5_sub'])}</td><td class=n>{f3(F0p[hk]['map5_all'])}</td><td class=n>{f3(F2p[hk]['map5_all'])}</td>"
              f"<td class=n>{t_tab:.0f} + {t_lgb:.0f}</td></tr>")
    t1.append(f"<tr><td><b>Final configuration</b> (sum + max labels)</td><td class=n><b>{f3(fin['truth_sum_all'])}</b></td>"
              f"<td class=n><b>{f3(fin['truth_sum_sub'])}</b></td><td class=n>&ndash;</td><td class=n>&ndash;</td><td class=n>2&times;</td></tr>")
    table1 = "\n".join(t1)

    # ---------------- Table 2: submissions
    sub_rows = [
        ("sub01_ranker_v1", "1", "Ranker v1: habit + schedule features, 63 leaves", f3(cv1["map5_sub"])),
        ("sub02_hybrid_lr31_r250_s5", "2", "Hybrid: + course-method features, 31 leaves, sum label, history A", f3(cv2["map5_sub"])),
        ("det_ens_D", "3", "Hybrid, sum + max labels, history D (weeks 20&ndash;24 + 28&ndash;30)", f3(fin["truth_sum_sub"]) + "&dagger;"),
        ("det_ens_AD", "4", "<b>Hybrid, sum + max labels, histories A and D &mdash; designated final</b>", f3(fin["truth_sum_sub"]) + "&dagger;"),
        ("det_ens_A", "5", "Hybrid, sum + max labels, history A (weeks 15&ndash;19 + 28&ndash;30)", f3(fin["truth_sum_sub"]) + "&dagger;"),
    ]
    t2 = []
    for name, num, desc, cvs in sub_rows:
        p, q = pub[name], prv[name]
        bp = "<b>" if name == max(pub, key=pub.get) else ""
        bq = "<b>" if name == best_priv_name else ""
        t2.append(f"<tr><td>{num}</td><td>{desc}</td><td class=n>{cvs}</td><td class=n>{bp}{f5(p)}{'</b>' if bp else ''}</td>"
                  f"<td class=n>{bq}{f5(q)}{'</b>' if bq else ''}</td></tr>")
    t2.append(f"<tr><td></td><td>Original winner, 2020 [1]</td><td class=n>&ndash;</td><td class=n>{f5(winner_pub)}</td><td class=n>{f5(winner_priv)}</td></tr>")
    table2 = "\n".join(t2)

    # ---------------- business tables: each method's top-5 with id, title, category
    def biz_table(rec):
        out = []
        names = [("POP", "Popularity (ref.)"), ("CBF", "CBF"), ("AR", "Assoc. rules"), ("MF (ALS)", "MF (ALS)"),
                 ("PPR", "PPR"), ("Hybrid (LGBM)", "Hybrid (final)")]
        for key, label in names:
            m = rec["methods"][key]
            cells = []
            for x in m["top5"]:
                mark = " &#10003;" if x["hit"] else ""
                cells.append(f"<td>{x['id']}{mark}<br>{esc(en_title(x['title']))} &middot; {esc(CAT_EN.get(x['category'], x['category']))}</td>")
            cells += ["<td></td>"] * (5 - len(cells))
            out.append(f"<tr><td>{label}<br>AP@5 {f3(m['ap5'])}</td>{''.join(cells)}</tr>")
        truth = "".join(f"<td>{x['id']}<br>{esc(en_title(x['title']))} &middot; {esc(CAT_EN.get(x['category'], x['category']))}"
                        f" ({x['full_views']}{'' if x['seen_in_history'] else ', new'})</td>" for x in rec["truth_19_30"])
        out.append(f"<tr><td><i>Truth, weeks 19&ndash;30</i></td>{truth}</tr>")
        return "\n".join(out)

    hab_tab, exp_tab = biz_table(hab), biz_table(exp_)
    hab_hist = ", ".join(f"{esc(en_title(x['title']))} ({x['full_views']})" for x in hab["top_history_shows"][:3])
    exp_hist = ", ".join(f"{esc(en_title(x['title']))} ({x['full_views']})" for x in exp_["top_history_shows"][:4])
    hab_ch = ", ".join(f"{esc(c['channel'])} {pct(c['share'], 0)}" for c in hab["top_channels_by_time"][:4])
    exp_ch = ", ".join(f"{esc(c['channel'])} {pct(c['share'], 0)}" for c in exp_["top_channels_by_time"][:4])
    hab_test = ", ".join(esc(en_title(x["title"])) for x in hab.get("test_31_42_final_top5", []))
    exp_test = ", ".join(esc(en_title(x["title"])) for x in exp_.get("test_31_42_final_top5", []))
    n_cn = sum(1 for x in exp_["truth_19_30"] if not x["channel_watched_in_history"])
    hab_hits_h = sum(1 for x in hab["methods"]["Hybrid (LGBM)"]["top5"] if x["hit"])
    hit_ranks = [i + 1 for i, x in enumerate(hab["methods"]["Hybrid (LGBM)"]["top5"]) if x["hit"]]
    ap_terms = " + ".join(f"{k + 1}/{r}" for k, r in enumerate(hit_ranks))
    hyb_ids = {x["id"] for x in hab["methods"]["Hybrid (LGBM)"]["top5"]}
    hab_missed = ", ".join(esc(en_title(x["title"])) for x in hab["truth_19_30"] if x["id"] not in hyb_ids)
    hab_extra = ", ".join(esc(en_title(x["title"])) for x in hab["methods"]["Hybrid (LGBM)"]["top5"] if not x["hit"])
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    shot = REP / "fig" / "leaderboard.png"
    shot_html = (f'<img src="fig/leaderboard.png" style="width:100%">' if shot.exists() else
                 '<div style="border:1px dashed #000;padding:14pt;text-align:center">Screenshot of the Kaggle Submissions page '
                 '(returned private score of the designated final submission) &mdash; to be inserted</div>')

    hp_close = (hp["mixed_12_14_20_24"]["map5_evalB"] - hp["recent_skip_worst_17_24"]["map5_evalB"],
                hp["mixed_12_14_20_24"]["map5_evalB_sub"] - hp["recent_skip_worst_17_24"]["map5_evalB_sub"])

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Midterm Report &mdash; SWEET.TV TV-show recommender</title>
<style>
  a {{ color: #000; text-decoration: none; }}
  @page {{ size: A4; margin: 12mm 13mm 12mm 13mm; }}
  body {{ font-family: "Times New Roman", "Liberation Serif", serif; font-size: 9.7pt; line-height: 1.22; color: #000; margin: 0; }}
  h1 {{ font-size: 14pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }}
  .meta {{ text-align: center; font-size: 9.4pt; margin-bottom: 5pt; }}
  h2 {{ font-size: 10.4pt; margin: 5pt 0 2pt; text-transform: uppercase; letter-spacing: .3pt; break-after: avoid; }}
  h3 {{ font-size: 9.9pt; margin: 3.5pt 0 1.5pt; font-style: italic; font-weight: normal; break-after: avoid; }}
  p {{ margin: 0 0 2.8pt; text-align: justify; }}
  ul {{ margin: 0 0 2.8pt; padding-left: 14pt; }}
  li {{ margin-bottom: 0.8pt; text-align: justify; }}
  .abstract {{ font-size: 9.3pt; margin: 0 5mm 3pt; }}
  code {{ font-family: "DejaVu Sans Mono", "Menlo", monospace; font-size: 7.7pt; }}
  table {{ border-collapse: collapse; margin: 2pt auto 1.5pt; font-size: 8.2pt; }}
  th, td {{ border-top: 1px solid #000; border-bottom: 1px solid #000; padding: 1pt 3.5pt; text-align: left; vertical-align: top; }}
  th {{ font-weight: bold; }}
  td.n, th.n {{ text-align: right; white-space: nowrap; }}
  table.top5 {{ font-size: 6.9pt; width: 100%; }}
  table.top5 td {{ padding: 0.6pt 2.5pt; line-height: 1.12; }}
  table.top5 td:first-child {{ white-space: nowrap; }}
  .cap {{ font-size: 8.2pt; text-align: center; margin: 1pt 0 4pt; }}
  .refs p {{ text-align: left; padding-left: 18pt; text-indent: -18pt; font-size: 8.5pt; margin-bottom: 1.6pt; }}
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
  Student: Konstantin Patrushev &nbsp;&nbsp; Team: Individual &nbsp;&nbsp; Kaggle: konstantinpatrushev<br>
  Email: kapatrushev@edu.hse.ru &nbsp;&nbsp; Date: 2026-10-21 &nbsp;&nbsp; Code and final submission file: <a href="{REPO_URL}">{REPO_URL}</a><br>
  Assignment: Midterm Quiz &mdash; TV Program Recommendation (Kaggle <code>sweettv-tv-program-recommender</code>)
</div>

<h2>Abstract</h2>
<p class="abstract">SWEET.TV asks for the five TV shows each of 1,260 users will most often watch to at least 80% of an airing during
twelve future weeks, scored by MAP@5 [1]. I matched {n0(prep['log_rows'])} viewing sessions to the programme schedule, rebuilt the
organisers' target exactly and validated on a time split with the same 12-week horizon. Four course methods (content-based
filtering, association rules, ALS matrix factorization, personalized PageRank) were evaluated individually on one candidate pool
and then combined in a LightGBM LambdaRank ranker with schedule-aware habit features: validation MAP@5 = {f3(fin['truth_sum_all'])}
against {f3(F1p['PPR']['map5_all'])} for the best single method, PPR. The designated final submission returned a private score
of {f5(prv[FINAL])} ({f5(pub[FINAL])} public), above the original winner's {f5(winner_priv)}. For linear TV, when a show will air,
crossed with when the user watches, carries more signal than collaborative similarity.</p>
<p class="abstract"><i>Index Terms</i> &mdash; recommender systems, linear television, learning to rank, personalized PageRank, implicit feedback</p>

<h2>1. Introduction</h2>
<p><b>Problem.</b> The competition [1] gives channel-viewing sessions of {n0(eda['n_users_logs'])} users for weeks 11&ndash;30 of 2020
and programme schedules for weeks 11&ndash;42. For every airing, screen time = seconds watched / airing duration; airings with
screen time &ge; 0.8 count, and a user's truth is the five shows with the most such airings in weeks 31&ndash;42 (exactly five per
scored user; public and private split the users 50/50) [1]. The exam requires at least two of CBF, association rules, MF and PPR,
compared on the same temporal split, candidate pool and protocol, with popularity as a reference only, and an analysis of two users.</p>
<p><b>Motivation.</b> For an operator the five shows are the &ldquo;your shows&rdquo; row and the reminder list. Unlike an on-demand
catalogue, a linear show can only be watched while it is on air, so the schedule of weeks 31&ndash;42 is part of the answer.</p>
<p><b>Concrete example.</b> Household u{hab['u']} fully watched <i>SpongeBob SquarePants</i> {hab['top_history_shows'][0]['full_views']} times
in weeks 11&ndash;18, mostly on {esc(hab['top_history_shows'][0]['channel'])}; in weeks 19&ndash;30 it was the household's
#1 show with {hab['truth_19_30'][0]['full_views']} full views, and the final model ranks it first (Table 3).</p>
<p><b>Contributions.</b></p>
<ul>
  <li>An exact rebuild of the organisers' target from raw sessions and a validation split whose estimate for the submission users
  ({f3(cv1['map5_sub'])}) matched the first public score ({f3(pub['sub01_ranker_v1'])}).</li>
  <li>The four course methods evaluated individually on one pool, three folds and with their cost, a hybrid ranker that combines
  them, five Kaggle submissions, and a two-user business analysis with failure cases.</li>
</ul>

<h2>2. Related Work</h2>
<p>Implicit-feedback matrix factorization with confidence weights was introduced on TV-viewing data [2]. Personalized PageRank
comes from topic-sensitive PageRank [3]; association rules from Agrawal et al. [4]; content-based recommendation is surveyed by
Pazzani and Billsus [5]. For linear TV, Turrin et al. [6] predict what a user watches from time of day and channel, and
Watch-It-Next [7] conditions on the programme being watched. In this competition the 2nd-place team generated candidates with
LightFM and re-ranked them with LightGBM [12].</p>
<p><b>Alternatives considered.</b> (i) A per-airing classifier P(full view) summed over future airings: closest to the target but
about ten times more rows; slot-habit features approximate it. (ii) Sequence models: {n0(eda['n_users_logs'])} users whose
behaviour follows the schedule rather than item order. (iii) Rank fusion of the four methods without learning: on the
all-airing-shows pool it reached {f3(rob['F1']['scores']['RRF_ppr+ar+als+cbf+repeat']['map5_all'])} on F1, far below the ranker.</p>

<h2>3. Method</h2>
<p><b>Matching rule.</b> A session (user, channel, start, stop) is matched to every airing on the same channel whose interval
[start, start + duration) overlaps it; the overlap in seconds is time watched (an interval join on sorted arrays, equal to brute
force on {ti['pairs_checked']:,} random pairs). Seconds are summed over a user's sessions on the same airing, and screen time &ge; 0.8
marks a <i>full view</i>: {n0(prep['full_views_sum_ge_0.8'])} of {n0(prep['user_airing_views'])} user&ndash;airing views. The rule's
wording does not say whether one uninterrupted session is required; the longest single session gives
{n0(prep['full_views_max_ge_0.8'])} full views, and the two readings give the same true top-5 for {pct(agree_lo, 0)}&ndash;{pct(agree_hi, 0)}
of users across folds, so the final model averages rankers trained on both readings.</p>
<p><b>Missing programme IDs.</b> <code>tv_show_id = 0</code> marks {pct(mt['share_schedule_rows_id0'])} of schedule rows
({n0(mt['schedule_rows_id0'])} of {n0(mt['schedule_rows_all'])}: films, series blocks, unidentified programmes). The organisers drop these
rows before building the truth, so I drop them before matching. Of all viewing time, {pct(mt['share_session_time_on_nonzero_airings'])}
falls on airings with an id, {pct(mt['share_session_time_on_id0_airings'])} on id-0 airings and {pct(mt['share_session_time_without_schedule'])}
on no schedule entry; {pct(mt['sessions_on_channels_without_any_nonzero_airing'] / mt['sessions_total'])} of sessions are on channels with no
identified programme at all. That time creates no interactions and is not imputed: a 0 can never be a target.</p>
<p><b>Data checks.</b> (a) <code>user_id</code> exceeds 2<sup>63</sup> and is read as unsigned 64-bit. (b) Logs and schedule share
a clock: {pct(tz['0'][1], 2)} of session ends fall within 60 s after an airing end at zero shift, at most
{pct(max(v[1] for k, v in tz.items() if k != '0'), 2)} at &plusmn;1&ndash;3 h. (c) Weeks 20&ndash;27 lost much of the logging (Fig. 1):
the weekly share of active users falls to {gap_min_sub:.2f} (submission users) and {gap_min_oth:.2f} (others), and {', '.join(no_days)}
has no sessions. (d) {pct(ties, 0)} of users tie between their 5th and 6th show in weeks 19&ndash;30.</p>
<div class="two avoid">
<div><img class="fig" src="fig/fig_activity.svg"></div>
<div><p class="cap" style="text-align:left;margin-top:8pt">Fig. 1. Share of users with any session per week, and the history and
target windows of the validation fold F1 and of the test. The test uses the eight most recent clean weeks (15&ndash;19 and
28&ndash;30, history A) or weeks 20&ndash;24 and 28&ndash;30 (history D).</p>
<p><b>Same protocol for every method.</b> Fold F1 takes features from weeks 11&ndash;18 and truth from weeks 19&ndash;30, the test's
12-week horizon; folds F0 (11&ndash;15 &rarr; 16&ndash;19) and F2 (11&ndash;27 &rarr; 28&ndash;30) repeat everything in other periods. Each
user's candidate pool is the union of the shows he watched to &ge; 20% that air in the target window, the 30 most-watched airing
shows, the 30 best shows by slot habit and the top 20 of each course method (median {cand['F1_candidates_per_user_median']:.0f} per user,
{pct(cand['F1_recall_all'])} of true shows covered). Every method, popularity included, ranks exactly this pool and is scored by the
same MAP@5 code on the same {n0(n_eval)} users ({n0(n_sub)} of them submission users).</p></div>
</div>
<p><b>Course methods.</b> <i>CBF</i>: TF-IDF show vectors (category, genres, shares of airings by channel, hour and weekend, duration,
decade); profile = count-weighted sum; cosine. <i>Association rules</i>: pairwise rules A&rarr;B over users' sets of fully watched
shows, support &ge; 20 users; score = max confidence. <i>MF</i>: implicit ALS [2] on 1 + &alpha;&middot;log(1 + count), chosen from
{len(als)} settings. <i>PPR</i>: random walk with restart on the user&ndash;show graph, edge weight log(1 + count), damping chosen from
{{0.3, 0.5, 0.7, 0.85}}. All four see only the history window.</p>
<p><b>Hybrid ranker.</b> LightGBM LambdaRank [8], [9] over the pool with {len(info2['features'])} features: full views per show over the
last 1/2/4/8 weeks and per week, recency, partial views; schedule-aware exposure &mdash; airings in the target window on the
channels where the user watched the show (the top feature by gain) and &Sigma;<sub>future airings</sub> P(user fully watches what starts
on that channel, weekday and hour); show popularity; user activity; category and genre; each course method's score and rank. Label
= &ldquo;in the true top 5&rdquo;; 31 leaves, learning rate 0.05, 250 rounds, 5 seeds averaged by within-user rank. <b>Final
submission</b>: the within-user rank average of four such rankers, {{sum, max}} label &times; {{history A, history D}}; a candidate missing
from one table counts as rank 0, which tilts the average toward the max-label ranker (fewer candidates).</p>
<p><b>Tools and practices.</b> Python 3.12, polars 2.0.0, NumPy 2.5.4, SciPy 1.18.1, LightGBM 4.7.0, implicit 0.7.3, Kaggle CLI 2.2.4;
Apple M5, 24 GB; deterministic LightGBM. Kaggle practices [10], [11]: read the data page through the API (it holds the exact target);
a fold that mirrors the test horizon; an early calibration submission; candidate recall measured apart from ranking; paired
comparisons with seed noise; seed averaging; a pre-planned budget of five submissions.</p>

<h2>4. Experiments</h2>
<h3>4.1 Each method individually, on one pool</h3>
<div class="avoid">
<table>
<tr><th>Method</th><th class=n>F1 all</th><th class=n>F1 sub.</th><th class=n>F0 all</th><th class=n>F2 all</th><th class=n>Cost, s</th></tr>
{table1}
</table>
<p class="cap">Table 1. MAP@5 on the common candidate pool. F1 {n0(n_eval)} users (&ldquo;sub.&rdquo; = {n0(n_sub)} submission users), F0
{n0(rob['F0']['n_users'])}, F2 {n0(rob['F2']['n_users'])}. Cost = wall-clock seconds to fit and score all users on F1 (Apple M5); for the
hybrid, the table with all four methods plus 25 LightGBM fits of cross-validation; the final configuration trains two such
rankers. <code>report_stats.json</code>.</p>
</div>
<p>The order is the same on all three folds: hybrid &gt; PPR &ge; repeat &gt; ALS &gt; association rules &gt; popularity &gt; CBF. PPR is
the best course method: the walk keeps most mass on the user's own shows and adds shows of similar users. ALS with common default
settings (64 factors, &alpha;=10, &lambda;=0.05) scored {f3(als_default['map5_all'])} on all airing shows, below popularity
({f3(meth['POP']['map5_all'])}); strong regularization raised it to {f3(best_als[0])}. CBF is below popularity on every fold: it finds
similar programmes rather than the exact show. In the ranker, the course methods add little on top of habit and schedule features ({f5(hyb['base']['map5_all'])} &rarr;
{f5(hyb['base+method_feats']['map5_all'])} on F1); 31 leaves beat 63 and 127, and 150&ndash;300 rounds are a plateau. Error analysis:
when all five true shows are in the pool ({pct(diag['share_users_all_5_in_candidates'], 0)} of users) MAP@5 is
{f3(diag['map5_if_all_5_in_candidates'])}; users whose truth contains three or more shows new to them ({n0(by_nov[3]['users'])}) reach
only {f3(by_nov[3]['map5'])}, against {f3(by_nov[0]['map5'])} when nothing is new. The bottleneck is novelty.</p>

<h3>4.2 Kaggle submissions</h3>
<div class="avoid">
<table>
<tr><th>#</th><th>Submission</th><th class=n>CV F1 sub.</th><th class=n>Public</th><th class=n>Private</th></tr>
{table2}
</table>
<p class="cap">Table 2. All five submissions (<code>kaggle_submissions.csv</code>). &dagger; F1 has one history window, so 3&ndash;5 share the
CV of the label ensemble. Final file: <code>submissions/{FINAL}.csv</code>, SHA-256 {repF['sha256_submitted'][:16]}&hellip;</p>
</div>
<div class="avoid">{shot_html}<p class="cap">Fig. 2. Kaggle Submissions page with the returned scores; the designated final submission is #4.</p></div>
<p>Submission 1, sent early as a calibration point, scored {f5(pub['sub01_ranker_v1'])} public against a CV of {f5(cv1['map5_sub'])}.
Later CV gains shrank on the test: submission 2 gained {f4(cv2['map5_sub'] - cv1['map5_sub'])} on CV but
{f4((pub['sub02_hybrid_lr31_r250_s5'] + prv['sub02_hybrid_lr31_r250_s5'] - pub['sub01_ranker_v1'] - prv['sub01_ranker_v1']) / 2)} on the whole test,
and each 630-user half moves a file by several thousandths. History D lost on both halves (#3 vs #5:
{f4(pub['det_ens_D'] - pub['det_ens_A'])} public, {f4(prv['det_ens_D'] - prv['det_ens_A'])} private): a proxy on target weeks 28&ndash;30 had
favoured recent weeks, but its closest analogue of A vs D was a tie ({hp_close[0]:+.4f} all, {hp_close[1]:+.4f} submission users),
and because of the logging gap {hb['weeks_20_24']['submission_users_without_full_view']} submission users have no full view in weeks 20&ndash;24
(against {hb['weeks_15_19']['submission_users_without_full_view']} in weeks 15&ndash;19). Averaging both histories (#4) gave the
best private score. The truth's tie-break is undocumented and matters: the same out-of-fold predictions score
{f3(tbh['id_asc']['map5_all'])} with ties broken by ascending id, {f3(min(rnd_all))}&ndash;{f3(max(rnd_all))} with random ties and
{f3(tbh['id_desc']['map5_all'])} with descending id; I train with ascending id.</p>

<h3>4.3 Verification</h3>
<ul>
  <li>MAP@5 equals {len(tm['cases'])} hand-computed cases and an independent re-implementation on {n0(tm['random_pairs'])} random pairs; the
  interval join equals brute force on {ti['trials']} random schedules.</li>
  <li><code>tests/reproduce_final.sh</code> rebuilds the final file from scratch: identical SHA-256; submission 1 also rebuilds byte-for-byte.</li>
  <li>By hand, u{hab['u']}: the final model's list hits at ranks {', '.join(map(str, hit_ranks))}, so AP@5 = ({ap_terms})/5 =
  {f3(hab['methods']['Hybrid (LGBM)']['ap5'])}, as the code reports.</li>
  <li>Every file has 1,260 rows in the sample's order and five distinct ids airing in weeks 31&ndash;42; no viewing after week 30 is read.</li>
</ul>

<h3>4.4 Business and algorithmic analysis</h3>
<p><b>Recommended strategy for SWEET.TV: the hybrid ranker.</b> It is the most accurate on every fold (Table 1:
+{f3(F1p[hk]['map5_all'] - F1p['PPR']['map5_all'])} MAP@5 over PPR on F1), and it is cheap enough for a nightly batch: each of the four
rankers of the final file took {min(run_secs):.0f}&ndash;{max(run_secs):.0f} s end to end on a laptop (tables with all four methods for
{n0(eda['n_users_logs'])} users, five LightGBM fits, prediction), {sum(run_secs):.0f} s in total, while PPR alone takes {cost['PPR']:.0f} s. It is the only method that
knows when a show will be on air. Its new-show suggestions are right {pct(nov['Hybrid (LGBM)']['precision_of_new_items'])} of the time,
against {pct(nov['CBF']['precision_of_new_items'])} for CBF and {pct(nov['AR']['precision_of_new_items'])} for association rules (submission users, F1).</p>
<p><b>Two users, same for every method.</b> Not hand-picked: among the {sel['n_typical_sub_users']} submission users with typical activity
({sel['typical_activity_range_full_views_11_18'][0]:.0f}&ndash;{sel['typical_activity_range_full_views_11_18'][1]:.0f} full views in weeks 11&ndash;18,
the interquartile range), one was drawn at random (seed {sel['seed']}) from the <i>habitual</i> segment (all five true shows already
watched; {sel['n_habitual']} users) and one from the <i>explorer</i> segment (&ge; 2 true shows never watched; {sel['n_explorer']} users).
Both draws are households with children's channels.</p>
<table class="top5 avoid">
<tr><th>u{hab['u']}, habitual</th><th>#1</th><th>#2</th><th>#3</th><th>#4</th><th>#5</th></tr>
{hab_tab}
</table>
<table class="top5 avoid">
<tr><th>u{exp_['u']}, explorer</th><th>#1</th><th>#2</th><th>#3</th><th>#4</th><th>#5</th></tr>
{exp_tab}
</table>
<p class="cap">Table 3. Each method's top-5 on F1 (programme id, title, category; &#10003; = in the truth). Truth: full views in weeks
19&ndash;30, &ldquo;new&rdquo; = never fully watched before. <code>business_two_users.json</code>.</p>
<p><b>u{hab['u']} (habitual).</b> History: {hab_hist}; channels {hab_ch}. <i>Familiar preferences</i> dominate: every personalised
method except association rules puts SpongeBob first, and the final model hits {words[hab_hits_h]} of five (AP@5
{f3(hab['methods']['Hybrid (LGBM)']['ap5'])}). It misses {hab_missed} and spends a slot on {hab_extra}, news that an adult in the same
household watches &mdash; one account mixes several viewers. Popularity, identical for everybody, recommends four news programmes and Chysto News. Test list
(31&ndash;42): {hab_test}.</p>
<p><b>u{exp_['u']} (explorer).</b> History: {exp_hist}; channels {exp_ch}. {words[n_cn].capitalize()} of the four <i>new discoveries</i> air on Cartoon Network,
a channel the household never watched in weeks 11&ndash;18 (first full view in week 20). None of the methods recommends them: CBF offers
other cartoons, and the collaborative methods stay with the household's usual shows and news. Once the shows enter the history (weeks 28&ndash;30), the final model ranks them:
{exp_test}.</p>
<p><b>How the methods differ and what it means for users.</b> PPR and ALS return almost only the user's own shows
({pct(nov['PPR']['share_new_in_top5'])} and {pct(nov['MF (ALS)']['share_new_in_top5'])} new items): reliable, but a &ldquo;continue
watching&rdquo; row with no discovery. Association rules push the staples co-watched by many households (news), which feels generic.
CBF fills {pct(nov['CBF']['share_new_in_top5'])} of slots with new but similar programmes that are rarely watched, which costs trust.
The hybrid keeps familiar shows first and adds a new show only when the schedule and the user's habits make it likely
({pct(nov['Hybrid (LGBM)']['share_new_in_top5'])} of slots). For the product: a &ldquo;your shows this week&rdquo; row with airtime reminders,
one discovery slot only when a new show scores high, and children's profiles to separate household viewers.</p>
<p><b>Limitations and failure cases.</b> Users with three or more new shows score {f3(by_nov[3]['map5'])}; channel-level cold start (u{exp_['u']});
household accounts (u{hab['u']}); the season change in weeks 31&ndash;42; the history-D loss and the undocumented tie-break (Sec. 4.2).
No online test was run.</p>

<h2>5. Discussion</h2>
<p><b>Failure case.</b> Two identical runs of candidate generation printed <code>features: 392690 rows</code> and
<code>features: 392686 rows</code> (session log); later, a repeated cross-validation gave {f5(ndet['cv_sub02_map5_all_recorded_at_47bffcd'])}
and then {f5(ndet['cv_sub02_map5_all_after_fix'])}. <i>Root cause:</i> ties without a tie-breaker (sorting, <code>mode().first()</code>,
ordinal ranks), and a float sum over rows whose order came from a hash join: re-running the old code twice changes
{ndet['exp_show_slot_values_differing_between_two_runs']} values of one feature by up to {ndet['max_abs_difference']:.1e} and
{ndet['exp_show_slot_rank_differing']} of its ranks, enough to change the trees. <i>Fix:</i> total orders, sorting before the sum,
deterministic LightGBM. <i>Verification:</i> the feature tables are identical across three processes
(<code>determinism_check.json</code>) and the final file rebuilds byte-for-byte; submission 2 was uploaded before the fix, and its
rebuild differs in {reb2['users_with_different_set']} of 1,260 top-5 sets. A second failure was methodological: the recent-weeks history looked better
on a proxy and lost on both test halves (Sec. 4.2).</p>
<p><b>What worked.</b> Reading the data page via the API (the exact target); a fold with the test's horizon; the schedule as features;
averaging rankers over the two label readings and two histories.</p>
<p><b>What surprised me or did not work.</b> ALS with default settings lost to popularity; the course methods add little once habit
and schedule features exist; a tie-break rule moves MAP@5 by {f3(tbh['id_asc']['map5_all'] - tbh['id_desc']['map5_all'])}; CV gains shrank on the test.</p>
<p><b>Next improvement.</b> Model each future airing (P(full view) from slot, channel and show history) and sum per show, and add a
&ldquo;recently adopted channel&rdquo; signal for explorers like u{exp_['u']}.</p>

<h2>6. AI Usage Disclosure</h2>
<p><b>AI tools used.</b> Claude Code (VS Code extension) with Claude Opus 5.5, one session, including sub-agents for the literature
and Kaggle-practice search and for an independent review of the submission plan.</p>
<p><b>How AI was used.</b> Code generation (matching, methods, ranker, diagnostics, report builder); data analysis and debugging;
search and verification of references through Crossref, OpenAlex and the publishers' pages; Kaggle submissions through the API;
draft of this report, rendered from <code>evidence/</code> by <code>src/build_report.py</code>.</p>
<p><b>What I personally verified.</b> <span class="todo">[to be completed by the student &mdash; keep only checks actually done, e.g. the
hand computation in Sec. 4.3, Tables 1&ndash;2 against <code>evidence/</code>, the screenshot.]</span></p>
<p><b>What I trusted without verification.</b> <span class="todo">[to be completed by the student.]</span></p>
<p><b>Session log.</b> <code>patrushev_konstantin_midterm_session.json</code> is the unmodified Claude Code session export.</p>

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
