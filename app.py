"""Minimal read-only viewer over the pipeline's saved outputs (run `python run_pipeline.py ...` first).

    pip install streamlit pandas && streamlit run app.py

Deliberately thin: all logic lives in src/ and run_pipeline.py; this file only displays precomputed JSON.
NOTE: Streamlit was not installable in the build environment, so this file was only smoke-tested against a stub (tests/test_app_smoke.py).
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(os.environ.get("PROMOAI_OUT", "outputs"))
st.set_page_config(page_title="Promo finder", layout="wide")


@st.cache_data
def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


ranked, top5, eps = load("ranked_promos.json"), load("top5_promos.json"), load("episode_summary.json")
stats, valid, sens, parse = load("run_stats.json"), load("validation_report.json"), load("evaluation/weight_sensitivity.json"), load("parse_summary.json")
st.title("Find the Promos")
if not ranked:
    st.warning(f"No outputs found in '{OUT}'. Run: python run_pipeline.py --input <zip|folder|file> --out {OUT}")
    st.stop()
st.caption("Rule-based engine: scores are heuristic proxies for hook / cliffhanger / emotion, not predictions of audience engagement.")

t_dash, t_rec, t_det, t_ep, t_q, t_exp = st.tabs(["Dashboard", "Recommendations", "Details", "Episode Explorer", "Quality Evaluation", "Export"])

with t_dash:
    c = st.columns(5)
    c[0].metric("Episodes", stats["episodes_processed"]); c[1].metric("Windows scored", stats["windows_scored"])
    c[2].metric("Promos recommended", stats["promos_recommended"]); c[3].metric("Episodes w/o valid promo", len(stats["episodes_without_valid_promo"]))
    c[4].metric("Validation checks passed", f"{stats['checks_run']}" if stats["validation_overall_ok"] else "FAILURES")
    st.subheader("Top 5")
    st.dataframe(pd.DataFrame([{"#": r["top5_rank"], "promo": r["promo_id"], "episode": r["episodes"][0], "time": f"{r['segments'][0]['start_ts']}-{r['segments'][-1]['end_ts']}",
                                "score": r["scoring"]["final"], "title": r["title"]} for r in top5]), hide_index=True)
    st.subheader("Promos per episode")
    st.bar_chart(pd.DataFrame({"promos": [e["promos_recommended"] for e in eps]}, index=[e["episode"] for e in eps]))

with t_rec:
    df = pd.DataFrame([{"rank": r["rank"], "promo": r["promo_id"], "ep": r["episodes"][0], "start": r["segments"][0]["start_ts"], "end": r["segments"][-1]["end_ts"],
                        "dur": r["total_duration"], "status": r["status"], "score": r["scoring"]["final"], **r["scoring"]["components"], "category": r["event_category"],
                        "spoiler": r["spoiler_risk"], "title": r["title"]} for r in ranked])
    status = st.selectbox("Status", ["all", "eligible", "flagged"])
    st.dataframe(df if status == "all" else df[df.status == status], hide_index=True)

with t_det:
    pid = st.selectbox("Promo", [r["promo_id"] for r in ranked])
    r = next(x for x in ranked if x["promo_id"] == pid)
    st.subheader(r["title"]); st.write(r["narrative_summary"])
    for i, s in enumerate(r["segments"], 1):
        st.markdown(f"**Segment {i}: episode {s['episode']} {s['start_ts']} -> {s['end_ts']} ({s['duration']}s), scene(s) {s['scene_numbers']}**")
        for d in s["selected_dialogue"]:
            st.write(f"`{d['start_ts']}` **{d['speaker']}**: {d['text']}")
        for a in s["selected_action"]:
            st.caption(f"{a['start_ts']} [{a['kind']}] {a['text']}")
    st.markdown("**Hook** — " + r["hook"]["explanation"]); st.markdown("**Cliffhanger** — " + r["cliffhanger"]["explanation"])
    st.json(r["scoring"]["components"]); st.markdown("**Why it works** — " + r["why_it_works"])
    st.markdown("**Limitations**"); [st.write("- " + x) for x in r["limitations"]]
    if r["why_multi_segment"]:
        st.info(r["why_multi_segment"])

with t_ep:
    n = st.selectbox("Episode", [e["episode"] for e in eps])
    e = next(x for x in eps if x["episode"] == n)
    st.json(e)
    st.dataframe(pd.DataFrame([{"promo": r["promo_id"], "rank": r["rank"], "score": r["scoring"]["final"], "status": r["status"]} for r in ranked if r["episodes"][0] == n]), hide_index=True)

with t_q:
    st.subheader("Validation (against raw text)")
    st.write({"parse level errors": valid["parse_level"]["episodes_with_errors"], "promos passed": valid["promo_level"]["promos_passed"],
              "promos failed": valid["promo_level"]["promos_failed"], "checks run": valid["promo_level"]["checks_run"], "overall ok": valid["overall_ok"]})
    st.subheader("Ranking stability to the weight assumption"); st.json(sens)
    st.subheader("Funnel"); st.json({k: stats[k] for k in ("status_counts", "rejection_reasons_first_failed_gate", "near_duplicates_suppressed")})
    st.info("Human evaluation: fill outputs/evaluation/human_eval_sheet.csv (blank - no human results exist yet).")

with t_exp:
    for name in ("ranked_promos.json", "ranked_promos.csv", "top5_promos.json", "episode_summary.csv", "validation_report.json"):
        p = OUT / name
        if p.exists():
            st.download_button(f"Download {name}", p.read_bytes(), file_name=name)
