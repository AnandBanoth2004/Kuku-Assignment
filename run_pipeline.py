#!/usr/bin/env python3
"""End-to-end pipeline:  ingest -> parse -> QC -> continuity -> events -> candidates -> rank -> validate -> export.

    python run_pipeline.py --input data/input/screenplays.zip --out outputs
    python run_pipeline.py --input path/to/NEW_screenplay.txt --out outputs_new        # same code path for a new show / episode
    python run_pipeline.py --input ... --llm anthropic --model claude-haiku-5-5 --api-key-env ANTHROPIC_API_KEY   # optional LLM layer

Fail-closed: a promo that fails ANY validation check is removed from the recommendations and listed in validation_report.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from config import ROOT, load_config
from src.assembly import build_record
from src.continuity import build_registry
from src.event_detection import detect_moments
from src.evaluation import human_eval_files, weight_sensitivity
from src.ingestion import load_pack
from src.llm_analysis import enrich_moments, review_top
from src.llm_client import LLMError, make_client
from src.promo_builder import generate_candidates, pick_top, promo_id, select_for_episode
from src.timeline import Timeline
from src.validation import RawIndex, validate_everything, validate_promo

log = logging.getLogger("promoai")


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run(input_path, out_dir, cfg, extract_dir=None, client=None, only=None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    raw_dir = out / "raw_outputs"
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "pipeline.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    try:
        # ---- 1. ingest + parse ------------------------------------------------------------------------------
        raw_texts: dict = {}
        episodes, summary = load_pack(Path(input_path), Path(extract_dir or ROOT / "data" / "extracted"), raw_out=raw_texts)
        if only:
            episodes = [e for e in episodes if e.number in only]
        log.info("parsed %d episode(s); failed: %s", len(episodes), summary.failed)
        _dump(out / "parse_summary.json", summary.to_dict())

        # ---- 2. continuity ------------------------------------------------------------------------------------
        alias_file = ROOT / cfg.aliases_file
        reg = build_registry(episodes, alias_file if alias_file.exists() else None)
        _dump(out / "continuity.json", reg.to_dict())

        # ---- 3. events + candidates per episode -----------------------------------------------------------------
        taken: set = set()
        files = {e.number: e.filename for e in episodes}
        tls, moments_all, per_ep, kept_all, best_rej_all, raw_cands = {}, {}, {}, [], {}, {}
        funnel = Counter()
        for ep in episodes:
            tl = Timeline(ep, reg)
            moments = detect_moments(tl, reg)
            n_llm = enrich_moments(client, ep, moments, tl) if client else 0
            cands, st = generate_candidates(tl, reg, cfg)
            for c in cands:
                c.cid = promo_id(c, taken)
            kept, dups, best_rej = select_for_episode(cands, cfg)
            tls[ep.number], moments_all[ep.number] = tl, moments
            per_ep[ep.number] = {"candidates_scored": len(cands), "stats": st, "kept": kept, "dups": dups, "best_rej": best_rej, "llm_moments": n_llm}
            kept_all += kept
            for c in cands:
                funnel[c.scored.status] += 1
                if c.scored.status == "rejected" and c.scored.reasons:
                    funnel["reject:" + c.scored.reasons[0].split(" (")[0]] += 1
            funnel["near_duplicates_suppressed"] += len(dups)
            raw_cands[ep.number] = [{"promo_id": c.cid, "segments": c.segs, "kind": c.kind, "status": c.scored.status, "final": c.scored.final,
                                     "components": c.scored.comps.as_dict(), "reasons": c.scored.reasons, "flags": c.scored.flags,
                                     "duplicate_of": c.duplicate_of} for c in sorted(cands, key=lambda c: -c.scored.final)]
        log.info("candidates: %s", dict(funnel))

        # ---- 4. records, optional LLM second opinion, global rank --------------------------------------------------
        records = {c.cid: build_record(c, tls[c.episode], reg, cfg, moments_all[c.episode], files[c.episode]) for c in kept_all}
        for ep_no, d in per_ep.items():
            for c in d["kept"]:
                records[c.cid]["duplicate_info"]["near_duplicates_suppressed"] = sum(1 for x in d["dups"] if x.duplicate_of == c.cid)
        rejected_rec = {ep_no: build_record(d["best_rej"], tls[ep_no], reg, cfg, moments_all[ep_no], files[ep_no])
                        for ep_no, d in per_ep.items() if not d["kept"] and d["best_rej"] is not None}
        llm_stats = review_top(client, list(records.values()), {e.number: e for e in episodes}, reg, cfg) if client else None

        # ---- 5. validate (fail closed) ---------------------------------------------------------------------------------
        raw_by_ep = {n: RawIndex(t, files.get(n, "")) for n, t in raw_texts.items()}
        invalid = []
        for pid, rec in list(records.items()):
            res = validate_promo(rec, raw_by_ep, cfg)
            if not res.passed:
                invalid.append({"promo_id": pid, "errors": [e.to_dict() for e in res.errors]})
                del records[pid]
                log.error("promo %s FAILED validation and is excluded: %s", pid, [e.code for e in res.errors])
        ranked = sorted(records.values(), key=lambda r: (-r["scoring"]["final"], r["episodes"][0], r["segments"][0]["start"]))
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        cat = {r["promo_id"]: r["event_category"] for r in ranked}

        class _Shim:  # pick_top works on objects with .cid/.episode
            def __init__(self, r):
                self.cid, self.episode = r["promo_id"], r["episodes"][0]
        top_shims, skipped = pick_top([_Shim(r) for r in ranked if r["status"] != "rejected"], cfg, lambda s: cat[s.cid])
        for j, s in enumerate(top_shims, 1):
            records[s.cid]["in_top5"], records[s.cid]["top5_rank"] = True, j
        top5 = [records[s.cid] for s in top_shims]

        # ---- 6. full validation report on exactly what is exported -----------------------------------------------------
        report = validate_everything(episodes, raw_texts, ranked, cfg)
        report["promos_excluded_for_failing_validation"] = invalid
        _dump(out / "validation_report.json", report)

        # ---- 7. episode-wise summary -------------------------------------------------------------------------------------
        ep_rows = []
        for ep in episodes:
            d, mine = per_ep[ep.number], sorted((r for r in ranked if r["episodes"][0] == ep.number), key=lambda r: r["rank"])
            row = {"episode": ep.number, "file": ep.filename, "duration_s": ep.duration, "scenes": len(ep.scenes), "windows_scored": d["candidates_scored"],
                   "promos_recommended": len(mine), "best_promo_id": mine[0]["promo_id"] if mine else None,
                   "best_rank": mine[0]["rank"] if mine else None, "best_final": mine[0]["scoring"]["final"] if mine else None,
                   "best_status": mine[0]["status"] if mine else None, "best_start": mine[0]["segments"][0]["start_ts"] if mine else None,
                   "best_end": mine[0]["segments"][-1]["end_ts"] if mine else None, "moments_detected": len(moments_all[ep.number])}
            if not mine:
                rr = rejected_rec.get(ep.number)
                row["no_valid_promo_reason"] = (rr["reasons"] if rr else ["no legal window of 20-60 s could be built"])
                row["closest_rejected"] = ({"promo_id": rr["promo_id"], "start": rr["segments"][0]["start_ts"], "end": rr["segments"][-1]["end_ts"],
                                            "final": rr["scoring"]["final"], "components": rr["scoring"]["components"]} if rr else None)
            ep_rows.append(row)
        _dump(out / "episode_summary.json", ep_rows)
        cols = ["episode", "promos_recommended", "best_promo_id", "best_rank", "best_final", "best_status", "best_start", "best_end", "windows_scored"]
        with open(out / "episode_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols + ["no_valid_promo_reason"])
            for r in ep_rows:
                w.writerow([r.get(c, "") for c in cols] + ["; ".join(r.get("no_valid_promo_reason", []))])

        # ---- 8. exports -------------------------------------------------------------------------------------------------------
        _dump(out / "ranked_promos.json", ranked)
        _dump(out / "top5_promos.json", top5)
        _dump(out / "closest_rejected_per_episode.json", list(rejected_rec.values()))
        _dump(out / "moments.json", {str(n): [m.to_dict() for m in ms] for n, ms in moments_all.items()})
        with open(out / "ranked_promos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "promo_id", "episode", "kind", "start", "end", "duration_s", "status", "final_score", "hook", "cliffhanger", "emotion",
                        "standalone", "coherence", "character", "event_category", "title", "hook_line", "cliffhanger_line", "flags"])
            for r in ranked:
                c = r["scoring"]["components"]
                w.writerow([r["rank"], r["promo_id"], r["episodes"][0], r["kind"], r["segments"][0]["start_ts"], r["segments"][-1]["end_ts"],
                            r["total_duration"], r["status"], r["scoring"]["final"], c["hook"], c["cliff"], c["emotion"], c["standalone"],
                            c["coherence"], c["character"], r["event_category"], r["title"], r["hook"]["text"], r["cliffhanger"]["text"], " | ".join(r["flags"])])
        for n, rows in raw_cands.items():
            _dump(raw_dir / f"candidates_episode_{n:03d}.json", rows)
        (raw_dir / "README.txt").write_text(
            "RAW, UNEDITED ENGINE OUTPUT of the run that produced ../ranked_promos.json.\n"
            "candidates_episode_NNN.json = every window the engine scored (status, components, rejection reasons) - nothing filtered by hand.\n"
            + ("LLM layer: ENABLED - responses are cached in llm_cache/.\n" if client else
               "LLM layer: NOT USED in this run (provider 'none'): no model output exists; every number is from the deterministic rule engine.\n"), encoding="utf-8")
        ev = out / "evaluation"
        sens = weight_sensitivity(ranked, cfg) if len(ranked) >= 5 else {"note": "too few promos"}
        _dump(ev / "weight_sensitivity.json", sens)
        hes = human_eval_files(ranked, sorted(rejected_rec.values(), key=lambda r: -r["scoring"]["final"]), ev) if ranked else {}
        stats = {"episodes_processed": len(episodes), "windows_scored": sum(d["candidates_scored"] for d in per_ep.values()),
                 "status_counts": {k: funnel[k] for k in ("eligible", "flagged", "rejected")},
                 "rejection_reasons_first_failed_gate": {k[7:]: v for k, v in funnel.items() if k.startswith("reject:")},
                 "near_duplicates_suppressed": funnel["near_duplicates_suppressed"], "promos_recommended": len(ranked),
                 "promos_by_status": dict(Counter(r["status"] for r in ranked)), "multi_segment_promos": sum(1 for r in ranked if r["n_segments"] > 1),
                 "episodes_without_valid_promo": [r["episode"] for r in ep_rows if not r["promos_recommended"]],
                 "promos_excluded_for_failing_validation": len(invalid), "validation_overall_ok": report["overall_ok"],
                 "checks_run": report["promo_level"]["checks_run"], "llm": llm_stats, "human_eval": hes, "skipped_for_top5_diversity": skipped,
                 "runtime_s": round(time.time() - t0, 1)}
        _dump(out / "run_stats.json", stats)
        log.info("done: %s", json.dumps({k: v for k, v in stats.items() if k not in ("human_eval", "skipped_for_top5_diversity")}))
        return {"stats": stats, "ranked": ranked, "top5": top5, "episode_rows": ep_rows, "report": report, "summary": summary}
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find promo candidates in screenplay files")
    ap.add_argument("--input", required=True, help=".zip | folder | single .txt screenplay")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--extract-dir", default=None)
    ap.add_argument("--llm", default=None, choices=["none", "anthropic", "openai_compat"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--only", default=None, help="comma-separated episode numbers (debug)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(a.config)
    for k, v in (("provider", a.llm), ("model", a.model), ("base_url", a.base_url), ("api_key_env", a.api_key_env)):
        if v:
            setattr(cfg.llm, k, v)
    try:
        client = make_client(cfg.llm)
    except LLMError as exc:
        log.error("LLM disabled (%s) - continuing with the rule-based engine", exc)
        client = None
    res = run(a.input, a.out, cfg, a.extract_dir, client, {int(x) for x in a.only.split(",")} if a.only else None)
    s = res["stats"]
    print(f"\nepisodes {s['episodes_processed']} | windows scored {s['windows_scored']} | promos recommended {s['promos_recommended']} "
          f"({s['promos_by_status']}) | validation ok: {s['validation_overall_ok']} ({s['checks_run']} checks) | {s['runtime_s']}s")
    print("episodes without a valid promo:", s["episodes_without_valid_promo"])
    for r in res["top5"]:
        print(f"  top{r['top5_rank']}  {r['promo_id']}  {r['segments'][0]['start_ts']}-{r['segments'][-1]['end_ts']}  score {r['scoring']['final']}  {r['title']}")
    return 0 if s["validation_overall_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
