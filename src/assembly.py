"""Turn a scored Candidate into the deliverable record (one dict per promo, JSON-serialisable).

Every quoted line / action in a record is copied from the parsed screenplay (never generated); the validator
re-checks each of them against the raw text.  Title / summary / "why it works" are TEMPLATED from those facts and
the score explanations in rule mode (an LLM can rewrite them - see llm_analysis.py - but never the timestamps).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .continuity import spoiler_risk
from .event_detection import CATEGORY_LABELS, Moment
from .promo_builder import Candidate
from .timeline import Timeline
from .timeutil import fmt_ts

ENGINE_NOTE = ("Hook / cliffhanger / emotion are rule-based proxies computed from wording, delivery tags and structure - "
               "they do not read meaning, and the score is not a prediction of audience engagement.")


def _el(e) -> dict:
    return {"element_id": e.id, "kind": e.kind, "speaker": e.speaker, "delivery": list(e.delivery), "voice": e.voice,
            "text": e.text, "start": e.start, "end": e.end, "start_ts": fmt_ts(e.start), "end_ts": fmt_ts(e.end), "line_no": e.line_no}


def moment_for(moments: List[Moment], segs) -> Optional[Moment]:
    best, best_ov = None, 0.0
    for m in moments:
        ov = sum(max(0.0, min(E, m.end) - max(S, m.start)) for S, E in segs)
        if ov > best_ov + 1e-9 or (best is not None and abs(ov - best_ov) < 1e-9 and m.significance > best.significance):
            best, best_ov = m, ov
    return best if best_ov >= 3.0 else None


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build_record(c: Candidate, tl: Timeline, reg, cfg, moments: List[Moment], source_file: str) -> dict:
    sc = c.scored
    segs_out, chars, scenes_seen = [], [], []
    for (S, E) in c.segs:
        dl = tl.dialogue_between(S, E)
        acts = tl.inside(S, E, ("action", "sfx"))
        cams = tl.inside(S, E, ("camera",))
        scenes = sorted({tl.scene_of(S), tl.scene_of(max(S, E - 0.05))})
        scenes_seen += [x for x in scenes if x not in scenes_seen]
        for d in dl:
            cn = reg.canon(d.speaker) if reg else d.speaker
            if cn and cn not in chars:
                chars.append(cn)
        mom = moment_for(moments, [(S, E)])
        lines = [x.line_no for x in dl + acts + cams if x.line_no]
        segs_out.append({
            "episode": c.episode, "source_file": source_file, "scene_numbers": scenes,
            "start": S, "end": E, "start_ts": fmt_ts(S), "end_ts": fmt_ts(E), "duration": round(E - S, 2),
            "selected_dialogue": [dict(_el(d), canonical=(reg.canon(d.speaker) if reg else None)) for d in dl],
            "selected_action": [_el(a) for a in acts], "selected_camera": [_el(x) for x in cams],
            "event_description": mom.event_description if mom else "No single dramatic beat covers this segment (rule engine).",
            "source_lines": [min(lines), max(lines)] if lines else None})
    mom = moment_for(moments, c.segs)
    cat = mom.event_category if mom else "general_drama"
    cat_label = CATEGORY_LABELS.get(cat, cat)
    hook, cliff = c.hook.to_dict(), c.cliff.to_dict()
    if c.cliff.element_id:
        a = tl.el(c.cliff.element_id)
        cliff["anchor_start"], cliff["anchor_end"] = a.start, a.end
    comps = sc.comps.as_dict()
    key = (mom.key_quote if mom and mom.key_quote else c.hook.text) or ""
    title = f"E{c.episode}: {cat_label} — “{_short(key, 52)}”" if key else f"E{c.episode}: {cat_label}"

    ctx = list(sc.details["standalone"].get("needs_context", []))
    dep = sc.details["standalone"].get("dependency_note")
    if dep:
        ctx.append(dep)
    if mom:
        ctx += [x for x in mom.context_required if x not in ctx]
    unresolved = {u["label"] for u in (reg.unresolved if reg else [])}
    unnamed = [d["speaker"] for s in segs_out for d in s["selected_dialogue"] if d["speaker"] and d["speaker"].replace("THE ", "", 1) in unresolved]
    risk = spoiler_risk(reg, c.episode) if reg else "low"
    limits = [ENGINE_NOTE]
    limits += [f"Flag: {f}" for f in sc.flags] + [f"Gate failed: {r}" for r in sc.reasons]
    limits += [f"Needs context: {x}" for x in ctx[:3]]
    if unnamed:
        limits.append("Speaker label(s) never resolved to a name by the continuity layer: " + ", ".join(sorted(set(unnamed))[:3]))
    if risk != "low":
        limits.append(f"Spoiler risk {risk}: this late-series moment reveals story beats a first-time viewer has not reached.")

    first_hook = hook["text"] or ""
    ends_on = (f'{(c.cliff.speaker or "").title() + ": " if c.cliff.kind == "dialogue" else ""}“{_short(c.cliff.text or "", 90)}”'
               if c.cliff.element_id else "a closing shot")
    n_lines = sum(len(s["selected_dialogue"]) for s in segs_out)
    summary = (f"{' & '.join(chars[:3]) or 'Unnamed speakers'} — {mom.event_description if mom else cat_label}. "
               f"Opens on {hook.get('speaker', '') or 'the screen'}: “{_short(first_hook, 80)}” and cuts at {ends_on}. "
               f"{n_lines} spoken line(s) across scene(s) {', '.join(map(str, scenes_seen))} "
               f"(template-generated summary; rule mode does not paraphrase).")
    ranked_parts = sorted((("hook", comps["hook"]), ("cliff", comps["cliff"]), ("emotion", comps["emotion"]),
                           ("standalone", comps["standalone"])), key=lambda kv: -kv[1])[:3]
    names = {"hook": "Hook", "cliff": "Cliffhanger", "emotion": "Emotion", "standalone": "Standalone clarity"}
    why = " ".join(f"{names[k]} {v:.1f}/10 — {sc.explanations[k]}" for k, v in ranked_parts)
    if sc.capped:
        why += f" The weakest-link rule capped the score at {sc.cap_value:.0f} (raw {sc.raw:.1f})."

    return {
        "promo_id": c.cid, "rank": None, "in_top5": False, "top5_rank": None, "title": title,
        "episodes": [c.episode], "total_duration": c.total, "n_segments": len(c.segs), "kind": c.kind,
        "status": sc.status, "flags": list(sc.flags), "reasons": list(sc.reasons),
        "segments": segs_out, "hook": hook, "cliffhanger": cliff,
        "narrative_summary": summary, "why_it_works": why, "limitations": limits, "context_required": ctx,
        "spoiler_risk": risk, "why_multi_segment": c.why_multi, "main_characters": chars,
        "event_category": cat, "secondary_category": mom.secondary_category if mom else None,
        "scoring": {"components": comps, "weights": dict(cfg.weights), "raw": sc.raw, "cap": sc.cap_value, "capped": sc.capped,
                    "final": sc.final, "explanations": dict(sc.explanations),
                    "details": {"standalone": sc.details["standalone"], "coherence": sc.details["coherence"]}},
        "duplicate_info": {"near_duplicates_suppressed": 0},
        "provenance": {"engine": "rules", "llm_used": False},
    }
