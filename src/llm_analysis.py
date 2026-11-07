"""Optional LLM layer (Layers 2-3-4-6 second opinion).  The deterministic pipeline never depends on it.

What the model may do : label dramatic beats, re-judge the six score components of the top-N promos, write title / summary / why-it-works.
What it may NOT do   : choose or change any timestamp, add dialogue, or reference material outside the clip.
Guards               : strict JSON + range checks; every element id must exist inside the clip; hook id must be in the first 3 s
                       (else the hook score cannot exceed the rule score); every quoted fragment must be a verbatim substring of the clip
                       (a hallucinated quote rejects the whole review); a rejected review leaves the rule-based record untouched.
Blending             : component = blend*LLM + (1-blend)*rules; final recomputed with the same weights and weakest-link cap; a gap larger
                       than ``llm.disagreement_flag`` on any component flags the promo for human review.
Only exercised with mock clients in this build environment (see tests/test_llm.py).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .event_detection import CATEGORY_LABELS, Moment
from .llm_client import LLMClient, LLMError, check_schema, extract_json
from .ranking import Components, combine, eligibility
from .validation import norm

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
COMPONENTS = ("hook", "cliff", "emotion", "standalone", "coherence", "character")
CATEGORIES = sorted(CATEGORY_LABELS)


def load_prompt(name: str) -> tuple:
    text = (PROMPTS / name).read_text(encoding="utf-8")
    sys_part, user_part = text.split("### USER")
    return sys_part.replace("### SYSTEM", "").strip(), user_part.strip()


def clip_text(record: dict) -> tuple:
    """Numbered clip listing sent to the model + the set of legal element ids + normalised full text for quote checks."""
    lines, ids, blob = [], set(), []
    for i, seg in enumerate(record["segments"], 1):
        lines.append(f"-- segment {i} ({seg['start_ts']}-{seg['end_ts']}) --")
        items = sorted(seg["selected_dialogue"] + seg["selected_action"] + seg.get("selected_camera", []), key=lambda e: (e["start"], e["kind"] != "camera"))
        for e in items:
            who = f" {e['speaker']}" + (f" ({', '.join(e['delivery'])})" if e["delivery"] else "") if e["kind"] == "dialogue" else ""
            lines.append(f"[{e['element_id']}] {e['start_ts']} {e['kind'].upper()}{who}: {e['text']}")
            ids.add(e["element_id"])
            blob.append(e["text"])
    return "\n".join(lines), ids, norm(" ".join(blob))


def validate_review(obj: dict, record: dict) -> dict:
    check_schema(obj, {"hook": dict, "cliffhanger": dict, "emotion": ((int, float), 0, 10), "standalone": ((int, float), 0, 10),
                       "coherence": ((int, float), 0, 10), "character": ((int, float), 0, 10), "event_category": str,
                       "title": str, "narrative_summary": str, "why_it_works": str, "limitations": list, "quotes": list})
    check_schema(obj["hook"], {"score": ((int, float), 0, 10), "element_id": str, "reason": str})
    check_schema(obj["cliffhanger"], {"score": ((int, float), 0, 10), "element_id": str, "reason": str})
    if obj["event_category"] not in CATEGORY_LABELS:
        raise LLMError(f"schema: unknown event_category {obj['event_category']!r}")
    _, ids, blob = clip_text(record)
    for who in ("hook", "cliffhanger"):
        if obj[who]["element_id"] not in ids:
            raise LLMError(f"hallucinated element id {obj[who]['element_id']!r} ({who}) - not inside the clip")
    for q in obj["quotes"]:
        if not isinstance(q, str) or norm(q) not in blob:
            raise LLMError(f"hallucinated quote rejected: {str(q)[:60]!r}")
    return obj


def apply_review(record: dict, obj: dict, cfg) -> dict:
    """Blend the validated review into the record (in place) and return a small audit dict."""
    rec_before = copy.deepcopy(record["scoring"])
    comps = record["scoring"]["components"]
    llm = {"hook": obj["hook"]["score"], "cliff": obj["cliffhanger"]["score"], "emotion": obj["emotion"], "standalone": obj["standalone"],
           "coherence": obj["coherence"], "character": obj["character"]}
    notes = []
    first_start = record["segments"][0]["start"]
    hook_el = next((e for s in record["segments"][:1] for e in s["selected_dialogue"] + s["selected_action"] + s["selected_camera"]
                    if e["element_id"] == obj["hook"]["element_id"]), None)
    if hook_el is None or hook_el["start"] > first_start + 3.0 + 0.011:
        llm["hook"] = min(llm["hook"], comps["hook"])
        notes.append("LLM hook element is not inside the first 3 s - its hook score was capped at the rule score")
    b = cfg.llm.blend
    new = {k: round(b * llm[k] + (1 - b) * comps[k], 2) for k in COMPONENTS}
    flags = [f"LLM and rules disagree on {k} (rules {comps[k]:.1f} vs LLM {llm[k]:.1f}) - human review" for k in COMPONENTS if abs(llm[k] - comps[k]) > cfg.llm.disagreement_flag]
    c = Components(**new)
    raw, final, capped, cap = combine(cfg, c)
    status, reasons, fl = eligibility(cfg, c, record["total_duration"], True, True)
    if capped and status == "eligible":
        status = "flagged"
    record["scoring"].update({"components": c.as_dict(), "raw": raw, "final": final, "cap": cap, "capped": capped,
                              "rule_scoring": rec_before, "llm_scores": llm, "blend": b})
    record["hook"]["score"], record["cliffhanger"]["score"] = c.hook, c.cliff
    record["status"], record["reasons"], record["flags"] = status, reasons, fl + flags + (["score capped by the weakest-link rule"] if capped else [])
    record["rule_text"] = {k: record[k] for k in ("title", "narrative_summary", "why_it_works")}
    record["title"], record["narrative_summary"], record["why_it_works"] = obj["title"].strip(), obj["narrative_summary"].strip(), obj["why_it_works"].strip()
    record["limitations"] = [str(x) for x in obj["limitations"]] + record["limitations"][:1]
    record["context_required"] = [str(x) for x in obj.get("context_required", [])] or record["context_required"]
    record["event_category"] = obj["event_category"]
    record["provenance"] = {"engine": "rules+llm", "llm_used": True, "blend": b, "notes": notes}
    return {"flags": flags, "notes": notes}


def review_top(client: Optional[LLMClient], records: List[dict], episodes: Dict[int, object], reg, cfg, top_n: Optional[int] = None) -> dict:
    """Re-judge the ``top_n`` best rule-scored records.  Failures are recorded, never raised."""
    stats = {"attempted": 0, "applied": 0, "failed": 0, "errors": []}
    if client is None:
        return stats
    system, user_t = load_prompt("promo_review.txt")
    for rec in sorted(records, key=lambda r: -r["scoring"]["final"])[: top_n or cfg.llm.rescore_top_n]:
        stats["attempted"] += 1
        ep = episodes[rec["episodes"][0]]
        clip, _, _ = clip_text(rec)
        ctx = "; ".join(f"ep {c['episode']}: {c['logline']}" for c in reg.context_for(ep.number, 2)[1:]) if reg else ""
        user = user_t.format(episode=ep.number, logline=ep.logline, context=ctx or "none", total=rec["total_duration"],
                             n_segments=rec["n_segments"], clip=clip, categories=CATEGORIES)
        try:
            obj = validate_review(extract_json(client.complete(system, user)), rec)
            apply_review(rec, obj, cfg)
            stats["applied"] += 1
        except LLMError as exc:
            stats["failed"] += 1
            stats["errors"].append({"promo_id": rec["promo_id"], "error": str(exc)})
            rec["provenance"] = {"engine": "rules", "llm_used": False, "llm_error": str(exc)}
            log.warning("LLM review of %s rejected: %s", rec["promo_id"], exc)
    return stats


def enrich_moments(client: Optional[LLMClient], ep, moments: List[Moment], tl) -> int:
    """Layer 2 with an LLM: replace the semantic fields of rule-detected moments.  Returns the number enriched."""
    if client is None or not moments:
        return 0
    system, user_t = load_prompt("event_analysis.txt")
    beats = "\n".join(f"{m.id} ({m.start:.1f}-{m.end:.1f}s): " + " | ".join(f"[{i}] {tl.el(i).text[:90]}" for i in m.element_ids[:12]) for m in moments)
    try:
        obj = extract_json(client.complete(system, user_t.format(episode=ep.number, logline=ep.logline, beats=beats, categories=CATEGORIES)))
        items = obj.get("moments")
        if not isinstance(items, list):
            raise LLMError("schema: 'moments' must be a list")
        by_id = {m.id: m for m in moments}
        staged = []
        for it in items:
            check_schema(it, {"id": str, "event_category": str, "event_description": str, "emotional_category": str, "significance": ((int, float), 0, 10)})
            if it["id"] not in by_id or it["event_category"] not in CATEGORY_LABELS:
                raise LLMError(f"unknown moment id or category in {it.get('id')!r}")
            staged.append(it)
    except LLMError as exc:
        log.warning("episode %s: LLM moment enrichment rejected (%s); keeping rule-based moments", ep.number, exc)
        return 0
    for it in staged:
        m = by_id[it["id"]]
        m.event_category, m.event_description, m.emotional_category = it["event_category"], it["event_description"], it["emotional_category"]
        m.significance, m.source = float(it["significance"]), "llm"
        m.context_required = [str(x) for x in it.get("context_required", [])] or m.context_required
    return len(staged)
