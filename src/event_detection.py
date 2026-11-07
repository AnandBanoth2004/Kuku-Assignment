"""Layer 2 - dramatic event understanding.

OFFLINE (rule-based) mode: each episode is cut into *beats* (runs of spoken/physical content bounded by scene
changes, pauses, over-long stretches or a new participant) and every beat is classified from cue patterns plus
structural signals.  This is a heuristic proxy - it reads wording, not meaning.  When an LLM is configured,
``llm_analysis.enrich_moments`` replaces the semantic fields (category, description, significance, context)
with schema-validated model output; times always come from the screenplay (the model refers to element ids).

Every moment keeps exact start/end timestamps that are boundaries of real screenplay elements.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .cliffhanger_detection import score_cliffhanger
from .hook_detection import score_hook
from .models import Element
from .timeline import Timeline
from .timeutil import EPS, fmt_ts

CATEGORY_LABELS = {
    "betrayal": "Betrayal", "revenge": "Revenge", "romantic_tension": "Romantic tension", "proposal": "Unexpected proposal",
    "public_humiliation": "Public humiliation", "confrontation": "Confrontation", "shocking_revelation": "Shocking revelation",
    "emotional_breakdown": "Emotional breakdown", "power_shift": "Power shift", "suspense": "Suspense", "threat": "Threat",
    "unexpected_decision": "Unexpected decision", "relationship_conflict": "Relationship conflict",
    "unresolved_situation": "Unresolved situation", "general_drama": "General drama",
}


def _rx(words) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % "|".join(words), re.I)


# category -> (pattern, weight).  Weights are hand-set engineering assumptions.
CATEGORY_CUES = {
    "betrayal": [(_rx([r"betray\w*", r"lied", r"lie[sd]?", r"liar", r"deceiv\w+", r"behind my back", r"changed my mind", r"never loved",
                       r"used you", r"abandon\w*", r"cast aside", r"forsak\w+", r"traitor\w*", r"because he is marrying", r"marrying me"]), 1.0)],
    "revenge": [(_rx([r"reveng\w+", r"vengeance", r"make you pay", r"you will pay", r"settle what", r"justice", r"pay for", r"never forgive",
                      r"avenge\w*", r"reckoning", r"what you stole"]), 1.0)],
    "romantic_tension": [(_rx([r"love", r"kiss\w*", r"embrace\w*", r"heart", r"beautiful", r"my wife", r"my husband", r"closer",
                               r"touch\w*", r"gaze\w*", r"breath\w*", r"tender\w*", r"hand in hand", r"fingers"]), 0.5)],
    "proposal": [(_rx([r"marry me", r"accept my hand", r"my hand", r"proposal", r"propos\w+", r"one knee", r"on his knee", r"be my wife", r"be my husband"]), 1.1)],
    "public_humiliation": [(_rx([r"humiliat\w+", r"disgrace\w*", r"shame\w*", r"worthless", r"beneath", r"beggar", r"peasant", r"laugh\w*",
                                 r"mock\w*", r"whisper\w*", r"crowd", r"guests", r"congregation", r"spectacle", r"everyone"]), 0.6)],
    "confrontation": [(_rx([r"how dare", r"you (?:will|dare|lied|stole|knew|killed)", r"get out", r"enough", r"answer me", r"explain",
                            r"accus\w+", r"stand aside", r"let (?:her|him|me) go", r"take your hands off", r"who are you", r"what have you"]), 0.8)],
    "shocking_revelation": [(_rx([r"you are not", r"you're not", r"alive", r"the truth", r"secret\w*", r"reveal\w*", r"identity", r"i am the",
                                  r"it was you", r"all along", r"mastermind", r"lawful", r"imposter", r"impostor", r"unmask\w*", r"rightful"]), 1.0)],
    "emotional_breakdown": [(_rx([r"sob\w*", r"weep\w*", r"tears?", r"collaps\w+", r"beg\w*", r"please", r"broken", r"trembl\w+", r"shatter\w*",
                                  r"can't breathe", r"cry\w*"]), 0.6)],
    "power_shift": [(_rx([r"guards?", r"arrest\w*", r"rightful", r"kneel\w*", r"bow\w*", r"command\w*", r"orders?", r"disown\w*", r"banish\w*",
                          r"sentence\w*", r"verdict", r"duke", r"duchess", r"crown", r"throne", r"obey", r"seize\w*", r"remove her"]), 0.7)],
    "suspense": [(_rx([r"footsteps", r"door", r"creak\w*", r"latch", r"silence", r"shadow\w*", r"ominous", r"stalk\w*", r"hidden", r"listen\w*",
                       r"watching", r"someone", r"slowly"]), 0.45)],
    "threat": [(_rx([r"gallows", r"hang\w*", r"burn\w*", r"kill\w*", r"die", r"dead", r"regret", r"destroy\w*", r"ruin\w*", r"poison\w*",
                     r"dungeon", r"witch\w*", r"execut\w+", r"flames?", r"ash", r"iron", r"blade", r"sword"]), 0.9)],
    "unexpected_decision": [(_rx([r"i (?:will|have decided|choose|accept|refuse)", r"i'm (?:leaving|staying)", r"i changed", r"decid\w+", r"choose",
                                  r"accept", r"refuse", r"declare\w*", r"vow\w*", r"swear"]), 0.7)],
    "relationship_conflict": [(_rx([r"husband", r"wife", r"sister", r"mother", r"father", r"brother", r"fianc\w+", r"betroth\w+", r"family",
                                    r"marriage", r"stepmother"]), 0.45)],
}
CATEGORY_ORDER = list(CATEGORY_CUES)


@dataclass
class Moment:
    id: str
    episode: int
    scene: int
    start: float
    end: float
    characters: List[str]
    event_category: str
    secondary_category: Optional[str]
    category_confidence: float
    intensity: float                    # 0-10
    significance: float                 # 0-10
    event_description: str
    emotional_category: str
    dramatic_significance: str
    context_required: List[str]
    promo_value: float                  # 0-10 quick preview, NOT the promo score
    element_ids: List[str] = field(default_factory=list)
    source: str = "rules"
    key_quote: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["start_ts"], d["end_ts"] = fmt_ts(self.start), fmt_ts(self.end)
        d["duration"] = round(self.end - self.start, 2)
        return d


def _saturate(x: float, k: float = 1.5) -> float:
    return 1.0 - math.exp(-x / k)


def segment_beats(tl: Timeline) -> List[List[Element]]:
    content = [e for e in tl.els if e.kind in ("dialogue", "action", "sfx")]
    beats: List[List[Element]] = []
    cur: List[Element] = []
    for e in content:
        if cur:
            gap = e.start - cur[-1].end
            new_scene = tl.scene_of(e.start) != tl.scene_of(cur[0].start)
            too_long = e.end - cur[0].start > 24.0
            recent = []
            for x in reversed(cur):
                if x.kind == "dialogue" and x.speaker not in recent:
                    recent.append(x.speaker)
                if len(recent) == 3:
                    break
            newcomer = e.kind == "dialogue" and recent and e.speaker not in recent and len([x for x in cur if x.kind == "dialogue"]) >= 4
            if gap > 2.5 or new_scene or (too_long and e.kind == "dialogue") or newcomer:
                beats.append(cur)
                cur = []
        cur.append(e)
    if cur:
        beats.append(cur)
    return [b for b in beats if len(b) >= 2 and b[-1].end - b[0].start >= 2.0]


def classify(tl: Timeline, beat: List[Element]) -> Dict[str, float]:
    dtext = " ".join(e.text for e in beat if e.kind == "dialogue")
    vtext = " ".join(e.text for e in beat if e.kind != "dialogue")
    scores: Dict[str, float] = {}
    for cat, cues in CATEGORY_CUES.items():
        x = 0.0
        for rx, w in cues:
            x += w * (len(rx.findall(dtext)) + 0.5 * len(rx.findall(vtext)))
        scores[cat] = _saturate(x)
    dl = [e for e in beat if e.kind == "dialogue"]
    last = dl[-1] if dl else None
    n_sp = len({e.speaker for e in dl})
    if n_sp >= 2 and sum(tl.feat[e.id].exclaim for e in dl) >= 2:
        scores["confrontation"] += 0.20
    if any(tl.feat[e.id].threat for e in dl):
        scores["threat"] += 0.25
    if any(tl.feat[e.id].reveal for e in dl):
        scores["shocking_revelation"] += 0.25
    if any(tl.feat[e.id].delivery >= 0.65 and re.search(r"tear|sob|broken|trembl|anguish|agoni", " ".join(e.delivery), re.I) for e in dl):
        scores["emotional_breakdown"] += 0.25
    if last is not None:
        f = tl.feat[last.id]
        if f.question or f.demand:
            scores["unresolved_situation"] = 0.35 + 0.25 * (1 if f.shock >= 0.2 else 0)
    return scores


def _chars(tl: Timeline, beat: List[Element], reg) -> List[str]:
    out: List[str] = []
    for e in beat:
        if e.kind == "dialogue" and e.speaker:
            c = reg.canon(e.speaker) if reg else e.speaker.title()
            if c and c not in out:
                out.append(c)
    return out


def detect_moments(tl: Timeline, reg, hook_cache=None) -> List[Moment]:
    out: List[Moment] = []
    ep = tl.ep.number
    mem = reg.memory.get(ep) if reg else None
    for i, beat in enumerate(segment_beats(tl), 1):
        S, E = beat[0].start, beat[-1].end
        scores = classify(tl, beat)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], CATEGORY_ORDER.index(kv[0]) if kv[0] in CATEGORY_ORDER else 99))
        cat, conf = ranked[0]
        sec = ranked[1][0] if len(ranked) > 1 and ranked[1][1] >= 0.6 * conf and ranked[1][1] >= 0.2 else None
        if conf < 0.18:
            cat, sec = "general_drama", None
        ints = sorted((tl.feat[e.id].intensity for e in beat), reverse=True)
        intensity = 10.0 * min(1.0, 1.35 * (0.6 * (sum(ints[:2]) / max(1, len(ints[:2]))) + 0.4 * (sum(ints) / len(ints))))
        dl = [e for e in beat if e.kind == "dialogue"]
        key = max(dl, key=lambda e: tl.feat[e.id].intensity + 0.3 * tl.feat[e.id].shock, default=None)
        chars = _chars(tl, beat, reg)
        hk = score_hook(tl, S)
        cf = score_cliffhanger(tl, E)
        weight = {"shocking_revelation": 1.0, "betrayal": 0.95, "power_shift": 0.9, "threat": 0.9, "proposal": 0.9, "revenge": 0.9,
                  "confrontation": 0.8, "public_humiliation": 0.8, "emotional_breakdown": 0.7, "unexpected_decision": 0.75,
                  "unresolved_situation": 0.7, "suspense": 0.65, "romantic_tension": 0.6, "relationship_conflict": 0.55,
                  "general_drama": 0.3}.get(cat, 0.5)
        significance = round(min(10.0, 0.55 * intensity + 4.5 * weight * min(1.0, conf + 0.3)), 2)
        ctx: List[str] = []
        if reg:
            for c in chars:
                ch = reg.characters.get(c)
                if ch and ch.kind == "named" and ch.first_episode < ep:
                    spoken = " ".join(e.text for e in dl)
                    if not re.search(r"\b%s\b" % re.escape(c.split()[0]), spoken, re.I):
                        ctx.append(f"{c} (introduced in episode {ch.first_episode}) is not named in this moment")
        if mem and mem.depends_on_previous and S <= 15.0 and mem.previous_ending:
            ctx.append(f"continues episode {ep - 1}'s ending: {mem.previous_ending['snippet'][:110]}")
        quote = key.text if key is not None else ""
        qs = f'"{quote[:90]}"' if quote else ""
        who = " & ".join(chars[:3]) if chars else "unspecified speakers"
        label = CATEGORY_LABELS[cat].lower()
        desc = f"{CATEGORY_LABELS[cat]} between {who}" if len(chars) >= 2 else f"{CATEGORY_LABELS[cat]} involving {who}"
        if qs:
            desc += f" - peak line {qs}"
        out.append(Moment(
            id=f"M{ep:03d}-{i:02d}", episode=ep, scene=tl.scene_of(S), start=round(S, 2), end=round(E, 2), characters=chars,
            event_category=cat, secondary_category=sec, category_confidence=round(conf, 2), intensity=round(intensity, 2),
            significance=significance, event_description=desc, emotional_category=label,
            dramatic_significance=("high" if significance >= 7 else "medium" if significance >= 4.5 else "low"),
            context_required=ctx, promo_value=round(0.45 * intensity + 0.30 * hk.score + 0.25 * cf.score, 2),
            element_ids=[e.id for e in beat], key_quote=quote))
    return out
