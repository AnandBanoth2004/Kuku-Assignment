"""Layer 4 - cliffhanger detection (rule-based).

A scene or episode ending is NOT automatically a cliffhanger (in the supplied pack eps 4-8, 17 and 19 close
on warm, resolved beats).  A cut counts as a cliffhanger only if the closing beat leaves something open:

  * an unanswered, directed question                       ("Who has been treating you?")
  * a demand / ultimatum / proposal awaiting an answer     ("Lady Juliana... accept my hand.")
  * a threat                                                ("...before he sends you to the gallows.")
  * a pronouncement / reveal                                ("You are not his lawful wife.")
  * a disbelief sting                                       ("Her?!")
  * a sudden event, shock image or key-object insert        (door latch turns, explosion, the planted watch)
  * the NEXT line (a response) exists and is withheld by cutting before it
  * a lingering reaction shot - SUPPORTING signal only (it ends almost every episode)

Penalised: resolution wording, warm / peaceful picture-and-sound cues, reflective voice-over.
The script's own word "cliffhanger" is ignored on purpose (it also labels tender, resolved beats).
When the promo ends on the last frame of an episode, the NEXT episode's continuity note ("Picks up from
Episode N's cliffhanger where ...") gives a small corroboration bonus - for scoring only, never as promo content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .models import Element
from .timeline import Timeline
from .timeutil import EPS, fmt_ts

CLOSING_BEAT_SECONDS = 6.0


@dataclass
class CliffResult:
    score: float                      # 0..10
    element_id: Optional[str] = None
    text: str = ""
    kind: str = ""
    speaker: Optional[str] = None
    timestamp: Optional[float] = None  # the cut (end of the promo segment)
    cues: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)
    withheld_next: Optional[str] = None
    explanation: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["timestamp_str"] = fmt_ts(self.timestamp) if self.timestamp is not None else None
        return d


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_cliffhanger(tl: Timeline, end: float) -> CliffResult:
    # ---- the last spoken line before the cut (weighted down if only reaction shots follow it for a long time)
    last_d: Optional[Element] = tl.last_dialogue_to(end)
    dweight = 1.0
    if last_d is not None:
        gap = end - last_d.end
        if gap > 12.0:
            last_d = None
        elif gap > 8.0:
            dweight = 0.65
        elif gap > 4.0:
            dweight = 0.85
    visuals = [e for e in tl.els if e.kind in ("action", "sfx", "camera") and end - CLOSING_BEAT_SECONDS <= e.end <= end + EPS
               and "out_of_scene_bounds" not in e.flags]
    nxt = tl.first_dialogue_from(end)
    if nxt is not None and (nxt.start - end > 3.0 or tl.scene_of(nxt.start) != tl.scene_of(max(end - 0.05, 0.0))):
        nxt = None
    at_episode_end = end >= tl.duration - 0.3

    cues: List[str] = []
    negs: List[str] = []
    signals: dict = {}
    anchor: Optional[Element] = last_d
    if last_d is not None:
        f = tl.feat[last_d.id]
        low = last_d.text.lower()
        if f.question:
            loaded = f.shock >= 0.2 or f.delivery >= 0.65          # stakes wording or menacing/urgent delivery
            wh = bool(re.search(r"\b(?:who|what|where|why|how|which)\b", low))
            signals["question"] = (1.0 if loaded else 0.8 if wh else 0.45) * dweight
            cues.append("loaded unanswered question" if loaded else "unanswered question" if wh else "mild yes/no question")
        if f.sting:
            signals["sting"] = 0.65 * dweight
            cues.append("disbelief sting")
        if f.demand:
            signals["demand"] = 0.8 * dweight
            cues.append("demand / ultimatum / proposal awaiting an answer")
        if f.threat:
            signals["threat"] = (0.9 if f.delivery >= 0.4 or f.shock >= 0.3 else 0.75) * dweight
            cues.append("threat")
        if f.reveal:
            signals["reveal"] = 0.85 * dweight
            cues.append("pronouncement / reveal")
        if f.resolved:
            negs.append("resolution wording")
        if last_d.voice == "V.O." and not (f.threat or f.reveal or f.question):
            negs.append("reflective voice-over")

    act_vis = [e for e in visuals if e.kind in ("action", "sfx")]
    shocking = [e for e in act_vis if tl.feat[e.id].strong_vis and tl.feat[e.id].shock >= 0.45] + \
               [e for e in visuals if e.kind == "camera" and tl.feat[e.id].strong_vis and tl.feat[e.id].shock >= 0.55]
    sudden = [e for e in act_vis if tl.feat[e.id].sudden]
    objects = [e for e in visuals if tl.feat[e.id].object_reveal]
    if shocking or sudden:
        pick = (shocking or sudden)[-1]
        signals["sudden"] = 0.9 if (shocking and shocking[-1].kind != "camera") else 0.75
        cues.append("sudden event / shock image: " + pick.text[:70])
        if anchor is None or pick.end > anchor.end:
            anchor = pick
    if objects:
        signals["object"] = 0.6
        cues.append("insert on a story object: " + objects[-1].text[:60])
        if anchor is None:
            anchor = objects[-1]
    hold = any(tl.feat[e.id].hold for e in visuals)
    if any(tl.feat[e.id].warm for e in visuals):
        negs.append("warm / peaceful picture or sound")

    primary = max(signals.values(), default=0.0)
    second = sorted(signals.values(), reverse=True)[1] if len(signals) > 1 else 0.0
    if hold and primary >= 0.5:
        cues.append("lingering reaction shot (supporting)")

    withheld, withheld_txt = 0.0, None
    if nxt is not None and last_d is not None:
        withheld = 0.6 if nxt.speaker != last_d.speaker else 0.3
        if tl.feat[nxt.id].reply_start:
            withheld = min(1.0, withheld + 0.15)
        withheld_txt = f'{(nxt.speaker or "").title()}: "{nxt.text[:60]}"'
        cues.append("the reply is cut off (next line withheld)")
    corroboration = 0.0
    if at_episode_end and primary >= 0.5:
        corroboration = 0.1
        if tl.end_signal:
            corroboration = 0.2
            cues.append(f"next episode's continuity note calls this ending a {tl.end_signal['type']}")

    raw = (0.50 * primary + 0.18 * withheld + 0.12 * (1.0 if hold and primary >= 0.5 else 0.0) + 0.08 * second
           + 0.07 * (1.0 if at_episode_end and primary >= 0.5 else 0.0) + 0.05 * (1.0 if corroboration >= 0.2 else 0.0))
    if "resolution wording" in negs:
        raw -= 0.40
    if "warm / peaceful picture or sound" in negs:
        raw -= 0.30 * (1.0 - 0.6 * primary)           # a strong open signal outweighs a warm backdrop
    if "reflective voice-over" in negs:
        raw -= 0.20
    if primary == 0 and withheld == 0:
        raw = min(raw, 0.15)                           # nothing is left open
    score = round(10.0 * _clip(raw), 2)

    if anchor is None:
        return CliffResult(score=min(score, 1.0), cues=cues, negatives=negs,
                           explanation="Ends on camera/sound only - nothing is left unresolved.")
    quote = anchor.text if len(anchor.text) <= 110 else anchor.text[:107] + "..."
    who = f"{anchor.speaker.title()}: " if anchor.kind == "dialogue" and anchor.speaker else ""
    parts = []
    if cues:
        parts.append("Open because: " + "; ".join(cues))
    if negs:
        parts.append("Weakened by: " + "; ".join(negs))
    if not cues and not negs:
        parts.append("The closing beat answers itself - it reads as a pause, not a cliffhanger")
    expl = f'Ends on {who}"{quote}" ' + ". ".join(parts) + "."
    return CliffResult(score=score, element_id=anchor.id, text=anchor.text, kind=anchor.kind, speaker=anchor.speaker,
                       timestamp=round(end, 2), cues=cues, negatives=negs, withheld_next=withheld_txt, explanation=expl)
