"""Layer 3 - hook detection (rule-based).

A hook must work in the FIRST THREE SECONDS of the clip, for someone who has never seen the show.  So the
hook is scored *from the proposed start time*, looking only at elements that begin in [start, start+3 s].

Signals (see cues.py for the lexicons): shock / stakes words, shouted-or-menacing delivery, direct address or
command ("Guards!", "Take your hands off her."), threat / reveal / demand patterns, short-and-clear wording,
and visual shock for action lines.  Penalised: a line that begins mid-sentence ("...and it isn't you"), a bare
reaction ("Why?", "No..."), a reply that needs its question ("Because he is marrying me."), greetings,
long exposition, voice-over, and a start with nothing happening in the first three seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .cues import ElFeat
from .models import Element
from .timeline import Timeline
from .timeutil import EPS, fmt_ts

HOOK_WINDOW = 3.0


@dataclass
class HookResult:
    score: float                      # 0..10
    element_id: Optional[str] = None
    text: str = ""
    kind: str = ""
    speaker: Optional[str] = None
    timestamp: Optional[float] = None
    in_first_3s: bool = False
    cues: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["timestamp_str"] = fmt_ts(self.timestamp) if self.timestamp is not None else None
        return d


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def hook_potential(el: Element, f: ElFeat) -> Tuple[float, List[str]]:
    """0..1 scroll-stopping potential of one element + the cues that fired."""
    cues: List[str] = []
    if el.kind == "dialogue":
        p = 0.16
        if f.shock >= 0.2:
            p += 0.34 * f.shock; cues.append("high-stakes wording")
        if f.delivery >= 0.65:
            p += 0.20 * f.delivery; cues.append(f"intense delivery ({', '.join(el.delivery[:2])})")
        elif f.delivery > 0:
            p += 0.20 * f.delivery
        if f.address:
            p += 0.12; cues.append("direct address / command")
        if f.threat:
            p += 0.10; cues.append("threat")
        elif f.reveal:
            p += 0.10; cues.append("reveal / accusation")
        elif f.demand:
            p += 0.08; cues.append("demand")
        if f.exclaim:
            p += 0.04 * min(f.exclaim, 2)
        if f.question and f.shock >= 0.25:
            p += 0.05; cues.append("loaded question")
        if f.words < 3:
            p -= 0.16 if f.exclaim == 0 else 0.06
            cues.append("very short line")
        elif f.words > 24:
            p -= 0.10; cues.append("long exposition")
        if f.ellipsis_start:
            p -= 0.30; cues.append("starts mid-sentence")
        if f.reply_start and f.words < 7:
            p -= 0.15; cues.append("a reply that needs its question")
        if f.greeting:
            p -= 0.20
        if el.voice == "V.O.":
            p -= 0.15; cues.append("voice-over")
        return _clip(p), cues
    # visual hook: action / sfx / camera text
    p = 0.10 + 0.62 * f.shock + (0.06 if f.sudden else 0.0)
    if f.shock >= 0.3:
        cues.append("strong physical / visual shock")
    elif f.sudden:
        cues.append("sudden event")
    if el.kind == "camera":
        p *= 0.8
    return _clip(p), cues


def score_hook(tl: Timeline, start: float, window: float = HOOK_WINDOW) -> HookResult:
    cands = [e for e in tl.els if e.kind in ("dialogue", "action", "sfx", "camera")
             and start - EPS <= e.start <= start + window + EPS]
    if not any(e.kind != "camera" for e in cands):
        return HookResult(0.0, explanation="Nothing is said or done in the first 3 seconds (camera-only opening) - viewers would scroll past.")
    scored = []
    for e in cands:
        p, cues = hook_potential(e, tl.feat[e.id])
        scored.append((p, e, cues))
    scored.sort(key=lambda x: (-x[0], x[1].start))
    best_p, best_e, best_cues = scored[0]
    second = next((p for p, e, _ in scored[1:] if e.id != best_e.id and e.kind != "camera"), 0.0)
    score = 10.0 * _clip(best_p + 0.20 * second)
    who = f"{best_e.speaker.title()} " if best_e.speaker else ""
    quote = best_e.text if len(best_e.text) <= 110 else best_e.text[:107] + "..."
    offset = best_e.start - start
    expl = (f"{'Says' if best_e.kind == 'dialogue' else 'Shows'} \"{quote}\"" if best_e.kind == "dialogue" else f"Shows: {quote}")
    expl = f"{who}{expl} {offset:.1f}s in. " if best_e.kind == "dialogue" else f"{expl} {offset:.1f}s in. "
    expl += ("Cues: " + "; ".join(best_cues) + ".") if best_cues else "No strong scroll-stopping cue found."
    return HookResult(score=round(score, 2), element_id=best_e.id, text=best_e.text, kind=best_e.kind,
                      speaker=best_e.speaker, timestamp=best_e.start, in_first_3s=True, cues=best_cues, explanation=expl)
