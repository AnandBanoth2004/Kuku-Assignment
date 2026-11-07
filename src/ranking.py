"""Layer 6 - explainable promo scoring, caps and eligibility gates.

Promo Score = 0.25*Hook + 0.25*Cliffhanger + 0.20*Emotion + 0.15*Standalone + 0.10*Coherence + 0.05*Character
(each component 0-10, final x10 -> 0-100).  The weights are the brief's *initial engineering assumptions*;
they are not empirically fitted industry weights and the score is NOT a prediction of audience engagement.

Guard-rails on top of the weighted sum (all documented, all visible in the output):
  * Weakest-link cap   - final <= 100 * (min(hook, cliffhanger)/10 + 0.25).  A great hook cannot buy back an
                         ending that leaves nothing open (and vice-versa).
  * Hard eligibility   - duration 20-60 s, hook effective inside the first 3 s, a real cliffhanger,
                         not dependent on unexplained context, coherent, valid timestamps.
  * Soft flags         - eligible but weak on one axis, or context/spoiler sensitive -> "flagged" for human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .cliffhanger_detection import CliffResult
from .hook_detection import HookResult
from .timeline import Timeline
from .timeutil import EPS

Segs = Sequence[Tuple[float, float]]


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _rx(words) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % "|".join(words), re.I)


SITUATION = {
    "marriage": _rx([r"wedding", r"marry\w*", r"married", r"husband", r"wife", r"bride", r"groom", r"ring", r"proposal",
                     r"accept my hand", r"betroth\w*", r"fianc\w*"]),
    "family": _rx([r"father", r"mother", r"sister", r"brother", r"son", r"daughter", r"family", r"stepmother", r"parents?"]),
    "authority": _rx([r"duke", r"duchess", r"lord", r"lady", r"guards?", r"judge", r"priest", r"crown", r"court", r"trial",
                      r"verdict", r"magistrate", r"constable", r"king", r"queen", r"throne", r"noble\w*"]),
    "danger": _rx([r"kill\w*", r"die[sd]?", r"dead", r"hang\w*", r"gallows", r"poison\w*", r"burn\w*", r"prison\w*", r"dungeon",
                   r"witch\w*", r"sword", r"blade", r"blood\w*", r"execut\w+", r"fire", r"flames?", r"chains?"]),
    "deceit": _rx([r"lie[sd]?", r"liar", r"betray\w*", r"secret\w*", r"truth", r"stole\w*", r"steal", r"frame[sd]?", r"forged?",
                   r"imposter", r"impostor", r"swore", r"fraud\w*"]),
    "emotion": _rx([r"love", r"heart", r"tears?", r"beg\w*", r"forgive", r"hurt", r"pain\w*"]),
}


@dataclass
class Components:
    hook: float = 0.0
    cliff: float = 0.0
    emotion: float = 0.0
    standalone: float = 0.0
    coherence: float = 0.0
    character: float = 0.0

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class Scored:
    segs: List[Tuple[float, float]]
    comps: Components
    raw: float
    final: float
    capped: bool
    cap_value: float
    status: str                       # eligible | flagged | rejected
    reasons: List[str] = field(default_factory=list)       # why rejected
    flags: List[str] = field(default_factory=list)         # why flagged
    explanations: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ helpers
def content_in(tl: Timeline, segs: Segs):
    out = []
    for S, E in segs:
        out += tl.inside(S, E, ("dialogue", "action", "sfx"))
    return out


def dialogue_in(tl: Timeline, segs: Segs):
    out = []
    for S, E in segs:
        out += tl.dialogue_between(S, E)
    return out


def visual_text(tl: Timeline, segs: Segs) -> str:
    out = []
    for S, E in segs:
        out += [e.text for e in tl.overlapping(S, E, ("camera", "action"))]
    return " ".join(out)


# ------------------------------------------------------------------------------------------------ emotion
def score_emotion(tl: Timeline, segs: Segs, scale: float) -> Tuple[float, str, dict]:
    els = content_in(tl, segs)
    if not els:
        return 0.0, "Nothing is said or done inside the clip.", {}
    vals = sorted(((tl.feat[e.id].intensity, e) for e in els), key=lambda x: -x[0])
    top = [v for v, _ in vals[:3]]
    mean_top = sum(top) / len(top)
    mean_all = sum(v for v, _ in vals) / len(vals)
    score = 10.0 * _clip(scale * (0.55 * mean_top + 0.45 * mean_all))
    peak_v, peak_e = vals[0]
    pk = peak_e.text if len(peak_e.text) <= 80 else peak_e.text[:77] + "..."
    who = f"{peak_e.speaker.title()}: " if peak_e.speaker else ""
    expl = (f"Peak beat \"{who}{pk}\" (intensity {peak_v:.2f}); average over {len(els)} spoken/physical beats {mean_all:.2f}. "
            f"Intensity = delivery tags + stakes wording + exclamations + threat/reveal patterns.")
    return round(score, 2), expl, {"peak_element": peak_e.id, "peak": round(peak_v, 3), "mean": round(mean_all, 3)}


# ------------------------------------------------------------------------------------------------ identity / standalone
def identity(tl: Timeline, segs: Segs, reg) -> Tuple[float, Dict[str, float], List[str]]:
    dl = dialogue_in(tl, segs)
    labels = []
    for d in dl:
        if d.speaker and d.speaker not in labels:
            labels.append(d.speaker)
    spoken = " ".join(d.text for d in dl)
    vis = visual_text(tl, segs)
    ids: Dict[str, float] = {}
    needs: List[str] = []
    for lab in labels:
        kind = reg.kind_of_label(lab) if reg else "role"
        if kind == "named":
            toks = (reg.name_tokens_of_label(lab) if reg else []) or [lab.title()]
            if any(re.search(r"\b%s\b" % re.escape(t), spoken, re.I) for t in toks):
                ids[lab] = 1.0
            elif any(re.search(r"\b%s\b" % re.escape(t), vis, re.I) for t in toks):
                ids[lab] = 0.6
                needs.append(f"{lab.title()} is never addressed by name in the clip")
            else:
                ids[lab] = 0.35
                needs.append(f"{lab.title()} is not introduced in the clip")
        else:
            ids[lab] = 0.8                   # a descriptive role label is self-explanatory on screen
        if all(d.voice for d in dl if d.speaker == lab) and kind == "named":
            ids[lab] *= 0.8
    return (sum(ids.values()) / len(ids) if ids else 0.3), ids, needs


def score_standalone(tl: Timeline, segs: Segs, reg) -> Tuple[float, str, dict]:
    dl = dialogue_in(tl, segs)
    ident, ids, needs = identity(tl, segs, reg)
    dtext = " ".join(d.text for d in dl)
    vtext = visual_text(tl, segs)
    cats_d = {k for k, rx in SITUATION.items() if rx.search(dtext)}
    cats_v = {k for k, rx in SITUATION.items() if rx.search(vtext)} - cats_d
    situation = _clip(0.30 * len(cats_d) + 0.15 * len(cats_v))
    ok = [d for d in dl if tl.feat[d.id].words >= 4 and not tl.feat[d.id].reply_start and not tl.feat[d.id].ellipsis_start]
    self_contained = (len(ok) / len(dl)) if dl else 0.3
    n_sp = len(ids)
    simplicity = 1.0 if n_sp <= 3 else 0.75 if n_sp == 4 else 0.5
    S0 = segs[0][0]
    near_scene_start = any(abs(S0 - s) <= 1.0 for _, s, _ in tl.scenes)
    first_cam = next((e for e in tl.inside(S0, S0 + 2.0, ("camera",))), None)
    estab = bool(first_cam and re.search(r"\b(?:wide|establishing|exterior|interior|full)\b", first_cam.text, re.I))
    setup = 1.0 if near_scene_start else 0.7 if estab else 0.4
    mem = reg.memory.get(tl.ep.number) if reg else None
    dep = 0.0
    dep_note = None
    if mem and mem.depends_on_previous and tl.ep.number > 1:
        dep = 0.04
        if S0 <= 12.0:
            dep += 0.14
            dep_note = f"opens {S0:.0f}s into an episode that picks up directly from episode {tl.ep.number - 1}'s ending"
    raw = 0.30 * ident + 0.25 * situation + 0.20 * self_contained + 0.15 * simplicity + 0.10 * setup - dep
    score = round(10.0 * _clip(raw), 2)
    bits = [f"{n_sp} speaker(s)", f"situation cues: {', '.join(sorted(cats_d | cats_v)) or 'none'}",
            f"{len(ok)}/{len(dl)} lines self-contained"]
    if needs:
        bits.append("; ".join(needs[:2]))
    if dep_note:
        bits.append(dep_note)
    expl = "First-time-viewer proxy: " + "; ".join(bits) + "."
    return score, expl, {"identity": round(ident, 2), "situation": round(situation, 2), "self_contained": round(self_contained, 2),
                         "simplicity": simplicity, "setup": setup, "dependency_penalty": dep, "needs_context": needs,
                         "dependency_note": dep_note, "speakers": list(ids)}


# ------------------------------------------------------------------------------------------------ coherence
def _segment_coherence(tl: Timeline, S: float, E: float) -> Tuple[float, dict]:
    dl = tl.dialogue_between(S, E)
    fd = dl[0] if dl else None
    notes = []
    if fd is None:
        clean_open = 0.5; notes.append("no spoken line")
    else:
        f = tl.feat[fd.id]
        if f.ellipsis_start:
            clean_open = 0.2; notes.append("opens mid-sentence")
        elif f.reply_start and f.words < 8:
            clean_open = 0.45; notes.append(f'opens on a reply ("{fd.text[:30]}") whose question is cut')
        elif fd.start - S > 4.0:
            clean_open = 0.7
        else:
            clean_open = 1.0
    els = sorted(content_in(tl, [(S, E)]), key=lambda e: e.start)
    ints = [tl.feat[e.id].intensity for e in els]
    if ints:
        k = max(1, len(ints) // 3)
        a, c = sum(ints[:k]) / k, sum(ints[-k:]) / k
        peak_pos = ints.index(max(ints)) / max(1, len(ints) - 1)
        mean = sum(ints) / len(ints)
        arc = 0.45 if mean < 0.2 else (1.0 if (c >= a + 0.05 or peak_pos >= 0.5) else 0.65)
    else:
        arc = 0.3
    spk = [d.speaker for d in dl]
    if len(dl) < 2:
        turn = 0.55
    else:
        alt = sum(1 for x, y in zip(spk, spk[1:]) if x != y) / (len(spk) - 1)
        n_sp = len(set(spk))
        turn = (0.6 + 0.4 * min(1.0, alt / 0.6)) if n_sp >= 2 else 0.6
        if n_sp > 5:
            turn *= 0.8
    last_end = max((e.end for e in els), default=S)
    slack = E - last_end
    closure = 1.0 if slack <= 2.0 else 0.8 if slack <= 3.5 else 0.5
    cam_starts = {e.start for e in tl.els if e.kind == "camera"} | {s for _, s, _ in tl.scenes}
    cam_ends = {e.end for e in tl.els if e.kind == "camera"} | {e for _, _, e in tl.scenes}
    cut_q = (0.5 if any(abs(S - x) <= EPS for x in cam_starts) else 0.0) + (0.5 if any(abs(E - x) <= EPS for x in cam_ends) else 0.0)
    cut_quality = 0.6 + 0.4 * cut_q
    mid = (S + E) / 2.0
    first = {d.speaker for d in dl if d.end <= mid + EPS}
    second = {d.speaker for d in dl if d.start >= mid - EPS}
    jac = (len(first & second) / len(first | second)) if first and second else 1.0
    continuity = 0.5 + 0.5 * min(1.0, jac / 0.34)                 # same people in both halves of the clip
    scene_cross = tl.scene_of(S) != tl.scene_of(E - 0.05)
    if scene_cross and jac < 0.5:
        notes.append("crosses a scene change with a different cast")
    score01 = 0.28 * clean_open + 0.20 * arc + 0.18 * turn + 0.12 * closure + 0.10 * cut_quality + 0.12 * continuity
    if scene_cross and jac < 0.5:
        score01 *= 0.85
    return score01, {"clean_open": clean_open, "arc": arc, "turn_taking": turn, "closure": closure, "cut_quality": cut_quality,
                     "speaker_continuity": round(continuity, 2), "scene_cross": scene_cross, "notes": notes}


def transition_score(tl: Timeline, a: Tuple[float, float], b: Tuple[float, float], reg) -> Tuple[float, dict]:
    sa = {reg.canon(d.speaker) if reg else d.speaker for d in tl.dialogue_between(*a)} - {None}
    sb = {reg.canon(d.speaker) if reg else d.speaker for d in tl.dialogue_between(*b)} - {None}
    shared = len(sa & sb) / max(1, min(len(sa), len(sb))) if sa and sb else 0.0
    sc_a, sc_b = tl.scene_of(a[1] - 0.05), tl.scene_of(b[0])
    scene = 1.0 if sc_a == sc_b else 0.7 if abs(sc_a - sc_b) == 1 else 0.4
    gap = b[0] - a[1]
    gapness = 1.0 if gap <= 30 else 0.8 if gap <= 60 else 0.6
    _, det_b = _segment_coherence(tl, *b)
    t = 0.40 * shared + 0.25 * scene + 0.15 * gapness + 0.20 * det_b["clean_open"]
    return t, {"shared_characters": sorted(sa & sb), "scene_a": sc_a, "scene_b": sc_b, "gap_s": round(gap, 2),
               "shared": round(shared, 2), "clean_open_b": det_b["clean_open"]}


def score_coherence(tl: Timeline, segs: Segs, reg) -> Tuple[float, str, dict]:
    parts = [_segment_coherence(tl, S, E) for S, E in segs]
    total = sum(E - S for S, E in segs)
    avg = sum(p[0] * (E - S) for p, (S, E) in zip(parts, segs)) / total
    dur_fit = 1.0 if 25 <= total <= 50 else 0.85 if (20 <= total < 25 or 50 < total <= 60) else 0.6
    det: dict = {"segments": [p[1] for p in parts], "duration_fit": dur_fit}
    if len(segs) == 1:
        score01 = 0.88 * avg + 0.12 * dur_fit
        notes = parts[0][1]["notes"]
        expl = ("Single continuous cut: " + ("; ".join(notes) if notes else "opens on a self-standing line, rising intensity, ends right after the beat") + ".")
    else:
        trans = [transition_score(tl, segs[i], segs[i + 1], reg) for i in range(len(segs) - 1)]
        t_min = min(t for t, _ in trans)
        det["transitions"] = [d for _, d in trans]
        det["transition_score"] = round(t_min, 3)
        score01 = 0.55 * avg + 0.35 * t_min + 0.10 * dur_fit
        d0 = trans[0][1]
        expl = (f"{len(segs)} segments joined: shared characters {d0['shared_characters'] or 'none'}, scenes {d0['scene_a']}->{d0['scene_b']}, "
                f"{d0['gap_s']:.0f}s skipped; transition score {t_min:.2f}.")
    return round(10.0 * _clip(score01), 2), expl, det


# ------------------------------------------------------------------------------------------------ character clarity
def score_character(tl: Timeline, segs: Segs, reg) -> Tuple[float, str, dict]:
    ident, ids, _ = identity(tl, segs, reg)
    n = len(ids)
    count_ok = 0.2 if n == 0 else 0.6 if n == 1 else 1.0 if n <= 4 else 0.45
    score = round(10.0 * _clip(0.55 * ident + 0.45 * count_ok), 2)
    canon = []
    for lab in ids:
        c = reg.canon(lab) if reg else lab.title()
        if c and c not in canon:
            canon.append(c)
    expl = f"{n} speaking role(s) ({', '.join(canon) or 'none'}); {'2-4 is easiest to follow' if 2 <= n <= 4 else 'outside the easy 2-4 range'}; identification {ident:.2f}."
    return score, expl, {"characters": canon}


# ------------------------------------------------------------------------------------------------ combine + gates
def combine(cfg, comps: Components) -> Tuple[float, float, bool, float]:
    raw = 10.0 * sum(w * getattr(comps, k) for k, w in cfg.weights.items())
    cap = 100.0 * (min(comps.hook, comps.cliff) / 10.0 + cfg.weakest_link_floor)
    capped = cap < raw - 1e-9 and cap < 100.0
    final = min(raw, cap, 100.0)
    return round(raw, 2), round(final, 2), capped, round(min(cap, 100.0), 2)


def eligibility(cfg, comps: Components, duration: float, timestamps_ok: bool, hook_in_3s: bool,
                extra_flags: Optional[List[str]] = None) -> Tuple[str, List[str], List[str]]:
    reasons: List[str] = []
    if not (cfg.min_duration - 1e-6 <= duration <= cfg.max_duration + 1e-6):
        reasons.append(f"total duration {duration:.1f}s is outside {cfg.min_duration:.0f}-{cfg.max_duration:.0f}s")
    if not hook_in_3s:
        reasons.append("nothing happens in the first 3 seconds")
    if comps.hook < cfg.hook_min:
        reasons.append(f"hook not effective in the first 3 seconds (score {comps.hook:.2f} < {cfg.hook_min:.2f})")
    if comps.cliff < cfg.cliff_min:
        reasons.append(f"no meaningful cliffhanger (score {comps.cliff:.2f} < {cfg.cliff_min:.2f})")
    if comps.standalone < cfg.standalone_min:
        reasons.append(f"depends heavily on unexplained context (standalone {comps.standalone:.2f} < {cfg.standalone_min:.2f})")
    if comps.coherence < cfg.coherence_min:
        reasons.append(f"incoherent cut (coherence {comps.coherence:.2f} < {cfg.coherence_min:.2f})")
    if comps.emotion < cfg.emotion_min:
        reasons.append(f"emotionally flat (emotion {comps.emotion:.2f} < {cfg.emotion_min:.2f})")
    if not timestamps_ok:
        reasons.append("invalid timestamps")
    flags: List[str] = list(extra_flags or [])
    if not reasons:
        if comps.hook < cfg.soft_flag:
            flags.append(f"modest hook ({comps.hook:.1f})")
        if comps.cliff < cfg.soft_flag:
            flags.append(f"modest cliffhanger ({comps.cliff:.1f})")
        if comps.standalone < cfg.soft_flag:
            flags.append(f"limited standalone clarity ({comps.standalone:.1f})")
        if comps.coherence < cfg.coherence_soft:
            flags.append(f"borderline narrative coherence ({comps.coherence:.1f})")
    status = "rejected" if reasons else ("flagged" if flags else "eligible")
    return status, reasons, flags


def score_window(tl: Timeline, reg, cfg, segs: Segs, hook: HookResult, cliff: CliffResult,
                 extra_flags: Optional[List[str]] = None) -> Scored:
    segs = [(round(S, 2), round(E, 2)) for S, E in segs]
    emo, emo_x, emo_d = score_emotion(tl, segs, cfg.emo_scale)
    sta, sta_x, sta_d = score_standalone(tl, segs, reg)
    coh, coh_x, coh_d = score_coherence(tl, segs, reg)
    cha, cha_x, cha_d = score_character(tl, segs, reg)
    comps = Components(hook=hook.score, cliff=cliff.score, emotion=emo, standalone=sta, coherence=coh, character=cha)
    raw, final, capped, cap = combine(cfg, comps)
    total = sum(E - S for S, E in segs)
    status, reasons, flags = eligibility(cfg, comps, total, True, hook.in_first_3s, extra_flags)
    if capped:
        flags.append(f"score capped at {cap:.0f} by the weakest-link rule (raw {raw:.1f})")
        if status == "eligible":
            status = "flagged"
    expl = {"hook": hook.explanation, "cliff": cliff.explanation, "emotion": emo_x, "standalone": sta_x,
            "coherence": coh_x, "character": cha_x}
    return Scored(segs=list(segs), comps=comps, raw=raw, final=final, capped=capped, cap_value=cap, status=status,
                  reasons=reasons, flags=flags, explanations=expl,
                  details={"emotion": emo_d, "standalone": sta_d, "coherence": coh_d, "character": cha_d})
