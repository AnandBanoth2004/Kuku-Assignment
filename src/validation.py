"""Deterministic QC (no LLM) - the layer that makes every timestamp in the deliverable checkable.

Nothing in here trusts the code that produced the promos.

1. Parse level  - ``validate_episode``: the parsed Episode is internally consistent AND agrees with ``RawIndex``,
   a second, deliberately tiny scanner written separately from ``screenplay_parser.py`` (episode number, number of
   timed blocks, every (start, end, text) triple, scene bounds, episode duration).  Two independent readers of the
   same file that disagree are reported as an error - this is how parser bugs show up.
2. Promo level - ``validate_promo``: each promo is re-checked against the RAW text of its episode:
   exact source boundaries, no cut inside a spoken line, no split "..."-sentence, duration recomputed, 20-60 s total,
   chronological non-overlapping segments, hook inside the first 3 s, cliffhanger anchor at the cut, every quoted
   line / action present verbatim and none silently omitted, scores in range and recomputed from the configured weights
   (including the weakest-link cap), status consistent with the gates, no reference to a later episode.
3. Collection level - ``validate_collection``: unique ids, no near-duplicates, consecutive ranks, top-5 diversity.

Limits (stated, not hidden): these checks prove the promos are *well-formed and faithful to the source text*.  They do not
(and cannot) prove a promo is *good* - that needs human judgement (see the human-evaluation sheet).
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Episode
from .ranking import Components, combine, eligibility

TOL = 0.011                       # timestamps in the source have 2 decimals
SUM_TOL = 0.021                   # tolerance when comparing sums of several 2-decimal values
CLIFF_LOOKBACK = 12.0             # cliffhanger anchor may be at most this far before the cut
HOOK_WINDOW = 3.0


# ====================================================================================== independent raw scanner
_TS = r"(\d+(?::\d{2}){1,2}(?:[.,]\d+)?)"
_RX_EPNUM = re.compile(r"^\s*episode\s+number\s*[:\-]\s*(?:episode\s*)?#?\s*(\d+)", re.I)
_RX_LABEL = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*$")
_RX_TIMES = re.compile(r"^\s*start\s*[:=]\s*\(?\s*" + _TS + r"\s*\)?\s*[|,;]\s*end\s*[:=]\s*\(?\s*" + _TS + r"\s*\)?\s*$", re.I)
_RX_SCENE = re.compile(r"^\s*scene\s+(\d+)\b", re.I)
_RX_SCENE_T = re.compile(r"^\s*scene\s+(start|end)\s*[:\-]\s*" + _TS + r"\s*$", re.I)
_NONSPEECH = {"CAMERA": "camera", "SHOT": "camera", "ACTION": "action", "ACTIONS": "action", "SFX": "sfx", "SOUND": "sfx",
              "SOUND FX": "sfx", "SOUND EFFECT": "sfx", "SOUND EFFECTS": "sfx"}
_QUOTE_CHARS = "\"'“”‘’„«»`´"
_WS = re.compile(r"\s+")
_RX_LEAD_ELLIPSIS = re.compile(r"^\W{0,2}(?:\.{2,}|…)")
_RX_TRAIL_ELLIPSIS = re.compile(r"(?:\.{2,}|…)\W{0,2}$")
_RX_EP_MENTION = re.compile(r"\bepisodes?\s+(\d+)\b", re.I)


def secs(ts: str) -> float:
    """'01:29.30' -> 89.3 (own tiny implementation on purpose, not ``timeutil.parse_ts``)."""
    total = 0.0
    for part in ts.replace(",", ".").split(":"):
        total = total * 60.0 + float(part)
    return round(total, 2)


def norm(text: str) -> str:
    """Comparison form: quote marks removed, ellipsis unified, whitespace collapsed, lower-case."""
    t = (text or "").replace("…", "...")
    t = "".join(ch for ch in t if ch not in _QUOTE_CHARS)
    return _WS.sub(" ", t).strip().lower()


def speaker_norm(label: Optional[str]) -> str:
    base = re.sub(r"\([^()]*\)", "", label or "")
    return _WS.sub(" ", base).strip(" :-").upper()


def _kind_of_label(label: str) -> str:
    base = speaker_norm(label)
    return _NONSPEECH.get(base, "dialogue")


@dataclass(frozen=True)
class RawBlock:
    line_no: int
    label: str
    kind: str                     # camera | action | sfx | dialogue
    start: float
    end: float
    text: str                     # body line, surrounding quotes removed for dialogue

    @property
    def speaker(self) -> str:
        return speaker_norm(self.label) if self.kind == "dialogue" else ""


class RawIndex:
    """Minimal second reader of a screenplay file - only what QC needs, written independently of the main parser."""

    def __init__(self, text: str, filename: str = ""):
        self.filename = filename
        self.episode_number: Optional[int] = None
        self.blocks: List[RawBlock] = []
        self.scenes: List[list] = []              # [number, start|None, end|None]
        lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        n, i = len(lines), 0
        while i < n:
            ln = lines[i]
            m = _RX_EPNUM.match(ln)
            if m and self.episode_number is None and not self.scenes:
                self.episode_number = int(m.group(1))
                i += 1
                continue
            m = _RX_SCENE.match(ln)
            if m:
                self.scenes.append([int(m.group(1)), None, None])
                i += 1
                continue
            m = _RX_SCENE_T.match(ln)
            if m and self.scenes:
                self.scenes[-1][1 if m.group(1).lower() == "start" else 2] = secs(m.group(2))
                i += 1
                continue
            m = _RX_LABEL.match(ln)
            if m:
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                t = _RX_TIMES.match(lines[j]) if j < n else None
                if t:
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    body = ""
                    if k < n and not _RX_LABEL.match(lines[k]) and not _RX_SCENE.match(lines[k]):
                        body = lines[k].strip()
                    label = m.group(1).strip().rstrip(":").strip()
                    kind = _kind_of_label(label)
                    if kind == "dialogue" and len(body) >= 2 and body[0] in _QUOTE_CHARS and body[-1] in _QUOTE_CHARS:
                        body = body[1:-1].strip()
                    elif kind == "dialogue" and body[:1] in _QUOTE_CHARS:
                        body = body[1:].strip()
                    self.blocks.append(RawBlock(i + 1, label, kind, secs(t.group(1)), secs(t.group(2)), body))
                    i = (k + 1) if body else (j + 1)
                    continue
            i += 1
        self.dialogue = [b for b in self.blocks if b.kind == "dialogue"]
        self._starts = sorted({b.start for b in self.blocks} | {s[1] for s in self.scenes if s[1] is not None})
        self._ends = sorted({b.end for b in self.blocks} | {s[2] for s in self.scenes if s[2] is not None})
        ends = [b.end for b in self.blocks] + [s[2] for s in self.scenes if s[2] is not None]
        self.duration = max(ends) if ends else 0.0

    # ---------------------------------------------------------------------------------------------- lookups
    @staticmethod
    def _near(sorted_vals: List[float], t: float) -> bool:
        k = bisect.bisect_left(sorted_vals, t - TOL)
        return k < len(sorted_vals) and sorted_vals[k] <= t + TOL

    def has_start(self, t: float) -> bool:
        return self._near(self._starts, t)

    def has_end(self, t: float) -> bool:
        return self._near(self._ends, t)

    def spoken_line_containing(self, t: float) -> Optional[RawBlock]:
        for b in self.dialogue:
            if b.start + TOL < t < b.end - TOL:
                return b
        return None

    def find(self, start: float, end: float, text: str, kinds: Optional[Iterable[str]] = None, speaker: Optional[str] = None) -> Optional[RawBlock]:
        want = norm(text)
        for b in self.blocks:
            if kinds is not None and b.kind not in kinds:
                continue
            if abs(b.start - start) <= TOL and abs(b.end - end) <= TOL and norm(b.text) == want:
                if speaker is None or b.speaker == speaker_norm(speaker):
                    return b
        return None

    def spoken_inside(self, S: float, E: float) -> List[RawBlock]:
        return [b for b in self.dialogue if b.start >= S - TOL and b.end <= E + TOL]

    def non_speech_starting_in(self, S: float, E: float, kinds: Iterable[str]) -> List[RawBlock]:
        ks = set(kinds)
        return [b for b in self.blocks if b.kind in ks and b.start >= S - TOL and b.start < E - TOL]

    def continues_into(self, E: float) -> Optional[RawBlock]:
        """If the spoken line that ends at the cut ends with '...' and the same speaker continues with '...', return the continuation."""
        prev = None
        for b in self.dialogue:
            if b.end <= E + TOL:
                prev = b
            else:
                break
        if prev is None or not _RX_TRAIL_ELLIPSIS.search(prev.text):
            return None
        for b in self.dialogue:
            if b.start >= E - TOL and b.start - E < 2.0 and b.speaker == prev.speaker and _RX_LEAD_ELLIPSIS.search(b.text):
                return b
        return None

    def continues_from(self, S: float) -> Optional[RawBlock]:
        """If the first spoken line (within 3 s of the cut) starts with '...' and the same speaker just trailed off, return that line."""
        first = None
        for b in self.dialogue:
            if b.start >= S - TOL:
                first = b
                break
        if first is None or first.start - S > HOOK_WINDOW + TOL or not _RX_LEAD_ELLIPSIS.search(first.text):
            return None
        for b in self.dialogue:
            if b.end <= S + TOL and S - b.end < 4.0 and b.speaker == first.speaker and _RX_TRAIL_ELLIPSIS.search(b.text):
                return first
        return None


# ====================================================================================== issues / results
@dataclass
class Issue:
    level: str                    # "error" | "warning"
    code: str
    message: str
    where: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PromoValidation:
    promo_id: str
    n_checks: int = 0
    errors: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"promo_id": self.promo_id, "passed": self.passed, "n_checks": self.n_checks,
                "errors": [e.to_dict() for e in self.errors], "warnings": [w.to_dict() for w in self.warnings]}


class _Checker:
    def __init__(self, pid: str):
        self.res = PromoValidation(promo_id=pid)

    def check(self, cond, code: str, message: str, where: str = "", level: str = "error") -> bool:
        self.res.n_checks += 1
        if not cond:
            (self.res.errors if level == "error" else self.res.warnings).append(Issue(level, code, message, where or self.res.promo_id))
        return bool(cond)


# ====================================================================================== parse-level validation
def validate_episode(ep: Episode, raw: Optional[RawIndex]) -> List[Issue]:
    """Internal consistency of the parsed episode + agreement with the independent raw scanner."""
    out: List[Issue] = []
    where = f"episode {ep.number} ({ep.filename})"

    def err(code, msg, level="error"):
        out.append(Issue(level, code, msg, where))

    if not ep.scenes:
        err("NO_SCENES", "no scenes were parsed")
        return out
    prev_end = None
    for sc in ep.scenes:
        if sc.start is None or sc.end is None:
            err("SCENE_NO_TIMES", f"scene {sc.number} has no start/end", "warning")
            continue
        if sc.end <= sc.start:
            err("SCENE_BAD_RANGE", f"scene {sc.number} ends ({sc.end}) before it starts ({sc.start})")
        if prev_end is not None and abs(sc.start - prev_end) > TOL:
            err("SCENE_GAP_OR_OVERLAP", f"scene {sc.number} starts at {sc.start} but the previous scene ended at {prev_end}", "warning")
        prev_end = sc.end
    for e in ep.elements():
        if e.start is None or e.end is None:
            err("ELEMENT_UNTIMED", f"{e.id} has no timestamps", "warning")
        elif e.end <= e.start:
            err("ELEMENT_BAD_RANGE", f"{e.id} end {e.end} <= start {e.start}")
        elif e.start < -TOL or e.end > ep.duration + TOL:
            err("ELEMENT_OUTSIDE_EPISODE", f"{e.id} [{e.start}, {e.end}] lies outside the episode (0..{ep.duration})")

    if raw is None:
        err("NO_RAW_TEXT", "independent raw-text cross-check skipped (raw text not supplied)", "warning")
        return out

    if raw.episode_number is not None and raw.episode_number != ep.number and ep.number_source == "header":
        err("EPISODE_NUMBER_MISMATCH", f"parser says episode {ep.number}, raw header says {raw.episode_number}")
    dups = sum(1 for w in ep.warnings if w.get("code") == "DUP_ELEMENT")
    parsed = [e for e in ep.elements() if e.start is not None and e.end is not None]
    if len(raw.blocks) != len(parsed) + dups:
        err("BLOCK_COUNT_MISMATCH", f"raw scanner finds {len(raw.blocks)} timed blocks, parser kept {len(parsed)} (+{dups} exact duplicates dropped)")
    for kind in ("dialogue", "action", "camera", "sfx"):
        r = sum(1 for b in raw.blocks if b.kind == kind)
        p = sum(1 for e in parsed if e.kind == kind)
        if abs(r - p) > dups:
            err("KIND_COUNT_MISMATCH", f"{kind}: raw scanner {r} vs parser {p}")
    raw_keys = Counter((b.start, b.end, norm(b.text)) for b in raw.blocks)
    par_keys = Counter((round(e.start, 2), round(e.end, 2), norm(e.text)) for e in parsed)
    missing = raw_keys - par_keys
    extra = par_keys - raw_keys
    n_missing, n_extra = sum(missing.values()), sum(extra.values())
    if n_missing - dups > 0 or n_extra > 0:
        ex = list(missing)[:2] + list(extra)[:2]
        err("CONTENT_MISMATCH", f"{n_missing} raw block(s) not reproduced by the parser and {n_extra} parser element(s) not in the raw text, e.g. {ex}")
    if abs(raw.duration - ep.duration) > TOL:
        err("DURATION_MISMATCH", f"raw scanner duration {raw.duration} vs parser {ep.duration}", "warning")
    parsed_scenes = {sc.number: sc for sc in ep.scenes}
    for num, s, e in raw.scenes:
        sc = parsed_scenes.get(num)
        if sc is None:
            err("SCENE_MISSING", f"scene {num} is in the raw text but not in the parse")
        elif s is not None and sc.start is not None and not sc.times_inferred and (abs(sc.start - s) > TOL or abs((sc.end or 0) - (e or 0)) > TOL):
            err("SCENE_BOUNDS_MISMATCH", f"scene {num}: raw [{s}, {e}] vs parser [{sc.start}, {sc.end}]")
    return out


# ====================================================================================== promo-level validation
def _components(scoring: dict) -> Optional[Components]:
    try:
        c = scoring["components"]
        return Components(**{k: float(c[k]) for k in ("hook", "cliff", "emotion", "standalone", "coherence", "character")})
    except (KeyError, TypeError, ValueError):
        return None


def _text_blobs(promo: dict) -> List[str]:
    blobs = [promo.get("narrative_summary", ""), promo.get("why_it_works", ""), promo.get("title", "")]
    blobs += [str(x) for x in promo.get("context_required", []) or []]
    blobs += [str(x) for x in promo.get("limitations", []) or []]
    for seg in promo.get("segments", []):
        blobs.append(seg.get("event_description", ""))
    return blobs


def validate_promo(promo: dict, raw_by_ep: Dict[int, RawIndex], cfg) -> PromoValidation:
    pid = str(promo.get("promo_id", "<missing id>"))
    c = _Checker(pid)
    ok = c.check

    # ---- schema ----------------------------------------------------------------------------------
    required = ("promo_id", "episodes", "segments", "total_duration", "hook", "cliffhanger", "scoring", "status")
    missing = [k for k in required if k not in promo]
    if not ok(not missing, "SCHEMA_MISSING_KEYS", f"missing keys: {missing}"):
        return c.res
    segs = promo["segments"]
    if not ok(isinstance(segs, list) and len(segs) >= 1, "SCHEMA_NO_SEGMENTS", "promo has no segments"):
        return c.res
    for i, s in enumerate(segs):
        if not ok(all(k in s for k in ("episode", "start", "end")), "SCHEMA_SEGMENT", f"segment {i + 1} lacks episode/start/end"):
            return c.res

    # ---- per segment: boundaries, speech, text ---------------------------------------------------
    for i, s in enumerate(segs, 1):
        w = f"{pid} segment {i}"
        ep, S, E = int(s["episode"]), float(s["start"]), float(s["end"])
        raw = raw_by_ep.get(ep)
        if not ok(raw is not None, "EPISODE_UNKNOWN", f"episode {ep} is not in the loaded pack", w):
            continue
        ok(raw.episode_number is None or raw.episode_number == ep, "EPISODE_HEADER", f"raw file header says episode {raw.episode_number}, promo says {ep}", w)
        ok(S < E - TOL, "START_BEFORE_END", f"start {S} is not before end {E}", w)
        ok(abs((E - S) - float(s.get("duration", E - S))) <= SUM_TOL, "SEGMENT_DURATION",
           f"stated duration {s.get('duration')} != recomputed {round(E - S, 2)}", w)
        for key, val in (("start_ts", S), ("end_ts", E)):
            if key in s:
                try:
                    good = abs(secs(s[key]) - val) <= TOL
                except Exception:
                    good = False
                ok(good, "TIMESTAMP_STRING", f"{key} {s[key]!r} does not equal {val}", w)
        ok(S >= -TOL and E <= raw.duration + TOL, "WITHIN_EPISODE", f"[{S}, {E}] outside episode length {raw.duration}", w)
        ok(raw.has_start(S), "START_NOT_A_SOURCE_BOUNDARY", f"{S} is not the start of any shot, line or scene in the source", w)
        ok(raw.has_end(E), "END_NOT_A_SOURCE_BOUNDARY", f"{E} is not the end of any shot, line or scene in the source", w)
        cut_in = raw.spoken_line_containing(S)
        ok(cut_in is None, "CUT_INSIDE_SPEECH_START", f"start {S} falls inside the spoken line at line {cut_in.line_no if cut_in else ''}", w)
        cut_in = raw.spoken_line_containing(E)
        ok(cut_in is None, "CUT_INSIDE_SPEECH_END", f"end {E} falls inside the spoken line at line {cut_in.line_no if cut_in else ''}", w)
        cont = raw.continues_from(S)
        ok(cont is None, "SPLIT_SENTENCE_START", f"clip opens on the tail of a sentence (line {cont.line_no if cont else ''}: '...')", w)
        cont = raw.continues_into(E)
        ok(cont is None, "SPLIT_SENTENCE_END", f"clip cuts a sentence in half (continues at line {cont.line_no if cont else ''})", w)

        # selected dialogue: verbatim, inside the segment, none omitted
        listed = s.get("selected_dialogue", []) or []
        bad = []
        matched_keys = set()
        for d in listed:
            rb = raw.find(float(d.get("start", -1)), float(d.get("end", -1)), d.get("text", ""), ("dialogue",), d.get("speaker"))
            if rb is None:
                bad.append(f"{d.get('speaker')}: {str(d.get('text'))[:40]!r} @ {d.get('start')}")
            else:
                matched_keys.add((rb.start, rb.end))
                if not (rb.start >= S - TOL and rb.end <= E + TOL):
                    bad.append(f"{rb.speaker} line at {rb.start} lies outside the segment")
        ok(not bad, "DIALOGUE_NOT_VERBATIM", f"{len(bad)} listed spoken line(s) not found verbatim inside the segment in the source, e.g. {bad[:2]}", w)
        omitted = [b for b in raw.spoken_inside(S, E) if (b.start, b.end) not in matched_keys]
        ok(not omitted, "DIALOGUE_OMITTED", f"{len(omitted)} spoken line(s) inside the segment are not listed, e.g. {[(b.speaker, b.text[:30]) for b in omitted[:2]]}", w)

        # selected actions / sound effects: verbatim, none omitted
        acts = s.get("selected_action", []) or []
        bad, matched_keys = [], set()
        for a in acts:
            rb = raw.find(float(a.get("start", -1)), float(a.get("end", -1)), a.get("text", ""), ("action", "sfx"))
            if rb is None:
                bad.append(f"{str(a.get('text'))[:40]!r} @ {a.get('start')}")
            else:
                matched_keys.add((rb.start, rb.end, norm(rb.text)))
        ok(not bad, "ACTION_NOT_VERBATIM", f"{len(bad)} listed action/sfx line(s) not found verbatim in the source, e.g. {bad[:2]}", w)
        omitted = [b for b in raw.non_speech_starting_in(S, E, ("action", "sfx")) if (b.start, b.end, norm(b.text)) not in matched_keys]
        ok(not omitted, "ACTION_OMITTED", f"{len(omitted)} action/sfx line(s) starting inside the segment are not listed", w)
        cams = s.get("selected_camera", []) or []
        bad = [x for x in cams if raw.find(float(x.get("start", -1)), float(x.get("end", -1)), x.get("text", ""), ("camera",)) is None]
        ok(not bad, "CAMERA_NOT_VERBATIM", f"{len(bad)} listed camera shot(s) not found verbatim in the source", w)

    # ---- whole promo: duration, order, bookkeeping -------------------------------------------------
    total = sum(float(s["end"]) - float(s["start"]) for s in segs)
    ok(abs(total - float(promo["total_duration"])) <= SUM_TOL, "TOTAL_DURATION_SUM", f"stated total {promo['total_duration']} != sum of segments {round(total, 2)}")
    ok(cfg.min_duration - TOL <= total <= cfg.max_duration + TOL, "TOTAL_DURATION_RANGE", f"total {round(total, 2)}s outside {cfg.min_duration:.0f}-{cfg.max_duration:.0f}s")
    order_ok, contiguous = True, False
    for a, b in zip(segs, segs[1:]):
        if (int(b["episode"]), float(b["start"])) < (int(a["episode"]), float(a["end"])) and int(b["episode"]) == int(a["episode"]) and float(b["start"]) < float(a["end"]) - TOL:
            order_ok = False
        if int(b["episode"]) < int(a["episode"]):
            order_ok = False
        if int(b["episode"]) == int(a["episode"]) and abs(float(b["start"]) - float(a["end"])) <= TOL:
            contiguous = True
    ok(order_ok, "SEGMENTS_NOT_CHRONOLOGICAL", "segments overlap or run backwards in time")
    ok(not contiguous, "SEGMENTS_CONTIGUOUS", "two segments touch - they should be one continuous segment", level="warning")
    eps = []
    for s in segs:
        if int(s["episode"]) not in eps:
            eps.append(int(s["episode"]))
    ok(list(promo["episodes"]) == eps, "EPISODES_FIELD", f"episodes field {promo['episodes']} != episodes of the segments {eps}")
    if "n_segments" in promo:
        ok(int(promo["n_segments"]) == len(segs), "N_SEGMENTS", f"n_segments {promo['n_segments']} != {len(segs)}")
    if "kind" in promo:
        ok((promo["kind"] == "continuous") == (len(segs) == 1), "KIND_FIELD", f"kind {promo['kind']!r} inconsistent with {len(segs)} segment(s)")
    if len(segs) > 1:
        ok(bool(promo.get("why_multi_segment")), "MULTI_SEGMENT_JUSTIFIED", "a multi-segment promo must carry a justification")

    # ---- hook ----------------------------------------------------------------------------------------
    S0, ep0 = float(segs[0]["start"]), int(segs[0]["episode"])
    hook = promo["hook"] or {}
    raw0 = raw_by_ep.get(ep0)
    if raw0 is not None:
        if hook.get("element_id"):
            hb = [b for b in raw0.blocks if S0 - TOL <= b.start <= S0 + HOOK_WINDOW + TOL and norm(b.text) == norm(hook.get("text", ""))]
            ok(bool(hb), "HOOK_NOT_IN_FIRST_3S", f"hook text {str(hook.get('text'))[:50]!r} is not an element starting within 3 s of the cut {S0}")
            if hb:
                ok(abs(float(hook.get("timestamp", -1)) - hb[0].start) <= TOL, "HOOK_TIMESTAMP", f"hook timestamp {hook.get('timestamp')} != source {hb[0].start}")
        else:
            ok(float(hook.get("score", 0)) <= TOL, "HOOK_EMPTY_BUT_SCORED", "no hook element but a non-zero hook score")

    # ---- cliffhanger --------------------------------------------------------------------------------
    Eend, epN = float(segs[-1]["end"]), int(segs[-1]["episode"])
    cl = promo["cliffhanger"] or {}
    rawN = raw_by_ep.get(epN)
    if rawN is not None:
        if cl.get("timestamp") is not None:
            ok(abs(float(cl["timestamp"]) - Eend) <= TOL, "CLIFF_TIMESTAMP", f"cliffhanger timestamp {cl['timestamp']} != promo end {Eend}")
        if cl.get("element_id"):
            ab = None
            if cl.get("anchor_start") is not None and cl.get("anchor_end") is not None:
                ab = rawN.find(float(cl["anchor_start"]), float(cl["anchor_end"]), cl.get("text", ""))
            else:
                ab = next((b for b in rawN.blocks if norm(b.text) == norm(cl.get("text", "")) and b.end <= Eend + TOL), None)
            if ok(ab is not None, "CLIFF_NOT_IN_SOURCE", f"cliffhanger anchor {str(cl.get('text'))[:50]!r} not found in the source"):
                ok(ab.end <= Eend + TOL, "CLIFF_AFTER_CUT", f"cliffhanger anchor ends {ab.end}, after the cut {Eend}")
                ok(Eend - ab.end <= CLIFF_LOOKBACK + TOL, "CLIFF_FAR_FROM_CUT", f"cliffhanger anchor ends {Eend - ab.end:.1f}s before the cut")
        else:
            ok(float(cl.get("score", 0)) <= 1.0 + TOL, "CLIFF_EMPTY_BUT_SCORED", "no cliffhanger anchor but a high cliffhanger score", level="warning")

    # ---- scores ---------------------------------------------------------------------------------------
    sc = promo["scoring"]
    comps = _components(sc)
    if ok(comps is not None, "SCORE_COMPONENTS", "scoring.components missing or malformed"):
        vals = comps.as_dict()
        ok(all(0.0 - 1e-9 <= v <= 10.0 + 1e-9 for v in vals.values()), "SCORE_RANGE", f"a component is outside 0-10: {vals}")
        raw_s, final, capped, cap = combine(cfg, comps)
        ok(abs(raw_s - float(sc.get("raw", raw_s))) <= 0.06, "SCORE_RAW", f"stated raw {sc.get('raw')} != recomputed {raw_s}")
        ok(abs(final - float(sc.get("final", final))) <= 0.06, "SCORE_FINAL", f"stated final {sc.get('final')} != recomputed {final} (weights {cfg.weights}, floor {cfg.weakest_link_floor})")
        ok(0.0 <= float(sc.get("final", 0)) <= 100.0 + 1e-9, "SCORE_FINAL_RANGE", f"final {sc.get('final')} outside 0-100")
        ok(abs(float(hook.get("score", comps.hook)) - comps.hook) <= 0.01, "HOOK_SCORE_CONSISTENT", "hook.score differs from scoring.components.hook")
        ok(abs(float(cl.get("score", comps.cliff)) - comps.cliff) <= 0.01, "CLIFF_SCORE_CONSISTENT", "cliffhanger.score differs from scoring.components.cliff")
        ok(final <= 100.0 * (min(comps.hook, comps.cliff) / 10.0 + cfg.weakest_link_floor) + 0.06, "WEAKEST_LINK_CAP", "final score exceeds the weakest-link cap")
        in3 = bool(hook.get("element_id"))
        exp_status, _, _ = eligibility(cfg, comps, total, True, in3)
        if capped and exp_status == "eligible":
            exp_status = "flagged"
        ok(promo["status"] == exp_status, "STATUS_CONSISTENT", f"stated status {promo['status']!r} but the gates give {exp_status!r}")
        if promo["status"] == "rejected":
            ok(True, "REJECTED_NOT_RECOMMENDED", "")

    # ---- no leakage from later episodes ---------------------------------------------------------------
    last_ep = max(eps)
    leaks = sorted({int(m.group(1)) for blob in _text_blobs(promo) for m in _RX_EP_MENTION.finditer(blob or "") if int(m.group(1)) > last_ep})
    ok(not leaks, "FUTURE_EPISODE_REFERENCE", f"narrative text mentions later episode(s) {leaks}")
    return c.res


# ====================================================================================== collection-level validation
def _overlap(a: Sequence[Tuple[int, float, float]], b: Sequence[Tuple[int, float, float]]) -> float:
    ov = sum(max(0.0, min(e1, e2) - max(s1, s2)) for ep1, s1, e1 in a for ep2, s2, e2 in b if ep1 == ep2)
    da, db = sum(e - s for _, s, e in a), sum(e - s for _, s, e in b)
    return ov / max(min(da, db), 1e-9)


def validate_collection(promos: List[dict], cfg) -> List[Issue]:
    out: List[Issue] = []
    ids = [p.get("promo_id") for p in promos]
    for pid, n in Counter(ids).items():
        if n > 1:
            out.append(Issue("error", "DUPLICATE_ID", f"promo id {pid} used {n} times", "collection"))
    spans = [(p["promo_id"], [(int(s["episode"]), float(s["start"]), float(s["end"])) for s in p["segments"]]) for p in promos]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            r = _overlap(spans[i][1], spans[j][1])
            if r >= cfg.dup_overlap:
                out.append(Issue("error", "NEAR_DUPLICATE", f"{spans[i][0]} and {spans[j][0]} overlap by {r:.0%} (limit {cfg.dup_overlap:.0%})", "collection"))
    ranks = sorted(p["rank"] for p in promos if "rank" in p)
    if ranks and ranks != list(range(1, len(ranks) + 1)):
        out.append(Issue("error", "RANKS_NOT_CONSECUTIVE", f"ranks are {ranks[:8]}...", "collection"))
    by_rank = sorted((p for p in promos if "rank" in p), key=lambda p: p["rank"])
    finals = [p["scoring"]["final"] for p in by_rank]
    if any(a < b - 1e-9 for a, b in zip(finals, finals[1:])):
        out.append(Issue("error", "RANK_ORDER", "a lower-ranked promo has a higher final score", "collection"))
    top = [p for p in promos if p.get("in_top5")]
    if len(top) > cfg.top_k:
        out.append(Issue("error", "TOP_K", f"{len(top)} promos flagged top-{cfg.top_k}", "collection"))
    per_ep = Counter(p["episodes"][0] for p in top)
    if any(n > cfg.top_max_per_episode for n in per_ep.values()):
        out.append(Issue("error", "TOP_DIVERSITY_EPISODE", f"a single episode holds more than {cfg.top_max_per_episode} of the top promos: {dict(per_ep)}", "collection"))
    if any(p.get("status") == "rejected" for p in top):
        out.append(Issue("error", "TOP_CONTAINS_REJECTED", "a rejected promo is in the top-5", "collection"))
    return out


def validate_everything(episodes: List[Episode], raw_texts: Dict[int, str], promos: List[dict], cfg) -> dict:
    """Run all three levels and return one JSON-serialisable report."""
    raw_by_ep = {n: RawIndex(t, f"episode {n}") for n, t in raw_texts.items()}
    ep_report = []
    for ep in episodes:
        issues = validate_episode(ep, raw_by_ep.get(ep.number))
        ep_report.append({"episode": ep.number, "file": ep.filename, "errors": [i.to_dict() for i in issues if i.level == "error"],
                          "warnings": [i.to_dict() for i in issues if i.level == "warning"]})
    per_promo = [validate_promo(p, raw_by_ep, cfg) for p in promos]
    coll = validate_collection(promos, cfg)
    err_codes: Counter = Counter()
    warn_codes: Counter = Counter()
    for r in per_promo:
        err_codes.update(e.code for e in r.errors)
        warn_codes.update(w.code for w in r.warnings)
    return {
        "parse_level": {"episodes_checked": len(ep_report),
                        "episodes_with_errors": sum(1 for r in ep_report if r["errors"]),
                        "episodes_with_warnings": sum(1 for r in ep_report if r["warnings"]),
                        "detail": ep_report},
        "promo_level": {"promos_checked": len(per_promo), "promos_passed": sum(1 for r in per_promo if r.passed),
                        "promos_failed": sum(1 for r in per_promo if not r.passed),
                        "checks_run": sum(r.n_checks for r in per_promo),
                        "errors_by_code": dict(err_codes), "warnings_by_code": dict(warn_codes),
                        "detail": [r.to_dict() for r in per_promo]},
        "collection_level": {"issues": [i.to_dict() for i in coll], "ok": not coll},
        "overall_ok": all(not r["errors"] for r in ep_report) and all(r.passed for r in per_promo) and not coll,
    }
