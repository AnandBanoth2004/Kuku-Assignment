"""Deterministic screenplay parser (Layer 1 - no LLM involved).

Source format (one .txt per episode)::

    EPISODE NUMBER: Episode 12
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL - DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    Actions and dialogue:
    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    <one line of text>
    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"
    --- SCRIPT END ---

Tolerances (all found or anticipated while profiling the 50 supplied files):
* label keywords are matched case-insensitively (episode 6 writes ``SCENE HEADING:`` in capitals);
* ``(O.S.)`` / ``(V.O.)`` may sit in their own parenthesis or inside the delivery list;
* a missing scene heading / start / end is recovered (from an INT./EXT. line or from the element
  times) and *flagged* - values are never invented;
* elements that overrun their scene, have start >= end, no timestamps, or are exact duplicates are
  flagged / dropped and reported as warnings;
* any line the parser does not understand is reported, never silently ignored.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .models import Element, Episode, Scene
from .timeutil import EPS, TS_PATTERN, fmt_ts, parse_ts

log = logging.getLogger(__name__)


class ScreenplayParseError(Exception):
    """Raised when a file cannot be turned into an Episode at all (reason is user-presentable)."""


_I = re.I
RE_EP_NUM = re.compile(r"^\s*EPISODE\s+(?:NUMBER|NO\.?|#)\s*[:\-]\s*(?:EPISODE\s*)?#?\s*(\d+)\b", _I)
RE_LOGLINE = re.compile(r"^\s*EPISODE\s+LOG\s*-?\s*LINE\s*[:\-]\s*(.*\S)\s*$", _I)
RE_CONT = re.compile(r"^\s*CONTINUITY(?:\s+NOTES?)?\s*[:\-]\s*(.*\S)\s*$", _I)
RE_SCRIPT_START = re.compile(r"^\s*-{2,}\s*SCRIPT\s+START\s*-{2,}\s*$", _I)
RE_SCRIPT_END = re.compile(r"^\s*-{2,}\s*SCRIPT\s+END\s*-{2,}\s*$", _I)
RE_SCENE = re.compile(r"^\s*SCENE\s+(\d+)\b\s*[:.\-–—]?\s*(.*)$", _I)
RE_HEADING = re.compile(r"^\s*SCENE\s+HEADING\s*[:\-]\s*(.*\S)\s*$", _I)
RE_SCENE_START = re.compile(r"^\s*SCENE\s+START\s*[:\-]\s*(" + TS_PATTERN + r")\s*$", _I)
RE_SCENE_END = re.compile(r"^\s*SCENE\s+END\s*[:\-]\s*(" + TS_PATTERN + r")\s*$", _I)
RE_AAD = re.compile(r"^\s*ACTIONS?\s*(?:AND|&)\s*DIALOGUE\s*:?\s*$", _I)
RE_SLUG = re.compile(r"^\s*(?:INT|EXT|INT\./EXT|INT/EXT|I/E)\b[.\s/]", _I)
RE_BLOCK = re.compile(r"^\s*\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")
RE_TIMES = re.compile(
    r"^\s*START\s*[:=]\s*\(?\s*(" + TS_PATTERN + r")\s*\)?\s*[|,;]\s*END\s*[:=]\s*\(?\s*(" + TS_PATTERN + r")\s*\)?\s*$",
    _I,
)
RE_PAREN = re.compile(r"\(([^()]*)\)")
RE_VOICE = re.compile(r"^(?:O\.?\s?S\.?|V\.?\s?O\.?|CONT'?D\.?|OFF[- ]?SCREEN|OFFSCREEN)$", _I)

_KIND_BY_LABEL = {
    "CAMERA": "camera",
    "SHOT": "camera",
    "ACTION": "action",
    "ACTIONS": "action",
    "SOUND FX": "sfx",
    "SOUND": "sfx",
    "SFX": "sfx",
    "SOUND EFFECT": "sfx",
    "SOUND EFFECTS": "sfx",
}

_QUOTES = "\"“”„'‘’"


def _warn(ep: Episode, code: str, message: str, line_no: int = 0, severity: str = "warning") -> None:
    ep.warnings.append({"code": code, "severity": severity, "message": message, "line_no": line_no})


def _is_structural(line: str) -> bool:
    return bool(
        RE_BLOCK.match(line) or RE_SCENE.match(line) or RE_SCRIPT_START.match(line) or RE_SCRIPT_END.match(line)
        or RE_HEADING.match(line) or RE_SCENE_START.match(line) or RE_SCENE_END.match(line)
        or RE_AAD.match(line)
    )


def split_label(label: str) -> tuple:
    """``'THE BEARDED STRANGER (O.S.) (Deep, Resonant)'`` -> (kind, speaker, delivery[], voice)."""
    base = RE_PAREN.sub("", label).strip()
    base = re.sub(r"\s+", " ", base).strip(" :-")
    key = base.upper()
    kind = _KIND_BY_LABEL.get(key)
    if kind:
        return kind, None, [], None
    delivery, voice = [], None
    for inner in RE_PAREN.findall(label):
        for tok in (t.strip() for t in inner.split(",")):
            if not tok:
                continue
            if RE_VOICE.match(tok):
                squashed = re.sub(r"[\s.'\-]", "", tok.upper())
                if squashed in ("OS", "OFFSCREEN"):
                    voice = "O.S."
                elif squashed == "VO":
                    voice = "V.O."
                # CONT'D carries no information for us and is ignored
            else:
                delivery.append(tok)
    return "dialogue", key, delivery, voice


def _strip_quotes(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] in _QUOTES and t[-1] in _QUOTES:
        t = t[1:-1].strip()
    elif t and t[0] in "\"“„":
        t = t[1:].strip()
    return t


def parse_screenplay_text(text: str, filename: str = "<memory>", expected_number: Optional[int] = None) -> Episode:
    """Parse one episode. Reusable for NEW screenplays (same layout, minor deviations tolerated)."""
    if text.startswith("﻿"):
        text = text[1:]
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    ep = Episode(number=-1, filename=filename)
    header_number: Optional[int] = None
    saw_start = saw_end = False
    cur: Optional[Scene] = None
    scene_meta_seen: dict = {}
    n = len(lines)
    i = 0
    while i < n:
        raw = lines[i]
        ln_no = i + 1
        if not raw.strip():
            i += 1
            continue

        m = RE_EP_NUM.match(raw)
        if m and cur is None:
            header_number = int(m.group(1)); i += 1; continue
        m = RE_LOGLINE.match(raw)
        if m and cur is None:
            ep.logline = m.group(1).strip(); i += 1; continue
        m = RE_CONT.match(raw)
        if m and cur is None:
            ep.continuity = m.group(1).strip(); i += 1; continue
        if RE_SCRIPT_START.match(raw):
            saw_start = True; i += 1; continue
        if RE_SCRIPT_END.match(raw):
            saw_end = True; i += 1; continue

        m = RE_SCENE.match(raw)
        if m:
            cur = Scene(number=int(m.group(1)), heading=None, start=None, end=None, line_no=ln_no)
            ep.scenes.append(cur)
            scene_meta_seen = {}
            inline = m.group(2).strip()
            if inline and RE_SLUG.match(inline):
                cur.heading = inline
            i += 1
            continue

        if cur is not None:
            m = RE_HEADING.match(raw)
            if m:
                cur.heading = m.group(1).strip(); i += 1; continue
            m = RE_SCENE_START.match(raw)
            if m:
                cur.start = parse_ts(m.group(1)); scene_meta_seen["start"] = True; i += 1; continue
            m = RE_SCENE_END.match(raw)
            if m:
                cur.end = parse_ts(m.group(1)); scene_meta_seen["end"] = True; i += 1; continue
            if RE_AAD.match(raw):
                i += 1; continue
            if cur.heading is None and not cur.elements and RE_SLUG.match(raw):
                cur.heading = raw.strip(); i += 1; continue

        b = RE_BLOCK.match(raw)
        if b:
            label = b.group(1).strip()
            if cur is None:
                cur = Scene(number=1, heading=None, start=None, end=None, line_no=ln_no)
                ep.scenes.append(cur)
                _warn(ep, "IMPLICIT_SCENE", "element block found before any 'SCENE nn' line; created scene 1", ln_no)
            kind, speaker, delivery, voice = split_label(label)
            j = i + 1
            while j < n and not lines[j].strip() and j - i <= 2:
                j += 1
            tm = RE_TIMES.match(lines[j]) if j < n else None
            start = end = None
            if tm:
                try:
                    start, end = parse_ts(tm.group(1)), parse_ts(tm.group(2))
                except ValueError as exc:  # pragma: no cover - guarded by regex
                    _warn(ep, "BAD_TIMESTAMP", str(exc), ln_no)
                j += 1
            else:
                _warn(ep, "BLOCK_NO_TIMES", f"block '{label}' has no 'Start: (..) | End: (..)' line; element kept untimed and excluded from promos", ln_no)
                j = i + 1
            # Skip any blank lines between the timestamp and the body text.
            # This tolerates CRLF files written through Python's text-mode on
            # Windows where \r\n becomes \r\r\n on disk and the extra \r later
            # normalises to an extra blank line after the timestamp.
            while j < n and not lines[j].strip():
                j += 1
            body = []
            while j < n and lines[j].strip() and not _is_structural(lines[j]):
                body.append(lines[j].strip())
                j += 1
            body_text = " ".join(body)
            if not body_text:
                _warn(ep, "EMPTY_BODY", f"block '{label}' has no text", ln_no, "info")
            if kind == "dialogue":
                body_text = _strip_quotes(body_text)
            el = Element(id="", episode=-1, scene=cur.number, kind=kind, start=start, end=end, text=body_text,
                         speaker=speaker, speaker_raw=label if kind == "dialogue" else None,
                         delivery=delivery, voice=voice, line_no=ln_no)
            if start is not None and end is not None and not end > start:
                el.flags.append("bad_times")
                _warn(ep, "ELEMENT_BAD_TIMES", f"{kind} element start {fmt_ts(start)} >= end {fmt_ts(end)}; excluded from promo cut points", ln_no)
            cur.elements.append(el)
            i = max(j, i + 1)
            continue

        _warn(ep, "UNPARSED_LINE", f"line not recognised: {raw.strip()[:100]!r}", ln_no, "info")
        i += 1

    _finalise(ep, header_number, expected_number, saw_start, saw_end)
    return ep


def _finalise(ep: Episode, header_number, expected_number, saw_start: bool, saw_end: bool) -> None:
    # --- episode number: header is authoritative, file name is the fallback -----------------------
    if header_number is not None:
        ep.number, ep.number_source = header_number, "header"
        if expected_number is not None and expected_number != header_number:
            _warn(ep, "EPISODE_NUMBER_MISMATCH",
                  f"header says episode {header_number} but file name says {expected_number}; header kept", 0)
    elif expected_number is not None:
        ep.number, ep.number_source = expected_number, "filename"
        _warn(ep, "MISSING_EPISODE_NUMBER", "no 'EPISODE NUMBER:' header; number taken from the file name", 0)
    else:
        raise ScreenplayParseError("no episode number in the header and none derivable from the file name")
    if not ep.logline:
        _warn(ep, "MISSING_LOGLINE", "no episode logline", 0, "info")
    if not ep.continuity:
        _warn(ep, "MISSING_CONTINUITY", "no continuity notes", 0, "info")
    if not (saw_start and saw_end):
        _warn(ep, "MISSING_SCRIPT_MARKERS", "'--- SCRIPT START/END ---' marker missing", 0, "info")
    if not ep.scenes:
        raise ScreenplayParseError("no scenes found (no 'SCENE nn' lines or element blocks)")

    # --- assign ids, drop exact duplicates --------------------------------------------------------
    for sc in ep.scenes:
        seen, kept = set(), []
        for el in sc.elements:
            key = (el.kind, el.speaker_raw, el.start, el.end, el.text)
            if key in seen:
                _warn(ep, "DUP_ELEMENT", f"duplicate {el.kind} element at {el.start}-{el.end} dropped", el.line_no)
                continue
            seen.add(key)
            kept.append(el)
        sc.elements = kept
        for idx, el in enumerate(sc.elements, 1):
            el.episode = ep.number
            el.id = f"E{ep.number:03d}-S{sc.number:02d}-{idx:03d}"

    # --- recover missing scene times from elements (flagged, never invented) ----------------------
    for sc in ep.scenes:
        timed = [e for e in sc.elements if e.timed]
        if sc.start is None or sc.end is None:
            if timed:
                if sc.start is None:
                    sc.start = min(e.start for e in timed)
                if sc.end is None:
                    sc.end = max(e.end for e in timed)
                sc.times_inferred = True
                _warn(ep, "SCENE_TIMES_INFERRED", f"scene {sc.number}: missing scene start/end; derived from element times", sc.line_no)
            else:
                _warn(ep, "SCENE_NO_TIMES", f"scene {sc.number}: no scene times and no timed elements", sc.line_no, "error")
        if sc.heading is None:
            _warn(ep, "MISSING_SCENE_HEADING", f"scene {sc.number}: no heading", sc.line_no, "info")

    # --- bounds / contiguity checks ----------------------------------------------------------------
    prev_end = 0.0
    for sc in ep.scenes:
        if sc.start is None or sc.end is None:
            continue
        if not sc.end > sc.start:
            _warn(ep, "SCENE_BAD_TIMES", f"scene {sc.number}: end {fmt_ts(sc.end)} <= start {fmt_ts(sc.start)}", sc.line_no, "error")
        if abs(sc.start - prev_end) > 0.05 and not (sc is ep.scenes[0] and abs(sc.start) < EPS):
            _warn(ep, "SCENE_GAP_OR_OVERLAP", f"scene {sc.number} starts at {fmt_ts(sc.start)} but previous scene ended at {fmt_ts(prev_end)}", sc.line_no)
        prev_end = sc.end
        for el in sc.elements:
            if el.timed and (el.start < sc.start - EPS or el.end > sc.end + EPS):
                el.flags.append("out_of_scene_bounds")
                _warn(ep, "ELEMENT_OUT_OF_SCENE",
                      f"{el.id} ({el.kind}) runs {fmt_ts(el.start)}-{fmt_ts(el.end)} outside scene {sc.number} "
                      f"[{fmt_ts(sc.start)}-{fmt_ts(sc.end)}]; its end is not used as a cut point", el.line_no)
    ends = [sc.end for sc in ep.scenes if sc.end is not None]
    ep.duration = round(max(ends), 2) if ends else 0.0
