"""Lightweight cross-episode continuity (no LLM needed).

What it maintains
* **Character registry** - canonical names, aliases *with the evidence for each alias*, first episode,
  episodes present, line counts.  The screenplays label people by description until they are named
  (``THE BEARDED STRANGER`` -> later ``ALEXANDER``; ``LADY IN GOLD GOWN`` -> ``CLARISSA``), so aliasing matters.
* **Story memory** - per-episode logline / continuity note, which earlier episode it picks up from, how the
  previous episode ended (parsed from this note), revelations and relationship hints (regex, low confidence).
* **Spoiler-safe context** - ``context_for(n)`` only ever returns material from episodes <= n.

Alias evidence (only two sources are trusted)
1. explicit hint inside the label:      ``BLOND NOBLEMAN (Rupert) (Cold)``
2. editor-supplied file ``configs/aliases.yaml`` (human-in-the-loop; optional)
Anything not resolved is listed in ``unresolved_descriptors`` for a human to fill in - never guessed.

Rejected approach (observed failure): voting by "a camera line that overlaps the spoken line names exactly one
character".  Cameras very often show the LISTENER, so it proposed the protagonist as an alias of MATRIARCH, JUDGE,
MAID ...  A same-scene clash check caught every wrong proposal and the method found no correct alias, so it was removed.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import Episode

log = logging.getLogger(__name__)

HONORIFICS = {"LADY", "LORD", "SIR", "DAME", "MADAM", "MISS", "MR", "MRS", "DR", "FATHER", "MOTHER", "DUKE", "DUCHESS"}
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")


def _title(label: str) -> str:
    return " ".join(w.capitalize() if not w.endswith("'S") else w[:-2].capitalize() + "'s" for w in label.split())


def strip_article(label: str) -> str:
    return re.sub(r"^(?:THE|A|AN)\s+", "", label.strip().upper())


@dataclass
class Character:
    canonical: str
    kind: str                       # "named" | "role"
    aliases: Dict[str, dict] = field(default_factory=dict)   # label -> {"evidence": str, "votes": int}
    first_episode: int = 10**6
    episodes: set = field(default_factory=set)
    lines: int = 0

    def to_dict(self) -> dict:
        return {"canonical": self.canonical, "kind": self.kind, "aliases": self.aliases,
                "first_episode": self.first_episode, "episodes": sorted(self.episodes), "lines": self.lines}


@dataclass
class EpisodeMemory:
    number: int
    logline: str
    continuity: str
    picks_up_from: List[int] = field(default_factory=list)
    depends_on_previous: bool = False
    previous_ending: Optional[dict] = None       # {"type": "cliffhanger", "snippet": "..."} described by THIS episode's note
    revelations: List[str] = field(default_factory=list)
    relationships: List[dict] = field(default_factory=list)
    characters_present: List[str] = field(default_factory=list)
    new_characters: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_RE_EP_REF = re.compile(r"\bEpisode\s+(\d+)\b", re.I)
_RE_PICKS_UP = re.compile(r"\b(?:picks? up|follows?|directly follows|continues?|immediately after|right after|"
                          r"cutting directly|transitions? directly)\b", re.I)
_RE_PREV_END = re.compile(r"\b(cliffhanger|climax|ending|conclusion|revelation|ordeal|confrontation)\b[^.;]*?"
                          r"(?:\bwhere\b|\bas\b|\bwith\b|\bthat\b)\s+(?P<snip>[^.;]{8,220})", re.I)
_RE_REVEAL = re.compile(r"\b(reveal\w*|discover\w*|unmask\w*|expos\w+|confession|proclaim\w*|learns?|realiz\w+|"
                        r"uncover\w*|recogni[sz]\w+|declare\w*)\b", re.I)
_RE_REL = re.compile(r"\b(?P<owner>her|his|their)\s+(?:[a-z\-]+\s+){0,2}?(?P<rel>stepmother|stepfather|sister|brother|mother|"
                     r"father|husband|wife|fianc[eé]e?|betrothed|daughter|son|cousin|bride|groom)\s+(?P<name>[A-Z][a-z]+)\b")
_RE_NAME_TOKENS = re.compile(r"\b[A-Z][a-z]{2,}\b")


class Registry:
    def __init__(self):
        self.characters: Dict[str, Character] = {}
        self.label_to_canon: Dict[str, str] = {}
        self.label_kind: Dict[str, str] = {}      # label -> 'named' | 'role' (before alias merging)
        self.unresolved: List[dict] = []
        self.name_tokens: set = set()
        self.memory: Dict[int, EpisodeMemory] = {}
        self.total_episodes = 0

    # ---- public API ---------------------------------------------------------------------------------
    def canon(self, speaker: Optional[str]) -> Optional[str]:
        return self.label_to_canon.get(speaker) if speaker else None

    def kind_of_label(self, speaker: Optional[str]) -> str:
        return self.label_kind.get(strip_article(speaker or ""), "role")

    def name_tokens_of_label(self, speaker: Optional[str]) -> List[str]:
        return [t.title() for t in re.findall(r"[A-Z]+", strip_article(speaker or "")) if t in self.name_tokens]

    def is_named(self, canonical: str) -> bool:
        c = self.characters.get(canonical)
        return bool(c and c.kind == "named")

    def context_for(self, ep_no: int, max_prior: int = 3) -> List[dict]:
        """Spoiler-safe context: only episodes <= ep_no, most recent first."""
        out = []
        for n in range(ep_no, max(0, ep_no - max_prior - 1), -1):
            m = self.memory.get(n)
            if m:
                out.append({"episode": n, "logline": m.logline, "continuity": m.continuity})
        return out

    def next_episode_ending_signal(self, ep_no: int) -> Optional[dict]:
        """How episode ``ep_no`` ended, according to episode ``ep_no+1``'s continuity note (scoring use only)."""
        nxt = self.memory.get(ep_no + 1)
        pe = nxt.previous_ending if nxt else None
        return pe if pe and pe.get("type") in ("cliffhanger", "climax") else None

    def to_dict(self) -> dict:
        return {
            "characters": {k: v.to_dict() for k, v in sorted(self.characters.items(), key=lambda kv: -kv[1].lines)},
            "label_to_canonical": self.label_to_canon,
            "unresolved_descriptors": self.unresolved,
            "episode_memory": {str(k): v.to_dict() for k, v in sorted(self.memory.items())},
        }


def _load_manual_aliases(path: Optional[Path]) -> Dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    try:
        import yaml  # PyYAML is optional
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return {strip_article(str(k)): str(v) for k, v in (data.get("aliases") or {}).items()}
    except Exception as exc:
        log.warning("could not read manual aliases %s: %s", path, exc)
        return {}


def build_registry(episodes: List[Episode], manual_aliases_path: Optional[Path] = None,
                   min_votes: int = 3, min_share: float = 0.6) -> Registry:
    reg = Registry()
    reg.total_episodes = len(episodes)
    labels = Counter()
    first_ep: Dict[str, int] = {}
    eps_by_label: Dict[str, set] = defaultdict(set)
    scenes_by_label: Dict[str, set] = defaultdict(set)
    for ep in episodes:
        for el in ep.elements():
            if el.kind == "dialogue" and el.speaker:
                lab = strip_article(el.speaker)
                labels[lab] += 1
                first_ep[lab] = min(first_ep.get(lab, 10**6), ep.number)
                eps_by_label[lab].add(ep.number)
                scenes_by_label[lab].add((ep.number, el.scene))

    # --- name lexicon: single tokens that look like proper names ------------------------------------
    cap_mid, lower = Counter(), Counter()
    corpus = []
    for ep in episodes:
        corpus.append(ep.logline); corpus.append(ep.continuity)
        corpus.extend(el.text for el in ep.elements() if el.kind != "dialogue")
        corpus.extend(el.text for el in ep.elements() if el.kind == "dialogue")
    for txt in corpus:
        for m in _WORD.finditer(txt or ""):
            tok = m.group(0).rstrip("'’s") if m.group(0).lower().endswith(("'s", "’s")) else m.group(0)
            if len(tok) < 3:
                continue
            at_start = m.start() == 0 or re.search(r"[.!?\"“]\s*$", txt[:m.start()]) is not None
            if tok[0].isupper() and not at_start:
                cap_mid[tok.lower()] += 1
            elif tok[0].islower():
                lower[tok.lower()] += 1
    label_tokens = set()
    for lab in labels:
        parts = lab.split()
        if len(parts) == 1:
            label_tokens.add(parts[0])
        elif len(parts) == 2 and parts[0] in HONORIFICS:
            label_tokens.add(parts[1])
    for tok in label_tokens:
        t = tok.lower()
        if cap_mid[t] >= 2 and lower[t] == 0 and tok not in HONORIFICS:
            reg.name_tokens.add(tok)

    # --- canonical key for each label -----------------------------------------------------------------
    bare_name_labels = {lab for lab in labels if lab in reg.name_tokens}
    def canonical_of(lab: str) -> tuple:
        if lab in bare_name_labels:
            return _title(lab), "named"
        parts = lab.split()
        if len(parts) == 2 and parts[0] in HONORIFICS and parts[1] in bare_name_labels:
            return _title(parts[1]), "named"      # LADY JULIANA -> Juliana (the bare name exists as its own label)
        if len(parts) == 2 and parts[0] in HONORIFICS and parts[1] in reg.name_tokens:
            return _title(lab), "named"           # LORD ASHCROFT stays its own person
        return _title(lab), "role"

    for lab, cnt in labels.items():
        canon, kind = canonical_of(lab)
        reg.label_to_canon[lab] = canon
        reg.label_kind[lab] = kind
        ch = reg.characters.setdefault(canon, Character(canonical=canon, kind=kind))
        ch.lines += cnt
        ch.episodes |= eps_by_label[lab]
        ch.first_episode = min(ch.first_episode, first_ep[lab])
        if _title(lab) != canon:
            ch.aliases[lab] = {"evidence": "honorific + bare name both occur as speaker labels", "votes": cnt}
    # speaker labels arrive in two flavours; the registry also answers for the raw (un-stripped) label
    for ep in episodes:
        for el in ep.elements():
            if el.kind == "dialogue" and el.speaker and el.speaker not in reg.label_to_canon:
                reg.label_to_canon[el.speaker] = reg.label_to_canon.get(strip_article(el.speaker), _title(el.speaker))

    # --- alias evidence for descriptor ("role") labels: explicit hints + editor file only ----------------
    named_canon = {c for c, ch in reg.characters.items() if ch.kind == "named"}
    explicit: Dict[str, Counter] = defaultdict(Counter)
    for ep in episodes:
        for el in ep.elements():
            if el.kind != "dialogue" or not el.speaker:
                continue
            lab = strip_article(el.speaker)
            if reg.characters[reg.label_to_canon[lab]].kind != "role":
                continue
            for tok in el.delivery:
                for c in named_canon:
                    if tok.strip().lower() == c.lower():
                        explicit[lab][c] += 1

    manual = _load_manual_aliases(manual_aliases_path)
    for lab in [l for l in labels if reg.characters[reg.label_to_canon[l]].kind == "role"]:
        chosen, evid, n = None, "", 0
        if lab in manual:
            chosen, evid, n = manual[lab], "editor-supplied alias file", labels[lab]
        elif explicit[lab]:
            chosen, n = explicit[lab].most_common(1)[0]
            evid = f"explicit name inside the speaker label ({n} line(s))"
        if chosen:
            target = _title(chosen)
            old = reg.label_to_canon[lab]
            reg.label_to_canon[lab] = target
            existing = reg.characters.get(target)
            tch = existing or Character(canonical=target, kind="named" if target.split()[0].upper() in reg.name_tokens else "role")
            reg.characters[target] = tch
            src = reg.characters.pop(old, None) if old != target else None
            if src:
                tch.lines += src.lines; tch.episodes |= src.episodes
                tch.first_episode = min(tch.first_episode, src.first_episode)
                tch.aliases.update(src.aliases)
            tch.aliases[lab] = {"evidence": evid, "votes": n}
        else:
            reg.unresolved.append({"label": lab, "lines": labels[lab], "episodes": sorted(eps_by_label[lab]),
                                   "reason": "descriptor label with no explicit name hint and no entry in the alias file"})
    # raw-label lookups (labels with O.S. etc. already stripped by the parser) now point to the final canon
    for ep in episodes:
        for el in ep.elements():
            if el.kind == "dialogue" and el.speaker:
                reg.label_to_canon[el.speaker] = reg.label_to_canon.get(strip_article(el.speaker), _title(el.speaker))

    _build_memory(reg, episodes)
    return reg


def _build_memory(reg: Registry, episodes: List[Episode]) -> None:
    seen_chars: set = set()
    for ep in sorted(episodes, key=lambda e: e.number):
        m = EpisodeMemory(number=ep.number, logline=ep.logline, continuity=ep.continuity)
        refs = [int(x) for x in _RE_EP_REF.findall(ep.continuity)]
        m.picks_up_from = sorted({r for r in refs if r < ep.number})
        m.depends_on_previous = bool(_RE_PICKS_UP.search(ep.continuity)) and bool(m.picks_up_from or ep.number > 1)
        pe = _RE_PREV_END.search(ep.continuity)
        if pe:
            m.previous_ending = {"type": pe.group(1).lower(), "snippet": pe.group("snip").strip(), "described_by_episode": ep.number}
        for sentence in re.split(r"(?<=[.!?])\s+", ep.logline):
            if _RE_REVEAL.search(sentence):
                m.revelations.append(sentence.strip())
        for text in (ep.logline, ep.continuity):
            for r in _RE_REL.finditer(text):
                m.relationships.append({"relation": r.group("rel").lower(), "person": r.group("name"),
                                        "owner_pronoun": r.group("owner").lower(), "confidence": "low",
                                        "evidence": text[max(0, r.start() - 30): r.end() + 10].strip()})
        present = []
        for el in ep.elements():
            c = reg.canon(el.speaker) if el.kind == "dialogue" else None
            if c and c not in present:
                present.append(c)
        m.characters_present = present
        m.new_characters = [c for c in present if c not in seen_chars]
        seen_chars.update(present)
        reg.memory[ep.number] = m


def spoiler_risk(reg: Registry, ep_no: int) -> str:
    """low | medium | high - flag for the editor, NOT part of the ranking score."""
    m = reg.memory.get(ep_no)
    late = ep_no > 0.7 * max(reg.total_episodes, 1)
    big_reveal = bool(m and m.revelations)
    if late and big_reveal:
        return "high"
    if late or big_reveal:
        return "medium"
    return "low"
