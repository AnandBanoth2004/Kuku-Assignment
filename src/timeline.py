"""Per-episode index used by hook / cliffhanger / candidate logic.

Its main job is to enumerate **legal cut points**. Every promo boundary must be an exact boundary that exists
in the screenplay (a shot, line, action or scene boundary) - timestamps are never interpolated or invented -
and a cut may never fall inside a spoken line or split a sentence that continues with "...".
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .cues import ElFeat, analyze_element
from .models import Element, Episode
from .timeutil import EPS


class Timeline:
    def __init__(self, ep: Episode, registry=None):
        self.ep = ep
        self.reg = registry
        self.els: List[Element] = sorted((e for e in ep.elements() if e.timed and "bad_times" not in e.flags),
                                         key=lambda e: (e.start, e.end))
        self.feat: Dict[str, ElFeat] = {e.id: analyze_element(e) for e in self.els}
        self.dlg: List[Element] = [e for e in self.els if e.kind == "dialogue"]
        self.duration: float = ep.duration
        self.scenes = [(s.number, s.start, s.end) for s in ep.scenes if s.start is not None and s.end is not None]
        self.end_signal = registry.next_episode_ending_signal(ep.number) if registry else None
        self._by_id = {e.id: e for e in self.els}

    # ------------------------------------------------------------------ lookups
    def el(self, element_id: str) -> Element:
        return self._by_id[element_id]

    def scene_of(self, t: float) -> int:
        for n, s, e in self.scenes:
            if s - EPS <= t < e - EPS:
                return n
        return self.scenes[-1][0] if self.scenes else 1

    def scene_bounds(self, n: int):
        for sn, s, e in self.scenes:
            if sn == n:
                return s, e
        return None

    def inside_dialogue(self, t: float) -> bool:
        return any(d.start + EPS < t < d.end - EPS for d in self.dlg)

    def dialogue_between(self, S: float, E: float) -> List[Element]:
        return [d for d in self.dlg if d.start >= S - EPS and d.end <= E + EPS]

    def first_dialogue_from(self, t: float) -> Optional[Element]:
        for d in self.dlg:
            if d.start >= t - EPS:
                return d
        return None

    def last_dialogue_to(self, t: float) -> Optional[Element]:
        last = None
        for d in self.dlg:
            if d.end <= t + EPS:
                last = d
            else:
                break
        return last

    def overlapping(self, S: float, E: float, kinds=None) -> List[Element]:
        return [e for e in self.els if e.start < E - EPS and e.end > S + EPS and (kinds is None or e.kind in kinds)]

    def inside(self, S: float, E: float, kinds=None) -> List[Element]:
        """Elements that start inside [S, E) (what the viewer meets in that stretch)."""
        return [e for e in self.els if e.start >= S - EPS and e.start < E - EPS and (kinds is None or e.kind in kinds)]

    # ------------------------------------------------------------------ legal cut points
    def _valid_start(self, t: float, min_len: float) -> bool:
        if t > self.duration - min_len + EPS:
            return False
        if self.inside_dialogue(t):
            return False
        d = self.first_dialogue_from(t)
        if d is not None and d.start - t <= 3.0 + EPS and self.feat[d.id].ellipsis_start:
            return False                      # would open on the tail of a sentence whose beginning is cut off
        return True

    def start_points(self, min_len: float = 20.0) -> List[float]:
        pts = {e.start for e in self.els if e.kind in ("camera", "action", "dialogue")}
        pts |= {s for _, s, _ in self.scenes}
        return sorted(p for p in pts if self._valid_start(round(p, 2), min_len))

    def _valid_end(self, t: float, min_len: float) -> bool:
        if t < min_len - EPS or t > self.duration + EPS:
            return False
        if self.inside_dialogue(t):
            return False
        nd = self.first_dialogue_from(t)
        last = self.last_dialogue_to(t)
        if nd and last and nd.speaker == last.speaker and nd.start - t < 2.0 and self.feat[nd.id].ellipsis_start:
            return False                      # would cut a sentence in half ("I said..." | "...let her go")
        return True

    def end_points(self, min_len: float = 20.0) -> List[float]:
        pts = {e.end for e in self.els if "out_of_scene_bounds" not in e.flags}
        pts |= {e for _, _, e in self.scenes}
        return sorted(p for p in pts if self._valid_end(round(p, 2), min_len))

    # ------------------------------------------------------------------ boundary membership (used by validation too)
    def is_source_boundary(self, t: float) -> bool:
        if any(abs(t - s) <= EPS or abs(t - e) <= EPS for _, s, e in self.scenes):
            return True
        return any(abs(t - e.start) <= EPS or abs(t - e.end) <= EPS for e in self.els)
