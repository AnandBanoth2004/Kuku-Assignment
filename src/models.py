"""Plain dataclasses shared by every layer (JSON-serialisable via ``to_dict``)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

KINDS = ("camera", "action", "dialogue", "sfx")


@dataclass
class Element:
    """One timed screenplay element (a camera shot, an action, a spoken line or a sound effect)."""

    id: str                       # e.g. "E003-S03-042" -> episode 3, scene 3, 42nd element of the scene
    episode: int
    scene: int
    kind: str                     # camera | action | dialogue | sfx
    start: Optional[float]        # seconds on the episode timeline (None if the source had no valid time)
    end: Optional[float]
    text: str                     # for dialogue: the spoken words without the surrounding quotes
    speaker: Optional[str] = None       # normalised label (upper-case, parentheticals removed)
    speaker_raw: Optional[str] = None   # label exactly as written in the file
    delivery: list = field(default_factory=list)   # e.g. ["Cold", "Commanding"]
    voice: Optional[str] = None         # "O.S." / "V.O." / None
    line_no: int = 0                    # 1-based line of the label in the source file (traceability)
    flags: list = field(default_factory=list)      # e.g. ["out_of_scene_bounds"]

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None and self.end > self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    number: int
    heading: Optional[str]
    start: Optional[float]
    end: Optional[float]
    elements: list = field(default_factory=list)
    times_inferred: bool = False  # True if start/end were derived from elements (never invented)
    line_no: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Episode:
    number: int
    filename: str
    logline: str = ""
    continuity: str = ""
    scenes: list = field(default_factory=list)
    duration: float = 0.0
    warnings: list = field(default_factory=list)  # list of dicts: {code, severity, message, line_no}
    number_source: str = "header"

    def elements(self) -> Iterator[Element]:
        for sc in self.scenes:
            yield from sc.elements

    def element_map(self) -> dict:
        return {e.id: e for e in self.elements()}

    def scene_by_number(self, number: int) -> Optional[Scene]:
        for sc in self.scenes:
            if sc.number == number:
                return sc
        return None

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "filename": self.filename,
            "logline": self.logline,
            "continuity": self.continuity,
            "duration": self.duration,
            "number_source": self.number_source,
            "warnings": self.warnings,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @staticmethod
    def from_dict(d: dict) -> "Episode":
        scenes = []
        for s in d.get("scenes", []):
            els = [Element(**e) for e in s.get("elements", [])]
            scenes.append(Scene(number=s["number"], heading=s.get("heading"), start=s.get("start"),
                                end=s.get("end"), elements=els, times_inferred=s.get("times_inferred", False),
                                line_no=s.get("line_no", 0)))
        return Episode(number=d["number"], filename=d.get("filename", ""), logline=d.get("logline", ""),
                       continuity=d.get("continuity", ""), scenes=scenes, duration=d.get("duration", 0.0),
                       warnings=d.get("warnings", []), number_source=d.get("number_source", "header"))
