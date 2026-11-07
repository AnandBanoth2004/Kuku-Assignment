"""Timestamp helpers.

Internal unit everywhere in the project: float seconds rounded to 0.01 s
(the precision of the source screenplays, e.g. ``00:04.80``).
"""
from __future__ import annotations

import re

# MM:SS[.ff]  or  H:MM:SS[.ff]   (comma accepted as the decimal separator)
TS_PATTERN = r"(?:\d{1,2}:)?\d{1,3}:\d{2}(?:[.,]\d{1,3})?"
_TS_RE = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,3}):(\d{2})(?:[.,](\d{1,3}))?\s*$")

EPS = 0.011  # tolerance used when comparing two timestamps (source precision is 0.01)


def parse_ts(text: str) -> float:
    """Parse a screenplay timestamp into seconds. Raises ValueError if malformed."""
    m = _TS_RE.match(text or "")
    if not m:
        raise ValueError(f"unparseable timestamp: {text!r}")
    hours, minutes, seconds, frac = m.groups()
    secs = int(minutes) * 60 + int(seconds)
    if hours:
        secs += int(hours) * 3600
    if int(seconds) >= 60:
        raise ValueError(f"seconds field out of range in timestamp: {text!r}")
    value = secs + (float("0." + frac) if frac else 0.0)
    return round(value, 2)


def fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS.ff (the screenplay style)."""
    cs = int(round(float(seconds) * 100))
    minutes, rem = divmod(cs, 6000)
    secs, hundredths = divmod(rem, 100)
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"


def close(a: float, b: float, tol: float = EPS) -> bool:
    return abs(a - b) <= tol
