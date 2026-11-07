"""Input handling: safe ZIP extraction, file discovery, per-file isolation of failures, summary.

``load_pack`` accepts a .zip, a directory of .txt files, or a single .txt file, so the exact same
function is used for the supplied 50-episode pack and for a NEW screenplay handed over later.
"""
from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Episode
from .screenplay_parser import ScreenplayParseError, parse_screenplay_text

log = logging.getLogger(__name__)


@dataclass
class ProcessingSummary:
    files_discovered: int = 0
    episodes_processed: int = 0
    failed: list = field(default_factory=list)            # [{file, reason}]
    total_scenes: int = 0
    dialogue_entries: int = 0
    action_entries: int = 0
    camera_entries: int = 0
    sfx_entries: int = 0
    timestamped_elements: int = 0
    timestamped_scenes: int = 0
    warnings_by_code: dict = field(default_factory=dict)
    episode_numbers: list = field(default_factory=list)
    encodings_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def safe_extract_zip(zip_path: Path, dest: Path) -> List[Path]:
    """Extract only regular .txt members, refusing absolute paths and '..' traversal (zip-slip)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith("."):
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise ValueError(f"unsafe path in archive refused: {name!r}")
            if not name.lower().endswith(".txt"):
                log.info("skipping non-text archive member %s", name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)
    return out


def _natural_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 10**9, path.name)


def discover_files(root: Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    files = [p for p in root.rglob("*.txt") if "__MACOSX" not in str(p) and not p.name.startswith(".")]
    return sorted(files, key=_natural_key)


def read_text_any_encoding(path: Path) -> Tuple[str, str]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ScreenplayParseError("file is not decodable as utf-8, cp1252 or latin-1")


def _number_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"episode[-_ ]?0*(\d+)", path.name, re.I)
    return int(m.group(1)) if m else None


def process_screenplay_file(path: Path) -> Episode:
    text, enc = read_text_any_encoding(path)
    ep = parse_screenplay_text(text, filename=Path(path).name, expected_number=_number_from_filename(Path(path)))
    if enc != "utf-8-sig":
        ep.warnings.append({"code": "ENCODING_FALLBACK", "severity": "warning",
                            "message": f"decoded as {enc}, not UTF-8", "line_no": 0})
    return ep


def process_screenplay_text(text: str, name: str = "pasted_screenplay.txt", episode_number: Optional[int] = None) -> Episode:
    """Entry point for a NEW screenplay supplied as text (used by the app and the interview workflow)."""
    return parse_screenplay_text(text, filename=name, expected_number=episode_number)


def load_pack(input_path: Path, extract_dir: Optional[Path] = None, raw_out: Optional[dict] = None) -> Tuple[List[Episode], ProcessingSummary]:
    """Load every screenplay found at ``input_path`` (zip | directory | single file).

    One bad file never stops the run: it is recorded in ``summary.failed`` with its exact reason.
    If ``raw_out`` is a dict it is filled with {episode_number: raw_text} for the independent QC scanner.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if input_path.suffix.lower() == ".zip":
        if extract_dir is None:
            raise ValueError("extract_dir is required when input is a .zip")
        files = safe_extract_zip(input_path, Path(extract_dir))
        files = sorted(files, key=_natural_key)
    else:
        files = discover_files(input_path)

    summary = ProcessingSummary(files_discovered=len(files))
    episodes: List[Episode] = []
    seen_numbers: dict = {}
    for f in files:
        try:
            ep = process_screenplay_file(f)
        except ScreenplayParseError as exc:
            summary.failed.append({"file": Path(f).name, "reason": str(exc)})
            log.error("FAILED %s: %s", Path(f).name, exc)
            continue
        except Exception as exc:  # defensive: report, never crash the batch
            summary.failed.append({"file": Path(f).name, "reason": f"{type(exc).__name__}: {exc}"})
            log.exception("unexpected failure parsing %s", f)
            continue
        if ep.number in seen_numbers:
            summary.failed.append({"file": Path(f).name,
                                   "reason": f"duplicate episode number {ep.number} (already loaded from {seen_numbers[ep.number]})"})
            continue
        seen_numbers[ep.number] = Path(f).name
        episodes.append(ep)
        if raw_out is not None:
            raw_out[ep.number] = read_text_any_encoding(f)[0]

    episodes.sort(key=lambda e: e.number)
    codes: Counter = Counter()
    for ep in episodes:
        summary.episodes_processed += 1
        summary.episode_numbers.append(ep.number)
        summary.total_scenes += len(ep.scenes)
        for sc in ep.scenes:
            if sc.start is not None and sc.end is not None:
                summary.timestamped_scenes += 1
            for el in sc.elements:
                if el.kind == "dialogue":
                    summary.dialogue_entries += 1
                elif el.kind == "action":
                    summary.action_entries += 1
                elif el.kind == "camera":
                    summary.camera_entries += 1
                elif el.kind == "sfx":
                    summary.sfx_entries += 1
                if el.timed:
                    summary.timestamped_elements += 1
        for w in ep.warnings:
            codes[w["code"]] += 1
    summary.warnings_by_code = dict(codes)
    return episodes, summary
