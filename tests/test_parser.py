"""Parser tests. They use small inline screenplays so they run without the (confidential) input pack."""
import pytest

from src.screenplay_parser import ScreenplayParseError, parse_screenplay_text, split_label
from src.timeutil import fmt_ts, parse_ts

BASE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Picks up from Episode 6.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Wide shot of the hall.

**LADY ANNA (Cold, Furious):**
Start: (00:01.00) | End: (00:03.00)
"You lied to me!"

**HOUSEKEEPER (Sharp, Stern, O.S.):**
Start: (00:05.00) | End: (00:07.00)
"Prudence."

**SOUND FX:**
Start: (00:08.00) | End: (00:09.50)
A door slams.

**ACTION:**
Start: (00:10.00) | End: (00:20.00)
Anna turns away.

--- SCRIPT END ---
"""


def test_timestamp_roundtrip():
    assert parse_ts("00:04.80") == 4.8
    assert parse_ts("02:25.75") == 145.75
    assert parse_ts("1:02:03.5") == 3723.5
    assert fmt_ts(145.75) == "02:25.75"
    assert fmt_ts(138.0) == "02:18.00"
    with pytest.raises(ValueError):
        parse_ts("00:75.00")


def test_basic_parse_counts():
    ep = parse_screenplay_text(BASE, "episode-007-x.txt", expected_number=7)
    assert ep.number == 7 and ep.number_source == "header"
    assert len(ep.scenes) == 1 and ep.duration == 20.0
    kinds = [e.kind for e in ep.elements()]
    assert kinds == ["camera", "dialogue", "dialogue", "sfx", "action"]
    d = [e for e in ep.elements() if e.kind == "dialogue"][0]
    assert d.text == "You lied to me!"          # quotes stripped
    assert d.speaker == "LADY ANNA" and d.delivery == ["Cold", "Furious"] and d.voice is None
    assert d.id == "E007-S01-002"


def test_os_inside_delivery_list_is_voice_not_delivery():
    kind, spk, delivery, voice = split_label("HOUSEKEEPER (Sharp, Stern, O.S.)")
    assert (kind, spk, delivery, voice) == ("dialogue", "HOUSEKEEPER", ["Sharp", "Stern"], "O.S.")
    assert split_label("THE BEARDED STRANGER (O.S.) (Deep, Resonant)")[3] == "O.S."
    assert split_label("MAID (Off-Screen)")[3] == "O.S."


def test_uppercase_metadata_labels_regression():
    """Episode 6 of the real pack writes SCENE HEADING / SCENE START / SCENE END in capitals.
    A case-sensitive first draft silently lost those scene times (98/100 scenes parsed)."""
    caps = (BASE.replace("Scene heading:", "SCENE HEADING:").replace("Scene start:", "SCENE START:")
            .replace("Scene end:", "SCENE END:").replace("Actions and dialogue:", "ACTIONS AND DIALOGUE:"))
    ep = parse_screenplay_text(caps, "x.txt", 7)
    sc = ep.scenes[0]
    assert sc.heading == "INT. HALL – DAY" and sc.start == 0.0 and sc.end == 20.0 and not sc.times_inferred
    assert ep.duration == 20.0


def test_missing_scene_times_are_inferred_and_flagged_not_invented():
    broken = BASE.replace("Scene start: 00:00.00\n", "").replace("Scene end: 00:20.00\n", "")
    ep = parse_screenplay_text(broken, "x.txt", 7)
    sc = ep.scenes[0]
    assert sc.times_inferred is True
    assert (sc.start, sc.end) == (0.0, 20.0)     # derived from the elements, not guessed
    assert any(w["code"] == "SCENE_TIMES_INFERRED" for w in ep.warnings)


def test_element_outside_scene_is_flagged():
    bad = BASE.replace("Start: (00:08.00) | End: (00:09.50)", "Start: (00:19.00) | End: (00:21.00)")
    ep = parse_screenplay_text(bad, "x.txt", 7)
    sfx = [e for e in ep.elements() if e.kind == "sfx"][0]
    assert "out_of_scene_bounds" in sfx.flags
    assert any(w["code"] == "ELEMENT_OUT_OF_SCENE" for w in ep.warnings)


def test_block_without_times_kept_untimed_and_reported():
    bad = BASE.replace("Start: (00:08.00) | End: (00:09.50)\n", "")
    ep = parse_screenplay_text(bad, "x.txt", 7)
    assert any(w["code"] == "BLOCK_NO_TIMES" for w in ep.warnings)
    sfx = [e for e in ep.elements() if e.kind == "sfx"][0]
    assert not sfx.timed


def test_duplicate_elements_dropped():
    dup_block = "**ACTION:**\nStart: (00:10.00) | End: (00:20.00)\nAnna turns away.\n\n"
    ep = parse_screenplay_text(BASE.replace("--- SCRIPT END ---", dup_block + "--- SCRIPT END ---"), "x.txt", 7)
    assert sum(1 for e in ep.elements() if e.kind == "action") == 1
    assert any(w["code"] == "DUP_ELEMENT" for w in ep.warnings)


def test_episode_number_falls_back_to_filename_and_mismatch_is_reported():
    no_hdr = BASE.replace("EPISODE NUMBER: Episode 7\n", "")
    ep = parse_screenplay_text(no_hdr, "episode-007-x.txt", expected_number=7)
    assert ep.number == 7 and ep.number_source == "filename"
    ep2 = parse_screenplay_text(BASE, "episode-009-x.txt", expected_number=9)
    assert ep2.number == 7 and any(w["code"] == "EPISODE_NUMBER_MISMATCH" for w in ep2.warnings)


def test_unrecoverable_files_raise_with_reason():
    with pytest.raises(ScreenplayParseError, match="no episode number"):
        parse_screenplay_text(BASE.replace("EPISODE NUMBER: Episode 7\n", ""), "mystery.txt", None)
    with pytest.raises(ScreenplayParseError, match="no scenes"):
        parse_screenplay_text("EPISODE NUMBER: Episode 1\nhello\n", "x.txt", 1)


def test_crlf_and_bom_tolerated():
    ep = parse_screenplay_text("﻿" + BASE.replace("\n", "\r\n"), "x.txt", 7)
    assert ep.number == 7 and len(list(ep.elements())) == 5
