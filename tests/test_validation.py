"""The validator must actually CATCH broken promos: take a valid record, break it one way at a time, expect the specific error code."""
import copy
import tempfile
from pathlib import Path

import pytest

from config import load_config
from run_pipeline import run
from src.screenplay_parser import parse_screenplay_text
from src.validation import RawIndex, validate_collection, validate_episode, validate_promo
from tests.fixtures import build_text


@pytest.fixture(scope="module")
def ctx():
    d = Path(tempfile.mkdtemp())
    text = build_text(7)
    (d / "s.txt").write_text(text, encoding="utf-8")
    cfg = load_config()
    res = run(d / "s.txt", d / "out", cfg, extract_dir=d / "ex")
    assert res["ranked"], "fixture must yield at least one promo"
    return {"rec": res["ranked"][0], "raw": {7: RawIndex(text)}, "cfg": cfg, "text": text}


def codes(ctx, rec):
    return {e.code for e in validate_promo(rec, ctx["raw"], ctx["cfg"]).errors}


def test_baseline_valid(ctx):
    r = validate_promo(ctx["rec"], ctx["raw"], ctx["cfg"])
    assert r.passed and r.n_checks > 20, [e.to_dict() for e in r.errors]


def mutated(ctx, fn):
    rec = copy.deepcopy(ctx["rec"])
    fn(rec)
    return codes(ctx, rec)


def test_start_not_a_source_boundary(ctx):
    def f(r): r["segments"][0]["start"] += 0.37
    assert "START_NOT_A_SOURCE_BOUNDARY" in mutated(ctx, f)


def test_cut_inside_spoken_line(ctx):
    def f(r):
        d = r["segments"][0]["selected_dialogue"][-1]
        r["segments"][0]["end"] = round((d["start"] + d["end"]) / 2, 2)
    got = mutated(ctx, f)
    assert "CUT_INSIDE_SPEECH_END" in got and "END_NOT_A_SOURCE_BOUNDARY" in got


def test_altered_quote(ctx):
    def f(r): r["segments"][0]["selected_dialogue"][0]["text"] += " and a secret the writers never wrote"
    assert "DIALOGUE_NOT_VERBATIM" in mutated(ctx, f)


def test_silently_dropped_line(ctx):
    def f(r): r["segments"][0]["selected_dialogue"].pop(1)
    assert "DIALOGUE_OMITTED" in mutated(ctx, f)


def test_total_duration_checks(ctx):
    def f(r): r["total_duration"] += 5
    assert "TOTAL_DURATION_SUM" in mutated(ctx, f)

    def g(r): r["segments"][0]["end"] = r["segments"][0]["start"] + 12.0
    assert "TOTAL_DURATION_RANGE" in mutated(ctx, g)


def test_hook_outside_first_three_seconds(ctx):
    def f(r):
        late = r["segments"][0]["selected_dialogue"][-1]
        r["hook"].update(text=late["text"], element_id=late["element_id"], timestamp=late["start"])
    assert "HOOK_NOT_IN_FIRST_3S" in mutated(ctx, f)


def test_tampered_scores(ctx):
    def f(r): r["scoring"]["final"] = 99.0
    assert "SCORE_FINAL" in mutated(ctx, f)

    def g(r):
        r["scoring"]["components"]["hook"] = 12.0
    assert "SCORE_RANGE" in mutated(ctx, g)

    def h(r):
        r["scoring"]["components"]["cliff"] = 1.0
        r["cliffhanger"]["score"] = 1.0
    assert "STATUS_CONSISTENT" in mutated(ctx, h) or "SCORE_FINAL" in mutated(ctx, h)


def test_overlapping_segments_and_bad_episode(ctx):
    def f(r):
        s = copy.deepcopy(r["segments"][0]); r["segments"].append(s); r["n_segments"] = 2; r["kind"] = "multi-segment"
    assert "SEGMENTS_NOT_CHRONOLOGICAL" in mutated(ctx, f)

    def g(r):
        r["segments"][0]["episode"] = 99; r["episodes"] = [99]
    assert "EPISODE_UNKNOWN" in mutated(ctx, g)


def test_future_episode_reference(ctx):
    def f(r): r["narrative_summary"] += " This echoes the finale in episode 12."
    assert "FUTURE_EPISODE_REFERENCE" in mutated(ctx, f)


def test_collection_level(ctx):
    a = ctx["rec"]; b = copy.deepcopy(a)
    assert {i.code for i in validate_collection([a, b], ctx["cfg"])} >= {"DUPLICATE_ID", "NEAR_DUPLICATE"}
    assert validate_collection([a], ctx["cfg"]) == []


def test_parser_and_scanner_agree_and_detect_tampering(ctx):
    ep = parse_screenplay_text(ctx["text"], "s.txt", 7)
    raw = RawIndex(ctx["text"])
    assert validate_episode(ep, raw) == []
    ep.scenes[0].elements.pop(2)
    assert "BLOCK_COUNT_MISMATCH" in {i.code for i in validate_episode(ep, raw)}
