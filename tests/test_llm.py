"""LLM layer tests use MOCK clients only (no network / keys in the build environment). They prove the guard-rails, not model quality."""
import copy
import json
import tempfile
from pathlib import Path

import pytest

from config import LLMConfig, load_config
from run_pipeline import run
from src.llm_analysis import review_top
from src.llm_client import (AnthropicClient, LLMError, ReplayClient, RetryableError, RobustClient, check_schema, extract_json, make_client)
from src.validation import RawIndex, validate_promo
from tests.fixtures import build_text


@pytest.fixture(scope="module")
def ctx():
    d = Path(tempfile.mkdtemp())
    text = build_text(7)
    (d / "s.txt").write_text(text, encoding="utf-8")
    cfg = load_config()
    res = run(d / "s.txt", d / "out", cfg, extract_dir=d / "ex")
    return {"rec": res["ranked"][0], "cfg": cfg, "raw": {7: RawIndex(text)}, "reg": None, "ep": res["summary"]}


def good_review(rec, **over):
    seg = rec["segments"][0]
    hook_id = rec["hook"]["element_id"]
    cliff_id = seg["selected_dialogue"][-1]["element_id"]
    obj = {"hook": {"score": 9, "element_id": hook_id, "reason": "shouted discovery"}, "cliffhanger": {"score": 8, "element_id": cliff_id, "reason": "open question"},
           "emotion": 8, "standalone": 7, "coherence": 8, "character": 7, "event_category": "threat", "title": "The ledger", "narrative_summary": "A captain is cornered.",
           "why_it_works": "Stakes are immediate.", "limitations": ["synthetic"], "quotes": [seg["selected_dialogue"][0]["text"][:20]]}
    obj.update(over)
    return obj


class _Ep:
    number, logline = 7, "synthetic"


def _run(ctx, answer, **kw):
    rec = copy.deepcopy(ctx["rec"])
    cfg = load_config()
    client = ReplayClient([answer] if not callable(answer) else answer)
    class _Reg:
        def context_for(self, n, k=3): return []
    stats = review_top(client, [rec], {7: _Ep()}, _Reg(), cfg, 1)
    return rec, stats, cfg


def test_valid_review_is_applied_blended_and_still_validates(ctx):
    rec, stats, cfg = _run(ctx, json.dumps(good_review(ctx["rec"])))
    assert stats["applied"] == 1 and rec["provenance"]["llm_used"]
    before = ctx["rec"]["scoring"]["components"]["hook"]
    assert rec["scoring"]["components"]["hook"] == round(0.5 * 9 + 0.5 * before, 2)
    assert rec["segments"] == ctx["rec"]["segments"], "the LLM can never change timestamps or quoted text"
    assert validate_promo(rec, ctx["raw"], cfg).passed


def test_hook_outside_first_3s_is_capped_at_rule_score(ctx):
    late = ctx["rec"]["segments"][0]["selected_dialogue"][-1]["element_id"]
    rec, stats, _ = _run(ctx, json.dumps(good_review(ctx["rec"], hook={"score": 10, "element_id": late, "reason": "x"})))
    assert stats["applied"] == 1
    assert rec["scoring"]["components"]["hook"] == ctx["rec"]["scoring"]["components"]["hook"]      # LLM's 10 was not allowed to raise it
    assert any("first 3 s" in n for n in rec["provenance"]["notes"])


@pytest.mark.parametrize("answer,why", [
    ("not json at all", "no JSON"),
    ('{"hook": ', "unterminated"),
    (None, "hallucinated element id"),
    (None, "hallucinated quote"),
    (None, "outside 0-10"),
    (None, "unknown event_category"),
])
def test_bad_reviews_are_rejected_and_record_left_untouched(ctx, answer, why):
    if why == "hallucinated element id":
        answer = json.dumps(good_review(ctx["rec"], hook={"score": 9, "element_id": "E999-S9-999", "reason": "x"}))
    elif why == "hallucinated quote":
        answer = json.dumps(good_review(ctx["rec"], quotes=["a line that does not exist in the clip"]))
    elif why == "outside 0-10":
        answer = json.dumps(good_review(ctx["rec"], emotion=11))
    elif why == "unknown event_category":
        answer = json.dumps(good_review(ctx["rec"], event_category="space_opera"))
    rec, stats, _ = _run(ctx, answer)
    assert stats["failed"] == 1 and stats["applied"] == 0
    assert why.split()[0] in stats["errors"][0]["error"] or why in stats["errors"][0]["error"]
    assert rec["scoring"] == ctx["rec"]["scoring"] and rec["provenance"]["llm_used"] is False


def test_extract_json_and_schema():
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    with pytest.raises(LLMError):
        check_schema({"x": 11}, {"x": ((int, float), 0, 10)})
    with pytest.raises(LLMError):
        check_schema({}, {"x": int})


def test_retry_backoff_then_success_and_give_up():
    sleeps = []
    c = RobustClient(ReplayClient([RetryableError("429"), RetryableError("429"), "ok"]), sleep=sleeps.append)
    assert c.complete("s", "u") == "ok" and sleeps == [1.0, 2.0]
    c2 = RobustClient(ReplayClient([RetryableError("429")] * 3), max_retries=2, sleep=lambda s: None)
    with pytest.raises(LLMError):
        c2.complete("s", "u")


def test_cache_prevents_second_call(tmp_path):
    inner = ReplayClient(["first"])
    c = RobustClient(inner, str(tmp_path))
    assert c.complete("s", "u") == "first" and c.complete("s", "u") == "first"
    assert inner.calls == 1 and c.hits == 1


def test_make_client_key_handling_and_http_error_paths():
    assert make_client(LLMConfig(provider="none")) is None
    with pytest.raises(LLMError):
        make_client(LLMConfig(provider="anthropic", model="m", api_key_env="NO_SUCH_VAR"), env={})
    class R:
        def __init__(s, code, body=None): s.status_code, s._b, s.text = code, body or {}, "x"
        def json(s): return s._b
    class Sess:
        def __init__(s): s.q = [R(429), R(200, {"content": [{"type": "text", "text": "{\"ok\": 1}"}]})]
        def post(s, *a, **k): return s.q.pop(0)
    cli = make_client(LLMConfig(provider="anthropic", model="m", api_key_env="K", max_retries=3), env={"K": "secret"}, session=Sess())
    cli.sleep = lambda s: None
    assert isinstance(cli.inner, AnthropicClient) and extract_json(cli.complete("s", "u")) == {"ok": 1}
