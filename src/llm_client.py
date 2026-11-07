"""Provider-independent LLM access: one tiny interface, swappable back-ends, caching, retries, strict JSON.

    client = make_client(cfg.llm)          # None when provider == "none"  -> the pipeline stays fully rule-based
    raw    = client.complete(system, user) # str
    obj    = extract_json(raw)             # dict, or LLMError

Back-ends (all via ``requests``; no vendor SDK needed, no key in code - only the NAME of an env var is configured):
  anthropic       Anthropic Messages API
  openai_compat   any OpenAI-style /chat/completions endpoint (OpenAI, Gemini's OpenAI endpoint, Groq, OpenRouter, local Ollama ...)
  replay          canned answers (tests / offline demos)

TESTED ONLY WITH MOCKS in this project's build environment (no network, no API keys): see tests/test_llm.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Any failure of the LLM layer (transport, rate limit, bad JSON, schema violation, missing key)."""


class RetryableError(LLMError):
    pass


class LLMClient:
    name = "abstract"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class _HTTPClient(LLMClient):
    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int, timeout: float, session=None):
        self.model, self.key, self.temperature, self.max_tokens, self.timeout = model, api_key, temperature, max_tokens, timeout
        if session is None:
            import requests
            session = requests
        self.http = session

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        try:
            r = self.http.post(url, headers=headers, json=body, timeout=self.timeout)
        except Exception as exc:                                   # network problem -> retryable
            raise RetryableError(f"transport error: {type(exc).__name__}: {exc}") from exc
        if r.status_code == 429 or r.status_code >= 500:
            raise RetryableError(f"HTTP {r.status_code} (rate limit / server)")
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {str(getattr(r, 'text', ''))[:200]}")
        try:
            return r.json()
        except Exception as exc:
            raise LLMError(f"response is not JSON: {exc}") from exc


class AnthropicClient(_HTTPClient):
    name = "anthropic"

    def complete(self, system: str, user: str) -> str:
        data = self._post("https://api.anthropic.com/v1/messages",
                          {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                           "system": system, "messages": [{"role": "user", "content": user}]})
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Anthropic response shape: {exc}") from exc


class OpenAICompatClient(_HTTPClient):
    name = "openai_compat"

    def __init__(self, base_url: str, *a, **kw):
        super().__init__(*a, **kw)
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        headers = {"content-type": "application/json"}
        if self.key:
            headers["authorization"] = f"Bearer {self.key}"
        data = self._post(f"{self.base_url}/chat/completions", headers,
                          {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat-completions response shape: {exc}") from exc


class ReplayClient(LLMClient):
    """Canned answers: ``responses`` is a callable (system, user) -> str or a list consumed in order."""
    name = "replay"

    def __init__(self, responses):
        self.responses, self.calls = responses, 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if callable(self.responses):
            return self.responses(system, user)
        if not self.responses:
            raise LLMError("replay client has no more canned responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RobustClient(LLMClient):
    """Wraps any client with an on-disk cache and retry-with-backoff (rate limits, 5xx, transport errors)."""

    def __init__(self, inner: LLMClient, cache_dir: Optional[str] = None, max_retries: int = 4, sleep: Callable[[float], None] = time.sleep,
                 model: str = "", temperature: float = 0.0):
        self.inner, self.max_retries, self.sleep = inner, max_retries, sleep
        self.cache = Path(cache_dir) if cache_dir else None
        self.salt = f"{inner.name}|{model}|{temperature}"
        self.name = inner.name
        self.hits = self.misses = 0
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, user: str) -> Path:
        return self.cache / (hashlib.sha256(f"{self.salt}\n{system}\n{user}".encode()).hexdigest() + ".json")

    def complete(self, system: str, user: str) -> str:
        if self.cache:
            p = self._key(system, user)
            if p.exists():
                self.hits += 1
                return json.loads(p.read_text(encoding="utf-8"))["response"]
        delay, last = 1.0, None
        for attempt in range(self.max_retries + 1):
            try:
                out = self.inner.complete(system, user)
                break
            except RetryableError as exc:
                last = exc
                log.warning("LLM retry %d/%d after: %s", attempt + 1, self.max_retries, exc)
                if attempt == self.max_retries:
                    raise LLMError(f"gave up after {self.max_retries} retries: {last}") from exc
                self.sleep(delay)
                delay = min(delay * 2, 30.0)
        self.misses += 1
        if self.cache:
            self._key(system, user).write_text(json.dumps({"salt": self.salt, "system": system, "user": user, "response": out}), encoding="utf-8")
        return out


def make_client(llm_cfg, env: Optional[dict] = None, session=None) -> Optional[LLMClient]:
    """Build the configured client or return None (rule-based run).  Raises LLMError if the configured key is missing."""
    env = os.environ if env is None else env
    prov = (llm_cfg.provider or "none").lower()
    if prov == "none":
        return None
    if prov == "replay":
        raise LLMError("the replay provider needs canned responses; construct ReplayClient directly")
    key = env.get(llm_cfg.api_key_env or "", "") if llm_cfg.api_key_env else ""
    if prov == "anthropic":
        if not key:
            raise LLMError(f"environment variable {llm_cfg.api_key_env or 'ANTHROPIC_API_KEY'} is not set")
        inner: LLMClient = AnthropicClient(llm_cfg.model, key, llm_cfg.temperature, llm_cfg.max_tokens, llm_cfg.timeout_s, session)
    elif prov == "openai_compat":
        if not llm_cfg.base_url:
            raise LLMError("openai_compat needs llm.base_url")
        if llm_cfg.api_key_env and not key:
            raise LLMError(f"environment variable {llm_cfg.api_key_env} is not set")
        inner = OpenAICompatClient(llm_cfg.base_url, llm_cfg.model, key, llm_cfg.temperature, llm_cfg.max_tokens, llm_cfg.timeout_s, session)
    else:
        raise LLMError(f"unknown provider {prov!r}")
    return RobustClient(inner, llm_cfg.cache_dir, llm_cfg.max_retries, model=llm_cfg.model, temperature=llm_cfg.temperature)


def extract_json(text: str) -> dict:
    """First balanced JSON object in ``text`` (tolerates ```json fences and chatter around it)."""
    t = re.sub(r"```(?:json)?", "", text or "")
    start = t.find("{")
    if start < 0:
        raise LLMError("no JSON object in the model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"invalid JSON from the model: {exc}") from exc
                if not isinstance(obj, dict):
                    raise LLMError("model output is not a JSON object")
                return obj
    raise LLMError("unterminated JSON object in the model output")


def check_schema(obj: dict, schema: Dict[str, tuple]) -> None:
    """Minimal schema: {key: (type|tuple_of_types, lo, hi)}; lo/hi apply to numbers.  Raises LLMError on the first violation."""
    for key, spec in schema.items():
        typ, lo, hi = (spec + (None, None))[:3] if isinstance(spec, tuple) else (spec, None, None)
        if key not in obj:
            raise LLMError(f"schema: missing key '{key}'")
        v = obj[key]
        if isinstance(v, bool) or not isinstance(v, typ):
            raise LLMError(f"schema: '{key}' has type {type(v).__name__}, expected {typ}")
        if lo is not None and isinstance(v, (int, float)) and not (lo <= v <= hi):
            raise LLMError(f"schema: '{key}'={v} outside {lo}-{hi}")
