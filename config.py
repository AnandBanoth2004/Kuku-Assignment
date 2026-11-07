"""Central configuration (defaults live here; override with a YAML file and/or environment variables).

Nothing secret is stored in code or in config files: API keys are read ONLY from environment variables whose
NAMES are configured below (default ANTHROPIC_API_KEY / OPENAI_API_KEY).

    cfg = load_config("configs/config.yaml")        # file is optional
    PROMOAI_LLM_PROVIDER=anthropic python run_pipeline.py ...   # env override
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent


@dataclass
class LLMConfig:
    provider: str = "none"            # none | anthropic | openai_compat | replay
    model: str = ""                   # e.g. claude-haiku-5-5 | gpt-4o-mini | gemini-2.0-flash | llama3.1 (ollama)
    base_url: str = ""                # openai_compat only (OpenAI, Gemini OpenAI-endpoint, Groq, OpenRouter, Ollama ...)
    api_key_env: str = ""             # NAME of the environment variable that holds the key
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 60.0
    max_retries: int = 4
    cache_dir: str = "outputs/raw_outputs/llm_cache"
    use_for_events: bool = True       # Layer 2: semantic event extraction per episode
    use_for_rescoring: bool = True    # re-judge the top-N promo candidates
    rescore_top_n: int = 25
    blend: float = 0.5                # weight of the LLM judgement vs the rule score (0 = rules only)
    disagreement_flag: float = 3.0    # |rule - llm| on any component above this -> flag for human review


@dataclass
class Config:
    # ---- ranking (initial ENGINEERING ASSUMPTIONS, not empirically fitted industry weights) -------------
    weights: dict = field(default_factory=lambda: {"hook": 0.25, "cliff": 0.25, "emotion": 0.20,
                                                   "standalone": 0.15, "coherence": 0.10, "character": 0.05})
    weakest_link_floor: float = 0.25   # cap = 100 * (min(hook, cliff)/10 + floor): a strong hook cannot rescue a weak ending
    emo_scale: float = 1.35            # maps raw 0..1 intensity statistics onto a usable 0..10 range

    # ---- promo construction ------------------------------------------------------------------------------
    min_duration: float = 20.0
    max_duration: float = 60.0
    hook_window: float = 3.0
    pre_hook_min: float = 3.5          # cut-point pre-filters (speed); best-effort fallbacks keep every episode reportable
    pre_cliff_min: float = 3.5
    max_pre: int = 40
    min_segment: float = 8.0           # shortest segment of a multi-segment promo
    min_gap: float = 2.5               # minimum skipped time between two segments of one episode
    multi_margin: float = 4.0          # a multi-segment promo must beat the best continuous one by this many points
    min_transition: float = 0.55       # minimum transition coherence (0..1) between joined segments

    # ---- eligibility gates (hard) and soft flags ---------------------------------------------------------
    hook_min: float = 5.0
    cliff_min: float = 5.0
    standalone_min: float = 4.5
    coherence_min: float = 5.0
    emotion_min: float = 3.0
    soft_flag: float = 6.0             # eligible but below this on hook / cliffhanger / standalone -> flagged for review
    coherence_soft: float = 6.5

    # ---- dedup / diversity -------------------------------------------------------------------------------
    dup_overlap: float = 0.5           # >= 50% overlap of the shorter promo => near-duplicate
    max_per_episode: int = 3
    top_k: int = 5
    top_max_per_episode: int = 1
    top_max_per_category: int = 2

    # ---- io ----------------------------------------------------------------------------------------------
    aliases_file: str = "configs/aliases.yaml"
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> Config:
    data = asdict(Config())
    if path and Path(path).exists():
        import yaml
        data = _merge(data, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    env = os.environ
    llm = data["llm"]
    for key, var in (("provider", "PROMOAI_LLM_PROVIDER"), ("model", "PROMOAI_LLM_MODEL"),
                     ("base_url", "PROMOAI_LLM_BASE_URL"), ("api_key_env", "PROMOAI_LLM_API_KEY_ENV")):
        if env.get(var):
            llm[key] = env[var]
    llm_cfg = LLMConfig(**llm)
    data.pop("llm")
    cfg = Config(**data)
    cfg.llm = llm_cfg
    s = sum(cfg.weights.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"ranking weights must sum to 1.0, got {s}")
    return cfg
