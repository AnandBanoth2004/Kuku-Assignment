# Find the Promos - Kuku AI Generalist Assignment, Problem 1

Finds 20-60 s promo clips (exact episode + timestamps) in timestamped microdrama screenplays, ranks them with an explainable score, and
**re-checks every timestamp and quoted line against the raw text**. Built and run on the supplied 50-episode pack
*"I Married a Broken Soldier. He was a Duke"*.

> **Honest status.** The delivered run is **fully rule-based (no LLM was called)** - there were no API keys or network in the build environment.
> The LLM layer exists and is unit-tested with *mock* clients only. Scores are heuristic proxies (wording, delivery tags, structure); they are **not** predictions
> of audience engagement and **no human evaluation has been done** (the rating sheet is blank). Lexicons and thresholds were tuned while reading these same 50 episodes.

## Result of the actual run (`outputs/`, 5.5 s on one core)
* 50/50 files parsed, 0 failures: 100 scenes, 1,487 dialogue + 372 action + 1,754 camera + 114 SFX = 3,727 timed elements.
* 6,981 candidate windows scored -> **86 promos recommended in 42 episodes** (25 eligible, 61 flagged for review; 4 multi-segment). Median duration 36.9 s.
* **8 episodes have no promo that passes the gates** (8, 17, 39, 40, 41, 48, 49, 50) - each lists its closest rejected candidate and the failing gate.
* Validation: 3,864 checks on the 86 promos against the raw text, 0 failures; parser and an independent raw scanner agree on all 50 files.
* Top 5: E03 58.30-89.30 (76.1), E13 45.80-80.50 (75.5), E42 17.90-77.80 (75.5), E30 32.00-50.50 + 94.50-125.10 (75.3), E12 63.20-87.21 (75.0).
  The top-5 scores span ~1 point and the order is not stable under weight perturbation - treat it as a short-list.

## Quick start
```bash
pip install -r requirements.txt                       # pyyaml is enough for the rule-based run
python run_pipeline.py --input data/input/screenplays.zip --out outputs
python validate_outputs.py --input data/input/screenplays.zip --promos outputs/ranked_promos.json   # stand-alone re-check
python -m pytest -q                                   # 39 tests
streamlit run app.py                                  # optional viewer (smoke-tested against a stub only)
```
**New screenplay / new show:** `python run_pipeline.py --input path/to/new.txt --out outputs_new` (a `.txt`, a folder or a `.zip`; same code path).
Unknown speakers can be mapped in `configs/aliases.yaml`. Tested on a *synthetic* screenplay and format-perturbed copies (CRLF, BOM, upper-case labels) - not on a real new show.

## Architecture (module per layer)
| Layer | Module | Notes |
|---|---|---|
| 1 Parse | `screenplay_parser.py`, `ingestion.py` | deterministic state machine; per-file failure isolation; never invents timestamps |
| QC | `validation.py` | independent raw scanner, parse-level + promo-level + collection-level checks |
| Continuity | `continuity.py` | character registry with alias *evidence*, spoiler-safe context (only episodes <= current) |
| 2 Events | `event_detection.py` (+ `llm_analysis.enrich_moments`) | beats -> category, description, context needed |
| 3 Hook | `hook_detection.py` | scored from the proposed start over the first 3 s only |
| 4 Cliffhanger | `cliffhanger_detection.py` | a scene end is not automatically a cliffhanger; warm/resolved endings are penalised; the script's own word "cliffhanger" is ignored |
| 5 Promo build | `promo_builder.py` | legal cut points only (existing boundaries, never inside a line, never splitting a "..." sentence), dedup, multi-segment with justification |
| 6 Rank | `ranking.py` | formula + gates below |
| LLM (optional) | `llm_client.py`, `llm_analysis.py`, `prompts/` | provider-independent, cache, retries, schema + hallucination guards, falls back to rules |

**Score** = 10 x (0.25 Hook + 0.25 Cliffhanger + 0.20 Emotion + 0.15 Standalone + 0.10 Coherence + 0.05 Character), components 0-10. The weights are *initial engineering assumptions*.
Guard-rails: **weakest-link cap** `final <= 100 x (min(hook, cliffhanger)/10 + 0.25)` so a strong hook cannot buy back a missing cliffhanger; **hard gates** (20-60 s, hook >= 5, cliffhanger >= 5, standalone >= 4.5, coherence >= 5, emotion >= 3);
**soft flags** send weak-but-eligible promos to human review. Multi-segment promos must beat the best continuous cut by 4 points and carry a written justification.

## Outputs
`ranked_promos.json|csv` (all fields per promo: ids, segments with verbatim dialogue/action, hook, cliffhanger, scores + explanations, why it works, limitations),
`top5_promos.json`, `episode_summary.json|csv`, `closest_rejected_per_episode.json`, `validation_report.json`, `parse_summary.json`, `continuity.json`, `moments.json`,
`evaluation/` (weight sensitivity, **blank** human-evaluation sheet + key), `raw_outputs/` (every scored window, unedited), `pipeline.log`.

## LLM mode (optional, untested against a real model)
`--llm anthropic --model <model> --api-key-env ANTHROPIC_API_KEY` or `--llm openai_compat --base-url <url> --model <m>`. The model may relabel beats, re-judge the six components of the top-N
promos (blended 50/50 with the rules; a gap > 3 points flags the promo) and write titles/summaries. It can never change a timestamp or quote; invented ids/quotes reject the review and the rule-based record is kept.

## Scalability
Episodes are independent -> embarrassingly parallel (`~0.1 s/episode` for the rule layers here). Work is O(legal cut points); the LLM, if used, sees only the top-N clips (cost grows with N, not with the number of windows) and results are cached by prompt hash.
New episodes only need their own parse plus a registry update. Not yet done: a database, incremental runs, real-endpoint load tests.

## Confidentiality
The input pack is assessment material marked confidential: it is git-ignored (`data/input`, `data/extracted`) and must not be published. `outputs/` quotes lines from it, so share it only with the assessors and keep any repository private. API keys are read only from env vars.

See `docs/FAILURE_ANALYSIS.md` (observed vs hypothetical failures), `docs/WALKTHROUGH.md`, `docs/INTERVIEW_PREP.md`.
