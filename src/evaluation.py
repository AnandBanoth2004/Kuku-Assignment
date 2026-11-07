"""Evaluation helpers.  Nothing here invents human judgement: the human-evaluation files are BLANK until people fill them in.

* ``weight_sensitivity``  - how stable is the top-5 when the (assumed) weights are perturbed?  Measures robustness of the ranking
                            to the weight assumption; it says NOTHING about whether the ranking matches audience taste.
* ``human_eval_files``    - writes a blind rating sheet (no system scores shown to raters) + a separate answer key.
* ``compare_human``       - once raters have filled the sheet: Spearman correlation and top-5 overlap between system and humans.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

from .ranking import Components, combine

RATER_COLS = ["rater_id", "hook_1to5", "cliffhanger_1to5", "standalone_clarity_1to5", "would_watch_next_yes_no", "overall_1to10", "notes"]


def _final(cfg, comps: dict, weights: dict) -> float:
    class _C:  # light stand-in so combine() can be reused with perturbed weights
        pass
    c = _C()
    c.weights, c.weakest_link_floor = weights, cfg.weakest_link_floor
    return combine(c, Components(**comps))[1]


def weight_sensitivity(records: List[dict], cfg, trials: int = 200, rel: float = 0.30, seed: int = 7, k: int = 5) -> dict:
    rng = random.Random(seed)
    base = sorted(records, key=lambda r: -_final(cfg, r["scoring"]["components"], cfg.weights))
    base_top = [r["promo_id"] for r in base[:k]]
    overlaps, ranks1 = [], []
    for _ in range(trials):
        w = {n: v * (1 + rng.uniform(-rel, rel)) for n, v in cfg.weights.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}
        order = sorted(records, key=lambda r: -_final(cfg, r["scoring"]["components"], w))
        overlaps.append(len(set(base_top) & {r["promo_id"] for r in order[:k]}) / k)
        ranks1.append(order[0]["promo_id"] == base[0]["promo_id"])
    return {"trials": trials, "relative_weight_perturbation": rel, "seed": seed, "k": k,
            "mean_top_k_overlap": round(mean(overlaps), 3), "min_top_k_overlap": min(overlaps),
            "share_of_trials_with_same_number_one": round(sum(ranks1) / trials, 3), "baseline_top_k": base_top,
            "note": "Robustness of the ranking to the weight ASSUMPTION only; not evidence that the ranking matches audience preference."}


def human_eval_files(records: List[dict], best_rejected: List[dict], out_dir: Path, seed: int = 11, n_random: int = 10) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ranked = sorted(records, key=lambda r: r["rank"])
    top = ranked[:10]
    rest = ranked[10:]
    sample = rng.sample(rest, min(n_random, len(rest)))
    chosen = top + sample + best_rejected[:5]
    rng.shuffle(chosen)                                        # raters must not see the order
    cols = ["promo_id", "episode", "start_ts", "end_ts", "duration_s", "first_line_of_clip", "last_line_of_clip", "summary_for_rater"] + RATER_COLS
    with open(out_dir / "human_eval_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in chosen:
            s0, sN = r["segments"][0], r["segments"][-1]
            first = (s0["selected_dialogue"] or [{"text": ""}])[0]["text"]
            last = (sN["selected_dialogue"] or [{"text": ""}])[-1]["text"]
            w.writerow([r["promo_id"], r["episodes"][0], s0["start_ts"], sN["end_ts"], r["total_duration"], first, last, r["narrative_summary"]] + [""] * len(RATER_COLS))
    with open(out_dir / "human_eval_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["promo_id", "system_rank", "system_final", "status", "group"])
        grp = {r["promo_id"]: "top10" for r in top} | {r["promo_id"]: "random_rest" for r in sample} | {r["promo_id"]: "best_rejected" for r in best_rejected[:5]}
        for r in chosen:
            w.writerow([r["promo_id"], r["rank"] if r["rank"] else "", r["scoring"]["final"], r["status"], grp[r["promo_id"]]])
    return {"rows": len(chosen), "sheet": str(out_dir / "human_eval_sheet.csv"), "key": str(out_dir / "human_eval_key.csv"),
            "note": "Rating columns are EMPTY - no human results exist yet."}


def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            rk[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def compare_human(sheet_csv: str, key_csv: str) -> dict:
    """Spearman(system final, mean human overall) over rated promos; also the share of system top-5 that humans rate in their top-5."""
    key = {r["promo_id"]: r for r in csv.DictReader(open(key_csv, encoding="utf-8"))}
    scores: Dict[str, List[float]] = {}
    for r in csv.DictReader(open(sheet_csv, encoding="utf-8")):
        try:
            scores.setdefault(r["promo_id"], []).append(float(r["overall_1to10"]))
        except (ValueError, KeyError):
            continue
    ids = [p for p in scores if p in key]
    if len(ids) < 5:
        return {"rated_promos": len(ids), "note": "Too few rated promos for a correlation (need >= 5)."}
    sys_f = [float(key[p]["system_final"]) for p in ids]
    hum = [mean(scores[p]) for p in ids]
    rs, rh = _ranks(sys_f), _ranks(hum)
    ms, mh = mean(rs), mean(rh)
    num = sum((a - ms) * (b - mh) for a, b in zip(rs, rh))
    den = (sum((a - ms) ** 2 for a in rs) * sum((b - mh) ** 2 for b in rh)) ** 0.5
    top_sys = set(sorted(ids, key=lambda p: -float(key[p]["system_final"]))[:5])
    top_hum = set(sorted(ids, key=lambda p: -mean(scores[p]))[:5])
    return {"rated_promos": len(ids), "spearman": round(num / den, 3) if den else None, "top5_overlap": len(top_sys & top_hum) / 5}
