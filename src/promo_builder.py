"""Layer 5 - promo construction, de-duplication and diversity.

Pipeline per episode
  1. enumerate LEGAL cut points (timeline.py): boundaries that exist in the screenplay, never inside a spoken
     line, never splitting a sentence that continues with "...";
  2. score every start with the hook detector (first 3 s) and every end with the cliffhanger detector;
  3. build continuous windows (20-60 s) between promising starts and ends and score them with ranking.py;
  4. try multi-segment promos (hook beat + cliffhanger beat from the SAME episode, chronological, gap skipped)
     and keep one only if it beats the best continuous cut by a margin and the join is coherent;
  5. suppress near-duplicates (>= 50% overlap) keeping the stronger one.
Cross-episode promos are intentionally NOT generated: the supplied continuity notes show every episode picks up
directly from the previous one, so stitching episodes risks an incoherent first-time-viewer story (the brief
says to prefer a continuous single-episode promo when cross-episode context is insufficient).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cliffhanger_detection import CliffResult, score_cliffhanger
from .hook_detection import HookResult, score_hook
from .ranking import Scored, score_window, transition_score
from .timeline import Timeline
from .timeutil import EPS


@dataclass
class Candidate:
    cid: str
    episode: int
    segs: List[Tuple[float, float]]
    scored: Scored
    hook: HookResult
    cliff: CliffResult
    kind: str = "continuous"            # continuous | multi-segment
    duplicate_of: Optional[str] = None
    overlap_with_duplicate: float = 0.0
    why_multi: Optional[str] = None
    moment_ids: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(E - S for S, E in self.segs), 2)


def _pick(scores: Dict[float, float], thr: float, max_n: int, fallback_n: int = 3) -> List[float]:
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [t for t, s in items if s >= thr][:max_n]
    if len(chosen) < fallback_n:
        chosen = [t for t, _ in items[:fallback_n]]
    return sorted(set(chosen))


def _clean_start(tl: Timeline, S: float) -> bool:
    d = tl.first_dialogue_from(S)
    if d is None or d.start - S > 4.0:
        return True
    f = tl.feat[d.id]
    return not (f.ellipsis_start or (f.reply_start and f.words < 8))


def generate_candidates(tl: Timeline, reg, cfg) -> Tuple[List[Candidate], dict]:
    ep = tl.ep.number
    starts = tl.start_points(cfg.min_duration)
    ends = tl.end_points(cfg.min_duration)
    stats = {"valid_starts": len(starts), "valid_ends": len(ends), "windows_scored": 0, "multi_scored": 0}
    if not starts or not ends:
        return [], stats
    hooks: Dict[float, HookResult] = {S: score_hook(tl, S, cfg.hook_window) for S in starts}
    cliffs: Dict[float, CliffResult] = {E: score_cliffhanger(tl, E) for E in ends}
    ps = _pick({S: h.score for S, h in hooks.items()}, cfg.pre_hook_min, cfg.max_pre)
    pe = _pick({E: c.score for E, c in cliffs.items()}, cfg.pre_cliff_min, cfg.max_pre)

    def windows(starts_, ends_):
        return [(S, E) for S in starts_ for E in ends_ if cfg.min_duration - 1e-6 <= E - S <= cfg.max_duration + 1e-6]

    wins = windows(ps, pe)
    if not wins:                                    # best-effort fallback so the episode can still be reported
        wins = sorted(windows(starts, ends), key=lambda w: -(hooks[w[0]].score + cliffs[w[1]].score))[:60]
    cands: List[Candidate] = []
    for S, E in wins:
        sc = score_window(tl, reg, cfg, [(S, E)], hooks[S], cliffs[E])
        cands.append(Candidate(cid="", episode=ep, segs=sc.segs, scored=sc, hook=hooks[S], cliff=cliffs[E]))
    stats["windows_scored"] = len(cands)

    multi = build_multi(tl, reg, cfg, hooks, cliffs, cands, stats)
    if multi is not None:
        cands.append(multi)
    return cands, stats


def _nearest(points: List[float], target: float, lo: float, hi: float) -> Optional[float]:
    pool = [p for p in points if lo - EPS <= p <= hi + EPS]
    return min(pool, key=lambda p: (abs(p - target), p)) if pool else None


def build_multi(tl: Timeline, reg, cfg, hooks, cliffs, singles: List[Candidate], stats: dict) -> Optional[Candidate]:
    good_starts = sorted((S for S, h in hooks.items() if h.score >= cfg.hook_min), key=lambda S: -hooks[S].score)[:6]
    good_ends = sorted((E for E, c in cliffs.items() if c.score >= cfg.cliff_min), key=lambda E: -cliffs[E].score)[:6]
    if not good_starts or not good_ends:
        return None
    vstarts, vends = sorted(hooks), sorted(cliffs)
    ok_single = [c for c in singles if c.scored.status != "rejected"]
    best_single = max(ok_single, key=lambda c: c.scored.final) if ok_single else None
    seen = set()
    best: Optional[Candidate] = None
    for Sa in good_starts:
        for Eb in good_ends:
            if Eb - Sa < cfg.min_duration + cfg.min_gap:
                continue
            for lenA in (10.0, 14.0, 18.0, 24.0):
                Ea = _nearest(vends, Sa + lenA, Sa + cfg.min_segment, Sa + lenA + 3.0)
                if Ea is None or Ea >= Eb - cfg.min_gap - cfg.min_segment:
                    continue
                for total in (30.0, 40.0, 50.0, 60.0):
                    lenB = total - (Ea - Sa)
                    if lenB < cfg.min_segment:
                        continue
                    pool = [s for s in vstarts if abs(s - (Eb - lenB)) <= 3.0 and s >= Ea + cfg.min_gap and Eb - s >= cfg.min_segment and _clean_start(tl, s)]
                    if not pool:
                        continue
                    Sb = min(pool, key=lambda s: (abs(s - (Eb - lenB)), s))
                    segs = [(Sa, Ea), (Sb, Eb)]
                    dur = (Ea - Sa) + (Eb - Sb)
                    key = (Sa, Ea, Sb, Eb)
                    if key in seen or not (cfg.min_duration - 1e-6 <= dur <= cfg.max_duration + 1e-6):
                        continue
                    seen.add(key)
                    t, tdet = transition_score(tl, segs[0], segs[1], reg)
                    if t < cfg.min_transition:
                        continue
                    stats["multi_scored"] += 1
                    sc = score_window(tl, reg, cfg, segs, hooks[Sa], cliffs[Eb])
                    if sc.status == "rejected":
                        continue
                    if best is None or sc.final > best.scored.final:
                        best = Candidate(cid="", episode=tl.ep.number, segs=sc.segs, scored=sc, hook=hooks[Sa], cliff=cliffs[Eb], kind="multi-segment")
    if best is None:
        return None
    if best_single is not None and best.scored.final < best_single.scored.final + cfg.multi_margin:
        return None
    Sa, Ea = best.segs[0]; Sb, Eb = best.segs[1]
    gap = Sb - Ea
    bs = best_single
    base = (f"The best continuous cut in episode {tl.ep.number} ({bs.segs[0][0]:.0f}-{bs.segs[0][1]:.0f}s) scores {bs.scored.final:.1f} "
            f"(hook {bs.scored.comps.hook:.1f}, cliffhanger {bs.scored.comps.cliff:.1f}). " if bs else
            f"No continuous cut in episode {tl.ep.number} passes the eligibility gates. ")
    best.why_multi = (base + f"The strongest hook ({Sa:.0f}s) and the strongest unresolved beat ({Eb:.0f}s) are {Eb - Sa:.0f}s apart, "
                      f"so one continuous cut cannot hold both. Joining them skips {gap:.0f}s of lower-value material and scores "
                      f"{best.scored.final:.1f} ({'+' if bs is None else ''}{(best.scored.final - bs.scored.final) if bs else best.scored.final:.1f}"
                      f"{' vs the best continuous cut' if bs else ''}). Join quality: shared characters "
                      f"{best.scored.details['coherence']['transitions'][0]['shared_characters'] or 'none'}, scenes "
                      f"{best.scored.details['coherence']['transitions'][0]['scene_a']}->{best.scored.details['coherence']['transitions'][0]['scene_b']}.")
    return best


def overlap_ratio(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    ov = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            ov += max(0.0, min(e1, e2) - max(s1, s2))
    da = sum(e - s for s, e in a)
    db = sum(e - s for s, e in b)
    return ov / max(min(da, db), 1e-9)


def select_for_episode(cands: List[Candidate], cfg) -> Tuple[List[Candidate], List[Candidate], Optional[Candidate]]:
    """Greedy non-maximum suppression -> (kept eligible/flagged, duplicates, best rejected)."""
    ok = sorted((c for c in cands if c.scored.status != "rejected"), key=lambda c: (-c.scored.final, c.segs[0][0]))
    kept: List[Candidate] = []
    dups: List[Candidate] = []
    for c in ok:
        hit = next((k for k in kept if overlap_ratio(c.segs, k.segs) >= cfg.dup_overlap), None)
        if hit is not None:
            c.duplicate_of = hit.cid or f"ep{hit.episode}@{hit.segs[0][0]}"
            c.overlap_with_duplicate = round(overlap_ratio(c.segs, hit.segs), 2)
            dups.append(c)
        elif len(kept) < cfg.max_per_episode:
            kept.append(c)
        # beyond max_per_episode: simply not selected
    rej = [c for c in cands if c.scored.status == "rejected"]
    best_rej = max(rej, key=lambda c: c.scored.final) if rej else None
    return kept, dups, best_rej


def promo_id(c: Candidate, taken: set) -> str:
    base = f"P-E{c.episode:03d}-{c.segs[0][0]:07.2f}-{c.segs[-1][1]:07.2f}" + ("-M%d" % len(c.segs) if len(c.segs) > 1 else "")
    pid, n = base, 2
    while pid in taken:
        pid = f"{base}-{n}"; n += 1
    taken.add(pid)
    return pid


def pick_top(ranked: List[Candidate], cfg, category_of) -> Tuple[List[Candidate], List[dict]]:
    """Top-k by score under two mild diversity rules (<=1 per episode, <=2 per event category)."""
    top: List[Candidate] = []
    skipped: List[dict] = []
    for c in ranked:
        if len(top) >= cfg.top_k:
            break
        same_ep = sum(1 for t in top if t.episode == c.episode)
        cat = category_of(c)
        same_cat = sum(1 for t in top if category_of(t) == cat)
        if same_ep >= cfg.top_max_per_episode:
            skipped.append({"promo_id": c.cid, "reason": f"episode {c.episode} already represented in the top {cfg.top_k}"})
        elif same_cat >= cfg.top_max_per_category:
            skipped.append({"promo_id": c.cid, "reason": f"event category '{cat}' already has {same_cat} promos in the top {cfg.top_k}"})
        else:
            top.append(c)
    return top, skipped
