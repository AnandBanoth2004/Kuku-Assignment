#!/usr/bin/env python3
"""Stand-alone re-validation of a finished run (does NOT use the parser or the pipeline - only the raw screenplay text).

    python validate_outputs.py --input data/input/screenplays.zip --promos outputs/ranked_promos.json

Exit code 0 only if every promo passes every check against the raw text.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

from config import load_config
from src.validation import RawIndex, validate_collection, validate_promo


def read_raw(path: Path) -> dict:
    blobs = {}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            items = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(".txt") and "__MACOSX" not in n]
    elif path.is_dir():
        items = [(str(p), p.read_bytes()) for p in sorted(path.rglob("*.txt"))]
    else:
        items = [(str(path), path.read_bytes())]
    for name, data in items:
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        idx = RawIndex(text, name)
        if idx.episode_number is not None:
            blobs[idx.episode_number] = idx
    return blobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--promos", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg, raw = load_config(a.config), read_raw(Path(a.input))
    promos = json.loads(Path(a.promos).read_text(encoding="utf-8"))
    bad, checks = 0, 0
    for p in promos:
        r = validate_promo(p, raw, cfg)
        checks += r.n_checks
        if not r.passed:
            bad += 1
            print("FAIL", p["promo_id"], [(e.code, e.message[:90]) for e in r.errors])
    coll = validate_collection(promos, cfg)
    for i in coll:
        print("COLLECTION", i.code, i.message)
    print(f"{len(promos)} promos, {checks} checks, {bad} failed promos, {len(coll)} collection issues, {len(raw)} raw episodes read")
    return 0 if not bad and not coll else 1


if __name__ == "__main__":
    sys.exit(main())
