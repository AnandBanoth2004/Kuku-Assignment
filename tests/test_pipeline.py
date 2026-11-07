"""New-screenplay workflow: the same entry point as the supplied pack. Tested on a SYNTHETIC screenplay and format-perturbed copies -
NOT on a real new show."""
import json
import tempfile
from pathlib import Path

from config import load_config
from run_pipeline import run
from src.ingestion import load_pack
from tests.fixtures import build_text


def _go(text, name="s.txt"):
    d = Path(tempfile.mkdtemp())
    (d / name).write_text(text, encoding="utf-8")
    res = run(d / name, d / "out", load_config(), extract_dir=d / "ex")
    return d, res


def test_single_new_file_end_to_end_and_outputs_written():
    d, res = _go(build_text(7))
    assert res["stats"]["validation_overall_ok"] and res["stats"]["promos_recommended"] >= 1
    for f in ("ranked_promos.json", "top5_promos.json", "episode_summary.json", "validation_report.json", "parse_summary.json", "pipeline.log"):
        assert (d / "out" / f).exists(), f
    r = json.loads((d / "out" / "ranked_promos.json").read_text(encoding="utf-8"))[0]
    assert 20 <= r["total_duration"] <= 60 and r["segments"][0]["start_ts"]


def test_format_perturbations_give_identical_promos():
    base = [(r["promo_id"], r["scoring"]["final"]) for r in _go(build_text(7))[1]["ranked"]]
    for kw in ({"upper_meta": True}, {"crlf": True}, {"bom": True}):
        got = [(r["promo_id"], r["scoring"]["final"]) for r in _go(build_text(7, **kw))[1]["ranked"]]
        assert got == base, kw


def test_one_bad_file_never_stops_the_batch():
    d = Path(tempfile.mkdtemp())
    (d / "episode-001.txt").write_text(build_text(1), encoding="utf-8")
    (d / "episode-002.txt").write_text("this is not a screenplay\nat all", encoding="utf-8")
    (d / "episode-003.txt").write_text(build_text(3), encoding="utf-8")
    eps, summ = load_pack(d)
    assert [e.number for e in eps] == [1, 3] and summ.files_discovered == 3
    assert len(summ.failed) == 1 and summ.failed[0]["file"] == "episode-002.txt" and summ.failed[0]["reason"]
