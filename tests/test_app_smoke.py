"""Streamlit is not installed in the build environment, so app.py is executed against a stub `streamlit` to catch wiring errors."""
import os
import runpy
import sys
import types
from pathlib import Path


class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def metric(self, *a, **k): pass


class _St(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.calls = []

    def cache_data(self, f): return f
    def tabs(self, names): return [_Ctx() for _ in names]
    def columns(self, n): return [_Ctx() for _ in range(n)]
    def selectbox(self, label, options, *a, **k): return list(options)[0]
    def stop(self): raise SystemExit

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append(name)


def test_app_runs_against_stub_with_real_outputs():
    out = Path(__file__).resolve().parent.parent / "outputs"
    if not (out / "ranked_promos.json").exists():
        import pytest
        pytest.skip("run the pipeline first")
    stub = _St()
    sys.modules["streamlit"] = stub
    os.environ["PROMOAI_OUT"] = str(out)
    try:
        runpy.run_path(str(Path(__file__).resolve().parent.parent / "app.py"))
    finally:
        sys.modules.pop("streamlit", None)
    assert "title" in stub.calls and "dataframe" in stub.calls and "download_button" in stub.calls
