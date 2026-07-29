"""Tests for UI configuration resolution.

The API URL resolver is worth testing on its own: it already broke the app
once (st.secrets raising on a missing secrets.toml), and its Render path
receives a bare hostname that would silently produce unreachable requests.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve(monkeypatch, value: str | None):
    """Import the resolver in isolation, with the environment controlled."""
    import streamlit as st

    # Simulate a deployment with no secrets.toml, where .get() raises.
    class _Raising:
        def get(self, *_a, **_kw):
            raise RuntimeError("No secrets found")

    monkeypatch.setattr(st, "secrets", _Raising(), raising=False)
    if value is None:
        monkeypatch.delenv("API_URL", raising=False)
    else:
        monkeypatch.setenv("API_URL", value)

    sys.modules.pop("ui.app", None)
    spec = importlib.util.spec_from_file_location(
        "_ui_probe", Path(__file__).resolve().parent.parent / "ui" / "app.py")
    # Only the resolver is needed; executing the whole script would start
    # rendering the dashboard.
    source = spec.origin
    namespace: dict = {"__file__": source}
    text = Path(source).read_text()
    start = text.index("def _resolve_api_url")
    end = text.index("API_URL = _resolve_api_url()")
    exec("import os\nimport streamlit as st\n" + text[start:end], namespace)
    return namespace["_resolve_api_url"]()


def test_missing_secrets_does_not_raise(monkeypatch):
    """The bug that blanked the whole page: st.secrets raises, not returns."""
    assert _resolve(monkeypatch, None) == "http://localhost:8000"


def test_env_var_is_used(monkeypatch):
    assert _resolve(monkeypatch, "http://nginx:80") == "http://nginx:80"


def test_trailing_slash_stripped(monkeypatch):
    assert _resolve(monkeypatch, "https://api.example.com/") == "https://api.example.com"


def test_bare_hostname_gets_https(monkeypatch):
    """Render's fromService injects a hostname with no scheme."""
    assert _resolve(monkeypatch, "crop-disease-api.onrender.com") == \
        "https://crop-disease-api.onrender.com"


@pytest.mark.parametrize("value", ["localhost:8000", "127.0.0.1:8000", "nginx:80"])
def test_local_hosts_get_http_not_https(monkeypatch, value):
    """Local targets must not be upgraded to https -- nothing serves TLS there."""
    assert _resolve(monkeypatch, value).startswith("http://")
