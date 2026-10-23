from pathlib import Path

import pytest

from mva.config import load_settings
from mva.synthetic import write_credit_package, write_market_package


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings(use_llm=False, openai_api_key=None, runs_dir=tmp_path / "runs")


@pytest.fixture
def credit_pkg(tmp_path: Path) -> Path:
    return write_credit_package(tmp_path / "credit_pd")


@pytest.fixture
def market_pkg(tmp_path: Path) -> Path:
    return write_market_package(tmp_path / "market_var")


class FakeResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


class FakeOpenAI:
    """Stands in for openai.OpenAI; records calls and returns a canned narrative."""

    def __init__(self, narrative=None, error: Exception | None = None):
        self.calls = []
        self._narrative = narrative
        self._error = error

        outer = self

        class _Responses:
            def parse(self, **kwargs):
                outer.calls.append(kwargs)
                if outer._error:
                    raise outer._error
                return FakeResponse(outer._narrative)

        self.responses = _Responses()
