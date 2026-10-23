"""Drive app/streamlit_app.py with a fake streamlit module (no server or browser needed)."""

import os
import runpy
import sys
from pathlib import Path

import pytest
from fake_streamlit import FakeStreamlit, Rerun, Stop, as_module

from mva.orchestrator import Orchestrator
from mva.review import ReviewSession, ReviewStatus

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _run_app(fake: FakeStreamlit, monkey_env: dict):
    old_mod = sys.modules.get("streamlit")
    old_env = {k: os.environ.get(k) for k in monkey_env}
    sys.modules["streamlit"] = as_module(fake)
    os.environ.update(monkey_env)
    try:
        runpy.run_path(str(APP), run_name="__main__")
        return "done"
    except Rerun:
        return "rerun"
    except Stop:
        return "stop"
    finally:
        if old_mod is not None:
            sys.modules["streamlit"] = old_mod
        else:
            sys.modules.pop("streamlit", None)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def run_dir(settings, market_pkg):
    return Orchestrator(settings).validate(market_pkg)[1]


def _env(run_dir):
    return {"MVA_RUNS_DIR": str(run_dir.parent), "MVA_USE_LLM": "0", "OPENAI_API_KEY": ""}


def test_app_without_a_run_asks_to_start(tmp_path):
    fake = FakeStreamlit()
    assert _run_app(fake, {"MVA_RUNS_DIR": str(tmp_path / "none"), "MVA_USE_LLM": "0"}) == "stop"
    assert ("info", ("Run a validation or open an earlier run from the sidebar.",)) in fake.calls


def test_app_renders_a_run(run_dir):
    fake = FakeStreamlit()
    fake.session_state["run_dir"] = str(run_dir)
    assert _run_app(fake, _env(run_dir)) == "done"
    titles = [a[0] for name, a in fake.calls if name == "expander" and a]
    assert any("DQ-001" in t for t in titles)
    assert not fake.errors


def test_app_accept_requires_reviewer_then_records_decision(run_dir):
    first = ReviewSession(run_dir).pending()[0]
    fake = FakeStreamlit(press={(f"accept-{first}", "Accept")})
    fake.session_state["run_dir"] = str(run_dir)
    assert _run_app(fake, _env(run_dir)) == "done"  # no reviewer name: rule enforced, error shown
    assert any("reviewer name" in e for e in fake.errors)

    fake = FakeStreamlit(inputs={"reviewer": "Tester"}, press={(f"accept-{first}", "Accept")})
    fake.session_state["run_dir"] = str(run_dir)
    assert _run_app(fake, _env(run_dir)) == "rerun"
    r = ReviewSession(run_dir).review(first)
    assert r.status is ReviewStatus.ACCEPTED and r.reviewer == "Tester"


def test_app_sign_off_and_report(run_dir):
    s = ReviewSession(run_dir)
    for fid in s.pending():
        s.accept(fid, "Tester")
    fake = FakeStreamlit(inputs={"reviewer": "Tester"}, press={"Sign off validation"})
    fake.session_state["run_dir"] = str(run_dir)
    assert _run_app(fake, _env(run_dir)) == "rerun"
    assert ReviewSession(run_dir).signed_off

    fake = FakeStreamlit(inputs={"reviewer": "Tester"}, press={"Generate report"})
    fake.session_state["run_dir"] = str(run_dir)
    assert _run_app(fake, _env(run_dir)) == "done"
    assert Path(fake.session_state["report"]).exists()
    assert any(name == "download_button" for name, _ in fake.calls)
