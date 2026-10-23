import json

from mva.cli import main
from mva.orchestrator import Orchestrator
from mva.schemas import ValidationRun


def test_run_writes_results_summary_and_audit(settings, credit_pkg):
    run, out_dir = Orchestrator(settings).validate(credit_pkg)
    assert run.finished_at is not None and run.duration_seconds >= 0

    saved = ValidationRun.model_validate_json((out_dir / "results.json").read_text())
    assert [f.finding_id for f in saved.findings] == [f.finding_id for f in run.findings]

    summary = (out_dir / "summary.md").read_text()
    assert "Suspected target leakage" in summary

    events = [json.loads(line)["event"] for line in (out_dir / "audit.jsonl").read_text().splitlines()]
    assert events == ["run_started", "agent_started", "agent_finished", "run_finished"]


def test_cli_validate_both_types(settings, credit_pkg, market_pkg, capsys):
    for pkg in (credit_pkg, market_pkg):
        assert main(["validate", str(pkg), "--no-llm", "--runs-dir", str(settings.runs_dir)]) == 0
    out = capsys.readouterr().out
    assert "CR-PD-001" in out and "MR-VAR-004" in out


def test_cli_bad_package_returns_2(tmp_path):
    assert main(["validate", str(tmp_path), "--no-llm", "--runs-dir", str(tmp_path / "r")]) == 2
