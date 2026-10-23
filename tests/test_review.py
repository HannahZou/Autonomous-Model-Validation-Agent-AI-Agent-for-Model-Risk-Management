import json
from datetime import date, timedelta

import pytest

from mva.orchestrator import Orchestrator
from mva.review import OverallRating, ReviewError, ReviewSession, ReviewStatus
from mva.schemas import Severity
from mva.turnaround import turnaround


@pytest.fixture
def run_dir(settings, market_pkg):
    return Orchestrator(settings).validate(market_pkg)[1]


def _events(run_dir, kind):
    lines = (run_dir / "audit.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == kind]


def test_all_findings_start_pending(run_dir):
    s = ReviewSession(run_dir)
    assert s.pending() == [f.finding_id for f in s.run.findings]
    assert s.progress()["pending"] == len(s.run.findings)


def test_accept_is_recorded_and_persisted(run_dir):
    s = ReviewSession(run_dir)
    fid = s.pending()[0]
    s.accept(fid, "Hannah Zou", "agree")
    again = ReviewSession(run_dir)  # reload from disk
    r = again.review(fid)
    assert r.status is ReviewStatus.ACCEPTED and r.reviewer == "Hannah Zou" and r.comment == "agree"
    (event,) = _events(run_dir, "review_decision")
    assert event["finding_id"] == fid and event["action"] == "accept"


def test_reject_and_edit_require_rationale(run_dir):
    s = ReviewSession(run_dir)
    fid = s.pending()[0]
    with pytest.raises(ReviewError, match="rationale"):
        s.reject(fid, "Hannah", "  ")
    with pytest.raises(ReviewError, match="rationale"):
        s.edit(fid, "Hannah", "", severity="Low")
    with pytest.raises(ReviewError, match="reviewer"):
        s.accept(fid, "")
    with pytest.raises(ReviewError, match="No change"):
        s.edit(fid, "Hannah", "because")  # nothing changed


def test_edit_overrides_appear_in_final_findings(run_dir):
    s = ReviewSession(run_dir)
    high = next(f for f in s.run.findings if f.severity is Severity.HIGH)
    s.edit(high.finding_id, "Hannah", "Duplicate date is a single row; impact immaterial.", severity="Low",
           recommendation="Remove the duplicate row.")
    final = {x.finding.finding_id: x for x in s.final_findings()}[high.finding_id]
    assert final.finding.severity is Severity.LOW and final.original_severity is Severity.HIGH
    assert final.finding.recommendation == "Remove the duplicate row."
    assert final.finding.description == high.description  # untouched fields keep the original
    (event,) = _events(run_dir, "review_decision")
    assert event["severity_before"] == "High" and event["severity_after"] == "Low"


def test_rejected_findings_leave_the_report_but_not_the_record(run_dir):
    s = ReviewSession(run_dir)
    fid = s.pending()[0]
    s.reject(fid, "Hannah", "Not a model issue.")
    assert fid not in {x.finding.finding_id for x in s.final_findings()}
    assert fid in {x.finding.finding_id for x in s.final_findings(include_rejected=True)}


def test_rating_and_remediation_dates(run_dir):
    s = ReviewSession(run_dir)
    for fid in s.pending():
        s.accept(fid, "Hannah")
    assert s.rating() is OverallRating.UNSATISFACTORY  # the example has High findings
    for x in s.final_findings():
        if x.finding.severity is Severity.HIGH:
            s.reject(x.finding.finding_id, "Hannah", "test")
    assert s.rating() is OverallRating.SATISFACTORY_WITH_FINDINGS
    due = {x.finding.severity: x.remediation_due for x in s.final_findings()}
    assert due[Severity.MEDIUM] == (date.today() + timedelta(days=180)).isoformat()


def test_sign_off_rules(run_dir):
    s = ReviewSession(run_dir)
    with pytest.raises(ReviewError, match="pending"):
        s.sign_off("Hannah")
    for fid in s.pending():
        s.accept(fid, "Hannah")
    signed = s.sign_off("Hannah", "ok")
    assert signed.rating is s.rating()
    with pytest.raises(ReviewError, match="signed off"):
        s.accept(s.run.findings[0].finding_id, "Hannah")
    with pytest.raises(ReviewError, match="Already"):
        s.sign_off("Hannah")
    assert len(_events(run_dir, "signed_off")) == 1


def test_reopen_returns_finding_to_pending(run_dir):
    s = ReviewSession(run_dir)
    fid = s.pending()[0]
    s.reject(fid, "Hannah", "no")
    s.reopen(fid, "Hannah")
    assert s.review(fid).status is ReviewStatus.PENDING


def test_turnaround_from_audit_log(run_dir):
    s = ReviewSession(run_dir)
    for fid in s.pending():
        s.accept(fid, "Hannah")
    s.sign_off("Hannah")
    t = turnaround(run_dir)
    assert t["decisions"] == len(s.run.findings)
    assert t["automated_seconds"] >= 0 and set(t["agent_seconds"]) == {r.agent for r in s.run.agent_reports}
    assert t["total_hours"] >= t["review_hours"] >= 0


def test_run_records_manifest_snapshot_and_artifacts(run_dir):
    snap = json.loads((run_dir / "manifest.json").read_text())
    assert snap["manifest"]["model_id"] == "MR-VAR-004"
    assert len(snap["file_sha256"]["prices.csv"]) == 64
    assert (run_dir / "artifacts" / "BT_var_backtest.csv").exists()
