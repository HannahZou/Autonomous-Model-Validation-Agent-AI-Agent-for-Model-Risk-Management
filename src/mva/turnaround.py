"""Turnaround time, measured from the audit trail rather than estimated."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mva.audit import AuditLog


def _ts(e: dict) -> datetime:
    return datetime.fromisoformat(e["ts"])


def turnaround(run_dir: str | Path) -> dict:
    """Elapsed times for a run: automated analysis, human review, and start to sign-off.

    Review time runs from the end of the automated run to sign-off, so it includes
    any time the run waited for a reviewer; `active_review_minutes` spans only
    the first to the last review decision.
    """
    events = AuditLog(Path(run_dir) / "audit.jsonl").events()
    first = {}
    for e in events:
        first.setdefault(e["event"], e)
    decisions = [e for e in events if e["event"] == "review_decision"]
    out: dict = {"decisions": len(decisions)}
    if "run_started" in first and "run_finished" in first:
        out["automated_seconds"] = round((_ts(first["run_finished"]) - _ts(first["run_started"])).total_seconds(), 2)
        out["agent_seconds"] = {e["agent"]: e["seconds"] for e in events if e["event"] == "agent_finished"}
    if decisions:
        out["active_review_minutes"] = round((_ts(decisions[-1]) - _ts(decisions[0])).total_seconds() / 60, 1)
    if "signed_off" in first and "run_finished" in first:
        out["review_hours"] = round((_ts(first["signed_off"]) - _ts(first["run_finished"])).total_seconds() / 3600, 2)
        out["total_hours"] = round((_ts(first["signed_off"]) - _ts(first["run_started"])).total_seconds() / 3600, 2)
    return out
