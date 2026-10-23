"""Shared data contracts for findings, agent reports and validation runs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"

    @property
    def rank(self) -> int:
        return {"High": 3, "Medium": 2, "Low": 1, "Info": 0}[self.value]


class SR117Pillar(str, Enum):
    """The three core elements of model validation under SR 11-7."""

    CONCEPTUAL_SOUNDNESS = "Evaluation of conceptual soundness"
    ONGOING_MONITORING = "Ongoing monitoring"
    OUTCOMES_ANALYSIS = "Outcomes analysis"


class Finding(BaseModel):
    """One validation finding. Produced by deterministic checks only."""

    finding_id: str
    agent: str
    check: str
    title: str
    severity: Severity
    pillar: SR117Pillar
    description: str
    recommendation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    affected: list[str] = Field(default_factory=list)
    # Filled in by the LLM narrative layer; never changes severity or evidence.
    commentary: str | None = None


class AgentReport(BaseModel):
    agent: str
    status: str = "completed"  # completed | failed
    findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    narrative_source: str = "template"  # template | llm
    error: str | None = None
    # Data files written alongside results.json (e.g. the VaR backtest series used for charts)
    artifacts: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def severity_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts


class ValidationRun(BaseModel):
    run_id: str
    model_id: str
    model_name: str
    model_type: str
    started_at: datetime
    finished_at: datetime | None = None
    agent_reports: list[AgentReport] = Field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.agent_reports for f in r.findings]

    def severity_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
