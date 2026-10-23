from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mva.checks.base import Issue
from mva.config import Settings
from mva.llm import NarrativeWriter, TemplateNarrativeWriter, apply_narrative
from mva.package import ModelPackage
from mva.schemas import AgentReport, Finding

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseAgent(ABC):
    """A specialised validation agent.

    Subclasses implement ``run_checks`` (pure, deterministic). The base class
    numbers the findings, adds the narrative and records timing and errors.
    """

    name: str = "base"
    code: str = "XX"

    def __init__(self, settings: Settings, writer: NarrativeWriter | None = None):
        self.settings = settings
        self.writer = writer or TemplateNarrativeWriter()
        self.fallback = TemplateNarrativeWriter()
        # Tables an agent wants saved with the run (for charts and audit), keyed by short name
        self.artifacts: dict[str, pd.DataFrame] = {}

    @abstractmethod
    def run_checks(self, package: ModelPackage) -> tuple[list[Issue], dict[str, Any]]:
        """Return the issues found and any metrics worth reporting."""

    def run(self, package: ModelPackage) -> AgentReport:
        started = _now()
        self.artifacts = {}
        try:
            issues, metrics = self.run_checks(package)
        except Exception as exc:  # an agent failure must not stop the whole validation
            log.exception("%s agent failed", self.name)
            return AgentReport(agent=self.name, status="failed", error=f"{type(exc).__name__}: {exc}",
                               started_at=started, finished_at=_now())

        issues.sort(key=lambda i: -i.severity.rank)  # stable: keeps check order within a severity
        findings = [
            Finding(finding_id=f"{self.code}-{n:03d}", agent=self.name, **vars(issue))
            for n, issue in enumerate(issues, start=1)
        ]

        context = {
            "model_id": package.manifest.model_id,
            "model_name": package.manifest.name,
            "model_type": package.model_type,
            "documentation": package.manifest.documentation.model_dump(),
        }
        source = self.writer.source
        try:
            narrative = self.writer.write(self.name, context, findings, metrics)
        except Exception as exc:
            log.warning("Narrative generation failed (%s); using template", exc)
            metrics = {**metrics, "narrative_error": f"{type(exc).__name__}: {exc}"}
            narrative = self.fallback.write(self.name, context, findings, metrics)
            source = self.fallback.source
        unknown = apply_narrative(findings, narrative)
        if unknown:
            metrics = {**metrics, "discarded_commentary_ids": unknown}

        return AgentReport(
            agent=self.name,
            findings=findings,
            metrics=metrics,
            summary=narrative.summary.strip(),
            narrative_source=source,
            started_at=started,
            finished_at=_now(),
        )
