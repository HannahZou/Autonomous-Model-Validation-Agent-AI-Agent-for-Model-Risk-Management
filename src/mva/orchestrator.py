"""Orchestrator: runs the agents over a model package and records the run."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mva.agents import DEFAULT_AGENTS, BaseAgent
from mva.config import Settings
from mva.llm import NarrativeWriter, OpenAINarrativeWriter, TemplateNarrativeWriter
from mva.package import ModelPackage
from mva.schemas import ValidationRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AuditLog:
    """Append-only JSONL trail of everything the system did during a run."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **data: Any) -> None:
        record = {"ts": _now().isoformat(), "event": kind, **data}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def default_writer(settings: Settings) -> NarrativeWriter:
    if settings.llm_enabled:
        return OpenAINarrativeWriter(model=settings.openai_model, api_key=settings.openai_api_key)
    return TemplateNarrativeWriter()


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        writer: NarrativeWriter | None = None,
        agents: list[type[BaseAgent]] | None = None,
    ):
        self.settings = settings
        self.writer = writer or default_writer(settings)
        self.agent_classes = agents or DEFAULT_AGENTS

    def validate(self, package_path: str | Path) -> tuple[ValidationRun, Path]:
        package = ModelPackage.load(package_path)
        m = package.manifest
        run_id = f"{_now():%Y%m%d-%H%M%S}-{m.model_id}-{uuid.uuid4().hex[:6]}"
        out_dir = self.settings.runs_dir / run_id
        audit = AuditLog(out_dir / "audit.jsonl")

        run = ValidationRun(run_id=run_id, model_id=m.model_id, model_name=m.name,
                            model_type=package.model_type, started_at=_now())
        audit.event("run_started", run_id=run_id, model_id=m.model_id, model_type=package.model_type,
                    package=str(Path(package_path).resolve()), narrative=self.writer.source)

        for cls in self.agent_classes:
            agent = cls(self.settings, self.writer)
            audit.event("agent_started", agent=agent.name)
            report = agent.run(package)
            run.agent_reports.append(report)
            audit.event(
                "agent_finished", agent=agent.name, status=report.status, error=report.error,
                findings=report.severity_counts(), narrative=report.narrative_source,
                seconds=round(report.duration_seconds, 3),
            )

        run.finished_at = _now()
        audit.event("run_finished", findings=run.severity_counts(), seconds=round(run.duration_seconds, 3))

        (out_dir / "results.json").write_text(run.model_dump_json(indent=2))
        (out_dir / "summary.md").write_text(render_summary(run))
        return run, out_dir


def render_summary(run: ValidationRun) -> str:
    """Interim markdown summary. The full validation report arrives in Week 4."""
    c = run.severity_counts()
    lines = [
        f"# Validation run {run.run_id}",
        "",
        f"**Model:** {run.model_name} ({run.model_id}, {run.model_type})  ",
        f"**Run time:** {run.duration_seconds:.1f} s  ",
        f"**Findings:** {c['High']} High · {c['Medium']} Medium · {c['Low']} Low",
        "",
    ]
    for r in run.agent_reports:
        lines += [f"## {r.agent}", ""]
        if r.status != "completed":
            lines += [f"> Agent failed: {r.error}", ""]
            continue
        lines += [r.summary, "", f"_Narrative: {r.narrative_source}_", ""]
        if r.findings:
            lines += ["| ID | Severity | Finding | SR 11-7 element |", "|---|---|---|---|"]
            lines += [f"| {f.finding_id} | {f.severity.value} | {f.title} | {f.pillar.value} |" for f in r.findings]
            lines.append("")
            for f in r.findings:
                lines += [f"### {f.finding_id} · {f.title} ({f.severity.value})", "", f.description, ""]
                if f.commentary:
                    lines += [f"*Validator commentary:* {f.commentary}", ""]
                lines += [f"**Recommendation:** {f.recommendation}", ""]
    return "\n".join(lines)
