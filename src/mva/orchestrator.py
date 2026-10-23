"""Orchestrator: runs the agents over a model package and records the run."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from mva.agents import DEFAULT_AGENTS, BaseAgent
from mva.audit import AuditLog, now
from mva.config import Settings
from mva.llm import NarrativeWriter, OpenAINarrativeWriter, TemplateNarrativeWriter
from mva.package import ModelPackage
from mva.schemas import ValidationRun

_now = now


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

        write_manifest_snapshot(package, Path(package_path), out_dir)

        for cls in self.agent_classes:
            agent = cls(self.settings, self.writer)
            audit.event("agent_started", agent=agent.name)
            report = agent.run(package)
            for name, table in agent.artifacts.items():
                rel = f"artifacts/{agent.code}_{name}.csv"
                (out_dir / "artifacts").mkdir(exist_ok=True)
                table.to_csv(out_dir / rel, index=False)
                report.artifacts.append(rel)
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


def write_manifest_snapshot(package: ModelPackage, package_path: Path, out_dir: Path) -> None:
    """Record exactly what was validated: the manifest and a SHA-256 of every file in the package."""
    files = {
        str(f.relative_to(package.root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(package.root.iterdir()) if f.is_file()
    }
    snapshot = {
        "package_path": str(package_path.resolve()),
        "manifest": package.manifest.model_dump(mode="json"),
        "file_sha256": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(snapshot, indent=2))


def render_summary(run: ValidationRun) -> str:
    """Quick-look markdown summary of the automated run (the reviewed report is built by mva.report)."""
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
                    lines += [f"*AI commentary (for the reviewer):* {f.commentary}", ""]
                lines += [f"**Recommendation:** {f.recommendation}", ""]
    return "\n".join(lines)
