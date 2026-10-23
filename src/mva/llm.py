"""Narrative layer: turns deterministic findings into validation-report prose.

Design rules (enforced in code, not just in the prompt):
* The LLM only receives aggregate statistics and documentation, never raw rows.
* It can write a summary and per-finding commentary; it cannot add, remove or
  re-grade findings. Commentary for unknown finding IDs is discarded.
* Any API failure falls back to a deterministic template, so a run never fails
  because the LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from pydantic import BaseModel

from mva.schemas import Finding, Severity

log = logging.getLogger(__name__)


class FindingCommentary(BaseModel):
    finding_id: str
    commentary: str


class AgentNarrative(BaseModel):
    summary: str
    commentary: list[FindingCommentary]


class NarrativeWriter(Protocol):
    source: str

    def write(self, agent: str, context: dict[str, Any], findings: list[Finding], metrics: dict[str, Any]) -> AgentNarrative: ...


# -- deterministic fallback -------------------------------------------------
class TemplateNarrativeWriter:
    source = "template"

    def write(self, agent, context, findings, metrics) -> AgentNarrative:
        counts = {s: sum(f.severity is s for f in findings) for s in Severity}
        if not findings:
            summary = f"The {agent} review of {context['model_name']} identified no exceptions."
        else:
            parts = [f"{n} {s.value.lower()}" for s, n in counts.items() if n]
            top = [f.title for f in findings if f.severity is Severity.HIGH][:3]
            summary = (
                f"The {agent} review of {context['model_name']} identified {len(findings)} finding(s) "
                f"({', '.join(parts)} severity)."
            )
            if top:
                summary += " High-severity items: " + "; ".join(top) + "."
        return AgentNarrative(summary=summary, commentary=[])


# -- OpenAI -----------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior model validator in a bank's Model Risk Management function, \
writing sections of a validation report under the Federal Reserve's SR 11-7 guidance.

You receive the model's documentation and a list of findings produced by deterministic tests.
Write:
1. `summary`: 3-5 sentences for the report section, stating the overall conclusion of this \
review area and the most material issues for the model's intended use.
2. `commentary`: for each finding, 1-3 sentences on why it matters for THIS model's purpose \
and assumptions, and what the validator should confirm with the model owner.

Rules:
- Use only facts in the input. Never invent statistics, dates or column names.
- Do not change, argue with, or restate a different severity than the one given.
- Use each finding_id exactly as given. Professional, neutral tone; no marketing language."""


class OpenAINarrativeWriter:
    source = "llm"

    def __init__(self, model: str, api_key: str | None = None, client: Any | None = None):
        self.model = model
        if client is None:
            from openai import OpenAI  # imported lazily so the package works offline

            client = OpenAI(api_key=api_key)
        self.client = client

    def write(self, agent, context, findings, metrics) -> AgentNarrative:
        payload = {
            "review_area": agent,
            "model": context,
            "findings": [
                f.model_dump(include={"finding_id", "title", "severity", "pillar", "description", "evidence"}, mode="json")
                for f in findings
            ],
        }
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            text_format=AgentNarrative,
        )
        narrative = response.output_parsed
        if narrative is None:
            raise ValueError("model returned no parsed output")
        return narrative


def apply_narrative(findings: list[Finding], narrative: AgentNarrative) -> list[str]:
    """Attach commentary to findings in place. Returns IDs the LLM made up (discarded)."""
    by_id = {f.finding_id: f for f in findings}
    unknown = []
    for c in narrative.commentary:
        if c.finding_id in by_id:
            by_id[c.finding_id].commentary = c.commentary.strip()
        else:
            unknown.append(c.finding_id)
    if unknown:
        log.warning("Discarded commentary for unknown finding IDs: %s", unknown)
    return unknown


# -- executive summary (written after human review) ----------------------------
class ExecutiveSummary(BaseModel):
    summary: str
    key_finding_ids: list[str]


EXEC_SYSTEM_PROMPT = """You are the lead model validator writing the executive summary of a \
validation report under SR 11-7, after a human reviewer has accepted, edited or rejected each \
automated finding.

Write `summary`: one paragraph of 4-6 sentences for senior management stating the overall rating \
and what it means for use of the model, the most material issues and their business impact, and \
the main remediation themes. List in `key_finding_ids` the 1-5 finding IDs the summary relies on.

Rules: use only the facts provided; do not invent numbers; do not change any severity or the \
overall rating; refer only to finding IDs that appear in the input; plain, neutral language."""


class ExecutiveSummaryWriter(Protocol):
    source: str

    def write(self, payload: dict[str, Any]) -> ExecutiveSummary: ...


class TemplateExecutiveSummaryWriter:
    source = "template"

    def write(self, payload: dict[str, Any]) -> ExecutiveSummary:
        findings = payload["findings"]
        counts = {s: sum(f["severity"] == s for f in findings) for s in ("High", "Medium", "Low")}
        text = (
            f"The validation of {payload['model']['name']} concludes with an overall rating of "
            f"{payload['rating']}. {payload['rating_meaning']} After review, {len(findings)} finding(s) were retained: "
            f"{counts['High']} High, {counts['Medium']} Medium and {counts['Low']} Low severity."
        )
        top = [f for f in findings if f["severity"] == "High"][:3] or findings[:3]
        if top:
            text += " The most material are " + "; ".join(f"{f['title']} ({f['finding_id']})" for f in top) + "."
        return ExecutiveSummary(summary=text, key_finding_ids=[f["finding_id"] for f in top])


class OpenAIExecutiveSummaryWriter:
    source = "llm"

    def __init__(self, model: str, api_key: str | None = None, client: Any | None = None):
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client

    def write(self, payload: dict[str, Any]) -> ExecutiveSummary:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": EXEC_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            text_format=ExecutiveSummary,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("model returned no parsed output")
        return parsed


def write_executive_summary(writer: ExecutiveSummaryWriter, payload: dict[str, Any]) -> tuple[ExecutiveSummary, str, str | None]:
    """Run the writer with guardrails; returns (summary, source, error)."""
    valid = {f["finding_id"] for f in payload["findings"]}
    try:
        result = writer.write(payload)
        source, error = writer.source, None
    except Exception as exc:  # never block the report on the API
        log.warning("Executive summary generation failed (%s); using template", exc)
        result, source, error = TemplateExecutiveSummaryWriter().write(payload), "template", f"{type(exc).__name__}: {exc}"
    result.key_finding_ids = [i for i in result.key_finding_ids if i in valid]
    return result, source, error
