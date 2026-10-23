import json

from conftest import FakeOpenAI

from mva.agents import DataQualityAgent
from mva.llm import AgentNarrative, FindingCommentary, OpenAINarrativeWriter
from mva.package import ModelPackage


def _baseline(settings, pkg):
    return DataQualityAgent(settings).run(ModelPackage.load(pkg))


def test_llm_commentary_attached_without_changing_findings(settings, credit_pkg):
    base = _baseline(settings, credit_pkg)
    narrative = AgentNarrative(
        summary="Data are not fit for purpose until leakage is removed.",
        commentary=[FindingCommentary(finding_id="DQ-001", commentary="Matters because...")],
    )
    client = FakeOpenAI(narrative)
    writer = OpenAINarrativeWriter(model="test-model", client=client)
    report = DataQualityAgent(settings, writer).run(ModelPackage.load(credit_pkg))

    assert report.narrative_source == "llm"
    assert report.summary.startswith("Data are not fit")
    assert report.findings[0].commentary == "Matters because..."
    strip = lambda r: [f.model_dump(exclude={"commentary"}) for f in r.findings]  # noqa: E731
    assert strip(report) == strip(base)  # severity, evidence, wording all unchanged
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["text_format"] is AgentNarrative


def test_unknown_finding_ids_are_discarded(settings, credit_pkg):
    narrative = AgentNarrative(
        summary="s", commentary=[FindingCommentary(finding_id="DQ-999", commentary="invented")]
    )
    writer = OpenAINarrativeWriter(model="m", client=FakeOpenAI(narrative))
    report = DataQualityAgent(settings, writer).run(ModelPackage.load(credit_pkg))
    assert all(f.commentary is None for f in report.findings)
    assert report.metrics["discarded_commentary_ids"] == ["DQ-999"]


def test_api_failure_falls_back_to_template(settings, market_pkg):
    writer = OpenAINarrativeWriter(model="m", client=FakeOpenAI(error=RuntimeError("rate limited")))
    report = DataQualityAgent(settings, writer).run(ModelPackage.load(market_pkg))
    assert report.status == "completed"
    assert report.narrative_source == "template"
    assert "rate limited" in report.metrics["narrative_error"]
    assert report.findings  # findings are still delivered


def test_only_aggregates_are_sent_to_the_api(settings, credit_pkg):
    client = FakeOpenAI(AgentNarrative(summary="s", commentary=[]))
    writer = OpenAINarrativeWriter(model="m", client=client)
    DataQualityAgent(settings, writer).run(ModelPackage.load(credit_pkg))
    payload = json.loads(client.calls[0]["input"][1]["content"])
    assert set(payload) == {"review_area", "model", "findings"}
    for f in payload["findings"]:
        assert set(f) == {"finding_id", "title", "severity", "pillar", "description", "evidence"}
    # 8k-row dataset; a row-level leak would make the payload far larger
    assert len(client.calls[0]["input"][1]["content"]) < 20_000
