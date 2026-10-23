from mva.agents import DataQualityAgent
from mva.package import ModelPackage
from mva.schemas import Severity
from mva.synthetic import write_credit_package, write_market_package


def _checks(report):
    return {f.check for f in report.findings}


def test_credit_agent_catches_injected_issues(settings, credit_pkg):
    report = DataQualityAgent(settings).run(ModelPackage.load(credit_pkg))
    assert report.status == "completed"
    assert {"missing_values", "duplicate_rows", "range_violations", "outliers",
            "target_leakage", "distribution_shift"} <= _checks(report)
    leak = next(f for f in report.findings if f.check == "target_leakage")
    assert leak.affected == ["recovery_amount"]
    shift = next(f for f in report.findings if f.check == "distribution_shift")
    assert "credit_utilization" in shift.affected


def test_market_agent_catches_injected_issues(settings, market_pkg):
    report = DataQualityAgent(settings).run(ModelPackage.load(market_pkg))
    assert {"duplicate_dates", "calendar_gaps", "stale_prices", "nonpositive_prices",
            "extreme_returns", "missing_values"} <= _checks(report)
    split = next(f for f in report.findings if f.check == "extreme_returns")
    assert split.affected == ["EQ_US"]


def test_clean_data_has_no_material_findings(settings, tmp_path):
    for seed in range(5):
        for pkg in (
            write_credit_package(tmp_path / f"c{seed}", seed=seed, inject_issues=False),
            write_market_package(tmp_path / f"m{seed}", seed=seed, inject_issues=False),
        ):
            report = DataQualityAgent(settings).run(ModelPackage.load(pkg))
            material = [f for f in report.findings if f.severity in (Severity.HIGH, Severity.MEDIUM)]
            assert material == [], (pkg, [f.title for f in material])


def test_findings_numbered_and_sorted_by_severity(settings, credit_pkg):
    report = DataQualityAgent(settings).run(ModelPackage.load(credit_pkg))
    ids = [f.finding_id for f in report.findings]
    assert ids == [f"DQ-{i:03d}" for i in range(1, len(ids) + 1)]
    ranks = [f.severity.rank for f in report.findings]
    assert ranks == sorted(ranks, reverse=True)


def test_results_are_deterministic(settings, credit_pkg):
    a = DataQualityAgent(settings).run(ModelPackage.load(credit_pkg))
    b = DataQualityAgent(settings).run(ModelPackage.load(credit_pkg))
    dump = lambda r: [f.model_dump(exclude={"commentary"}) for f in r.findings]  # noqa: E731
    assert dump(a) == dump(b)


def test_agent_failure_is_recorded_not_raised(settings, credit_pkg):
    (credit_pkg / "test.csv").unlink()
    report = DataQualityAgent(settings).run(ModelPackage.load(credit_pkg))
    assert report.status == "failed"
    assert "test.csv" in report.error
