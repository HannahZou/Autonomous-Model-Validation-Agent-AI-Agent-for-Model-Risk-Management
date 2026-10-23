import json

import yaml

from mva.agents import AssumptionsAgent, BacktestingAgent
from mva.package import ModelPackage
from mva.schemas import Severity
from mva.synthetic import write_credit_package, write_market_package


def _by_check(report):
    return {f.check: f for f in report.findings}


# -- credit -------------------------------------------------------------------
def test_credit_assumptions_on_injected_package(settings, credit_pkg):
    report = AssumptionsAgent(settings).run(ModelPackage.load(credit_pkg))
    assert report.status == "completed", report.error
    found = _by_check(report)
    assert set(found["multicollinearity"].affected) == {"credit_utilization", "utilization_6m_avg"}
    assert found["multicollinearity"].severity is Severity.HIGH
    assert "replication" not in found  # the delivered coefficients are reproducible
    assert report.metrics["max_pd_difference"] < settings.assumptions.replication_tolerance


def test_credit_backtesting_on_injected_package(settings, credit_pkg):
    report = BacktestingAgent(settings).run(ModelPackage.load(credit_pkg))
    found = _by_check(report)
    level = found["calibration_level"]
    assert level.severity is Severity.HIGH and "under-predicts" in level.title
    assert level.evidence["observed_defaults"] > level.evidence["expected_defaults"]
    assert len(report.metrics["calibration"]["deciles"]) == 10
    assert report.metrics["discrimination"]["validation"]["auc"] > 0.7


def test_replication_failure_detected(settings, credit_pkg):
    path = credit_pkg / "model.json"
    model = json.loads(path.read_text())
    model["coefficients"]["debt_to_income"] *= 2  # delivered coefficients don't match the data
    path.write_text(json.dumps(model))
    found = _by_check(AssumptionsAgent(settings).run(ModelPackage.load(credit_pkg)))
    assert found["replication"].severity is Severity.HIGH


def test_wrong_sign_detected(settings, credit_pkg):
    path = credit_pkg / "model.json"
    model = json.loads(path.read_text())
    model["expected_signs"]["debt_to_income"] = "-"
    path.write_text(json.dumps(model))
    found = _by_check(AssumptionsAgent(settings).run(ModelPackage.load(credit_pkg)))
    assert found["coefficient_signs"].affected == ["debt_to_income"]


def test_missing_model_fails_agent_cleanly(settings, credit_pkg):
    path = credit_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    del manifest["model"]
    path.write_text(yaml.safe_dump(manifest))
    for agent in (AssumptionsAgent, BacktestingAgent):
        report = agent(settings).run(ModelPackage.load(credit_pkg))
        assert report.status == "failed" and "model" in report.error


# -- market -------------------------------------------------------------------
def test_market_assumptions_on_injected_package(settings, market_pkg):
    report = AssumptionsAgent(settings).run(ModelPackage.load(market_pkg))
    found = _by_check(report)
    assert {"volatility_clustering", "regime_representativeness"} <= set(found)
    assert report.metrics["recent_vs_window_vol_ratio"] > settings.assumptions.regime_ratio
    assert report.metrics["data_treatment"]  # cleaning steps are disclosed
    assert "normality" not in found  # historical VaR does not assume normality


def test_market_backtesting_on_injected_package(settings, market_pkg):
    report = BacktestingAgent(settings).run(ModelPackage.load(market_pkg))
    found = _by_check(report)
    assert found["traffic_light"].evidence["zone"] in {"yellow", "red"}
    assert found["unconditional_coverage"].severity is Severity.HIGH
    m = report.metrics
    assert m["exceptions"] > m["expected_exceptions"]
    assert m["traffic_light"]["days"] == 250


def test_parametric_var_flags_fat_tails(settings, market_pkg):
    path = market_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["var"]["method"] = "parametric"
    path.write_text(yaml.safe_dump(manifest))
    found = _by_check(AssumptionsAgent(settings).run(ModelPackage.load(market_pkg)))
    assert found["normality"].severity is Severity.HIGH


# -- false positives ------------------------------------------------------------
def test_clean_models_rarely_trigger_high_findings(settings, tmp_path):
    """Statistical tests reject a correct model at roughly their significance level.

    Across 12 clean credit and 12 clean market packages we allow at most 2 packages
    with a High finding from the Week 2 agents (the first 12 seeds give 1).
    """
    with_high = []
    for seed in range(12):
        for kind, write in (("credit", write_credit_package), ("market", write_market_package)):
            pkg = ModelPackage.load(write(tmp_path / f"{kind}{seed}", seed=seed, inject_issues=False))
            for agent in (AssumptionsAgent, BacktestingAgent):
                report = agent(settings).run(pkg)
                assert report.status == "completed", report.error
                if any(f.severity is Severity.HIGH for f in report.findings):
                    with_high.append((kind, seed, agent.name))
    assert len(with_high) <= 2, with_high
