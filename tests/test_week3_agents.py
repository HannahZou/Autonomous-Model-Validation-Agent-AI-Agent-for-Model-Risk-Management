import json

import numpy as np
import pandas as pd
import pytest
import yaml

from mva import stats
from mva.agents import ChallengerAgent, SensitivityAgent
from mva.checks.challenger import paired_bootstrap_auc_gap
from mva.models import market as mk
from mva.package import ModelPackage, PackageError
from mva.schemas import Severity
from mva.synthetic import write_credit_package, write_market_package

rng = np.random.default_rng(5)


def _by_check(report):
    return {f.check: f for f in report.findings}


def _returns(n=600, vol=0.01):
    return pd.Series(rng.standard_t(5, n) * vol / np.sqrt(5 / 3), index=pd.bdate_range("2023-01-02", periods=n))


# -- building blocks -------------------------------------------------------------
def test_logit_covariance_consistent_with_standard_errors():
    X = rng.normal(size=(3000, 2)) * [1, 20] + [0, 100]
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2 + 0.7 * X[:, 0] + 0.02 * (X[:, 1] - 100)))))
    res = stats.fit_logit(X, y)
    assert res.cov.shape == (3, 3)
    assert np.allclose(np.sqrt(np.diag(res.cov))[1:], res.se)


@pytest.mark.parametrize("fn", ["ewma", "fhs"])
def test_challenger_var_has_no_look_ahead(fn):
    r = _returns()
    make = (lambda s: mk.ewma_var(s, 0.99)) if fn == "ewma" else (lambda s: mk.fhs_var(s, 250, 0.99))
    r2 = r.copy()
    r2.iloc[-1] = -0.3
    assert make(r).iloc[-1] == make(r2).iloc[-1]


def test_quantile_loss_prefers_correct_quantile():
    r = _returns(5000)
    true_q = -np.quantile(r, 0.01)
    good = pd.DataFrame({"return": r, "var": true_q})
    bad = pd.DataFrame({"return": r, "var": true_q * 0.5})
    assert mk.quantile_loss(good, 0.99) < mk.quantile_loss(bad, 0.99)


def test_expected_shortfall_exceeds_var():
    r = _returns()
    assert mk.expected_shortfall(r, 250, 0.99) >= mk.latest_var(r, 250, 0.99, "historical")


def test_paired_bootstrap_identical_models_has_zero_gap():
    y = rng.binomial(1, 0.2, 1000)
    p = rng.uniform(size=1000)
    lo, hi = paired_bootstrap_auc_gap(y, p, p, n=50, seed=0)
    assert lo == hi == 0.0


def test_stress_scenario_must_reference_portfolio_assets(market_pkg):
    path = market_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["stress_scenarios"]["Bad"] = {"GOLD": -0.1}
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(PackageError, match="GOLD"):
        ModelPackage.load(market_pkg)


# -- Challenger agent ------------------------------------------------------------
def test_credit_challenger_detects_threshold_effect(settings, credit_pkg):
    report = ChallengerAgent(settings).run(ModelPackage.load(credit_pkg))
    assert report.status == "completed", report.error
    outperform = [f for f in report.findings if f.check == "challenger_outperforms"]
    assert outperform, "tree challengers should capture the DTI step the linear champion misses"
    for f in outperform:
        assert f.evidence["auc_gap"] > settings.challenger.auc_gap_medium
        assert f.evidence["auc_gap_95ci"][0] > 0
    assert set(report.metrics["models"]) == {"Champion (delivered model)", "Gradient boosting", "Random forest"}


def test_market_challenger_beats_lagging_champion(settings, market_pkg):
    report = ChallengerAgent(settings).run(ModelPackage.load(market_pkg))
    found = _by_check(report)
    assert found["challenger_outperforms"].severity is Severity.MEDIUM
    models = report.metrics["models"]
    assert len({m["days"] for m in models.values()}) == 1  # compared on the same days
    assert models["EWMA (RiskMetrics)"]["exceptions"] < models["Champion"]["exceptions"]


# -- Sensitivity agent -----------------------------------------------------------
def test_credit_sensitivity_metrics(settings, credit_pkg):
    report = SensitivityAgent(settings).run(ModelPackage.load(credit_pkg))
    m = report.metrics
    assert m["adverse_scenario"]["stressed_mean_pd"] > m["adverse_scenario"]["base_mean_pd"]
    shares = [v["share"] for v in m["feature_shocks"].values()]
    assert sum(shares) == pytest.approx(1.0, abs=0.01)
    lo, hi = m["parameter_uncertainty"]["pd_90_interval"]
    assert lo < m["parameter_uncertainty"]["mean_pd"] < hi


def test_counterintuitive_model_fails_adverse_scenario(settings, credit_pkg):
    path = credit_pkg / "model.json"
    model = json.loads(path.read_text())
    model["coefficients"] = {k: -v for k, v in model["coefficients"].items()}
    path.write_text(json.dumps(model))
    found = _by_check(SensitivityAgent(settings).run(ModelPackage.load(credit_pkg)))
    assert found["adverse_scenario_response"].severity is Severity.HIGH


def test_market_sensitivity_metrics_and_stress(settings, market_pkg):
    report = SensitivityAgent(settings).run(ModelPackage.load(market_pkg))
    m = report.metrics
    assert set(m["var_by_window"]) == {"125", "250", "500"}
    assert sum(m["component_var_share"].values()) == pytest.approx(1.0, abs=0.01)
    assert set(m["hypothetical_scenarios"]) == {
        "Equity crash", "Rates sell-off", "Risk-off (equities down, euro down)"
    }
    # The worst "loss" in the history is the unadjusted stock split
    stress = _by_check(report)["stress_vs_var"]
    assert stress.evidence["multiple_of_var"] > settings.sensitivity.stress_multiple
    assert m["es_97_5_to_var_ratio"] > 0


# -- false positives ---------------------------------------------------------------
def test_clean_models_have_no_material_week3_findings(settings, tmp_path):
    material = []
    packages = [write_credit_package(tmp_path / f"c{s}", seed=s, inject_issues=False) for s in range(4)]
    packages += [write_market_package(tmp_path / f"m{s}", seed=s, inject_issues=False) for s in range(12)]
    for path in packages:
        pkg = ModelPackage.load(path)
        for agent in (ChallengerAgent, SensitivityAgent):
            report = agent(settings).run(pkg)
            assert report.status == "completed", report.error
            material += [(path.name, f.check) for f in report.findings
                         if f.severity in (Severity.HIGH, Severity.MEDIUM)]
    assert material == []
