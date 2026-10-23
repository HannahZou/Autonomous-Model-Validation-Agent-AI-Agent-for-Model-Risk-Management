import numpy as np
import pandas as pd

from mva.checks import common, credit
from mva.config import DataQualityThresholds
from mva.schemas import Severity

TH = DataQualityThresholds()
rng = np.random.default_rng(0)


def test_psi_near_zero_for_same_distribution():
    a = pd.Series(rng.normal(size=5000))
    b = pd.Series(rng.normal(size=5000))
    assert credit.population_stability_index(a, b) < 0.02


def test_psi_large_for_shifted_distribution():
    a = pd.Series(rng.normal(size=5000))
    b = pd.Series(rng.normal(loc=1.0, size=5000))
    assert credit.population_stability_index(a, b) > TH.psi_high


def test_missing_values_severity_buckets():
    df = pd.DataFrame({"a": [np.nan] * 30 + [1] * 70, "b": [np.nan] * 10 + [1] * 90, "c": [np.nan] + [1] * 99})
    issues = common.missing_values(df, ["a", "b", "c"], "development", TH)
    by_sev = {i.severity: i.affected for i in issues}
    assert by_sev == {Severity.HIGH: ["a"], Severity.MEDIUM: ["b"], Severity.LOW: ["c"]}


def test_no_missing_values_no_issue():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert common.missing_values(df, ["a"], "development", TH) == []


def test_duplicates_and_conflicting_ids():
    df = pd.DataFrame({"id": [1, 1, 2, 2, 3], "x": [5, 5, 7, 8, 9]})
    issues = common.duplicate_rows(df, "development", TH, id_column="id")
    checks = {i.check: i for i in issues}
    assert checks["duplicate_rows"].evidence["duplicate_rows"] == 1
    assert checks["duplicate_ids"].severity is Severity.HIGH
    assert checks["duplicate_ids"].evidence["conflicting_ids"] == 1


def test_range_violation_is_high():
    df = pd.DataFrame({"age": [25, 40, 150, -1]})
    (issue,) = credit.range_violations(df, {"age": (18, 100)}, "development")
    assert issue.severity is Severity.HIGH
    assert issue.evidence["violations"]["age"]["count"] == 2


def test_outliers_flag_data_errors_but_not_short_tenure():
    income = pd.Series(rng.lognormal(11, 0.5, 2000))
    income.iloc[:30] = 99_999_999
    tenure = pd.Series(np.clip(rng.gamma(2.0, 30.0, 2000), 0, 480).round())
    df = pd.DataFrame({"income": income, "tenure": tenure})
    (issue,) = credit.outliers(df, ["income", "tenure"], "development", TH)
    assert issue.affected == ["income"]
    assert issue.evidence["features"]["income"]["count"] == 30
    assert issue.severity is Severity.MEDIUM  # 1.5% of rows


def test_leakage_detected_only_for_post_outcome_feature():
    n = 3000
    y = rng.binomial(1, 0.1, n)
    df = pd.DataFrame(
        {
            "honest": rng.normal(size=n) + 0.5 * y,
            "recovery": np.where(y == 1, rng.uniform(100, 500, n), 0.0),
            "target": y,
        }
    )
    issues, metrics = credit.target_leakage(df, ["honest", "recovery"], "target", TH)
    assert len(issues) == 1 and issues[0].affected == ["recovery"]
    assert metrics["univariate_auc"]["honest"] < TH.leakage_auc


def test_non_binary_target():
    df = pd.DataFrame({"target": [0, 1, 2, 1]})
    (issue,) = credit.target_validity(df, "target", "development")
    assert issue.severity is Severity.HIGH


def test_thin_defaults_flagged():
    train = pd.DataFrame({"target": [1] * 20 + [0] * 980})
    test = pd.DataFrame({"target": [1] * 15 + [0] * 485})
    issues, metrics = credit.event_counts(train, test, "target", TH)
    assert any(i.check == "event_counts" for i in issues)
    assert metrics["train_events"] == 20
