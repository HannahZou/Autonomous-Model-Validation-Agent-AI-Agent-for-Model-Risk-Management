import numpy as np
import pandas as pd

from mva.checks import market
from mva.config import DataQualityThresholds
from mva.schemas import Severity

TH = DataQualityThresholds()


def _prices(n=300):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-02", periods=n)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": dates, "A": p.round(4)})


def test_clean_series_has_no_issues():
    df = _prices()
    assert market.date_integrity(df, "date", TH) == []
    assert market.price_validity(df, ["A"], TH) == []
    assert market.extreme_returns(df, "date", ["A"], TH) == []


def test_duplicate_dates_high():
    df = _prices()
    df = pd.concat([df, df.iloc[[10]]]).sort_values("date")
    issues = market.date_integrity(df, "date", TH)
    assert [i.check for i in issues] == ["duplicate_dates"]
    assert issues[0].severity is Severity.HIGH


def test_calendar_gap():
    df = _prices().drop(index=range(50, 60))
    (issue,) = market.date_integrity(df, "date", TH)
    assert issue.check == "calendar_gaps"
    assert issue.evidence["gaps"][0]["days"] > TH.max_calendar_gap_days


def test_unsorted_dates():
    df = _prices().iloc[::-1]
    assert any(i.check == "date_order" for i in market.date_integrity(df, "date", TH))


def test_stale_and_nonpositive_prices():
    df = _prices()
    df.loc[100:107, "A"] = df.loc[99, "A"]
    df.loc[200, "A"] = 0.0
    checks = {i.check: i for i in market.price_validity(df, ["A"], TH)}
    assert checks["stale_prices"].evidence["stale"]["A"]["longest_days"] == 9
    assert checks["nonpositive_prices"].severity is Severity.HIGH


def test_unadjusted_split_flagged():
    df = _prices()
    df.loc[150:, "A"] = df.loc[150:, "A"] / 2
    (issue,) = market.extreme_returns(df, "date", ["A"], TH)
    assert issue.severity is Severity.HIGH
    assert issue.evidence["extreme_returns"]["A"][0]["return"] < -0.45


def test_history_length():
    issues, m = market.history_length(_prices(200), "date", window=250)
    assert issues[0].severity is Severity.HIGH
    issues, m = market.history_length(_prices(400), "date", window=250)
    assert issues[0].severity is Severity.MEDIUM
    issues, m = market.history_length(_prices(600), "date", window=250)
    assert issues == [] and m["backtest_days_available"] == 349
