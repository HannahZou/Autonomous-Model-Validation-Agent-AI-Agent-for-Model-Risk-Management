"""Data quality checks specific to market risk (VaR) price histories."""

from __future__ import annotations

import pandas as pd

from mva.checks.base import Issue, rnd
from mva.config import DataQualityThresholds
from mva.schemas import Severity, SR117Pillar

CS = SR117Pillar.CONCEPTUAL_SOUNDNESS
OA = SR117Pillar.OUTCOMES_ANALYSIS

# Trading days of out-of-sample history needed on top of the VaR window for a
# one-year backtest (Basel traffic-light uses 250 observations).
BACKTEST_DAYS = 250


def date_integrity(df: pd.DataFrame, date_col: str, th: DataQualityThresholds) -> list[Issue]:
    issues = []
    dates = df[date_col]
    n_bad = int(dates.isna().sum())
    if n_bad:
        issues.append(
            Issue(
                check="date_integrity",
                title="Unparseable dates in price history",
                severity=Severity.HIGH,
                pillar=CS,
                description=f"{n_bad} row(s) have a date that could not be parsed.",
                recommendation="Fix the date format at source; rows without a valid date cannot be ordered.",
                evidence={"unparseable_dates": n_bad},
                affected=[date_col],
            )
        )
    dup = dates[dates.duplicated(keep=False) & dates.notna()]
    if len(dup):
        uniq = sorted({d.strftime("%Y-%m-%d") for d in dup})
        issues.append(
            Issue(
                check="duplicate_dates",
                title="Duplicate dates in price history",
                severity=Severity.HIGH,
                pillar=CS,
                description=f"{len(uniq)} date(s) appear more than once (e.g. {', '.join(uniq[:5])}), producing zero or double-counted returns.",
                recommendation="De-duplicate the time series and confirm which observation is the official close.",
                evidence={"duplicate_dates": uniq[:20], "count": len(uniq)},
                affected=[date_col],
            )
        )
    valid = dates.dropna()
    if not valid.is_monotonic_increasing:
        issues.append(
            Issue(
                check="date_order",
                title="Price history is not in date order",
                severity=Severity.LOW,
                pillar=CS,
                description="Rows are not sorted by date; returns computed in file order would be wrong.",
                recommendation="Sort by date in the data pipeline before computing returns.",
                evidence={},
                affected=[date_col],
            )
        )
    gaps = valid.drop_duplicates().sort_values().diff().dt.days
    big = gaps[gaps > th.max_calendar_gap_days]
    if len(big):
        ends = valid.drop_duplicates().sort_values()
        listed = [
            {"from": ends.iloc[ends.index.get_loc(i) - 1].strftime("%Y-%m-%d"), "to": ends.loc[i].strftime("%Y-%m-%d"), "days": int(g)}
            for i, g in big.items()
        ]
        issues.append(
            Issue(
                check="calendar_gaps",
                title="Gaps in price history",
                severity=Severity.MEDIUM,
                pillar=CS,
                description=(
                    f"{len(listed)} gap(s) longer than {th.max_calendar_gap_days} calendar days, e.g. "
                    f"{listed[0]['from']} to {listed[0]['to']}. Returns spanning a gap overstate one-day moves."
                ),
                recommendation="Back-fill the missing history from an alternate source or exclude returns that span gaps.",
                evidence={"gaps": listed[:10], "threshold_days": th.max_calendar_gap_days},
                affected=[date_col],
            )
        )
    return issues


def price_validity(df: pd.DataFrame, assets: list[str], th: DataQualityThresholds) -> list[Issue]:
    issues = []
    nonpos = {a: int((df[a] <= 0).sum()) for a in assets if (df[a] <= 0).any()}
    if nonpos:
        issues.append(
            Issue(
                check="nonpositive_prices",
                title="Zero or negative prices",
                severity=Severity.HIGH,
                pillar=CS,
                description=f"Non-positive prices found: {nonpos}. Log returns are undefined and simple returns are -100%.",
                recommendation="Replace with validated closes; if the series can legitimately be negative (e.g. spreads), model it in levels or changes, not returns.",
                evidence={"nonpositive_counts": nonpos},
                affected=list(nonpos),
            )
        )
    miss = {a: rnd(df[a].isna().mean()) for a in assets if df[a].isna().any()}
    if miss:
        worst = max(miss.values())
        sev = Severity.HIGH if worst > th.missing_high else Severity.MEDIUM if worst > th.missing_medium else Severity.LOW
        issues.append(
            Issue(
                check="missing_values",
                title="Missing prices",
                severity=sev,
                pillar=CS,
                description=f"Missing prices by asset (share of days): {miss}.",
                recommendation="Document the fill rule (e.g. previous close) and quantify its effect on VaR.",
                evidence={"missing_rate": miss},
                affected=list(miss),
            )
        )
    stale: dict[str, dict] = {}
    for a in assets:
        s = df[a]
        run_id = (s != s.shift()).cumsum()
        runs = s.groupby(run_id).agg(["size", "first"])
        long = runs[(runs["size"] >= th.stale_run_days) & runs["first"].notna()]
        if len(long):
            stale[a] = {"runs": int(len(long)), "longest_days": int(long["size"].max())}
    if stale:
        issues.append(
            Issue(
                check="stale_prices",
                title="Stale (unchanged) prices",
                severity=Severity.MEDIUM,
                pillar=CS,
                description=(
                    f"Prices unchanged for {th.stale_run_days}+ consecutive observations: {stale}. "
                    "Stale data understates volatility and VaR."
                ),
                recommendation="Check the market-data feed for the affected period and replace stale points with observed prices.",
                evidence={"stale": stale, "threshold_days": th.stale_run_days},
                affected=list(stale),
            )
        )
    return issues


def extreme_returns(df: pd.DataFrame, date_col: str, assets: list[str], th: DataQualityThresholds) -> list[Issue]:
    clean = df.dropna(subset=[date_col]).drop_duplicates(subset=[date_col]).sort_values(date_col)
    flagged: dict[str, list] = {}
    worst = 0.0
    for a in assets:
        p = clean[a].where(clean[a] > 0)
        r = p.pct_change(fill_method=None)
        hits = r[r.abs() > th.extreme_return]
        if len(hits):
            flagged[a] = [
                {"date": clean.loc[i, date_col].strftime("%Y-%m-%d"), "return": rnd(v)} for i, v in hits.items()
            ][:5]
            worst = max(worst, float(hits.abs().max()))
    if not flagged:
        return []
    example = next(iter(flagged.items()))
    return [
        Issue(
            check="extreme_returns",
            title="Implausible one-day price moves",
            severity=Severity.HIGH if worst >= th.extreme_return_high else Severity.MEDIUM,
            pillar=CS,
            description=(
                f"One-day moves larger than {th.extreme_return:.0%} found, e.g. {example[0]} "
                f"{example[1][0]['return']:+.1%} on {example[1][0]['date']}. A single move of this size "
                "usually signals an unadjusted corporate action (split, dividend) or a bad tick."
            ),
            recommendation=(
                "Verify against corporate-action records; use total-return adjusted prices. If the move is genuine, "
                "document it as a stress observation."
            ),
            evidence={"extreme_returns": flagged, "threshold": th.extreme_return},
            affected=list(flagged),
        )
    ]


def history_length(df: pd.DataFrame, date_col: str, window: int) -> tuple[list[Issue], dict]:
    n = int(df[date_col].dropna().nunique())
    returns = max(n - 1, 0)
    metrics = {"observations": n, "returns": returns, "var_window": window,
               "backtest_days_available": max(returns - window, 0)}
    if returns < window:
        sev, msg = Severity.HIGH, f"Only {returns} returns available but the VaR window needs {window}."
    elif returns < window + BACKTEST_DAYS:
        sev, msg = Severity.MEDIUM, (
            f"{returns} returns leave {returns - window} out-of-window days; a one-year backtest needs "
            f"{BACKTEST_DAYS}."
        )
    else:
        return [], metrics
    return [
        Issue(
            check="history_length",
            title="Insufficient price history",
            severity=sev,
            pillar=OA,
            description=msg,
            recommendation="Source a longer history so the model can be estimated and backtested over at least one year.",
            evidence=metrics,
        )
    ], metrics
