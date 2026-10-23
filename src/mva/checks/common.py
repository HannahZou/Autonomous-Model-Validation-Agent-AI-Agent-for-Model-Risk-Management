"""Checks that apply to any tabular dataset."""

from __future__ import annotations

import pandas as pd

from mva.checks.base import Issue, rnd
from mva.config import DataQualityThresholds
from mva.schemas import Severity, SR117Pillar

CS = SR117Pillar.CONCEPTUAL_SOUNDNESS


def missing_values(df: pd.DataFrame, columns: list[str], label: str, th: DataQualityThresholds) -> list[Issue]:
    rates = df[columns].isna().mean()
    rates = rates[rates > 0]
    buckets: dict[Severity, dict[str, float]] = {}
    for col, rate in rates.items():
        if rate > th.missing_high:
            sev = Severity.HIGH
        elif rate > th.missing_medium:
            sev = Severity.MEDIUM
        else:
            sev = Severity.LOW
        buckets.setdefault(sev, {})[col] = rnd(rate)

    issues = []
    for sev, cols in buckets.items():
        names = ", ".join(f"{c} ({r:.1%})" for c, r in cols.items())
        issues.append(
            Issue(
                check="missing_values",
                title=f"Missing values in {label} data",
                severity=sev,
                pillar=CS,
                description=f"{len(cols)} column(s) in the {label} sample contain missing values: {names}.",
                recommendation=(
                    "Confirm the root cause of missingness (system outage, optional field, data feed), "
                    "document the treatment applied by the developer, and test whether missingness is "
                    "correlated with the target or model output."
                ),
                evidence={"dataset": label, "missing_rate": cols},
                affected=list(cols),
            )
        )
    return issues


def duplicate_rows(
    df: pd.DataFrame, label: str, th: DataQualityThresholds, id_column: str | None = None
) -> list[Issue]:
    issues = []
    n = len(df)
    dup_full = int(df.duplicated().sum())
    if dup_full:
        rate = dup_full / n
        issues.append(
            Issue(
                check="duplicate_rows",
                title=f"Duplicate records in {label} data",
                severity=Severity.MEDIUM if rate > th.duplicate_medium else Severity.LOW,
                pillar=CS,
                description=f"{dup_full} of {n} rows ({rate:.2%}) in the {label} sample are exact duplicates.",
                recommendation=(
                    "Remove duplicates before estimation and re-run development tests; duplicated records "
                    "overweight some observations and understate standard errors."
                ),
                evidence={"dataset": label, "duplicate_rows": dup_full, "rows": n, "rate": rnd(rate)},
            )
        )
    if id_column and id_column in df.columns:
        dup_ids = int(df[id_column].duplicated().sum())
        if dup_ids > dup_full:  # IDs repeated with different content are worse than exact copies
            issues.append(
                Issue(
                    check="duplicate_ids",
                    title=f"Conflicting records share an ID in {label} data",
                    severity=Severity.HIGH,
                    pillar=CS,
                    description=(
                        f"{dup_ids - dup_full} {id_column} value(s) appear on rows with different contents "
                        f"in the {label} sample."
                    ),
                    recommendation="Reconcile with the source system to establish the record of truth for each ID.",
                    evidence={"dataset": label, "id_column": id_column, "conflicting_ids": dup_ids - dup_full},
                    affected=[id_column],
                )
            )
    return issues


def constant_columns(df: pd.DataFrame, columns: list[str], label: str) -> list[Issue]:
    const = [c for c in columns if df[c].nunique(dropna=True) <= 1]
    if not const:
        return []
    return [
        Issue(
            check="constant_columns",
            title=f"Constant features in {label} data",
            severity=Severity.MEDIUM,
            pillar=CS,
            description=f"Feature(s) {', '.join(const)} take a single value in the {label} sample and carry no information.",
            recommendation="Remove the feature(s) or explain why they are retained; check for a broken data feed.",
            evidence={"dataset": label, "columns": const},
            affected=const,
        )
    ]
