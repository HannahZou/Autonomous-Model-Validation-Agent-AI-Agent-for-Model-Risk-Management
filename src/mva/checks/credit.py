"""Data quality checks specific to credit PD models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mva.checks.base import Issue, rnd
from mva.config import DataQualityThresholds
from mva.schemas import Severity, SR117Pillar

CS = SR117Pillar.CONCEPTUAL_SOUNDNESS
OM = SR117Pillar.ONGOING_MONITORING


# -- helpers ----------------------------------------------------------------
def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI with quantile bins from the expected (development) sample.

    Missing values are excluded; differences in missingness are reported by the
    missing-values check, so they are not double counted here.
    """
    exp = pd.to_numeric(expected, errors="coerce").dropna()
    act = pd.to_numeric(actual, errors="coerce").dropna()
    if exp.empty or act.empty:
        return 0.0
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(s: pd.Series) -> np.ndarray:
        counts = pd.cut(s, edges, include_lowest=True).value_counts(sort=False).to_numpy()
        return counts / len(s)

    e, a = dist(exp), dist(act)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def outlier_scores(values: pd.Series) -> pd.Series:
    """Median/MAD z-scores, oriented so that only implausible values score high.

    Non-negative, right-skewed variables (income, balances, tenure) are scored on
    a log1p scale and only their upper tail counts: a short tenure or a small
    income is not an outlier, an income of 100 million is. Other variables are
    scored on both tails.
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    upper_only = len(v) > 0 and (v >= 0).all() and v.skew() > 1
    if upper_only:
        v = np.log1p(v)
    med = v.median()
    mad = (v - med).abs().median() * 1.4826
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=v.index)
    z = (v - med) / mad
    return z.clip(lower=0) if upper_only else z.abs()


# -- checks -----------------------------------------------------------------
def target_validity(df: pd.DataFrame, target: str, label: str) -> list[Issue]:
    values = set(pd.unique(df[target].dropna()))
    issues = []
    if not values <= {0, 1}:
        issues.append(
            Issue(
                check="target_validity",
                title=f"Target is not binary in {label} data",
                severity=Severity.HIGH,
                pillar=CS,
                description=f"{target} contains values other than 0/1: {sorted(map(str, values))[:10]}.",
                recommendation="Confirm the default definition and recode the target before any further testing.",
                evidence={"dataset": label, "values": sorted(map(str, values))[:10]},
                affected=[target],
            )
        )
    n_na = int(df[target].isna().sum())
    if n_na:
        issues.append(
            Issue(
                check="target_validity",
                title=f"Missing target values in {label} data",
                severity=Severity.HIGH,
                pillar=CS,
                description=f"{n_na} row(s) in the {label} sample have no observed outcome.",
                recommendation="Exclude unresolved outcomes explicitly and document the exclusion rule.",
                evidence={"dataset": label, "missing_target": n_na},
                affected=[target],
            )
        )
    return issues


def event_counts(train: pd.DataFrame, test: pd.DataFrame, target: str, th: DataQualityThresholds):
    t = pd.to_numeric(train[target], errors="coerce")
    v = pd.to_numeric(test[target], errors="coerce")
    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_events": int(t.sum()),
        "test_events": int(v.sum()),
        "train_event_rate": rnd(t.mean()),
        "test_event_rate": rnd(v.mean()),
    }
    issues = []
    thin = {k: metrics[f"{k}_events"] for k in ("train", "test") if metrics[f"{k}_events"] < th.min_events}
    if thin:
        issues.append(
            Issue(
                check="event_counts",
                title="Too few defaults for reliable testing",
                severity=Severity.MEDIUM,
                pillar=SR117Pillar.OUTCOMES_ANALYSIS,
                description=(
                    f"Default counts below {th.min_events}: {thin}. Discrimination and calibration statistics "
                    "will have wide confidence intervals."
                ),
                recommendation="Extend the observation window, pool vintages, or report confidence intervals for all metrics.",
                evidence={**thin, "threshold": th.min_events},
            )
        )
    shift = abs(metrics["test_event_rate"] - metrics["train_event_rate"])
    if shift > th.event_rate_shift:
        issues.append(
            Issue(
                check="event_rate_shift",
                title="Default rate differs materially between samples",
                severity=Severity.MEDIUM,
                pillar=OM,
                description=(
                    f"Default rate moved from {metrics['train_event_rate']:.2%} (development) to "
                    f"{metrics['test_event_rate']:.2%} (validation)."
                ),
                recommendation="Assess whether the shift reflects the credit cycle or a population change; consider recalibration.",
                evidence={"train_rate": metrics["train_event_rate"], "test_rate": metrics["test_event_rate"]},
            )
        )
    return issues, metrics


def range_violations(df: pd.DataFrame, ranges: dict[str, tuple[float, float]], label: str) -> list[Issue]:
    bad: dict[str, dict] = {}
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mask = (s < lo) | (s > hi)
        if mask.any():
            bad[col] = {
                "count": int(mask.sum()),
                "allowed": [lo, hi],
                "observed_min": rnd(s.min()),
                "observed_max": rnd(s.max()),
            }
    if not bad:
        return []
    names = ", ".join(f"{c} ({v['count']} rows outside {v['allowed']})" for c, v in bad.items())
    return [
        Issue(
            check="range_violations",
            title=f"Values outside documented valid ranges in {label} data",
            severity=Severity.HIGH,
            pillar=CS,
            description=f"Impossible or undocumented values found: {names}.",
            recommendation=(
                "Trace the records to source, correct or exclude them, and add range validation to the data pipeline."
            ),
            evidence={"dataset": label, "violations": bad},
            affected=list(bad),
        )
    ]


def outliers(df: pd.DataFrame, features: list[str], label: str, th: DataQualityThresholds) -> list[Issue]:
    flagged: dict[str, dict] = {}
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 20:
            continue
        n_out = int((outlier_scores(df[col]) > th.outlier_z).sum())
        if n_out:
            flagged[col] = {"count": n_out, "share": rnd(n_out / len(df)), "max_value": rnd(df[col].max(), 2)}
    if not flagged:
        return []
    worst = max(v["share"] for v in flagged.values())
    names = ", ".join(f"{c} ({v['count']} rows)" for c, v in flagged.items())
    return [
        Issue(
            check="outliers",
            title=f"Extreme outliers in {label} data",
            severity=Severity.MEDIUM if worst > th.outlier_share_medium else Severity.LOW,
            pillar=CS,
            description=f"Values more than {th.outlier_z:g} robust standard deviations from the median: {names}.",
            recommendation=(
                "Determine whether these are data-entry errors or genuine extremes; document the capping or "
                "winsorisation rule and test model sensitivity to it."
            ),
            evidence={"dataset": label, "z_threshold": th.outlier_z, "features": flagged},
            affected=list(flagged),
        )
    ]


def distribution_shift(train: pd.DataFrame, test: pd.DataFrame, features: list[str], th: DataQualityThresholds):
    psi = {c: rnd(population_stability_index(train[c], test[c])) for c in features}
    issues = []
    for sev, lo, hi in ((Severity.HIGH, th.psi_high, np.inf), (Severity.MEDIUM, th.psi_medium, th.psi_high)):
        cols = {c: v for c, v in psi.items() if lo < v <= hi}
        if not cols:
            continue
        names = ", ".join(f"{c} (PSI {v:.3f})" for c, v in cols.items())
        issues.append(
            Issue(
                check="distribution_shift",
                title="Feature distribution shift between development and validation samples"
                + (" (significant)" if sev is Severity.HIGH else " (moderate)"),
                severity=sev,
                pillar=OM,
                description=f"Population Stability Index above {lo:g}: {names}.",
                recommendation=(
                    "Investigate drivers of the shift (policy, channel, macro) and confirm the model remains fit "
                    "for the current population; re-test performance on the shifted segment."
                ),
                evidence={"psi": cols, "threshold": lo},
                affected=list(cols),
            )
        )
    return issues, {"psi": psi}


def target_leakage(df: pd.DataFrame, features: list[str], target: str, th: DataQualityThresholds):
    y = pd.to_numeric(df[target], errors="coerce")
    auc: dict[str, float] = {}
    for col in features:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 50 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
            continue
        a = roc_auc_score(y[mask], x[mask])
        auc[col] = rnd(max(a, 1 - a))
    leaky = {c: a for c, a in auc.items() if a >= th.leakage_auc}
    issues = []
    if leaky:
        names = ", ".join(f"{c} (AUC {a:.3f})" for c, a in leaky.items())
        issues.append(
            Issue(
                check="target_leakage",
                title="Suspected target leakage",
                severity=Severity.HIGH,
                pillar=CS,
                description=(
                    f"On its own, {names} separates defaulters from non-defaulters almost perfectly. "
                    "A feature this predictive is usually only observable after the outcome."
                ),
                recommendation=(
                    "Confirm each feature's availability at the decision (application) date; remove any "
                    "post-outcome variable and re-estimate the model."
                ),
                evidence={"univariate_auc": leaky, "threshold": th.leakage_auc},
                affected=list(leaky),
            )
        )
    return issues, {"univariate_auc": auc}
