"""Assumption checks: does the model's specification hold up?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import expit

from mva import stats
from mva.checks.base import Issue, rnd
from mva.config import AssumptionThresholds
from mva.models.credit import CreditModelSpec
from mva.schemas import Severity, SR117Pillar

CS = SR117Pillar.CONCEPTUAL_SOUNDNESS


def _p_severity(p: float, th: AssumptionThresholds, material: Severity = Severity.MEDIUM) -> Severity | None:
    if p < th.p_material:
        return material
    if p < th.p_low:
        return Severity.LOW
    return None


# ===========================================================================
# Credit PD (logistic regression)
# ===========================================================================
def credit_replication(model: CreditModelSpec, X: pd.DataFrame, y: np.ndarray, th: AssumptionThresholds):
    """Re-estimate the developer's specification and compare predicted PDs."""
    res = stats.fit_logit(X.to_numpy(), y)
    beta = np.array([model.coefficients[f] for f in model.features])
    pd_model = expit(model.intercept + X.to_numpy() @ beta)
    pd_refit = expit(res.intercept + X.to_numpy() @ res.coef)
    max_diff = float(np.max(np.abs(pd_model - pd_refit)))
    metrics = {
        "refit_converged": res.converged,
        "refit_coefficients": {f: rnd(c, 6) for f, c in zip(model.features, res.coef)},
        "refit_z": {f: rnd(z, 2) for f, z in zip(model.features, res.z)},
        "max_pd_difference": rnd(max_diff, 6),
    }
    issues = []
    if max_diff > th.replication_tolerance or not res.converged:
        issues.append(
            Issue(
                check="replication",
                title="Model could not be independently replicated",
                severity=Severity.HIGH,
                pillar=CS,
                description=(
                    f"Re-estimating the documented specification on the development sample gives PDs that differ "
                    f"from the delivered model by up to {max_diff:.4f} (tolerance {th.replication_tolerance})."
                ),
                recommendation=(
                    "Obtain the developer's exact estimation code, sample filters and preprocessing; "
                    "the delivered coefficients cannot be relied on until they are reproduced."
                ),
                evidence={"max_pd_difference": rnd(max_diff, 6), "tolerance": th.replication_tolerance},
            )
        )
    return issues, metrics, res


def credit_multicollinearity(X: pd.DataFrame, th: AssumptionThresholds):
    v = {k: rnd(val, 2) for k, val in stats.vif(X).items()}
    issues = []
    for sev, lo, hi in ((Severity.HIGH, th.vif_high, np.inf), (Severity.MEDIUM, th.vif_medium, th.vif_high)):
        cols = {c: x for c, x in v.items() if lo < x <= hi}
        if cols:
            corr = X[list(cols)].corr().round(3) if len(cols) > 1 else None
            pairs = {}
            if corr is not None:
                for i, a in enumerate(corr.columns):
                    for b in corr.columns[i + 1:]:
                        pairs[f"{a} ~ {b}"] = float(corr.loc[a, b])
            issues.append(
                Issue(
                    check="multicollinearity",
                    title="Multicollinearity among model features" + (" (severe)" if sev is Severity.HIGH else ""),
                    severity=sev,
                    pillar=CS,
                    description=(
                        "Variance inflation factors above "
                        f"{lo:g}: " + ", ".join(f"{c} (VIF {x:.1f})" for c, x in cols.items()) + ". "
                        "Coefficients of collinear features are unstable and their signs cannot be interpreted."
                    ),
                    recommendation=(
                        "Keep one feature from each collinear group (or combine them), re-estimate, and confirm the "
                        "remaining coefficients have the expected signs."
                    ),
                    evidence={"vif": cols, "threshold": lo, "correlations": pairs},
                    affected=list(cols),
                )
            )
    return issues, {"vif": v}


def credit_signs(model: CreditModelSpec) -> list[Issue]:
    wrong = {}
    for f, expected in model.expected_signs.items():
        c = model.coefficients[f]
        if (expected == "+" and c < 0) or (expected == "-" and c > 0):
            wrong[f] = {"expected": expected, "coefficient": rnd(c, 6)}
    if not wrong:
        return []
    return [
        Issue(
            check="coefficient_signs",
            title="Coefficients contradict documented business expectations",
            severity=Severity.HIGH,
            pillar=CS,
            description=(
                "The delivered model has the opposite sign to the documented expectation for: "
                + ", ".join(f"{f} (expected {v['expected']}, estimated {v['coefficient']:+.4g})" for f, v in wrong.items())
                + "."
            ),
            recommendation=(
                "Explain each counter-intuitive sign or re-specify the model; a wrong sign usually indicates "
                "collinearity, omitted variables or a data issue, and will produce unexplainable decisions."
            ),
            evidence={"wrong_signs": wrong},
            affected=list(wrong),
        )
    ]


def credit_significance(model: CreditModelSpec, res: stats.LogitResult) -> list[Issue]:
    weak = {f: rnd(z, 2) for f, z in zip(model.features, res.z) if abs(z) < 1.96}
    if not weak:
        return []
    return [
        Issue(
            check="significance",
            title="Features without statistically significant effect",
            severity=Severity.LOW,
            pillar=CS,
            description=(
                "In the independent re-estimation these features are not significant at 5% (|z| < 1.96): "
                + ", ".join(f"{f} (z = {z})" for f, z in weak.items()) + "."
            ),
            recommendation="Document the business rationale for retaining them or remove them to simplify the model.",
            evidence={"z_statistics": weak},
            affected=list(weak),
        )
    ]


def credit_stability(model: CreditModelSpec, dev: stats.LogitResult, X_val: pd.DataFrame, y_val: np.ndarray):
    """Re-estimate on the validation sample; a significant effect that flips sign is unstable."""
    val = stats.fit_logit(X_val.to_numpy(), y_val)
    flips = {}
    for f, cd, zd, cv in zip(model.features, dev.coef, dev.z, val.coef):
        if abs(zd) >= 1.96 and np.sign(cd) != np.sign(cv):
            flips[f] = {"development": rnd(cd, 6), "validation": rnd(cv, 6)}
    metrics = {"validation_refit_coefficients": {f: rnd(c, 6) for f, c in zip(model.features, val.coef)}}
    if not flips:
        return [], metrics
    return [
        Issue(
            check="coefficient_stability",
            title="Coefficients change sign between development and validation samples",
            severity=Severity.MEDIUM,
            pillar=CS,
            description=(
                "Features that are significant in development but have the opposite sign when the same "
                "specification is estimated on the validation sample: " + ", ".join(flips) + "."
            ),
            recommendation="Investigate the instability (collinearity, population change) before relying on these effects.",
            evidence={"sign_flips": flips},
            affected=list(flips),
        )
    ], metrics


def credit_linearity(model: CreditModelSpec, X: pd.DataFrame, y: np.ndarray, base_loglik: float, th: AssumptionThresholds):
    """Is the log-odds linear in each continuous feature? LR test for an added squared term.

    One test per feature, so p-values are Bonferroni-adjusted for the number of features tested.
    """
    raw = {}
    for f in model.features:
        if X[f].nunique() <= 10:
            continue
        z = (X[f] - X[f].mean()) / X[f].std()
        X2 = np.column_stack([X.to_numpy(), z.to_numpy() ** 2])
        raw[f] = stats.lr_test(base_loglik, stats.fit_logit(X2, y).loglik, df=1)
    pvals = {f: min(1.0, p * len(raw)) for f, p in raw.items()}
    issues = []
    for sev in (Severity.MEDIUM, Severity.LOW):
        cols = {f: rnd(p, 5) for f, p in pvals.items() if _p_severity(p, th) is sev}
        if cols:
            issues.append(
                Issue(
                    check="linearity",
                    title="Non-linear relationship with log-odds",
                    severity=sev,
                    pillar=CS,
                    description=(
                        "Adding a squared term significantly improves fit (Bonferroni-adjusted) for: "
                        + ", ".join(f"{f} (adjusted p = {p:.2g})" for f, p in cols.items())
                        + ". This contradicts the documented assumption that log-odds are linear in the features."
                    ),
                    recommendation="Consider binning (WoE), a transformation or a spline for these features.",
                    evidence={"lr_test_p_values": cols},
                    affected=list(cols),
                )
            )
    return issues, {"linearity_p_values": {f: rnd(p, 5) for f, p in raw.items()},
                    "linearity_p_values_adjusted": {f: rnd(p, 5) for f, p in pvals.items()}}


# ===========================================================================
# Market VaR
# ===========================================================================
def var_independence(returns: pd.Series, th: AssumptionThresholds):
    q_r, p_r = stats.ljung_box(returns.to_numpy(), th.ljung_box_lags)
    q_s, p_s = stats.ljung_box(returns.to_numpy() ** 2, th.ljung_box_lags)
    metrics = {"ljung_box_returns": {"q": rnd(q_r, 2), "p": rnd(p_r, 5)},
               "ljung_box_squared_returns": {"q": rnd(q_s, 2), "p": rnd(p_s, 5)}}
    issues = []
    sev = _p_severity(p_r, th)
    if sev:
        issues.append(
            Issue(
                check="serial_correlation",
                title="Portfolio returns are serially correlated",
                severity=sev,
                pillar=CS,
                description=(
                    f"Ljung-Box test on returns ({th.ljung_box_lags} lags): Q = {q_r:.1f}, p = {p_r:.4f}. "
                    "The model assumes daily returns are independent."
                ),
                recommendation="Assess the impact on VaR, and on any square-root-of-time scaling to longer horizons.",
                evidence=metrics["ljung_box_returns"],
            )
        )
    sev = _p_severity(p_s, th)
    if sev:
        issues.append(
            Issue(
                check="volatility_clustering",
                title="Volatility clustering in portfolio returns",
                severity=sev,
                pillar=CS,
                description=(
                    f"Ljung-Box test on squared returns: Q = {q_s:.1f}, p = {p_s:.4f}. Volatility is persistent, "
                    "so an equally weighted historical window reacts slowly when market conditions change."
                ),
                recommendation=(
                    "Benchmark against a volatility-weighted (filtered) historical simulation or EWMA approach; "
                    "monitor exception clustering in backtesting."
                ),
                evidence=metrics["ljung_box_squared_returns"],
            )
        )
    return issues, metrics


def var_distribution(returns: pd.Series, method: str, th: AssumptionThresholds):
    jb, p = sps.jarque_bera(returns.to_numpy())
    metrics = {"skewness": rnd(sps.skew(returns), 3), "excess_kurtosis": rnd(sps.kurtosis(returns), 3),
               "jarque_bera": {"stat": rnd(jb, 2), "p": rnd(p, 5)}}
    if method != "parametric" or p >= th.p_material:
        return [], metrics
    return [
        Issue(
            check="normality",
            title="Normal-distribution assumption rejected",
            severity=Severity.HIGH,
            pillar=CS,
            description=(
                f"Parametric VaR assumes normal returns, but Jarque-Bera rejects normality (p = {p:.2g}; "
                f"excess kurtosis {metrics['excess_kurtosis']}). Tail losses will be understated."
            ),
            recommendation="Use a fat-tailed distribution (e.g. Student-t) or historical simulation.",
            evidence=metrics,
        )
    ], metrics


def var_regime(returns: pd.Series, window: int, th: AssumptionThresholds):
    recent = returns.iloc[-th.regime_window:]
    earlier = returns.iloc[-window:-th.regime_window]
    if len(earlier) < 20:
        return [], {}
    ratio = float(recent.std() / earlier.std())
    metrics = {"recent_vs_window_vol_ratio": rnd(ratio, 3), "recent_days": th.regime_window}
    if ratio > th.regime_ratio:
        sev, direction = Severity.MEDIUM, "higher"
    elif ratio < 1 / th.regime_ratio:
        sev, direction = Severity.LOW, "lower"
    else:
        return [], metrics
    return [
        Issue(
            check="regime_representativeness",
            title="Recent volatility differs from the VaR window",
            severity=sev,
            pillar=CS,
            description=(
                f"Volatility over the last {th.regime_window} days is {ratio:.2f}x the rest of the "
                f"{window}-day window ({direction}). The documented assumption that the window represents "
                "next-day risk is weakened."
            ),
            recommendation=(
                "Compare against a shorter or volatility-weighted window; if volatility is higher, VaR is likely "
                "understated until the window catches up."
            ),
            evidence=metrics,
        )
    ], metrics
