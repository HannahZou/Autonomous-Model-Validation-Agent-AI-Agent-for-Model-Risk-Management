"""Sensitivity analysis: how do outputs respond to inputs, parameters and model choices?

SR 11-7 lists sensitivity analysis under evaluation of conceptual soundness:
checking the impact of small changes in inputs and parameter values, and
stressing the model to find the limits of its reliability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from mva import stats
from mva.checks.base import Issue, rnd
from mva.config import SensitivityThresholds
from mva.models import market as mk
from mva.models.credit import CreditModelSpec
from mva.schemas import Severity, SR117Pillar

CS = SR117Pillar.CONCEPTUAL_SOUNDNESS


# ===========================================================================
# Credit PD
# ===========================================================================
def _mean_pd(model: CreditModelSpec, X: pd.DataFrame, intercept=None, beta=None) -> float:
    b = np.array([model.coefficients[f] for f in model.features]) if beta is None else beta
    a = model.intercept if intercept is None else intercept
    return float(expit(a + X.to_numpy() @ b).mean())


def robust_sd(x: pd.Series) -> float:
    """IQR-based standard deviation, so shock sizes are not inflated by data errors;
    falls back to the ordinary SD for discrete variables with a zero IQR."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr_sd = (q3 - q1) / 1.349
    return float(iqr_sd) if iqr_sd > 0 else float(x.std())


def credit_feature_shocks(model: CreditModelSpec, X_dev: pd.DataFrame, X_val: pd.DataFrame, th: SensitivityThresholds):
    """One-at-a-time shocks of +/- `shock_sd` (robust) development standard deviations in model-input space."""
    base = _mean_pd(model, X_val)
    table: dict[str, dict] = {}
    for f in model.features:
        if X_dev[f].nunique() <= 2:
            continue
        sd = robust_sd(X_dev[f])
        up, down = X_val.copy(), X_val.copy()
        up[f] += th.shock_sd * sd
        down[f] -= th.shock_sd * sd
        pu, pdn = _mean_pd(model, up), _mean_pd(model, down)
        table[f] = {"shock": rnd(th.shock_sd * sd, 6), "mean_pd_up": rnd(pu, 5), "mean_pd_down": rnd(pdn, 5),
                    "relative_impact": rnd((abs(pu - base) + abs(pdn - base)) / 2 / base, 4)}
    total = sum(v["relative_impact"] for v in table.values()) or 1.0
    for v in table.values():
        v["share"] = rnd(v["relative_impact"] / total, 3)
    metrics = {"base_mean_pd": rnd(base, 5), "feature_shocks": table}
    top, info = max(table.items(), key=lambda kv: kv[1]["share"]) if table else (None, None)
    if top is None or info["share"] <= th.driver_concentration:
        return [], metrics
    return [
        Issue(
            check="driver_concentration",
            title="Model output is dominated by a single driver",
            severity=Severity.LOW,
            pillar=CS,
            description=(
                f"A {th.shock_sd:g}-standard-deviation shock to {top} moves the portfolio's mean PD by "
                f"{info['relative_impact']:.1%}, {info['share']:.0%} of the combined sensitivity of all features."
            ),
            recommendation=(
                f"Confirm the strength of {top} is justified and monitor its data quality closely, since errors "
                "in this one input drive most of the model's output."
            ),
            evidence={"feature": top, **info},
            affected=[top],
        )
    ], metrics


def credit_adverse_scenario(model: CreditModelSpec, X_dev: pd.DataFrame, X_val: pd.DataFrame, th: SensitivityThresholds):
    """Move every feature with a documented expected sign in its risk-increasing direction at once."""
    if not model.expected_signs:
        return [], {}
    base = _mean_pd(model, X_val)
    stressed = X_val.copy()
    moves = {}
    for f, sign in model.expected_signs.items():
        step = th.adverse_sd * robust_sd(X_dev[f]) * (1 if sign == "+" else -1)
        stressed[f] += step
        moves[f] = rnd(step, 6)
    s = _mean_pd(model, stressed)
    metrics = {"adverse_scenario": {"moves": moves, "base_mean_pd": rnd(base, 5), "stressed_mean_pd": rnd(s, 5),
                                    "relative_change": rnd(s / base - 1, 4)}}
    if s > base:
        return [], metrics
    return [
        Issue(
            check="adverse_scenario_response",
            title="PDs fall under an adverse scenario",
            severity=Severity.HIGH,
            pillar=CS,
            description=(
                f"Moving every feature {th.adverse_sd:g} standard deviations in its documented risk-increasing "
                f"direction lowers the mean PD from {base:.2%} to {s:.2%}. The model's response contradicts "
                "its own documented expectations."
            ),
            recommendation="Re-specify the model so that its joint response to deteriorating inputs is intuitive.",
            evidence=metrics["adverse_scenario"],
        )
    ], metrics


def credit_parameter_uncertainty(model: CreditModelSpec, X_dev: pd.DataFrame, y_dev: np.ndarray,
                                 X_val: pd.DataFrame, th: SensitivityThresholds):
    """Propagate estimation uncertainty: draw parameters from their asymptotic distribution
    (centred on the delivered model) and record the portfolio mean PD for each draw."""
    res = stats.fit_logit(X_dev.to_numpy(), y_dev)
    centre = np.array([model.intercept] + [model.coefficients[f] for f in model.features])
    rng = np.random.default_rng(th.seed)
    draws = rng.multivariate_normal(centre, res.cov, size=th.parameter_draws, method="cholesky")
    X = X_val.to_numpy()
    pds = expit(draws[:, :1] + draws[:, 1:] @ X.T).mean(axis=1)
    base = _mean_pd(model, X_val)
    lo, hi = np.percentile(pds, [5, 95])
    width = (hi - lo) / base
    metrics = {"parameter_uncertainty": {"draws": th.parameter_draws, "mean_pd": rnd(base, 5),
                                         "pd_90_interval": [rnd(lo, 5), rnd(hi, 5)], "relative_width": rnd(width, 4)}}
    if width <= th.parameter_width_low:
        return [], metrics
    sev = Severity.MEDIUM if width > th.parameter_width_medium else Severity.LOW
    return [
        Issue(
            check="parameter_uncertainty",
            title="Estimation uncertainty has a material effect on portfolio PD",
            severity=sev,
            pillar=CS,
            description=(
                f"Across {th.parameter_draws} draws of the coefficients from their estimated sampling "
                f"distribution, the validation portfolio's mean PD ranges from {lo:.2%} to {hi:.2%} "
                f"(90% interval, {width:.0%} of the point estimate)."
            ),
            recommendation="Consider a conservative margin of estimation uncertainty or a larger development sample.",
            evidence=metrics["parameter_uncertainty"],
        )
    ], metrics


# ===========================================================================
# Market VaR
# ===========================================================================
def var_window_sensitivity(returns: pd.Series, window: int, confidence: float, method: str, th: SensitivityThresholds):
    current = mk.latest_var(returns, window, confidence, method)
    by_window = {w: mk.latest_var(returns, w, confidence, method) for w in th.windows if w <= len(returns)}
    by_window.setdefault(window, current)
    by_conf = {f"{c:.1%}": rnd(mk.latest_var(returns, window, c, method), 5) for c in (0.975, 0.99, 0.995)}
    es = mk.expected_shortfall(returns, window, 0.975)
    spread = (max(by_window.values()) - min(by_window.values())) / current
    metrics = {
        "current_var": rnd(current, 5),
        "var_by_window": {str(w): rnd(v, 5) for w, v in sorted(by_window.items())},
        "window_dispersion": rnd(spread, 3),
        "var_by_confidence": by_conf,
        "expected_shortfall_97_5": rnd(es, 5),
        "es_97_5_to_var_ratio": rnd(es / current, 3),
    }
    if spread <= th.window_dispersion:
        return [], metrics
    lo_w = min(by_window, key=by_window.get)
    hi_w = max(by_window, key=by_window.get)
    return [
        Issue(
            check="window_sensitivity",
            title="VaR is highly sensitive to the look-back window",
            severity=Severity.LOW,
            pillar=CS,
            description=(
                f"Current VaR ranges from {by_window[lo_w]:.2%} ({lo_w}-day window) to {by_window[hi_w]:.2%} "
                f"({hi_w}-day window), a spread of {spread:.0%} of the champion's {window}-day VaR. The choice of "
                "window, rather than market risk, drives a large part of the number."
            ),
            recommendation="Document the rationale for the window and report the range to risk committees.",
            evidence={k: metrics[k] for k in ("var_by_window", "window_dispersion")},
        )
    ], metrics


def var_contributions(returns_by_asset: pd.DataFrame, weights: dict[str, float], window: int,
                      confidence: float, th: SensitivityThresholds, k: int = 5):
    """Historical component VaR: average asset P&L on the k days nearest the VaR quantile."""
    r = returns_by_asset.iloc[-window:]
    w = pd.Series(weights)[r.columns]
    contrib = r * w
    port = contrib.sum(axis=1)
    q = np.quantile(port, 1 - confidence)
    nearest = (port - q).abs().nsmallest(k).index
    comp = contrib.loc[nearest].mean()
    share = comp / comp.sum()
    metrics = {"component_var_share": {a: rnd(v, 3) for a, v in share.items()}}
    top = share.idxmax()
    if share[top] <= th.asset_concentration:
        return [], metrics
    return [
        Issue(
            check="risk_concentration",
            title="VaR is concentrated in one position",
            severity=Severity.LOW,
            pillar=CS,
            description=f"{top} contributes {share[top]:.0%} of portfolio VaR (historical component VaR).",
            recommendation="Make sure the position's price history receives the closest data-quality scrutiny.",
            evidence=metrics["component_var_share"],
            affected=[top],
        )
    ], metrics


def var_stress(returns: pd.Series, current_var: float, weights: dict[str, float],
               scenarios: dict[str, dict[str, float]], th: SensitivityThresholds):
    worst_day = returns.idxmin()
    worst = float(-returns.min())
    hypo = {
        name: {"loss": rnd(-sum(weights[a] * s for a, s in shocks.items()), 5),
               "multiple_of_var": rnd(-sum(weights[a] * s for a, s in shocks.items()) / current_var, 2)}
        for name, shocks in scenarios.items()
    }
    metrics = {"worst_historical_day": {"date": worst_day.strftime("%Y-%m-%d"), "loss": rnd(worst, 5),
                                        "multiple_of_var": rnd(worst / current_var, 2)},
               "hypothetical_scenarios": hypo}
    if worst / current_var <= th.stress_multiple:
        return [], metrics
    return [
        Issue(
            check="stress_vs_var",
            title="Historical worst day far exceeds current VaR",
            severity=Severity.LOW,
            pillar=CS,
            description=(
                f"The largest one-day loss in the history, {worst:.2%} on {metrics['worst_historical_day']['date']}, "
                f"is {worst / current_var:.1f}x the current VaR. Either the history contains a genuine tail event "
                "that VaR at this confidence level will not reflect, or the observation is a data error "
                "(check the Data Quality findings)."
            ),
            recommendation="Confirm the observation; if genuine, cover it through stress testing alongside VaR.",
            evidence=metrics["worst_historical_day"],
        )
    ], metrics
