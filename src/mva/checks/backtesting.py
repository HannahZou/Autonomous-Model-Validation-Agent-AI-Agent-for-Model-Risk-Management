"""Backtesting checks: do the model's outputs agree with what actually happened?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mva import stats
from mva.checks.base import Issue, rnd
from mva.config import BacktestThresholds
from mva.schemas import Severity, SR117Pillar

OA = SR117Pillar.OUTCOMES_ANALYSIS


# ===========================================================================
# Credit PD
# ===========================================================================
def discrimination(y: np.ndarray, pd_hat: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, pd_hat))
    return {"auc": rnd(auc), "gini": rnd(2 * auc - 1), "ks": rnd(stats.ks_statistic(pd_hat, y))}


def credit_discrimination(dev: dict, val: dict, th: BacktestThresholds) -> list[Issue]:
    issues = []
    auc = val["auc"]
    if auc < th.auc_medium:
        sev = Severity.HIGH if auc < th.auc_high else Severity.MEDIUM
        issues.append(
            Issue(
                check="discrimination",
                title="Weak discriminatory power on the validation sample",
                severity=sev,
                pillar=OA,
                description=(
                    f"Validation AUC is {auc:.3f} (Gini {val['gini']:.3f}, KS {val['ks']:.3f}), below the "
                    f"{th.auc_medium:.2f} benchmark for an application scorecard."
                ),
                recommendation="Review feature selection and segmentation; benchmark against a challenger model.",
                evidence={"validation": val, "benchmark_auc": th.auc_medium},
            )
        )
    drop = dev["auc"] - auc
    if drop > th.auc_drop:
        issues.append(
            Issue(
                check="discrimination_decay",
                title="Discriminatory power deteriorates out of sample",
                severity=Severity.MEDIUM,
                pillar=OA,
                description=f"AUC falls from {dev['auc']:.3f} in development to {auc:.3f} in validation (-{drop:.3f}).",
                recommendation="Check for over-fitting and population change; consider re-development.",
                evidence={"development": dev, "validation": val, "max_drop": th.auc_drop},
            )
        )
    return issues


def credit_calibration(y: np.ndarray, pd_hat: np.ndarray, th: BacktestThresholds):
    z, p = stats.calibration_in_the_large(pd_hat, y)
    hl, hl_p, table = stats.hosmer_lemeshow(pd_hat, y)
    observed, expected = int(y.sum()), float(pd_hat.sum())
    metrics = {
        "observed_defaults": observed,
        "expected_defaults": rnd(expected, 1),
        "observed_rate": rnd(y.mean()),
        "mean_predicted_pd": rnd(pd_hat.mean()),
        "calibration_z": rnd(z, 3),
        "calibration_p": rnd(p, 6),
        "hosmer_lemeshow": {"stat": rnd(hl, 2), "p": rnd(hl_p, 6)},
        "deciles": [
            {"decile": int(i) + 1, "n": int(r.n), "predicted_rate": rnd(r.predicted_rate), "observed_rate": rnd(r.observed_rate)}
            for i, r in table.iterrows()
        ],
    }
    issues = []
    if p < th.p_low:
        under = observed > expected
        sev = (Severity.HIGH if under else Severity.MEDIUM) if p < th.p_material else Severity.LOW
        issues.append(
            Issue(
                check="calibration_level",
                title="Model " + ("under-predicts" if under else "over-predicts") + " defaults",
                severity=sev,
                pillar=OA,
                description=(
                    f"{observed} defaults observed against {expected:.0f} predicted "
                    f"(observed rate {y.mean():.2%} vs mean PD {pd_hat.mean():.2%}; z = {z:.2f}, p = {p:.2g}). "
                    + ("Under-prediction is non-conservative: losses, pricing and reserves would be understated."
                       if under else "Over-prediction is conservative but may reject creditworthy applicants.")
                ),
                recommendation=(
                    "Recalibrate the intercept to the current default rate (or a through-the-cycle target) and "
                    "document the calibration target."
                ),
                evidence={k: metrics[k] for k in ("observed_defaults", "expected_defaults", "calibration_z", "calibration_p")},
            )
        )
    if hl_p < th.p_low:
        issues.append(
            Issue(
                check="calibration_fit",
                title="Predicted PDs do not match observed default rates across risk deciles",
                severity=Severity.MEDIUM if hl_p < th.p_material else Severity.LOW,
                pillar=OA,
                description=f"Hosmer-Lemeshow test on the validation sample: statistic {hl:.1f}, p = {hl_p:.2g}.",
                recommendation="Review the decile table for where the model mis-states risk; recalibrate by segment if needed.",
                evidence={"hosmer_lemeshow": metrics["hosmer_lemeshow"], "deciles": metrics["deciles"]},
            )
        )
    return issues, metrics


# ===========================================================================
# Market VaR
# ===========================================================================
def var_backtest(hits: pd.DataFrame, confidence: float, th: BacktestThresholds):
    """`hits` has columns return, var, exception (one row per out-of-sample day)."""
    p = 1 - confidence
    n, x = len(hits), int(hits["exception"].sum())
    lr_uc, p_uc = stats.kupiec_pof(x, n, p)
    lr_ind, p_ind, counts = stats.christoffersen_independence(hits["exception"].to_numpy())
    last = hits.iloc[-th.traffic_light_days:]
    x_last = int(last["exception"].sum())
    zone, cum = stats.traffic_light(x_last, len(last), p)
    exc_dates = [d.strftime("%Y-%m-%d") for d in hits.index[hits["exception"] == 1]]
    metrics = {
        "backtest_days": n,
        "exceptions": x,
        "expected_exceptions": rnd(n * p, 1),
        "kupiec": {"lr": rnd(lr_uc, 3), "p": rnd(p_uc, 6)},
        "christoffersen_independence": {"lr": rnd(lr_ind, 3), "p": rnd(p_ind, 6), **counts},
        "traffic_light": {"days": len(last), "exceptions": x_last, "zone": zone, "cumulative_probability": rnd(cum, 6)},
        "exception_dates": exc_dates[-20:],
    }
    issues = []
    if len(last) >= th.traffic_light_days and zone != "green":
        issues.append(
            Issue(
                check="traffic_light",
                title=f"Basel traffic light in the {zone} zone",
                severity=Severity.HIGH if zone == "red" else Severity.MEDIUM,
                pillar=OA,
                description=(
                    f"{x_last} exceptions in the last {len(last)} trading days against "
                    f"{len(last) * p:.1f} expected at {confidence:.0%} confidence "
                    f"(cumulative binomial probability {cum:.4%})."
                ),
                recommendation=(
                    "Investigate each exception (model vs. P&L issue), escalate per the backtesting policy, and "
                    "assess whether a capital multiplier add-on or model change is required."
                ),
                evidence=metrics["traffic_light"],
            )
        )
    if p_uc < th.p_low:
        too_many = x > n * p
        sev = (Severity.HIGH if too_many else Severity.LOW) if p_uc < th.p_material else Severity.LOW
        issues.append(
            Issue(
                check="unconditional_coverage",
                title="Exception frequency inconsistent with the VaR confidence level"
                + (" (too many)" if too_many else " (too few)"),
                severity=sev,
                pillar=OA,
                description=(
                    f"Kupiec POF test over {n} days: {x} exceptions vs {n * p:.1f} expected "
                    f"(LR = {lr_uc:.2f}, p = {p_uc:.2g})."
                    + (" VaR is understating risk." if too_many else " VaR is overly conservative.")
                ),
                recommendation=(
                    "Re-examine the window length, weighting and data treatment; compare with a challenger VaR."
                ),
                evidence={"exceptions": x, "expected": metrics["expected_exceptions"], **metrics["kupiec"]},
            )
        )
    if p_ind < th.p_low:
        issues.append(
            Issue(
                check="exception_clustering",
                title="VaR exceptions cluster in time",
                severity=Severity.MEDIUM if p_ind < th.p_material else Severity.LOW,
                pillar=OA,
                description=(
                    f"Christoffersen independence test: LR = {lr_ind:.2f}, p = {p_ind:.2g}; "
                    f"{counts['n11']} exception(s) followed another exception. The model is slow to react to "
                    "changing volatility."
                ),
                recommendation="Consider volatility scaling (EWMA / filtered historical simulation).",
                evidence=metrics["christoffersen_independence"],
            )
        )
    return issues, metrics
