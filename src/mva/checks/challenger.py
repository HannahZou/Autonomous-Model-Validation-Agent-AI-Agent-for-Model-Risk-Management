"""Challenger benchmarking: does an alternative approach materially beat the champion?

SR 11-7 treats benchmarking against alternative models as part of ongoing
monitoring, and comparison with alternative approaches as part of conceptual
soundness. Challengers are built here by the validator, independently of the
developer, on the same information set as the champion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from mva import stats
from mva.checks.base import Issue, rnd
from mva.config import ChallengerThresholds
from mva.models import market as mk
from mva.schemas import Severity, SR117Pillar

OM = SR117Pillar.ONGOING_MONITORING
CS = SR117Pillar.CONCEPTUAL_SOUNDNESS


# ===========================================================================
# Credit PD
# ===========================================================================
def build_credit_challengers(seed: int) -> dict:
    return {
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.2, random_state=seed,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=seed,
        ),
    }


def paired_bootstrap_auc_gap(y, champion, challenger, n: int, seed: int) -> tuple[float, float]:
    """95% interval for AUC(challenger) - AUC(champion), resampling the same rows for both."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    gaps = []
    for _ in range(n):
        idx = rng.choice(idx_all, len(idx_all), replace=True)
        if y[idx].min() == y[idx].max():
            continue
        gaps.append(roc_auc_score(y[idx], challenger[idx]) - roc_auc_score(y[idx], champion[idx]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


def credit_benchmark(
    X_dev: pd.DataFrame, y_dev: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
    champion_val: np.ndarray, th: ChallengerThresholds,
):
    champ = {"auc": rnd(roc_auc_score(y_val, champion_val)), "brier": rnd(brier_score_loss(y_val, champion_val), 5)}
    results: dict[str, dict] = {"Champion (delivered model)": champ}
    issues: list[Issue] = []
    for name, est in build_credit_challengers(th.seed).items():
        est.fit(X_dev, y_dev)
        p = est.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        lo, hi = paired_bootstrap_auc_gap(y_val, champion_val, p, th.bootstrap_samples, th.seed)
        rho = float(sps.spearmanr(champion_val, p).statistic)
        results[name] = {
            "auc": rnd(auc), "brier": rnd(brier_score_loss(y_val, p), 5),
            "auc_gap": rnd(auc - champ["auc"]), "auc_gap_95ci": [rnd(lo), rnd(hi)],
            "rank_correlation": rnd(rho, 3),
        }
        gap = auc - champ["auc"]
        if gap > th.auc_gap_medium and lo > 0:
            sev = Severity.HIGH if gap > th.auc_gap_high else Severity.MEDIUM
            issues.append(
                Issue(
                    check="challenger_outperforms",
                    title=f"{name} challenger outperforms the champion",
                    severity=sev,
                    pillar=OM,
                    description=(
                        f"On the validation sample the {name.lower()} challenger reaches AUC {auc:.3f} vs "
                        f"{champ['auc']:.3f} for the delivered model (gap {gap:+.3f}, paired-bootstrap 95% "
                        f"interval {lo:+.3f} to {hi:+.3f}). The challenger uses the same features, so the "
                        "champion's functional form is leaving predictive information unused."
                    ),
                    recommendation=(
                        "Identify where the challenger gains (e.g. partial dependence of the main features) and "
                        "capture it in the champion through binning or transformations, or document why the "
                        "simpler model is preferred despite the gap."
                    ),
                    evidence=results[name],
                )
            )
        if rho < th.rank_correlation_low:
            issues.append(
                Issue(
                    check="challenger_rank_disagreement",
                    title=f"Champion and {name.lower()} challenger rank applicants differently",
                    severity=Severity.LOW,
                    pillar=OM,
                    description=(
                        f"Spearman rank correlation between champion and challenger PDs is {rho:.2f}. "
                        "Approval decisions near the cut-off would differ materially between the two models."
                    ),
                    recommendation="Review the applicants on which the models disagree most.",
                    evidence={"rank_correlation": rnd(rho, 3)},
                )
            )
    return issues, {"models": results}


# ===========================================================================
# Market VaR
# ===========================================================================
def var_benchmark(returns: pd.Series, champion_var: pd.Series, window: int, confidence: float,
                  method: str, th: ChallengerThresholds):
    p = 1 - confidence
    models = {
        "Champion": champion_var,
        "EWMA (RiskMetrics)": mk.ewma_var(returns, confidence, th.ewma_lambda),
        "Filtered historical simulation": mk.fhs_var(returns, window, confidence, th.ewma_lambda),
    }
    # Compare every model on the same days: those where all forecasts exist.
    frame = pd.DataFrame(models).dropna()
    results: dict[str, dict] = {}
    for name in models:
        hits = mk.exceptions(returns, frame[name])
        x, n = int(hits["exception"].sum()), len(hits)
        last = hits.iloc[-250:]
        zone, _ = stats.traffic_light(int(last["exception"].sum()), len(last), p)
        results[name] = {
            "days": n, "exceptions": x, "expected": rnd(n * p, 1),
            "kupiec_p": rnd(stats.kupiec_pof(x, n, p)[1], 5),
            "christoffersen_p": rnd(stats.christoffersen_independence(hits["exception"].to_numpy())[1], 5),
            "traffic_light_last_250": zone,
            "quantile_loss": rnd(mk.quantile_loss(hits, confidence), 7),
            "average_var": rnd(hits["var"].mean(), 5),
        }

    sigma_next = mk.ewma_sigma(pd.concat([returns, pd.Series([0.0])], ignore_index=True), th.ewma_lambda).iloc[-1]
    current = {
        "Champion": mk.latest_var(returns, window, confidence, method),
        "EWMA (RiskMetrics)": float(-sps.norm.ppf(p) * sigma_next),
    }
    z = (returns / mk.ewma_sigma(returns, th.ewma_lambda)).iloc[-window:]
    current["Filtered historical simulation"] = float(-np.quantile(z, p) * sigma_next)
    for k, v in current.items():
        results[k]["current_var"] = rnd(v, 5)

    issues: list[Issue] = []
    champ = results["Champion"]
    order = {"green": 0, "yellow": 1, "red": 2}
    better = [
        k for k, r in results.items()
        if k != "Champion" and r["quantile_loss"] < champ["quantile_loss"]
        and (order[r["traffic_light_last_250"]] < order[champ["traffic_light_last_250"]]
             or (champ["kupiec_p"] < th.p_material and r["kupiec_p"] >= th.p_low))
    ]
    if better:
        issues.append(
            Issue(
                check="challenger_outperforms",
                title="Volatility-sensitive challengers backtest better than the champion",
                severity=Severity.MEDIUM,
                pillar=OM,
                description=(
                    f"Over the same {champ['days']} days the champion has {champ['exceptions']} exceptions "
                    f"({champ['traffic_light_last_250']} zone), while "
                    + "; ".join(f"{k} has {results[k]['exceptions']} ({results[k]['traffic_light_last_250']} zone)" for k in better)
                    + f" against {champ['expected']} expected. The challengers also have lower quantile loss."
                ),
                recommendation=(
                    "Evaluate adopting volatility scaling (EWMA / FHS) in the production VaR, or document why the "
                    "equally weighted window is retained and how its lag is compensated."
                ),
                evidence={k: results[k] for k in ["Champion", *better]},
            )
        )
    # Require every challenger to agree: a single challenger's point estimate is noisy.
    challengers = [k for k in current if k != "Champion"]
    ratio = min(current[k] for k in challengers) / current["Champion"]
    if ratio > th.var_gap_medium:
        issues.append(
            Issue(
                check="current_var_gap",
                title="Current VaR is low relative to challenger models",
                severity=Severity.HIGH if ratio > th.var_gap_high else Severity.MEDIUM,
                pillar=CS,
                description=(
                    f"Next-day VaR from the champion is {current['Champion']:.2%} of portfolio value; every "
                    f"challenger is at least {ratio:.2f}x higher ("
                    + ", ".join(f"{k} {current[k]:.2%}" for k in challengers)
                    + "). The champion's limit utilisation and capital may be understated in the current market."
                ),
                recommendation=(
                    "Report the challenger VaR alongside the champion until the gap closes; consider a "
                    "temporary overlay approved by the model risk committee."
                ),
                evidence={"current_var": {k: rnd(v, 5) for k, v in current.items()}, "min_ratio": rnd(ratio, 3)},
            )
        )
    return issues, {"models": results}
