"""Statistical tests used by the validation agents.

Implemented with numpy/scipy only, so every number can be traced to a formula
in this file. References are given next to each test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, xlogy


# -- logistic regression (independent replication of the developer's fit) ---
@dataclass
class LogitResult:
    intercept: float
    coef: np.ndarray        # on the original scale of X
    se: np.ndarray          # standard errors on the original scale
    loglik: float
    converged: bool

    @property
    def z(self) -> np.ndarray:
        return self.coef / self.se


def fit_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> LogitResult:
    """Maximum-likelihood logistic regression by Newton-Raphson (IRLS).

    Columns are standardised internally for numerical stability and the
    results are mapped back to the original scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    ridge = 1e-8 * np.eye(Z.shape[1])
    converged = False
    for _ in range(max_iter):
        p = expit(Z @ beta)
        w = np.clip(p * (1 - p), 1e-10, None)
        H = (Z * w[:, None]).T @ Z + ridge
        step = np.linalg.solve(H, Z.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    p = np.clip(expit(Z @ beta), 1e-12, 1 - 1e-12)
    w = p * (1 - p)
    cov = np.linalg.inv((Z * w[:, None]).T @ Z + ridge)
    se_std = np.sqrt(np.diag(cov))
    coef = beta[1:] / sd
    intercept = beta[0] - np.sum(beta[1:] * mu / sd)
    loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return LogitResult(float(intercept), coef, se_std[1:] / sd, loglik, converged)


def lr_test(loglik_restricted: float, loglik_full: float, df: int) -> float:
    """Likelihood-ratio test p-value."""
    stat = max(2 * (loglik_full - loglik_restricted), 0.0)
    return float(stats.chi2.sf(stat, df))


def vif(X: pd.DataFrame) -> dict[str, float]:
    """Variance inflation factor of each column (regressed on all the others)."""
    out = {}
    A = X.to_numpy(dtype=float)
    for j, col in enumerate(X.columns):
        y = A[:, j]
        others = np.column_stack([np.ones(len(A)), np.delete(A, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
        out[col] = float(np.inf) if r2 >= 1 - 1e-12 else float(1 / (1 - r2))
    return out


# -- discrimination and calibration (credit) --------------------------------
def ks_statistic(scores: np.ndarray, y: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between score distributions of bads and goods."""
    y = np.asarray(y)
    return float(stats.ks_2samp(scores[y == 1], scores[y == 0]).statistic)


def hosmer_lemeshow(pd_hat: np.ndarray, y: np.ndarray, groups: int = 10) -> tuple[float, float, pd.DataFrame]:
    """Hosmer-Lemeshow goodness-of-fit test on predicted-PD deciles (df = groups - 2)."""
    df = pd.DataFrame({"p": pd_hat, "y": y})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    table = df.groupby("bucket").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"))
    table["predicted_rate"] = table["expected"] / table["n"]
    table["observed_rate"] = table["observed"] / table["n"]
    e = table["expected"]
    stat = float((((table["observed"] - e) ** 2) / (e * (1 - e / table["n"]))).sum())
    dof = len(table) - 2
    return stat, float(stats.chi2.sf(stat, dof)), table


def calibration_in_the_large(pd_hat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Observed vs expected defaults; z-test using the sum of Bernoulli variances."""
    expected = pd_hat.sum()
    var = (pd_hat * (1 - pd_hat)).sum()
    z = (y.sum() - expected) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


# -- time series (market) ---------------------------------------------------
def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q test for autocorrelation up to `lags`."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x * x)
    acf = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def kupiec_pof(exceptions: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test. p = expected exception rate (1 - confidence)."""
    x = exceptions
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float, dict]:
    """Christoffersen (1998) test that exceptions do not cluster (first-order Markov)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    counts = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    if n01 + n11 == 0:
        return 0.0, 1.0, counts
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1)), counts


def traffic_light(exceptions: int, n: int, p: float) -> tuple[str, float]:
    """Basel traffic-light zone from the binomial cumulative probability of the exception count.

    Green below 95%, yellow below 99.99%, red otherwise. For 250 days at 99%
    this reproduces the familiar 0-4 green, 5-9 yellow, 10+ red.
    """
    cum = float(stats.binom.cdf(exceptions, n, p))
    zone = "green" if cum < 0.95 else "yellow" if cum < 0.9999 else "red"
    return zone, cum
