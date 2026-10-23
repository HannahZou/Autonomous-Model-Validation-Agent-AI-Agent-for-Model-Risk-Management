"""Portfolio returns and rolling VaR, as the model under validation computes them."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mva.package import MarketVaRSpec


def portfolio_returns(prices: pd.DataFrame, spec: MarketVaRSpec) -> tuple[pd.Series, list[str]]:
    """Daily simple returns of the constant-weight portfolio.

    Applies a minimal, documented treatment so that the downstream tests can
    run on defective data. Every step taken is returned so the report can
    disclose it (the Data Quality agent reports the defects themselves).
    """
    date_col = spec.data.date_column
    notes: list[str] = []
    df = prices.copy()

    n_bad = int(df[date_col].isna().sum())
    if n_bad:
        df = df.dropna(subset=[date_col])
        notes.append(f"dropped {n_bad} row(s) with unparseable dates")
    n_dup = int(df[date_col].duplicated().sum())
    if n_dup:
        df = df.drop_duplicates(subset=[date_col], keep="first")
        notes.append(f"kept first of {n_dup} duplicated date(s)")
    df = df.sort_values(date_col).set_index(date_col)[spec.assets]

    n_nonpos = int((df <= 0).sum().sum())
    if n_nonpos:
        df = df.where(df > 0)
        notes.append(f"treated {n_nonpos} non-positive price(s) as missing")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        df = df.ffill()
        notes.append(f"forward-filled {n_missing} missing price(s)")

    rets = df.pct_change(fill_method=None).dropna()
    w = pd.Series(spec.weights)[spec.assets]
    port = rets @ w
    port.name = "portfolio_return"
    return port, notes


def rolling_var(returns: pd.Series, window: int, confidence: float, method: str) -> pd.Series:
    """One-day VaR (as a positive loss) for each day, using only the previous `window` returns."""
    alpha = 1 - confidence
    if method == "historical":
        q = returns.rolling(window).quantile(alpha)
        var = -q
    elif method == "parametric":
        r = returns.rolling(window)
        var = -(r.mean() + stats.norm.ppf(alpha) * r.std())
    else:  # pragma: no cover - guarded by the manifest schema
        raise ValueError(method)
    return var.shift(1).rename("var")  # VaR for day t is known at the close of t-1


def exceptions(returns: pd.Series, var: pd.Series) -> pd.DataFrame:
    """Align returns with the VaR forecast for that day and flag losses beyond VaR."""
    df = pd.DataFrame({"return": returns, "var": var}).dropna()
    df["exception"] = (-df["return"] > df["var"]).astype(int)
    return df


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))
