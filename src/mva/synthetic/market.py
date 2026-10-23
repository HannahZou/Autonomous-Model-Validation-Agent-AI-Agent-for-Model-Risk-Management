"""Synthetic daily price series for a market-risk VaR model.

With ``inject_issues=True`` the series contains: a duplicated date, a two-week
gap in the history, a run of stale (unchanged) prices, an unadjusted 2-for-1
stock split, a zero price and a few missing values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ASSETS = ["EQ_US", "EQ_EU", "RATES_10Y", "FX_EURUSD"]
WEIGHTS = {"EQ_US": 0.40, "EQ_EU": 0.25, "RATES_10Y": 0.25, "FX_EURUSD": 0.10}


def make_price_data(n_days: int = 1000, seed: int = 11, inject_issues: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    vols = np.array([0.012, 0.013, 0.005, 0.006])
    corr = np.array(
        [
            [1.0, 0.75, -0.30, 0.20],
            [0.75, 1.0, -0.25, 0.35],
            [-0.30, -0.25, 1.0, -0.10],
            [0.20, 0.35, -0.10, 1.0],
        ]
    )
    cov = np.outer(vols, vols) * corr
    # Student-t shocks give the fat tails real returns have
    z = rng.multivariate_normal(np.zeros(4), cov, n_days)
    # (scale floored so tails stay realistic: no 25% daily moves in FX or rates)
    scale = np.clip(np.sqrt(rng.chisquare(5, n_days) / 5), 0.4, None)[:, None]
    rets = z / scale
    start = np.array([420.0, 380.0, 98.0, 1.08])
    prices = start * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices.round(4), columns=ASSETS)
    df.insert(0, "date", dates)

    if inject_issues:
        # Unadjusted 2-for-1 split in EQ_US from day 600
        df.loc[600:, "EQ_US"] = (df.loc[600:, "EQ_US"] / 2).round(4)
        # Stale prices: FX feed froze for 9 days
        df.loc[701:709, "FX_EURUSD"] = df.loc[700, "FX_EURUSD"]
        # Zero price and some missing values
        df.loc[812, "RATES_10Y"] = 0.0
        df.loc[[300, 301, 450], "EQ_EU"] = np.nan
        # Two-week gap in history
        df = df.drop(index=range(400, 410))
        # Duplicated date
        df = pd.concat([df, df.iloc[[200]]]).sort_values("date", kind="stable")
    return df.reset_index(drop=True)


def write_market_package(root: str | Path, seed: int = 11, inject_issues: bool = True) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prices = make_price_data(seed=seed, inject_issues=inject_issues)
    prices.to_csv(root / "prices.csv", index=False, date_format="%Y-%m-%d")
    manifest = {
        "model_id": "MR-VAR-004",
        "name": "Trading Book 1-Day 99% Historical VaR",
        "version": "1.4",
        "model_type": "market_var",
        "documentation": {
            "purpose": "Daily 1-day 99% Value-at-Risk for the multi-asset trading book.",
            "intended_use": "Daily risk limits monitoring and internal capital reporting.",
            "owner": "Market Risk Analytics",
            "assumptions": [
                "The past 250 trading days are representative of next-day risk.",
                "Daily returns are independent across days.",
                "Portfolio weights are constant over the 1-day horizon.",
                "Price history is adjusted for corporate actions.",
            ],
            "limitations": ["Does not capture intraday liquidity risk."],
        },
        "data": {"prices": "prices.csv", "date_column": "date"},
        "assets": ASSETS,
        "weights": WEIGHTS,
        "var": {"method": "historical", "confidence": 0.99, "window": 250, "horizon_days": 1},
    }
    (root / "model_package.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return root
