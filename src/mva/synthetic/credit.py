"""Synthetic retail-loan data for a credit PD model.

With ``inject_issues=True`` the data carries realistic defects that a validator
should catch: missing income, duplicate records, impossible ages, extreme
income outliers, a drifted feature in the validation sample, and a feature
that leaks the outcome (recovery amount is only known after default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FEATURES = [
    "age",
    "annual_income",
    "debt_to_income",
    "credit_utilization",
    "num_open_accounts",
    "delinquencies_2y",
    "months_employed",
    "loan_amount",
]
TARGET = "default_flag"


def _sample(n: int, rng: np.random.Generator, utilization_shift: float = 0.0) -> pd.DataFrame:
    age = rng.integers(21, 75, n).astype(float)
    income = np.round(rng.lognormal(mean=11.0, sigma=0.5, size=n), -2)
    dti = np.clip(rng.normal(0.32, 0.12, n), 0.0, 1.2)
    util = np.clip(rng.beta(2.0, 4.0, n) + utilization_shift, 0.0, 1.0)
    open_acc = rng.poisson(6, n).astype(float)
    delinq = rng.poisson(0.3, n).astype(float)
    employed = np.clip(rng.gamma(2.0, 30.0, n), 0, 480).round()
    loan = np.round(rng.lognormal(mean=9.6, sigma=0.6, size=n), -2)

    logit = (
        -3.0
        + 2.2 * (dti - 0.32)
        + 2.8 * (util - 0.33)
        + 0.55 * delinq
        - 0.004 * (employed - 60)
        - 0.6 * (np.log(income) - 11.0)
        - 0.01 * (age - 45)
    )
    pd_true = 1.0 / (1.0 + np.exp(-logit))
    default = rng.binomial(1, pd_true)

    return pd.DataFrame(
        {
            "loan_id": np.arange(n),
            "age": age,
            "annual_income": income,
            "debt_to_income": dti.round(4),
            "credit_utilization": util.round(4),
            "num_open_accounts": open_acc,
            "delinquencies_2y": delinq,
            "months_employed": employed,
            "loan_amount": loan,
            TARGET: default,
        }
    )


def make_credit_data(
    n_train: int = 8000, n_test: int = 3000, seed: int = 7, inject_issues: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train = _sample(n_train, rng)
    test = _sample(n_test, rng, utilization_shift=0.18 if inject_issues else 0.0)
    test["loan_id"] += n_train

    if inject_issues:
        # 1. Missing income (8% of development sample)
        idx = rng.choice(n_train, int(0.08 * n_train), replace=False)
        train.loc[idx, "annual_income"] = np.nan
        # 2. Duplicate records (2%)
        dup = train.sample(frac=0.02, random_state=seed)
        train = pd.concat([train, dup], ignore_index=True)
        # 3. Impossible ages
        bad = rng.choice(len(train), 12, replace=False)
        train.loc[bad[:6], "age"] = 150
        train.loc[bad[6:], "age"] = -1
        # 4. Extreme income outliers (data entry errors)
        out = rng.choice(len(train), 120, replace=False)
        train.loc[out, "annual_income"] = 99_999_999
        # 5. Leakage: recovery amount is only observed after a default
        for df in (train, test):
            df["recovery_amount"] = np.where(
                df[TARGET] == 1, np.round(df["loan_amount"] * rng.uniform(0.2, 0.6, len(df)), -1), 0.0
            )
    return train, test


def write_credit_package(root: str | Path, seed: int = 7, inject_issues: bool = True) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train, test = make_credit_data(seed=seed, inject_issues=inject_issues)
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)

    features = FEATURES + (["recovery_amount"] if inject_issues else [])
    manifest = {
        "model_id": "CR-PD-001",
        "name": "Retail Unsecured Loan PD Scorecard",
        "version": "2.3",
        "model_type": "credit_pd",
        "documentation": {
            "purpose": "Estimate 12-month probability of default for unsecured personal loan applicants.",
            "intended_use": "Application approval cut-offs and risk-based pricing; input to CECL PD estimates.",
            "owner": "Retail Credit Risk Modeling",
            "assumptions": [
                "Log-odds of default are linear in the (transformed) features.",
                "Development sample is representative of the current applicant population.",
                "Features are available at application time.",
                "Default definition: 90+ days past due within 12 months of origination.",
            ],
            "limitations": ["Not validated for small-business or secured lending."],
        },
        "data": {"train": "train.csv", "test": "test.csv"},
        "target": TARGET,
        "id_column": "loan_id",
        "features": features,
        "feature_ranges": {
            "age": [18, 100],
            "debt_to_income": [0, 2],
            "credit_utilization": [0, 1],
        },
    }
    (root / "model_package.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return root
