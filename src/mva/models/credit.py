"""Logistic-regression PD model, specified as data (model.json) rather than a pickle.

A JSON specification is safe to load, readable in a diff, and is how many
banks document scorecards: coefficients plus the preprocessing applied.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from scipy.special import expit


class CreditModelSpec(BaseModel):
    model_form: Literal["logistic_regression"] = "logistic_regression"
    intercept: float
    coefficients: dict[str, float] = Field(min_length=1)
    # Applied in this order: impute -> cap -> transform
    imputation: dict[str, float] = Field(default_factory=dict)
    caps: dict[str, tuple[float, float]] = Field(default_factory=dict)
    transforms: dict[str, Literal["log", "none"]] = Field(default_factory=dict)
    # Business expectation for each coefficient's direction: "+" or "-"
    expected_signs: dict[str, Literal["+", "-"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _keys_are_model_features(self) -> "CreditModelSpec":
        feats = set(self.coefficients)
        for name in ("imputation", "caps", "transforms", "expected_signs"):
            unknown = set(getattr(self, name)) - feats
            if unknown:
                raise ValueError(f"{name} refers to features not in coefficients: {sorted(unknown)}")
        return self

    @property
    def features(self) -> list[str]:
        return list(self.coefficients)

    def design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model inputs after the documented preprocessing."""
        X = pd.DataFrame(index=df.index)
        for f in self.features:
            x = pd.to_numeric(df[f], errors="coerce").astype(float)
            if f in self.imputation:
                x = x.fillna(self.imputation[f])
            if f in self.caps:
                lo, hi = self.caps[f]
                x = x.clip(lo, hi)
            if self.transforms.get(f) == "log":
                x = np.log(x.clip(lower=1.0))
            X[f] = x
        return X

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        X = self.design_matrix(df)
        if X.isna().any().any():
            bad = X.columns[X.isna().any()].tolist()
            raise ValueError(f"model inputs contain missing values with no imputation rule: {bad}")
        beta = np.array([self.coefficients[f] for f in self.features])
        return expit(self.intercept + X.to_numpy() @ beta)
