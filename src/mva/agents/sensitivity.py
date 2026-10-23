"""Sensitivity Analysis agent: responses to input shocks, parameter uncertainty and model choices."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mva.agents.base import BaseAgent
from mva.checks import sensitivity as S
from mva.checks.base import Issue
from mva.models.market import portfolio_returns
from mva.package import CreditPDSpec, MarketVaRSpec, ModelPackage


class SensitivityAgent(BaseAgent):
    name = "Sensitivity Analysis"
    code = "SA"

    def run_checks(self, package: ModelPackage) -> tuple[list[Issue], dict[str, Any]]:
        if isinstance(package.spec, CreditPDSpec):
            return self._credit(package)
        if isinstance(package.spec, MarketVaRSpec):
            return self._market(package)
        raise NotImplementedError(package.model_type)

    def _credit(self, package: ModelPackage):
        th = self.settings.sensitivity
        spec: CreditPDSpec = package.spec
        model = package.load_credit_model()
        data = package.load_data()
        X_dev, X_val = model.design_matrix(data["train"]), model.design_matrix(data["test"])
        issues: list[Issue] = []
        metrics: dict[str, Any] = {}
        for new, m in (
            S.credit_feature_shocks(model, X_dev, X_val, th),
            S.credit_adverse_scenario(model, X_dev, X_val, th),
            S.credit_parameter_uncertainty(model, X_dev, data["train"][spec.target].to_numpy(), X_val, th),
        ):
            issues += new
            metrics.update(m)
        return issues, metrics

    def _market(self, package: ModelPackage):
        th = self.settings.sensitivity
        spec: MarketVaRSpec = package.spec
        cfg = spec.var
        prices = package.load_data()["prices"]
        returns, notes = portfolio_returns(prices, spec)
        asset_returns = _asset_returns(prices, spec)
        issues: list[Issue] = []
        metrics: dict[str, Any] = {"data_treatment": notes}
        new, m = S.var_window_sensitivity(returns, cfg.window, cfg.confidence, cfg.method, th)
        issues += new
        metrics.update(m)
        for new, m in (
            S.var_contributions(asset_returns, spec.weights, cfg.window, cfg.confidence, th),
            S.var_stress(returns, m["current_var"], spec.weights, spec.stress_scenarios, th),
        ):
            issues += new
            metrics.update(m)
        return issues, metrics


def _asset_returns(prices: pd.DataFrame, spec: MarketVaRSpec) -> pd.DataFrame:
    """Per-asset returns with the same treatment as `portfolio_returns`."""
    date_col = spec.data.date_column
    df = (prices.dropna(subset=[date_col]).drop_duplicates(subset=[date_col], keep="first")
          .sort_values(date_col).set_index(date_col)[spec.assets])
    df = df.where(df > 0).ffill()
    return df.pct_change(fill_method=None).dropna()
