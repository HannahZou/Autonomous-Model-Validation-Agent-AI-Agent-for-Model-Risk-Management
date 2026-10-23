"""Data Quality agent: is the data fit for building and validating this model?"""

from __future__ import annotations

from typing import Any

from mva.agents.base import BaseAgent
from mva.checks import common, credit, market
from mva.checks.base import Issue
from mva.package import CreditPDSpec, MarketVaRSpec, ModelPackage


class DataQualityAgent(BaseAgent):
    name = "Data Quality"
    code = "DQ"

    def run_checks(self, package: ModelPackage) -> tuple[list[Issue], dict[str, Any]]:
        data = package.load_data()
        spec = package.spec
        if isinstance(spec, CreditPDSpec):
            return self._credit(data, spec)
        if isinstance(spec, MarketVaRSpec):
            return self._market(data, spec)
        raise NotImplementedError(f"No data quality checks for model type {package.model_type}")

    def _credit(self, data, spec: CreditPDSpec):
        th = self.settings.dq
        train, test = data["train"], data["test"]
        cols = spec.features + [spec.target]
        issues: list[Issue] = []
        metrics: dict[str, Any] = {}

        for label, df in (("development", train), ("validation", test)):
            issues += credit.target_validity(df, spec.target, label)
            issues += common.missing_values(df, cols, label, th)
            issues += common.duplicate_rows(df, label, th, spec.id_column)
            issues += common.constant_columns(df, spec.features, label)
            issues += credit.range_violations(df, spec.feature_ranges, label)
            issues += credit.outliers(df, spec.features, label, th)

        new, m = credit.event_counts(train, test, spec.target, th)
        issues += new
        metrics["sample"] = m
        new, m = credit.target_leakage(train, spec.features, spec.target, th)
        issues += new
        metrics.update(m)
        new, m = credit.distribution_shift(train, test, spec.features, th)
        issues += new
        metrics.update(m)
        return issues, metrics

    def _market(self, data, spec: MarketVaRSpec):
        th = self.settings.dq
        prices = data["prices"]
        date_col = spec.data.date_column
        issues: list[Issue] = []
        issues += market.date_integrity(prices, date_col, th)
        issues += market.price_validity(prices, spec.assets, th)
        issues += market.extreme_returns(prices, date_col, spec.assets, th)
        new, metrics = market.history_length(prices, date_col, spec.var.window)
        issues += new
        valid = prices[date_col].dropna()
        metrics["date_range"] = [valid.min().strftime("%Y-%m-%d"), valid.max().strftime("%Y-%m-%d")] if len(valid) else None
        return issues, metrics
