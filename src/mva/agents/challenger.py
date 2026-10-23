"""Challenger Benchmarking agent: compare the champion with independently built alternatives."""

from __future__ import annotations

from typing import Any

from mva.agents.base import BaseAgent
from mva.checks import challenger as C
from mva.checks.base import Issue
from mva.models.market import portfolio_returns, rolling_var
from mva.package import CreditPDSpec, MarketVaRSpec, ModelPackage


class ChallengerAgent(BaseAgent):
    name = "Challenger Benchmarking"
    code = "CH"

    def run_checks(self, package: ModelPackage) -> tuple[list[Issue], dict[str, Any]]:
        if isinstance(package.spec, CreditPDSpec):
            return self._credit(package)
        if isinstance(package.spec, MarketVaRSpec):
            return self._market(package)
        raise NotImplementedError(package.model_type)

    def _credit(self, package: ModelPackage):
        spec: CreditPDSpec = package.spec
        model = package.load_credit_model()
        data = package.load_data()
        dev, val = data["train"], data["test"]
        # Same information set as the champion: the model's features after its own preprocessing
        X_dev, X_val = model.design_matrix(dev), model.design_matrix(val)
        issues, metrics = C.credit_benchmark(
            X_dev, dev[spec.target].to_numpy(), X_val, val[spec.target].to_numpy(),
            model.predict_pd(val), self.settings.challenger,
        )
        metrics["features"] = model.features
        return issues, metrics

    def _market(self, package: ModelPackage):
        spec: MarketVaRSpec = package.spec
        cfg = spec.var
        returns, notes = portfolio_returns(package.load_data()["prices"], spec)
        champion = rolling_var(returns, cfg.window, cfg.confidence, cfg.method)
        issues, metrics = C.var_benchmark(returns, champion, cfg.window, cfg.confidence, cfg.method,
                                          self.settings.challenger)
        metrics["data_treatment"] = notes
        return issues, metrics
