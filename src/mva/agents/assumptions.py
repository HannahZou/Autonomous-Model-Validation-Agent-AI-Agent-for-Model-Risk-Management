"""Assumptions agent: are the model's specification and documented assumptions supported?"""

from __future__ import annotations

from typing import Any

from mva.agents.base import BaseAgent
from mva.checks import assumptions as A
from mva.checks.base import Issue
from mva.models.market import portfolio_returns
from mva.package import CreditPDSpec, MarketVaRSpec, ModelPackage


class AssumptionsAgent(BaseAgent):
    name = "Assumptions"
    code = "AS"

    def run_checks(self, package: ModelPackage) -> tuple[list[Issue], dict[str, Any]]:
        if isinstance(package.spec, CreditPDSpec):
            return self._credit(package)
        if isinstance(package.spec, MarketVaRSpec):
            return self._market(package)
        raise NotImplementedError(package.model_type)

    def _credit(self, package: ModelPackage):
        th = self.settings.assumptions
        spec: CreditPDSpec = package.spec
        model = package.load_credit_model()
        data = package.load_data()
        dev, val = data["train"], data["test"]
        X_dev, y_dev = model.design_matrix(dev), dev[spec.target].to_numpy()
        X_val, y_val = model.design_matrix(val), val[spec.target].to_numpy()

        issues: list[Issue] = []
        metrics: dict[str, Any] = {"model_features": model.features}
        new, m, refit = A.credit_replication(model, X_dev, y_dev, th)
        issues += new
        metrics.update(m)
        new, m = A.credit_multicollinearity(X_dev, th)
        issues += new
        metrics.update(m)
        issues += A.credit_signs(model)
        issues += A.credit_significance(model, refit)
        new, m = A.credit_stability(model, refit, X_val, y_val)
        issues += new
        metrics.update(m)
        new, m = A.credit_linearity(model, X_dev, y_dev, refit.loglik, th)
        issues += new
        metrics.update(m)
        return issues, metrics

    def _market(self, package: ModelPackage):
        th = self.settings.assumptions
        spec: MarketVaRSpec = package.spec
        returns, notes = portfolio_returns(package.load_data()["prices"], spec)
        # Test the assumptions on the data the current VaR is actually built from.
        window = returns.iloc[-spec.var.window:]
        issues: list[Issue] = []
        metrics: dict[str, Any] = {
            "data_treatment": notes,
            "tested_period": [window.index[0].strftime("%Y-%m-%d"), window.index[-1].strftime("%Y-%m-%d")],
            "tested_days": len(window),
        }
        for new, m in (
            A.var_independence(window, th),
            A.var_distribution(window, spec.var.method, th),
            A.var_regime(returns, spec.var.window, th),
        ):
            issues += new
            metrics.update(m)
        return issues, metrics
