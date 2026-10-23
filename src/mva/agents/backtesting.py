"""Backtesting agent: outcomes analysis of model outputs against realised results."""

from __future__ import annotations

from typing import Any

from mva.agents.base import BaseAgent
from mva.checks import backtesting as B
from mva.checks.base import Issue
from mva.models.market import exceptions, portfolio_returns, rolling_var
from mva.package import CreditPDSpec, MarketVaRSpec, ModelPackage, PackageError


class BacktestingAgent(BaseAgent):
    name = "Backtesting"
    code = "BT"

    def run_checks(self, package: ModelPackage) -> tuple[list[Issue], dict[str, Any]]:
        if isinstance(package.spec, CreditPDSpec):
            return self._credit(package)
        if isinstance(package.spec, MarketVaRSpec):
            return self._market(package)
        raise NotImplementedError(package.model_type)

    def _credit(self, package: ModelPackage):
        th = self.settings.backtest
        spec: CreditPDSpec = package.spec
        model = package.load_credit_model()
        data = package.load_data()
        dev, val = data["train"], data["test"]
        y_dev, y_val = dev[spec.target].to_numpy(), val[spec.target].to_numpy()
        pd_dev, pd_val = model.predict_pd(dev), model.predict_pd(val)

        disc_dev = B.discrimination(y_dev, pd_dev)
        disc_val = B.discrimination(y_val, pd_val)
        issues = B.credit_discrimination(disc_dev, disc_val, th)
        new, cal = B.credit_calibration(y_val, pd_val, th)
        issues += new
        return issues, {"discrimination": {"development": disc_dev, "validation": disc_val}, "calibration": cal}

    def _market(self, package: ModelPackage):
        th = self.settings.backtest
        spec: MarketVaRSpec = package.spec
        cfg = spec.var
        returns, notes = portfolio_returns(package.load_data()["prices"], spec)
        hits = exceptions(returns, rolling_var(returns, cfg.window, cfg.confidence, cfg.method))
        if hits.empty:
            raise PackageError(f"not enough history to backtest a {cfg.window}-day VaR")
        issues, metrics = B.var_backtest(hits, cfg.confidence, th)
        metrics["data_treatment"] = notes
        metrics["var_config"] = cfg.model_dump()
        if cfg.horizon_days > 1:
            metrics["note"] = f"backtest performed on 1-day VaR; the {cfg.horizon_days}-day scaling is not backtested"
        return issues, metrics
