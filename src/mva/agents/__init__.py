from mva.agents.assumptions import AssumptionsAgent
from mva.agents.backtesting import BacktestingAgent
from mva.agents.base import BaseAgent
from mva.agents.data_quality import DataQualityAgent

# Agents run in this order. Week 3 adds Challenger Benchmarking and
# Sensitivity Analysis here.
DEFAULT_AGENTS: list[type[BaseAgent]] = [DataQualityAgent, AssumptionsAgent, BacktestingAgent]

__all__ = ["BaseAgent", "DataQualityAgent", "AssumptionsAgent", "BacktestingAgent", "DEFAULT_AGENTS"]
