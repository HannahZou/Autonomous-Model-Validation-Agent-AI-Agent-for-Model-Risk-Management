from mva.agents.assumptions import AssumptionsAgent
from mva.agents.backtesting import BacktestingAgent
from mva.agents.base import BaseAgent
from mva.agents.challenger import ChallengerAgent
from mva.agents.data_quality import DataQualityAgent
from mva.agents.sensitivity import SensitivityAgent

# Agents run in this order; together they cover the SR 11-7 validation scope.
DEFAULT_AGENTS: list[type[BaseAgent]] = [
    DataQualityAgent,
    AssumptionsAgent,
    BacktestingAgent,
    ChallengerAgent,
    SensitivityAgent,
]

__all__ = [
    "BaseAgent",
    "DataQualityAgent",
    "AssumptionsAgent",
    "BacktestingAgent",
    "ChallengerAgent",
    "SensitivityAgent",
    "DEFAULT_AGENTS",
]
