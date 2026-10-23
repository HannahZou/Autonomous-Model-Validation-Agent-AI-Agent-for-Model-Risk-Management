from mva.agents.base import BaseAgent
from mva.agents.data_quality import DataQualityAgent

# Agents run in this order. Weeks 2-3 add Assumptions, Backtesting,
# Challenger Benchmarking and Sensitivity Analysis here.
DEFAULT_AGENTS: list[type[BaseAgent]] = [DataQualityAgent]

__all__ = ["BaseAgent", "DataQualityAgent", "DEFAULT_AGENTS"]
