"""Runtime settings and validation thresholds.

Severity is always decided by these deterministic thresholds, never by the LLM,
so that the same data always produces the same findings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class DataQualityThresholds:
    # Share of missing values in a column
    missing_high: float = 0.20
    missing_medium: float = 0.05
    # Share of fully duplicated rows
    duplicate_medium: float = 0.01
    # Population Stability Index between development and validation samples
    psi_high: float = 0.25
    psi_medium: float = 0.10
    # A single feature that separates the target this well is a leakage suspect
    leakage_auc: float = 0.95
    # Minimum number of defaults needed for a meaningful calibration test
    min_events: int = 100
    # Absolute change in default rate between samples (percentage points / 100)
    event_rate_shift: float = 0.05
    # Robust z-score beyond which a value counts as an extreme outlier
    outlier_z: float = 6.0
    outlier_share_medium: float = 0.01
    # Market data
    stale_run_days: int = 5
    max_calendar_gap_days: int = 7
    extreme_return: float = 0.25
    # A move this large is almost certainly a split or bad tick (2-for-1 split = -50%)
    extreme_return_high: float = 0.40


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    use_llm: bool
    runs_dir: Path
    dq: DataQualityThresholds = field(default_factory=DataQualityThresholds)

    @property
    def llm_enabled(self) -> bool:
        return self.use_llm and bool(self.openai_api_key)


def load_settings(**overrides) -> Settings:
    load_dotenv()
    values = dict(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        use_llm=os.getenv("MVA_USE_LLM", "1") not in {"0", "false", "False"},
        runs_dir=Path(os.getenv("MVA_RUNS_DIR", "runs")),
    )
    values.update(overrides)
    return Settings(**values)
