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
class StatTestThresholds:
    """Shared significance levels. Rejection at 1% is a Medium (or High) finding,
    rejection only at 5% is Low, so a well-specified model rarely trips a material finding."""

    p_material: float = 0.01
    p_low: float = 0.05


@dataclass(frozen=True)
class AssumptionThresholds(StatTestThresholds):
    vif_high: float = 10.0
    vif_medium: float = 5.0
    # Independent re-estimation must reproduce the developer's PDs within this tolerance
    replication_tolerance: float = 0.01
    # Volatility of the latest 60 days relative to the rest of the VaR window
    regime_window: int = 60
    regime_ratio: float = 1.5
    ljung_box_lags: int = 10


@dataclass(frozen=True)
class BacktestThresholds(StatTestThresholds):
    auc_high: float = 0.60
    auc_medium: float = 0.70
    auc_drop: float = 0.05
    traffic_light_days: int = 250


@dataclass(frozen=True)
class ChallengerThresholds(StatTestThresholds):
    # Credit: a challenger beating the champion's validation AUC by this much,
    # with a paired-bootstrap 95% interval that excludes zero, is a finding
    auc_gap_medium: float = 0.01
    auc_gap_high: float = 0.03
    bootstrap_samples: int = 300
    rank_correlation_low: float = 0.80
    # Market: every challenger's current VaR exceeding the champion's by this ratio
    var_gap_medium: float = 1.25
    var_gap_high: float = 1.50
    ewma_lambda: float = 0.94
    seed: int = 0


@dataclass(frozen=True)
class SensitivityThresholds:
    # Credit
    shock_sd: float = 1.0              # size of one-at-a-time feature shocks
    adverse_sd: float = 0.5            # joint adverse move of every feature with an expected sign
    driver_concentration: float = 0.50  # one feature's share of total sensitivity
    parameter_draws: int = 500
    parameter_width_low: float = 0.15   # 90% interval width of portfolio PD / mean PD
    parameter_width_medium: float = 0.25
    # Market
    windows: tuple[int, ...] = (125, 250, 500)
    window_dispersion: float = 0.50     # (max - min) / champion VaR across windows
    asset_concentration: float = 0.60   # one asset's share of VaR
    stress_multiple: float = 3.0        # worst loss as a multiple of current VaR
    seed: int = 0


@dataclass(frozen=True)
class ReviewPolicy:
    # Days allowed to remediate an accepted finding, by severity
    remediation_days: tuple[tuple[str, int], ...] = (("High", 90), ("Medium", 180), ("Low", 365))


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    use_llm: bool
    runs_dir: Path
    dq: DataQualityThresholds = field(default_factory=DataQualityThresholds)
    assumptions: AssumptionThresholds = field(default_factory=AssumptionThresholds)
    backtest: BacktestThresholds = field(default_factory=BacktestThresholds)
    challenger: ChallengerThresholds = field(default_factory=ChallengerThresholds)
    sensitivity: SensitivityThresholds = field(default_factory=SensitivityThresholds)
    review: ReviewPolicy = field(default_factory=ReviewPolicy)

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
