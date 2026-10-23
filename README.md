# Autonomous Model Validation Agent

A multi-agent framework that automates end-to-end model validation in line with the Federal Reserve's **SR 11-7** guidance on model risk management. Specialised agents review a model package — data quality, assumptions, backtesting, challenger benchmarking and sensitivity — and draft a structured validation report with severity-rated findings that a human validator reviews before sign-off.

It supports two model families out of the box:

| Model type | Example | What gets validated |
|---|---|---|
| `credit_pd` | Retail unsecured-loan PD scorecard | Development and validation samples, target definition, features, logistic-regression model (`model.json`) |
| `market_var` | Trading-book 1-day 99% historical VaR | Daily price history, portfolio weights, VaR method / confidence / window |

## How it works

```
model package (YAML manifest + data)
        │
        ▼
   Orchestrator ──► Data Quality ──► Assumptions ──► Backtesting ──► Challenger* ──► Sensitivity*
        │                 │
        │        deterministic checks  →  findings (severity set by thresholds)
        │                 │
        │        LLM narrative layer   →  summary + commentary (cannot change findings)
        ▼
 results.json · summary.md · audit.jsonl          (* = coming in week 3)
```

**Design principles**

1. **Numbers come from code, not the LLM.** Every statistic and every severity rating is produced by deterministic, unit-tested checks with thresholds in `config.py`. The same data always gives the same findings.
2. **The LLM writes, it doesn't judge.** OpenAI structured outputs (Pydantic schemas) turn findings into report prose. Commentary for finding IDs the model invents is discarded, and severity and evidence cannot be altered.
3. **Only aggregates leave the machine.** The API receives finding statistics and model documentation, never row-level data.
4. **Never blocked by the API.** If the call fails or no key is set, a deterministic template narrative is used and the error is recorded.
5. **Everything is auditable.** Each run writes an append-only `audit.jsonl` with timings, so turnaround time is measured, not claimed.

## Week 1: Data Quality agent

| Check | Credit PD | Market VaR | SR 11-7 element |
|---|---|---|---|
| Missing values (by severity band) | ✓ | ✓ | Conceptual soundness |
| Duplicate records / conflicting IDs / duplicate dates | ✓ | ✓ | Conceptual soundness |
| Values outside documented ranges | ✓ | | Conceptual soundness |
| Extreme outliers (robust z-score) | ✓ | | Conceptual soundness |
| Target validity and default counts | ✓ | | Outcomes analysis |
| Target leakage (univariate AUC) | ✓ | | Conceptual soundness |
| Distribution shift (PSI) | ✓ | | Ongoing monitoring |
| Calendar gaps, date order | | ✓ | Conceptual soundness |
| Zero/negative and stale prices | | ✓ | Conceptual soundness |
| Implausible moves (unadjusted splits, bad ticks) | | ✓ | Conceptual soundness |
| History length vs. VaR window + 1-year backtest | | ✓ | Outcomes analysis |

The synthetic example packages contain deliberately injected defects (a post-default "recovery amount" feature, an unadjusted 2-for-1 split, a frozen FX feed, …). Clean versions of the same generators produce no High or Medium findings, which the test suite checks across several random seeds.

## Week 2: Assumptions and Backtesting agents

**Assumptions agent** (SR 11-7: evaluation of conceptual soundness)

| Check | Credit PD | Market VaR |
|---|---|---|
| Independent replication: re-estimate the documented specification and compare PDs | ✓ | |
| Multicollinearity (VIF) | ✓ | |
| Coefficient signs vs documented business expectations | ✓ | |
| Significance of each feature (re-estimated z-statistics) | ✓ | |
| Coefficient stability: re-estimate on the validation sample, flag sign flips | ✓ | |
| Linearity of log-odds (LR test for a squared term, Bonferroni-adjusted) | ✓ | |
| Independence of returns (Ljung-Box) | | ✓ |
| Volatility clustering (Ljung-Box on squared returns) | | ✓ |
| Normality (Jarque-Bera) when the method is parametric | | ✓ |
| Regime representativeness: recent vs window volatility | | ✓ |

**Backtesting agent** (SR 11-7: outcomes analysis)

| Check | Credit PD | Market VaR |
|---|---|---|
| Discrimination: AUC, Gini, KS on development and validation samples | ✓ | |
| Out-of-sample decay in AUC | ✓ | |
| Calibration in the large (observed vs expected defaults, z-test) | ✓ | |
| Calibration across risk deciles (Hosmer-Lemeshow) with decile table | ✓ | |
| Rolling one-day VaR with no look-ahead, exception series | | ✓ |
| Kupiec proportion-of-failures test | | ✓ |
| Christoffersen independence test (exception clustering) | | ✓ |
| Basel traffic light on the last 250 days | | ✓ |

Statistical tests are implemented in `src/mva/stats.py` with numpy/scipy only and unit-tested against known values (e.g. Kupiec LR = 12.96 for 10 exceptions in 250 days at 99%; the traffic-light function reproduces Basel's 0–4 / 5–9 / 10+ zones).

Severity for statistical tests: rejection at 1% is Medium (High where the error is non-conservative, e.g. under-predicted defaults or too many VaR exceptions); rejection only at 5% is Low. A correct model is still rejected at roughly the test's significance level, so the suite measures this: across 24 clean synthetic packages, at most 2 may produce a High finding.

**Model-level issues planted in the examples**

- *Credit*: the developer added `utilization_6m_avg`, a near-copy of `credit_utilization` (VIF ≈ 36), and the validation sample comes from a worse vintage, so the model under-predicts defaults (15.1% observed vs ≈11% predicted).
- *VaR*: a stress episode makes the last 60 days about three times as volatile; the 250-day historical VaR lags, giving 16 exceptions against 7.4 expected and a yellow traffic light.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add OPENAI_API_KEY to get LLM narratives

mva make-examples               # writes examples/credit_pd and examples/market_var
mva validate examples/credit_pd --no-llm
mva validate examples/market_var          # uses OpenAI when a key is set
pytest
```

Example output:

```
Retail Unsecured Loan PD Scorecard (CR-PD-001)
  Data Quality      8 findings  (0.17s, narrative: template)
    DQ-001  High    Conflicting records share an ID in development data
    DQ-002  High    Values outside documented valid ranges in development data
    DQ-003  High    Suspected target leakage
    DQ-004  High    Feature distribution shift between development and validation samples (significant)
    DQ-005  Medium  Missing values in development data
    DQ-006  Medium  Duplicate records in development data
    DQ-007  Medium  Extreme outliers in development data
    DQ-008  Medium  Default rate differs materially between samples
  Assumptions       3 findings  (0.08s, narrative: template)
    AS-001  High    Multicollinearity among model features (severe)
    AS-002  Medium  Non-linear relationship with log-odds
    AS-003  Low     Features without statistically significant effect
  Backtesting       2 findings  (0.06s, narrative: template)
    BT-001  High    Model under-predicts defaults
    BT-002  Medium  Predicted PDs do not match observed default rates across risk deciles
Total: 6 High, 6 Medium, 1 Low  in 0.35s
```

Each run is saved to `runs/<run_id>/` as `results.json` (machine-readable), `summary.md` (readable findings with recommendations) and `audit.jsonl`.

## Model package format

A package is a folder with `model_package.yaml` and the data it references. See `examples/credit_pd/model_package.yaml` and `examples/market_var/model_package.yaml`. Documentation (purpose, intended use, assumptions, limitations) lives in the manifest so agents can test claims against it.

A credit package also points to `model.json`: the delivered logistic regression as data — intercept, coefficients, the preprocessing applied (imputation → caps → transforms) and the expected sign of each coefficient. A JSON specification is safe to load (no pickles), readable in code review, and mirrors how scorecards are documented. A VaR model is fully described by its `var` block (method, confidence, window, horizon).

## Roadmap

- **Week 1** ✅ Project skeleton, model package format, Data Quality agent, LLM narrative layer, orchestrator, audit log
- **Week 2** ✅ Model specification in the package, statistics library, Assumptions agent, Backtesting agent
- **Week 3** Challenger benchmarking (gradient boosting / filtered historical simulation) and Sensitivity analysis (input shocks, stress scenarios, window and confidence sensitivity)
- **Week 4** Validation report (Word/PDF), Streamlit human-in-the-loop review (accept / edit / reject findings), turnaround-time benchmark

## Project layout

```
src/mva/
  config.py          thresholds and settings
  schemas.py         Finding, AgentReport, ValidationRun
  package.py         model package manifest and loader
  llm.py             narrative layer (OpenAI structured outputs + template fallback)
  orchestrator.py    runs agents, writes results and audit trail
  stats.py           statistical tests (logit MLE, VIF, Hosmer-Lemeshow, Kupiec, Christoffersen, ...)
  models/            the models under validation: PD scoring, portfolio returns, rolling VaR
  agents/            one module per validation agent
  checks/            deterministic tests used by the agents
  synthetic/         synthetic credit and market data generators
tests/               pytest suite (OpenAI is mocked)
```

All data in this repository is synthetic.
