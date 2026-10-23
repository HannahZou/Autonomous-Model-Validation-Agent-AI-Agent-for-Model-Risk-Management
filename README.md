# Autonomous Model Validation Agent

A multi-agent framework that automates end-to-end model validation in line with the Federal Reserve's **SR 11-7** guidance on model risk management. Specialised agents review a model package — data quality, assumptions, backtesting, challenger benchmarking and sensitivity — and draft a structured validation report with severity-rated findings that a human validator reviews before sign-off.

It supports two model families out of the box:

| Model type | Example | What gets validated |
|---|---|---|
| `credit_pd` | Retail unsecured-loan PD scorecard | Development and validation samples, target definition, features |
| `market_var` | Trading-book 1-day 99% historical VaR | Daily price history, portfolio weights, VaR configuration |

## How it works

```
model package (YAML manifest + data)
        │
        ▼
   Orchestrator ──► Data Quality ──► Assumptions* ──► Backtesting* ──► Challenger* ──► Sensitivity*
        │                 │
        │        deterministic checks  →  findings (severity set by thresholds)
        │                 │
        │        LLM narrative layer   →  summary + commentary (cannot change findings)
        ▼
 results.json · summary.md · audit.jsonl          (* = coming in weeks 2–3)
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
  Data Quality      7 findings  (0.15s, narrative: template)
    DQ-001  High    Conflicting records share an ID in development data
    DQ-002  High    Values outside documented valid ranges in development data
    DQ-003  High    Suspected target leakage
    DQ-004  High    Feature distribution shift between development and validation samples (significant)
    DQ-005  Medium  Missing values in development data
    DQ-006  Medium  Duplicate records in development data
    DQ-007  Medium  Extreme outliers in development data
```

Each run is saved to `runs/<run_id>/` as `results.json` (machine-readable), `summary.md` (readable findings with recommendations) and `audit.jsonl`.

## Model package format

A package is a folder with `model_package.yaml` and the data it references. See `examples/credit_pd/model_package.yaml` and `examples/market_var/model_package.yaml`. Documentation (purpose, intended use, assumptions, limitations) lives in the manifest so agents can test claims against it.

## Roadmap

- **Week 1** ✅ Project skeleton, model package format, Data Quality agent, LLM narrative layer, orchestrator, audit log
- **Week 2** Assumptions agent (linearity in log-odds, multicollinearity/VIF, return independence and normality) and Backtesting agent (AUC/KS/Gini, Hosmer-Lemeshow, Kupiec POF, Christoffersen, Basel traffic light)
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
  agents/            one module per validation agent
  checks/            deterministic tests used by the agents
  synthetic/         synthetic credit and market data generators
tests/               pytest suite (OpenAI is mocked)
```

All data in this repository is synthetic.
